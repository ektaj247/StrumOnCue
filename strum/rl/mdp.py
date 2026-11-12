"""MDP terms for goal-conditioned chord fretting.

One episode: the hand starts in one chord, is told a target chord, and has
to reach and hold it; every few seconds the target changes to another chord.
The policy sees joint state, fingertip positions and the target chord's
frets. It is never told which finger goes where: that only enters through
the reward.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .chords import ACTION_JOINTS, ACTION_TENDONS, FINGERS, MAX_FRET, load_table, pairs

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

FORCE_THRESHOLD = .01       # N; same threshold as the scripted evaluation
STRING_SENSORS = tuple(f'tip_string_{s}' for s in 'GCEA')
NECK_SENSOR = 'tip_neck'
HAND_SENSOR = 'hand_contacts'


class ChordTable:
    """Chord table as tensors, indexed [chord] or [chord, finger]."""

    def __init__(self, hand, device):
        table = load_table()
        self.names = table['chord_names']
        n = len(self.names)
        joint = {name: i for i, name in enumerate(hand.joint_names)}
        order = [joint[name] for name in table['joint_names']]
        self.qpos = torch.zeros(n, len(order), device=device)
        self.string = torch.full((n, 4), -1, dtype=torch.long, device=device)
        self.fret = torch.zeros((n, 4), dtype=torch.long, device=device)
        self.pads = torch.zeros(n, 4, 3, device=device)
        self.goal = torch.zeros(n, 4 * (MAX_FRET + 1), device=device)
        target = dict(zip(table['actuator_targets'], range(len(table['actuator_targets']))))
        self.joint_cmd = torch.zeros(n, len(ACTION_JOINTS), device=device)
        self.tendon_cmd = torch.zeros(n, len(ACTION_TENDONS), device=device)
        for c, name in enumerate(self.names):
            chord = table['chords'][name]
            self.qpos[c, order] = torch.tensor(chord['qpos'], device=device)
            for finger, string, fret in chord['fingering']:
                f = FINGERS.index(finger)
                self.string[c, f] = table['string_names'].index(string)
                self.fret[c, f] = fret
                self.pads[c, f] = torch.tensor(chord['pads'][finger], device=device)
            for s, fret in enumerate(chord['frets']):
                self.goal[c, s * (MAX_FRET + 1) + fret] = 1.
            ctrl = torch.tensor(chord['ctrl'], device=device)
            self.joint_cmd[c] = ctrl[[target[x] for x in ACTION_JOINTS]]
            self.tendon_cmd[c] = ctrl[[target[x] for x in ACTION_TENDONS]]
        self.assigned = self.string >= 0
        self.edges = torch.tensor(table['fret_edges'], device=device)
        fixed = [t for t in table['actuator_targets'] if t not in ACTION_JOINTS + ACTION_TENDONS]
        self.fixed_names = tuple(fixed)
        self.fixed_cmd = torch.tensor([table['rest']['ctrl'][target[x]] for x in fixed], device=device)
        self.table = table

    def successors(self, split, device):
        """[chord, k] successor ids and how many are valid, for one split."""
        allowed = [[self.names.index(b) for a, b in pairs(self.table, split) if a == name] for name in self.names]
        width = max(1, max(len(x) for x in allowed))
        ids = torch.zeros(len(self.names), width, dtype=torch.long, device=device)
        for i, row in enumerate(allowed):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return ids, torch.tensor([len(x) for x in allowed], device=device)


##
# Actions.
##


@dataclass(kw_only=True)
class FretActionCfg(ActionTermCfg):
    """Position targets for the twelve finger actuators.

    An action in [-1, 1] maps linearly onto each actuator's working range and
    is low-pass filtered. Wrist, thumb and LFJ5 hold their rest command.
    """
    command_name: str = 'chord'
    ema_alpha: float = .5
    tendon_range: tuple[float, float] = (0., 2.)

    def build(self, env: ManagerBasedRlEnv) -> 'FretAction':
        return FretAction(self, env)


class FretAction(ActionTerm):
    cfg: FretActionCfg

    def __init__(self, cfg: FretActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        hand = self._entity
        self._chords = _chords(env)
        self._joint_ids = torch.tensor(hand.find_joints(ACTION_JOINTS, preserve_order=True)[0], device=self.device)
        self._tendon_ids = torch.tensor(hand.find_tendons(ACTION_TENDONS, preserve_order=True)[0], device=self.device)
        self._fixed_ids = torch.tensor(hand.find_joints(self._chords.fixed_names, preserve_order=True)[0], device=self.device)
        limits = hand.data.default_joint_pos_limits[0, self._joint_ids]
        low = torch.cat([limits[:, 0], torch.full((len(ACTION_TENDONS),), cfg.tendon_range[0], device=self.device)])
        high = torch.cat([limits[:, 1], torch.full((len(ACTION_TENDONS),), cfg.tendon_range[1], device=self.device)])
        self._offset, self._scale = (high + low) / 2, (high - low) / 2
        self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._target = self._offset.repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return len(ACTION_JOINTS) + len(ACTION_TENDONS)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def target(self) -> torch.Tensor:
        return self._target

    def to_action(self, command: torch.Tensor) -> torch.Tensor:
        """Inverse of the action map (used by the scripted check)."""
        return (command - self._offset) / self._scale

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        wanted = self._offset + self._scale * actions.clamp(-1., 1.)
        self._target[:] = self.cfg.ema_alpha * wanted + (1. - self.cfg.ema_alpha) * self._target

    def apply_actions(self) -> None:
        split = len(ACTION_JOINTS)
        self._entity.set_joint_position_target(self._target[:, :split], joint_ids=self._joint_ids)
        self._entity.set_tendon_len_target(self._target[:, split:], tendon_ids=self._tendon_ids)
        self._entity.set_joint_position_target(self._chords.fixed_cmd.expand(self.num_envs, -1), joint_ids=self._fixed_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.
        # Start from the command that holds the chord the hand was reset into.
        start = self._env.command_manager.get_term(self.cfg.command_name).source[env_ids]
        self._target[env_ids] = torch.cat([self._chords.joint_cmd[start], self._chords.tendon_cmd[start]], dim=-1)


##
# Command: which chord to fret, and bookkeeping for success.
##


@dataclass(kw_only=True)
class ChordCommandCfg(CommandTermCfg):
    split: str = 'train'
    """Which chord-to-chord pairs may occur: train, heldout or all."""
    hold_window_s: float = 1.
    """Final part of each target period over which the hold is judged."""
    source_grace_s: float = 1.
    """After a switch, contacts of the previous chord are not counted as unwanted."""
    required_hold_fraction: float = .95

    def build(self, env: ManagerBasedRlEnv) -> 'ChordCommand':
        return ChordCommand(self, env)


class ChordCommand(CommandTerm):
    cfg: ChordCommandCfg

    def __init__(self, cfg: ChordCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.chords = _chords(env)
        self._next, self._count = self.chords.successors(cfg.split, self.device)
        self.sources = (self._count > 0).nonzero().flatten()
        zeros = lambda dtype=torch.float: torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self.source, self.target = zeros(torch.long), zeros(torch.long)
        self.fresh = zeros(torch.bool)
        self.forced = None
        self.version = 0            # bumped whenever any source or target changes
        self._restart_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.since_switch = zeros()
        self.window_steps, self.window_ok = zeros(), zeros()
        self.period_unwanted = zeros(torch.bool)
        self._steps, self._ok_sum, self._periods, self._held, self._clean = zeros(), zeros(), zeros(), zeros(), zeros()
        for name in ('chord_ok', 'transition_success', 'clean_success', 'unwanted_force', 'contact_overflow'):
            self.metrics[name] = zeros()

    @property
    def command(self) -> torch.Tensor:
        return self.chords.goal[self.target]

    def assign(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Fix each env's first chord change (evaluation). Applies from the next reset."""
        self.forced = (source.to(self.device), target.to(self.device))

    def begin_episode(self, env_ids: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        if self.forced is not None:
            start = self.forced[0][env_ids]
        self.source[env_ids] = start
        self.fresh[env_ids] = True
        self._restart_step[env_ids] = self._env.common_step_counter
        for buffer in (self._steps, self._ok_sum, self._periods, self._held, self._clean):
            buffer[env_ids] = 0.
        return start

    def period_result(self):
        """(held, clean) for the target period in progress, judged so far."""
        held = self.window_ok >= self.cfg.required_hold_fraction * self.window_steps.clamp(min=1.)
        held &= self.window_steps > 0
        return held, held & ~self.period_unwanted

    def _update_metrics(self) -> None:
        # Envs reset earlier in this env step have not simulated their new
        # episode yet; nothing measured this step belongs to it.
        live = self._restart_step != self._env.common_step_counter
        state = contact_state(self._env)
        self.since_switch += self._env.step_dt * live
        in_window = (self.time_left <= self.cfg.hold_window_s) & live
        self.window_steps += in_window
        self.window_ok += in_window & state['chord_ok']
        self.period_unwanted |= (state['unwanted'] >= FORCE_THRESHOLD) & live
        self._steps += live
        self._ok_sum += state['chord_ok'] & live
        self.metrics['contact_overflow'][:] = state['overflow']
        self.metrics['chord_ok'][:] = self._ok_sum / self._steps.clamp(min=1.)
        self.metrics['unwanted_force'][:] = state['unwanted']
        self.metrics['transition_success'][:] = self._held / self._periods.clamp(min=1.)
        self.metrics['clean_success'][:] = self._clean / self._periods.clamp(min=1.)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self.version += 1
        _forget_substeps(self._env, env_ids)
        old = env_ids[~self.fresh[env_ids]]
        if len(old):
            held, clean = self.period_result()
            self._periods[old] += 1.
            self._held[old] += held[old]
            self._clean[old] += clean[old]
            self.source[old] = self.target[old]
        first = self.fresh[env_ids]
        self.fresh[env_ids] = False
        source = self.source[env_ids]
        count = self._count[source]
        pick = (torch.rand(len(env_ids), device=self.device) * count).long().clamp(max=self._next.shape[1] - 1)
        chosen = self._next[source, pick]
        # A chord with no allowed successor in this split keeps its target.
        self.target[env_ids] = torch.where(count > 0, chosen, source)
        if self.forced is not None:
            self.target[env_ids] = torch.where(first, self.forced[1][env_ids], self.target[env_ids])
        self.since_switch[env_ids] = 0.
        self.window_steps[env_ids] = 0.
        self.window_ok[env_ids] = 0.
        self.period_unwanted[env_ids] = False

    def _update_command(self) -> None:
        pass


def reset_to_chord(env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str = 'chord',
                   noise: float = .02) -> None:
    """Start each episode holding a chord that has an allowed successor."""
    term: ChordCommand = env.command_manager.get_term(command_name)
    hand = env.scene['hand']
    start = term.sources[torch.randint(len(term.sources), (len(env_ids),), device=env.device)]
    start = term.begin_episode(env_ids, start)
    qpos = term.chords.qpos[start].clone()
    if noise:
        # Same offset on J2 and J1 keeps the distal coupling satisfied.
        for finger in FINGERS:
            ids = hand.find_joints([f'lh_{finger}J4', f'lh_{finger}J3', f'lh_{finger}J2', f'lh_{finger}J1'], preserve_order=True)[0]
            delta = (torch.rand(len(env_ids), 3, device=env.device) * 2 - 1) * noise
            # Only lift: positive J3 offsets raise the fingertip off the strings.
            delta[:, 1] = delta[:, 1].abs()
            qpos[:, ids[0]] += delta[:, 0]
            qpos[:, ids[1]] += delta[:, 1]
            qpos[:, ids[2]] += delta[:, 2]
            qpos[:, ids[3]] += delta[:, 2]
        limits = hand.data.default_joint_pos_limits[env_ids]
        qpos = torch.maximum(torch.minimum(qpos, limits[..., 1]), limits[..., 0])
    hand.write_joint_state_to_sim(qpos, torch.zeros_like(qpos), env_ids=env_ids)


##
# Contact bookkeeping shared by rewards, observations and success.
##


def _chords(env: ManagerBasedRlEnv) -> ChordTable:
    if not hasattr(env, '_strum_chords'):
        env._strum_chords = ChordTable(env.scene['hand'], env.device)
    return env._strum_chords


def pad_positions(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fingertip pad markers in the instrument frame, [envs, 4, 3]."""
    hand = env.scene['hand']
    if not hasattr(env, '_strum_pad_ids'):
        env._strum_pad_ids = torch.tensor(hand.find_sites([f'pad_{f.lower()}' for f in FINGERS], preserve_order=True)[0], device=env.device)
    origin = env.scene['ukulele'].data.root_link_pos_w
    return hand.data.site_pos_w[:, env._strum_pad_ids] - origin[:, None]


def _fingertip_order(sensor) -> list[int]:
    names = [n.split('/')[-1] for n in sensor.primary_names]
    return [names.index(f'lh_{f.lower()}distal_collision') for f in FINGERS]


def _measure(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Classify the contacts the sensors report right now under the current target."""
    chords = _chords(env)
    term: ChordCommand = env.command_manager.get_term('chord')
    origin = env.scene['ukulele'].data.root_link_pos_w[:, None]
    force, fret = [], []
    for name in STRING_SENSORS:
        sensor = env.scene[name]
        order = _fingertip_order(sensor)
        data = sensor.data
        force.append(data.force[:, order, 0].abs() * (data.found[:, order] > 0))
        x = (data.pos[:, order] - origin)[..., 0]
        fret.append(torch.bucketize(x.contiguous(), chords.edges))
    force, fret = torch.stack(force, dim=-1), torch.stack(fret, dim=-1)   # [envs, finger, string]
    neck = env.scene[NECK_SENSOR]
    order = _fingertip_order(neck)
    neck_force = neck.data.force[:, order, 0].abs() * (neck.data.found[:, order] > 0)
    neck_fret = torch.bucketize(((neck.data.pos[:, order] - origin)[..., 0]).contiguous(), chords.edges)
    hand = env.scene[HAND_SENSOR].data
    total = (hand.force[..., 0].abs() * (hand.found > 0)).sum(dim=-1)
    # found holds the number of matching contacts; more than the sensor keeps
    # means some contact was not measured.
    overflow = hand.found.max(dim=-1).values > hand.found.shape[-1]

    def allowed(chord):
        string, want = chords.string[chord], chords.fret[chord]             # [envs, finger]
        on_string = torch.nn.functional.one_hot(string.clamp(min=0), 4).bool() & (string >= 0)[..., None]
        good = on_string & (fret == want[..., None])
        return good, (string >= 0) & (neck_fret == want)

    good, good_neck = allowed(term.target)
    assigned = chords.assigned[term.target]
    correct = ((force * good).sum(dim=-1) >= FORCE_THRESHOLD) & assigned
    grace = (term.since_switch < term.cfg.source_grace_s)[:, None]
    old, old_neck = allowed(term.source)
    permitted = good | (old & grace[..., None])
    permitted_neck = good_neck | (old_neck & grace)
    unwanted = (total - (force * permitted).sum(dim=(1, 2)) - (neck_force * permitted_neck).sum(dim=1)).clamp(min=0.)
    return {'force': force, 'neck_force': neck_force, 'correct': correct, 'assigned': assigned,
            'unwanted': unwanted, 'overflow': overflow}


def _substeps(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    if not hasattr(env, '_strum_substeps'):
        env._strum_substeps = {'unwanted': torch.zeros(env.num_envs, device=env.device),
                               'overflow': torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)}
    return env._strum_substeps


def _forget_substeps(env: ManagerBasedRlEnv, env_ids) -> None:
    seen = _substeps(env)
    seen['unwanted'][env_ids] = 0.
    seen['overflow'][env_ids] = False


def substep_unwanted(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-physics-substep check, so a contact shorter than one policy step still counts.

    Registered as a per-substep metric. Keeps the largest unwanted force seen
    during the env step for contact_state(); the logged value is the fraction
    of substeps with unwanted contact.
    """
    seen = _substeps(env)
    if (env._sim_step_counter - 1) % env.cfg.decimation == 0:
        _forget_substeps(env, slice(None))
    now = _measure(env)
    seen['unwanted'] = torch.maximum(seen['unwanted'], now['unwanted'])
    seen['overflow'] |= now['overflow']
    return (now['unwanted'] >= FORCE_THRESHOLD).float()


def contact_state(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    """Per-env contact summary under the benchmark rule.

    correct[f]: finger f presses its assigned string in its assigned fret.
    unwanted: largest hand contact force during this env step that is neither
    an assigned string contact nor an assigned fingertip on the fretboard
    inside its own fret. overflow: the hand had more contacts than the sensor
    keeps, so the measurement is incomplete; treated as not ok.

    Cached per env step and per command version: a reset or target change
    inside the step recomputes it against the new target.
    """
    term: ChordCommand = env.command_manager.get_term('chord')
    key = (env.common_step_counter, term.version)
    cache = getattr(env, '_strum_contact', None)
    if cache is not None and cache[0] == key:
        return cache[1]
    state = _measure(env)
    seen = _substeps(env)
    state['unwanted'] = torch.maximum(state['unwanted'], seen['unwanted'])
    state['overflow'] = state['overflow'] | seen['overflow']
    state['chord_ok'] = ((state['correct'] | ~state['assigned']).all(dim=1)
                         & (state['unwanted'] < FORCE_THRESHOLD) & ~state['overflow'])
    env._strum_contact = (key, state)
    return state


def _pad_error(env: ManagerBasedRlEnv):
    chords = _chords(env)
    target = env.command_manager.get_term('chord').target
    return pad_positions(env) - chords.pads[target], chords.assigned[target]


##
# Observations.
##


def joint_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene['hand'].data.joint_pos


def joint_vel(env: ManagerBasedRlEnv, scale: float = .1) -> torch.Tensor:
    return env.scene['hand'].data.joint_vel * scale


def fingertip_pos(env: ManagerBasedRlEnv, scale: float = 20.) -> torch.Tensor:
    return (pad_positions(env) * scale).flatten(1)


def goal_frets(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Target fret of each string, one-hot over frets 0..3 (0 is open)."""
    return env.command_manager.get_command('chord')


def privileged_targets(env: ManagerBasedRlEnv, scale: float = 20.) -> torch.Tensor:
    error, assigned = _pad_error(env)
    return torch.cat([(error * assigned[..., None] * scale).flatten(1), assigned.float()], dim=1)


def privileged_contacts(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = contact_state(env)
    term = env.command_manager.get_term('chord')
    return torch.cat([state['force'].flatten(1).clamp(max=2.), state['neck_force'].clamp(max=2.),
                      state['unwanted'][:, None].clamp(max=2.), term.since_switch[:, None],
                      term.time_left[:, None]], dim=1)


##
# Rewards.
##


def reach(env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    """Assigned fingertips near where they sit when the chord is fretted."""
    error, assigned = _pad_error(env)
    close = 1. - torch.tanh(error.norm(dim=-1) / std)
    return (close * assigned).sum(dim=1) / assigned.sum(dim=1).clamp(min=1)


def clear(env: ManagerBasedRlEnv, height: float = .015) -> torch.Tensor:
    """Fingers the chord does not use stay above the strings."""
    _, assigned = _pad_error(env)
    z = pad_positions(env)[..., 2]
    up = (z / height).clamp(0., 1.)
    free = ~assigned
    return torch.where(free.any(dim=1), (up * free).sum(dim=1) / free.sum(dim=1).clamp(min=1), torch.ones_like(z[:, 0]))


def correct_contacts(env: ManagerBasedRlEnv) -> torch.Tensor:
    state = contact_state(env)
    return state['correct'].sum(dim=1) / state['assigned'].sum(dim=1).clamp(min=1)


def chord_ok(env: ManagerBasedRlEnv) -> torch.Tensor:
    return contact_state(env)['chord_ok'].float()


def unwanted_contact(env: ManagerBasedRlEnv, scale: float = .5) -> torch.Tensor:
    return torch.tanh(contact_state(env)['unwanted'] / scale)
