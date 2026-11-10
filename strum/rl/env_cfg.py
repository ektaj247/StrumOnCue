"""Environment and PPO configuration for the chord-fretting task."""
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.config import RslRlModelCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp
from .scene import hand_cfg, ukulele_cfg

FINGERTIPS = ContactMatch(mode='geom', pattern=r'lh_(ff|mf|rf|lf)distal_collision', entity='hand')


def _sensors():
    def tip(name, geom):
        return ContactSensorCfg(name=name, primary=FINGERTIPS,
                                secondary=ContactMatch(mode='geom', pattern=geom, entity='ukulele'),
                                fields=('found', 'force', 'pos'), reduce='maxforce', num_slots=1)
    strings = tuple(tip(name, 'string_' + name[-1]) for name in mdp.STRING_SENSORS)
    whole_hand = ContactSensorCfg(name=mdp.HAND_SENSOR,
                                  primary=ContactMatch(mode='subtree', pattern='lh_forearm', entity='hand'),
                                  fields=('found', 'force'), reduce='maxforce', num_slots=8)
    return strings + (tip(mdp.NECK_SENSOR, 'neck'), whole_hand)


def make_fret_env_cfg(play: bool = False, split: str = 'train') -> ManagerBasedRlEnvCfg:
    actor = {
        'joint_pos': ObservationTermCfg(func=mdp.joint_pos, noise=Unoise(n_min=-.005, n_max=.005)),
        'joint_vel': ObservationTermCfg(func=mdp.joint_vel, noise=Unoise(n_min=-.05, n_max=.05)),
        'fingertips': ObservationTermCfg(func=mdp.fingertip_pos),
        'goal': ObservationTermCfg(func=mdp.goal_frets),
        'actions': ObservationTermCfg(func=base_mdp.last_action),
    }
    critic = {**actor,
              'targets': ObservationTermCfg(func=mdp.privileged_targets),
              'contacts': ObservationTermCfg(func=mdp.privileged_contacts)}
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type='plane'), num_envs=1, env_spacing=.8,
                       sensors=_sensors()),
        observations={'actor': ObservationGroupCfg(actor, enable_corruption=not play),
                      'critic': ObservationGroupCfg(critic, enable_corruption=False)},
        actions={'fingers': mdp.FretActionCfg(entity_name='hand')},
        commands={'chord': mdp.ChordCommandCfg(resampling_time_range=(2.5, 3.5), split=split)},
        events={'reset_to_chord': EventTermCfg(func=mdp.reset_to_chord, mode='reset',
                                               params={'noise': 0. if play else .02})},
        rewards={
            'reach': RewardTermCfg(func=mdp.reach, weight=1., params={'std': .04}),
            'reach_fine': RewardTermCfg(func=mdp.reach, weight=1., params={'std': .006}),
            'clear': RewardTermCfg(func=mdp.clear, weight=.5),
            'contacts': RewardTermCfg(func=mdp.correct_contacts, weight=1.),
            'chord_ok': RewardTermCfg(func=mdp.chord_ok, weight=3.),
            'unwanted': RewardTermCfg(func=mdp.unwanted_contact, weight=-1.),
            'action_rate': RewardTermCfg(func=base_mdp.action_rate_l2, weight=-.01),
        },
        terminations={'time_out': TerminationTermCfg(func=base_mdp.time_out, time_out=True),
                      'nan': TerminationTermCfg(func=base_mdp.nan_detection)},
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_ROOT, entity_name='ukulele',
                            lookat=(.05, 0., .02), distance=.45, elevation=-25., azimuth=60.),
        sim=SimulationCfg(nconmax=64, njmax=256,
                          mujoco=MujocoCfg(timestep=.004, iterations=20, ls_iterations=20,
                                           impratio=10, cone='elliptic')),
        decimation=5,
        episode_length_s=9.,
    )
    cfg.scene.entities = {'hand': hand_cfg(), 'ukulele': ukulele_cfg()}
    if play:
        cfg.episode_length_s = int(1e9)
    return cfg


def fret_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True,
                            distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .5, 'std_type': 'log'}),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=.003),
        clip_actions=1.,
        experiment_name='strum_fret',
        wandb_project='strum-on-cue',
        num_steps_per_env=24,
        max_iterations=3000,
        save_interval=100,
    )
