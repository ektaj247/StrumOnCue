"""Load a Shadow Hand checkpoint and record a bounded GPU rollout."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import mjlab.tasks
import mjlab_hand

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.wrappers import VideoRecorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    assert torch.cuda.is_available(), "Run inside a GPU allocation"
    assert args.steps > 0
    output.mkdir(parents=True, exist_ok=True)

    task = "Grasp-Shadow"
    device = "cuda:0"

    env_cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)

    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = 640
    env_cfg.viewer.height = 480

    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array",
    )

    try:
        env = VideoRecorder(
            env,
            video_folder=output,
            step_trigger=lambda step: step == 0,
            video_length=args.steps,
            disable_logger=True,
        )
        env = RslRlVecEnvWrapper(
            env,
            clip_actions=agent_cfg.clip_actions,
        )

        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)

        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        print(f"Loaded checkpoint: {checkpoint}", flush=True)

        observations = env.get_observations()
        episode_endings = 0
        reward_sum = 0.0

        with torch.inference_mode():
            for step in range(args.steps):
                actions = policy(observations)
                assert torch.isfinite(actions).all(), "Nonfinite actions"

                observations, rewards, dones, _ = env.step(actions)
                assert torch.isfinite(rewards).all(), "Nonfinite rewards"

                reward_sum += rewards.sum().item()
                episode_endings += int(dones.sum().item())

                if (step + 1) % 50 == 0:
                    print(
                        f"Completed {step + 1}/{args.steps} steps",
                        flush=True,
                    )

        print(f"Reward sum: {reward_sum:.6f}")
        print(f"Episode endings: {episode_endings}")
        print("PASS: checkpoint loaded and rollout completed.")
    finally:
        env.close()

    videos = sorted(output.glob("*.mp4"))
    if not videos:
        raise RuntimeError(
            "Rollout completed, but no MP4 was found in the output directory"
        )

    for video in videos:
        print(f"VIDEO: {video}")


if __name__ == "__main__":
    main()
