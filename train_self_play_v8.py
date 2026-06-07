"""
Self-Play Training for SADIE v8 (policy-aware environment).
Saves models to models/v8_policy/ by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn

from .load_parameters import default_models_dir
from .strategic_intel_env_v8 import StrategicWargameEnvV8
from .self_play_env import SelfPlayEnv
from .eval_utils import evaluate_matchup, DEFAULT_EVAL_SEEDS

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = str(PACKAGE_DIR / "models" / "v8_contest")


def _make_ppo(env, learning_rate):
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )


def train(
    iterations: int = 20,
    steps_per_round: int = 60_000,
    model_dir: str = DEFAULT_MODEL_DIR,
    eval_episodes: int = 12,
):
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    blue_models = []
    red_models = []
    blue = None
    red = None
    lr_schedule = get_linear_fn(3e-4, 5e-5, 1.0)

    training_stats = {
        "round": [],
        "blue_avg_reward": [],
        "red_avg_reward": [],
        "avg_tension": [],
        "avg_stability": [],
        "blue_win_rate": [],
        "red_win_rate": [],
        "mutual_loss_rate": [],
        "mean_clamps": [],
        "mean_commitments_blue": [],
        "mean_commitments_red": [],
    }

    print("=" * 70, flush=True)
    print("SADIE v8 Policy-Aware Self-Play Training", flush=True)
    print("=" * 70, flush=True)
    print(f"Model directory: {model_path}", flush=True)
    print(f"Iterations: {iterations} | Steps/round/side: {steps_per_round}", flush=True)
    print(
        f"Eval: {eval_episodes} episodes (no COA diversity guard) | "
        f"Seeds: {DEFAULT_EVAL_SEEDS[0]}..{DEFAULT_EVAL_SEEDS[-1]}",
        flush=True,
    )
    print("Reward: contest margin + terminal outcome alignment (v8_contest patch)", flush=True)
    print("=" * 70 + "\n", flush=True)

    # One verbose env init for policy summary
    StrategicWargameEnvV8(max_steps=60, quiet=False)

    for i in range(iterations):
        print(f"\n{'=' * 70}", flush=True)
        print(f"ITERATION {i + 1}/{iterations}", flush=True)
        print(f"{'=' * 70}", flush=True)

        opponent_red = red_models[np.random.randint(0, len(red_models))] if red_models else None
        opponent_blue = blue_models[np.random.randint(0, len(blue_models))] if blue_models else None
        print(
            f"  Opponents: Blue vs {'Red pool #' + str(red_models.index(opponent_red)) if opponent_red else 'random'}, "
            f"Red vs {'Blue pool #' + str(blue_models.index(opponent_blue)) if opponent_blue else 'random'}",
            flush=True,
        )

        blue_base = StrategicWargameEnvV8(max_steps=60, quiet=True)
        red_base = StrategicWargameEnvV8(max_steps=60, quiet=True)

        print("  Training Blue...", flush=True)
        blue_env = SelfPlayEnv(blue_base, "Blue", opponent_red)
        if blue is None:
            blue = _make_ppo(blue_env, lr_schedule)
        else:
            blue.set_env(blue_env)
            blue.learning_rate = lr_schedule
        blue.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        blue.save(str(model_path / f"blue_{i:02d}"))
        blue_models.append(blue)

        print("  Training Red...", flush=True)
        red_env = SelfPlayEnv(red_base, "Red", opponent_blue)
        if red is None:
            red = _make_ppo(red_env, lr_schedule)
        else:
            red.set_env(red_env)
            red.learning_rate = lr_schedule
        red.learn(total_timesteps=steps_per_round, reset_num_timesteps=False)
        red.save(str(model_path / f"red_{i:02d}"))
        red_models.append(red)

        print("  Evaluating...", flush=True)
        metrics = evaluate_matchup(
            blue,
            red,
            n_episodes=eval_episodes,
            policy_mechanics=True,
            coa_diversity_guard=False,
        )

        training_stats["round"].append(i + 1)
        training_stats["blue_avg_reward"].append(metrics["mean_blue_reward"])
        training_stats["red_avg_reward"].append(metrics["mean_red_reward"])
        training_stats["avg_tension"].append(metrics["mean_tension"])
        training_stats["avg_stability"].append(metrics["mean_final_stability"])
        training_stats["blue_win_rate"].append(metrics["blue_win_rate"])
        training_stats["red_win_rate"].append(metrics["red_win_rate"])
        training_stats["mutual_loss_rate"].append(metrics["mutual_loss_rate"])
        training_stats["mean_clamps"].append(metrics["mean_clamps_per_episode"])
        training_stats["mean_commitments_blue"].append(metrics["mean_commitments_blue"])
        training_stats["mean_commitments_red"].append(metrics["mean_commitments_red"])

        with open(model_path / "training_stats.json", "w", encoding="utf-8") as f:
            json.dump(training_stats, f, indent=2)

        print(f"  Blue reward: {metrics['mean_blue_reward']:.2f} | Red: {metrics['mean_red_reward']:.2f}", flush=True)
        print(
            f"  Tension: {metrics['mean_tension']:.2f} | Wins B/R/loss: "
            f"{metrics['blue_win_rate']*100:.0f}% / {metrics['red_win_rate']*100:.0f}% / "
            f"{metrics['mutual_loss_rate']*100:.0f}%",
            flush=True,
        )
        print(
            f"  Clamps/ep: {metrics['mean_clamps_per_episode']:.1f} | "
            f"Commitments B/R: {metrics['mean_commitments_blue']:.1f} / {metrics['mean_commitments_red']:.1f}",
            flush=True,
        )
        cm = metrics.get("commander_metrics", {})
        if cm:
            print(
                f"  DSS: margin {cm.get('mean_score_margin', 0):+.3f} | "
                f"collapse B/R {cm.get('blue_collapse_rate', 0)*100:.0f}%/{cm.get('red_collapse_rate', 0)*100:.0f}% | "
                f"unique COAs B/R {cm.get('mean_blue_unique_actions', 0):.1f}/{cm.get('mean_red_unique_actions', 0):.1f}",
                flush=True,
            )

    blue.save(str(model_path / "blue_final"))
    red.save(str(model_path / "red_final"))

    print("\n" + "=" * 70, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print(f"  Blue: {model_path / 'blue_final.zip'}", flush=True)
    print(f"  Red:  {model_path / 'red_final.zip'}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return blue, red, training_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SADIE v8 policy-aware self-play")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--eval-episodes", type=int, default=12)
    args = parser.parse_args()
    train(
        iterations=args.iterations,
        steps_per_round=args.steps,
        model_dir=args.model_dir,
        eval_episodes=args.eval_episodes,
    )
