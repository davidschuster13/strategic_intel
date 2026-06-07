"""
Compare random agents and (optional) trained models with policy mechanics on vs off.
Supports publication ablation: does document-derived structure change outcomes?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .eval_utils import evaluate_matchup, rollout_episode, DEFAULT_EVAL_SEEDS
from .load_parameters import default_models_dir
from .strategic_intel_env_v8 import StrategicWargameEnvV8


class RandomAgent:
    def __init__(self, n_actions: int = 24):
        self.n_actions = n_actions

    def predict(self, obs, deterministic=True):
        return np.random.randint(0, self.n_actions), None


def _summarize_random(n_episodes: int, policy_mechanics: bool, seeds) -> dict:
    env = StrategicWargameEnvV8(max_steps=60, policy_mechanics=policy_mechanics, quiet=True)
    blue = RandomAgent(env.num_actions)
    red = RandomAgent(env.num_actions)
    episodes = [rollout_episode(env, blue, red, seed=seeds[i % len(seeds)]) for i in range(n_episodes)]
    outcomes = [e["outcome"] for e in episodes]
    return {
        "label": f"random_policy_mechanics_{'on' if policy_mechanics else 'off'}",
        "n_episodes": n_episodes,
        "blue_win_rate": outcomes.count("blue") / n_episodes,
        "red_win_rate": outcomes.count("red") / n_episodes,
        "mutual_loss_rate": outcomes.count("mutual_loss") / n_episodes,
        "mean_tension": float(np.mean([e["mean_tension"] for e in episodes])),
        "mean_final_tension": float(np.mean([e["final_tension"] for e in episodes])),
        "mean_clamps": float(np.mean([e["clamps"] for e in episodes])),
        "mean_commitments_blue": float(np.mean([e["commitments_blue"] for e in episodes])),
        "mean_commitments_red": float(np.mean([e["commitments_red"] for e in episodes])),
    }


def run_ablation(n_episodes: int = 30, model_dir: str | None = None, output: str | None = None):
    results = []
    seeds = DEFAULT_EVAL_SEEDS

    print("=== Policy Mechanics Ablation (random policies) ===\n")
    for mechanics in (False, True):
        row = _summarize_random(n_episodes, mechanics, seeds)
        results.append(row)
        print(f"{row['label']}:")
        print(f"  mean tension: {row['mean_tension']:.3f} | final: {row['mean_final_tension']:.3f}")
        print(f"  clamps/ep: {row['mean_clamps']:.1f} | commitments B/R: "
              f"{row['mean_commitments_blue']:.1f} / {row['mean_commitments_red']:.1f}\n")

    models_path = Path(model_dir or default_models_dir())
    blue_zip = models_path / "blue_final.zip"
    red_zip = models_path / "red_final.zip"
    if blue_zip.exists() and red_zip.exists():
        print("=== Trained models (policy mechanics ON) ===\n")
        blue = PPO.load(str(blue_zip), device="cpu")
        red = PPO.load(str(red_zip), device="cpu")
        trained_on = evaluate_matchup(blue, red, n_episodes=n_episodes, policy_mechanics=True)
        row = {
            "label": "trained_policy_mechanics_on",
            "n_episodes": n_episodes,
            "blue_win_rate": trained_on["blue_win_rate"],
            "red_win_rate": trained_on["red_win_rate"],
            "mutual_loss_rate": trained_on["mutual_loss_rate"],
            "mean_tension": trained_on["mean_tension"],
            "mean_final_tension": trained_on["mean_final_tension"],
            "mean_clamps": trained_on["mean_clamps_per_episode"],
            "mean_commitments_blue": trained_on["mean_commitments_blue"],
            "mean_commitments_red": trained_on["mean_commitments_red"],
        }
        results.append(row)
        print(f"  wins B/R/loss: {row['blue_win_rate']*100:.0f}% / {row['red_win_rate']*100:.0f}% / "
              f"{row['mutual_loss_rate']*100:.0f}%")
        print(f"  mean tension: {row['mean_tension']:.3f}\n")
    else:
        print(f"(Skipping trained eval — no models at {models_path})\n")

    out_path = Path(output or models_path / "ablation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SADIE policy mechanics ablation")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_ablation(args.episodes, args.model_dir, args.output)
