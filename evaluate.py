"""Evaluate trained Blue/Red PPO policies with publication metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .load_parameters import default_models_dir
from .eval_utils import ACTION_NAMES, evaluate_matchup, rollout_episode
from .strategic_intel_env_v8 import StrategicWargameEnvV8


def _latest_model_path(models_dir: Path, prefix: str) -> str:
    candidates = []
    for p in models_dir.glob(f"{prefix}_*.zip"):
        stem_parts = p.stem.split("_")
        if len(stem_parts) < 2:
            continue
        tag = "_".join(stem_parts[1:])
        if tag.isdigit():
            score = int(tag)
        elif tag in {"final", "best"}:
            score = int(1e9)
        else:
            continue
        candidates.append((score, p.stat().st_mtime, p))
    if not candidates:
        raise FileNotFoundError(f"No models found for {prefix}_*.zip in {models_dir}")
    candidates.sort(key=lambda x: (x[0], x[1]))
    return str(candidates[-1][2].with_suffix(""))


def evaluate(
    n_games: int = 50,
    blue_model_path: str | None = None,
    red_model_path: str | None = None,
    model_dir: str | None = None,
    output_json: str | None = None,
    policy_mechanics: bool = True,
    coa_diversity_guard: bool = False,
):
    models_dir = Path(model_dir or default_models_dir())
    try:
        blue_path = blue_model_path or _latest_model_path(models_dir, "blue")
        red_path = red_model_path or _latest_model_path(models_dir, "red")
        print(f"Using Blue model: {blue_path}")
        print(f"Using Red model : {red_path}")
        blue = PPO.load(blue_path, device="cpu")
        red = PPO.load(red_path, device="cpu")
    except Exception as e:
        print("Could not load models:", e)
        print(f"Check {models_dir} for blue_*.zip and red_*.zip")
        return None

    metrics = evaluate_matchup(
        blue,
        red,
        n_episodes=n_games,
        policy_mechanics=policy_mechanics,
        coa_diversity_guard=coa_diversity_guard,
    )
    if coa_diversity_guard:
        print("(COA diversity guard ON — breaks 2+ identical Blue / 3+ Red COAs for DSS realism)")

    blue_hist = np.zeros(len(ACTION_NAMES), dtype=int)
    red_hist = np.zeros(len(ACTION_NAMES), dtype=int)
    for ep in metrics["episodes"]:
        blue_hist += ep["blue_actions"]
        red_hist += ep["red_actions"]

    cm = metrics.get("commander_metrics", {})

    print(f"\n=== COMMANDER / DSS METRICS ({n_games} games, policy_mechanics={policy_mechanics}) ===")
    print(f"Score margin (Blue−Red, final): mean {cm.get('mean_score_margin', 0):+.3f} | std {cm.get('std_score_margin', 0):.3f}")
    print(
        f"End-state trajectories (episode avg): "
        f"mil bal {metrics.get('mean_final_military_balance', 0):.3f} | "
        f"coalition {metrics.get('mean_final_coalition', 0):.3f} | "
        f"Red leverage {metrics.get('mean_final_resource_leverage', 0):.3f}"
    )
    print(
        f"Episode drift (start→end): "
        f"Δmil {cm.get('delta_military_balance', 0):+.3f} | "
        f"Δcoalition {cm.get('delta_coalition_strength', 0):+.3f} | "
        f"ΔRed leverage {cm.get('delta_resource_leverage', 0):+.3f}"
    )
    print(f"Tension in realistic band (0.45–0.72): {cm.get('tension_in_band_rate', 0)*100:.1f}% of episodes")
    print(
        f"COA diversity: Blue unique {cm.get('mean_blue_unique_actions', 0):.1f} | "
        f"Red unique {cm.get('mean_red_unique_actions', 0):.1f} | "
        f"entropy (norm) B/R {cm.get('mean_blue_entropy_norm', 0):.2f}/{cm.get('mean_red_entropy_norm', 0):.2f}"
    )
    print(
        f"Policy collapse rate (>80% one action): Blue {cm.get('blue_collapse_rate', 0)*100:.1f}% | "
        f"Red {cm.get('red_collapse_rate', 0)*100:.1f}%"
    )
    print(f"Avg tension (mean/final): {metrics['mean_tension']:.3f} / {metrics['mean_final_tension']:.3f}")
    print(f"Avg stability (final): {metrics['mean_final_stability']:.3f}")
    print(f"Doctrine clamps/ep: {metrics['mean_clamps_per_episode']:.1f} | Commitments B/R: "
          f"{metrics['mean_commitments_blue']:.1f} / {metrics['mean_commitments_red']:.1f}")

    print(f"\n--- Outcome split (secondary diagnostic) ---")
    print(f"Blue Win Rate : {metrics['blue_win_rate']*100:.1f}%")
    print(f"Red Win Rate  : {metrics['red_win_rate']*100:.1f}%")
    print(f"Mutual Loss   : {metrics['mutual_loss_rate']*100:.1f}%")
    print(f"Draw Rate     : {metrics['draw_rate']*100:.1f}%")
    print(
        f"Strategic score (final avg): Blue {metrics.get('mean_blue_strategic_score', 0):.3f} | "
        f"Red {metrics.get('mean_red_strategic_score', 0):.3f}"
    )
    print(f"RL reward (episode avg): Blue {metrics['mean_blue_reward']:.1f} | Red {metrics['mean_red_reward']:.1f}")

    print("\nBlue COA mix (top actions):")
    for i in np.argsort(blue_hist)[::-1][:8]:
        if blue_hist[i] > 0:
            print(f"  {ACTION_NAMES[i]:40} | {blue_hist[i]:5d} ({100*blue_hist[i]/blue_hist.sum():5.1f}%)")

    print("\nRed COA mix (top actions):")
    for i in np.argsort(red_hist)[::-1][:8]:
        if red_hist[i] > 0:
            print(f"  {ACTION_NAMES[i]:40} | {red_hist[i]:5d} ({100*red_hist[i]/red_hist.sum():5.1f}%)")

    report = {k: v for k, v in metrics.items() if k != "episodes"}
    report["commander_metrics"] = cm
    report["blue_action_hist"] = blue_hist.tolist()
    report["red_action_hist"] = red_hist.tolist()
    report["blue_model"] = blue_path
    report["red_model"] = red_path

    if output_json:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report: {out}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SADIE v8 models")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--blue", type=str, default=None)
    parser.add_argument("--red", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no-policy-mechanics", action="store_true")
    parser.add_argument(
        "--diversity-guard",
        action="store_true",
        help="Enable eval-only COA diversity shim (not used during training).",
    )
    args = parser.parse_args()
    if args.model_dir:
        report_dir = Path(args.model_dir)
    elif args.blue or args.red:
        report_dir = Path(args.blue or args.red).parent
    else:
        report_dir = Path(default_models_dir())
    default_out = report_dir / "eval_report.json"
    evaluate(
        n_games=args.games,
        blue_model_path=args.blue,
        red_model_path=args.red,
        model_dir=args.model_dir,
        output_json=args.output or str(default_out),
        policy_mechanics=not args.no_policy_mechanics,
        coa_diversity_guard=args.diversity_guard,
    )
