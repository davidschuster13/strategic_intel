"""Shared rollout and metrics for training checkpoints and publication eval."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .strategic_intel_env_v8 import StrategicWargameEnvV8
from .multi_domain_operations import MultiDomainActionLibrary

ACTION_NAMES = list(MultiDomainActionLibrary.get_all_actions().keys())
DEFAULT_EVAL_SEEDS = tuple(range(2000, 2020))


def action_diversity_metrics(action_counts: np.ndarray) -> Dict[str, float]:
    total = int(action_counts.sum())
    if total <= 0:
        return {
            "total_actions": 0,
            "unique_actions": 0,
            "dominant_action_pct": 0.0,
            "collapsed": False,
            "entropy_norm": 0.0,
        }
    probs = action_counts[action_counts > 0] / total
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    max_entropy = float(np.log(len(action_counts)))
    dominant_pct = float(action_counts.max() / total)
    return {
        "total_actions": total,
        "unique_actions": int((action_counts > 0).sum()),
        "dominant_action_pct": dominant_pct,
        "collapsed": dominant_pct >= 0.80,
        "entropy_norm": entropy / max_entropy if max_entropy > 0 else 0.0,
    }


def aggregate_commander_metrics(episodes: List[Dict]) -> Dict:
    """Trajectory- and COA-focused metrics for decision-support reporting."""
    score_margins = [
        e["blue_strategic_score"] - e["red_strategic_score"] for e in episodes
    ]
    blue_div = [action_diversity_metrics(e["blue_actions"]) for e in episodes]
    red_div = [action_diversity_metrics(e["red_actions"]) for e in episodes]

    def _mean_traj(key: str) -> float:
        vals = []
        for ep in episodes:
            traj = ep.get("trajectory", {})
            series = traj.get(key, [])
            if series:
                vals.append(float(np.mean(series)))
        return float(np.mean(vals)) if vals else 0.0

    def _mean_delta(key: str) -> float:
        deltas = []
        for ep in episodes:
            traj = ep.get("trajectory", {})
            series = traj.get(key, [])
            if len(series) >= 2:
                deltas.append(float(series[-1] - series[0]))
        return float(np.mean(deltas)) if deltas else 0.0

    return {
        "mean_score_margin": float(np.mean(score_margins)),
        "std_score_margin": float(np.std(score_margins)),
        "mean_blue_coalition_trajectory": _mean_traj("coalition_strength"),
        "mean_red_leverage_trajectory": _mean_traj("resource_leverage"),
        "mean_military_balance_trajectory": _mean_traj("military_balance"),
        "delta_military_balance": _mean_delta("military_balance"),
        "delta_coalition_strength": _mean_delta("coalition_strength"),
        "delta_resource_leverage": _mean_delta("resource_leverage"),
        "blue_collapse_rate": float(np.mean([d["collapsed"] for d in blue_div])),
        "red_collapse_rate": float(np.mean([d["collapsed"] for d in red_div])),
        "mean_blue_unique_actions": float(np.mean([d["unique_actions"] for d in blue_div])),
        "mean_red_unique_actions": float(np.mean([d["unique_actions"] for d in red_div])),
        "mean_blue_entropy_norm": float(np.mean([d["entropy_norm"] for d in blue_div])),
        "mean_red_entropy_norm": float(np.mean([d["entropy_norm"] for d in red_div])),
        "tension_in_band_rate": float(
            np.mean([0.45 <= e["mean_tension"] <= 0.72 for e in episodes])
        ),
    }


def classify_outcome(
    tension: float,
    stability: float,
    blue_reward: float = 0.0,
    red_reward: float = 0.0,
    military_balance: float = 0.5,
    coalition_strength: float = 0.5,
    resource_leverage: float = 0.5,
    blue_strategic_score: float | None = None,
    red_strategic_score: float | None = None,
) -> str:
    # Systemic breakdown only (high tension with order intact is still a scored contest)
    if stability <= 0.06:
        return "mutual_loss"

    if blue_strategic_score is not None and red_strategic_score is not None:
        blue_score = blue_strategic_score
        red_score = red_strategic_score
    else:
        # Fallback if older info dict lacks strategic scores
        blue_score = (
            1.15 * (military_balance - 0.50)
            + 0.70 * (coalition_strength - 0.78)
            + 1.00 * (0.42 - resource_leverage)
        )
        red_score = (
            1.15 * (0.50 - military_balance)
            + 1.05 * (resource_leverage - 0.38)
        )

    margin = abs(blue_score - red_score)
    if margin < 0.12:
        return "draw"
    if blue_score > red_score:
        return "blue"
    return "red"


def rollout_episode(
    env: StrategicWargameEnvV8,
    blue,
    red,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> Dict:
    obs_dict, _ = env.reset(seed=seed)
    blue_obs = obs_dict["Blue"]
    red_obs = obs_dict["Red"]
    done = False
    blue_reward = 0.0
    red_reward = 0.0
    tensions: List[float] = []
    clamps = 0
    commitments_blue = 0
    commitments_red = 0
    blue_actions = np.zeros(env.num_actions, dtype=int)
    red_actions = np.zeros(env.num_actions, dtype=int)
    trajectory = {
        "tension": [],
        "stability": [],
        "military_balance": [],
        "coalition_strength": [],
        "resource_leverage": [],
        "blue_strategic_score": [],
        "red_strategic_score": [],
    }

    while not done:
        b_action, _ = blue.predict(blue_obs, deterministic=deterministic)
        r_action, _ = red.predict(red_obs, deterministic=deterministic)
        actions = {"Blue": int(b_action), "Red": int(r_action)}

        obs_dict, rewards, term, _, info = env.step(actions)
        blue_obs = obs_dict["Blue"]
        red_obs = obs_dict["Red"]

        blue_reward += rewards["Blue"]
        red_reward += rewards["Red"]
        tensions.append(float(info.get("tension", env.tension)))
        clamps += int(info.get("blue_action_clamped", False)) + int(info.get("red_action_clamped", False))
        commitments_blue = int(info.get("blue_commitments", commitments_blue))
        commitments_red = int(info.get("red_commitments", commitments_red))
        blue_actions[int(b_action)] += 1
        red_actions[int(r_action)] += 1
        trajectory["tension"].append(float(info.get("tension", env.tension)))
        trajectory["stability"].append(float(info.get("stability", env.stability)))
        trajectory["military_balance"].append(float(info.get("military_balance", 0.5)))
        trajectory["coalition_strength"].append(float(info.get("coalition_strength", 0.5)))
        trajectory["resource_leverage"].append(float(info.get("resource_leverage", 0.5)))
        trajectory["blue_strategic_score"].append(float(info.get("blue_strategic_score", 0.0)))
        trajectory["red_strategic_score"].append(float(info.get("red_strategic_score", 0.0)))

        done = term["Blue"] if isinstance(term, dict) else term

    return {
        "blue_reward": blue_reward,
        "red_reward": red_reward,
        "final_tension": float(info.get("tension", env.tension)),
        "final_stability": float(info.get("stability", env.stability)),
        "mean_tension": float(np.mean(tensions)) if tensions else 0.0,
        "military_balance": float(info.get("military_balance", 0.5)),
        "coalition_strength": float(info.get("coalition_strength", 0.5)),
        "resource_leverage": float(info.get("resource_leverage", 0.5)),
        "outcome": classify_outcome(
            float(info.get("tension", env.tension)),
            float(info.get("stability", env.stability)),
            blue_reward,
            red_reward,
            float(info.get("military_balance", 0.5)),
            float(info.get("coalition_strength", 0.5)),
            float(info.get("resource_leverage", 0.5)),
            info.get("blue_strategic_score"),
            info.get("red_strategic_score"),
        ),
        "blue_strategic_score": float(info.get("blue_strategic_score", 0.0)),
        "red_strategic_score": float(info.get("red_strategic_score", 0.0)),
        "clamps": clamps,
        "commitments_blue": commitments_blue,
        "commitments_red": commitments_red,
        "blue_actions": blue_actions,
        "red_actions": red_actions,
        "trajectory": trajectory,
        "schelling_point": info.get("schelling_point", ""),
        "blue_diversity": action_diversity_metrics(blue_actions),
        "red_diversity": action_diversity_metrics(red_actions),
    }


def evaluate_matchup(
    blue,
    red,
    n_episodes: int = 20,
    seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    max_steps: int = 60,
    policy_mechanics: bool = True,
    coa_diversity_guard: bool = False,
) -> Dict:
    env = StrategicWargameEnvV8(
        max_steps=max_steps,
        policy_mechanics=policy_mechanics,
        quiet=True,
        coa_diversity_guard=coa_diversity_guard,
    )
    episodes = []
    for i in range(n_episodes):
        ep = rollout_episode(env, blue, red, seed=seeds[i % len(seeds)])
        episodes.append(ep)

    outcomes = [e["outcome"] for e in episodes]
    commander = aggregate_commander_metrics(episodes)
    return {
        "n_episodes": n_episodes,
        "commander_metrics": commander,
        "blue_win_rate": outcomes.count("blue") / n_episodes,
        "red_win_rate": outcomes.count("red") / n_episodes,
        "mutual_loss_rate": outcomes.count("mutual_loss") / n_episodes,
        "draw_rate": outcomes.count("draw") / n_episodes,
        "mean_blue_strategic_score": float(np.mean([e["blue_strategic_score"] for e in episodes])),
        "mean_red_strategic_score": float(np.mean([e["red_strategic_score"] for e in episodes])),
        "mean_blue_reward": float(np.mean([e["blue_reward"] for e in episodes])),
        "mean_red_reward": float(np.mean([e["red_reward"] for e in episodes])),
        "mean_tension": float(np.mean([e["mean_tension"] for e in episodes])),
        "mean_final_tension": float(np.mean([e["final_tension"] for e in episodes])),
        "mean_final_stability": float(np.mean([e["final_stability"] for e in episodes])),
        "mean_final_military_balance": float(np.mean([e["military_balance"] for e in episodes])),
        "mean_final_coalition": float(np.mean([e["coalition_strength"] for e in episodes])),
        "mean_final_resource_leverage": float(np.mean([e["resource_leverage"] for e in episodes])),
        "mean_clamps_per_episode": float(np.mean([e["clamps"] for e in episodes])),
        "mean_commitments_blue": float(np.mean([e["commitments_blue"] for e in episodes])),
        "mean_commitments_red": float(np.mean([e["commitments_red"] for e in episodes])),
        "episodes": episodes,
    }
