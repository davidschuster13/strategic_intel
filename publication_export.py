"""
Export SADIE evaluation results for publication (JSON, CSV, figures).

Tables align with LaTeX templates:
  - tab:main_results   — key evaluation metrics
  - tab:coa_mix        — aggregated action counts (%)
  - tab:diversity      — COA diversity metrics
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .eval_utils import ACTION_NAMES, action_diversity_metrics


def _fmt_pm(mean: float, std: float, decimals: int = 2) -> str:
    std_decimals = 3 if std < 0.01 and std > 0 else decimals
    return f"{mean:.{decimals}f} ± {std:.{std_decimals}f}"


def _pct(value: float, decimals: int = 1) -> str:
    return f"{100.0 * value:.{decimals}f}%"


def _episode_series(episodes: List[Dict], key: str) -> np.ndarray:
    return np.array([float(e[key]) for e in episodes], dtype=float)


def _per_episode_diversity(episodes: List[Dict], side: str) -> Dict[str, float]:
    key = "blue_actions" if side == "blue" else "red_actions"
    divs = [action_diversity_metrics(e[key]) for e in episodes]
    unique = np.array([d["unique_actions"] for d in divs], dtype=float)
    entropy = np.array([d["entropy_norm"] for d in divs], dtype=float)
    collapsed = np.array([1.0 if d["collapsed"] else 0.0 for d in divs], dtype=float)
    return {
        "mean_unique": float(unique.mean()) if len(unique) else 0.0,
        "std_unique": float(unique.std()) if len(unique) else 0.0,
        "mean_entropy_norm": float(entropy.mean()) if len(entropy) else 0.0,
        "std_entropy_norm": float(entropy.std()) if len(entropy) else 0.0,
        "collapse_rate": float(collapsed.mean()) if len(collapsed) else 0.0,
    }


def build_publication_tables(
    metrics: Dict,
    episodes: List[Dict],
    action_names: Sequence[str] = ACTION_NAMES,
    checkpoint: str = "v8_contest",
) -> Dict:
    """Build structured tables matching publication LaTeX layouts."""
    cm = metrics.get("commander_metrics", {}) or {}
    n = int(metrics.get("n_episodes", len(episodes)))

    margins = np.array(
        [e["blue_strategic_score"] - e["red_strategic_score"] for e in episodes],
        dtype=float,
    )
    final_tension = _episode_series(episodes, "final_tension")
    final_stability = _episode_series(episodes, "final_stability")
    clamps = _episode_series(episodes, "clamps")

    blue_div = _per_episode_diversity(episodes, "blue")
    red_div = _per_episode_diversity(episodes, "red")

    blue_hist = np.zeros(len(action_names), dtype=int)
    red_hist = np.zeros(len(action_names), dtype=int)
    for ep in episodes:
        blue_hist += ep["blue_actions"]
        red_hist += ep["red_actions"]
    blue_total = int(blue_hist.sum())
    red_total = int(red_hist.sum())

    coa_rows: List[Dict] = []
    for i, name in enumerate(action_names):
        b_cnt = int(blue_hist[i])
        r_cnt = int(red_hist[i])
        if b_cnt == 0 and r_cnt == 0:
            continue
        coa_rows.append(
            {
                "action": name,
                "blue_count": b_cnt,
                "red_count": r_cnt,
                "blue_pct": (100.0 * b_cnt / blue_total) if blue_total else 0.0,
                "red_pct": (100.0 * r_cnt / red_total) if red_total else 0.0,
                "blue_pct_formatted": f"{100.0 * b_cnt / blue_total:.1f}%" if blue_total else "--",
                "red_pct_formatted": f"{100.0 * r_cnt / red_total:.1f}%" if red_total else "--",
            }
        )
    coa_rows.sort(key=lambda r: r["blue_count"] + r["red_count"], reverse=True)

    main_results = [
        {
            "metric": "Mean Score Margin (Blue - Red)",
            "value": float(cm.get("mean_score_margin", margins.mean())),
            "std": float(cm.get("std_score_margin", margins.std())),
            "formatted": _fmt_pm(
                float(cm.get("mean_score_margin", margins.mean())),
                float(cm.get("std_score_margin", margins.std())),
                2,
            ),
            "notes": "Slight Red edge, contested",
        },
        {
            "metric": "Blue Win Rate",
            "value": float(metrics.get("blue_win_rate", 0)),
            "std": None,
            "formatted": _pct(metrics.get("blue_win_rate", 0)),
            "notes": "Strategic score margin > 0.12",
        },
        {
            "metric": "Red Win Rate",
            "value": float(metrics.get("red_win_rate", 0)),
            "std": None,
            "formatted": _pct(metrics.get("red_win_rate", 0)),
            "notes": "Strategic score margin < -0.12",
        },
        {
            "metric": "Draw Rate",
            "value": float(metrics.get("draw_rate", 0)),
            "std": None,
            "formatted": _pct(metrics.get("draw_rate", 0)),
            "notes": "|margin| < 0.12",
        },
        {
            "metric": "Mutual Loss Rate",
            "value": float(metrics.get("mutual_loss_rate", 0)),
            "std": None,
            "formatted": _pct(metrics.get("mutual_loss_rate", 0)),
            "notes": "Stability collapse",
        },
        {
            "metric": "Mean Tension (final)",
            "value": float(final_tension.mean()),
            "std": float(final_tension.std()),
            "formatted": _fmt_pm(float(final_tension.mean()), float(final_tension.std()), 2),
            "notes": "Capped at 0.82",
        },
        {
            "metric": "Tension in Realistic Band (0.45-0.72)",
            "value": float(cm.get("tension_in_band_rate", 0)),
            "std": None,
            "formatted": _pct(cm.get("tension_in_band_rate", 0)),
            "notes": "Sustainable competition",
        },
        {
            "metric": "Mean Final Stability",
            "value": float(final_stability.mean()),
            "std": float(final_stability.std()),
            "formatted": _fmt_pm(float(final_stability.mean()), float(final_stability.std()), 2),
            "notes": "",
        },
        {
            "metric": "Mean Clamps per Episode",
            "value": float(clamps.mean()),
            "std": float(clamps.std()),
            "formatted": f"{float(clamps.mean()):.1f}",
            "notes": "Low doctrine override",
        },
    ]

    diversity = [
        {
            "metric": "Mean Unique Actions per Episode",
            "blue_value": blue_div["mean_unique"],
            "blue_std": blue_div["std_unique"],
            "red_value": red_div["mean_unique"],
            "red_std": red_div["std_unique"],
            "blue_formatted": _fmt_pm(blue_div["mean_unique"], blue_div["std_unique"], 1),
            "red_formatted": _fmt_pm(red_div["mean_unique"], red_div["std_unique"], 1),
        },
        {
            "metric": "Collapse Rate (>80% one action)",
            "blue_value": blue_div["collapse_rate"],
            "blue_std": None,
            "red_value": red_div["collapse_rate"],
            "red_std": None,
            "blue_formatted": _pct(blue_div["collapse_rate"]),
            "red_formatted": _pct(red_div["collapse_rate"]),
        },
        {
            "metric": "Normalized Entropy",
            "blue_value": blue_div["mean_entropy_norm"],
            "blue_std": blue_div["std_entropy_norm"],
            "red_value": red_div["mean_entropy_norm"],
            "red_std": red_div["std_entropy_norm"],
            "blue_formatted": f"{blue_div['mean_entropy_norm']:.2f}",
            "red_formatted": f"{red_div['mean_entropy_norm']:.2f}",
        },
    ]

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint,
            "n_episodes": n,
            "policy_mechanics": metrics.get("policy_mechanics"),
            "coa_diversity_guard": metrics.get("coa_diversity_guard"),
            "blue_model": metrics.get("blue_model"),
            "red_model": metrics.get("red_model"),
        },
        "table_main_results": main_results,
        "table_coa_mix": coa_rows,
        "table_diversity": diversity,
        "summary": {
            "mean_score_margin": float(cm.get("mean_score_margin", margins.mean())),
            "std_score_margin": float(cm.get("std_score_margin", margins.std())),
            "blue_win_rate": float(metrics.get("blue_win_rate", 0)),
            "red_win_rate": float(metrics.get("red_win_rate", 0)),
            "draw_rate": float(metrics.get("draw_rate", 0)),
            "mutual_loss_rate": float(metrics.get("mutual_loss_rate", 0)),
        },
    }


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_episode_csv(path: Path, episodes: List[Dict]) -> None:
    rows = []
    for i, ep in enumerate(episodes):
        margin = ep["blue_strategic_score"] - ep["red_strategic_score"]
        rows.append(
            {
                "episode": i,
                "outcome": ep["outcome"],
                "score_margin": margin,
                "blue_strategic_score": ep["blue_strategic_score"],
                "red_strategic_score": ep["red_strategic_score"],
                "blue_reward": ep["blue_reward"],
                "red_reward": ep["red_reward"],
                "mean_tension": ep["mean_tension"],
                "final_tension": ep["final_tension"],
                "final_stability": ep["final_stability"],
                "military_balance": ep["military_balance"],
                "coalition_strength": ep["coalition_strength"],
                "resource_leverage": ep["resource_leverage"],
                "clamps": ep["clamps"],
                "commitments_blue": ep["commitments_blue"],
                "commitments_red": ep["commitments_red"],
                "blue_unique_actions": ep["blue_diversity"]["unique_actions"],
                "red_unique_actions": ep["red_diversity"]["unique_actions"],
                "blue_entropy_norm": ep["blue_diversity"]["entropy_norm"],
                "red_entropy_norm": ep["red_diversity"]["entropy_norm"],
            }
        )
    _write_csv(
        path,
        rows,
        fieldnames=list(rows[0].keys()) if rows else ["episode"],
    )


def _save_figures(
    output_dir: Path,
    episodes: List[Dict],
    coa_rows: List[Dict],
    tables: Dict,
) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    saved: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    margins = [e["blue_strategic_score"] - e["red_strategic_score"] for e in episodes]
    outcomes = [e["outcome"] for e in episodes]

    # 1. Score margin histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(margins, bins=min(15, max(5, len(margins) // 3)), color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.12, color="#C44E52", linewidth=0.8, linestyle=":")
    ax.axvline(0.12, color="#55A868", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Final Score Margin (Blue − Red)")
    ax.set_ylabel("Episodes")
    ax.set_title("Score Margin Distribution")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = output_dir / f"fig_score_margin.{ext}"
        fig.savefig(p, dpi=200)
        saved.append(str(p))
    plt.close(fig)

    # 2. Outcome bar chart
    labels = ["blue", "red", "draw", "mutual_loss"]
    counts = [outcomes.count(l) for l in labels]
    colors = ["#4C72B0", "#C44E52", "#8172B3", "#CCB974"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color=colors)
    ax.set_ylabel("Episodes")
    ax.set_title("Outcome Split")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = output_dir / f"fig_outcomes.{ext}"
        fig.savefig(p, dpi=200)
        saved.append(str(p))
    plt.close(fig)

    # 3. Top COA mix (Blue vs Red %)
    top = coa_rows[:10]
    if top:
        names = [r["action"].replace("_", "\n") for r in top]
        x = np.arange(len(top))
        width = 0.38
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, [r["blue_pct"] for r in top], width, label="Blue", color="#4C72B0")
        ax.bar(x + width / 2, [r["red_pct"] for r in top], width, label="Red", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, rotation=0)
        ax.set_ylabel("Action share (%)")
        ax.set_title("Top COA Mix (aggregated)")
        ax.legend()
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = output_dir / f"fig_coa_mix.{ext}"
            fig.savefig(p, dpi=200)
            saved.append(str(p))
        plt.close(fig)

    # 4. Mean tension trajectory (across episodes)
    max_len = max(len(e["trajectory"]["tension"]) for e in episodes)
    tension_mat = np.full((len(episodes), max_len), np.nan)
    for i, ep in enumerate(episodes):
        series = ep["trajectory"]["tension"]
        tension_mat[i, : len(series)] = series
    mean_traj = np.nanmean(tension_mat, axis=0)
    std_traj = np.nanstd(tension_mat, axis=0)
    steps = np.arange(len(mean_traj))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, mean_traj, color="#4C72B0", label="Mean tension")
    ax.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj, alpha=0.25, color="#4C72B0")
    ax.axhspan(0.45, 0.72, alpha=0.12, color="#55A868", label="Realistic band")
    ax.axhline(0.82, color="#C44E52", linestyle=":", linewidth=0.8, label="Cap (0.82)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tension")
    ax.set_title("Mean Tension Trajectory (±1 std)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = output_dir / f"fig_tension_trajectory.{ext}"
        fig.savefig(p, dpi=200)
        saved.append(str(p))
    plt.close(fig)

    # 5. Diversity comparison
    div = tables["table_diversity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    metrics_plot = ["Unique\nactions", "Entropy\n(norm)"]
    blue_vals = [div[0]["blue_value"], div[2]["blue_value"]]
    red_vals = [div[0]["red_value"], div[2]["red_value"]]
    x = np.arange(len(metrics_plot))
    width = 0.35
    ax.bar(x - width / 2, blue_vals, width, label="Blue", color="#4C72B0")
    ax.bar(x + width / 2, red_vals, width, label="Red", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_plot)
    ax.set_title("COA Diversity Summary")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = output_dir / f"fig_diversity.{ext}"
        fig.savefig(p, dpi=200)
        saved.append(str(p))
    plt.close(fig)

    return saved


def export_publication_artifacts(
    metrics: Dict,
    episodes: List[Dict],
    output_dir: str | Path,
    action_names: Sequence[str] = ACTION_NAMES,
    checkpoint: str = "v8_contest",
    save_figures: bool = True,
) -> Dict[str, str]:
    """
    Write JSON + CSV tables (+ optional figures) for publication.

    Returns dict of artifact paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = build_publication_tables(metrics, episodes, action_names, checkpoint)
    tables["meta"]["policy_mechanics"] = metrics.get("policy_mechanics")
    tables["meta"]["coa_diversity_guard"] = metrics.get("coa_diversity_guard")
    tables["meta"]["blue_model"] = metrics.get("blue_model")
    tables["meta"]["red_model"] = metrics.get("red_model")

    paths: Dict[str, str] = {}

    json_path = out / "publication_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2)
    paths["publication_json"] = str(json_path)

    main_csv = out / "table_main_results.csv"
    _write_csv(
        main_csv,
        tables["table_main_results"],
        ["metric", "value", "std", "formatted", "notes"],
    )
    paths["table_main_results_csv"] = str(main_csv)

    coa_csv = out / "table_coa_mix.csv"
    _write_csv(
        coa_csv,
        tables["table_coa_mix"],
        ["action", "blue_count", "red_count", "blue_pct", "red_pct", "blue_pct_formatted", "red_pct_formatted"],
    )
    paths["table_coa_mix_csv"] = str(coa_csv)

    div_csv = out / "table_diversity.csv"
    _write_csv(
        div_csv,
        tables["table_diversity"],
        ["metric", "blue_value", "blue_std", "red_value", "red_std", "blue_formatted", "red_formatted"],
    )
    paths["table_diversity_csv"] = str(div_csv)

    ep_csv = out / "episodes.csv"
    _write_episode_csv(ep_csv, episodes)
    paths["episodes_csv"] = str(ep_csv)

    if save_figures:
        fig_paths = _save_figures(out / "figures", episodes, tables["table_coa_mix"], tables)
        for i, fp in enumerate(fig_paths):
            paths[f"figure_{i}"] = fp

    return paths
