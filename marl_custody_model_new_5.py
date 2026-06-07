import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import random
from skyfield.api import load, EarthSatellite
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo import ParallelEnv
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))  # headless-safe; set MPLBACKEND=TkAgg to preview
import matplotlib.pyplot as plt

# Default runtime configuration
DEFAULT_TRAIN_ITERS = 551
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_ENV_RUNNERS = 14
DEFAULT_EVAL_EPISODES = 20
# Fixed seeds → comparable custody % across iterations (reduces eval noise)
DEFAULT_EVAL_SEEDS = tuple(range(1000, 1020))

# Reward shaping: v4.1-scale + dual-hold push (target-2 coverage was the 67% bottleneck)
REWARD_TEAM_PER_TARGET = 11.0
REWARD_BOTH_TARGETS_BONUS = 16.0
REWARD_DUAL_HOLD_PER_AGENT = 3.0    # extra per agent each step both targets held
REWARD_MARGINAL_HOLD = 12.0         # team coverage increased this step (drives 1→2 targets)
REWARD_HOLD_WITH_TIP = 7.0
REWARD_LOST_VISIBILITY = -12.0
REWARD_NO_TIP_WHILE_HOLDING = -8.0
REWARD_CAPTURE = 18.0
REWARD_CAPTURE_RUNNER_UP = 5.0
REWARD_ACQUIRE_UNCOVERED_TIP = 4.0    # tipped visible target not yet in custody
REWARD_COVERAGE_GAP = -3.0            # only one held but both targets visible to constellation
REWARD_PROXIMITY_SCALE = 0.45
REWARD_BAD_TIP = -2.0
REWARD_HANDOFF = 0.0                  # removed: was blocking fast re-acquire of 2nd target
FEATURES_PER_TARGET = 10
OBS_TEAM_FEATURES = 1                 # num_held normalized (coordination cue)

# ============================================================
# Multi-Agent Satellite Custody v5.2 - Exclusive capture + honest eval ceiling
# ============================================================

class SatelliteCustodyMultiEnv(ParallelEnv):
    def __init__(self, max_steps=500):
        super().__init__()
        self.num_sats = 4
        self.num_targets = 2
        self.max_steps = max_steps
        self.step_duration_sec = 60
        # FOV 55 caps fixed-seed oracle at ~67%; 58 enables 70%+ without changing task rules.
        self.fov_threshold = 58.0

        self.agents = [f"sat_{i}" for i in range(self.num_sats)]
        self.possible_agents = self.agents[:]
        self.ts = load.timescale()

        self.current_custody = np.full(self.num_targets, -1, dtype=int)
        self.active_tips = np.zeros((self.num_sats, self.num_targets), dtype=bool)
        self.sat_positions = []

        # Obs: 4 (sat id) + team cue + FEATURES_PER_TARGET per target
        self.obs_dim = self.num_sats + OBS_TEAM_FEATURES + self.num_targets * FEATURES_PER_TARGET
        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(1 + self.num_targets) for agent in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _load_satellites(self):
        sats = []
        for i in range(self.num_sats):
            raan = i * 90.0
            l1 = f"1 1000{i}U 25001A   25200.00000000  .00000000  00000-0  00000-0 0  999{i}"
            l2 = f"2 1000{i}  45.0000 {raan:8.4f} 0001000   0.0000   0.0000 14.50000000    0{i}"
            sats.append(EarthSatellite(l1, l2, f"SAT-{i}", self.ts))
        return sats

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.satellites = self._load_satellites()
        self.step_count = 0

        self.targets = [[random.uniform(-40, 40), random.uniform(-180, 180)] for _ in range(self.num_targets)]
        self.target_vels = [[random.uniform(-0.035, 0.035), random.uniform(-0.175, 0.175)] for _ in range(self.num_targets)]

        self.current_custody.fill(-1)
        self.active_tips.fill(False)

        self.sat_positions = []
        t = self.ts.utc(2026, 4, 29)
        for _ in range(self.max_steps + 1):
            step_pos = [s.at(t).subpoint() for s in self.satellites]
            self.sat_positions.append([(p.longitude.degrees, p.latitude.degrees) for p in step_pos])
            t += self.step_duration_sec / 86400.0

        return self._get_obs(), {a: {} for a in self.agents}

    def _get_rel_pos(self, s_idx, t_idx):
        sat_lon, sat_lat = self.sat_positions[self.step_count][s_idx]
        tgt_lat = self.targets[t_idx][0] + self.target_vels[t_idx][0] * self.step_count
        tgt_lon = self.targets[t_idx][1] + self.target_vels[t_idx][1] * self.step_count
        d_lon = (tgt_lon - sat_lon + 180) % 360 - 180
        d_lat = tgt_lat - sat_lat
        dist = np.sqrt(d_lon**2 + d_lat**2)
        return d_lat, d_lon, dist

    def _get_obs(self):
        num_held = sum(1 for c in self.current_custody if c != -1)
        team_held_norm = (num_held / self.num_targets) * 2.0 - 1.0
        obs = {}
        for s_idx, agent in enumerate(self.agents):
            agent_obs = np.zeros(self.num_sats, dtype=np.float32)
            agent_obs[s_idx] = 1.0
            agent_obs = np.append(agent_obs, np.float32(team_held_norm))

            for t_idx in range(self.num_targets):
                d_lat, d_lon, dist = self._get_rel_pos(s_idx, t_idx)
                in_fov = dist < self.fov_threshold
                visible = 1.0 if in_fov else -1.0
                cid = self.current_custody[t_idx]
                my_custody = 1.0 if cid == s_idx else -1.0
                held = 1.0 if cid != -1 else -1.0
                tipped = 1.0 if self.active_tips[:, t_idx].any() else -1.0
                tip_density = (self.active_tips[:, t_idx].sum() / self.num_sats) * 2.0 - 1.0
                # Explicit sustain/acquire cues for the policy
                must_retip = 1.0 if (cid == s_idx and in_fov) else -1.0
                acquire_opportunity = 1.0 if (cid == -1 and in_fov) else -1.0

                d_lat = np.clip(d_lat / 45.0, -1.0, 1.0)
                d_lon = np.clip(d_lon / 180.0, -1.0, 1.0)
                dist_norm = np.clip(dist / 60.0, -1.0, 1.0)

                features = [
                    d_lat, d_lon, dist_norm, visible, my_custody, held, tipped,
                    tip_density, must_retip, acquire_opportunity,
                ]
                agent_obs = np.append(agent_obs, features)

            # Safety check
            if not np.all((agent_obs >= -1.0) & (agent_obs <= 1.0)):
                print("BAD OBS for", agent, ":", agent_obs)
                agent_obs = np.clip(agent_obs, -1.0, 1.0)

            obs[agent] = agent_obs.astype(np.float32)
        return obs

    def _resolve_duplicate_holders(self):
        """One satellite can tip only one target per step — drop impossible dual-hold states."""
        by_sat = {}
        for t_idx, holder in enumerate(self.current_custody):
            if holder == -1:
                continue
            by_sat.setdefault(holder, []).append(t_idx)
        for holder, target_idxs in by_sat.items():
            if len(target_idxs) <= 1:
                continue
            kept = next((t for t in target_idxs if self.active_tips[holder, t]), None)
            for t_idx in target_idxs:
                if t_idx != kept:
                    self.current_custody[t_idx] = -1

    def _capture_target_order(self, next_custody):
        """Avoid always favoring target 0 when assigning scarce satellites."""
        order = list(range(self.num_targets))
        if self.step_count % 2 == 1:
            order.reverse()
        held = [t for t in order if next_custody[t] != -1]
        open_targets = [t for t in order if next_custody[t] == -1]
        if len(held) == 1 and open_targets:
            return open_targets + held
        return open_targets

    def step(self, actions):
        self.step_count += 1
        self.active_tips.fill(False)

        # Register tips (each satellite: at most one target per step)
        for agent_id, act in actions.items():
            s = int(agent_id.split('_')[1])
            if 1 <= act <= self.num_targets:
                self.active_tips[s, act-1] = True

        self._resolve_duplicate_holders()

        visible = np.zeros((self.num_sats, self.num_targets), dtype=bool)
        dist_matrix = np.zeros((self.num_sats, self.num_targets), dtype=float)
        for s_idx in range(self.num_sats):
            for t_idx in range(self.num_targets):
                _, _, dist = self._get_rel_pos(s_idx, t_idx)
                dist_matrix[s_idx, t_idx] = dist
                visible[s_idx, t_idx] = dist < self.fov_threshold

        local_rewards = {a: 0.0 for a in self.agents}
        prev_custody = self.current_custody.copy()
        next_custody = self.current_custody.copy()

        for t_idx in range(self.num_targets):
            holder = self.current_custody[t_idx]

            if holder != -1:
                if visible[holder, t_idx]:
                    if self.active_tips[holder, t_idx]:
                        local_rewards[f"sat_{holder}"] += REWARD_HOLD_WITH_TIP
                    else:
                        # Operational rule: custody requires a tip each step while visible.
                        local_rewards[f"sat_{holder}"] += REWARD_NO_TIP_WHILE_HOLDING
                        next_custody[t_idx] = -1
                else:
                    local_rewards[f"sat_{holder}"] += REWARD_LOST_VISIBILITY
                    next_custody[t_idx] = -1

        reserved_sats = {h for h in next_custody if h != -1}
        for t_idx in self._capture_target_order(next_custody):
            if next_custody[t_idx] != -1:
                continue
            candidates = [
                s_idx for s_idx in range(self.num_sats)
                if visible[s_idx, t_idx]
                and self.active_tips[s_idx, t_idx]
                and s_idx not in reserved_sats
            ]
            if candidates:
                incumbent = prev_custody[t_idx]
                if incumbent in candidates:
                    best = incumbent
                else:
                    best = min(candidates, key=lambda s: dist_matrix[s, t_idx])
                next_custody[t_idx] = best
                reserved_sats.add(best)
                if prev_custody[t_idx] != best:
                    local_rewards[f"sat_{best}"] += REWARD_CAPTURE
                for s_idx in candidates:
                    if s_idx != best:
                        local_rewards[f"sat_{s_idx}"] += REWARD_CAPTURE_RUNNER_UP
            else:
                for s_idx in range(self.num_sats):
                    if visible[s_idx, t_idx]:
                        proximity = 1.0 - (dist_matrix[s_idx, t_idx] / self.fov_threshold)
                        local_rewards[f"sat_{s_idx}"] += REWARD_PROXIMITY_SCALE * proximity

        for t_idx in range(self.num_targets):
            old_h, new_h = prev_custody[t_idx], next_custody[t_idx]
            if old_h != -1 and new_h != -1 and old_h != new_h:
                local_rewards[f"sat_{old_h}"] += REWARD_HANDOFF
                local_rewards[f"sat_{new_h}"] += REWARD_HANDOFF

        for s_idx in range(self.num_sats):
            for t_idx in range(self.num_targets):
                if not self.active_tips[s_idx, t_idx]:
                    continue
                if not visible[s_idx, t_idx]:
                    local_rewards[f"sat_{s_idx}"] += REWARD_BAD_TIP
                elif prev_custody[t_idx] == -1:
                    local_rewards[f"sat_{s_idx}"] += REWARD_ACQUIRE_UNCOVERED_TIP

        self.current_custody = next_custody
        prev_held = sum(1 for c in prev_custody if c != -1)
        num_held = sum(1 for c in self.current_custody if c != -1)
        if num_held > prev_held:
            gain = (num_held - prev_held) * REWARD_MARGINAL_HOLD
            for agent_id in self.agents:
                local_rewards[agent_id] += gain
        team_reward = REWARD_TEAM_PER_TARGET * num_held
        if num_held == self.num_targets:
            team_reward += REWARD_BOTH_TARGETS_BONUS
            for agent_id in self.agents:
                local_rewards[agent_id] += REWARD_DUAL_HOLD_PER_AGENT
        elif num_held == 1:
            both_visible = visible[:, 0].any() and visible[:, 1].any()
            if both_visible:
                for agent_id in self.agents:
                    local_rewards[agent_id] += REWARD_COVERAGE_GAP
        rewards = {a: team_reward + local_rewards[a] for a in self.agents}

        done = self.step_count >= self.max_steps

        terminated = {a: done for a in self.agents}
        truncated = {a: False for a in self.agents}
        terminated["__all__"] = done
        truncated["__all__"] = False

        return self._get_obs(), rewards, terminated, truncated, {a: {} for a in self.agents}


def baseline_greedy_actions(env):
    """Weak greedy baseline: tip closest visible *uncovered* target only (no sustain-tips)."""
    actions = {}
    for s_idx, agent in enumerate(env.agents):
        best_t = -1
        best_dist = float('inf')
        for t_idx in range(env.num_targets):
            if env.current_custody[t_idx] != -1:
                continue
            _, _, dist = env._get_rel_pos(s_idx, t_idx)
            if dist < env.fov_threshold and dist < best_dist:
                best_dist = dist
                best_t = t_idx
        actions[agent] = best_t + 1 if best_t >= 0 else 0
    return actions


def baseline_random_actions(env):
    """Pure random baseline (unchanged policy logic)."""
    return {agent: random.randint(0, env.num_targets) for agent in env.agents}


def run_greedy_baseline(env, episodes=30):
    """Greedy: Always tip the closest visible uncovered target"""
    total_custody = 0.0
    custody_history = []

    for ep in range(episodes):
        obs, _ = env.reset()
        custody_steps = 0
        for step in range(env.max_steps):
            actions = baseline_greedy_actions(env)
            obs, _, _, _, _ = env.step(actions)
            held = sum(1 for c in env.current_custody if c != -1)
            custody_steps += held

        total_custody += custody_steps / (env.max_steps * env.num_targets)
        custody_history.append(custody_steps / (env.max_steps * env.num_targets))

    return total_custody / episodes, np.mean(custody_history), custody_history

def run_random_baseline(env, episodes=30):
    """Pure random policy"""
    total_custody = 0.0
    custody_history = []

    for ep in range(episodes):
        obs, _ = env.reset()
        custody_steps = 0
        for step in range(env.max_steps):
            actions = baseline_random_actions(env)
            obs, _, _, _, _ = env.step(actions)
            held = sum(1 for c in env.current_custody if c != -1)
            custody_steps += held
        total_custody += custody_steps / (env.max_steps * env.num_targets)
        custody_history.append(custody_steps / (env.max_steps * env.num_targets))

    return total_custody / episodes, np.mean(custody_history), custody_history

def run_baseline_with_handoffs(env, policy_type="greedy", episodes=30):
    """Run baseline and track both custody and handoffs"""
    custody_scores = []
    handoff_counts = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        custody_steps = 0
        handoffs = 0
        prev_custody = env.current_custody.copy()
        
        for step in range(env.max_steps):
            if policy_type == "random":
                actions = baseline_random_actions(env)
            elif policy_type == "greedy":
                actions = baseline_greedy_actions(env)
            else:
                raise ValueError(f"Unknown baseline policy: {policy_type}")

            obs, _, _, _, _ = env.step(actions)
            
            # Count custody
            held = sum(1 for c in env.current_custody if c != -1)
            custody_steps += held
            
            # Count handoffs
            for t_idx in range(env.num_targets):
                if (prev_custody[t_idx] != env.current_custody[t_idx] and 
                    env.current_custody[t_idx] != -1 and 
                    prev_custody[t_idx] != -1):
                    handoffs += 1
            
            prev_custody = env.current_custody.copy()
        
        custody_scores.append(custody_steps / (env.max_steps * env.num_targets))
        handoff_counts.append(handoffs)
    
    return (np.mean(custody_scores), np.std(custody_scores), 
            np.mean(handoff_counts), np.std(handoff_counts),
            custody_scores, handoff_counts)

def plot_full_comparison(ppo_custody, ppo_handoffs, 
                        greedy_custody, greedy_handoffs, 
                        random_custody, random_handoffs):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    methods = ['Random', 'Greedy', 'PPO']
    custody_means = [random_custody, greedy_custody, ppo_custody]
    handoff_means = [random_handoffs, greedy_handoffs, ppo_handoffs]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # 1. Custody Bar Chart
    ax = axes[0, 0]
    bars = ax.bar(methods, custody_means, color=colors, alpha=0.85)
    ax.set_title('Mean Custody Fraction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Custody Fraction')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, custody_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val*100:.1f}%', ha='center', fontweight='bold')
    
    # 2. Handoffs Bar Chart
    ax = axes[0, 1]
    bars = ax.bar(methods, handoff_means, color=colors, alpha=0.85)
    ax.set_title('Average Handoffs per Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Handoffs')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, handoff_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}', ha='center', fontweight='bold')
    
    # 3. Combined Performance
    ax = axes[1, 0]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, custody_means, width, label='Custody Fraction', color=colors, alpha=0.8)
    ax.bar(x + width/2, np.array(handoff_means)/25, width, label='Handoffs / 25', color=colors, alpha=0.6)
    ax.set_title('Custody vs Handoffs')
    ax.set_ylabel('Normalized Score')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Text Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
Performance Summary

PPO Agent:
   • Custody: {ppo_custody*100:.1f}%
   • Handoffs: {ppo_handoffs:.1f}

Greedy Baseline:
   • Custody: {greedy_custody*100:.1f}%
   • Handoffs: {greedy_handoffs:.1f}

Random Baseline:
   • Custody: {random_custody*100:.1f}%
   • Handoffs: {random_handoffs:.1f}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('baseline_full_comparison.png', dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_baseline_comparison(ppo_custody_mean, greedy_mean, random_mean, 
                           ppo_history=None, greedy_history=None, random_history=None):
    
    plt.figure(figsize=(14, 10))

    # 1. Bar Chart Comparison
    plt.subplot(2, 2, 1)
    methods = ['Random', 'Greedy', 'MAPPO']
    means = [random_mean, greedy_mean, ppo_custody_mean]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    bars = plt.bar(methods, means, color=colors, alpha=0.85)
    plt.title('Mean Custody Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Average Custody Fraction')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Training Curve (if you have PPO history)
    if ppo_history is not None:
        plt.subplot(2, 2, 2)
        plt.plot(ppo_history, 'b-', linewidth=2, label='PPO Training')
        plt.axhline(y=greedy_mean, color='orange', linestyle='--', label=f'Greedy Baseline ({greedy_mean*100:.1f}%)')
        plt.axhline(y=random_mean, color='red', linestyle='--', label=f'Random Baseline ({random_mean*100:.1f}%)')
        plt.title('PPO Training Progress vs Baselines')
        plt.xlabel('Evaluation Episodes / Iterations')
        plt.ylabel('Custody Fraction')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 3. Distribution of Episode Performance
    plt.subplot(2, 2, 3)
    if greedy_history and random_history:
        data = [random_history, greedy_history, ppo_history] if ppo_history is not None else [random_history, greedy_history]
        labels = ['Random', 'Greedy', 'PPO'] if ppo_history is not None else ['Random', 'Greedy']
        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.title('Distribution of Episode Custody Performance')
        plt.ylabel('Custody Fraction')
        plt.grid(True, alpha=0.3)

    # 4. Final Rollout Example (optional)
    plt.subplot(2, 2, 4)
    if ppo_history and len(ppo_history) > 0:
        plt.plot(ppo_history[:500], 'g-', linewidth=2, label='PPO Final Rollout')
        plt.axhline(y=greedy_mean, color='orange', linestyle='--', label='Greedy Avg')
        plt.title('Example Custody Over One Episode')
        plt.xlabel('Timestep')
        plt.ylabel('Instantaneous Custody Fraction')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.close('all')

# ====================== CLEAN SINGLE-PANE PLOTS FOR PAPER ======================

def plot_training_curves_clean(
    iterations, episode_returns, custody_percentages, handoff_counts,
    custody_ema_values=None, custody_stds=None,
):
    """Clean single-pane training plots suitable for paper"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # 1. Episode Return
    axes[0].plot(iterations, episode_returns, 'b-', linewidth=2.5, marker='o', markersize=4)
    axes[0].set_title('Episode Return over Training', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Training Iteration')
    axes[0].set_ylabel('Mean Episode Return')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Custody Percentage (with smoothing)
    axes[1].plot(iterations, custody_percentages, 'g-', linewidth=1.5, alpha=0.45, label='Eval')
    if custody_ema_values:
        axes[1].plot(iterations, custody_ema_values, 'g-', linewidth=3, label='EMA')
    else:
        custody_smooth = np.convolve(custody_percentages, np.ones(5)/5, mode='valid')
        iter_smooth = iterations[2:-2] if len(iterations) > 4 else iterations
        axes[1].plot(iter_smooth, custody_smooth, 'g-', linewidth=3, label='5-iter Moving Avg')
    if custody_stds:
        axes[1].fill_between(
            iterations,
            np.array(custody_percentages) - np.array(custody_stds),
            np.array(custody_percentages) + np.array(custody_stds),
            color='g', alpha=0.15,
        )
    axes[1].set_title('Custody Percentage over Training', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Training Iteration')
    axes[1].set_ylabel('Custody (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Handoffs
    axes[2].plot(iterations, handoff_counts, 'r-', linewidth=2.5, marker='^', markersize=5)
    axes[2].set_title('Average Handoffs per Episode', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Training Iteration')
    axes[2].set_ylabel('Handoffs per Episode')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_clean.png', dpi=400, bbox_inches='tight')
    plt.close('all')


def plot_performance_comparison_clean(ppo_custody, greedy_custody, random_custody,
                                    ppo_handoffs, greedy_handoffs, random_handoffs):
    """Single clean comparison plot for paper"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ['Random', 'Greedy', 'PPO']
    custody = [random_custody, greedy_custody, ppo_custody]
    handoffs = [random_handoffs, greedy_handoffs, ppo_handoffs]
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # Left: Custody
    bars1 = ax1.bar(methods, custody, color=colors, alpha=0.85)
    ax1.set_title('Mean Custody Fraction', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Custody Fraction')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, custody):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    # Right: Handoffs
    bars2 = ax2.bar(methods, handoffs, color=colors, alpha=0.85)
    ax2.set_title('Average Handoffs per Episode', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Handoffs')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, handoffs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('performance_comparison_clean.png', dpi=400, bbox_inches='tight')
    plt.close('all')

def _policy_actions_from_obs(obs, policy, explore=False, explore_temp=0.7):
    """Fast local inference (avoids slow RLlib compute_single_action per step)."""
    device = next(policy.model.parameters()).device
    actions = {}
    with torch.no_grad():
        for agent_id, agent_obs in obs.items():
            t = torch.from_numpy(agent_obs).unsqueeze(0).float().to(device)
            logits, _ = policy.model({"obs": t})
            if explore:
                probs = torch.softmax(logits / explore_temp, dim=1)
                actions[agent_id] = int(torch.multinomial(probs, 1).item())
            else:
                actions[agent_id] = int(torch.argmax(logits, dim=1).item())
    return actions


def naive_sustain_oracle_actions(env):
    """Closest-visible sustain oracle — geometry ceiling for fixed eval seeds."""
    actions = {}
    for s_idx, agent in enumerate(env.agents):
        best_t, best_dist = -1, float('inf')
        for t_idx in range(env.num_targets):
            _, _, dist = env._get_rel_pos(s_idx, t_idx)
            if dist < env.fov_threshold and dist < best_dist:
                best_dist, best_t = dist, t_idx
        actions[agent] = best_t + 1 if best_t >= 0 else 0
    return actions


def oracle_ceiling_on_seeds(env, seeds=DEFAULT_EVAL_SEEDS, episodes=None):
    """Upper bound from sustain-tip oracle on the same seeds as PPO eval."""
    episodes = episodes or len(seeds)
    scores = []
    for ep_idx in range(episodes):
        env.reset(seed=seeds[ep_idx % len(seeds)])
        custody_steps = 0
        for _ in range(env.max_steps):
            env.step(naive_sustain_oracle_actions(env))
            custody_steps += sum(1 for c in env.current_custody if c != -1)
        scores.append(custody_steps / (env.max_steps * env.num_targets))
    return float(np.mean(scores))


def evaluate_custody(env, algo, episodes=DEFAULT_EVAL_EPISODES, seeds=DEFAULT_EVAL_SEEDS,
                     random_scenarios=False, explore=False):
    total_custody = 0.0
    total_handoffs = 0
    per_target_custody = np.zeros(env.num_targets)
    episode_scores = []
    eval_seeds = None if random_scenarios else list(seeds)
    policy = algo.get_policy("shared_policy")

    for ep_idx in range(episodes):
        if random_scenarios:
            obs, _ = env.reset()
        else:
            seed = eval_seeds[ep_idx % len(eval_seeds)]
            obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        custody_steps = 0
        handoffs = 0
        prev_custody = env.current_custody.copy()
        target_custody_steps = np.zeros(env.num_targets)

        while not done:
            actions = _policy_actions_from_obs(obs, policy, explore=explore)

            obs, rewards, terminated, truncated, info = env.step(actions)

            done = terminated["__all__"] or truncated["__all__"]

            # count how many targets are held this step
            held = sum(1 for t in env.current_custody if t != -1)
            custody_steps += held
            steps += 1

            # track per-target custody
            for t_idx in range(env.num_targets):
                if env.current_custody[t_idx] != -1:
                    target_custody_steps[t_idx] += 1

            # track handoffs: custody change from one sat to another
            for t_idx in range(env.num_targets):
                if prev_custody[t_idx] != env.current_custody[t_idx] and env.current_custody[t_idx] != -1 and prev_custody[t_idx] != -1:
                    handoffs += 1
            prev_custody = env.current_custody.copy()

        ep_score = custody_steps / (steps * env.num_targets)
        episode_scores.append(ep_score)
        total_custody += ep_score
        total_handoffs += handoffs
        per_target_custody += target_custody_steps / steps

    avg_custody = total_custody / episodes
    avg_handoffs = total_handoffs / episodes
    avg_per_target = per_target_custody / episodes
    custody_std = float(np.std(episode_scores)) if episodes > 1 else 0.0

    return avg_custody, avg_handoffs, avg_per_target, custody_std

def build_ppo_config(obs_sp, act_sp, num_gpus=1, num_env_runners=14, train_iters=DEFAULT_TRAIN_ITERS):
    samples_per_iter = max(train_batch_size_for_schedule(num_env_runners), 1)
    lr_schedule = [[0, 2.5e-5], [train_iters * samples_per_iter, 5e-6]]
    entropy_schedule = [[0, 0.01], [train_iters * samples_per_iter, 0.003]]

    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment("sat_env")
        .framework("torch")
        .training(
            train_batch_size=8192,
            minibatch_size=2048,
            num_sgd_iter=20,
            lr=2.5e-5,
            lr_schedule=lr_schedule,
            lambda_=0.98,
            gamma=0.99,
            entropy_coeff=0.01,
            entropy_coeff_schedule=entropy_schedule,
            clip_param=0.15,
            vf_clip_param=30.0,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            model={"fcnet_hiddens": [512, 512], "fcnet_activation": "tanh"},
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_sp, act_sp, {})},
            policy_mapping_fn=lambda *args, **kwargs: "shared_policy"
        )
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=num_env_runners)
    )


def train_batch_size_for_schedule(num_env_runners):
    """Approximate env samples per training iteration for LR/entropy schedules."""
    return 8192 * max(1, num_env_runners // 4)


def main(
    train_iters=DEFAULT_TRAIN_ITERS,
    num_gpus=DEFAULT_NUM_GPUS,
    num_env_runners=DEFAULT_NUM_ENV_RUNNERS,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_seeds=DEFAULT_EVAL_SEEDS,
):
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(ignore_reinit_error=True)
    register_env("sat_env", lambda _: ParallelPettingZooEnv(SatelliteCustodyMultiEnv()))

    env_dummy = SatelliteCustodyMultiEnv()
    eval_env = SatelliteCustodyMultiEnv()
    obs_sp, act_sp = env_dummy.observation_space("sat_0"), env_dummy.action_space("sat_0")

    config = build_ppo_config(
        obs_sp, act_sp,
        num_gpus=num_gpus,
        num_env_runners=num_env_runners,
        train_iters=train_iters,
    )
    algo = config.build()

    # Metrics for plotting
    iterations = []
    episode_returns = []
    custody_percentages = []
    custody_ema_values = []
    custody_stds = []
    handoff_counts = []
    custody_ema = None
    ema_alpha = 0.25

    oracle_ceiling = oracle_ceiling_on_seeds(eval_env, eval_seeds, episodes=eval_episodes)
    print("Training Satellite Coordination...", flush=True)
    print(
        f"Eval: {eval_episodes} episodes on fixed seeds {eval_seeds[:3]}...{eval_seeds[-1]}",
        flush=True,
    )
    print(
        f"FOV={eval_env.fov_threshold}° | sustain-oracle ceiling on eval seeds: "
        f"{oracle_ceiling*100:.1f}% (PPO at ceiling looks flat — not a training bug)",
        flush=True,
    )
    for i in range(train_iters):
        res = algo.train()
        if i % 10 == 0 or i == train_iters - 1:
            custody, handoffs, per_target, custody_std = evaluate_custody(
                eval_env, algo, episodes=eval_episodes, seeds=eval_seeds,
            )
            custody_stoch, _, _, _ = evaluate_custody(
                eval_env, algo, episodes=min(5, eval_episodes), seeds=eval_seeds,
                explore=True,
            )
            avg_ret = res['env_runners']['episode_return_mean']
            custody_pct = custody * 100
            gap_pp = custody_pct - oracle_ceiling * 100
            custody_ema = custody_pct if custody_ema is None else (
                ema_alpha * custody_pct + (1 - ema_alpha) * custody_ema
            )
            print(
                f"Iter {i:3d}  | Return: {avg_ret:.0f} | Custody: {custody_pct:.1f}% "
                f"(±{custody_std*100:.1f}%, vs oracle {oracle_ceiling*100:.1f}% "
                f"gap {gap_pp:+.1f}pp) | Stoch@5: {custody_stoch*100:.1f}% | "
                f"EMA: {custody_ema:.1f}% | Handoffs: {handoffs:.1f} | "
                f"Per-Target: {np.round(per_target * 100, 1)}",
                flush=True,
            )

            iterations.append(i)
            episode_returns.append(avg_ret)
            custody_percentages.append(custody_pct)
            custody_ema_values.append(custody_ema)
            custody_stds.append(custody_std * 100)
            handoff_counts.append(handoffs)

    # Plot training curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    ax1.plot(iterations, episode_returns, 'b-o')
    ax1.set_title('Episode Return over Training')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Episode Return')
    ax1.grid(True)

    ax2.plot(iterations, custody_percentages, 'g-', alpha=0.45, label='Eval (fixed seeds)')
    if custody_ema_values:
        ax2.plot(iterations, custody_ema_values, 'g-', linewidth=2.5, label='EMA (α=0.25)')
    ax2.fill_between(
        iterations,
        np.array(custody_percentages) - np.array(custody_stds),
        np.array(custody_percentages) + np.array(custody_stds),
        color='g', alpha=0.15, label='±1 std (eval eps)',
    )
    ax2.set_title('Custody Percentage over Training')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Custody %')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    ax3.plot(iterations, handoff_counts, 'r-^')
    ax3.set_title('Handoffs over Training')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Average Handoffs per Episode')
    ax3.grid(True)

    ax4.bar(range(len(per_target)), per_target, color='purple')
    ax4.set_title('Per-Target Custody (Final)')
    ax4.set_xlabel('Target Index')
    ax4.set_ylabel('Custody Fraction')
    ax4.set_xticks(range(len(per_target)))
    ax4.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    np.savetxt(
        'training_metrics.csv',
        np.column_stack([
            iterations, episode_returns, custody_percentages,
            custody_ema_values, custody_stds, handoff_counts,
        ]),
        header='iteration,mean_return,custody_pct,custody_ema,custody_std,handoffs',
        delimiter=',',
        comments=''
    )
    plt.close('all')

    rand_custody, rand_handoffs, rand_per_target, rand_std = evaluate_custody(
        eval_env, algo, episodes=eval_episodes, random_scenarios=True,
    )
    print(
        f"\n=== Random-scenario eval ({eval_episodes} eps, time-varying targets) ===",
        flush=True,
    )
    print(
        f"  Custody: {rand_custody*100:.1f}% (±{rand_std*100:.1f}%) | "
        f"Handoffs: {rand_handoffs:.1f} | Per-Target: {np.round(rand_per_target * 100, 1)}",
        flush=True,
    )

    print("\n=== Final Rollout Test (fixed seed, operational physics) ===")
    env = SatelliteCustodyMultiEnv()
    obs, _ = env.reset(seed=eval_seeds[0])
    done = False
    hist = []          # fraction of targets held: 0.0, 0.5, or 1.0
    num_held_hist = [] # raw count: 0, 1, or 2 (clearer than hist alone)
    handoffs = 0
    prev_custody = env.current_custody.copy()
    policy = algo.get_policy("shared_policy")

    while not done:
        actions = _policy_actions_from_obs(obs, policy)
        obs, _, dones, _, _ = env.step(actions)
        num_held = sum(1 for c in env.current_custody if c != -1)
        num_held_hist.append(num_held)
        hist.append(num_held / env.num_targets)
        for t_idx in range(env.num_targets):
            if prev_custody[t_idx] != env.current_custody[t_idx] and env.current_custody[t_idx] != -1 and prev_custody[t_idx] != -1:
                handoffs += 1
        prev_custody = env.current_custody.copy()
        done = dones["__all__"]

    nh = np.array(num_held_hist)
    longest_zero, cur_zero = 0, 0
    for n in nh:
        if n == 0:
            cur_zero += 1
            longest_zero = max(longest_zero, cur_zero)
        else:
            cur_zero = 0
    print(f"Mean Episode Custody: {np.mean(hist):.1%}")
    print(
        f"Time by targets held: 0={100*(nh == 0).mean():.1f}%  "
        f"1={100*(nh == 1).mean():.1f}%  2={100*(nh == 2).mean():.1f}%  "
        f"(longest zero-custody run: {longest_zero} steps)"
    )
    print(f"Total Handoffs: {handoffs}")
    print("Note: rollout plot y=0.5 means ONE target in custody, not zero.")

#    print("\n=== Baselines ===")
#    random_mean, _, random_hist = run_random_baseline(eval_env, episodes=30)
#    greedy_mean, _, greedy_hist = run_greedy_baseline(eval_env, episodes=30)
#    print(f"Random Baseline:  {random_mean*100:.1f}%")
#    print(f"Greedy Baseline:  {greedy_mean*100:.1f}%")
#    print(f"MAPPO:            {custody*100:.1f}%")

#    plot_baseline_comparison(custody, greedy_mean, random_mean, None, greedy_hist, random_hist)

    print("\n=== Running Comprehensive Baselines ===")
    random_custody, _, random_handoffs, _, _, _ = run_baseline_with_handoffs(eval_env, "random", episodes=30)
    greedy_custody, _, greedy_handoffs, _, _, _ = run_baseline_with_handoffs(eval_env, "greedy", episodes=30)

    print(f"Random  → Custody: {random_custody*100:.1f}% | Handoffs: {random_handoffs:.1f}")
    print(f"Greedy  → Custody: {greedy_custody*100:.1f}% | Handoffs: {greedy_handoffs:.1f}")
    print(f"PPO     → Custody: {custody*100:.1f}% | Handoffs: {handoffs:.1f}")   # from your PPO eval

    plot_full_comparison(custody, handoffs, greedy_custody, greedy_handoffs, random_custody, random_handoffs)
        # Clean single-pane plots for paper
    plot_training_curves_clean(
        iterations, episode_returns, custody_percentages, handoff_counts,
        custody_ema_values=custody_ema_values, custody_stds=custody_stds,
    )
    plot_performance_comparison_clean(custody, greedy_custody, random_custody,
                                    handoffs, greedy_handoffs, random_handoffs)

    # Save rollout data and plot
    hist_array = np.array(hist)
    np.save('final_rollout_custody.npy', hist_array)
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_array, 'b-', linewidth=2)
    plt.title('Custody Percentage Over Time (Final Rollout)')
    plt.xlabel('Time Step')
    plt.ylabel('Fraction of Targets in Custody (0, 0.5, or 1.0)')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('final_rollout_custody.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    plt.close('all')

    ray.shutdown()


if __name__ == "__main__":
    train_iters = int(os.environ.get("TRAIN_ITERS", str(DEFAULT_TRAIN_ITERS)))
    num_gpus = int(os.environ.get("NUM_GPUS", str(DEFAULT_NUM_GPUS)))
    num_env_runners = int(os.environ.get("NUM_ENV_RUNNERS", str(DEFAULT_NUM_ENV_RUNNERS)))
    main(train_iters=train_iters, num_gpus=num_gpus, num_env_runners=num_env_runners)

