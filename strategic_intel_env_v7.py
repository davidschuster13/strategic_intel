"""
⚠️  DEPRECATED: Use strategic_intel_env_v8.py instead
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This v7 environment is LEGACY code kept for baseline comparison only.

v8 IMPROVEMENTS (use v8 instead):
  ✓ Enhanced state space: 5 dimensions (military/economic/territorial/alliance/resources)
  ✓ Multi-domain actions: 24 actions (kinetic/cyber/economic/information)
  ✓ Realistic escalation: Schelling points + irreversible commitments + de-escalation
  ✓ Cross-domain spillover: Actions trigger responses in other domains
  ✓ Domain-specific observations: Asymmetric sensor models

MIGRATE TO v8:
  1. Replace: from strategic_intel_env_v8 import StrategicWargameEnvV8
  2. Use: env = StrategicWargameEnvV8() instead of StrategicWargameEnv()
  3. Run: python train_self_play_v8.py

For v7 baseline comparison, see: legacy/LEGACY.md
"""

import gymnasium as gym
import numpy as np
from typing import Dict

from .load_parameters import load_all_parameters

ACTION_NAMES = {
    0: "Passive Monitoring / ISR",
    1: "Diplomatic Engagement",
    2: "Increase Intelligence Sharing",
    3: "Calibrated Domain Awareness",
    4: "Measured Military Posturing",
    5: "Economic / Sanctions Pressure",
    6: "Forward Deployment / Exercises",
    7: "Aggressive Show of Force",
    8: "Cyber / Hybrid Operations",
    9: "High-Risk Escalation Response"
}
ACTION_ESCALATION = np.array([0.0, 0.1, 0.15, 0.25, 0.4, 0.45, 0.65, 0.8, 0.75, 1.0], dtype=np.float32)

class StrategicWargameEnv(gym.Env):
    """v7 - Clean Self-Play Ready"""
    def __init__(self, max_steps=40):
        super().__init__()
        self.params = load_all_parameters()
        self.blue_params = self.params.get("Blue_US", {}) or {}
        self.red_params = self.params.get("Red_Adversary", {}) or {}
        self.blue_bias = self._build_side_bias(self.blue_params, "Blue")
        self.red_bias = self._build_side_bias(self.red_params, "Red")

        self.max_steps = max_steps
        self.current_step = 0
        
        self.tension = 0.35
        self.blue_awareness = 0.75
        self.red_influence = 0.45
        self.stability = 1.0
        self.last_blue_action = 0
        self.last_red_action = 0
        
        self.action_space = gym.spaces.Discrete(10)
        # Single Box for both agents (simpler for self-play wrapper)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(32,), dtype=np.float32)
        
        print("✅ Strategic Wargame v7 - Self-Play Ready")

    def _normalize_priority_map(self, params: Dict):
        priorities = params.get("capability_priorities", {})
        if not isinstance(priorities, dict):
            return {}

        normalized = {}
        for key, value in priorities.items():
            try:
                normalized[str(key).lower()] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized

    def _build_side_bias(self, params: Dict, side: str):
        priorities = self._normalize_priority_map(params)
        militarized_keywords = ("military", "force", "projection", "strike", "deployment", "naval")
        hybrid_keywords = ("cyber", "hybrid", "jamming", "intelligence")
        deescalation_keywords = ("diplomacy", "stability", "deescalation", "sanction", "coordination")

        militarized = sum(v for k, v in priorities.items() if any(token in k for token in militarized_keywords))
        hybrid = sum(v for k, v in priorities.items() if any(token in k for token in hybrid_keywords))
        deescalatory = sum(v for k, v in priorities.items() if any(token in k for token in deescalation_keywords))

        red_lines = params.get("red_lines", [])
        escalation_steps = params.get("escalation_ladder", [])

        assertiveness = np.clip(0.2 + 0.35 * militarized + 0.15 * hybrid + 0.03 * len(escalation_steps), 0.0, 1.0)
        restraint = np.clip(0.2 + 0.3 * deescalatory - 0.2 * militarized + 0.02 * len(red_lines), 0.0, 1.0)
        risk_tolerance = np.clip(0.2 + 0.4 * militarized + 0.2 * hybrid - 0.1 * deescalatory, 0.0, 1.0)

        # If no extracted signals exist for this side, fall back to a moderate profile.
        if not priorities and not red_lines and not escalation_steps:
            assertiveness, restraint, risk_tolerance = 0.45, 0.45, 0.45

        if side == "Blue":
            risk_tolerance *= 0.85
        return {
            "assertiveness": float(assertiveness),
            "restraint": float(restraint),
            "risk_tolerance": float(risk_tolerance),
        }
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        # Curriculum starts: mostly baseline, with some low- and high-tension starts.
        regime = np.random.choice(["low", "baseline", "high"], p=[0.25, 0.5, 0.25])
        if regime == "low":
            self.tension = float(np.random.uniform(0.18, 0.32))
            self.blue_awareness = float(np.random.uniform(0.72, 0.9))
            self.red_influence = float(np.random.uniform(0.25, 0.42))
            self.stability = float(np.random.uniform(1.0, 1.15))
        elif regime == "high":
            self.tension = float(np.random.uniform(0.52, 0.7))
            self.blue_awareness = float(np.random.uniform(0.55, 0.78))
            self.red_influence = float(np.random.uniform(0.5, 0.72))
            self.stability = float(np.random.uniform(0.82, 1.0))
        else:
            self.tension = float(np.random.uniform(0.3, 0.45))
            self.blue_awareness = float(np.random.uniform(0.68, 0.82))
            self.red_influence = float(np.random.uniform(0.38, 0.52))
            self.stability = float(np.random.uniform(0.92, 1.08))
        self.last_blue_action = 0
        self.last_red_action = 0
        obs = self._get_obs()
        return obs, {}   
    
    def step(self, actions: Dict[str, int]):
        self.current_step += 1
        blue_act = actions["Blue"]
        red_act = actions["Red"]

        blue_escalation = float(ACTION_ESCALATION[blue_act])
        red_escalation = float(ACTION_ESCALATION[red_act])

        blue_effect = (0.55 * self.blue_bias["assertiveness"] - 0.65 * self.blue_bias["restraint"]) * blue_escalation
        red_effect = (0.85 * self.red_bias["assertiveness"] - 0.25 * self.red_bias["restraint"]) * red_escalation
        doctrine_drift = red_effect - blue_effect

        crisis_multiplier = 1.0 + max(0.0, self.tension - 0.7) * 1.6
        interaction_push = (red_escalation - blue_escalation) * 0.23 * crisis_multiplier
        joint_escalation = max(0.0, (blue_escalation + red_escalation) - 1.3) * 0.06
        self.tension += interaction_push + joint_escalation + doctrine_drift + np.random.normal(0, 0.03)
        self.tension += (0.35 - self.tension) * 0.02  # mild reversion in absence of pressure

        # Exogenous shocks: occasional incidents that perturb the strategic picture.
        shock = 0.0
        if np.random.random() < (0.015 + 0.04 * max(0.0, self.tension - 0.6)):
            shock = float(np.random.normal(0.04, 0.025))
            self.tension += shock
            self.red_influence += 0.25 * shock
            self.blue_awareness += 0.18 * shock

        self.tension = np.clip(self.tension, 0.0, 1.0)

        blue_awareness_gain = 0.04 + 0.03 * self.blue_bias["restraint"]
        self.blue_awareness += blue_awareness_gain if blue_act <= 4 else -0.03 * (1.0 + self.blue_bias["risk_tolerance"])

        red_influence_gain = 0.05 + 0.04 * self.red_bias["assertiveness"]
        self.red_influence += red_influence_gain if red_act >= 6 else -0.03 * (1.0 - self.red_bias["assertiveness"])
        # Influence becomes harder to accrue in crisis/instability; can reverse under severe stress.
        crisis_drag = max(0.0, self.tension - 0.65) * 0.06
        fragility_drag = max(0.0, 0.55 - self.stability) * 0.15
        saturation_drag = max(0.0, self.red_influence - 0.8) * 0.10
        self.red_influence -= crisis_drag + fragility_drag + saturation_drag

        self.blue_awareness = np.clip(self.blue_awareness, 0.0, 1.2)
        self.red_influence = np.clip(self.red_influence, 0.0, 1.2)
        self.stability -= self.tension * 0.022
        self.stability -= joint_escalation * 0.18
        self.stability = np.clip(self.stability, 0.0, 1.2)
        self.last_blue_action = blue_act
        self.last_red_action = red_act
        
        obs = self._get_obs()
        rewards = self._compute_rewards(blue_act, red_act)
        
        terminated = self.current_step >= self.max_steps or self.tension > 0.95 or self.stability < 0.15
        info = {
            "tension": self.tension,
            "stability": self.stability,
            "red_influence": self.red_influence,
            "blue_awareness": self.blue_awareness,
            "shock": shock,
        }
        
        return obs, rewards, {"Blue": terminated, "Red": terminated}, {"Blue": False, "Red": False}, info
    
    def _get_obs(self):
        """Asymmetric observations: each agent sees own policy bias, not opponent's."""
        # Shared environment state (visible to both)
        shared = np.zeros(32, dtype=np.float32)
        shared[0] = self.tension
        shared[1] = self.blue_awareness
        shared[2] = self.red_influence
        shared[3] = self.stability
        shared[4] = self.current_step / self.max_steps
        shared[5] = ACTION_ESCALATION[self.last_blue_action]
        shared[6] = ACTION_ESCALATION[self.last_red_action]
        
        # Blue sees own parameters, Red's actions but not Red's introspection
        blue_obs = shared.copy()
        blue_obs[7] = self.blue_bias["assertiveness"]
        blue_obs[8] = self.blue_bias["restraint"]
        blue_obs[9] = self.blue_bias["risk_tolerance"]
        blue_obs[10:13] = 0  # Blue doesn't see Red's internal state
        
        # Red sees own parameters, Blue's actions but not Blue's introspection
        red_obs = shared.copy()
        red_obs[7:10] = 0  # Red doesn't see Blue's internal state
        red_obs[10] = self.red_bias["assertiveness"]
        red_obs[11] = self.red_bias["restraint"]
        red_obs[12] = self.red_bias["risk_tolerance"]
        
        return {"Blue": blue_obs, "Red": red_obs}
    
    def _compute_rewards(self, blue_act, red_act):
        """Compute rewards grounded in policy objectives and strategic dynamics.
        
        BLUE OBJECTIVES (from parameterized US/NATO policy):
        - Minimize escalation/tension (deterrence without provoking)
        - Maintain awareness of adversary actions
        - Preserve stability of strategic order
        - Prevent Red influence expansion
        
        RED OBJECTIVES (from parameterized adversary policy):
        - Maximize regional influence (limited expansion)
        - Maintain strategic ambiguity (keep opponents uncertain)
        - Exploit stability gaps (only when order won't collapse)
        - Avoid systemic breakdown (too much cost)
        
        Reward structure:
        - Base reward: state payoff aligned with objective
        - Action preference: bonus/penalty based on escalation level
        - Cost functions: penalties for mutual escalation & crisis
        - Temporal discount: small penalty per step (encourages efficiency)
        """
        blue_r = 5.5 * (1 - self.tension) + 3.5 * self.blue_awareness + 2.5 * self.stability
        red_r = 4.5 * self.tension + 3.5 * self.red_influence
        blue_deterrence = 1.0 - self.red_influence

        blue_escalation = float(ACTION_ESCALATION[blue_act])
        red_escalation = float(ACTION_ESCALATION[red_act])
        blue_pref_low_escalation = 1.0 - blue_escalation
        red_pref_high_escalation = red_escalation
        blue_r += 1.2 * self.blue_bias["restraint"] * blue_pref_low_escalation
        blue_r += 0.7 * self.blue_bias["assertiveness"] * blue_escalation
        blue_r -= 0.9 * self.blue_bias["risk_tolerance"] * max(0.0, self.tension - 0.75)

        red_r += 1.0 * self.red_bias["assertiveness"] * red_pref_high_escalation
        red_r -= 0.8 * self.red_bias["restraint"] * red_pref_high_escalation
        red_r += 0.7 * self.red_bias["risk_tolerance"] * max(0.0, self.tension - 0.6)

        # Escalation has real costs, especially during crisis.
        escalation_cost = (blue_escalation + red_escalation) * (0.25 + 0.9 * max(0.0, self.tension - 0.55))
        blue_r -= 0.5 * escalation_cost
        red_r -= 0.5 * escalation_cost

        # Blue objective: preserve stability while containing adversary influence.
        blue_r += 1.6 * blue_deterrence
        blue_r += 0.9 * self.stability * (1.0 - self.tension)

        # Red objective: gain influence, but only if broader order does not collapse.
        red_r += 1.1 * self.red_influence * (0.35 + 0.65 * self.stability)
        red_r -= 1.1 * max(0.0, 0.45 - self.stability)
        red_r -= 1.6 * max(0.0, self.red_influence - 0.85) * max(0.0, 0.75 - self.stability)

        # Prolonged crisis has mounting costs for both actors.
        crisis_level = max(0.0, self.tension - 0.72)
        blue_r -= 1.0 * crisis_level
        red_r -= 1.25 * crisis_level

        # Brinkmanship beyond near-crisis is especially costly.
        near_breakdown = max(0.0, self.tension - 0.88)
        blue_r -= 1.4 * near_breakdown
        red_r -= 1.7 * near_breakdown

        # De-escalation diplomacy has value when tension is already high.
        if self.tension > 0.7 and blue_act <= 2:
            blue_r += 0.7
        if self.tension > 0.7 and red_act <= 2:
            red_r += 0.5

        blue_r -= 0.1 * self.current_step
        red_r -= 0.09 * self.current_step
        return {"Blue": float(blue_r), "Red": float(red_r)}
