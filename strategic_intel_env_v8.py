"""
Strategic Wargame Environment v8 - Enhanced Multi-Domain + Realistic Escalation
Integrates:
- Enhanced state space (military, economic, territorial, alliances, resources)
- Multi-domain operations (kinetic, cyber, economic, information)
- Realistic escalation pathways (Schelling points, irreversible actions, de-escalation)
- Policy-document constraints (merged corpora, ladders, red lines, commitments)
"""

import gymnasium as gym
import numpy as np
from typing import Dict

from .load_parameters import load_all_parameters
from .strategic_state_manager import StrategicStateManager
from .multi_domain_operations import (
    MultiDomainActionLibrary, CrossDomainSpilloverModel, Domain, ActionEffect
)
from .escalation_pathways import (
    SchelllingPointManager, DeescalationMechanism, EscalationCommitment,
)
from .policy_mechanics import (
    SidePolicyProfile,
    build_action_cost_table,
    map_action_to_commitment,
    observe_for_side,
)


class StrategicWargameEnvV8(gym.Env):
    """v8 - Multi-Domain Enhanced with Realistic Escalation Dynamics"""

    def __init__(
        self,
        max_steps=60,
        policy_mechanics: bool = True,
        quiet: bool = False,
        coa_diversity_guard: bool = False,
    ):
        super().__init__()
        self.policy_mechanics = policy_mechanics
        self.quiet = quiet
        self.coa_diversity_guard = coa_diversity_guard

        self.params = load_all_parameters(quiet=quiet)
        self.blue_params = self.params.get("Blue_US", {}) or {}
        self.red_params = self.params.get("Red_Adversary", {}) or {}
        if policy_mechanics:
            self.blue_bias = self._build_side_bias(self.blue_params, "Blue")
            self.red_bias = self._build_side_bias(self.red_params, "Red")
        else:
            self.blue_bias = {"assertiveness": 0.45, "restraint": 0.45, "risk_tolerance": 0.45}
            self.red_bias = {"assertiveness": 0.45, "restraint": 0.45, "risk_tolerance": 0.45}

        all_actions = MultiDomainActionLibrary.get_all_actions()
        self.action_names = list(all_actions.keys())
        self.action_costs = build_action_cost_table()
        self.num_actions = len(self.action_names)
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.blue_policy = SidePolicyProfile.from_merged(
            "Blue", self.blue_params, self.action_names, self.action_costs
        )
        self.red_policy = SidePolicyProfile.from_merged(
            "Red", self.red_params, self.action_names, self.action_costs
        )

        self.strategic_state = StrategicStateManager()
        self.spillover_model = CrossDomainSpilloverModel()
        self.schelling_manager = SchelllingPointManager()
        self.deescalation = DeescalationMechanism()

        self.tension = 0.35
        self.stability = 1.0
        self.max_steps = max_steps
        self.current_step = 0
        self.last_blue_action = 0
        self.last_red_action = 0
        self._last_policy_penalties = {"Blue": 0.0, "Red": 0.0}
        self._blue_repeat_streak = 0
        self._red_repeat_streak = 0
        self._blue_recent_actions: list[int] = []
        self._red_recent_actions: list[int] = []
        self._prev_contest_dims: tuple[float, float, float] | None = None
        self._recent_window = 10
        self._blue_info_spam_count = 0
        self._episode_blue_score_baseline = 0.0
        self._episode_red_score_baseline = 0.0
        self._prev_score_margin: float | None = None
        try:
            self._info_sharing_idx = self.action_names.index("info_intelligence_sharing")
        except ValueError:
            self._info_sharing_idx = -1

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(64,), dtype=np.float32)

        if not quiet:
            print("✅ Strategic Wargame v8 - Multi-Domain Enhanced")
            print(f"   Actions: {self.num_actions} multi-domain")
            print(
                f"   Policy mechanics: {'on' if policy_mechanics else 'off (ablation)'} | "
                f"Blue {self.blue_policy.merged.get('n_source_documents', 0)} docs, "
                f"Red {self.red_policy.merged.get('n_source_documents', 0)} docs"
            )

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
        hybrid_keywords = ("cyber", "hybrid", "jamming", "intelligence", "sensor")
        deescalation_keywords = ("diplomacy", "stability", "deescalation", "sanction", "coordination", "engage")

        militarized = sum(v for k, v in priorities.items() if any(t in k for t in militarized_keywords))
        hybrid = sum(v for k, v in priorities.items() if any(t in k for t in hybrid_keywords))
        deescalatory = sum(v for k, v in priorities.items() if any(t in k for t in deescalation_keywords))

        red_lines = params.get("red_lines", [])
        escalation_steps = params.get("escalation_ladder", [])
        heuristics = params.get("decision_heuristics", [])

        assertiveness = np.clip(
            0.2 + 0.35 * militarized + 0.15 * hybrid + 0.03 * len(escalation_steps) + 0.01 * len(heuristics),
            0.0, 1.0,
        )
        restraint = np.clip(
            0.2 + 0.3 * deescalatory - 0.2 * militarized + 0.02 * len(red_lines),
            0.0, 1.0,
        )
        risk_tolerance = np.clip(0.2 + 0.4 * militarized + 0.2 * hybrid - 0.1 * deescalatory, 0.0, 1.0)

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
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.strategic_state.reset()
        self.spillover_model = CrossDomainSpilloverModel()
        self.schelling_manager.reset()
        self.deescalation.reset()
        self.blue_policy.reset_episode()
        self.red_policy.reset_episode()
        self._last_policy_penalties = {"Blue": 0.0, "Red": 0.0}

        regime = np.random.choice(["low", "baseline", "high"], p=[0.25, 0.5, 0.25])
        if regime == "low":
            self.tension = float(np.random.uniform(0.18, 0.32))
            self.stability = float(np.random.uniform(1.0, 1.15))
        elif regime == "high":
            self.tension = float(np.random.uniform(0.52, 0.7))
            self.stability = float(np.random.uniform(0.82, 1.0))
        else:
            self.tension = float(np.random.uniform(0.3, 0.45))
            self.stability = float(np.random.uniform(0.92, 1.08))

        self.last_blue_action = 0
        self.last_red_action = 0
        self._prev_strategic_scores = None
        self._prev_score_margin = None
        self._blue_repeat_streak = 0
        self._red_repeat_streak = 0
        self._blue_recent_actions = []
        self._red_recent_actions = []
        self._blue_info_spam_count = 0
        summary = self.strategic_state.get_dimension_summary()
        self._prev_contest_dims = (
            float(summary["military_balance"]),
            float(summary["coalition_strength"]),
            float(summary["resource_leverage"]),
        )
        self._episode_blue_score_baseline, self._episode_red_score_baseline = (
            self._strategic_objective_scores()
        )

        return self._get_obs(), {}

    def _break_action_monopoly(self, side: str, action_idx: int, streak: int) -> int:
        """Eval/DSS shim: after repeated identical COAs, force a different legal action."""
        if not self.coa_diversity_guard:
            return action_idx
        threshold = 2 if side == "Blue" else 3
        if streak < threshold:
            return action_idx
        pool = [
            i for i in range(self.num_actions)
            if i != action_idx
        ]
        if not pool:
            pool = [i for i in range(self.num_actions) if i != action_idx]
        return int(np.random.choice(pool)) if pool else action_idx

    def _apply_policy_action_gates(self, actions: Dict[str, int]) -> Dict[str, int]:
        """Clamp actions to policy escalation ladders before physics update."""
        if self.coa_diversity_guard:
            actions = {
                "Blue": self._break_action_monopoly(
                    "Blue", actions["Blue"], self._blue_repeat_streak
                ),
                "Red": self._break_action_monopoly(
                    "Red", actions["Red"], self._red_repeat_streak
                ),
            }
        if not self.policy_mechanics:
            return actions
        blue_idx, blue_clamped = self.blue_policy.clamp_action_index(
            actions["Blue"], self.action_names, self.action_costs,
            self.tension, self.blue_bias["risk_tolerance"],
        )
        red_idx, red_clamped = self.red_policy.clamp_action_index(
            actions["Red"], self.action_names, self.action_costs,
            self.tension, self.red_bias["risk_tolerance"],
        )
        if blue_clamped:
            self._last_policy_penalties["Blue"] += 0.15
        if red_clamped:
            self._last_policy_penalties["Red"] += 0.15
        return {"Blue": blue_idx, "Red": red_idx}

    def _record_commitments(self, side: str, action_name: str, effect: ActionEffect):
        if not self.policy_mechanics:
            return
        commitment = map_action_to_commitment(action_name, effect)
        if commitment is None:
            return
        self.schelling_manager.make_commitment(
            side, commitment, effect.intensity, self.current_step
        )

    def _apply_red_line_dynamics(self, blue_name: str, red_name: str,
                                blue_effect: ActionEffect, red_effect: ActionEffect):
        """Penalize side that pushes tension past document-derived red-line band."""
        if not self.policy_mechanics:
            return
        if self.tension < self.blue_policy.min_tension_red_line:
            return

        blue_esc = self.action_costs.get(blue_name, 0.0)
        red_esc = self.action_costs.get(red_name, 0.0)
        if blue_esc >= red_esc and blue_esc >= 0.35:
            self._last_policy_penalties["Blue"] += 0.25
            self.stability -= 0.008
        elif red_esc > blue_esc and red_esc >= 0.35:
            self._last_policy_penalties["Red"] += 0.25
            self.stability -= 0.008

    def step(self, actions: Dict[str, int]):
        self.current_step += 1
        self.schelling_manager.step(self.tension)
        self._last_policy_penalties = {"Blue": 0.0, "Red": 0.0}

        actions = self._apply_policy_action_gates(actions)
        blue_act = actions["Blue"]
        red_act = actions["Red"]

        blue_action_name = self.action_names[blue_act] if blue_act < self.num_actions else self.action_names[0]
        red_action_name = self.action_names[red_act] if red_act < self.num_actions else self.action_names[0]

        all_actions = MultiDomainActionLibrary.get_all_actions()
        blue_effect = all_actions.get(blue_action_name)
        red_effect = all_actions.get(red_action_name)

        if not blue_effect or not red_effect:
            blue_effect = ActionEffect(Domain.INFORMATION, 0.0, 0.0, 1.0, 0.0)
            red_effect = ActionEffect(Domain.INFORMATION, 0.0, 0.0, 1.0, 0.0)

        self.blue_policy.note_action(blue_action_name, self.action_costs)
        self.red_policy.note_action(red_action_name, self.action_costs)
        self._record_commitments("Blue", blue_action_name, blue_effect)
        self._record_commitments("Red", red_action_name, red_effect)

        self.strategic_state.apply_military_action(
            "Blue", blue_effect.intensity if blue_effect.domain == Domain.KINETIC else 0
        )
        self.strategic_state.apply_military_action(
            "Red", red_effect.intensity if red_effect.domain == Domain.KINETIC else 0
        )
        self.strategic_state.apply_economic_action(
            "Blue", blue_effect.intensity if blue_effect.domain == Domain.ECONOMIC else 0
        )
        self.strategic_state.apply_economic_action(
            "Red", red_effect.intensity if red_effect.domain == Domain.ECONOMIC else 0
        )
        self.strategic_state.apply_territorial_action(
            "Blue", blue_effect.intensity if blue_effect.domain == Domain.KINETIC else 0
        )
        self.strategic_state.apply_territorial_action(
            "Red", red_effect.intensity if red_effect.domain == Domain.KINETIC else 0
        )
        if blue_effect.alliance_impact:
            self.strategic_state.alliance.nato_cohesion = np.clip(
                self.strategic_state.alliance.nato_cohesion + 0.12 * blue_effect.alliance_impact,
                0.0,
                1.0,
            )
        if red_effect.alliance_impact:
            self.strategic_state.alliance.nato_cohesion = np.clip(
                self.strategic_state.alliance.nato_cohesion + 0.08 * red_effect.alliance_impact,
                0.0,
                1.0,
            )

        escalation_from_actions = (blue_effect.escalation_cost + red_effect.escalation_cost) * 0.42
        blue_spillover = self.spillover_model.compute_spillover(
            blue_action_name, blue_effect, self.tension, self.stability
        )
        red_spillover = self.spillover_model.compute_spillover(
            red_action_name, red_effect, self.tension, self.stability
        )
        spillover_escalation = 0.04 * (len(blue_spillover) + len(red_spillover))

        tension_change = (
            escalation_from_actions + blue_effect.tension_impact + red_effect.tension_impact
            + spillover_escalation + np.random.normal(0, 0.015)
        )
        tension_change = float(np.clip(tension_change, -0.07, 0.07))
        self.tension += tension_change
        # Mean reversion keeps competition in a negotiable band (avoids permanent crisis)
        self.tension += (0.50 - self.tension) * 0.018
        if self.tension > 0.82:
            self.tension -= 0.05 * (self.tension - 0.82)
        self.tension = float(np.clip(self.tension, 0.0, 0.82))

        stability_change = blue_effect.stability_impact + red_effect.stability_impact
        self.stability += stability_change + np.random.normal(0, 0.006)
        self.stability -= self.tension * 0.0025
        if 0.40 <= self.tension <= 0.70:
            self.stability += 0.018
        self.stability = np.clip(self.stability, 0.45, 1.2)

        self._apply_red_line_dynamics(blue_action_name, red_action_name, blue_effect, red_effect)

        self.strategic_state.natural_decay(decay_rate=0.02)

        obs = self._get_obs()
        episode_ending = (
            self.current_step >= self.max_steps
            or self.stability < 0.05
        )
        rewards = self._compute_rewards(
            blue_act,
            red_act,
            blue_action_name,
            red_action_name,
            blue_effect,
            red_effect,
            episode_ending=episode_ending,
        )

        self.last_blue_action = blue_act
        self.last_red_action = red_act

        terminated = episode_ending

        blue_strat, red_strat = self._strategic_objective_scores()
        info = {
            "tension": self.tension,
            "stability": self.stability,
            "current_step": self.current_step,
            "blue_action": blue_action_name,
            "red_action": red_action_name,
            "blue_action_clamped": self._last_policy_penalties["Blue"] > 0,
            "red_action_clamped": self._last_policy_penalties["Red"] > 0,
            "blue_ladder_progress": self.blue_policy.ladder_progress,
            "red_ladder_progress": self.red_policy.ladder_progress,
            "blue_commitments": len(self.schelling_manager.commitments_blue),
            "red_commitments": len(self.schelling_manager.commitments_red),
            "schelling_point": self.schelling_manager.last_schelling_point_crossed.name,
            "blue_strategic_score": blue_strat,
            "red_strategic_score": red_strat,
            **self.strategic_state.get_dimension_summary(),
        }

        return obs, rewards, {"Blue": terminated, "Red": terminated}, {"Blue": False, "Red": False}, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Asymmetric observations with domain-specific sensing and policy context."""
        state_vector = self.strategic_state.get_full_state_vector()
        if self.policy_mechanics:
            blue_visible = observe_for_side(state_vector, "Blue")
            red_visible = observe_for_side(state_vector, "Red")
        else:
            blue_visible = state_vector.copy()
            red_visible = state_vector.copy()

        blue_obs = np.zeros(64, dtype=np.float32)
        blue_obs[:36] = blue_visible
        blue_obs[36] = self.tension
        blue_obs[37] = self.stability
        blue_obs[38] = self.blue_bias["assertiveness"]
        blue_obs[39] = self.blue_bias["restraint"]
        blue_obs[40] = self.blue_bias["risk_tolerance"]
        blue_obs[41] = self.schelling_manager.steps_in_current_regime / max(1, self.max_steps)
        blue_obs[42] = len(self.schelling_manager.commitments_blue) / 10.0
        blue_obs[43] = self.strategic_state.alliance.nato_cohesion
        blue_obs[44] = len(self.schelling_manager.commitments_red) / 10.0
        blue_obs[45] = self.blue_policy.ladder_progress / max(1, len(self.blue_policy.ladder_steps))
        blue_obs[46] = float(len(self.blue_policy.red_lines)) / 10.0

        red_obs = np.zeros(64, dtype=np.float32)
        red_obs[:36] = red_visible
        red_obs[36] = self.tension
        red_obs[37] = self.stability
        red_obs[38] = self.red_bias["assertiveness"]
        red_obs[39] = self.red_bias["restraint"]
        red_obs[40] = self.red_bias["risk_tolerance"]
        red_obs[41] = self.schelling_manager.steps_in_current_regime / max(1, self.max_steps)
        red_obs[42] = len(self.schelling_manager.commitments_red) / 10.0
        red_obs[43] = self.strategic_state.resources.resource_leverage()
        red_obs[44] = len(self.schelling_manager.commitments_blue) / 10.0
        red_obs[45] = self.red_policy.ladder_progress / max(1, len(self.red_policy.ladder_steps))
        red_obs[46] = float(len(self.red_policy.red_lines)) / 10.0

        return {"Blue": np.clip(blue_obs, -1.0, 1.0), "Red": np.clip(red_obs, -1.0, 1.0)}

    def _strategic_objective_scores(self) -> tuple[float, float]:
        """
        Contest-relative scores centered on neutral military balance and
        side-specific objectives (NATO cohesion vs resource/energy leverage).
        """
        m = self.strategic_state.military
        e = self.strategic_state.economic
        a = self.strategic_state.alliance
        r = self.strategic_state.resources
        t = self.strategic_state.territorial

        mil_bal = m.balance_ratio()
        coalition = a.coalition_strength()
        res_lev = r.resource_leverage()
        press_red = e.economic_pressure_on_red()
        press_blue = e.economic_pressure_on_blue()

        # Baselines chosen so initial strategic posture is near 0 for both sides
        blue_score = (
            1.00 * (mil_bal - 0.50)
            + 0.62 * (coalition - 0.72)
            + 0.65 * (0.40 - res_lev)
            + 0.55 * press_red
            - 0.55 * press_blue
            + 0.30 * (t.blue_territorial_control - t.red_territorial_control)
            - 0.12 * max(0.0, self.tension - 0.82)
        )

        red_score = (
            1.00 * (0.50 - mil_bal)
            + 0.62 * (res_lev - 0.40)
            + 0.52 * press_blue
            + 0.38 * (t.red_logistics_access - t.blue_logistics_access)
            + 0.28 * (t.red_territorial_control - t.blue_territorial_control)
            - 0.48 * press_red
            - 0.20 * max(0.0, 0.55 - self.tension)
        )

        return float(blue_score), float(red_score)

    def _track_action_patterns(self, blue_act: int, red_act: int) -> tuple[float, float]:
        """Returns (blue_repeat_penalty, red_repeat_penalty) for anti-collapse."""
        if self._blue_recent_actions and self._blue_recent_actions[-1] == blue_act:
            self._blue_repeat_streak += 1
        else:
            self._blue_repeat_streak = 1
        if self._red_recent_actions and self._red_recent_actions[-1] == red_act:
            self._red_repeat_streak += 1
        else:
            self._red_repeat_streak = 1

        self._blue_recent_actions.append(blue_act)
        self._red_recent_actions.append(red_act)
        if len(self._blue_recent_actions) > self._recent_window:
            self._blue_recent_actions.pop(0)
        if len(self._red_recent_actions) > self._recent_window:
            self._red_recent_actions.pop(0)

        blue_pen = 0.0
        red_pen = 0.0
        if self._blue_repeat_streak >= 4:
            blue_pen = 0.08 * (self._blue_repeat_streak - 3)
        if self._blue_repeat_streak >= 7:
            blue_pen += 0.25
        if self._red_repeat_streak >= 4:
            red_pen = 0.07 * (self._red_repeat_streak - 3)
        if self._red_repeat_streak >= 7:
            red_pen += 0.22

        if len(self._blue_recent_actions) >= 8:
            unique_blue = len(set(self._blue_recent_actions))
            if unique_blue <= 2:
                blue_pen += 0.12 * (3 - unique_blue)
            dominant = max(self._blue_recent_actions.count(a) for a in set(self._blue_recent_actions))
            if dominant >= 7:
                blue_pen += 0.20 * (dominant - 6)
        if len(self._red_recent_actions) >= 8:
            unique_red = len(set(self._red_recent_actions))
            if unique_red <= 2:
                red_pen += 0.10 * (3 - unique_red)
            dominant = max(self._red_recent_actions.count(a) for a in set(self._red_recent_actions))
            if dominant >= 7:
                red_pen += 0.18 * (dominant - 6)

        return blue_pen, red_pen

    def _contest_deltas(self) -> tuple[float, float, float]:
        summary = self.strategic_state.get_dimension_summary()
        mil = float(summary["military_balance"])
        coal = float(summary["coalition_strength"])
        res = float(summary["resource_leverage"])
        if self._prev_contest_dims is None:
            self._prev_contest_dims = (mil, coal, res)
            return 0.0, 0.0, 0.0
        prev_mil, prev_coal, prev_res = self._prev_contest_dims
        self._prev_contest_dims = (mil, coal, res)
        return mil - prev_mil, coal - prev_coal, res - prev_res

    def _compute_rewards(
        self,
        blue_act: int,
        red_act: int,
        blue_action: str,
        red_action: str,
        blue_effect: ActionEffect,
        red_effect: ActionEffect,
        episode_ending: bool = False,
    ) -> Dict[str, float]:
        blue_repeat_pen, red_repeat_pen = self._track_action_patterns(blue_act, red_act)

        blue_now, red_now = self._strategic_objective_scores()
        margin = blue_now - red_now
        dmil, dcoal, dres = self._contest_deltas()

        # Progress vs episode start (neutral posture ~0) — primary training signal
        blue_r = 0.06 + 0.48 * (blue_now - self._episode_blue_score_baseline)
        red_r = 0.06 + 0.48 * (red_now - self._episode_red_score_baseline)

        if self._prev_strategic_scores is not None:
            prev_b, prev_r = self._prev_strategic_scores
            blue_r += 0.28 * (blue_now - prev_b)
            red_r += 0.28 * (red_now - prev_r)
        self._prev_strategic_scores = (blue_now, red_now)

        # Contest-relative margin (aligns RL with win/loss classification)
        if self._prev_score_margin is not None:
            dm = margin - self._prev_score_margin
            blue_r += 0.45 * dm
            red_r -= 0.45 * dm
        self._prev_score_margin = margin

        # Contest movement (scaled down; anti-collapse should not punish exploration)
        blue_r += 0.38 * max(0.0, dmil) + 0.36 * max(0.0, dcoal) + 0.26 * max(0.0, -dres)
        red_r += 0.34 * max(0.0, -dmil) + 0.28 * max(0.0, dres) + 0.22 * max(0.0, -dcoal)

        # Shared tension band — reward sustainable competition band
        target_t = 0.58
        band_bonus = 0.20 * float(np.exp(-((self.tension - target_t) ** 2) / (2 * 0.11**2)))
        blue_r += band_bonus
        red_r += band_bonus
        if self.tension > 0.78:
            over = self.tension - 0.78
            blue_r -= 0.32 * over
            red_r -= 0.34 * over

        if self.tension < 0.38:
            stalemate = (0.38 - self.tension) / 0.38
            blue_r -= 0.50 * stalemate
            red_r -= 0.50 * stalemate
        elif self.tension > 0.88:
            crisis = self.tension - 0.88
            blue_r -= 0.60 * crisis
            red_r -= 0.60 * crisis

        if blue_effect.escalation_cost > 0:
            blue_r -= 0.18 * blue_effect.escalation_cost
        if red_effect.escalation_cost > 0:
            red_r -= 0.18 * red_effect.escalation_cost
        if (
            blue_effect.escalation_cost < 0.09
            and red_effect.escalation_cost < 0.09
            and self._blue_repeat_streak >= 4
            and self._red_repeat_streak >= 4
        ):
            blue_r -= 0.08
            red_r -= 0.08

        if self.policy_mechanics:
            blue_r += 0.03 * self.blue_policy.ladder_progress
            red_r += 0.03 * self.red_policy.ladder_progress

        mil_bal = self.strategic_state.military.balance_ratio()
        # Domain shaping aligned to each side's competitive toolkit
        if "economic" in blue_action:
            blue_r += 0.30 * blue_effect.intensity
        if "economic" in red_action:
            red_r += 0.28 * red_effect.intensity
        if "cyber" in blue_action:
            blue_r += 0.24 * blue_effect.intensity
        if "cyber" in red_action:
            red_r += 0.22 * red_effect.intensity
        if "kinetic" in blue_action:
            blue_r += 0.26 * blue_effect.intensity * (1.0 if mil_bal >= 0.48 else 0.55)
        if "kinetic" in red_action:
            red_r += 0.24 * red_effect.intensity * (1.0 if mil_bal <= 0.52 else 0.45)
        if "info" in blue_action and blue_effect.intensity < 0.25:
            if dmil > 0.002 or dcoal > 0.002:
                blue_r += 0.06
            elif self._blue_repeat_streak >= 3:
                blue_r -= 0.10
        if "info" in red_action and red_effect.intensity < 0.25:
            if (-dmil) > 0.002 or dres > 0.002:
                red_r += 0.06
            elif self._red_repeat_streak >= 3:
                red_r -= 0.08

        if blue_act == self._info_sharing_idx:
            self._blue_info_spam_count += 1
            if self._blue_info_spam_count >= 4:
                blue_r -= 0.12 * (self._blue_info_spam_count - 3)
            if self._blue_repeat_streak >= 5 and blue_act == self._info_sharing_idx:
                blue_r -= 0.15 * (self._blue_repeat_streak - 4)

        blue_r -= blue_repeat_pen
        red_r -= red_repeat_pen

        commitment_penalty_blue = self.schelling_manager.get_commitment_penalty("Blue")
        commitment_penalty_red = self.schelling_manager.get_commitment_penalty("Red")
        blue_r -= 0.32 * commitment_penalty_blue
        red_r -= 0.32 * commitment_penalty_red

        if self.deescalation.get_deescalation_window(
            self.tension, self.schelling_manager.steps_in_current_regime
        ):
            if "info" in blue_action and blue_effect.intensity < 0.3 and self._blue_repeat_streak <= 1:
                blue_r += 0.12
            if "info" in red_action and red_effect.intensity < 0.3 and self._red_repeat_streak <= 1:
                red_r += 0.12

        blue_r -= self._last_policy_penalties["Blue"]
        red_r -= self._last_policy_penalties["Red"]

        if episode_ending:
            if margin > 0.12:
                blue_r += 0.95
                red_r -= 0.65
            elif margin < -0.12:
                red_r += 0.95
                blue_r -= 0.65
            else:
                blue_r += 0.20
                red_r += 0.20

        # Keep per-step signal bounded for long episodes (60-step horizons)
        clip_hi = 0.85 if episode_ending else 0.45
        blue_r = float(np.clip(blue_r, -clip_hi, clip_hi))
        red_r = float(np.clip(red_r, -clip_hi, clip_hi))

        return {"Blue": blue_r, "Red": red_r}
