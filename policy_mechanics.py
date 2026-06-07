"""
Bridge ingested policy JSON to simulation mechanics: merged corpora, ladder
constraints, commitment triggers, and asymmetric sensing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .multi_domain_operations import (
    ActionEffect,
    Domain,
    DomainSeparationObservation,
    MultiDomainActionLibrary,
)
from .escalation_pathways import EscalationCommitment


def _dedupe_strings(items: List) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def merge_side_records(records: List[dict]) -> dict:
    """Confidence-weighted merge of all valid policy extractions for one side."""
    valid = [r for r in records if isinstance(r, dict) and "error" not in r]
    if not valid:
        return {}

    total_w = sum(float(r.get("confidence", 0.5)) for r in valid) or float(len(valid))
    merged_priority: Dict[str, float] = {}
    objectives: List[str] = []
    red_lines: List[str] = []
    heuristics: List[str] = []
    ladder: List[str] = []
    quotes: List[str] = []
    sources: List[str] = []

    for rec in valid:
        w = float(rec.get("confidence", 0.5)) / total_w
        for key, val in (rec.get("capability_priorities") or {}).items():
            if not isinstance(key, str):
                continue
            try:
                merged_priority[key.lower()] = merged_priority.get(key.lower(), 0.0) + w * float(val)
            except (TypeError, ValueError):
                continue
        objectives.extend(rec.get("key_objectives") or [])
        red_lines.extend(rec.get("red_lines") or [])
        heuristics.extend(rec.get("decision_heuristics") or [])
        ladder.extend(rec.get("escalation_ladder") or [])
        quotes.extend(rec.get("key_quotes") or [])
        src = rec.get("original_filename") or rec.get("source_file")
        if src:
            sources.append(str(src))

    return {
        "document_title": "merged_side_policy",
        "source_files": sorted(set(sources)),
        "key_objectives": _dedupe_strings(objectives),
        "red_lines": _dedupe_strings(red_lines),
        "decision_heuristics": _dedupe_strings(heuristics),
        "escalation_ladder": _dedupe_strings(ladder),
        "key_quotes": _dedupe_strings(quotes)[:12],
        "capability_priorities": merged_priority,
        "confidence": max(float(r.get("confidence", 0.5)) for r in valid),
        "n_source_documents": len(valid),
    }


@dataclass
class SidePolicyProfile:
    """Runtime policy constraints derived from merged JSON."""

    side: str
    merged: dict
    ladder_steps: List[str] = field(default_factory=list)
    red_lines: List[str] = field(default_factory=list)
    sorted_action_names: List[str] = field(default_factory=list)
    sorted_action_indices: List[int] = field(default_factory=list)
    min_tension_red_line: float = 0.55
    ladder_progress: int = 0
    recent_clamp_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_merged(cls, side: str, merged: dict, action_names: List[str], action_costs: Dict[str, float]):
        ladder = merged.get("escalation_ladder") or []
        red_lines = merged.get("red_lines") or []
        sorted_names = sorted(action_names, key=lambda n: action_costs.get(n, 0.0))
        sorted_indices = [action_names.index(n) for n in sorted_names]
        min_tension = float(np.clip(0.42 + 0.04 * len(red_lines), 0.45, 0.78))
        return cls(
            side=side,
            merged=merged,
            ladder_steps=ladder,
            red_lines=red_lines,
            sorted_action_names=sorted_names,
            sorted_action_indices=sorted_indices,
            min_tension_red_line=min_tension,
        )

    def reset_episode(self):
        self.ladder_progress = 0
        self.recent_clamp_indices = []

    def max_allowed_rank(self, num_actions: int, tension: float, risk_tolerance: float) -> int:
        n = max(len(self.ladder_steps), 1)
        ladder_frac = min(1.0, (self.ladder_progress + 1) / (n + 1))
        tension_frac = min(0.4, tension * 0.45)
        risk_frac = 0.12 * risk_tolerance
        allowed_frac = min(1.0, ladder_frac + tension_frac + risk_frac)
        return int(allowed_frac * max(0, num_actions - 1))

    def clamp_action_index(
        self,
        action_idx: int,
        action_names: List[str],
        action_costs: Dict[str, float],
        tension: float,
        risk_tolerance: float,
    ) -> Tuple[int, bool]:
        """Downgrade over-escalatory actions relative to ladder progress. Returns (idx, was_clamped)."""
        num_actions = len(action_names)
        if action_idx < 0 or action_idx >= num_actions:
            return 0, True

        max_rank = self.max_allowed_rank(num_actions, tension, risk_tolerance)
        name = action_names[action_idx]
        try:
            rank = self.sorted_action_names.index(name)
        except ValueError:
            return action_idx, False

        if rank <= max_rank:
            return action_idx, False

        allowed_names = self.sorted_action_names[: max_rank + 1]
        target_cost = action_costs.get(name, 0.0)
        # Nearest-allowed downgrade (not always the ceiling action — avoids patrol lock-in)
        clamped_name = min(
            allowed_names,
            key=lambda n: abs(action_costs.get(n, 0.0) - target_cost),
        )
        clamped_idx = action_names.index(clamped_name)

        if len(self.recent_clamp_indices) >= 2:
            recent = self.recent_clamp_indices[-3:]
            if recent.count(clamped_idx) >= 2 and len(allowed_names) > 1:
                alternatives = [n for n in allowed_names if action_names.index(n) != clamped_idx]
                clamped_name = str(np.random.choice(alternatives))
                clamped_idx = action_names.index(clamped_name)

        self.recent_clamp_indices.append(clamped_idx)
        if len(self.recent_clamp_indices) > 8:
            self.recent_clamp_indices.pop(0)
        return clamped_idx, True

    def note_action(self, action_name: str, action_costs: Dict[str, float]):
        if action_costs.get(action_name, 0.0) >= 0.22:
            self.ladder_progress = min(len(self.ladder_steps), self.ladder_progress + 1)


def map_action_to_commitment(action_name: str, effect: ActionEffect) -> Optional[EscalationCommitment]:
    """Map meaningful escalatory actions to commitment types."""
    if effect.intensity < 0.32 and effect.reversibility > 0.65 and effect.escalation_cost < 0.25:
        return None

    if "air_strike" in action_name or (effect.domain == Domain.KINETIC and effect.intensity >= 0.95):
        return EscalationCommitment.FIRST_STRIKE
    if "airspace" in action_name:
        return EscalationCommitment.AIRSPACE_CLOSURE
    if "mobilization" in action_name or "limited_deployment" in action_name:
        return EscalationCommitment.DEPLOYMENT
    if effect.domain == Domain.ECONOMIC and effect.intensity >= 0.55:
        return EscalationCommitment.SANCTIONS
    if effect.domain == Domain.CYBER and effect.intensity >= 0.65:
        return EscalationCommitment.CYBER_ATTACK
    if "existential" in action_name or "ultimatum" in action_name:
        return EscalationCommitment.PUBLIC_ULTIMATUM
    if effect.reversibility < 0.35:
        return EscalationCommitment.DEPLOYMENT
    return None


def build_action_cost_table() -> Dict[str, float]:
    return {name: eff.escalation_cost for name, eff in MultiDomainActionLibrary.get_all_actions().items()}


def observe_for_side(state_vector: np.ndarray, side: str) -> np.ndarray:
    """Asymmetric, domain-biased partial observability."""
    if side == "Blue":
        visible = DomainSeparationObservation.get_visible_state(
            state_vector, Domain.KINETIC, "Blue", noise_level=0.06
        )
        # Blue sees alliance clearly; red force structure is noisier
        for idx in (1, 3, 5, 7):
            visible[idx] += np.random.normal(0, 0.12)
        visible[24:30] *= 1.0  # alliance block
        visible[30:36] += np.random.normal(0, 0.08, 6)  # resources (adversary leverage) fuzzier
    else:
        visible = DomainSeparationObservation.get_visible_state(
            state_vector, Domain.ECONOMIC, "Red", noise_level=0.06
        )
        for idx in range(24, 30):
            visible[idx] += np.random.normal(0, 0.14)
        visible[0:8] += np.random.normal(0, 0.10, 8)

    return np.clip(visible, 0.0, 1.0).astype(np.float32)
