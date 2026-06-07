"""
Multi-Domain Operations Framework
Separates strategic actions across cyber, kinetic, economic, and information domains
with realistic cross-domain spillover effects and domain-specific escalation costs.
"""

import numpy as np
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass


class Domain(Enum):
    """Strategic action domains."""
    KINETIC = "kinetic"
    CYBER = "cyber"
    ECONOMIC = "economic"
    INFORMATION = "information"


class Intensity(Enum):
    """Action intensity levels."""
    PASSIVE = 0.1
    CALIBRATED = 0.3
    MEASURED = 0.5
    AGGRESSIVE = 0.8
    MAXIMUM = 1.0


@dataclass
class ActionEffect:
    """Effects of an action across strategic dimensions."""
    domain: Domain
    intensity: float
    escalation_cost: float  # How much this escalates tension
    reversibility: float  # 0=permanent, 1=easily reversible
    visibility: float  # 0=hidden, 1=highly visible (affects detection/response)
    
    # Cross-domain spillover potentials
    kinetic_spillover: float = 0.0  # Probability this triggers kinetic response
    cyber_spillover: float = 0.0
    economic_spillover: float = 0.0
    information_spillover: float = 0.0
    
    # Target effects
    tension_impact: float = 0.0
    stability_impact: float = 0.0
    influence_impact: float = 0.0
    alliance_impact: float = 0.0


class MultiDomainActionLibrary:
    """Define all available multi-domain actions with effects."""
    
    # ============ KINETIC DOMAIN ACTIONS ============
    KINETIC_ACTIONS = {
        "readiness_exercise": ActionEffect(
            domain=Domain.KINETIC,
            intensity=0.15,
            escalation_cost=0.08,
            reversibility=0.95,
            visibility=0.70,
            kinetic_spillover=0.05,
            tension_impact=0.06,
            influence_impact=0.02,
        ),
        "increased_patrols": ActionEffect(
            domain=Domain.KINETIC,
            intensity=0.25,
            escalation_cost=0.10,
            reversibility=0.85,
            visibility=0.85,
            kinetic_spillover=0.06,
            cyber_spillover=0.03,
            tension_impact=0.05,
            influence_impact=0.03,
        ),
        "airspace_restriction": ActionEffect(
            domain=Domain.KINETIC,
            intensity=0.35,
            escalation_cost=0.20,
            reversibility=0.75,
            visibility=0.95,
            kinetic_spillover=0.25,
            cyber_spillover=0.08,
            tension_impact=0.25,
            influence_impact=0.05,
        ),
        "limited_deployment": ActionEffect(
            domain=Domain.KINETIC,
            intensity=0.55,
            escalation_cost=0.35,
            reversibility=0.50,
            visibility=0.98,
            kinetic_spillover=0.45,
            cyber_spillover=0.10,
            information_spillover=0.15,
            tension_impact=0.40,
            stability_impact=-0.15,
            influence_impact=0.08,
        ),
        "full_mobilization": ActionEffect(
            domain=Domain.KINETIC,
            intensity=0.85,
            escalation_cost=0.70,
            reversibility=0.20,
            visibility=1.00,
            kinetic_spillover=0.80,
            economic_spillover=0.30,
            tension_impact=0.75,
            stability_impact=-0.35,
            influence_impact=0.15,
        ),
        "air_strike_operations": ActionEffect(
            domain=Domain.KINETIC,
            intensity=1.0,
            escalation_cost=1.0,
            reversibility=0.0,
            visibility=1.0,
            kinetic_spillover=1.0,
            cyber_spillover=0.20,
            economic_spillover=0.15,
            information_spillover=0.40,
            tension_impact=1.0,
            stability_impact=-0.70,
            influence_impact=0.25,
        ),
    }
    
    # ============ CYBER DOMAIN ACTIONS ============
    CYBER_ACTIONS = {
        "passive_monitoring": ActionEffect(
            domain=Domain.CYBER,
            intensity=0.05,
            escalation_cost=0.02,
            reversibility=1.0,
            visibility=0.05,
            tension_impact=0.01,
            influence_impact=0.01,
        ),
        "network_probing": ActionEffect(
            domain=Domain.CYBER,
            intensity=0.15,
            escalation_cost=0.08,
            reversibility=0.95,
            visibility=0.15,
            tension_impact=0.04,
            stability_impact=-0.02,
        ),
        "infrastructure_scanning": ActionEffect(
            domain=Domain.CYBER,
            intensity=0.25,
            escalation_cost=0.12,
            reversibility=0.80,
            visibility=0.25,
            information_spillover=0.05,
            tension_impact=0.08,
            stability_impact=-0.05,
        ),
        "cyber_harassment": ActionEffect(
            domain=Domain.CYBER,
            intensity=0.40,
            escalation_cost=0.20,
            reversibility=0.60,
            visibility=0.50,
            kinetic_spillover=0.15,
            information_spillover=0.20,
            tension_impact=0.15,
            stability_impact=-0.10,
            influence_impact=0.04,
        ),
        "critical_infrastructure_attack": ActionEffect(
            domain=Domain.CYBER,
            intensity=0.75,
            escalation_cost=0.55,
            reversibility=0.15,
            visibility=0.85,
            kinetic_spillover=0.50,
            economic_spillover=0.40,
            information_spillover=0.30,
            tension_impact=0.60,
            stability_impact=-0.25,
            influence_impact=0.12,
        ),
        "command_control_disruption": ActionEffect(
            domain=Domain.CYBER,
            intensity=0.90,
            escalation_cost=0.80,
            reversibility=0.10,
            visibility=0.95,
            kinetic_spillover=0.80,
            economic_spillover=0.20,
            tension_impact=0.85,
            stability_impact=-0.50,
        ),
    }
    
    # ============ ECONOMIC DOMAIN ACTIONS ============
    ECONOMIC_ACTIONS = {
        "trade_monitoring": ActionEffect(
            domain=Domain.ECONOMIC,
            intensity=0.05,
            escalation_cost=0.01,
            reversibility=1.0,
            visibility=0.10,
            tension_impact=0.00,
        ),
        "selective_tariffs": ActionEffect(
            domain=Domain.ECONOMIC,
            intensity=0.20,
            escalation_cost=0.10,
            reversibility=0.90,
            visibility=0.80,
            information_spillover=0.10,
            influence_impact=-0.05,
            tension_impact=0.06,
        ),
        "targeted_sanctions": ActionEffect(
            domain=Domain.ECONOMIC,
            intensity=0.40,
            escalation_cost=0.20,
            reversibility=0.60,
            visibility=0.95,
            information_spillover=0.15,
            tension_impact=0.12,
            influence_impact=-0.08,
            stability_impact=-0.05,
        ),
        "sectoral_sanctions": ActionEffect(
            domain=Domain.ECONOMIC,
            intensity=0.65,
            escalation_cost=0.40,
            reversibility=0.40,
            visibility=0.98,
            cyber_spillover=0.20,
            kinetic_spillover=0.10,
            information_spillover=0.25,
            tension_impact=0.30,
            stability_impact=-0.15,
            influence_impact=-0.20,
        ),
        "financial_isolation": ActionEffect(
            domain=Domain.ECONOMIC,
            intensity=0.85,
            escalation_cost=0.60,
            reversibility=0.20,
            visibility=1.0,
            cyber_spillover=0.30,
            information_spillover=0.20,
            tension_impact=0.50,
            stability_impact=-0.25,
            influence_impact=-0.35,
        ),
        "energy_weaponization": ActionEffect(
            domain=Domain.ECONOMIC,
            intensity=0.75,
            escalation_cost=0.45,
            reversibility=0.30,
            visibility=0.85,
            information_spillover=0.25,
            kinetic_spillover=0.15,
            tension_impact=0.35,
            stability_impact=-0.20,
            influence_impact=0.10,
        ),
    }
    
    # ============ INFORMATION DOMAIN ACTIONS ============
    INFORMATION_ACTIONS = {
        "intelligence_sharing": ActionEffect(
            domain=Domain.INFORMATION,
            intensity=0.10,
            escalation_cost=0.02,
            reversibility=0.95,
            visibility=0.40,
            alliance_impact=0.03,
            tension_impact=-0.02,
            stability_impact=0.02,
        ),
        "public_statements": ActionEffect(
            domain=Domain.INFORMATION,
            intensity=0.15,
            escalation_cost=0.05,
            reversibility=0.90,
            visibility=1.0,
            alliance_impact=0.05,
            tension_impact=0.04,
            information_spillover=0.20,
        ),
        "disinformation_campaign": ActionEffect(
            domain=Domain.INFORMATION,
            intensity=0.40,
            escalation_cost=0.15,
            reversibility=0.50,
            visibility=0.30,
            kinetic_spillover=0.10,
            cyber_spillover=0.15,
            alliance_impact=-0.10,
            tension_impact=0.10,
            stability_impact=-0.08,
        ),
        "narrative_dominance": ActionEffect(
            domain=Domain.INFORMATION,
            intensity=0.35,
            escalation_cost=0.12,
            reversibility=0.60,
            visibility=0.60,
            alliance_impact=-0.08,
            tension_impact=0.08,
            influence_impact=0.06,
        ),
        "information_warfare": ActionEffect(
            domain=Domain.INFORMATION,
            intensity=0.65,
            escalation_cost=0.35,
            reversibility=0.25,
            visibility=0.45,
            kinetic_spillover=0.20,
            cyber_spillover=0.25,
            economic_spillover=0.15,
            alliance_impact=-0.20,
            tension_impact=0.25,
            stability_impact=-0.15,
            influence_impact=0.08,
        ),
        "existential_messaging": ActionEffect(
            domain=Domain.INFORMATION,
            intensity=0.90,
            escalation_cost=0.70,
            reversibility=0.10,
            visibility=1.0,
            kinetic_spillover=0.50,
            cyber_spillover=0.30,
            alliance_impact=-0.35,
            tension_impact=0.60,
            stability_impact=-0.40,
            influence_impact=0.15,
        ),
    }
    
    @classmethod
    def get_all_actions(cls) -> Dict[str, ActionEffect]:
        """Get all 24+ multi-domain actions."""
        return {
            **{f"kinetic_{k}": v for k, v in cls.KINETIC_ACTIONS.items()},
            **{f"cyber_{k}": v for k, v in cls.CYBER_ACTIONS.items()},
            **{f"economic_{k}": v for k, v in cls.ECONOMIC_ACTIONS.items()},
            **{f"info_{k}": v for k, v in cls.INFORMATION_ACTIONS.items()},
        }
    
    @classmethod
    def get_actions_by_domain(cls, domain: Domain) -> Dict[str, ActionEffect]:
        """Get all actions in a specific domain."""
        all_actions = cls.get_all_actions()
        domain_prefix = domain.value.split("_")[0]
        return {k: v for k, v in all_actions.items() if k.startswith(domain_prefix)}


class CrossDomainSpilloverModel:
    """Models how actions in one domain trigger responses/escalation in others."""
    
    def __init__(self):
        self.spillover_history: List[Tuple[str, str]] = []  # Track spillover chain
        self.mutual_escalation_lock = False  # Once kinetic, harder to de-escalate
        self.cyber_retaliation_pending = 0
        self.information_counter_pending = 0
    
    def compute_spillover(self, action_name: str, action_effect: ActionEffect, 
                         current_tension: float, current_stability: float) -> Dict[str, float]:
        """
        Compute spillover effects from taking an action.
        Returns dict of cross-domain triggered actions (action_name -> probability).
        """
        spillover_actions = {}
        
        # High tension amplifies spillover
        spillover_multiplier = 1.0 + 0.5 * max(0.0, current_tension - 0.6)
        
        # Kinetic actions trigger kinetic reciprocation
        if action_effect.kinetic_spillover > 0:
            trigger_prob = action_effect.kinetic_spillover * spillover_multiplier
            if np.random.random() < trigger_prob:
                spillover_actions["enemy_kinetic_response"] = trigger_prob
                self.mutual_escalation_lock = True
        
        # Cyber actions may trigger kinetic or economic response
        if action_effect.cyber_spillover > 0:
            trigger_prob = action_effect.cyber_spillover * spillover_multiplier * 0.7
            if np.random.random() < trigger_prob:
                spillover_actions["enemy_cyber_retaliation"] = trigger_prob
                self.cyber_retaliation_pending += 1
        
        # Economic actions trigger information counters
        if action_effect.economic_spillover > 0:
            trigger_prob = action_effect.economic_spillover * spillover_multiplier * 0.5
            if np.random.random() < trigger_prob:
                spillover_actions["enemy_information_campaign"] = trigger_prob
                self.information_counter_pending += 1
        
        # Information actions may escalate to kinetic in crisis
        if action_effect.information_spillover > 0 and current_tension > 0.7:
            trigger_prob = action_effect.information_spillover * max(0.0, current_tension - 0.7)
            if np.random.random() < trigger_prob:
                spillover_actions["crisis_kinetic_escalation"] = trigger_prob
        
        self.spillover_history.append((action_name, str(spillover_actions)))
        return spillover_actions
    
    def get_commitment_penalty(self) -> float:
        """
        Once mutual escalation locks, de-escalation becomes harder.
        Commitment penalty increases with each step of mutual escalation.
        """
        if not self.mutual_escalation_lock:
            return 0.0
        return min(0.5, 0.1 * len([x for x in self.spillover_history if "kinetic" in str(x)]))
    
    def allow_deescalation_window(self, steps_in_crisis: int) -> bool:
        """
        Limited windows for coordinated de-escalation after crisis peaks.
        Once locked in, both sides pay cost to back down.
        """
        return steps_in_crisis > 5 and steps_in_crisis < 15


class DomainSeparationObservation:
    """
    Agents see domain-specific sensor models, not full state.
    Reflects that intelligence is incomplete and domain-specific.
    """
    
    @staticmethod
    def get_visible_state(full_state: np.ndarray, observer_domain: Domain, 
                         own_side: str, noise_level: float = 0.05) -> np.ndarray:
        """
        Return domain-specific observation with realistic sensor noise and gaps.
        """
        visible = full_state.copy()
        
        # Kinetic observers (military) see military well, cyber poorly
        if observer_domain == Domain.KINETIC:
            # Add gaussian noise to non-military dimensions
            visible[8:16] += np.random.normal(0, noise_level * 0.3, 8)  # Economic less visible
            visible[24:32] += np.random.normal(0, noise_level * 0.5, 8)  # Info less visible
        
        # Cyber observers see cyber operations, miss economic/kinetic
        elif observer_domain == Domain.CYBER:
            visible[0:8] += np.random.normal(0, noise_level * 0.4, 8)    # Military less visible
            visible[16:24] += np.random.normal(0, noise_level * 0.3, 8)  # Territorial less visible
        
        # Economic observers see economic pressure/sanctions
        elif observer_domain == Domain.ECONOMIC:
            visible[0:8] += np.random.normal(0, noise_level * 0.3, 8)    # Military less visible
            visible[24:32] += np.random.normal(0, noise_level * 0.2, 8)  # Info less visible
        
        # Information operators see information effects
        elif observer_domain == Domain.INFORMATION:
            visible[8:16] += np.random.normal(0, noise_level * 0.4, 8)   # Economic less visible
        
        return np.clip(visible, -1.0, 1.0)
