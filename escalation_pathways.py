"""
Realistic Escalation Pathways
Implements Schelling points, irreversible actions, de-escalation windows,
and face-saving mechanisms for realistic strategic crisis behavior.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SchelllingPoint(Enum):
    """Schelling focal points: natural thresholds where decisions must be made."""
    PEACETIME = 0.3           # Baseline tension
    FIRST_ALERT = 0.45        # Initial crisis indicators
    MOBILIZATION_TRIGGER = 0.60  # Military mobilization threshold
    AIRSPACE_CLOSURE = 0.70    # Territorial air/sea restrictions
    FORWARD_DEPLOYMENT = 0.75  # Major force deployments
    STRATEGIC_STRIKE = 0.88    # Imminent kinetic strike threshold
    MUTUAL_DESTRUCTION = 0.95  # Near-total breakdown


class EscalationCommitment(Enum):
    """Types of irreversible commitments."""
    DEPLOYMENT = "deployment"  # Forward military forces
    SANCTIONS = "sanctions"    # Economic isolation
    CYBER_ATTACK = "cyber_attack"  # Destructive cyber ops
    AIRSPACE_CLOSURE = "airspace"
    PUBLIC_ULTIMATUM = "ultimatum"
    FIRST_STRIKE = "kinetic"


@dataclass
class IrreversibleAction:
    """Tracks irreversible commitments made by each side."""
    action_type: EscalationCommitment
    timestamp: int  # Step when action was taken
    intensity: float  # 0-1 scale
    can_reverse_by: int  # Steps until reversal becomes impossible
    face_saving_required: float  # What opponent must concede to allow reversal
    
    def is_reversible(self, current_step: int) -> bool:
        """Check if this commitment can still be reversed."""
        return current_step < self.timestamp + self.can_reverse_by


@dataclass
class DeescalationOffer:
    """Formal offer to de-escalate."""
    offering_side: str  # "Blue" or "Red"
    concessions: Dict[str, float]  # What they'll give up (sanctions levels, force levels, etc)
    face_saving: str  # How opponent saves face (agreement phrasing, etc)
    confidence: float  # 0-1 likelihood opponent accepts
    expires_in: int  # Steps until offer expires


class SchelllingPointManager:
    """Manages escalation points and commitment dynamics."""
    
    def __init__(self):
        self.commitments_blue: List[IrreversibleAction] = []
        self.commitments_red: List[IrreversibleAction] = []
        self.past_offers: List[DeescalationOffer] = []
        self.current_step = 0
        self.last_schelling_point_crossed = SchelllingPoint.PEACETIME
        self.steps_in_current_regime = 0
        self.breakdown_threshold = 0.95  # Tension level where spiral becomes uncontrollable
    
    def reset(self):
        """Reset commitment tracking."""
        self.commitments_blue = []
        self.commitments_red = []
        self.past_offers = []
        self.current_step = 0
        self.last_schelling_point_crossed = SchelllingPoint.PEACETIME
        self.steps_in_current_regime = 0
    
    def step(self, tension: float):
        """Update commitment tracking."""
        self.current_step += 1
        
        # Check if we've crossed a Schelling point
        new_point = self._get_schelling_point(tension)
        if new_point != self.last_schelling_point_crossed:
            self.last_schelling_point_crossed = new_point
            self.steps_in_current_regime = 0
        else:
            self.steps_in_current_regime += 1
        
        # Expire old offers
        self.past_offers = [o for o in self.past_offers 
                           if self.current_step < o.expires_in]
    
    def _get_schelling_point(self, tension: float) -> SchelllingPoint:
        """Determine which Schelling point we're at."""
        if tension < 0.35:
            return SchelllingPoint.PEACETIME
        elif tension < 0.50:
            return SchelllingPoint.FIRST_ALERT
        elif tension < 0.65:
            return SchelllingPoint.MOBILIZATION_TRIGGER
        elif tension < 0.72:
            return SchelllingPoint.AIRSPACE_CLOSURE
        elif tension < 0.80:
            return SchelllingPoint.FORWARD_DEPLOYMENT
        elif tension < 0.90:
            return SchelllingPoint.STRATEGIC_STRIKE
        else:
            return SchelllingPoint.MUTUAL_DESTRUCTION
    
    def make_commitment(self, side: str, commitment_type: EscalationCommitment, 
                       intensity: float, current_step: int):
        """Record an irreversible commitment."""
        action = IrreversibleAction(
            action_type=commitment_type,
            timestamp=current_step,
            intensity=intensity,
            can_reverse_by=self._get_reversal_window(commitment_type),
            face_saving_required=self._get_face_saving_cost(commitment_type, intensity)
        )
        
        if side == "Blue":
            self.commitments_blue.append(action)
        else:
            self.commitments_red.append(action)
    
    def _get_reversal_window(self, commitment_type: EscalationCommitment) -> int:
        """How many steps before commitment becomes truly irreversible."""
        windows = {
            EscalationCommitment.DEPLOYMENT: 8,           # Can pull back units within 8 steps
            EscalationCommitment.SANCTIONS: 15,           # Can lift sanctions slowly
            EscalationCommitment.CYBER_ATTACK: 3,         # Cyber attacks are immediately escalatory
            EscalationCommitment.AIRSPACE_CLOSURE: 5,     # Airspace closures are easy to reverse
            EscalationCommitment.PUBLIC_ULTIMATUM: 4,     # Public statements are hard to walk back
            EscalationCommitment.FIRST_STRIKE: 1,         # Kinetic strike is point of no return
        }
        return windows.get(commitment_type, 5)
    
    def _get_face_saving_cost(self, commitment_type: EscalationCommitment, 
                             intensity: float) -> float:
        """What must opponent concede for us to back down from this commitment."""
        base_costs = {
            EscalationCommitment.DEPLOYMENT: 0.30,
            EscalationCommitment.SANCTIONS: 0.25,
            EscalationCommitment.CYBER_ATTACK: 0.60,
            EscalationCommitment.AIRSPACE_CLOSURE: 0.20,
            EscalationCommitment.PUBLIC_ULTIMATUM: 0.40,
            EscalationCommitment.FIRST_STRIKE: 1.00,  # Cannot retreat
        }
        return base_costs.get(commitment_type, 0.3) * intensity
    
    def get_commitment_penalty(self, side: str) -> float:
        """
        Cost to escalate further given existing commitments.
        More commitments = higher cost to reverse or escalate further.
        """
        commitments = self.commitments_blue if side == "Blue" else self.commitments_red
        
        penalty = 0.0
        for commitment in commitments:
            if commitment.is_reversible(self.current_step):
                # Reversible commitments have escalation cost (sunk cost fallacy)
                penalty += 0.1 * commitment.intensity
            else:
                # Irreversible commitments lock you in (no option to retreat)
                penalty += 0.05  # Fixed cost of being locked in
        
        return penalty
    
    def get_reversible_commitments(self, side: str) -> List[IrreversibleAction]:
        """Get list of commitments that can still be reversed."""
        commitments = self.commitments_blue if side == "Blue" else self.commitments_red
        return [c for c in commitments if c.is_reversible(self.current_step)]
    
    def get_irreversible_commitments(self, side: str) -> List[IrreversibleAction]:
        """Get list of commitments that cannot be reversed."""
        commitments = self.commitments_blue if side == "Blue" else self.commitments_red
        return [c for c in commitments if not c.is_reversible(self.current_step)]
    
    def propose_deescalation(self, offering_side: str, 
                            concessions: Dict[str, float],
                            face_saving_offer: str,
                            confidence: float = 0.6) -> DeescalationOffer:
        """Create formal de-escalation proposal."""
        offer = DeescalationOffer(
            offering_side=offering_side,
            concessions=concessions,
            face_saving=face_saving_offer,
            confidence=confidence,
            expires_in=self.current_step + 5  # Offer valid for 5 steps
        )
        self.past_offers.append(offer)
        return offer
    
    def can_accept_deescalation_offer(self, side: str, offer: DeescalationOffer) -> bool:
        """
        Check if accepting this offer allows side to save face.
        Considers existing commitments and face-saving requirements.
        """
        own_commitments = self.get_reversible_commitments(side)
        
        if not own_commitments:
            return True  # No commitments to back down from
        
        # Calculate if offered face-saving is enough
        face_saving_required = sum(c.face_saving_required for c in own_commitments)
        face_saving_offered = offer.confidence  # Simplified: confidence = face-saving value
        
        return face_saving_offered >= face_saving_required * 0.6


class DeescalationMechanism:
    """Implements de-escalation dynamics with windows and face-saving."""
    
    def __init__(self):
        self.deescalation_locked = False  # Once kinetic strikes happen, no de-escalation
        self.crisis_peak_step = None
        self.deescalation_attempted = False
        self.successful_de_escalations = 0
    
    def reset(self):
        self.deescalation_locked = False
        self.crisis_peak_step = None
        self.deescalation_attempted = False
        self.successful_de_escalations = 0
    
    def get_deescalation_window(self, current_tension: float, steps_in_crisis: int) -> bool:
        """
        Limited window exists for coordinated de-escalation.
        Window: after crisis peaks but before commitment becomes irreversible.
        """
        if self.deescalation_locked:
            return False
        
        # Window opens after 5 steps of high tension, closes after 15
        high_tension = current_tension > 0.72
        return high_tension and 5 < steps_in_crisis < 15
    
    def compute_deescalation_discount(self, mutual_agreement: bool, 
                                     face_saving_enabled: bool) -> float:
        """
        Reward multiplier for de-escalation.
        Increases if both sides agree and face-saving is available.
        """
        base_discount = 2.0 if mutual_agreement else 1.0
        if face_saving_enabled:
            base_discount *= 1.5
        
        return base_discount
    
    def lock_deescalation(self):
        """Lock out de-escalation once kinetic strikes occur."""
        self.deescalation_locked = True
    
    def compute_unraveling_cost(self, num_irreversible_commitments: int) -> float:
        """
        Cost of unraveling escalation spiral.
        Increases exponentially with number of irreversible commitments.
        """
        return 0.1 * (1.5 ** num_irreversible_commitments)


class StrategicNegotiation:
    """Models strategic signaling and negotiation for de-escalation."""
    
    @staticmethod
    def get_credible_deescalation_signal(side: str, current_tension: float,
                                        own_commitments: int, 
                                        enemy_commitments: int) -> Tuple[str, float]:
        """
        Generate credible de-escalation signal that enemy will believe.
        
        Returns: (signal_type, credibility)
        - Unilateral retreat: Low credibility but high reward if successful
        - Mutual step-back: Moderate credibility, moderate reward
        - Third-party mediation: Low immediate effect but face-saving
        - Economic sweetener: Direct incentive to back down
        """
        signals = []
        
        # Signal 1: Unilateral retreat from forward positions
        if own_commitments > 2:
            credibility = 0.3 + 0.1 * min(1, enemy_commitments / own_commitments)
            signals.append(("unilateral_pullback", credibility))
        
        # Signal 2: Mutual step-back proposal
        if current_tension > 0.7:
            credibility = 0.6 + 0.2 * min(1, own_commitments / max(1, enemy_commitments))
            signals.append(("mutual_step_back", credibility))
        
        # Signal 3: Third-party mediation (always available)
        credibility = 0.5
        signals.append(("third_party_mediation", credibility))
        
        # Signal 4: Economic incentive (sanctions relief, trade)
        if side == "Blue" and current_tension > 0.65:
            credibility = 0.7
            signals.append(("economic_sweetener", credibility))
        
        if not signals:
            return ("status_quo", 0.0)
        
        # Return most credible signal
        return max(signals, key=lambda x: x[1])
    
    @staticmethod
    def compute_mutual_agreement_probability(blue_signal: Tuple[str, float],
                                            red_signal: Tuple[str, float],
                                            tension: float,
                                            schelling_point: SchelllingPoint) -> float:
        """
        Probability both sides agree to same de-escalation path.
        """
        signal_compatibility = {
            ("mutual_step_back", "mutual_step_back"): 0.85,
            ("third_party_mediation", "third_party_mediation"): 0.75,
            ("mutual_step_back", "third_party_mediation"): 0.60,
            ("third_party_mediation", "mutual_step_back"): 0.60,
            ("economic_sweetener", "mutual_step_back"): 0.70,
        }
        
        blue_name, blue_cred = blue_signal
        red_name, red_cred = red_signal
        
        base_compat = signal_compatibility.get((blue_name, red_name), 0.3)
        credibility_boost = 0.1 * (blue_cred + red_cred) / 2.0
        
        # At Schelling points, agreement is harder
        schelling_penalty = {
            SchelllingPoint.PEACETIME: 0.0,
            SchelllingPoint.FIRST_ALERT: -0.05,
            SchelllingPoint.MOBILIZATION_TRIGGER: -0.10,
            SchelllingPoint.AIRSPACE_CLOSURE: -0.20,
            SchelllingPoint.FORWARD_DEPLOYMENT: -0.30,
            SchelllingPoint.STRATEGIC_STRIKE: -0.50,
            SchelllingPoint.MUTUAL_DESTRUCTION: -0.70,
        }
        
        penalty = schelling_penalty.get(schelling_point, -0.15)
        
        return np.clip(base_compat + credibility_boost + penalty, 0.0, 1.0)
