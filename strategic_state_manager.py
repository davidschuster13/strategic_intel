"""
Strategic State Manager - Enhanced multi-dimensional competition model
Manages military, economic, territorial, alliance, and resource dimensions
for realistic two-nation strategic competition.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class MilitaryDimension:
    """Military balance: force structure, readiness, positioning."""
    blue_naval_strength: float = 0.72  # Fleet size, capability, readiness
    red_naval_strength: float = 0.70
    blue_air_superiority: float = 0.71  # Fighter aircraft, air defense systems
    red_air_superiority: float = 0.70
    blue_ground_forces: float = 0.70  # Ground army quality, readiness
    red_ground_forces: float = 0.72
    blue_missile_readiness: float = 0.68  # ICBM, cruise missile capability
    red_missile_readiness: float = 0.74
    blue_forward_deployment: float = 0.45  # Forces positioned in contested zone
    red_forward_deployment: float = 0.55
    
    def balance_ratio(self) -> float:
        """Overall military balance (0=Red dominant, 0.5=balanced, 1=Blue dominant)."""
        blue_total = (self.blue_naval_strength + self.blue_air_superiority + 
                     self.blue_ground_forces + self.blue_missile_readiness) / 4.0
        red_total = (self.red_naval_strength + self.red_air_superiority + 
                    self.red_ground_forces + self.red_missile_readiness) / 4.0
        return np.clip(blue_total / (blue_total + red_total + 1e-6), 0.0, 1.0)


@dataclass
class EconomicDimension:
    """Economic pressure: sanctions, trade disruption, financial isolation, energy leverage."""
    blue_sanctions_level: float = 0.0  # Cumulative sanctions (0=none, 1=severe)
    red_sanctions_level: float = 0.0
    blue_trade_disruption: float = 0.0  # Trade volume impact (0=normal, 1=blocked)
    red_trade_disruption: float = 0.0
    blue_financial_isolation: float = 0.0  # Access to capital markets (0=normal, 1=frozen)
    red_financial_isolation: float = 0.0
    energy_leverage: float = 0.38  # Russia's leverage over Europe energy (0=none, 1=full)
    energy_vulnerability: float = 0.38  # Europe's vulnerability (increases with sanctions)
    
    def economic_pressure_on_red(self) -> float:
        """Aggregate economic pressure on Russia."""
        return np.clip(0.4 * self.blue_sanctions_level + 0.3 * self.blue_trade_disruption + 
                       0.3 * self.blue_financial_isolation, 0.0, 1.0)
    
    def economic_pressure_on_blue(self) -> float:
        """Aggregate economic pressure on US/NATO (via energy leverage)."""
        return np.clip(self.energy_leverage * self.energy_vulnerability, 0.0, 1.0)


@dataclass
class TerritorialDimension:
    """Territorial control and claims: contested terrain and strategic positioning."""
    blue_territorial_control: float = 0.40  # General strategic terrain control ratio for Blue
    red_territorial_control: float = 0.45
    disputed_zone_control: float = 0.50  # 0=Red controls, 1=Blue controls, 0.5=contested
    blue_forward_bases: int = 2  # Forward military/logistics bases
    red_forward_bases: int = 3
    blue_logistics_access: float = 0.30  # Access to key logistics and maritime routes
    red_logistics_access: float = 0.70
    territorial_claims_tension: float = 0.0  # Escalation from overlapping territorial claims

    def territorial_stability(self) -> float:
        """Stability of territorial arrangement (1=stable, 0=crisis)."""
        control_gap = abs(self.blue_territorial_control - self.red_territorial_control)
        base_threat = abs(self.blue_forward_bases - self.red_forward_bases) / 5.0
        return np.clip(1.0 - (control_gap + base_threat + self.territorial_claims_tension) / 3.0, 0.0, 1.0)


@dataclass
class AllianceDimension:
    """Alliance strength: NATO cohesion, partner reliability, coalition capability."""
    nato_cohesion: float = 0.85  # Unified response (0.5=fractured, 1.0=united)
    nato_military_commitment: float = 0.70  # Willingness to deploy forces
    partner_reliability: Dict[str, float] = field(default_factory=lambda: {
        "Norway": 0.95,
        "Poland": 0.90,
        "UK": 0.92,
        "Canada": 0.88,
        "Finland": 0.87,
        "Sweden": 0.82
    })
    schengen_cohesion: float = 0.80  # European unity against Russian actions
    us_commitment: float = 0.85  # US commitment to NATO/Pacific
    
    def coalition_strength(self) -> float:
        """Overall Western coalition military and political strength."""
        avg_partner = np.mean(list(self.partner_reliability.values()))
        return np.clip(0.3 * self.nato_cohesion + 0.3 * self.us_commitment + 
                       0.2 * avg_partner + 0.2 * self.schengen_cohesion, 0.0, 1.0)


@dataclass
class ResourceDimension:
    """Strategic resources: energy, minerals, rare earths, and critical supply vulnerabilities."""
    critical_resource_access: float = 0.40  # General access to critical materials and energy
    strategic_supply_vulnerability: float = 0.35  # How vulnerable supply chains are to disruption
    resource_dependency: float = 0.6  # Dependency on adversary-controlled resources
    strategic_minerals: float = 0.5  # Access to critical minerals for tech/military
    food_security: float = 0.7  # Food security and supply resilience
    resource_competition_intensity: float = 0.2  # Level of resource competition

    def resource_leverage(self) -> float:
        """Adversary resource leverage in the strategic competition."""
        return np.clip((self.critical_resource_access * 0.4 + 
                       self.resource_dependency * 0.3 + 
                       (1 - self.food_security) * 0.3) / 1.0, 0.0, 1.0)


class StrategicStateManager:
    """Manages all dimensions of strategic competition."""
    
    def __init__(self):
        self.military = MilitaryDimension()
        self.economic = EconomicDimension()
        self.territorial = TerritorialDimension()
        self.alliance = AllianceDimension()
        self.resources = ResourceDimension()
        
        # Track escalation history for commitment modeling
        self.blue_committed_actions = []  # Irreversible commitments
        self.red_committed_actions = []
        
    def reset(self):
        """Reset to baseline state."""
        self.military = MilitaryDimension()
        self.economic = EconomicDimension()
        self.territorial = TerritorialDimension()
        self.alliance = AllianceDimension()
        self.resources = ResourceDimension()
        self.blue_committed_actions = []
        self.red_committed_actions = []
    
    def get_full_state_vector(self) -> np.ndarray:
        """Return complete state as normalized vector for neural networks."""
        state = np.array([
            # Military (8)
            self.military.blue_naval_strength,
            self.military.red_naval_strength,
            self.military.blue_air_superiority,
            self.military.red_air_superiority,
            self.military.blue_ground_forces,
            self.military.red_ground_forces,
            self.military.blue_forward_deployment,
            self.military.red_forward_deployment,
            
            # Economic (8)
            self.economic.blue_sanctions_level,
            self.economic.red_sanctions_level,
            self.economic.blue_trade_disruption,
            self.economic.red_trade_disruption,
            self.economic.blue_financial_isolation,
            self.economic.red_financial_isolation,
            self.economic.energy_leverage,
            self.economic.energy_vulnerability,
            
            # Territorial (8)
            self.territorial.blue_territorial_control,
            self.territorial.red_territorial_control,
            self.territorial.disputed_zone_control,
            self.territorial.blue_forward_bases / 5.0,  # Normalize
            self.territorial.red_forward_bases / 5.0,
            self.territorial.blue_logistics_access,
            self.territorial.red_logistics_access,
            self.territorial.territorial_claims_tension,

            # Alliance (6)
            self.alliance.nato_cohesion,
            self.alliance.nato_military_commitment,
            self.alliance.schengen_cohesion,
            self.alliance.us_commitment,
            np.mean(list(self.alliance.partner_reliability.values())),
            self.alliance.coalition_strength(),
            
            # Resources (6)
            self.resources.critical_resource_access,
            self.resources.strategic_supply_vulnerability,
            self.resources.resource_dependency,
            self.resources.strategic_minerals,
            self.resources.food_security,
            self.resources.resource_competition_intensity,
        ], dtype=np.float32)
        
        return np.clip(state, 0.0, 1.0)
    
    def apply_military_action(self, side: str, intensity: float):
        """Apply effects of military escalation action."""
        if side == "Blue":
            if intensity < 0.3:  # Low: exercises, readiness
                self.military.blue_forward_deployment += 0.02
            elif intensity < 0.7:  # Medium: mobilization
                self.military.blue_forward_deployment += 0.08
                self.military.blue_naval_strength += 0.03
            else:  # High: deployment
                self.military.blue_forward_deployment += 0.15
                self.military.blue_naval_strength += 0.05
                self.military.blue_air_superiority += 0.03
        else:
            if intensity < 0.3:
                self.military.red_forward_deployment += 0.03
            elif intensity < 0.7:
                self.military.red_forward_deployment += 0.10
                self.military.red_naval_strength += 0.04
            else:
                self.military.red_forward_deployment += 0.18
                self.military.red_naval_strength += 0.06
                self.military.red_ground_forces += 0.04
        
        # Normalize
        self.military.blue_forward_deployment = np.clip(self.military.blue_forward_deployment, 0.0, 1.0)
        self.military.red_forward_deployment = np.clip(self.military.red_forward_deployment, 0.0, 1.0)
    
    def apply_economic_action(self, side: str, intensity: float):
        """Apply effects of economic action (sanctions, trade pressure)."""
        if side == "Blue":
            if intensity < 0.3:  # Light sanctions
                self.economic.blue_sanctions_level += 0.05
            elif intensity < 0.7:  # Medium sanctions
                self.economic.blue_sanctions_level += 0.12
                self.economic.blue_trade_disruption += 0.08
            else:  # Severe sanctions
                self.economic.blue_sanctions_level += 0.20
                self.economic.blue_trade_disruption += 0.15
                self.economic.blue_financial_isolation += 0.10
                # Sanctions increase energy vulnerability
                self.economic.energy_vulnerability += 0.05
        else:
            # Red economic actions are more limited (weaker economy)
            if intensity < 0.5:  # Energy leverage
                self.economic.energy_leverage += 0.03
            else:  # Disruption
                self.economic.energy_leverage += 0.08
                self.economic.red_trade_disruption += 0.05
        
        # Normalize
        for attr in ["blue_sanctions_level", "red_sanctions_level", "blue_trade_disruption", 
                     "red_trade_disruption", "blue_financial_isolation", "red_financial_isolation",
                     "energy_leverage", "energy_vulnerability"]:
            setattr(self.economic, attr, np.clip(getattr(self.economic, attr), 0.0, 1.0))
    
    def apply_territorial_action(self, side: str, intensity: float):
        """Apply effects of territorial/positioning action."""
        if side == "Blue":
            if intensity < 0.5:
                self.territorial.blue_territorial_control += 0.02
            else:
                self.territorial.blue_territorial_control += 0.05
                self.territorial.blue_forward_bases = min(5, self.territorial.blue_forward_bases + 1)
                self.territorial.territorial_claims_tension += 0.10
        else:
            if intensity < 0.5:
                self.territorial.red_territorial_control += 0.03
            else:
                self.territorial.red_territorial_control += 0.06
                self.territorial.red_forward_bases = min(5, self.territorial.red_forward_bases + 1)
                self.territorial.territorial_claims_tension += 0.15
        
        self.territorial.blue_territorial_control = np.clip(self.territorial.blue_territorial_control, 0.0, 1.0)
        self.territorial.red_territorial_control = np.clip(self.territorial.red_territorial_control, 0.0, 1.0)
        self.territorial.territorial_claims_tension = np.clip(self.territorial.territorial_claims_tension, 0.0, 1.0)
    
    def natural_decay(self, decay_rate: float = 0.02):
        """Slow natural decay of escalation (tension reduces if no new pressure)."""
        # Economic measures have long halflife
        self.economic.blue_sanctions_level -= decay_rate * 0.3
        self.economic.red_sanctions_level -= decay_rate * 0.3
        self.economic.blue_trade_disruption -= decay_rate * 0.5
        self.economic.red_trade_disruption -= decay_rate * 0.5
        
        # Military posture decays faster
        self.military.blue_forward_deployment -= decay_rate * 0.8
        self.military.red_forward_deployment -= decay_rate * 0.8
        
        # Territorial tension decays slowly
        self.territorial.territorial_claims_tension -= decay_rate * 0.2
        
        # Energy leverage is sticky
        self.economic.energy_leverage -= decay_rate * 0.1
        
        # Normalize
        for dim in [self.military, self.economic, self.territorial, self.alliance, self.resources]:
            for attr, value in dim.__dict__.items():
                if isinstance(value, float):
                    setattr(dim, attr, np.clip(value, 0.0, 1.0))
    
    def get_dimension_summary(self) -> Dict[str, float]:
        """Get summary metrics for each strategic dimension."""
        return {
            "military_balance": self.military.balance_ratio(),  # 0=Red dom, 1=Blue dom
            "economic_pressure_red": self.economic.economic_pressure_on_red(),
            "economic_pressure_blue": self.economic.economic_pressure_on_blue(),
            "territorial_stability": self.territorial.territorial_stability(),
            "coalition_strength": self.alliance.coalition_strength(),
            "resource_leverage": self.resources.resource_leverage(),
        }
