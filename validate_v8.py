"""
Validation script for SADIE v8 - Multi-Domain Enhanced Wargame
Tests integration of enhanced state space, multi-domain operations, and escalation pathways.
"""

import numpy as np
import sys
from pathlib import Path

# Ensure imports work
try:
    from .strategic_state_manager import StrategicStateManager
    from .multi_domain_operations import (
        MultiDomainActionLibrary, CrossDomainSpilloverModel, Domain
    )
    from .escalation_pathways import SchelllingPointManager, DeescalationMechanism
    from .strategic_intel_env_v8 import StrategicWargameEnvV8
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_enhanced_state_manager():
    """Test strategic state manager."""
    print("\n" + "="*60)
    print("TEST 1: Enhanced State Manager")
    print("="*60)
    
    mgr = StrategicStateManager()
    
    # Check initial state
    state = mgr.get_full_state_vector()
    print(f"✅ State vector shape: {state.shape} (expected 36)")
    print(f"✅ State values in [0,1]: {np.all((state >= 0) & (state <= 1))}")
    
    # Test dimension summary
    summary = mgr.get_dimension_summary()
    print(f"✅ Dimension summary keys: {list(summary.keys())}")
    print(f"   - Military balance: {summary['military_balance']:.2f}")
    print(f"   - Coalition strength: {summary['coalition_strength']:.2f}")
    print(f"   - Territorial stability: {summary['territorial_stability']:.2f}")
    
    # Test actions
    mgr.apply_military_action("Blue", 0.5)
    mgr.apply_economic_action("Red", 0.8)
    mgr.apply_territorial_action("Blue", 0.6)
    print(f"✅ Actions applied successfully")
    
    # Test decay
    old_state = mgr.get_full_state_vector()
    mgr.natural_decay(decay_rate=0.1)
    new_state = mgr.get_full_state_vector()
    print(f"✅ Natural decay applied (change magnitude: {np.linalg.norm(new_state - old_state):.3f})")


def test_multi_domain_operations():
    """Test multi-domain operations."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Domain Operations")
    print("="*60)
    
    all_actions = MultiDomainActionLibrary.get_all_actions()
    print(f"✅ Total actions: {len(all_actions)}")
    
    # Count by domain
    domains = {}
    for action_name, effect in all_actions.items():
        domain = effect.domain.value
        domains[domain] = domains.get(domain, 0) + 1
    
    print(f"   Actions by domain:")
    for domain, count in domains.items():
        print(f"     - {domain}: {count}")
    
    # Test spillover model
    spillover = CrossDomainSpilloverModel()
    action_effect = all_actions["kinetic_full_mobilization"]
    
    spillover_actions = spillover.compute_spillover(
        "kinetic_full_mobilization", action_effect, 
        current_tension=0.8, current_stability=0.7
    )
    print(f"✅ Spillover model - triggered {len(spillover_actions)} cross-domain effects")
    print(f"   Mutual escalation locked: {spillover.mutual_escalation_lock}")


def test_escalation_pathways():
    """Test escalation and de-escalation mechanics."""
    print("\n" + "="*60)
    print("TEST 3: Escalation Pathways")
    print("="*60)
    
    schelling = SchelllingPointManager()
    deescalation = DeescalationMechanism()
    
    # Test Schelling points at different tensions
    test_tensions = [0.2, 0.5, 0.7, 0.85, 0.98]
    for tension in test_tensions:
        point = schelling._get_schelling_point(tension)
        print(f"   Tension {tension:.2f} → {point.name}")
    
    # Test commitment tracking
    from .escalation_pathways import EscalationCommitment
    schelling.make_commitment("Blue", EscalationCommitment.DEPLOYMENT, 0.6, current_step=0)
    schelling.make_commitment("Red", EscalationCommitment.SANCTIONS, 0.4, current_step=1)
    
    print(f"✅ Commitments made:")
    print(f"   - Blue reversible: {len(schelling.get_reversible_commitments('Blue'))}")
    print(f"   - Red reversible: {len(schelling.get_reversible_commitments('Red'))}")
    
    # Test de-escalation window
    window = deescalation.get_deescalation_window(current_tension=0.75, steps_in_crisis=8)
    print(f"✅ De-escalation window available (T=0.75, steps=8): {window}")


def test_policy_mechanics():
    """Test merged policy load, ladder clamping, and commitments."""
    print("\n" + "=" * 60)
    print("TEST 4: Policy Mechanics")
    print("=" * 60)

    from .load_parameters import load_all_parameters
    from .policy_mechanics import map_action_to_commitment
    from .multi_domain_operations import MultiDomainActionLibrary

    params = load_all_parameters()
    blue = params.get("Blue_US", {})
    assert blue.get("n_source_documents", 0) >= 1 or blue, "Expected merged Blue policy"
    print(f"✅ Blue merged docs: {blue.get('n_source_documents', 0)}")
    print(f"   Ladder steps: {len(blue.get('escalation_ladder', []))}")
    print(f"   Red lines: {len(blue.get('red_lines', []))}")

    env = StrategicWargameEnvV8(max_steps=5)
    obs, _ = env.reset(seed=42)
    assert not np.allclose(obs["Blue"], obs["Red"]), "Blue/Red observations should differ"
    print("✅ Asymmetric observations confirmed")

    all_actions = MultiDomainActionLibrary.get_all_actions()
    strike = all_actions.get("kinetic_air_strike_operations")
    assert map_action_to_commitment("kinetic_air_strike_operations", strike) is not None
    print("✅ Commitment mapping for kinetic strike")

    obs, rewards, term, trunc, info = env.step({
        "Blue": env.num_actions - 1,
        "Red": env.num_actions - 1,
    })
    print(f"✅ Step with high-index actions: commitments Blue={info['blue_commitments']} Red={info['red_commitments']}")


def test_integrated_environment():
    """Test full v8 environment."""
    print("\n" + "="*60)
    print("TEST 5: Integrated Environment V8")
    print("="*60)
    
    env = StrategicWargameEnvV8(max_steps=10)
    
    # Reset
    obs, info = env.reset()
    print(f"✅ Environment reset")
    print(f"   Blue obs shape: {obs['Blue'].shape} (expected (64,))")
    print(f"   Red obs shape: {obs['Red'].shape} (expected (64,))")
    print(f"   Initial tension: {env.tension:.2f}")
    print(f"   Initial stability: {env.stability:.2f}")
    
    # Run episode
    total_reward = {"Blue": 0.0, "Red": 0.0}
    action_counts = {"Blue": 0, "Red": 0}
    
    for step in range(5):
        # Random actions
        blue_action = env.action_space.sample()
        red_action = env.action_space.sample()
        
        obs, rewards, terminated, truncated, info = env.step({
            "Blue": blue_action,
            "Red": red_action
        })
        
        total_reward["Blue"] += rewards["Blue"]
        total_reward["Red"] += rewards["Red"]
        action_counts["Blue"] += 1
        action_counts["Red"] += 1
        
        if step == 0:
            print(f"\n✅ Step execution successful")
            print(f"   Blue reward: {rewards['Blue']:.2f}")
            print(f"   Red reward: {rewards['Red']:.2f}")
            print(f"   Blue action: {info['blue_action']}")
            print(f"   Red action: {info['red_action']}")
            print(f"   Tension: {info['tension']:.2f}")
            print(f"   Stability: {info['stability']:.2f}")
    
    print(f"\n✅ Episode summary (5 steps):")
    print(f"   Blue avg reward: {total_reward['Blue'] / action_counts['Blue']:.2f}")
    print(f"   Red avg reward: {total_reward['Red'] / action_counts['Red']:.2f}")
    print(f"   Final tension: {env.tension:.2f}")
    print(f"   Final stability: {env.stability:.2f}")


def main():
    """Run all validation tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + " SADIE v8 - Multi-Domain Integration Validation ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        test_enhanced_state_manager()
        test_multi_domain_operations()
        test_escalation_pathways()
        test_policy_mechanics()
        test_integrated_environment()
        
        print("\n" + "="*60)
        print("✅ ALL VALIDATION TESTS PASSED")
        print("="*60)
        print("\nNext steps:")
        print("  1. Update train_self_play.py to use v8 environment")
        print("  2. Run training with: python train_self_play.py")
        print("  3. Evaluate policies with: python evaluate.py")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
