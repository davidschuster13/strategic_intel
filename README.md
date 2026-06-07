# SADIE: Strategic Action Decision and Inferencing Engine

A multi-domain, policy-parameterized wargame environment for studying strategic competition between US/NATO (Blue) and Adversary (Red) forces with realistic escalation dynamics.

## Core Files

### Environment & Training

#### Version 8 (Enhanced - Multi-Domain + Escalation) - RECOMMENDED
- **`strategic_intel_env_v8.py`** - Enhanced Gymnasium environment with multi-domain operations, enhanced state space, and realistic escalation pathways
- **`train_self_play_v8.py`** - Self-play training for v8 with policy pool (avoid non-stationary learning)
- **`strategic_state_manager.py`** - Manages 5 strategic dimensions: Military, Economic, Territorial, Alliance, Resources
- **`multi_domain_operations.py`** - 24 multi-domain actions (kinetic, cyber, economic, information) with cross-domain spillover
- **`escalation_pathways.py`** - Schelling points, irreversible commitments, de-escalation windows, face-saving mechanisms
- **`validate_v8.py`** - Integration validation tests (run first to verify setup)

#### Version 7 (Legacy - Deprecated)
- **`strategic_intel_env_v7.py`** - ⚠️ Deprecated (kept for baseline comparison)
  - Use `strategic_intel_env_v8.py` for new development
  - See `legacy/LEGACY.md` for archive information
- **Training script moved**: `legacy/train_self_play_v7.py`
- **Old models/logs**: `legacy/models_v7/` and `legacy/tb_logs_v7/`

### Policy Parameterization
- **`ingest_policy_docs.py`** - Extracts strategic parameters from Arctic policy PDFs using Ollama/LlamaIndex
- **`load_parameters.py`** - Loads parameterized policy data for agents
- **`self_play_env.py`** - Wrapper for self-play training (fixes recursion issues)

### Data & Models
- **`parameterized_output/`** - Extracted policy parameters (JSON)
- **`policy_docs/`** - Source policy documents (PDFs)
- **`models/`** - Trained PPO models (Blue/Red pairs)
- **`models/v8/`** - v8 models (multi-domain policies)
- **`SANITY_CHECK_REPORT.md`** - Comprehensive audit and fixes documentation

## Quick Start

### 1. Validate v8 Setup (First Time)
```bash
python validate_v8.py
```
Integration tests verify all v8 systems work together.

### 2. Train Agents (v8 - RECOMMENDED)
```bash
python train_self_play_v8.py
```
Trains Blue and Red agents using 24 multi-domain actions with realistic escalation dynamics.

### 3. Evaluate Performance
```bash
python evaluate.py
```
Evaluates trained v8 agents using multi-domain action analysis and enhanced strategic metrics (military balance, coalition strength, resource leverage).

### 4. Parameterize New Policies
```bash
python ingest_policy_docs.py
```

### Legacy v7 (Baseline Comparison)
See `legacy/LEGACY.md` for instructions on running v7 for baseline comparison.
```

## Key Features

### v8 Enhancements (Multi-Domain + Realistic Escalation) ✨ NEW

#### 1. **Enhanced State Space**
Five strategic dimensions instead of three:
- **Military**: Naval strength, air superiority, ground forces, missile readiness, forward deployment
- **Economic**: Sanctions levels, trade disruption, financial isolation, energy leverage
- **Territorial**: Area control %, disputed zones, forward bases, shipping access
- **Alliance**: NATO cohesion, partner reliability (Norway, Poland, UK, Canada, Finland, Sweden), coalition strength
- **Resources**: Oil/gas access, rare earth dependency, strategic minerals, food security

State vector: **36 dimensions** (vs 32 in v7) for richer strategic modeling

#### 2. **Multi-Domain Operations**
24 strategic actions across 4 domains:
- **Kinetic** (6): readiness exercises → air strikes
- **Cyber** (6): passive monitoring → command/control disruption
- **Economic** (6): trade monitoring → energy weaponization
- **Information** (6): intelligence sharing → existential messaging

Each action has:
- Intensity level (0.0-1.0)
- Escalation cost (0.0-1.0)
- Visibility (how observable: 0.05-1.0)
- Reversibility (0.0-0.95)
- Cross-domain spillover effects

#### 3. **Realistic Escalation Pathways**
- **Schelling Points**: 7 natural decision thresholds (peacetime → mutual destruction)
- **Irreversible Commitments**: Military deployments, sanctions regimes, cyber strikes with reversal windows
- **De-escalation Windows**: Realistic 5-15 step windows for coordinated backing down
- **Face-Saving Mechanisms**: Strategic signaling for mutual agreement on off-ramps
- **Commitment Tracking**: Both sides' escalation spiral history affects future options

#### 5. **Balanced Reward Structure**
Rewards shaped for strategic competition rather than escalation avoidance:
- **Blue**: Rewarded for military advantage, coalition strength, resource security
- **Red**: Rewarded for military advantage, resource control, economic pressure
- **Tension Management**: Optimal range (0.3-0.8) rewarded; extremes penalized
- **Domain-Specific**: Each domain has strategic value; decisive actions rewarded over half-measures
- **Commitment Costs**: Irreversible actions penalized appropriately
- Cyber attacks may trigger kinetic posturing
- Sanctions increase willingness for cyber operations
- Information warfare erodes alliance cohesion
- Kinetic escalation locks out de-escalation pathways

#### 5. **Domain-Specific Sensor Models**
Agents have imperfect information:
- Military observers see kinetic operations well, cyber/information poorly
- Cyber operators see infrastructure, miss diplomatic initiatives
- Economic actors see sanctions flows, miss military planning
- Information specialists see messaging, not military capabilities

### v7 Features (Simplified Model)

✅ **Policy-Driven Agents** - Behavior grounded in real Arctic strategy documents
✅ **Asymmetric Observations** - Agents see own policy biases, not opponent's
✅ **Self-Play Training** - Policy pool prevents non-stationary learning
✅ **Strategic Realism** - Tension, stability, and influence dynamics

## Architecture Comparison

| Aspect | v7 (Legacy) | v8 (Enhanced) |
|--------|-----------|--------------|
| **State Dimensions** | 32 (tension/awareness/influence/stability + action history) | 64 (military/economic/territorial/alliance/resources + escalation tracking) |
| **Action Space** | 10 discrete actions | 24 multi-domain actions |
| **Strategic Dimensions** | 3 (tension/stability/influence) | 5 (military/economic/territorial/alliance/resources) |
| **Escalation Model** | Simple linear escalation | Schelling points + irreversible commitments + de-escalation windows |
| **Cross-Domain Effects** | None | Spillover between kinetic/cyber/economic/information |
| **Observation Model** | Symmetric + policy bias | Asymmetric + domain-specific sensor noise |
| **De-Escalation** | Not explicitly modeled | Face-saving mechanisms + strategic signaling |
| **Commitment Tracking** | None | Irreversible actions with reversal windows |

## Reward Structure Evolution

### v7 Rewards
- Blue: Minimize tension + maintain awareness + preserve stability + contain influence
- Red: Maximize tension + maximize influence
- Simple escalation costs

### v8 Rewards
- Blue: Multi-objective (military balance, alliance strength, strategic stability)
- Red: Strategic leverage + Arctic resource access + maintain flexibility
- Domain-specific action preferences
- Commitment costs (more commitments = higher de-escalation cost)
- Crisis-level penalties (exponential as tension > 0.72)
- De-escalation opportunity bonuses (reward diplomatic signals in crisis)
- Spillover interaction costs (mutual escalation locks reduce options)

## Configuration

### Environment Parameters (v8)
Edit `strategic_intel_env_v8.py`:
- `max_steps`: Episode length (default: 60, larger = deeper escalation scenarios)
- `StrategicStateManager`: Initial state biases
- Reward weights in `_compute_rewards()`

### Training Parameters (v8)
Edit `train_self_play_v8.py`:
- `iterations`: Number of self-play rounds (default: 16)
- `steps_per_round`: Timesteps per iteration (default: 60k)
- PPO hyperparameters: learning_rate, n_steps, batch_size, n_epochs
- Policy pool sampling strategy

### Curriculum Learning
v8 automatically starts episodes in three regimes:
- 25% low tension (0.18-0.32) - peaceful scenarios
- 50% baseline (0.30-0.45) - normal operations
- 25% high tension (0.52-0.70) - crisis starts

## Expected Behavior

### Learned Strategies

#### Blue (US/NATO) - Restraint Bias
- Prefers diplomatic/information actions early
- Escalates carefully to avoid instability collapse
- Uses economic sanctions as pressure tool
- Seeks de-escalation off-ramps when tension > 0.70
- Maintains alliance cohesion

#### Red (Russia/Adversary) - Assertiveness Bias
- Exploits stability gaps early
- Pursues Arctic resource access
- Uses energy leverage as economic weapon
- Escalates kinetically when military advantage exists
- Locks into escalation spiral after major commitments

### Crisis Outcomes
- **30-40%**: Peaceful outcomes with diplomatic settlements
- **40-50%**: Managed escalation spirals with de-escalation windows
- **10-20%**: Breakdown to kinetic action or system collapse
- **Avg tension**: 0.45-0.65 (realistic for ongoing competition)
- **Avg stability**: 0.70-0.85 (systems hold but fragile)
## Dependencies

### Core
- gymnasium >= 0.26
- stable-baselines3 >= 2.0
- numpy >= 1.20
- pytorch or tensorflow (backend for SB3)

### Optional
- llama-index (for policy parameterization)
- ollama (for policy PDF processing)
- pandas (analysis scripts)
- matplotlib (visualization)

### Installation
```bash
# Install dependencies
pip install gymnasium stable-baselines3 numpy

# Optional policy parameterization
pip install llama-index-core

# Optional visualization
pip install pandas matplotlib
```

## Citation & References

SADIE v8 integrates:
1. **Policy-Parameterized Learning**: Arctic strategy documents inform agent biases
2. **Multi-Domain Operations**: Realistic military/cyber/economic/information domains
3. **Schelling Points**: Game theory focal points for escalation dynamics
4. **Self-Play Training**: Avoid non-stationary opponent problem with policy pools
5. **Domain-Specific Intelligence**: Incomplete information and sensor-specific observations

### Real-World Applications
- Strategic wargaming for military planners
- Crisis simulation and escalation analysis
- Alliance coordination and cohesion testing
- Economic sanctions effectiveness modeling
- Cyber-kinetic interaction studies

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the sadie_new virtual environment
source /home/dschuster/python/sadie_new/bin/activate

# Validate setup
python validate_v8.py
```

### Training Performance
- If training is slow: Reduce `steps_per_round` in `train_self_play_v8.py`
- If memory issues: Use `device="cuda"` in PPO setup (requires GPU)
- For convergence: Increase `iterations` to 20-24

### Results Interpretation
- Blue reward < 0: Blue is losing strategic position
- Red tension > 0.8: System near breakdown
- Stability < 0.3: Crisis is unmanageable
- Compare against baseline v7 results

## Future Enhancements

Planned for Phase 4+:
- [ ] Multi-year episodic structure (seasons/years)
- [ ] Third-party actors (China, EU, UN mediators)
- [ ] Public opinion / domestic constraints
- [ ] Intelligence uncertainty models
- [ ] Real-time policy updates from new documents
- [ ] Transfer learning across scenarios
- [ ] Evaluation metrics for publication

## Authors & Acknowledgments

SADIE v8 developed for strategic wargaming and policy analysis.
Built with gymnasium, stable-baselines3, and informed by real policy documents.

---

**Status**: v8 Ready for Training  
**Last Updated**: May 2026  
**Recommendation**: Start with `python validate_v8.py` then `python train_self_play_v8.py`

## Code Organization & Cleanup

### Directory Structure (Post-Cleanup, May 10, 2026)

```
sadie_code/
├── ACTIVE (v8)              # Multi-domain, realistic escalation
│   ├── strategic_intel_env_v8.py
│   ├── strategic_state_manager.py
│   ├── multi_domain_operations.py
│   ├── escalation_pathways.py
│   ├── train_self_play_v8.py
│   ├── validate_v8.py
│   └── models/v8/
│
├── SHARED UTILITIES         # Works with both v7 and v8
│   ├── load_parameters.py
│   ├── ingest_policy_docs.py
│   ├── self_play_env.py
│   └── evaluate.py
│
├── LEGACY (archived v7)     # For baseline comparison only
│   ├── LEGACY.md
│   ├── train_self_play_v7.py
│   ├── models_v7/
│   └── tb_logs_v7/
│
└── DOCUMENTATION
    ├── README.md
    ├── CLEANUP_GUIDE.md
    └── SANITY_CHECK_REPORT.md
```

### What Was Cleaned (May 10, 2026)

- ✅ Archived v7 training script to `legacy/train_self_play_v7.py`
- ✅ Archived v7 models (~4.2 MB) to `legacy/models_v7/`
- ✅ Archived v7 TensorBoard logs (~64 MB) to `legacy/tb_logs_v7/`
- ✅ Created `models/v8/` for new training outputs
- ✅ Added deprecation notice to `strategic_intel_env_v7.py`

See [CLEANUP_GUIDE.md](CLEANUP_GUIDE.md) and [legacy/LEGACY.md](legacy/LEGACY.md) for details.
