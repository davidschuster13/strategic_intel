# SADIE: Strategic Action Decision and Inferencing Engine

A multi-domain, policy-parameterized wargame environment for studying strategic competition between US/NATO (Blue) and Adversary (Red) forces with realistic escalation dynamics.

## Blue vs Red simulation

Each episode is a **60-step turn-based contest** between two PPO policies:

| Side | Role | Typical objectives in sim |
|------|------|---------------------------|
| **Blue** | US / NATO | Military balance, NATO cohesion, limiting Red resource leverage, economic pressure on Red |
| **Red** | Adversary | Military pressure, Arctic/resource leverage, economic pressure on Blue |

Both sides choose from **24 multi-domain COAs** (kinetic, cyber, economic, information) each step. Policy PDFs are ingested into JSON parameters that constrain escalation ladders, red lines, and doctrine biases. Outcomes are scored by **strategic score margin**, not RL reward alone — suited for commander / decision-support reporting rather than binary win-rate tuning.

**Recommended checkpoint:** `models/v8_contest/` (contested draws, multi-domain COAs, no eval diversity guard required).

## Usage

### Prerequisites

```bash
cd /home/dschuster/python
source ~/python/sadie_new/bin/activate   # or your venv
```

Run modules as **`python -m sadie_code.<module>`** from the parent of `sadie_code/` (not from inside the package unless `PYTHONPATH` is set). For policy ingestion, start Ollama: `ollama serve`.

### 1. Policy documents → parameters

Place PDFs under `policy_docs/blue/` and `policy_docs/red/`, then ingest:

```bash
python -m sadie_code.ingest_policy_docs              # both sides
python -m sadie_code.ingest_policy_docs --side blue
python -m sadie_code.ingest_policy_docs --side red
```

Outputs merge into `parameterized_output/blue/` and `parameterized_output/red/` (loaded automatically by the environment).

### 2. Train Blue and Red (self-play PPO)

Default: **20 rounds × 60,000 steps per side** → `models/v8_contest/`:

```bash
python -m sadie_code.train_self_play_v8
```

Custom run or smoke test:

```bash
python -m sadie_code.train_self_play_v8 \
  --model-dir sadie_code/models/v8_contest \
  --iterations 4 --steps 20000
```

Produces `blue_00.zip` … `blue_final.zip`, `red_00.zip` … `red_final.zip`, and `training_stats.json`. Training eval uses **raw policy** (no COA diversity guard).

**Note:** Model `.zip` files are **local only** (see `.gitignore`). Clone the repo for code; retrain or copy checkpoints between machines.

### 3. Evaluate the matchup

```bash
python -m sadie_code.evaluate --model-dir sadie_code/models/v8_contest --games 50
```

Saves `eval_report.json` under the same `--model-dir`. **Publication exports** (CSV/JSON + figures for LaTeX) go to `<model-dir>/publication/` automatically — see [USAGE.md](USAGE.md#4-evaluate-raw-policy-by-default).

Optional eval-only COA nudge for DSS demos:

```bash
python -m sadie_code.evaluate --model-dir sadie_code/models/v8_contest --games 50 --diversity-guard
```

### GUI (user-friendly runner)

If you want a clickable workflow (upload policy docs → ingest → train → eval), run the Streamlit GUI:

```bash
pip install streamlit
streamlit run sadie_code/gui_app.py
```

### 4. Smoke test / ablation

```bash
python -m sadie_code.validate_v8
python -m sadie_code.policy_ablation --model-dir sadie_code/models/v8_contest
```

### Reading eval output

| Metric | Meaning |
|--------|---------|
| **Score margin (Blue − Red)** | Primary outcome; negative = Red ahead |
| **Win / draw / mutual loss** | From final strategic scores (draw if \|margin\| < 0.12) |
| **Tension in band (0.45–0.72)** | Share of episodes in “steady competition” |
| **COA diversity / collapse** | Unique actions; collapse = one action > 80% |
| **Clamps/ep** | Doctrine ladder overriding chosen COAs |
| **RL reward** | Training signal only — can disagree with who “won” |

Use **default eval (no guard)** to see what the network learned; use **`--diversity-guard`** only for commander UI “what-if” COA variety.

More detail: [USAGE.md](USAGE.md) · mechanics: [METHODS.md](METHODS.md)

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
- **`models/`** - Trained PPO checkpoints (local; `*.zip` gitignored)
- **`models/v8_contest/`** - Current default training/eval target
- **`evaluate.py`** / **`eval_utils.py`** - Commander/DSS metrics and rollouts
- **`policy_mechanics.py`** - Doctrine ladders, red lines, action clamps
- **`SANITY_CHECK_REPORT.md`** - Comprehensive audit and fixes documentation

### Legacy v7 (baseline only)

See [legacy/LEGACY.md](legacy/LEGACY.md) for archived v7 training and models.

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
Edit `train_self_play_v8.py` or use CLI flags:
- `iterations`: Self-play rounds (default: **20**)
- `steps_per_round`: Timesteps per side per round (default: **60,000**)
- `--model-dir`: Output directory (default: `models/v8_contest/`)
- PPO: `ent_coef=0.05`, linear LR 3e-4 → 5e-5, opponent sampled from prior checkpoint pool

### Curriculum Learning
v8 automatically starts episodes in three regimes:
- 25% low tension (0.18-0.32) - peaceful scenarios
- 50% baseline (0.30-0.45) - normal operations
- 25% high tension (0.52-0.70) - crisis starts

## Expected Behavior (v8_contest benchmark)

With the current reward patch, a well-trained pair typically shows:

- **Contested outcomes** — many episodes classify as **draws** (|Blue − Red strategic score| < 0.12), not 100% one-side wins
- **Multi-domain COAs** — neither side collapsed to a single action (>80%); Blue often mixes info/economic/cyber; Red spreads across domains
- **Score margin** near **−0.10** (slight Red edge) with low variance is normal for self-play equilibrium
- **Tension** mean ~0.73, final ~0.82 (cap); **16–50%** of episodes in the 0.45–0.72 “competition” band depending on run
- **Coalition drift** — Blue coalition strength often rises over the episode (+0.04 typical)
- **Doctrine clamps** near **0/ep** when policy and learned COAs align

Treat **RL episode reward** as a training diagnostic; brief stakeholders on **margin, trajectories, and COA mix**.
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
cd /home/dschuster/python
source ~/python/sadie_new/bin/activate
python -m sadie_code.validate_v8
```

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sadie_code` | Run from parent dir or set `PYTHONPATH` |
| Observation shape mismatch | Retrain after env changes; old `.zip` incompatible |
| Eval report in wrong folder | Pass `--model-dir` to `evaluate` |
| Ollama errors on ingest | Run `ollama serve` locally |

### Training Performance
- Slow runs: `--iterations 4 --steps 20000` for smoke tests
- Convergence: full default (20 × 60k) or increase iterations
- GPU: change `device="cpu"` in `train_self_play_v8.py` if CUDA available

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

**Status**: v8 production-ready (`v8_contest` checkpoint)  
**Last Updated**: May 2026  
**Quick path**: `python -m sadie_code.validate_v8` → `python -m sadie_code.train_self_play_v8` → `python -m sadie_code.evaluate --model-dir sadie_code/models/v8_contest --games 50`

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
│   └── models/v8_contest/   # default checkpoints (local .zip)
│
├── SHARED UTILITIES
│   ├── load_parameters.py
│   ├── ingest_policy_docs.py
│   ├── policy_mechanics.py
│   ├── eval_utils.py
│   ├── self_play_env.py
│   ├── evaluate.py
│   └── policy_ablation.py
│
├── LEGACY (archived v7)     # For baseline comparison only
│   ├── LEGACY.md
│   ├── train_self_play_v7.py
│   ├── models_v7/
│   └── tb_logs_v7/
│
└── DOCUMENTATION
    ├── README.md
    ├── USAGE.md
    ├── METHODS.md
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
