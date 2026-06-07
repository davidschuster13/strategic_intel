# SADIE — Usage Guide

Arctic strategic competition wargame: policy documents → parameters → multi-domain RL (Blue US vs Red adversary).

## Prerequisites

- Python 3.10+ with venv activated, e.g. `source ~/python/sadie_new/bin/activate`
- [Ollama](https://ollama.com) running locally for policy ingestion (`ollama serve`)
- Run all commands from the **parent** of the package:

```bash
cd /home/dschuster/python
source ~/python/sadie_new/bin/activate
```

Imports use `python -m sadie_code.<module>` (do not `cd` into `sadie_code` alone unless `PYTHONPATH` includes the parent).

## Quick start (full pipeline)

### 1. Add policy PDFs

```text
sadie_code/policy_docs/blue/   # US / allied documents
sadie_code/policy_docs/red/    # adversary documents
```

### 2. Ingest → JSON parameters

```bash
python -m sadie_code.ingest_policy_docs          # both sides
python -m sadie_code.ingest_policy_docs --side blue
python -m sadie_code.ingest_policy_docs --side red
```

Output:

```text
sadie_code/parameterized_output/blue/all_parameterized_policies.json
sadie_code/parameterized_output/red/all_parameterized_policies.json
```

### 3. Train (self-play PPO, no COA diversity guard)

Default run writes to `sadie_code/models/v8_contest/` (20 rounds × 60k steps per side):

```bash
python -m sadie_code.train_self_play_v8
```

Custom directory or shorter smoke test:

```bash
python -m sadie_code.train_self_play_v8 --model-dir sadie_code/models/v8_contest --iterations 4 --steps 20000
```

Checkpoints: `blue_00.zip` … `blue_final.zip`, `red_00.zip` … `red_final.zip`, plus `training_stats.json`.

### 4. Evaluate (raw policy by default)

```bash
python -m sadie_code.evaluate --model-dir sadie_code/models/v8_contest --games 50
```

Report saved to `sadie_code/models/v8_contest/eval_report.json` when `--model-dir` is set.

Optional **eval-only** COA diversity shim (not used in training):

```bash
python -m sadie_code.evaluate --model-dir sadie_code/models/v8_contest --games 50 --diversity-guard
```

## What the numbers mean

| Metric | Use for |
|--------|---------|
| Score margin (Blue − Red) | Primary contest outcome; negative = Red ahead |
| Win / draw / mutual loss | Derived from final **strategic scores**, not RL reward |
| Tension in band (0.45–0.72) | Share of episodes in a “steady competition” band |
| COA diversity / collapse | Whether one action dominates (>80%) |
| Clamps/ep | How often doctrine ladder overrides the policy |
| RL reward (episode avg) | Training signal only; can disagree with who “won” |

**Default eval = learned policy.** Use `--diversity-guard` only when demonstrating a commander UI that nudges repeated COAs.

## Model directories (examples)

| Directory | Notes |
|-----------|--------|
| `models/v8_contest` | Current default training target (margin + terminal rewards) |
| `models/v8_rebalance2` | Prior tuned run (good COA spread without guard) |
| `models/v8_balanced` | Older default in `load_parameters.py` if no `--model-dir` |

Always pass `--model-dir` to `evaluate` so checkpoints and reports match.

## Other commands

```bash
# Policy on vs off (ablation)
python -m sadie_code.policy_ablation --model-dir sadie_code/models/v8_contest

# Environment smoke test
python -m sadie_code.validate_v8
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: sadie_code` | Run from `/home/dschuster/python` or set `PYTHONPATH` |
| Ollama connection errors | Start `ollama serve`; confirm model in ingest script |
| Observation shape mismatch after env change | Retrain; old `.zip` checkpoints are incompatible |
| 100% Red wins | Check **no-guard** eval; tune rewards / train longer (`v8_contest`) |
| Report in wrong folder | Use `--model-dir` (or `--output` explicitly) |

## Methods detail

See [METHODS.md](METHODS.md) for publication-style mechanics (state space, policy JSON mapping, limitations).
