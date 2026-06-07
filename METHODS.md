# SADIE v8 — Methods (Publication Reference)

## Pipeline

1. **Ingest** (`ingest_policy_docs.py`): PDFs under `policy_docs/blue/` and `policy_docs/red/` → per-document JSON via LLM extraction + validation pass.
2. **Merge** (`load_parameters.py` + `policy_mechanics.py`): Confidence-weighted fusion of all documents per side.
3. **Simulate** (`strategic_intel_env_v8.py`): Multi-domain turn-based environment (60 steps default).
4. **Train** (`train_self_play_v8.py`): Independent Blue/Red PPO with opponent policy pool.
5. **Evaluate** (`evaluate.py`, `policy_ablation.py`): Fixed-seed rollouts and ablations.

## Policy JSON → Simulation

| JSON field | Simulation use |
|------------|----------------|
| `capability_priorities` | Keyword aggregation → assertiveness, restraint, risk_tolerance biases (obs dims 38–40) |
| `escalation_ladder` | Ladder length sets max escalation rank per episode; progress advances on costly actions |
| `red_lines` | Raises `min_tension_red_line`; penalties when tension exceeded and side escalates |
| `decision_heuristics` | Small boost to assertiveness (proxy for doctrinal aggressiveness) |
| `key_objectives` | Stored in merged corpus (future: objective-based rewards) |

## Environment mechanics

- **State (36 dims)**: Military, economic, territorial, alliance, resource subsystems (`strategic_state_manager.py`).
- **Actions (24)**: 6 per domain — kinetic, cyber, economic, information (`multi_domain_operations.py`).
- **Obs (64 dims)**: Partially observed state + tension/stability + policy biases + Schelling/commitment/ladder features.
- **Asymmetric sensing**: Blue kinetic-biased, Red economic-biased (`observe_for_side`).
- **Commitments**: High-intensity / low-reversibility actions call `SchelllingPointManager.make_commitment`.
- **Ablation flag**: `policy_mechanics=False` disables ladder clamp, commitments, red-line penalties (neutral biases).

## Training defaults (v8_policy_r2 — reward patch after v8_policy run)

- Algorithm: PPO (Stable-Baselines3), MLP [256, 256]
- 16 iterations × 50k steps per side per iteration
- Opponent sampled from prior checkpoint pool
- Eval: 10 episodes, seeds 2000–2009, after each iteration
- **Reward patch (May 2026):** Contest-relative strategic scores (military balance centered at 0.5, coalition vs baseline 0.78, Red leverage/energy pressure); wins from `blue_strategic_score` vs `red_strategic_score`, not cumulative step reward or raw NATO strength
- Training output directory: `models/v8_balanced/`

## Recommended experiments

1. Train: `python -m sadie_code.train_self_play_v8`
2. Eval: `python -m sadie_code.evaluate --model-dir sadie_code/models/v8_policy`
3. Ablation: `python -m sadie_code.policy_ablation`
4. Swap or remove `parameterized_output/` and retrain for “no policy” comparison

## Limitations (state in paper)

- LLM extraction error; single 30k-char slice per PDF at ingest
- No attrition, logistics, or domestic politics
- Turn-based simultaneous moves with stylized payoffs (not validated against classified wargames)
