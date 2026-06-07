# Legacy Code Archive

This directory contains v7 (simplified) environment code that has been superseded by v8 (multi-domain, realistic escalation).

## What's Here

### Code
- **`train_self_play_v7.py`** - Original self-play training script for v7 environment
  - Archived May 10, 2026
  - Replaced by: `../train_self_play_v8.py`
  - Use v7 only for baseline comparison or debugging

### Artifacts
- **`models_v7/`** - Trained v7 models (Blue/Red policy pools)
  - 12 iterations of self-play training
  - Files: blue_0.zip through blue_11.zip, red_0.zip through red_11.zip
  - Size: ~4.2 MB total
  - Last trained: May 9, 2026

- **`tb_logs_v7/`** - TensorBoard logs from v7 training
  - 14 training runs (PPO_1 through PPO_14)
  - Used for monitoring early v7 development
  - Size: ~64 MB

## Why Archived

v7 was a **simplified baseline** (10 actions, 32-dim state, 3 strategic dimensions).

v8 **supersedes v7** with:
- ✅ 24 multi-domain actions vs 10
- ✅ 64-dim state space vs 32
- ✅ 5 strategic dimensions vs 3
- ✅ Realistic escalation pathways (Schelling points, irreversible commitments)
- ✅ Cross-domain spillover effects
- ✅ De-escalation mechanisms with face-saving

## Reference Use Cases

Keep v7 code for:
1. **Baseline comparison** - Quantify v8 improvements
2. **Simplified learning** - v7 trains faster (validation purposes)
3. **Debugging** - v7 is easier to understand and trace

## Migration Path

If you need to compare v7 vs v8:

```bash
# Train v7 (for comparison)
cd legacy
python train_self_play_v7.py

# Train v8 (recommended)
cd ..
python train_self_play_v8.py

# Compare results
python analyze_v7_vs_v8.py  # (when implemented)
```

## Restore v7 to Main

If needed, restore v7 to active development:

```bash
mv legacy/train_self_play_v7.py train_self_play.py
# Also restore strategic_intel_env_v7.py is still in root
```

## File Size Summary

```
legacy/models_v7/      ~4.2 MB (v7 model checkpoints)
legacy/tb_logs_v7/     ~64 MB  (TensorBoard event files)
legacy/train_self_play_v7.py   2.9 KB (training script)

Total: ~68 MB (safe to delete if space needed)
```

---

**Last Archived**: May 10, 2026  
**Reason**: v8 multi-domain system complete and validated  
**Recommendation**: Keep for baseline comparison; don't use for new training
