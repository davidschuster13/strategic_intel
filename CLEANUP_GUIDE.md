# CLEANUP GUIDE - SADIE v8 Migration (May 10, 2026)

## Summary

Cleaned up legacy v7 code to make room for v8 (multi-domain, realistic escalation) development.

## What Was Cleaned

### ✅ Archived to `legacy/` directory

| Item | Size | Status |
|------|------|--------|
| `train_self_play_v7.py` | 2.9 KB | Moved to legacy |
| `models/` (v7 models) | 4.2 MB | Moved to legacy/models_v7 |
| `tb_logs/` (v7 logs) | ~64 MB | Moved to legacy/tb_logs_v7 |

### ✅ Created for v8

| Item | Purpose |
|------|---------|
| `models/v8/` | New directory for v8 training outputs |
| `legacy/LEGACY.md` | Explanation of what was archived |

### ✅ Marked as Deprecated

- `strategic_intel_env_v7.py` - Added deprecation notice at top
  - Keep for reference/baseline comparison
  - Use `strategic_intel_env_v8.py` for new work

## File Structure (After Cleanup)

```
sadie_code/
├── ACTIVE CODE (v8)
│   ├── strategic_intel_env_v8.py          ← Main environment
│   ├── strategic_state_manager.py         ← 5-D state
│   ├── multi_domain_operations.py         ← 24 multi-domain actions
│   ├── escalation_pathways.py             ← Schelling points + de-escalation
│   ├── train_self_play_v8.py              ← Training script (NEW)
│   ├── validate_v8.py                     ← Integration tests (NEW)
│   └── models/v8/                         ← Training outputs (NEW)
│
├── SHARED UTILITIES
│   ├── load_parameters.py                 ← Policy parameterization
│   ├── ingest_policy_docs.py              ← PDF policy extraction
│   ├── self_play_env.py                   ← Self-play wrapper
│   └── evaluate.py                        ← Evaluation tools
│
├── LEGACY (v7 - for reference only)
│   ├── strategic_intel_env_v7.py          ← Deprecated (see header)
│   ├── train_self_play_v7.py              ← Moved from root
│   ├── models_v7/                         ← Old v7 models
│   ├── tb_logs_v7/                        ← Old v7 TensorBoard logs
│   └── LEGACY.md                          ← Archive documentation
│
├── DOCUMENTATION
│   ├── README.md                          ← Updated for v8
│   ├── SANITY_CHECK_REPORT.md
│   ├── CLEANUP_GUIDE.md                   ← This file
│   └── legacy/LEGACY.md
│
└── DATA
    ├── parameterized_output/
    ├── policy_docs/
    └── marl_models/
```

## What to Do Next

### Option 1: Use v8 (Recommended)
```bash
# Validate v8 setup
python validate_v8.py

# Train v8 agents
python train_self_play_v8.py

# Evaluate
python evaluate.py
```

### Option 2: Compare v7 vs v8
```bash
# Train legacy v7 for baseline
cd legacy && python train_self_play_v7.py

# Train v8
cd .. && python train_self_play_v8.py

# Compare results manually
```

### Option 3: Restore v7 (if needed)
```bash
# Restore v7 training script
mv legacy/train_self_play_v7.py train_self_play.py
```

## Disk Space Freed

The following were moved to `legacy/` (can be deleted if space needed):
- `models_v7/`: ~4.2 MB
- `tb_logs_v7/`: ~64 MB
- **Total**: ~68 MB recoverable

```bash
# WARNING: Only delete if sure you don't need baseline comparison
rm -rf legacy/models_v7 legacy/tb_logs_v7
```

## Files Kept in Root (Backward Compatible)

These remain in the root directory for shared use:
- ✅ `strategic_intel_env_v7.py` - Deprecated but available
- ✅ `load_parameters.py` - Used by both v7 and v8
- ✅ `self_play_env.py` - Shared wrapper
- ✅ `evaluate.py` - Works with both versions
- ✅ `ingest_policy_docs.py` - Shared utility

## Migration Checklist

- [x] Create `legacy/` directory
- [x] Move v7 training script: `train_self_play.py` → `legacy/train_self_play_v7.py`
- [x] Move v7 models: `models/` → `legacy/models_v7/`
- [x] Move v7 logs: `tb_logs/` → `legacy/tb_logs_v7/`
- [x] Create `models/v8/` for new training outputs
- [x] Add deprecation notice to `strategic_intel_env_v7.py`
- [x] Document archive in `legacy/LEGACY.md`
- [x] Create this cleanup guide

## Questions?

See `legacy/LEGACY.md` for detailed archive information.
See `README.md` for v8 usage instructions.

---

**Cleanup Date**: May 10, 2026  
**Reason**: v8 system complete and validated  
**v7 Status**: Archived for baseline comparison  
**Recommendation**: Use v8 for all new development
