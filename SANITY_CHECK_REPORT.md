# SADIE Wargame Codebase Sanity Check Report

**Date**: 2026-05-09  
**Files Reviewed**: `ingest_policy_docs.py`, `load_parameters.py`, `strategic_intel_env_v7.py`, `train_self_play.py`, `evaluate.py`, `self_play_env.py`

---

## Executive Summary

The codebase **does implement the intended architecture** (policy parameterization → RL wargaming), but has **critical bugs and validation gaps** that prevent operational and publication readiness.

### Status
- ✅ **Core Concept**: Sound (policy-to-RL pipeline)
- ⚠️ **Implementation**: Multiple critical issues
- ❌ **Operational Ready**: No (requires validation loops, explainability)
- ❌ **Publication Ready**: No (missing ablations, baselines, significance testing)

---

## Critical Issues (Must Fix)

### 1. **Observation Space Breaks Multi-Agent Learning** 🔴 CRITICAL
**File**: `strategic_intel_env_v7.py`, line ~315  
**Issue**: 
```python
def _get_obs(self):
    base = np.zeros(32, dtype=np.float32)
    # ... populate base with state ...
    return {"Blue": base.copy(), "Red": base.copy()}  # <-- IDENTICAL OBSERVATIONS
```
Both agents see **identical state**, violating multi-agent RL assumption. Agents cannot learn asymmetric strategies (e.g., Blue defensive vs. Red offensive).

**Impact**: 
- Agents will converge to trivial strategies (both play the same action)
- Policy parameters are not actually being differentiated in agent behavior
- Results are invalid for a two-player wargame

**Fix**:
```python
def _get_obs(self):
    base = np.zeros(32, dtype=np.float32)
    base[0] = self.tension
    base[1] = self.blue_awareness
    base[2] = self.red_influence
    base[3] = self.stability
    base[4] = self.current_step / self.max_steps
    base[5] = ACTION_ESCALATION[self.last_blue_action]
    base[6] = ACTION_ESCALATION[self.last_red_action]
    base[7:10] = [self.blue_bias["assertiveness"], ...]  # Blue's own bias
    base[10:13] = [self.red_bias["assertiveness"], ...]    # Red's own bias
    
    blue_obs = base.copy()
    blue_obs[13:] = 0  # Red doesn't see Blue's introspection
    
    red_obs = base.copy()
    red_obs[7:10] = 0  # Blue doesn't see Red's introspection
    
    return {"Blue": blue_obs, "Red": red_obs}
```

---

### 2. **LLM Parameter Extraction is Unvalidated** 🔴 CRITICAL
**File**: `ingest_policy_docs.py`, line ~79  
**Issue**:
- Document truncated to 50K characters (Arctic Strategy policy is 100+ pages)
- No validation that extracted JSON actually reflects document content
- Confidence scores are self-reported by LLM, not grounded
- Single-pass extraction with no error recovery

**Example Problem**:
```python
combined = combined[:50_000]  # Loses second half of long documents
# LLM may hallucinate parameters not in the document
```

**Impact**:
- Parameters may not represent actual policy
- "Ground truth" policy priors are corrupted
- Cannot justify agent behavior as policy-driven

**Fix**:
```python
# 1. Implement validation pass: re-query LLM to confirm extracted parameters appear in text
# 2. Chunk longer documents, extract separately, aggregate via voting
# 3. Add extractive quotes to every parameter (force quotation from original)
# 4. Flag low-confidence extractions for manual review
```

---

### 3. **Self-Play Training Has Non-Stationary Learning** 🔴 CRITICAL
**File**: `train_self_play.py`, lines 18-29  
**Issue**:
```python
for i in range(iterations):
    blue_env = SelfPlayEnv(base_env, "Blue", red)  # Train Blue vs fixed Red
    blue.set_env(blue_env)
    blue.learn(total_timesteps=steps_per_round)
    # Then train Red vs fixed Blue...
```

Alternating training means:
- Blue learns against an increasingly strong Red
- Then Red trains against a suddenly different Blue
- Non-stationary opponent leads to unstable convergence
- No guarantee of equilibrium

**Impact**:
- Results may not reflect genuine strategic equilibrium
- Agents may just be exploiting each other's weaknesses, not learning robust policies
- Cannot claim the learned policies are "optimal"

**Fix**: Use proper self-play framework (e.g., RLlib's built-in self-play or maintain a policy population)

---

### 4. **Reward Function is Unjustified and Ad-Hoc** 🔴 CRITICAL
**File**: `strategic_intel_env_v7.py`, lines ~275-330  
**Issue**: 17+ weighted reward terms with magic numbers:
```python
blue_r = 5.5 * (1 - self.tension) + 3.5 * self.blue_awareness + 2.5 * self.stability
red_r = 4.5 * self.tension + 3.5 * self.red_influence
# ... 10+ more terms with unexplained weights
```

**Impact**:
- Impossible to debug why agents behave as they do
- Weights look arbitrary; may not reflect actual policy objectives
- Sensitive to small weight changes (needed for publication ablation study)
- No connection to policy documents

**Fix**:
1. Extract explicit objectives from parameterized policies (e.g., "Red maximizes influence")
2. Justify each reward term with reference to extracted policy
3. Conduct ablation: disable each term, measure performance impact
4. Publish sensitivity analysis

---

## High-Priority Issues

### 5. **Parameter Extraction Mapping is Ad-Hoc** 🟠 HIGH
**File**: `strategic_intel_env_v7.py`, lines ~42-54  
**Issue**: Keywords mapped to assertiveness/restraint/risk via hardcoded keyword lists:
```python
militarized_keywords = ("military", "force", "projection", "strike", ...)
militarized = sum(v for k, v in priorities.items() 
                  if any(token in k for token in militarized_keywords))
assertiveness = np.clip(0.2 + 0.35 * militarized + 0.15 * hybrid + ..., 0.0, 1.0)
```

**Issues**:
- Brittle substring matching (e.g., "military" ≠ "military doctrine")
- Weights (0.35, 0.15, etc.) unjustified
- No validation against actual policy doctrine
- Different papers may use different terminology

**Fix**: 
- Use structured policy extraction template (objectives, red-lines, constraints as separate JSON fields)
- Require human expert to validate extracted vectors before use
- Document how each extracted parameter maps to game mechanic

---

### 6. **No Baseline Comparisons** 🟠 HIGH
**File**: All files  
**Issue**: No evaluation against random policy, rule-based baseline, or other RL algorithms

**Impact**:
- Cannot claim learned policies are actually better than simple heuristics
- No proof that policy parameters influence agent behavior
- Publication will be rejected without baselines

**Fix**:
```python
# Add evaluate.py comparison:
# - Random action baseline
# - Rule-based (if tension > 0.7, de-escalate)
# - Single-agent RL trained without policy parameters
# - Mixed: trained with policy but no parameterization
```

---

### 7. **No Uncertainty or Confidence Quantification** 🟠 HIGH
**Files**: All  
**Issue**: Point estimates with no confidence intervals or uncertainty bands

**Example**:
```python
print(f"Blue Win Rate : {blue_wins/n_games*100:.1f}%")  # Just a point, no CI
```

**Impact**:
- Cannot determine if observed win rates are statistically significant
- Differences between models may be noise
- Publication reviewers will reject without significance testing

**Fix**:
```python
from scipy import stats
ci = stats.binomtest(blue_wins, n_games, 0.5).proportion_ci()
print(f"Blue Win Rate: {blue_wins/n_games:.1%} (95% CI: {ci})")
```

---

## Medium-Priority Issues

### 8. **No Per-Game Logging** 🟡 MEDIUM
**File**: `evaluate.py`  
**Issue**: Aggregates statistics; cannot debug individual anomalies

**Fix**: Add per-game CSV output with columns: game_id, blue_action_seq, red_action_seq, final_tension, outcome

### 9. **Self-Play Wrapper Has Action Indexing Bug** 🟡 MEDIUM
**File**: `self_play_env.py`, line ~33  
**Issue**:
```python
def step(self, action):
    # ...
    done = terminated if isinstance(terminated, bool) else terminated[self.agent_name]
    return obs_dict[self.agent_name], rewards[self.agent_name], done, done, info  # <-- done twice (deprecated)
```

Gymnasium API returns `(obs, reward, terminated, truncated, info)`, not `(obs, reward, done, done, info)`.

**Fix**:
```python
return obs_dict[self.agent_name], rewards[self.agent_name], done, False, info
```

### 10. **No Curriculum Learning Documentation** 🟡 MEDIUM
**File**: `strategic_intel_env_v7.py`, line ~102  
**Issue**: Curriculum learning (low/baseline/high tension) has no justification or ablation

**Fix**: Document why these regimes were chosen; ablate to show impact on convergence

### 11. **Hardcoded Model Paths and No Versioning** 🟡 MEDIUM
**Files**: `train_self_play.py`, `evaluate.py`  
**Issue**: No experiment ID, timestamp, or version tracking

**Fix**: Add experiment config file (YAML/JSON) tracking:
- Ollama model version
- Document sources (Blue/Red files with hash)
- Training hyperparameters
- Random seed
- RL algorithm and parameters

---

## Operational Validity Assessment

### For Real-World Operational Use ❌
**Current state: NOT READY**

**Required for operational deployment**:
1. ✅ Policy expert review of extracted parameters (manual validation)
2. ❌ **MISSING**: Uncertainty quantification (e.g., "Red assertiveness = 0.65 ± 0.15")
3. ❌ **MISSING**: Explainability module (why did the agent choose this action?)
4. ❌ **MISSING**: Audit trail (decisions logged with parameter values)
5. ❌ **MISSING**: Sensitivity analysis (how much does outcome change if parameters shift by ±10%?)
6. ⚠️ **INCOMPLETE**: Single LLM model (Qwen 2.5); no robustness to model selection

**Recommendation**: 
- Do NOT use in operational intelligence workflows without expert validation loops
- Add human-in-the-loop (expert must approve parameter vectors before training)
- Add decision logging/tracing for every agent action

---

## Publication Readiness Assessment

### Current State: NOT READY ❌

**Required for publication**:

| Item | Status | Priority |
|------|--------|----------|
| Baseline comparisons | ❌ MISSING | P0 |
| Ablation studies | ❌ MISSING | P0 |
| Statistical significance testing | ❌ MISSING | P0 |
| Multi-agent assumption validation | ❌ FAILS (identical obs) | P0 |
| Parameter validation analysis | ❌ MISSING | P1 |
| Hyperparameter sensitivity | ❌ MISSING | P1 |
| Reproducibility (seeds, model versions) | ⚠️ INCOMPLETE | P1 |
| Comparison to prior work (game theory, wargaming) | ❌ MISSING | P1 |
| Per-game trajectories | ❌ MISSING | P2 |
| Code release + documentation | ❌ NOT READY | P2 |

**Estimated effort to publication-ready**:
- Fix critical issues (1-2): **1-2 weeks**
- Add baselines & ablations (6): **2-3 weeks**
- Statistical analysis & writing: **2-3 weeks**
- **Total: 5-8 weeks** (if fixes go smoothly)

---

## Recommended Fix Priority

### Phase 1: Critical Correctness (Do This First) 🔴
1. **Fix observation space** (Issue #1) - agents need asymmetric observations
2. **Validate LLM parameters** (Issue #2) - extract with confidence, validation pass
3. **Fix reward function** (Issue #4) - ground in policy, ablate components
4. **Use proper self-play** (Issue #3) - switch to RLlib or policy pool

**Estimated time**: 5-7 days  
**Test**: Can Blue and Red learn distinct strategies? Do agent actions differ by policy?

### Phase 2: Publication-Grade Analysis 🟠
5. Add baselines (random, rule-based, no-policy RL)
6. Ablation study: disable reward terms one-by-one
7. Statistical significance testing (binomial CI on win rates)
8. Sensitivity analysis: perturb parameters ±10%, rerun

**Estimated time**: 10-14 days

### Phase 3: Operational Readiness 🟡
9. Add parameter confidence intervals
10. Add decision explainability (action→reward breakdown)
11. Create human validation UI for extracted parameters
12. Add reproducibility config (seed, model version tracking)

**Estimated time**: 5-7 days

---

## Detailed Fix Examples

### Fix #1: Asymmetric Observations

Replace `_get_obs()` in `strategic_intel_env_v7.py`:

```python
def _get_obs(self):
    """Return asymmetric observations; each agent sees only own bias."""
    base = np.zeros(32, dtype=np.float32)
    
    # Shared environment state
    base[0] = self.tension
    base[1] = self.blue_awareness
    base[2] = self.red_influence
    base[3] = self.stability
    base[4] = self.current_step / self.max_steps
    base[5] = ACTION_ESCALATION[self.last_blue_action]
    base[6] = ACTION_ESCALATION[self.last_red_action]
    
    # Blue observation: sees own bias + history, not Red's internal state
    blue_obs = base.copy()
    blue_obs[7:10] = [self.blue_bias["assertiveness"], 
                      self.blue_bias["restraint"], 
                      self.blue_bias["risk_tolerance"]]
    blue_obs[10:13] = 0  # Blue doesn't see Red's internal bias
    
    # Red observation: sees own bias + history, not Blue's internal state
    red_obs = base.copy()
    red_obs[10:13] = [self.red_bias["assertiveness"], 
                      self.red_bias["restraint"], 
                      self.red_bias["risk_tolerance"]]
    red_obs[7:10] = 0  # Red doesn't see Blue's internal bias
    
    # Remaining slots for future expansion
    blue_obs[13:] = 0
    red_obs[13:] = 0
    
    return {"Blue": blue_obs, "Red": red_obs}
```

---

### Fix #2: Parameter Extraction with Validation

Add to `ingest_policy_docs.py`:

```python
def _validate_extraction(llm, original_text: str, extracted_params: dict, max_retries=2):
    """Validate that extracted parameters appear in source text."""
    objectives = extracted_params.get("key_objectives", [])
    
    for attempt in range(max_retries):
        validation_prompt = f"""
        Given this original policy text, do these extracted objectives appear?
        
        Extracted: {objectives}
        
        Original text (first 5000 chars): {original_text[:5000]}
        
        For each objective, respond:
        - FOUND: appears in text
        - INFERRED: implied but not explicit
        - HALLUCINATED: not supported by text
        """
        
        try:
            result = llm.complete(validation_prompt)
            if "FOUND" in result.text or "INFERRED" in result.text:
                return True, result.text
        except Exception as e:
            print(f"Validation attempt {attempt+1} failed: {e}")
    
    return False, "Validation failed"
```

---

### Fix #3: Baseline Comparison

Add to `evaluate.py`:

```python
def evaluate_all_baselines(n_games=50):
    """Compare learned RL against baselines."""
    env = StrategicWargameEnv()
    
    # Random baseline
    random_results = _eval_random_policy(env, n_games)
    
    # Rule-based baseline
    rule_results = _eval_rule_policy(env, n_games)
    
    # Learned RL
    rl_results = evaluate(n_games)
    
    print("\n=== COMPARISON ===")
    print(f"Random baseline  : {random_results['win_rate']:.1%} blue wins")
    print(f"Rule-based       : {rule_results['win_rate']:.1%} blue wins")
    print(f"Learned RL       : {rl_results['win_rate']:.1%} blue wins")
    
    # Statistical test
    from scipy import stats
    z_stat, p_val = stats.ttest_ind(rl_results['win_outcomes'], 
                                     random_results['win_outcomes'])
    print(f"Learned vs Random: p={p_val:.4f} {'✓ significant' if p_val < 0.05 else '✗ not significant'}")
```

---

## Recommendations Summary

| # | Issue | Fix | Time | Impact |
|---|-------|-----|------|--------|
| 1 | Identical observations | Asymmetric obs by agent | 2 hrs | CRITICAL |
| 2 | Unvalidated parameters | Add validation pass + expert review | 3 days | CRITICAL |
| 3 | Non-stationary self-play | Switch to policy pool or RLlib | 3 days | CRITICAL |
| 4 | Ad-hoc reward function | Ground in policy + ablate | 5 days | CRITICAL |
| 5 | No baselines | Add 3 baselines (random, rule, no-param) | 5 days | PUBLICATION |
| 6 | No significance testing | Add binomial CI, t-tests | 1 day | PUBLICATION |
| 7 | No parameter validation study | Extract from >2 docs, compare | 3 days | PUBLICATION |
| 8 | No logging | Add CSV per-game output | 1 day | DEBUG |
| 9 | Deprecation warnings | Fix Gymnasium API | 2 hrs | CODE QUALITY |
| 10 | No reproducibility | Add config file + seeding | 1 day | CODE QUALITY |

---

## Conclusion

✅ **Concept is Sound**: Policy parameterization + self-play RL is valid  
⚠️ **Implementation Has Critical Flaws**: Identical observations, unvalidated parameters, non-stationary training  
❌ **Not Operational Ready**: No validation loops, no explainability  
❌ **Not Publication Ready**: No baselines, ablations, significance testing  

**Next Step**: Prioritize fixing the 4 critical issues (observation space, parameter validation, reward grounding, self-play convergence). Once fixed, conduct baseline comparisons and ablation studies before proceeding to publication or operational deployment.

