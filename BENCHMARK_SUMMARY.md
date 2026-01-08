# Benchmark Results Summary

**Date:** 2026-01-08
**Test:** Initial benchmark of stable parameter cases

---

## Results

### Overall Performance
- **Category:** Stable (Df ∈ [1.8, 2.0])
- **Expected:** 95%+ success rate
- **Actual:** **33.3%** success rate (3/9 trials)
- **Status:** ❌ SEVERE UNDERPERFORMANCE

### Detailed Breakdown

| Configuration | Df | kf | N | Trials | Successes | Failure Rate |
|--------------|-----|-----|---|--------|-----------|--------------|
| Original paper | 1.8 | 1.0 | 128 | 3 | 2 | 33.3% |
| Default config | 2.0 | 1.0 | 128 | 3 | 0 | **100%** |
| Moderate polydisperse | 1.9 | 1.2 | 256 | 3 | 1 | 66.7% |

---

## Root Cause Identified

**Critical Bug in Particle Swapping Mechanism**

### The Problem
`_search_and_select_candidate()` in `pca_agg.py` doesn't swap particles when:
1. Candidates exist (len > 0)
2. BUT all candidates fail overlap check after 360 rotations

This creates an infinite loop where the same doomed candidate is retried 12 times.

### Example Failure (Seed: 2104002276)
```
PCA k=2: Found 1 candidate
  - Candidate 0: Initial overlap = 20.9% (impossible to resolve with tol_ov=1e-6)
  - Failed after 360 rotations

Retry attempt 1: Same particle k=2 → Same candidate → FAIL
Retry attempt 2: Same particle k=2 → Same candidate → FAIL
...
Retry attempt 12: Same particle k=2 → Same candidate → FAIL

ERROR: PCA failed for subcluster 4
```

### Why Swapping Doesn't Occur

**Current Code (Line 320):**
```python
if len(candidates) > 0:
    return (..., candidates)  # Returns immediately - NO SWAP!
else:
    # Swap only triggered when NO candidates found
```

**Fortran Code (Correct Behavior):**
```fortran
if (Cov_max .GT. tol_ov) then
   list = list*0  ! Clear candidates → forces swap
end if
```

---

## Solution

### Fix Type: Add `force_swap` Parameter

**Modification:** `pca_agg.py:281` - `_search_and_select_candidate`

```python
def _search_and_select_candidate(
    self, k, considered_indices, force_swap=False  # NEW
):
    if len(candidates) > 0 and not force_swap:  # NEW check
        return (..., candidates)
    else:
        # Swap particle k with another...
```

**Caller update:** After all candidates fail:
```python
# Retry with force_swap=True
search_result = self._search_and_select_candidate(
    k, considered_indices, force_swap=True
)
```

---

## Expected Impact

### Success Rate Improvements
| Category | Before | After Fix | Improvement |
|----------|--------|-----------|-------------|
| Stable (Df=1.8-2.0) | 33% | **90-95%** | +3x |
| Default (Df=2.0) | 0% | **85-90%** | +∞ |
| Moderate polydisperse | 33% | **80-85%** | +2.5x |

### Why This Fixes It
- Allows algorithm to try different particles when geometry is constrained
- Escapes "local minima" in particle ordering
- Matches intended Fortran behavior
- No algorithm changes (same aggregate properties guaranteed)

---

## Implementation Steps

1. ✅ **Identify root cause** (COMPLETED)
2. ✅ **Document in STICKING_ANALYSIS.md** (COMPLETED)
3. ✅ **Create ACTION_PLAN.md** (COMPLETED)
4. ⏳ **Implement force_swap fix** (READY)
5. ⏳ **Test with failed seeds** (READY)
6. ⏳ **Re-run benchmarks** (READY)
7. ⏳ **Validate 90%+ success rate** (PENDING)

---

## Files Created/Updated

### New Files
- `BENCHMARK_SUMMARY.md` (this file)
- `ACTION_PLAN.md` (detailed implementation guide)
- `benchmark_results/stable_summary.json` (test results)
- `/tmp/test_single.py` (debug test)
- `/tmp/test_failed_seed.py` (failure investigation)

### Updated Files
- `STICKING_ANALYSIS.md` (added "CRITICAL UPDATE" section)

---

## Next Steps

### Option 1: Implement Fix Now (Recommended)
Follow `ACTION_PLAN.md`:
1. Modify `_search_and_select_candidate` (15 min)
2. Test with seed 2104002276 (10 min)
3. Re-run benchmarks (30 min)
4. Validate results (15 min)

**Total time:** ~1-2 hours
**Expected result:** 90%+ success for stable cases

### Option 2: Review First
1. Read `ACTION_PLAN.md` for implementation details
2. Read `STICKING_ANALYSIS.md` (section "CRITICAL UPDATE") for full analysis
3. Decide on implementation approach

---

## Key Insights

### What Worked
✅ Benchmark infrastructure revealed the bug immediately
✅ DEBUG logging pinpointed exact failure mode
✅ Fortran comparison showed intended behavior

### What Failed
❌ Initial analysis missed swap mechanism bug
❌ Theoretical predictions assumed working swap logic
❌ Code "worked" for lucky seeds, hiding the bug

### Lessons Learned
- Always run benchmarks before and after changes
- Stochastic failures (67% not 100%) are hardest to debug
- Compare against reference implementation systematically

---

## Confidence Level

**Root Cause Analysis:** 🟢 VERY HIGH
- Reproduced failure deterministically
- Identified exact code path causing issue
- Confirmed Fortran does it differently

**Proposed Fix:** 🟢 HIGH
- Simple parameter addition
- Clear control flow
- Matches Fortran logic

**Expected Improvement:** 🟡 MEDIUM-HIGH
- Based on geometric analysis: should hit 90%+
- Will be validated by post-fix benchmarks
- May discover additional edge cases

---

**Recommendation:** Proceed with fix implementation per `ACTION_PLAN.md`
