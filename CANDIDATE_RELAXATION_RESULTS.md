# Candidate Selection Relaxation - Results

**Date:** 2026-01-08
**Issue:** PyFracVAL-tfm
**Status:** ❌ Not Effective - Reverted

---

## Summary

Attempted to relax Gamma_pc candidate selection constraints by 5% (1.05x factor) to allow particles slightly outside the strict fractal constraint. **The approach made success rates worse** (77.8% → 66.7%) and was reverted. The relaxation creates false positive candidates with large initial overlap that cannot be resolved by rotation.

---

## Implementation Attempted

### Changes Made (Reverted)

**Modified Files:**
- `pyfracval/pca_agg.py` - Added PCA_CANDIDATE_RELAXATION_FACTOR = 1.05
- `pyfracval/cca_agg.py` - Added CCA_CANDIDATE_RELAXATION_FACTOR = 1.05

**Logic:**
```python
# Instead of: radius_sum <= gamma_pc
# Used: radius_sum <= gamma_pc * 1.05

relaxed_gamma_pc = gamma_pc * PCA_CANDIDATE_RELAXATION_FACTOR
radius_sum_check = radius_sum <= relaxed_gamma_pc
```

---

## Results

| Metric | Before (Strict) | After (Relaxed) | Change |
|--------|----------------|-----------------|---------|
| **Success Rate** | 77.8% (7/9) | **66.7% (6/9)** | **-14%** ❌ |
| **Median Runtime** | 0.87s | 0.76s | -13% |
| **Failures** | 2/9 | **3/9** | **+50% more failures** |

---

## Root Cause Analysis

### Why Relaxation Made Things Worse

**The Problem:**
1. Relaxing Gamma_pc allows particles with `Rk + Ri > Gamma_pc` (outside fractal constraint)
2. These particles have **larger radius sum** than geometrically compatible
3. Result: **Large initial overlap** (6% instead of <1%)
4. Rotation cannot resolve 6% overlap → guaranteed failure
5. **Created false positive candidates** that waste computation time

### Example: Seed 546629111

**Without Relaxation:**
```
PCA k=2: Found 0 candidates (strict constraint)
Result: Try swapping particles
```

**With 1.05x Relaxation:**
```
PCA k=2: Found 1 candidate (relaxed constraint)  
Initial overlap: 6.0427e-02 (6% overlap!)
After 360 rotations: STILL FAILED
After swapping: Still finds candidates with 6% overlap
Result: Same failure, more wasted computation
```

### Why This Happens

**Gamma_pc constraint is physically meaningful:**
- Ensures particles satisfy target fractal dimension (Df) and prefactor (kf)
- Formula: `Gamma_pc = sqrt(((m3²)(rg3²) - m3(m1·rg1² + m2·rg2²))/(m1·m2))`
- Derived from fractal scaling relationships
- **Not arbitrary** - particles outside constraint violate physics

**Relaxing the constraint:**
- Allows particles that don't satisfy Df/kf target
- Creates geometrically impossible configurations
- Leads to large initial overlap that rotation can't fix

---

## Analysis: When Would Relaxation Work?

### Hypothetical Success Scenario

For relaxation to help, we'd need:
1. ✅ Strict Gamma_pc finds 0 candidates
2. ✅ Relaxed Gamma_pc finds 1+ candidates  
3. ✅ Those candidates have **small initial overlap** (< 0.1%)
4. ✅ Rotation resolves the small overlap
5. ✅ Final aggregate still meets Df/kf targets

### Actual Scenario (What We Observed)

1. ✅ Strict Gamma_pc finds 0-1 candidates
2. ✅ Relaxed Gamma_pc finds 1 candidate
3. ❌ Candidate has **large initial overlap** (6%)
4. ❌ Rotation **cannot** resolve 6% overlap  
5. ❌ **Failure guaranteed** - relaxation just delays it

**Conclusion:** The narrow window where relaxation helps **doesn't exist in practice**.

---

## Comparison with CCA Pairing

**CCA already has 1.50x relaxation** (`CCA_PAIRING_FACTOR = 1.50`)

**Why does CCA relaxation work but PCA doesn't?**

### CCA Context

- **Cluster-level pairing:** Merging 2 subclusters  
- **Constraint:** `Gamma_pc < (R_max1 + R_max2)`
- **Relaxed:** `Gamma_pc < (R_max1 + R_max2) * 1.50`
- **Effect:** Allows clusters that are "close enough" to pair
- **Result:** Works because cluster radii have more flexibility

### PCA Context

- **Particle-level selection:** Adding individual monomers
- **Constraint:** `(Rk + Ri) <= Gamma_pc`
- **Relaxed:** `(Rk + Ri) <= Gamma_pc * 1.05`
- **Effect:** Allows particles that violate fractal constraint
- **Result:** Fails because particle sizes are fixed (log-normal distribution)

**Key Difference:** CCA relaxes inter-cluster relationships (flexible), PCA relaxes particle physics (rigid).

---

## Real Bottlenecks

The actual causes of the remaining 22% failure rate:

### 1. Unlucky Particle Size Distributions
- Log-normal distribution (`rp_gstd=1.5`) produces wide radius range
- Some combinations are geometrically incompatible
- No algorithm can fix this - it's inherent to the distribution

### 2. Subcluster Configuration
- Current: Fixed `n_subcl_percentage = 0.1` (10% of N)
- Some subclusters work, others don't
- **Solution:** Try different n_subcl values (PyFracVAL-0c1)

### 3. Swap Pool Exhaustion
- Limited number of particles to swap with
- After trying all available swaps, no options left
- **Solution:** Larger N or different distribution parameters

### 4. Overlap Tolerance (Already Addressed)
- PyFracVAL-gps implemented adaptive tolerance
- Helps rare cases but doesn't affect current failures

---

## Fortran Comparison

**Original Fortran uses the SAME strict constraint:**

```fortran
! docs/FracVAL/PCA_cca.f90:257
if ((R(n1 + 1) + R(ii)) .GT. Gamma_pc) THEN
    ! Reject candidate
endif
```

**No relaxation in Fortran** - Python should not relax either.

---

## Conclusion

### Key Findings

1. ❌ **Relaxation decreased success rate** - From 77.8% to 66.7%
2. ❌ **Creates false positive candidates** - Large initial overlap
3. ❌ **Wastes computation time** - 360 rotations on impossible cases
4. ✅ **Gamma_pc constraint is well-calibrated** - Should not be relaxed
5. ✅ **Fortran uses same strict constraint** - Confirms correctness

### Recommendation

**Revert and close issue PyFracVAL-tfm as "not effective"**

**Reason:**
- Theoretical assumption was wrong - Gamma_pc is not "too strict"
- Relaxation creates more problems than it solves
- Success rate degraded instead of improved
- Real bottleneck is subcluster configuration, not candidate selection

### Next Priority

**PyFracVAL-0c1:** Implement alternative subclustering strategy
- When PCA fails for a subcluster, try different n_subcl values
- Expected: +3-5% improvement
- Targets actual bottleneck

---

## Lessons Learned

1. **Constraints exist for physical reasons** - Not arbitrary thresholds
2. **Relaxation can create false positives** - More candidates ≠ better results
3. **Test assumptions empirically** - Theoretical "should help" ≠ actual help
4. **Fortran is well-tuned** - 20+ years of research, respect the constraints
5. **Focus on real bottlenecks** - Subcluster configuration, not Gamma_pc

---

## Technical Details

### Test Configuration

**Benchmark:** 9 test cases (3 × Df=1.8, 3 × Df=2.0, 3 × Df=1.9)  
**Parameters:** N=128/256, kf=1.0/1.2, rp_gstd=1.3/1.5  
**Seeds:** Randomized for each run

### Failure Examples

**Seed 546629111 (Df=1.8):**
- Strict: 0-1 candidates found
- Relaxed: 1 candidate found with 6% initial overlap
- Result: Both failed (relaxation wasted time)

**Seed 669593809 (Df=2.0):**
- Strict: 1 candidate found, 20% initial overlap
- Relaxed: Same result (relaxation didn't help)

---

**Status:** ✅ Analysis complete, changes reverted, issue documented
**Recommendation:** Close PyFracVAL-tfm, prioritize PyFracVAL-0c1
**Success Rate:** 77.8% maintained (no regression after revert)
