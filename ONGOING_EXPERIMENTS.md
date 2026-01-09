# Ongoing PyFracVAL Experiments

**Started:** 2026-01-09
**Status:** Running

## Active Benchmarks

### 1. Full Parameter Sweep
- **Status:** Running (task ID: b1d7ef9)
- **Grid:** 11 Df × 16 kf × 3 sigma = 528 configurations
- **Trials:** 5 per configuration = 2640 total trials
- **Estimated time:** ~44 minutes
- **Output:** `benchmark_results/parameter_sweep/sweep_results.json`

**Purpose:** Fine-grained mapping of parameter space to identify exact boundaries between feasible and infeasible regions.

**Current Progress:** Check with:
```bash
tail -5 /tmp/claude/-home-mar-Development-PyFracVAL/tasks/b1d7ef9.output | grep Testing
```

### 2. Large Aggregate Scaling Test
- **Status:** Running (task ID: b41ef78)
- **Configurations:** 5 optimal combinations × 4 N values = 20 tests
- **N values:** 128, 256, 512, 1024
- **Trials:** 5 per configuration = 100 total trials
- **Estimated time:** ~20-40 minutes (depends on N)
- **Output:** `benchmark_results/large_aggregates/large_agg_results.json`

**Purpose:** Test whether optimal N=128 parameters (Df, kf, sigma) scale to larger aggregates.

**Optimal combinations being tested:**
| Df  | kf  | sigma | Expected @N=128 |
|-----|-----|-------|-----------------|
| 1.6 | 1.2 | 1.3   | 100%            |
| 1.8 | 1.0 | 1.3   | 100%            |
| 2.0 | 1.0 | 1.3   | 100%            |
| 2.2 | 0.8 | 1.3   | 100%            |
| 2.4 | 0.6 | 1.3   | 100%            |

**Current Progress:** Check with:
```bash
tail -10 /tmp/claude/-home-mar-Development-PyFracVAL/tasks/b41ef78.output
```

### 3. Sigma Failure Mode Investigation
- **Status:** Running (task ID: bf597b0)
- **Sigma values:** 1.3, 1.5, 1.8, 2.0, 2.5
- **Fixed params:** Df=2.0, kf=1.0 (known good combination)
- **Trials:** 5 seeds per sigma = 25 total trials
- **Estimated time:** ~15-20 minutes
- **Output:** `benchmark_results/sigma_investigation/sigma_investigation.json`

**Purpose:** Understand exactly why wide particle size distributions fail by instrumenting:
- Particle size statistics (mean, std, min, max, size ratio, CV)
- Failure stages (radii generation, PCA, CCA)
- Correlation between size distribution metrics and failure

**Hypothesis:** Wide distributions create large size ratios (max/min) that violate geometric packing constraints in PCA.

**Current Progress:** Check with:
```bash
tail -30 /tmp/claude/-home-mar-Development-PyFracVAL/tasks/bf597b0.output | grep -E "seed=|✓|✗"
```

## Expected Insights

### From Full Parameter Sweep
- Precise Df-kf relationship formula
- Exact boundaries of feasible regions
- Identification of "sweet spots" for reliable generation
- Statistics on failure rates vs parameter distance from optimal

### From Large Aggregate Scaling
- Whether geometric constraints relax or tighten with N
- Runtime scaling (expected O(N²) for CCA)
- If different Df values scale differently
- Whether we need to adjust kf for larger N

### From Sigma Investigation
- Quantitative threshold for size ratio (max/min) causing failure
- Coefficient of variation threshold
- Whether failures are deterministic (always fail) or stochastic
- Physical explanation for why algorithm can't handle wide distributions

## Quick Status Check

Run all three at once:
```bash
echo "=== Full Sweep ===" && grep -E "^\[.*Testing" /tmp/claude/-home-mar-Development-PyFracVAL/tasks/b1d7ef9.output | tail -1
echo "=== Large Aggregate ===" && tail -3 /tmp/claude/-home-mar-Development-PyFracVAL/tasks/b41ef78.output
echo "=== Sigma Investigation ===" && tail -3 /tmp/claude/-home-mar-Development-PyFracVAL/tasks/bf597b0.output
```

## Next Steps (After Completion)

1. Create comprehensive analysis documents for each benchmark
2. Update `PARAMETER_SWEEP_ANALYSIS.md` with full sweep data
3. Create `SCALING_ANALYSIS.md` for large aggregate results
4. Create `SIGMA_FAILURE_ANALYSIS.md` with physical interpretation
5. Update config.py defaults based on findings
6. Consider implementing:
   - Automatic parameter validation with warnings
   - Adaptive kf selection based on Df
   - Size distribution pre-check before simulation
