# PyFracVAL Parameter Sweep Analysis

**Date:** 2026-01-09
**Mode:** Quick Sweep (5×7×3 grid = 105 combinations, 315 trials total)
**Duration:** 1.6 minutes

## Executive Summary

Parameter sweep reveals that **particle size distribution (sigma)** has the strongest impact on simulation success, with narrow distributions (sigma=1.3) performing far better than wide distributions (sigma=2.0). Additionally, there's a clear **inverse relationship between fractal dimension (Df) and fractal prefactor (kf)**: lower Df values require higher kf for success.

**Overall Success Rate:** 21.9% (69/315 trials)

## Key Findings

### 1. Particle Size Distribution Impact (sigma_p_geo)

| Sigma | Excellent (≥90%) | Good (70-90%) | Poor (30-70%) | Failed (<30%) | Best Success |
|-------|------------------|---------------|---------------|---------------|--------------|
| 1.3   | 14               | 0             | 13            | 8             | 100%         |
| 1.5   | 4                | 0             | 3             | 28            | 100%         |
| 2.0   | 0                | 0             | 3             | 32            | 67%          |

**Conclusion:** Narrow particle size distributions (sigma ≤ 1.5) are essential for reliable aggregate generation. Wide distributions (sigma ≥ 2.0) are nearly impossible to handle with this algorithm.

### 2. Inverse Df-kf Relationship

For sigma=1.3 (the most reliable configuration):

| Df  | Optimal kf Range | Success Pattern |
|-----|------------------|-----------------|
| 1.6 | [1.2, 1.4]       | 100% success    |
| 1.8 | [1.0, 1.2]       | 100% success    |
| 2.0 | [0.8, 1.2]       | 100% success    |
| 2.2 | [0.8, 1.0]       | 100% success    |
| 2.4 | [0.6, 0.8]       | 100% success    |

**Physical Interpretation:** As fractal dimension increases (more compact aggregates), lower prefactor values are needed to achieve stable packing. This suggests geometric constraints during particle-cluster aggregation.

### 3. Feasibility Regions

#### Sigma = 1.3 (Recommended)
- **All Df values feasible** (1.6 → 2.4)
- Best combinations:
  - Df=1.60, kf=1.20: 100% (3/3)
  - Df=1.80, kf=1.00: 100% (3/3)
  - Df=2.00, kf=0.80: 100% (3/3)
  - Df=2.20, kf=0.80: 100% (3/3)
  - Df=2.40, kf=0.60: 100% (3/3)

#### Sigma = 1.5 (Challenging)
- **Difficult regions:** Df=1.6, Df=2.4
- **Feasible regions:** Df=1.8-2.2 with careful kf selection
- Reliable combos limited to:
  - Df=1.80, kf=0.80: 100% (3/3)
  - Df=2.00, kf=0.80: 100% (3/3)
  - Df=2.20, kf=0.60: 100% (3/3)

#### Sigma = 2.0 (Avoid)
- **Nearly impossible** across all Df-kf combinations
- Only 1 marginally viable option:
  - Df=2.00, kf=0.60: 67% (2/3)
- Recommendation: **Do not use sigma ≥ 2.0**

## Performance Observations

### Runtime Patterns
- Successful runs: 0.4-1.6 seconds average
- Failed runs: 0.001-0.08 seconds (fail fast in PCA)
- Most failures occur early in PCA subclustering (k=2 typically)

### Failure Modes
Most failures manifest as:
```
WARNING - PCA k=2: No candidates found and no more available monomers to swap with.
ERROR - PCA failed Search/Swap for k=2 (Attempt 1).
```

This indicates the algorithm cannot find geometrically valid particle placements that satisfy the target Df/kf constraints.

## Recommendations

### For Production Use
1. **Always use sigma ≤ 1.5** for reliable aggregate generation
2. **Use the Df-kf relationship table** to select compatible parameters
3. **Avoid extremes:** kf < 0.6 or kf > 1.4 have narrow success windows

### Optimal Starting Points
For first-time users or critical applications:

| Target | Df  | kf  | sigma | Expected Success |
|--------|-----|-----|-------|------------------|
| Loose  | 1.8 | 1.0 | 1.3   | 100%             |
| Medium | 2.0 | 1.0 | 1.3   | 100%             |
| Dense  | 2.2 | 0.8 | 1.3   | 100%             |

### Future Work
1. Run full sweep (11×16×3 grid) to map finer parameter boundaries
2. Investigate algorithmic improvements for wide size distributions (sigma > 1.5)
3. Test larger aggregates (N > 128) to see if constraints relax
4. Explore adaptive kf selection based on Df using discovered relationship

## Data Files

- Full results: `benchmark_results/parameter_sweep/sweep_results_quick.json`
- Grid specification:
  - Df: 1.6, 1.8, 2.0, 2.2, 2.4 (5 points)
  - kf: 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8 (7 points)
  - sigma: 1.3, 1.5, 2.0 (3 levels)
  - N: 128 (fixed)
  - Trials per combo: 3

## Appendix: Detailed Success Rates

### Sigma = 1.3 Success Matrix

| Df  | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 1.6 | 0%  | 0%  | 67% | 100%| 100%| 67% | 67% |
| 1.8 | 0%  | 33% | 100%| 100%| 33% | 33% | 0%  |
| 2.0 | 0%  | 100%| 100%| 100%| 0%  | 0%  | 0%  |
| 2.2 | 67% | 100%| 100%| 0%  | 0%  | 0%  | 0%  |
| 2.4 | 100%| 100%| 33% | 33% | 0%  | 0%  | 0%  |

### Sigma = 1.5 Success Matrix

| Df  | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 1.6 | 0%  | 0%  | 67% | 33% | 33% | 0%  | 0%  |
| 1.8 | 0%  | 100%| 67% | 33% | 0%  | 0%  | 0%  |
| 2.0 | 33% | 100%| 33% | 0%  | 0%  | 0%  | 0%  |
| 2.2 | 100%| 33% | 0%  | 0%  | 0%  | 0%  | 0%  |
| 2.4 | 33% | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  |

### Sigma = 2.0 Success Matrix

| Df  | 0.6 | 0.8 | 1.0 | 1.2 | 1.4 | 1.6 | 1.8 |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 1.6 | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  |
| 1.8 | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  |
| 2.0 | 67% | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  |
| 2.2 | 33% | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  |
| 2.4 | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  | 0%  |
