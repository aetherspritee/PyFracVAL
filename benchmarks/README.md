# PyFracVAL Sticking Benchmark Suite

Comprehensive testing infrastructure for analyzing the sticking process convergence across various Df/kf parameter combinations.

## Quick Start

```bash
# Run quick test (3 trials of stable cases)
python benchmarks/sticking_benchmark.py

# Run specific category (10 trials each)
python -c "from benchmarks.sticking_benchmark import StickingBenchmark; \
           StickingBenchmark().run_suite('low_df', n_trials=10)"

# Run full benchmark (all categories, 10 trials - takes hours!)
python -c "from benchmarks.sticking_benchmark import StickingBenchmark; \
           StickingBenchmark().run_all(n_trials=10)"
```

## Benchmark Categories

### 1. **Stable** (Baseline)
Known stable parameter combinations for validating the benchmark infrastructure.
- 3 test cases
- Expected success rate: >95%

### 2. **Low Df** (Known Problematic)
Tests convergence for low fractal dimensions (Df < 1.7).
- 5 test cases
- Current estimated success: 20-60%
- Target after optimization: 80-90%

### 3. **High Df** (Dense Packing)
Tests convergence for high fractal dimensions (Df > 2.2).
- 5 test cases
- Current estimated success: 40-70%
- Target after optimization: 85-95%

### 4. **Extreme kf** (Prefactor Limits)
Tests unusual fractal prefactor values.
- 4 test cases
- Tests kf ∈ [0.5, 2.0]

### 5. **Polydisperse** (Size Distribution)
Tests various polydispersity levels (rp_gstd).
- 4 test cases
- Includes monodisperse (1.0) to very polydisperse (2.0)

### 6. **Scaling** (Particle Count)
Tests performance scaling with N.
- 4 test cases
- N ∈ [64, 512]

### 7. **Corner** (Combined Extremes)
Stress tests combining multiple extreme parameters.
- 4 test cases
- Most challenging scenarios

## Output Structure

```
benchmark_results/
├── BENCHMARK_REPORT.md           # Human-readable summary
├── stable_summary.json            # Per-category JSON summaries
├── low_df_summary.json
├── ...
├── stable_Original_paper_example.json  # Per-test-case detailed results
├── low_df_Lower_bound.json
├── ...
└── aggregates/                    # Generated aggregate files
    ├── stable/
    ├── low_df/
    └── ...
```

## Interpreting Results

### Success Rate
- **>90%**: Excellent, algorithm handles these parameters well
- **70-90%**: Good, occasional failures expected
- **50-70%**: Moderate, needs investigation
- **<50%**: Poor, optimization urgently needed

### Runtime
- N=128: typically 1-5 seconds
- N=256: typically 3-15 seconds
- N=512: typically 10-60 seconds

Unusually long runtimes indicate rotation struggles.

### Failure Stages
- **PCA**: Particle-cluster aggregation failed (subcluster creation)
- **CCA**: Cluster-cluster aggregation failed (merging subclusters)
- **EXCEPTION**: Unexpected error (check logs)
- **UNKNOWN**: Generic failure (requires log analysis)

## Example Report Section

```markdown
### LOW_DF

**Lower bound**
- Parameters: N=128, Df=1.5, kf=1.0, rp_gstd=1.5
- Success: 4/10 (40.0%)
- Avg Runtime: 8.34s
- Failures: ['PCA', 'PCA', 'PCA', 'PCA', 'PCA', 'PCA']
```

**Interpretation:** Low success rate with all failures in PCA stage suggests gamma calculation or candidate selection issues at low Df.

## Adding Custom Test Cases

Edit `sticking_benchmark.py`:

```python
def _define_custom_cases(self) -> List[Dict]:
    return [
        {'N': 256, 'Df': 1.75, 'kf': 1.1, 'rp_gstd': 1.4,
         'description': 'My custom test'},
    ]

# In __init__:
self.test_suites = {
    ...
    'custom': self._define_custom_cases(),
}
```

## Integration with Optimization Workflow

1. **Baseline**: Run full benchmark with current implementation
2. **Implement fix**: Modify PCA/CCA code
3. **Re-benchmark**: Run same suite with optimized code
4. **Compare**: Diff the JSON summaries or reports

Example comparison:
```bash
# Before optimization
Success rate: stable=96%, low_df=35%, high_df=58%

# After Fibonacci spiral rotation
Success rate: stable=99%, low_df=72%, high_df=81%
```

## Benchmark Metrics

Each `BenchmarkResult` contains:

**Always Available:**
- `success`: bool
- `runtime_seconds`: float
- `failure_stage`: str | None
- `failure_reason`: str | None
- All input parameters (N, Df, kf, etc.)

**If Successful:**
- `final_N`: int (should equal input N)
- `final_Rg`: float (radius of gyration)

**Future Instrumentation:**
- `total_rotations_pca`: Total rotation attempts in PCA
- `total_rotations_cca`: Total rotation attempts in CCA
- `gamma_failures`: Count of gamma calculation failures
- `candidate_failures`: Count of empty candidate list occurrences

## Performance Notes

- Each trial is deterministic (seeded by category + description + trial_num)
- Aggregates are saved to disk (can be large for N=512)
- Full benchmark (7 categories × ~4 cases × 10 trials = 280 runs) takes 2-6 hours
- Use `n_trials=3` for rapid iteration during development

## Related Documentation

- `../STICKING_ANALYSIS.md`: Full analysis of sticking issues
- `../pyfracval/pca_agg.py`: PCA implementation
- `../pyfracval/cca_agg.py`: CCA implementation
- `../docs/FracVAL/`: Original Fortran code for comparison

## Citation

If you use these benchmarks in research, please cite:

```bibtex
@software{pyfracval_benchmarks,
  title={PyFracVAL Sticking Benchmark Suite},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/pyfracval}
}
```

Also cite the original FracVAL paper:
- Moran et al. (2019). "FracVAL: A General Tool for Generating Fractal Aggregates"
