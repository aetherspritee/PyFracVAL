# Phase 3: CPU Parallelization Strategy

## Goal
Achieve 2-3x additional speedup using **CPU parallelization** only. No GPU required.

## Current Bottleneck Analysis

From benchmarks:
- **Rotation loops**: 360 sequential attempts per particle (PCA) or pair (CCA)
- **Overlap calculations**: Already optimized (Phase 1), but computed sequentially
- **Current state**: N=512 @ ~2.0s after Phase 1

## Key Insight: Embarrassingly Parallel Rotations

The rotation loop (line 805-843 in pca_agg.py) tries angles sequentially:
```python
while cov_max > tol_ov and intento < max_rotations:
    intento += 1
    coord_k_new, theta_a_new = _reintento(...)  # New rotation angle
    cov_max = calculate_max_overlap_pca_fast(...)  # Check overlap
```

**Problem**: Each rotation is independent, but we compute them one-by-one.
**Solution**: Batch rotations and evaluate in parallel across CPU cores.

---

## Optimization Strategies

### 1. Batch Rotation Evaluation (Highest Impact)

**Idea**: Pre-compute 32-64 rotation angles, evaluate overlap for all simultaneously.

**Current approach**:
```python
for rotation in range(360):
    coords_new = rotate(coords, angle=rotation)
    overlap = check_overlap(coords_new)
    if overlap < tolerance:
        break  # Found valid position
```

**Parallelized approach**:
```python
# Generate batch of angles
angles = np.linspace(0, 2*pi, num=32)  # 32 parallel evaluations

# Parallel rotation using Numba parallel=True
@njit(parallel=True)
def batch_rotate_and_check(coords, angles, ...):
    overlaps = np.empty(len(angles))
    for i in prange(len(angles)):  # Parallel loop
        coords_rotated = rodrigues_rotation(coords, angles[i])
        overlaps[i] = calculate_max_overlap(coords_rotated, ...)
    return overlaps

# Find first angle with acceptable overlap
overlaps = batch_rotate_and_check(coords, angles, ...)
valid_idx = np.where(overlaps < tolerance)[0]
if len(valid_idx) > 0:
    best_angle = angles[valid_idx[0]]
```

**Expected speedup**: 2-4x (depends on CPU core count)
- 8 cores → ~4x for rotation loops
- Rotation loops are ~60% of remaining runtime
- **Overall: 1.5-2x total speedup**

---

### 2. Re-enable Numba parallel=True (Medium Impact)

**Current state**:
```python
@jit(parallel=False, fastmath=True, cache=True)  # parallel=False!
def calculate_max_overlap_pca_fast(...): ...
```

**Why disabled?** Early termination (`return overlap`) breaks parallelization.

**Solution**: Create two versions:
1. **Sequential with early exit** (current) - for small N
2. **Parallel without early exit** - for large N

```python
@njit(parallel=True, fastmath=True, cache=True)
def calculate_max_overlap_pca_parallel(coords_agg, radii_agg, coord_new, radius_new):
    """Parallel version (no early termination) for N > 200."""
    n_agg = coords_agg.shape[0]
    max_overlap_val = -np.inf
    
    # Parallel reduction (all iterations complete)
    for j in prange(n_agg):  # prange = parallel range
        coord_agg = coords_agg[j]
        radius_agg = radii_agg[j]
        radius_sum = radius_new + radius_agg
        
        d_sq = 0.0
        for dim in range(3):
            d_sq += (coord_new[dim] - coord_agg[dim]) ** 2
        
        radius_sum_sq = radius_sum * radius_sum
        if d_sq > radius_sum_sq:
            continue
        
        dist = np.sqrt(d_sq)
        overlap = 1.0 - dist / radius_sum
        max_overlap_val = max(overlap, max_overlap_val)
    
    return max_overlap_val

# Dispatch based on size
def calculate_max_overlap_auto(coords_agg, radii_agg, coord_new, radius_new, tolerance):
    if coords_agg.shape[0] > 200:
        return calculate_max_overlap_pca_parallel(coords_agg, radii_agg, coord_new, radius_new)
    else:
        return calculate_max_overlap_pca_fast(coords_agg, radii_agg, coord_new, radius_new, tolerance)
```

**Expected speedup**: 1.2-1.5x for N > 200 (reduces sequential bottleneck)

---

### 3. Parallel PCA Subcluster Building (Low-Medium Impact)

**Current**: Build subclusters sequentially (one after another).

**Opportunity**: Initial subclusters are independent, build in parallel.

```python
from multiprocessing import Pool

def build_subcluster(particle_indices, config):
    """Build one subcluster (independent task)."""
    aggregator = PCAggregator(...)
    return aggregator.run()

# Build N subclusters in parallel
with Pool(processes=cpu_count()) as pool:
    subclusters = pool.map(build_subcluster, subcluster_particle_groups)
```

**Expected speedup**: 1.3-1.8x (for N_subclusters > 4)
**Caveat**: Requires process-level parallelism (GIL bypass), adds complexity.

---

### 4. Vectorized Candidate Selection (Already Done in Phase 2)

✅ Spatial indexing with k-d tree (Phase 2)
✅ Vectorized distance calculations

**No additional work needed.**

---

### 5. SIMD Auto-Vectorization (Compiler-Level)

**Idea**: Let Numba/LLVM auto-vectorize inner loops with SIMD instructions.

```python
@njit(parallel=False, fastmath=True, cache=True)
def calculate_distances_simd(coords1, coords2):
    """Inner loop optimized for SIMD (AVX2/AVX512)."""
    n = coords1.shape[0]
    distances = np.empty(n, dtype=np.float64)
    
    # This loop will be auto-vectorized by LLVM
    for i in range(n):
        dx = coords1[i, 0] - coords2[i, 0]
        dy = coords1[i, 1] - coords2[i, 1]
        dz = coords1[i, 2] - coords2[i, 2]
        distances[i] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return distances
```

**Expected speedup**: 1.1-1.2x (automatic with `fastmath=True`)
**Already active** in current code.

---

## Implementation Roadmap

### Week 1: Batch Rotation Evaluation

**Priority**: ⭐⭐⭐ (Highest impact)

1. Create `batch_rotate_and_check()` function with `prange`
2. Modify PCA `_sticking_process()` to use batched approach
3. Modify CCA `_cca_sticking_v1()` similarly
4. Add configuration parameter: `batch_size` (default: 32)
5. Benchmark: Expect **1.5-2x speedup**

**Files to modify**:
- `pyfracval/pca_agg.py`: Lines 805-843 (rotation loop)
- `pyfracval/cca_agg.py`: Lines 879-894 (rotation loop)
- `pyfracval/utils.py`: Add `batch_rodrigues_rotation()` helper

**Estimated effort**: 2-3 days

---

### Week 2: Parallel Overlap Calculations

**Priority**: ⭐⭐ (Medium impact for large N)

1. Create `calculate_max_overlap_*_parallel()` versions
2. Add size-based dispatch logic
3. Benchmark threshold (where parallel beats sequential)
4. Update all call sites with auto-dispatch

**Files to modify**:
- `pyfracval/utils.py`: Add parallel versions, dispatcher

**Estimated effort**: 1-2 days

---

### Week 3: Parallel Subcluster Building (Optional)

**Priority**: ⭐ (Lower priority, adds complexity)

1. Refactor `Subclusterer` to support parallel execution
2. Add process pool for independent subclusters
3. Handle seed management for reproducibility
4. Test with different core counts

**Files to modify**:
- `pyfracval/pca_subclusters.py`: Refactor for parallelism
- `pyfracval/main_runner.py`: Add multiprocessing pool

**Estimated effort**: 3-5 days
**Defer if unnecessary** - only if targeting N > 1000

---

## Expected Performance Gains

### Conservative Estimates (8-core CPU)

| Optimization | Speedup | Cumulative | Runtime (N=512) |
|--------------|---------|------------|-----------------|
| Baseline (Stock) | 1.0x | 1.0x | 4.4s |
| Phase 1 (done) | 2.2x | 2.2x | 2.0s |
| Batch rotations | 1.5x | 3.3x | **1.3s** |
| Parallel overlap | 1.2x | 4.0x | **1.1s** |
| Parallel subclusters | 1.3x | 5.2x | **0.85s** |

**Target**: **3-4x speedup over Phase 1** (6-8x over stock)

---

## Technical Considerations

### Thread Count Configuration

```python
import os
os.environ['NUMBA_NUM_THREADS'] = str(cpu_count())  # Auto-detect cores
```

**Recommendation**: Default to `cpu_count()`, allow user override via config.

### Memory Overhead

Batch rotation creates temporary arrays:
- Batch size 32: 32 × (N × 3 × 8 bytes) = ~8 KB per batch (N=256)
- Negligible for modern systems

### Reproducibility

Parallel execution must maintain determinism:
```python
# Use thread-safe RNG with fixed seeds per batch
rng = np.random.Generator(np.random.PCG64(seed))
angles = rng.uniform(0, 2*np.pi, size=batch_size)
```

---

## Comparison: CPU Parallelization vs GPU

| Aspect | CPU (This Plan) | GPU (Numba CUDA) |
|--------|-----------------|------------------|
| **Hardware req** | ✅ Any multi-core CPU | ❌ NVIDIA GPU |
| **Speedup** | 3-4x | 4-6x |
| **Portability** | ✅ Cross-platform | ❌ NVIDIA only |
| **Code complexity** | Low (Numba prange) | Medium (kernel code) |
| **User accessibility** | ✅ Everyone | ⚠️ GPU owners only |
| **Cost** | $0 | $500-$2000 |

**Decision**: CPU parallelization aligns with accessibility goals.

---

## Benchmark Plan

### Test Configurations

```python
configs = [
    {"N": 128, "Df": 1.8, "kf": 1.0},
    {"N": 256, "Df": 1.9, "kf": 1.2},
    {"N": 512, "Df": 1.9, "kf": 1.1},
    {"N": 1024, "Df": 2.0, "kf": 1.1},  # New: large aggregate
]

# Test different core counts
thread_counts = [1, 2, 4, 8, 16]  # Measure scaling
```

### Metrics to Track

1. **Runtime**: Total wall-clock time
2. **Speedup**: vs Phase 1 baseline
3. **Scaling efficiency**: speedup / core_count
4. **CPU utilization**: `htop` monitoring
5. **Memory usage**: RSS peak

---

## Implementation Guidelines

### Code Style

```python
# Use Numba's prange for parallel loops
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def parallel_function(array):
    result = np.empty_like(array)
    for i in prange(len(array)):  # Parallel loop
        result[i] = expensive_operation(array[i])
    return result
```

### Error Handling

```python
# Graceful fallback if parallelization fails
try:
    result = parallel_version(data)
except Exception as e:
    logger.warning(f"Parallel execution failed: {e}. Falling back to sequential.")
    result = sequential_version(data)
```

### Configuration

Add to `config.py`:
```python
class SimulationConfig(BaseModel):
    # ... existing fields ...
    
    # Parallelization settings
    num_threads: int = Field(default=-1, description="CPU threads (-1 = auto-detect)")
    rotation_batch_size: int = Field(default=32, description="Parallel rotation batch size")
    enable_parallel_overlap: bool = Field(default=True, description="Use parallel overlap for N>200")
```

---

## Testing Strategy

### Unit Tests

```python
def test_batch_rotation_correctness():
    """Verify batch rotation gives same results as sequential."""
    # Same seed → same results
    result_seq = sequential_rotation(seed=42)
    result_batch = batch_rotation(seed=42)
    np.testing.assert_allclose(result_seq, result_batch)

def test_parallel_overlap_matches_sequential():
    """Parallel and sequential overlap must agree."""
    coords = np.random.rand(500, 3)
    radii = np.random.rand(500)
    
    ov_seq = calculate_max_overlap_pca_fast(coords, radii, ...)
    ov_par = calculate_max_overlap_pca_parallel(coords, radii, ...)
    
    assert abs(ov_seq - ov_par) < 1e-10
```

### Performance Tests

```python
def test_parallel_speedup():
    """Verify parallel version is faster for large N."""
    coords = np.random.rand(1000, 3)
    
    t_seq = timeit(lambda: sequential_version(coords), number=10)
    t_par = timeit(lambda: parallel_version(coords), number=10)
    
    assert t_par < t_seq * 0.7  # At least 30% faster
```

---

## Future Enhancements (Post Phase 3)

### 1. Adaptive Batch Sizing
Automatically tune batch size based on N and core count:
```python
optimal_batch_size = min(64, max(16, cpu_count() * 4))
```

### 2. Hierarchical Parallelism
- **Coarse-grained**: Parallel subclusters (process-level)
- **Fine-grained**: Parallel rotations (thread-level)

### 3. Cache-Aware Scheduling
Reorder computations to maximize data locality:
```python
# Group particles by spatial proximity (better cache hits)
sorted_indices = kdtree.query_ball_tree(...)
```

---

## Summary

**Phase 3 CPU parallelization provides:**
- ✅ **3-4x speedup** over Phase 1 (cumulative 6-8x vs stock)
- ✅ **No GPU required** - accessible to all users
- ✅ **Low implementation complexity** - mostly Numba `prange`
- ✅ **Scalable** - benefits from more CPU cores
- ✅ **Maintainable** - minimal code changes

**Timeline**: 2-3 weeks for full implementation + testing

**Recommendation**: Proceed with Week 1 (batch rotations) first, benchmark, then decide on Weeks 2-3.
