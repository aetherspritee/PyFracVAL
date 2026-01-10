# Phase 3 Hybrid Strategy: Parallel Overlap with Sequential Rotation

## Problem Analysis

Batch rotation failed because:
- Early termination finds valid positions in 10-50 attempts
- Batch computes 32 positions (20-25 wasted)
- Overhead > savings

## Key Insight

**Separate the costs:**

1. **Rotation computation**: O(1) - very cheap
   - Just sin/cos and vector math
   - ~10-50 attempts before success

2. **Overlap check**: O(n) - expensive for large N
   - Loops over all aggregate particles  
   - 512 distance calculations for N=512
   - Called 10-50 times per particle

**Current**: Sequential rotation + Sequential overlap check
**Proposed**: Sequential rotation + **Parallel overlap check**

## Proposed Implementation

### Conditional Parallelization

```python
# In utils.py - add size-based dispatcher
def calculate_max_overlap_auto(coords_agg, radii_agg, coord_new, radius_new, tolerance):
    """Automatically choose parallel vs sequential based on aggregate size."""
    n = coords_agg.shape[0]
    
    if n > PARALLEL_THRESHOLD:  # e.g., 200
        # Large aggregate: parallel overlap (no early termination)
        return calculate_max_overlap_parallel(coords_agg, radii_agg, coord_new, radius_new)
    else:
        # Small aggregate: sequential with early termination
        return calculate_max_overlap_fast(coords_agg, radii_agg, coord_new, radius_new, tolerance)
```

### Parallel Overlap Check

```python
@njit(parallel=True, fastmath=True)
def calculate_max_overlap_parallel(coords_agg, radii_agg, coord_new, radius_new):
    """Parallel overlap check - no early termination."""
    n_agg = coords_agg.shape[0]
    overlaps = np.empty(n_agg)
    
    # Parallel loop over all particles
    for j in prange(n_agg):  # ← Parallelize this!
        coord_agg = coords_agg[j]
        radius_agg = radii_agg[j]
        radius_sum = radius_new + radius_agg
        
        # Compute distance
        d_sq = 0.0
        for dim in range(3):
            d_sq += (coord_new[dim] - coord_agg[dim]) ** 2
        
        # Bounding check
        if d_sq > radius_sum * radius_sum:
            overlaps[j] = -np.inf
            continue
        
        # Compute overlap
        dist = np.sqrt(d_sq)
        overlaps[j] = 1.0 - dist / radius_sum
    
    return np.max(overlaps)
```

### Usage in Rotation Loop

```python
# Sequential rotation with parallel overlap
while overlap > tolerance and attempt < 360:
    attempt += 1
    new_position = rotate(attempt)  # Cheap O(1)
    
    # Expensive O(n) - but now parallel!
    overlap = calculate_max_overlap_auto(
        coords, radii, new_position, radius_new, tolerance
    )
    
    if overlap < tolerance:
        break  # Early exit still works!
```

## Expected Performance

### For N=512 (512 particles per overlap check)

**Sequential overlap check**:
- 512 distance calculations (serial)
- 10-50 rotation attempts
- Total: 5,120 - 25,600 serial operations

**Parallel overlap check (8 cores)**:
- 512 distance calculations / 8 = 64 per core
- 10-50 rotation attempts
- Total: 640 - 3,200 operations per core

**Expected speedup**: ~4-6x for overlap checks
**Overall speedup**: ~2-3x (overlap is 70-80% of time)

### For N=128 (128 particles)

**Overhead dominates**:
- Thread spawning: ~0.1ms
- 128 / 8 = 16 operations per core (too small)
- Use sequential version

### Breakeven Analysis

```
Parallel time = overhead + work / cores
Sequential time = work

Parallel faster when:
overhead + work / cores < work
overhead < work × (1 - 1/cores)

For 8 cores:
overhead < work × 0.875

If overhead = 0.1ms:
work > 0.11ms
```

For distance calculations (~10ns each):
- 128 particles: 1.28μs (too small)
- 200 particles: 2.0μs (marginal)
- 512 particles: 5.12μs (worth it!)

**Threshold**: n > 200 particles

## Implementation Plan

### Step 1: Create Parallel Overlap Function

```python
# In utils.py
@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def calculate_max_overlap_pca_parallel(coords_agg, radii_agg, coord_new, radius_new):
    # ... (see above)

@jit(parallel=True, fastmath=True, cache=True, nopython=True)
def calculate_max_overlap_cca_parallel(coords1, radii1, coords2, radii2):
    # Similar, but for cluster pairs
```

### Step 2: Add Auto-Dispatcher

```python
PARALLEL_OVERLAP_THRESHOLD = 200  # Tune based on benchmarks

def calculate_max_overlap_pca_auto(coords_agg, radii_agg, coord_new, radius_new, tolerance):
    if coords_agg.shape[0] > PARALLEL_OVERLAP_THRESHOLD:
        return calculate_max_overlap_pca_parallel(coords_agg, radii_agg, coord_new, radius_new)
    else:
        return calculate_max_overlap_pca_fast(coords_agg, radii_agg, coord_new, radius_new, tolerance)
```

### Step 3: Update Call Sites

```python
# In pca_agg.py - sequential rotation loop
overlap = utils.calculate_max_overlap_pca_auto(
    self.coords[:self.n1],
    self.radii[:self.n1],
    self.coords[k],
    self.radii[k],
    tolerance=self.tol_ov,
)
```

### Step 4: Benchmark

Test with:
- N = [128, 256, 512, 1024]
- Vary threshold: [100, 150, 200, 250, 300]
- Measure total runtime

## Advantages Over Batch Rotation

| Aspect | Batch Rotation | Hybrid (This) |
|--------|----------------|---------------|
| **Early exit** | ❌ Lost | ✅ Kept |
| **Wasted work** | High (21-31 of 32) | None |
| **Memory** | Large batches | Single position |
| **Parallelism** | 32 rotations | n particles |
| **Overhead** | Per batch | Per overlap check |
| **Speedup** | 0.5x (slower!) | 2-3x (predicted) |

## Potential Issues & Mitigations

### Issue 1: Early Termination Loss

Parallel version can't exit early when one particle exceeds tolerance.

**Mitigation**: Only use parallel for n > 200, where:
- Early termination less effective anyway
- More particles = harder to find valid positions
- Fewer attempts benefit from early exit

### Issue 2: Thread Overhead

Spawning threads for each overlap check adds overhead.

**Mitigation**:
- Numba reuses thread pool (no repeated spawning)
- First call pays JIT + thread init cost
- Subsequent calls just dispatch to pool (~10-50μs)

### Issue 3: Cache Coherency

Parallel threads may thrash cache.

**Mitigation**:
- Each thread reads different particles (no contention)
- `coord_new` is read-only (broadcasted)
- Modern CPUs handle this well

### Issue 4: Load Imbalance

Some threads may finish early (bounding check).

**Mitigation**:
- Most particles checked (aggregate is compact)
- prange uses dynamic scheduling
- Load imbalance < 10%

## Configuration

Add to `config.py`:
```python
# --- Performance Tuning ---
USE_PARALLEL_OVERLAP: bool = True  # Enable for N > threshold
PARALLEL_OVERLAP_THRESHOLD: int = 200  # Minimum particles for parallel
```

## Testing Strategy

### Correctness

```python
def test_parallel_matches_sequential():
    coords = np.random.rand(500, 3)
    radii = np.random.rand(500)
    new_pos = np.random.rand(3)
    new_radius = 1.0
    
    seq_result = calculate_max_overlap_fast(coords, radii, new_pos, new_radius, 1e-6)
    par_result = calculate_max_overlap_parallel(coords, radii, new_pos, new_radius)
    
    assert abs(seq_result - par_result) < 1e-10
```

### Performance

```python
# Benchmark with different thresholds
for threshold in [100, 150, 200, 250, 300]:
    config.PARALLEL_OVERLAP_THRESHOLD = threshold
    runtime = benchmark_simulation(N=512)
    print(f"Threshold {threshold}: {runtime:.3f}s")
```

## Expected Results

### Conservative Estimate

| N | Sequential | Hybrid (Parallel Overlap) | Speedup |
|---|-----------|--------------------------|---------|
| 128 | 0.50s | 0.50s | 1.0x (sequential used) |
| 256 | 0.67s | 0.50s | **1.3x** |
| 512 | 2.02s | 1.00s | **2.0x** |
| 1024 | 8.00s | 3.00s | **2.7x** |

### Optimistic Estimate

If overhead is lower than expected:

| N | Speedup |
|---|---------|
| 256 | **1.5x** |
| 512 | **2.5x** |
| 1024 | **3.5x** |

## Why This Should Work

1. **Keeps early termination**: Sequential rotation loop exits after 10-50 attempts
2. **No wasted rotations**: Only computes needed positions
3. **Parallelizes expensive part**: O(n) overlap check, not O(1) rotation
4. **Size-adaptive**: Uses parallel only when beneficial
5. **Low memory**: No batch arrays, just thread-local variables
6. **Numba-native**: Reuses thread pool, minimal overhead

## Next Steps

1. Implement parallel overlap functions
2. Add auto-dispatcher
3. Benchmark with different thresholds
4. Compare to Phase 1 baseline
5. Document findings

This should give us the 2-3x speedup we originally hoped for!
