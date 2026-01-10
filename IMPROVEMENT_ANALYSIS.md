# PyFracVAL Improvement Analysis

**Date:** 2026-01-10  
**Status:** Comprehensive codebase analysis completed  
**Scope:** Performance optimization and algorithmic enhancement opportunities

## Executive Summary

PyFracVAL is a well-structured implementation of the FracVAL algorithm with solid foundational optimizations (Numba JIT, NumPy vectorization, Fibonacci spiral sampling). Current runtimes for N=128 aggregates average ~0.95s with optimal parameters.

**Key Finding:** The algorithm performs ~9,000-32,000 overlap checks per aggregate (N=128-512), making this the critical bottleneck. Multiple optimization opportunities exist with potential speedups ranging from 2-100x.

**Recommended Path:** Incremental optimization starting with low-hanging fruit (2-3x speedup) before considering architectural changes.

---

## 1. Current State Assessment

### Strengths

✅ **Already Optimized:**
- Numba JIT compilation on overlap calculations (`@jit(parallel=True, fastmath=True, cache=True)`)
- NumPy vectorization for CCA candidate selection
- Fibonacci spiral rotation sampling (better coverage than random)
- Adaptive tolerance relaxation (1e-6 → 1e-5 after 180 attempts)
- Particle swapping fallback mechanism
- Clean code architecture with separation of concerns

✅ **Comprehensive Testing:**
- Extensive benchmark infrastructure
- Parameter sweeps validated Df=1.3-2.6 range
- Sigma threshold (4.9x) identified
- Empirical kf=3.0-Df relationship confirmed

### Performance Characteristics

**Runtime Scaling (N=128, optimal params):**
- PCA: ~0.4-0.7s (dominant)
- CCA: ~0.2-0.4s
- Total: ~0.6-1.5s average

**Overlap Check Frequency:**
```
N=128:  ~9,000 checks  (6,300 PCA + 2,800 CCA)
N=512:  ~32,700 checks (25,500 PCA + 7,200 CCA)
N=1024: ~70,000+ checks (scaling quadratically with iterations)
```

**Success Rates:**
- Optimal region (Df=1.8, kf=1.0, σ=1.3): 100%
- Marginal regions: 60-80%
- Failed configurations: 0-40%

---

## 2. Performance Bottlenecks (Verified by Code Inspection)

### Critical Path: Overlap Calculations

**Location:** `pyfracval/utils.py:549-654`

**Functions:**
- `calculate_max_overlap_cca()`: O(n1 × n2) per rotation attempt
- `calculate_max_overlap_pca()`: O(n_agg) per rotation attempt

**Code Analysis:**

```python
@jit(parallel=True, fastmath=True, cache=True)
def calculate_max_overlap_pca(...):
    max_overlap_val = 0.0
    
    for j in prange(n_agg):  # Parallelized
        # Distance calculation
        d_sq = 0.0
        for dim in range(3):
            d_sq += (coord_new[dim] - coord_agg[dim]) ** 2
        dist = np.sqrt(d_sq)
        
        # Overlap calculation
        overlap = 1 - dist / (radius_new + radius_agg)
        max_overlap_val = max(overlap, max_overlap_val)
    
    return max_overlap_val  # Always scans ALL particles
```

**Issue #1:** No early termination - always checks all particles even after finding overlap  
**Issue #2:** No spatial filtering - checks particles far outside interaction range  
**Issue #3:** No bounding checks before expensive sqrt operation

### Secondary Bottleneck: Rotation Loop

**Location:** `pyfracval/pca_agg.py:684-722`

**Code:**
```python
while cov_max > self.tol_ov and intento < max_rotations:  # Up to 360 iterations
    intento += 1
    coord_k_new, theta_a_new = self._reintento(k, ...)  # Sequential rotation
    self.coords[k] = coord_k_new
    cov_max = utils.calculate_max_overlap_pca(...)  # Full O(n) overlap check
    
    if intento >= 180 and cov_max <= 1e-5:  # Adaptive tolerance
        break
```

**Issue #1:** Sequential trial-and-error approach  
**Issue #2:** Each rotation triggers full overlap recalculation  
**Issue #3:** No intelligent angle selection based on collision geometry

### Tertiary Bottleneck: Candidate Selection

**PCA Location:** `pyfracval/pca_agg.py:218-279` (sequential loop)  
**CCA Location:** `pyfracval/cca_agg.py:352-402` (already vectorized)

**PCA Code:**
```python
for idx_cand in range(self.n1):  # Linear scan
    dist_cm_cand = np.linalg.norm(coord_cm - self.coords[idx_cand])
    
    if np.abs(dist_cm_cand - gamma_pc) <= (self.radii[idx_cand] + self.radii[k]):
        candidates.append(idx_cand)
```

**Issue:** O(n) sequential search when spatial indexing could achieve O(log n)

---

## 3. Low-Hanging Fruit Optimizations (2-3x Speedup)

### A. Early Termination in Overlap Checks ⭐⭐⭐

**Current:** Always scans all particles  
**Proposed:** Return immediately when overlap exceeds tolerance

**Implementation:**
```python
@jit(parallel=False, fastmath=True, cache=True)  # Sequential for early exit
def calculate_max_overlap_pca_fast(...):
    threshold = 1e-6  # tolerance parameter
    
    for j in range(n_agg):
        d_sq = 0.0
        for dim in range(3):
            d_sq += (coord_new[dim] - coords_agg[j, dim]) ** 2
        
        # Early bounding check (avoid sqrt)
        if d_sq > (radius_new + radii_agg[j]) ** 2:
            continue  # No overlap possible
        
        dist = np.sqrt(d_sq)
        overlap = 1 - dist / (radius_new + radii_agg[j])
        
        if overlap > threshold:  # Early exit on violation
            return overlap
    
    return 0.0  # No significant overlap found
```

**Expected Impact:** 2-3x speedup in overlap checks (50-70% of total runtime)  
**Trade-off:** Loses parallelization, but early exit compensates  
**Effort:** Low (1-2 hours)

### B. Bounding Sphere Pre-checks ⭐⭐

**Current:** Computes sqrt for every particle pair  
**Proposed:** Compare squared distances before sqrt

**Already shown in implementation above** - check `d_sq > (r1 + r2)^2` before sqrt.

**Expected Impact:** 1.3-1.5x speedup (avoids ~40% of sqrt calls)  
**Effort:** Trivial (included in early termination)

### C. Vectorize PCA Candidate Selection ⭐⭐

**Current:** Sequential loop with NumPy calls inside  
**Proposed:** Use broadcasting like CCA

**Implementation:**
```python
# Current PCA approach (sequential)
candidates = []
for idx_cand in range(self.n1):
    dist_cm_cand = np.linalg.norm(coord_cm - self.coords[idx_cand])
    if np.abs(dist_cm_cand - gamma_pc) <= (self.radii[idx_cand] + self.radii[k]):
        candidates.append(idx_cand)

# Proposed (vectorized, like CCA)
dist_cm_all = np.linalg.norm(self.coords[:self.n1] - coord_cm, axis=1)
condition = np.abs(dist_cm_all - gamma_pc) <= (self.radii[:self.n1] + self.radii[k])
candidates = np.where(condition)[0]
```

**Expected Impact:** 2-3x speedup in candidate selection (~5% of total runtime)  
**Overall Impact:** ~1.1-1.15x total speedup  
**Effort:** Low (30 minutes)

### D. Remove TRACE Logging Overhead ⭐

**Current:** Even when disabled, `logger.isEnabledFor(TRACE_LEVEL_NUM)` is checked on every rotation  
**Proposed:** Compile-time or environment variable control

**Expected Impact:** 1.05-1.1x speedup (minor but trivial to fix)  
**Effort:** Trivial (5 minutes)

**Combined Phase 1 Speedup: ~2.5-3.5x**

---

## 4. Medium-Term Improvements (5-10x Speedup)

### E. Spatial Indexing for Candidate Selection ⭐⭐⭐⭐

**Problem:** O(n) linear search through all particles  
**Solution:** k-d tree or octree for O(log n) range queries

**Implementation Approach:**

```python
from scipy.spatial import cKDTree

class PCASubclustersOptimized:
    def __init__(self, ...):
        self.tree = None  # Rebuild after each particle addition
    
    def _find_candidates_spatial(self, k, coord_cm, gamma_pc):
        # Build tree from current aggregate (amortized cost)
        if self.n1 > 10:  # Only worth it for larger aggregates
            self.tree = cKDTree(self.coords[:self.n1])
            
            # Range query: particles within gamma_pc ± max_radius
            search_radius = gamma_pc + 2 * np.max(self.radii[:self.n1])
            candidates_idx = self.tree.query_ball_point(coord_cm, search_radius)
            
            # Refine with exact geometric constraint
            candidates = [
                idx for idx in candidates_idx
                if abs(np.linalg.norm(coord_cm - self.coords[idx]) - gamma_pc)
                   <= (self.radii[idx] + self.radii[k])
            ]
            return candidates
        else:
            return self._find_candidates_linear(...)  # Fallback
```

**Challenges:**
- Tree rebuild cost: O(n log n) per particle addition
- Trade-off: Rebuild vs persistent structure with updates
- For small n1 (<50), linear scan is faster

**Alternative: Persistent Octree**
- Custom implementation with incremental updates
- Avoids O(n log n) rebuild cost
- More complex but asymptotically better

**Expected Impact:** 
- Small N (128): 1.5-2x (tree rebuild overhead)
- Large N (1024): 5-10x (logarithmic scaling wins)

**Effort:** Medium (4-8 hours for cKDTree, 2-3 days for custom octree)

### F. Batch Rotation with Vectorized Overlap ⭐⭐⭐

**Problem:** Sequential rotation → test → rotation cycle  
**Solution:** Pre-compute batch of angles, vectorize overlap checks

**Concept:**
```python
def batch_rotation_search(self, k, vec_0, i_vec, j_vec, batch_size=32):
    # Pre-compute batch of rotation angles (Fibonacci spiral)
    angles = fibonacci_spiral_angles(batch_size)
    
    # Vectorize: compute all rotated positions at once
    coords_batch = np.array([
        rotate_particle(vec_0, i_vec, j_vec, angle) 
        for angle in angles
    ])  # Shape: (batch_size, 3)
    
    # Vectorized overlap check (custom Numba kernel)
    overlaps = calculate_overlap_batch(
        coords_batch, self.radii[k],
        self.coords[:self.n1], self.radii[:self.n1]
    )  # Shape: (batch_size,)
    
    # Select best angle (minimum overlap)
    best_idx = np.argmin(overlaps)
    if overlaps[best_idx] <= self.tol_ov:
        return coords_batch[best_idx], angles[best_idx]
    
    # Fallback: refine around best region
    return self._refine_rotation(angles[best_idx], ...)
```

**Expected Impact:** 1.5-2x (better parallelization, reduced Python overhead)  
**Effort:** Medium (1-2 days)

### G. Analytical Constraint Pruning ⭐⭐

**Problem:** Rotation attempts in geometrically impossible regions  
**Solution:** Analytical bounds from Gamma_pc geometry

**Concept:**
- Given `gamma_pc` and particle radii, calculate feasible rotation angle ranges
- Eliminate regions where contact is geometrically impossible
- Focus rotation attempts in viable zones

**Expected Impact:** 1.3-1.5x (reduces wasted rotation attempts by ~30%)  
**Effort:** High (2-3 days, requires geometric analysis)

### H. Incremental Rg Calculation ⭐

**Problem:** Recalculates Rg from scratch for every Gamma_pc call  
**Solution:** Update Rg incrementally when adding particles

**Implementation:**
```python
def update_aggregate_properties_incremental(self, k):
    # Current: Recalculate Rg for all n1 particles
    # Proposed: Incremental update formula
    
    # Rg^2 = (1/M) * sum(m_i * r_i^2) - r_cm^2
    # When adding particle k:
    M_old = self.mass_sum
    M_new = M_old + self.mass[k]
    
    r_cm_old = self.r_cm.copy()
    r_cm_new = (M_old * r_cm_old + self.mass[k] * self.coords[k]) / M_new
    
    Rg_sq_old = self.Rg ** 2
    Rg_sq_new = (M_old / M_new) * (Rg_sq_old + np.sum((r_cm_new - r_cm_old)**2)) + \
                (self.mass[k] / M_new) * np.sum((self.coords[k] - r_cm_new)**2)
    
    self.Rg = np.sqrt(Rg_sq_new)
    self.r_cm = r_cm_new
    self.mass_sum = M_new
```

**Expected Impact:** 1.1-1.2x (Rg calculation is ~5% of runtime)  
**Effort:** Medium (1 day, requires careful validation)

**Combined Phase 2 Speedup: ~5-10x (cumulative with Phase 1)**

---

## 5. Advanced Optimizations (10-100x Speedup)

### I. GPU Acceleration ⭐⭐⭐⭐⭐

**Target Operations:**
1. Overlap calculations (embarrassingly parallel)
2. Batch rotation attempts
3. Distance matrix computations

**Implementation Frameworks:**

**Option 1: CuPy** (easiest)
```python
import cupy as cp

@cp.fuse()  # JIT kernel fusion
def calculate_overlap_gpu(coords1, radii1, coords2, radii2):
    # Vectorized distance calculation on GPU
    dist_matrix = cp.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=2)
    radii_sum = radii1[:, None] + radii2[None, :]
    overlap_matrix = 1 - dist_matrix / radii_sum
    return cp.max(overlap_matrix)
```

**Option 2: JAX** (more flexible)
```python
import jax.numpy as jnp
from jax import jit, vmap

@jit
def overlap_single_pair(coord1, r1, coord2, r2):
    dist = jnp.linalg.norm(coord1 - coord2)
    return 1 - dist / (r1 + r2)

# Vectorize over all pairs
overlap_all_pairs = vmap(vmap(overlap_single_pair, (None, None, 0, 0)), (0, 0, None, None))

def calculate_overlap_jax(coords1, radii1, coords2, radii2):
    overlaps = overlap_all_pairs(coords1, radii1, coords2, radii2)
    return jnp.max(overlaps)
```

**Option 3: Custom CUDA Kernels** (highest performance)
- Direct memory management
- Shared memory for radii/coordinates
- Warp-level optimizations

**Expected Impact:**
- N=128: 5-10x (memory transfer overhead)
- N=512: 20-50x (GPU compute dominates)
- N=1024+: 50-100x (quadratic speedup scales)

**Challenges:**
- Memory transfer CPU ↔ GPU overhead
- Algorithm restructuring for batch processing
- Dependency: CUDA-capable GPU

**Effort:** High (1-2 weeks for CuPy/JAX, 3-4 weeks for CUDA)

### J. Hierarchical Sticking Algorithm ⭐⭐⭐⭐

**Problem:** Fine-grained rotation search is expensive  
**Solution:** Coarse → fine hierarchical placement

**Algorithm:**
1. **Coarse phase**: Grid-based angle search (e.g., 36 angles, 10° spacing)
2. **Identify promising regions**: Angles with overlap ≤ 2 × tolerance
3. **Fine phase**: Local refinement around best coarse angles (1° spacing)

**Expected Impact:** 2-3x (reduces rotation attempts from 360 → ~50)  
**Effort:** Medium-High (2-3 days)

### K. Parallel PCA Subclustering ⭐⭐⭐

**Current:** Sequential subclustering (one subcluster at a time)  
**Proposed:** Parallel subclustering with process pool

**Implementation:**
```python
from multiprocessing import Pool

def build_subcluster_parallel(subcluster_args):
    """Worker function for parallel execution."""
    pca = PCASubclusters(**subcluster_args)
    return pca.build_aggregate()

def parallel_subclustering(radii, N, num_subclusters, num_workers=4):
    # Divide radii into chunks
    chunks = partition_radii(radii, num_subclusters)
    
    # Prepare arguments for each subcluster
    args_list = [prepare_subcluster_args(chunk, ...) for chunk in chunks]
    
    # Parallel execution
    with Pool(num_workers) as pool:
        results = pool.map(build_subcluster_parallel, args_list)
    
    return concatenate_subclusters(results)
```

**Expected Impact:** 2-4x (depending on CPU cores available)  
**Challenge:** CCA still sequential (harder to parallelize)  
**Effort:** Medium (1-2 days)

### L. Smart Initial Placement ⭐⭐

**Problem:** First rotation attempt is arbitrary  
**Solution:** Analytical initial angle from Gamma_pc geometry

**Concept:**
- Use `gamma_pc` to estimate optimal initial contact angle
- Reduces rotation attempts by starting closer to solution

**Expected Impact:** 1.2-1.4x (reduces average rotations from ~50 → ~35)  
**Effort:** Medium (1-2 days)

**Combined Phase 3 Potential: 10-100x (with GPU + architectural changes)**

---

## 6. Algorithmic Alternatives

### Alternative 1: Monte Carlo Placement

**Replace:** Deterministic rotation search  
**With:** Monte Carlo sampling + Metropolis acceptance

**Advantages:**
- Natural parallelization (many samples simultaneously)
- Probabilistic guarantee instead of exhaustive search
- Potentially faster convergence

**Disadvantages:**
- Non-deterministic (harder to reproduce results)
- May require more samples for tight tolerances
- Conceptual departure from original FracVAL

### Alternative 2: Gradient-Based Optimization

**Replace:** Grid/spiral rotation search  
**With:** Numerical optimization (e.g., L-BFGS)

**Objective Function:** Minimize max_overlap(θ)  
**Variables:** Rotation angles (θ, φ) for particle placement

**Advantages:**
- Faster convergence (O(log n) iterations vs O(n) grid search)
- Works well for smooth overlap landscapes

**Disadvantages:**
- May get stuck in local minima
- Requires differentiable overlap calculation
- More complex implementation

### Alternative 3: Constraint Satisfaction Solver

**Replace:** Trial-and-error sticking  
**With:** Declarative constraint solver (e.g., Z3, OR-Tools)

**Constraints:**
- No overlap: `distance(p_i, p_j) ≥ r_i + r_j` for all i,j
- Fractal scaling: `distance(p_new, aggregate_cm) ≈ gamma_pc`
- Contact: `exists i: distance(p_new, p_i) = r_new + r_i`

**Advantages:**
- Guaranteed feasible solution (if exists)
- Handles complex constraint combinations

**Disadvantages:**
- Slow for large systems
- Black-box solver (less control)
- May not scale well

**Recommendation:** Stick with current approach but incorporate optimizations. Alternatives are research-level explorations.

---

## 7. Code Quality Improvements

### Profiling Infrastructure

**Add:** Per-function timing instrumentation

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return result
    return wrapper

# Apply to main functions
@profile_function
def run_simulation(...):
    ...
```

**Add:** Operation counters

```python
class PerformanceCounters:
    overlap_checks = 0
    rotation_attempts = 0
    particle_swaps = 0
    candidate_searches = 0
    
    @classmethod
    def reset(cls):
        cls.overlap_checks = 0
        cls.rotation_attempts = 0
        ...
    
    @classmethod
    def report(cls):
        print(f"Overlap checks: {cls.overlap_checks:,}")
        print(f"Rotation attempts: {cls.rotation_attempts:,}")
        ...
```

### Configuration Constants

**Replace magic numbers with named constants:**

```python
# pyfracval/constants.py
MAX_ROTATION_ATTEMPTS = 360
ADAPTIVE_TOLERANCE_THRESHOLD = 180
STANDARD_OVERLAP_TOLERANCE = 1e-6
RELAXED_OVERLAP_TOLERANCE = 1e-5
FIBONACCI_GOLDEN_RATIO = 1.6180339887
```

### Unit Tests for Critical Paths

**Add tests for:**
- Overlap calculation correctness (boundary cases)
- Rotation geometry (verify contact conditions)
- Incremental Rg updates (vs full recalculation)
- Spatial index correctness (vs linear search)

---

## 8. Implementation Roadmap

### Phase 1: Quick Wins (1 week, 2-3x speedup)

**Week 1:**
- [ ] Implement early termination in overlap checks
- [ ] Add bounding sphere pre-checks
- [ ] Vectorize PCA candidate selection
- [ ] Remove/optimize TRACE logging overhead
- [ ] Add profiling instrumentation
- [ ] Benchmark and validate speedup

**Deliverables:**
- Optimized `utils.py` with fast overlap functions
- Benchmarks showing 2-3x improvement
- Profiling data confirming hotspot elimination

### Phase 2: Algorithmic Improvements (2-3 weeks, 5-10x cumulative)

**Weeks 2-3:**
- [ ] Implement spatial indexing (cKDTree first, custom octree later)
- [ ] Add batch rotation with vectorized overlap
- [ ] Implement incremental Rg calculation
- [ ] Add analytical constraint pruning
- [ ] Comprehensive benchmarking across N=[128, 256, 512, 1024]

**Deliverables:**
- Optimized PCA/CCA implementations
- Scaling analysis showing logarithmic candidate search
- 5-10x total speedup on large N

### Phase 3: Advanced Optimizations (1-2 months, 10-100x for large N)

**Month 2:**
- [ ] GPU acceleration exploration (CuPy proof-of-concept)
- [ ] Hierarchical sticking implementation
- [ ] Parallel subclustering
- [ ] JAX implementation comparison
- [ ] Performance comparison: CPU vs GPU across N scales

**Deliverables:**
- GPU-accelerated overlap calculations
- Scaling benchmarks up to N=10,000
- Documentation for GPU setup

### Phase 4: Production Hardening (ongoing)

- [ ] Comprehensive unit tests (>80% coverage)
- [ ] Performance regression tests
- [ ] Documentation updates
- [ ] API stability
- [ ] Continuous benchmarking infrastructure

---

## 9. Risk Assessment

### Low Risk (Phase 1)
- **Early termination:** No algorithmic change, just optimization
- **Bounding checks:** Well-understood geometry
- **Vectorization:** Proven technique

### Medium Risk (Phase 2)
- **Spatial indexing:** Complexity in tree updates, edge cases
- **Batch rotation:** Requires careful validation of convergence
- **Incremental Rg:** Numerical stability concerns

### High Risk (Phase 3)
- **GPU acceleration:** Platform dependency, memory constraints
- **Parallel subclustering:** Reproducibility with random seeds
- **Algorithm changes:** May affect aggregate properties (needs validation)

---

## 10. Validation Strategy

For EVERY optimization:

1. **Correctness:** Compare outputs with original implementation
   - Same seed → same aggregate (bit-exact)
   - Visual inspection of aggregate geometry
   - Statistical properties (Rg, fractal dimension)

2. **Performance:** Benchmark with multiple N values
   - Runtime comparison
   - Scaling analysis (plot runtime vs N)
   - Operation counts

3. **Success Rate:** Ensure optimizations don't reduce robustness
   - Run parameter sweeps from benchmarks
   - Compare success rates before/after

4. **Regression Tests:** Add automated tests
   - Known-good test cases
   - Performance thresholds

---

## 11. Conclusion

PyFracVAL has significant optimization potential without compromising algorithmic integrity. The recommended approach is **incremental optimization**:

1. **Start with Phase 1** (quick wins, low risk, 2-3x speedup)
2. **Validate thoroughly** before proceeding
3. **Move to Phase 2** only after Phase 1 is production-ready
4. **Phase 3 is optional** - pursue only if large N (>1000) is a priority

**Expected Total Speedup:**
- Phase 1: 2-3x
- Phase 1 + 2: 5-10x
- Phase 1 + 2 + 3: 10-100x (N-dependent, with GPU)

**Effort vs Reward:**
- Phase 1: Best ROI (1 week for 2-3x)
- Phase 2: Good ROI (3 weeks for 5-10x)
- Phase 3: Research-level (months for 10-100x, hardware-dependent)

The current implementation is already **production-ready** for typical use cases (N ≤ 512). Optimizations are **enhancements**, not **necessities**.
