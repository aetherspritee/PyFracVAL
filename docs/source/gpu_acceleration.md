# GPU Acceleration Evaluation: JAX vs. Numba

This page documents a 2026-07-26 evaluation of whether PyFracVAL's
numba-jitted hot path would benefit from a JAX/GPU port. The conclusion is
negative: numba's existing CPU kernels outperform JAX (both CPU and GPU)
by one to four orders of magnitude at every problem size PyFracVAL
operates at, and the gap does not close as problems grow larger. The
sections below give the reasoning, with measurements from this machine's
GPU (NVIDIA TITAN X, Pascal, compute capability 6.1).

## Scope of the evaluation

Every `@jit`-decorated function in `pyfracval/` was catalogued (kernels in
`overlap.py`, `geometry.py`, `pca_kernels.py`, `cca_kernels.py`,
`densify.py`, and `experimental/`). The numerical hot path concentrates
in two places:

- `overlap.py`: pairwise overlap/distance checks between two point sets,
  called up to ~18,000 times per aggregate (once per rotation attempt
  during PCA/CCA sticking).
- `geometry._rodrigues_rotation_2d`: rotates an (N,3) cluster around an
  axis, called once per rotation attempt alongside the overlap check.

Both are, in principle, reasonable candidates for GPU acceleration: they
are numerical array operations, called frequently, on a problem that
nominally scales with particle count - the profile a JAX/GPU port
typically targets.

## Environment setup

`devenv.nix` mirrors `../YASF-new/devenv.nix`'s CUDA setup: a
`cudaPackages_12` toolkit via `symlinkJoin`, `CUDA_HOME`, and
`NUMBA_CUDA_DRIVER` pointing at `/run/opengl-driver/lib/libcuda.so`, with
`/run/opengl-driver/lib` appended to `LD_LIBRARY_PATH` (JAX's CUDA plugin
requires the driver directory on the standard dynamic-linker search path;
numba does not, since it reads `NUMBA_CUDA_DRIVER` directly).
`devenv.yaml` sets `allow_unfree: true`, since the CUDA toolkit's license
is non-free. Both `NUMBA_CACHE_DIR` and `JAX_COMPILATION_CACHE_DIR` point
at `.devenv/state/`, so compiled kernels persist across process restarts
for both frameworks.

`jax[cuda12]` is not synced by default, since it pulls approximately
1.5GB of CUDA wheels (cudnn, nccl, nvshmem); install it explicitly with
`uv sync --group test --group docs --group jax_bench` (all three groups
together, since `uv sync --group X` alone replaces rather than extends
the synced set).

With this in place:

```
>>> import jax; jax.devices()
[CudaDevice(id=0)]
```

## Benchmark methodology

`benchmarks/jax_kernels.py` ports the two hot-path kernels to JAX
(`rodrigues_rotation_jax`, `max_overlap_pairwise_jax`,
`max_overlap_single_vs_agg_jax`, `batch_check_overlaps_pca_jax` - the
last mirrors `pca_kernels.batch_check_overlaps_pca`, batching K candidate
rotations into a single call, the shape JAX/GPU is best suited for).
`jax_enable_x64` is enabled, since PyFracVAL uses float64 throughout and
JAX defaults to float32, which would silently change simulation
precision.

`benchmarks/jax_vs_numba_benchmark.py` times numba against JAX-CPU and
JAX-GPU for each kernel across a size sweep. Each kernel/framework/size
combination is warmed up once (to pay JIT-compile cost, excluded from the
timed loop) and then timed over 50 calls with fresh random data each
call, matching how these kernels are invoked in the simulation
(coordinates and radii differ on every rotation attempt, so nothing can
remain resident on the GPU across calls without a substantially larger
rewrite of the sticking loop). JAX timings include host-device transfer
and `block_until_ready()` per call, since the caller needs the concrete
result before the next step of the algorithm; this reflects actual usage
rather than a benchmark artifact. Raw results:
`benchmark_results/jax_vs_numba_summary.json`.

## Results

### Rodrigues rotation (no early exit, purely vectorized - JAX's most favorable case)

| N particles | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|
| 16 | 6.6 µs | 11.5 µs | 62.6 µs |
| 64 | 7.0 µs | 11.5 µs | 60.4 µs |
| 256 | 8.0 µs | 12.2 µs | 45.4 µs |
| 1,024 | 10.7 µs | 15.9 µs | 46.5 µs |
| 4,096 | 20.9 µs | 29.6 µs | 59.9 µs |
| 8,192 | 34.1 µs | 54.2 µs | 58.6 µs |

Even here, a kernel with no branching and nothing for JAX to lose out on
algorithmically, numba is faster at every size tested. JAX-GPU's
~45-60 µs floor reflects dispatch, kernel-launch, and PCIe transfer
latency; it changes little with N because at these sizes the compute
itself is negligible relative to that fixed cost. numba's cached,
compiled-to-native-code dispatch has no equivalent floor.

### Pairwise overlap (numba's early-exit and bounding-sphere pre-check vs. JAX's dense O(N²) matrix)

| N per cluster | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|
| 16 | 1.0 µs | 13.2 µs | 50.6 µs |
| 64 | 0.8 µs | 68.1 µs | 55.8 µs |
| 256 | 1.4 µs | 438.2 µs | 120.1 µs |
| 1,024 | 3.5 µs | 2,452 µs | 412 µs |
| 4,096 | 32.3 µs | 45,647 µs | 3,609 µs |

The gap widens substantially here, reaching roughly three orders of
magnitude at N=4,096. numba's `calculate_max_overlap_cca_fast` performs a
cheap squared-distance-vs-squared-radius-sum comparison before computing
any `sqrt`, skipping the expensive math for pairs that clearly cannot
overlap, and returns as soon as it finds one pair that does. JAX/XLA has
no equivalent: on SIMD/GPU hardware, a per-element `where` does not skip
work, since it computes both branches for every lane and selects between
them, so JAX must materialize the full N×N distance matrix and reduce it
unconditionally.

Two follow-up checks indicate this gap is architectural rather than a
tuning artifact:

- No crossover, even far past realistic sizes. Sweeping the
  sparse/scattered scenario up to N=32,768 (32,768² ≈ 1.07 billion
  pairs), numba stayed at 67-200 µs throughout (the pre-check continues
  to skip nearly everything), while JAX-GPU grew to 165 ms; the gap
  widens with N rather than closing.
- Densely-packed data, where numba's early return fires almost
  immediately, widens the gap further rather than narrowing it: numba
  drops to ~0.5-0.7 µs (returning on the first overlapping pair found)
  while JAX-GPU still pays 100 µs - 2.9 ms for N=256-4,096, since it
  always computes the full matrix regardless of how quickly a CPU scan
  would have stopped.

No data distribution in this problem favors JAX: sparse data lets
numba's pre-check skip almost everything, and dense data lets its early
return fire almost immediately. JAX pays the full O(N²) cost in either
case.

### Batched overlap check (K candidate rotations per call - JAX's most favorable comparison)

Mirrors `pca_kernels.batch_check_overlaps_pca`: check K candidate
positions against one aggregate in a single call, amortizing per-call
dispatch/launch overhead over K. This is the shape expected to favor
JAX/GPU most.

| N_agg | K | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|---:|
| 64 | 1 | 5.8 µs | 13.5 µs | 43.2 µs |
| 64 | 360 | 9.0 µs | 219.9 µs | 60.2 µs |
| 512 | 360 | 29.9 µs | 640.2 µs | 131.4 µs |
| 2,048 | 1 | 10.3 µs | 40.8 µs | 50.6 µs |
| 2,048 | 360 | 100.5 µs | 1,532.6 µs | 332.4 µs |

Batching improves JAX's relative standing: the numba/JAX-GPU ratio
narrows from ~7x (K=1) to ~3.3x (K=360, N_agg=2,048), the closest result
JAX achieves anywhere in this evaluation, though still a clear loss in
absolute terms. numba's `batch_check_overlaps_pca` is itself already
parallelized (`prange` across candidates) and cached, leaving JAX little
margin to close.

## Compilation caching

Both frameworks' persistent caches were verified directly (empty cache
directory vs. warm cache directory, same kernel, same input size):

| Framework | Cold compile (empty cache) | Cache hit (warm `.devenv/state/`) |
|---|---:|---:|
| numba (`cache=True`) | 501 ms | 299 ms |
| JAX (`jax_compilation_cache_dir`) | 155 ms | 19 ms |

JAX's cache produces a larger relative improvement (~8x), since XLA
compilation is inherently heavier than numba's LLVM path for these small
kernels, plus a one-time ~300 ms CUDA context/backend initialization cost
per process, unrelated to caching and paid once regardless. Both caches
function as intended and matter for startup latency of short-lived
processes (a benchmark run, a single CLI invocation). Neither changes the
steady-state figures above, which are what determine the outcome here:
PyFracVAL's simulation loop runs each kernel thousands of times per
aggregate within a single long-lived process, so warm-cache startup cost
is amortized to irrelevance either way. The per-call dispatch/launch
overhead that caching does not address is what decides the comparison.

## Conclusion

A JAX/GPU port of `overlap.py` or `geometry.rodrigues_rotation` is not
being pursued. The existing numba implementation is close to optimal for
this problem's shape: many small, branch-heavy, early-exit-friendly
pairwise geometry computations, called in a tight sequential loop, on
modest per-call array sizes (tens to low thousands of particles). This is
a well-suited case for a tuned CPU kernel rather than a SIMD/GPU one:
branching that skips work is precisely what GPU-style parallelism cannot
exploit, and the fixed per-call dispatch/launch/transfer floor (tens of
microseconds minimum, independent of framework maturity or caching)
dominates every problem size PyFracVAL generates aggregates at.

This parallels the project's [CCA sticking retrospective](experiments.md):
another case where a plausible "more compute" approach does not improve
the outcome, because the binding constraint is not raw throughput but the
shape of the search itself. The devenv CUDA setup and the JAX kernel
ports are retained (`benchmarks/jax_kernels.py`,
`benchmarks/jax_vs_numba_benchmark.py`) in case a different
parameterization - generating a large batch of independent aggregates in
parallel on the GPU, rather than accelerating one aggregate's inner
rotation loop - is worth revisiting later.
