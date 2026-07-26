# GPU Acceleration: Why We're Not Porting to JAX

This page documents a 2026-07-26 evaluation of whether PyFracVAL's numba-jitted
hot path would benefit from a JAX/GPU port. Short answer: **no** - numba's
existing CPU kernels beat JAX (CPU and GPU) by 1-4 orders of magnitude at
every problem size PyFracVAL actually operates at, and the gap does not close
as problems get larger. The reasoning below explains why, with real numbers
from this machine's GPU (NVIDIA TITAN X, Pascal, compute capability 6.1).

## What was surveyed

Every `@jit`-decorated function in `pyfracval/` was catalogued (kernels in
`overlap.py`, `geometry.py`, `pca_kernels.py`, `cca_kernels.py`, `densify.py`,
and `experimental/`). The numerical hot path concentrates in two places:

- **`overlap.py`** - pairwise overlap/distance checks between two point sets,
  called up to ~18,000 times per aggregate (once per rotation attempt during
  PCA/CCA sticking).
- **`geometry._rodrigues_rotation_2d`** - rotates an (N,3) cluster around an
  axis, called once per rotation attempt alongside the overlap check.

Both are excellent candidates for GPU acceleration *on paper*: they're
numerical array operations, called extremely frequently, on a problem that in
principle scales with the number of particles. That's exactly the profile a
JAX/GPU port usually targets.

## Setup: making the GPU actually visible

`devenv.nix` now mirrors `../YASF-new/devenv.nix`'s CUDA setup: a
`cudaPackages_12` toolkit via `symlinkJoin`, `CUDA_HOME`, `NUMBA_CUDA_DRIVER`
pointing at `/run/opengl-driver/lib/libcuda.so`, and (new) `/run/opengl-driver/lib`
itself appended to `LD_LIBRARY_PATH` - JAX's CUDA plugin needs the driver
directory on the standard dynamic-linker search path; numba doesn't, since it
reads `NUMBA_CUDA_DRIVER` directly. `devenv.yaml` gained `allow_unfree: true`
(the CUDA toolkit's EULA is unfree). Both `NUMBA_CACHE_DIR` and
`JAX_COMPILATION_CACHE_DIR` point at `.devenv/state/`, so compiled kernels
persist across process restarts for both frameworks.

`jax[cuda12]` is **not** auto-synced (it pulls ~1.5GB of CUDA wheels -
cudnn, nccl, nvshmem) - install it explicitly with
`uv sync --group test --group docs --group jax_bench` (all three groups
together, since `uv sync --group X` alone replaces rather than extends the
synced set).

With this in place:

```
>>> import jax; jax.devices()
[CudaDevice(id=0)]
```

## Benchmark methodology

`benchmarks/jax_kernels.py` ports the two hot-path kernels to JAX
(`rodrigues_rotation_jax`, `max_overlap_pairwise_jax`,
`max_overlap_single_vs_agg_jax`, `batch_check_overlaps_pca_jax` - the last one
mirrors `pca_kernels.batch_check_overlaps_pca`, batching K candidate rotations
into a single call, which is the shape JAX/GPU is actually suited for).
`jax_enable_x64` is on (PyFracVAL uses float64 throughout; JAX defaults to
float32, which would silently change simulation precision).

`benchmarks/jax_vs_numba_benchmark.py` times numba vs JAX-CPU vs JAX-GPU for
each kernel across a size sweep. Each kernel/framework/size combination is
warmed up once (to pay JIT-compile cost, excluded from the timed loop) then
timed over 50 calls with **fresh random data each call** - this matches how
these kernels are actually invoked in the simulation (coordinates and radii
differ on every rotation attempt, so nothing can be kept resident on the GPU
across calls without a much larger rewrite of the sticking loop itself).
JAX timings include host↔device transfer and `block_until_ready()` per call,
since the caller needs the concrete result back before the next step of the
algorithm - not a benchmark artifact, the real usage pattern. Raw results:
`benchmark_results/jax_vs_numba_summary.json`.

## Results

### Rodrigues rotation (no early-exit, purely vectorized - JAX's best case)

| N particles | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|
| 16 | 6.6 µs | 11.5 µs | 62.6 µs |
| 64 | 7.0 µs | 11.5 µs | 60.4 µs |
| 256 | 8.0 µs | 12.2 µs | 45.4 µs |
| 1,024 | 10.7 µs | 15.9 µs | 46.5 µs |
| 4,096 | 20.9 µs | 29.6 µs | 59.9 µs |
| 8,192 | 34.1 µs | 54.2 µs | 58.6 µs |

Even here - a kernel with no branching, nothing for JAX to lose out on
algorithmically - numba wins at every size tested. JAX-GPU's ~45-60 µs floor
is dispatch + kernel-launch + PCIe transfer latency; it barely moves with N
because at these sizes the actual compute is negligible next to that fixed
cost. numba's cached, compiled-to-native-code dispatch has no such floor.

### Pairwise overlap (numba's early-exit + bounding-sphere pre-check vs. JAX's dense O(N²) matrix)

| N per cluster | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|
| 16 | 1.0 µs | 13.2 µs | 50.6 µs |
| 64 | 0.8 µs | 68.1 µs | 55.8 µs |
| 256 | 1.4 µs | 438.2 µs | 120.1 µs |
| 1,024 | 3.5 µs | 2,452 µs | 412 µs |
| 4,096 | 32.3 µs | 45,647 µs | 3,609 µs |

This is where the gap becomes dramatic: **up to 3 orders of magnitude** at
N=4,096. numba's `calculate_max_overlap_cca_fast` does a cheap
squared-distance-vs-squared-radius-sum comparison before ever computing a
`sqrt`, skipping the expensive math for any pair that clearly can't overlap
- and returns immediately the moment it finds one pair that does. JAX/XLA has
no equivalent: on SIMD/GPU hardware, a per-element `where` doesn't skip work,
it computes both branches for every lane and selects - so JAX must always
materialize the full N×N distance matrix and reduce it, unconditionally.

Two follow-up checks confirm this is architectural, not a tuning gap:

- **No crossover, even far past realistic sizes.** Sweeping the sparse/scattered
  scenario up to N=32,768 (32,768² ≈ 1.07 billion pairs), numba stayed at
  67-200 µs the whole way (the pre-check keeps skipping nearly everything),
  while JAX-GPU grew to 165 ms - the gap *widens* with N, it doesn't close.
- **Densely-packed data (numba's early-*return* fires almost instantly)**
  makes the gap worse for JAX, not better: numba drops to ~0.5-0.7 µs
  (returns on the very first overlapping pair it finds) while JAX-GPU still
  pays 100 µs - 2.9 ms for N=256-4,096, since it always computes the full
  matrix regardless of how quickly a CPU scan would have stopped.

There is no data distribution in this problem where JAX wins: sparse data
lets numba's pre-check skip almost everything; dense data lets its early
return fire almost immediately. JAX always pays the full O(N²) cost either
way.

### Batched overlap check (K candidate rotations per call - JAX's fairest shot)

Mirrors `pca_kernels.batch_check_overlaps_pca`: check K candidate positions
against one aggregate in a single call, amortizing per-call dispatch/launch
overhead over K. This is the shape that should favor JAX/GPU most.

| N_agg | K | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|---:|
| 64 | 1 | 5.8 µs | 13.5 µs | 43.2 µs |
| 64 | 360 | 9.0 µs | 219.9 µs | 60.2 µs |
| 512 | 360 | 29.9 µs | 640.2 µs | 131.4 µs |
| 2,048 | 1 | 10.3 µs | 40.8 µs | 50.6 µs |
| 2,048 | 360 | 100.5 µs | 1,532.6 µs | 332.4 µs |

Batching genuinely helps JAX's *relative* standing - the numba/JAX-GPU ratio
narrows from ~7x (K=1) to ~3.3x (K=360, N_agg=2,048), the best result JAX
gets anywhere in this evaluation. It's still a clear loss in absolute terms.
numba's `batch_check_overlaps_pca` is itself already parallelized
(`prange` across candidates) and cached, so it isn't leaving much on the
table for JAX to beat.

## Caching: confirmed working, doesn't change the conclusion

Both frameworks' persistent caches were verified directly (empty cache dir
vs. warm cache dir, same kernel, same input size):

| Framework | Cold compile (empty cache) | Cache hit (warm `.devenv/state/`) |
|---|---:|---:|
| numba (`cache=True`) | 501 ms | 299 ms |
| JAX (`jax_compilation_cache_dir`) | 155 ms | 19 ms |

JAX's cache gives a much larger relative win (~8x) since XLA compilation is
inherently heavier than numba's LLVM path for these small kernels, plus a
one-time ~300 ms CUDA context/backend initialization cost per process
(unrelated to caching - pure driver/runtime bring-up, paid once regardless).
Both are working as intended and both matter for keeping short-lived
processes (a benchmark run, a single CLI invocation) fast to start. Neither
changes the steady-state numbers above, which is what actually matters here:
PyFracVAL's simulation loop runs each kernel thousands of times per
aggregate in a single long-lived process, so warm-cache startup cost is
amortized to irrelevance either way - the per-call dispatch/launch overhead
that caching *can't* fix is what decides the outcome.

## Conclusion

**Not pursuing a JAX/GPU port of `overlap.py` or `geometry.rodrigues_rotation`.**
The existing numba implementation is already close to optimal for this
problem's actual shape: many small, branch-heavy, early-exit-friendly
pairwise geometry computations, called in a tight sequential loop, on modest
per-call array sizes (tens to low-thousands of particles). That's a
textbook case for a well-tuned CPU kernel, not for a SIMD/GPU one - branching
that skips work is exactly what GPU-style parallelism can't exploit, and the
fixed per-call dispatch/launch/transfer floor (tens of microseconds minimum,
regardless of framework maturity or caching) dominates every problem size
PyFracVAL actually generates aggregates at.

This mirrors the project's [CCA sticking retrospective](experiments.md):
another case where a plausible-sounding "throw more compute at it" idea
doesn't move the needle, because the real constraint isn't raw throughput -
it's the shape of the search itself. The devenv CUDA setup and the JAX
kernel ports are kept (`benchmarks/jax_kernels.py`,
`benchmarks/jax_vs_numba_benchmark.py`) in case a different angle - a
genuinely large batch of independent aggregates generated in parallel on the
GPU, rather than accelerating one aggregate's inner rotation loop - turns out
to be worth revisiting later.
