# GPU Acceleration Evaluation: JAX vs. Numba

This page documents a 2026-07-26 evaluation of whether PyFracVAL's
numba-jitted hot path would benefit from a JAX/GPU port. The conclusion
is negative: the existing numba CPU kernels outperform JAX (both CPU
and GPU) by one to four orders of magnitude at every problem size
PyFracVAL operates at, and the gap does not close as problems grow.
Measurements were taken on an NVIDIA TITAN X (Pascal, compute
capability 6.1).

## Scope

Every `@jit`-decorated function in `pyfracval/` was catalogued (kernels
in `overlap.py`, `geometry.py`, `pca_kernels.py`, `cca_kernels.py`,
`densify.py`, and `experimental/`). The numerical hot path concentrates
in two places:

- `overlap.py`: pairwise overlap/distance checks between two point
  sets, called up to ~18,000 times per aggregate (once per rotation
  attempt during PCA/CCA sticking);
- `geometry._rodrigues_rotation_2d`: rotation of an (N,3) cluster
  around an axis, called once per rotation attempt alongside the
  overlap check.

Both are, on their face, reasonable candidates for GPU acceleration:
numerical array operations, called frequently, on a problem that
nominally scales with particle count — the profile a JAX/GPU port
typically targets.

## Method

`benchmarks/jax_kernels.py` ports the two hot-path kernels to JAX
(`rodrigues_rotation_jax`, `max_overlap_pairwise_jax`,
`max_overlap_single_vs_agg_jax`, `batch_check_overlaps_pca_jax` — the
last mirrors `pca_kernels.batch_check_overlaps_pca`, batching K
candidate rotations into a single call, the shape best suited to
JAX/GPU). `jax_enable_x64` is enabled, since PyFracVAL uses float64
throughout and JAX's float32 default would silently change simulation
precision.

`benchmarks/jax_vs_numba_benchmark.py` times numba against JAX-CPU and
JAX-GPU for each kernel across a size sweep. Each
kernel/framework/size combination is warmed up once (JIT-compile cost,
excluded from the timed loop) and then timed over 50 calls with fresh
random data per call, matching how the kernels are invoked in the
simulation: coordinates and radii differ on every rotation attempt, so
nothing can remain resident on the GPU across calls without a
substantially larger rewrite of the sticking loop. JAX timings include
host–device transfer and `block_until_ready()` per call, since the
caller needs the concrete result before the next step of the algorithm;
this reflects actual usage rather than a benchmark artifact. Raw
results: `benchmark_results/jax_vs_numba_summary.json`.

## Results

### Rodrigues rotation

No early exit, purely vectorized — the case most favorable to JAX:

| N particles | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|
| 16 | 6.6 µs | 11.5 µs | 62.6 µs |
| 64 | 7.0 µs | 11.5 µs | 60.4 µs |
| 256 | 8.0 µs | 12.2 µs | 45.4 µs |
| 1,024 | 10.7 µs | 15.9 µs | 46.5 µs |
| 4,096 | 20.9 µs | 29.6 µs | 59.9 µs |
| 8,192 | 34.1 µs | 54.2 µs | 58.6 µs |

Even for this kernel, with no branching and nothing for JAX to lose
algorithmically, numba is faster at every size tested. JAX-GPU's
~45–60 µs floor reflects dispatch, kernel-launch, and PCIe transfer
latency; it changes little with N because at these sizes the compute is
negligible relative to that fixed cost. numba's cached
compiled-to-native dispatch has no equivalent floor.

### Pairwise overlap

numba's early exit and bounding-sphere pre-check against JAX's dense
O(N²) matrix:

| N per cluster | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|
| 16 | 1.0 µs | 13.2 µs | 50.6 µs |
| 64 | 0.8 µs | 68.1 µs | 55.8 µs |
| 256 | 1.4 µs | 438.2 µs | 120.1 µs |
| 1,024 | 3.5 µs | 2,452 µs | 412 µs |
| 4,096 | 32.3 µs | 45,647 µs | 3,609 µs |

The gap widens here, reaching roughly three orders of magnitude at
N=4,096. numba's `calculate_max_overlap_cca_fast` performs a cheap
squared-distance vs. squared-radius-sum comparison before computing any
`sqrt`, skipping the expensive arithmetic for pairs that cannot
overlap, and returns as soon as it finds one pair that does. JAX/XLA
has no equivalent: on SIMD/GPU hardware a per-element `where` does not
skip work — both branches are computed for every lane — so JAX must
materialize the full N×N distance matrix and reduce it unconditionally.

Two follow-up checks indicate the gap is architectural rather than a
tuning artifact:

- No crossover exists even far past realistic sizes. Sweeping the
  sparse/scattered scenario up to N=32,768 (≈1.07 billion pairs), numba
  stays at 67–200 µs throughout (the pre-check continues to skip nearly
  everything) while JAX-GPU grows to 165 ms; the gap widens with N.
- Densely-packed data, where numba's early return fires almost
  immediately, widens the gap further: numba drops to ~0.5–0.7 µs while
  JAX-GPU still pays 100 µs–2.9 ms for N=256–4,096, computing the full
  matrix regardless.

No data distribution in this problem favors JAX: sparse data lets the
pre-check skip almost everything, and dense data lets the early return
fire almost immediately, while JAX pays the full O(N²) cost in either
case.

### Batched overlap check

K candidate rotations per call, mirroring
`pca_kernels.batch_check_overlaps_pca` — the comparison expected to
favor JAX/GPU most, since per-call dispatch overhead is amortized
over K:

| N_agg | K | numba | JAX (CPU) | JAX (GPU) |
|---:|---:|---:|---:|---:|
| 64 | 1 | 5.8 µs | 13.5 µs | 43.2 µs |
| 64 | 360 | 9.0 µs | 219.9 µs | 60.2 µs |
| 512 | 360 | 29.9 µs | 640.2 µs | 131.4 µs |
| 2,048 | 1 | 10.3 µs | 40.8 µs | 50.6 µs |
| 2,048 | 360 | 100.5 µs | 1,532.6 µs | 332.4 µs |

Batching improves JAX's relative standing: the numba/JAX-GPU ratio
narrows from ~7× (K=1) to ~3.3× (K=360, N_agg=2,048), JAX's closest
result in this evaluation, though still a clear loss in absolute terms.
numba's `batch_check_overlaps_pca` is itself parallelized (`prange`
across candidates) and cached, leaving little margin to close.

### Compilation caching

Both frameworks' persistent caches were verified directly (empty vs.
warm cache directory, same kernel, same input size):

| Framework | Cold compile (empty cache) | Cache hit (warm `.devenv/state/`) |
|---|---:|---:|
| numba (`cache=True`) | 501 ms | 299 ms |
| JAX (`jax_compilation_cache_dir`) | 155 ms | 19 ms |

JAX's cache produces the larger relative improvement (~8×), since XLA
compilation is heavier than numba's LLVM path for these small kernels;
JAX additionally pays a one-time ~300 ms CUDA context initialization
per process, unrelated to caching. Both caches function as intended and
matter for the startup latency of short-lived processes (a benchmark
run, a single CLI invocation). Neither changes the steady-state figures
above, which determine the outcome: the simulation loop runs each
kernel thousands of times per aggregate within one long-lived process,
so warm-cache startup cost is amortized away, and the per-call
dispatch/launch overhead that caching does not address decides the
comparison.

## Conclusion

A JAX/GPU port of `overlap.py` or `geometry.rodrigues_rotation` is not
being pursued. The existing numba implementation is close to optimal
for this problem's shape: many small, branch-heavy,
early-exit-friendly pairwise geometry computations, called in a tight
sequential loop, on modest per-call array sizes. Branching that skips
work is precisely what GPU-style parallelism cannot exploit, and the
fixed per-call dispatch/launch/transfer floor (tens of microseconds,
independent of framework maturity or caching) dominates at every
problem size PyFracVAL generates aggregates at.

The result parallels the CCA sticking retrospective
([experiments.md](experiments.md)): a second case in which additional
compute does not improve the outcome, because the binding constraint is
not throughput but the structure of the problem. The devenv CUDA setup
and the JAX kernel ports are retained (`benchmarks/jax_kernels.py`,
`benchmarks/jax_vs_numba_benchmark.py`) in case a different
parameterization — generating a large batch of independent aggregates
in parallel on the GPU, rather than accelerating one aggregate's inner
rotation loop — is revisited later.

## Implementation notes

`devenv.nix` mirrors `../YASF-new/devenv.nix`'s CUDA setup: a
`cudaPackages_12` toolkit via `symlinkJoin`, `CUDA_HOME`, and
`NUMBA_CUDA_DRIVER` pointing at `/run/opengl-driver/lib/libcuda.so`,
with `/run/opengl-driver/lib` appended to `LD_LIBRARY_PATH` (JAX's CUDA
plugin requires the driver directory on the standard dynamic-linker
search path; numba does not, since it reads `NUMBA_CUDA_DRIVER`
directly). `devenv.yaml` sets `allow_unfree: true` for the CUDA
toolkit's license. Both `NUMBA_CACHE_DIR` and
`JAX_COMPILATION_CACHE_DIR` point at `.devenv/state/`, so compiled
kernels persist across process restarts for both frameworks.

`jax[cuda12]` is not synced by default, since it pulls approximately
1.5 GB of CUDA wheels (cudnn, nccl, nvshmem); install it explicitly
with `uv sync --group test --group docs --group jax_bench` (all three
groups together, since `uv sync --group X` alone replaces rather than
extends the synced set). With this in place:

```
>>> import jax; jax.devices()
[CudaDevice(id=0)]
```
