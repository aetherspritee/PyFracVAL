# Profiling and Performance

`benchmarks/profile_pipeline.py` profiles a full `run_simulation` call
rather than a micro-benchmark of one kernel, because the costs that
matter here are distributional: which stage dominates depends on N, on
how often sticking fails, and on how many retries a regime forces. It
rolls `cProfile` output up into the pipeline's own stages, so the report
answers "what should I optimize" rather than only "what is hot".

```
devenv shell -- uv run python benchmarks/profile_pipeline.py --n 1024 --regime medium
```

## What the first run found

Three separate problems, none of them where intuition would have looked
(the overlap kernels, which are already JIT-compiled and were never the
bottleneck).

### 1. `np.cross` on 3-vectors costs 21 µs

`np.cross` supports 2- and 3-component inputs over arbitrary axes with
broadcasting, and pays for that generality on every call. Against three
lines of explicit component arithmetic:

| Operation | Time | Ratio |
|---|---:|---:|
| `np.cross(vec3, vec3)` | 21.0 µs | — |
| explicit `cross3` | 1.7 µs | **12.4x** |
| `np.linalg.norm(vec3)` | 1.28 µs | — |
| `math.sqrt` inline | 0.63 µs | **2.0x** |

The CCA sticking path made ~19,500 `norm` and ~1,550 `cross` calls per
N=512 aggregate, all on plain 3-vectors. `geometry.norm3` / `cross3` now
serve those sites. These are deliberately *not* JIT-compiled: they are
called with plain Python floats and small arrays from interpreted code,
where numba's dispatch overhead would eat the gain.

### 2. The per-aggregate quality record was O(N²) in memory

`quality.max_self_overlap` built an `(N, N, 3)` difference array. It runs
on **every** aggregate, so this was a self-inflicted regression from
adding the quality record:

| N | Before | After (pdist) |
|---:|---:|---:|
| 512 | 12.2 ms | ~0.4 ms |
| 1024 | 49.3 ms | ~1.1 ms |
| 2048 | 211 ms, 101 MB temporary | ~5.5 ms |

`scipy.spatial.distance.pdist` walks the upper triangle directly in C.

### 3. Disabled telemetry was still being computed

`_leaf_mask_for_cluster` is O(n²) per cluster and ran on every sticking
call, as did `_candidate_score` per candidate attempt. Under the
production defaults *neither result is read*: the baseline candidate
policy ignores leaf classification, and the leaf/score counters are only
printed when their `profile_*` flags are set. Both are now computed only
when something actually consumes them — an experimental candidate
policy, a profiling flag, or an attached merge log.

### 4. The sticking placement was scalar glue around fast kernels

`_cca_sticking_v1` ran 2176 times per N=1024 aggregate and accounted for
~26% of the run, almost none of it arithmetic: it drove ~30k `norm`
calls, ~24k `np.array` allocations and ~20k `np.zeros`, all on 3-vectors
and small temporaries between operations that were already fast.

`cca_kernels.cca_sticking_v1_kernel` fuses the whole default
(`ext_case=0`) placement — translate cluster 2, sample the first contact
point, rotate cluster 1, sample the second contact point, rotate cluster
2 — into one compiled function, including both cluster transforms as
tight loops. Its self time dropped **5.8x**, from 133 ms to 23 ms.

Randomness is hoisted out: the two angles the sphere-sphere intersections
would have sampled are passed in as arguments, so the caller keeps sole
ownership of the RNG stream and seeded runs stay reproducible. That also
makes the kernel a pure function, which is what lets
`tests/test_sticking_kernel.py` pin it against the interpreted reference
*exactly* — seeding both identically makes the comparison bit-for-bit
rather than statistical. `ext_case=1` (spherical-cap sampling) still runs
the interpreted path, and `_cca_sticking_v1_interpreted` is kept as the
readable definition of correct behaviour rather than deleted.

## Measured effect

End-to-end medians, 5 seeds per cell, single-threaded:

| Regime | N | Before | After | Speedup |
|---|---:|---:|---:|---:|
| easy | 256 | 130.9 ms | 111.4 ms | 1.18x |
| easy | 512 | 267.4 ms | 203.5 ms | 1.31x |
| easy | 1024 | 498.8 ms | 329.9 ms | **1.51x** |
| medium | 256 | 174.4 ms | 125.2 ms | 1.39x |
| medium | 512 | 450.7 ms | 251.1 ms | **1.80x** |
| medium | 1024 | 998.7 ms | 571.9 ms | **1.75x** |

The gain grows with N and with difficulty, as it should: the O(N²) fixes
scale with size, and the sticking kernel scales with how many candidate
placements a regime forces.

## What did *not* help

Worth recording, so nobody repeats it. Three `logger.trace(f"...")` calls
in the rotation loop evaluate their f-string before `trace()` can check
the level, which is a genuine (and common) bug — but guarding them
produced **no measurable change**, because only ~900 rotations run per
aggregate and a float format is ~1 µs. The guards were kept as correct,
not as an optimization.

That miss came from trusting cProfile's *self time*, which charges some
per-call instrumentation overhead to the calling function. It made
`_perform_cca_sticking`'s loop body look like it held ~47% of the run.
Wall-clock A/B measurement is the arbiter here; the profiler is only good
for pointing at candidates.

Results are unchanged — a seeded run is still bit-identical to itself,
and produces the same overlap-free geometry. (Aggregates generated by
*older* versions at the same seed may differ in the last bits, since
`math.sqrt` of an explicit sum and `np.linalg.norm` need not associate
identically. That shifts a trajectory without changing its statistics.)

## Where the remaining time goes

After the above, the profile at N=1024/medium is dominated by
`_perform_cca_sticking`'s own loop body, with the geometry and sticking
stages having largely disappeared (`_cca_sticking_v1` fell from 14% of
the run to under 4%). Bear the self-time caveat above in mind when
reading that: a loop making thousands of calls is exactly the shape
cProfile over-attributes.

Further gains would mean compiling the whole candidate/rotation loop,
not just the placement inside it. That is harder than the placement was,
because the loop interleaves RNG draws, overlap checks and the retry
policy, and it would have to keep the RNG draw *order* identical to stay
reproducible. It is a real project rather than a tweak, and the payoff is
now smaller: PCA does not register on the profile at all, and a
representative N=1024 aggregate builds in about half a second.
