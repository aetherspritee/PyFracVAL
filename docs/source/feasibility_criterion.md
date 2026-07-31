# Predicting the Feasibility Boundary

Morán et al. (2019) state only that FracVAL works "as long as the pair
of Df and kf falls in the valid range where it is possible to generate
such fractal aggregates"; the valid range is left uncharacterized, both
in that paper and in the literature since. Two sweeps in this project
measure it ([hard_regime_boundary_sweep.md](hard_regime_boundary_sweep.md),
[boundary_sweep_v2.md](boundary_sweep_v2.md)). This page describes an
empirical model fitted to those measurements, used to warn up front
that a requested parameter combination is unlikely to succeed rather
than letting the user discover this after twenty internal retries.

## Model

`pyfracval/feasibility.py` evaluates a logistic model of per-trial
success:

```
z = b0 + b1·Df + b2·kf + b3·log10(σ) + b4·log10(N)
       + b5·(Df·kf) + b6·(Df·log10(σ))
P(success) = 1 / (1 + e^-z)
```

Both interaction terms are required, because every sweep found effects
an additive model cannot express:

- **Df × kf** is sign-flipping: at low Df a larger kf helps, at high Df
  a smaller kf helps. A single kf coefficient would average these away.
- **Df × log σ** captures the boundary moving down in Df as
  polydispersity rises — the most consistent finding across all sweeps.

The model is fitted by `benchmarks/fit_feasibility_boundary.py` (plain
gradient ascent on the binomial log-likelihood; the design matrix is
840×7, so no optimizer dependency is needed) against the 4200 trials of
`boundary_sweep_v2`.

## Fit quality

Trial-weighted, on the grid it was trained on:

| Metric | Value |
|---|---|
| Mean absolute error in predicted success rate | 0.035 |
| Brier score | 0.011 |
| Agreement with the measured ≥50% feasibility call | 97.7% |

## Implied Df ceiling

The largest Df with predicted success ≥ 50%, as a function of the
other parameters:

| σ | kf | N=64 | N=256 | N=1024 |
|---|---|---|---|---|
| 1.0 | 0.8 | 2.50 | 2.50 | 2.50 |
| 1.0 | 1.0 | 2.50 | 2.50 | 2.46 |
| 1.0 | 1.4 | 2.50 | 2.41 | 2.31 |
| 1.5 | 0.8 | 2.50 | 2.49 | 2.38 |
| 1.5 | 1.4 | 2.37 | 2.27 | 2.17 |
| 1.9 | 0.8 | 2.50 | 2.40 | 2.29 |
| 1.9 | 1.0 | 2.43 | 2.33 | 2.22 |
| 1.9 | 1.4 | 2.29 | 2.20 | 2.10 |

(2.50 is the top of the fitted grid, not a prediction that generation
fails above it.)

The model reproduces, as a continuous function, the three qualitative
regularities every sweep found: lower kf survives further into high Df,
wider size distributions lower the ceiling, and larger N lowers it
further.

## Use

`run_simulation` calls `feasibility.warn_if_difficult` once per run:

```
WARNING: Requested Df=2.4, kf=1.2, sigma=1.9, N=512 has an estimated
success probability of 0% per attempt. At this kf/sigma/N, Df up to about
2.21 is reliable.
```

The warning is advisory and never blocks. The model is an empirical
fit, and the sweep it derives from itself found success at points
earlier implementations could not reach at all; a low prediction means
"expect this to be slow and to fail often", not "this is impossible".
Requests outside the fitted grid are still evaluated, with the
extrapolation stated:

```
Note this is extrapolation: df=2.9 (fitted range 1.8-2.5),
n=4096 (fitted range 64-1024).
```

## Limitations

The fit is interpolation over one grid, against one set of defaults. It
is not a theory and offers no mechanism for why the boundary sits where
it does; a geometric criterion (comparing the required contact distance
Γ against the available surface shell) would be a stronger result and
remains open. Any change that moves the boundary invalidates the
coefficients.

## Implementation notes

To refit after a boundary-moving change, run
`benchmarks/fit_feasibility_boundary.py` and paste the result into
`feasibility.py::_COEFFS`; the script prints the coefficients in the
form the module expects, along with the fit report above.
