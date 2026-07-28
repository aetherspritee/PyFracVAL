# Usage

This guide covers the two ways to run PyFracVAL: the command-line interface,
and the `run_simulation` function as a library call. Both implement the
PCA/CCA workflow of {cite:p}`Moran2019FracVAL`, with morphology conventions
from {cite:p}`Filippov2000Tunable`.

## Command-line interface

Generating a single aggregate with default parameters:

```bash
pyfracval
```

This writes one `.dat` file to a `RESULTS/` subdirectory. Parameters are set
via flags, for example an aggregate of 512 particles with `Df=1.9`,
`kf=1.4`, and polydisperse primary particles (`rp_g=50`, `rp_gstd=1.25`):

```bash
pyfracval -n 512 --df 1.9 --kf 1.4 --rp-g 50 --rp-gstd 1.25
```

### Options

| Flag | Meaning |
|---|---|
| `-n`, `--num-particles` | Total number of primary particles (N). |
| `--df` | Target fractal dimension. |
| `--kf` | Target fractal prefactor. |
| `--rp-g` | Geometric mean radius of primary particles. |
| `--rp-gstd` | Geometric standard deviation of radii (>= 1.0). Takes precedence over `--rp-std` if both are given. |
| `--rp-std` | Approximate arithmetic standard deviation of radii, used to estimate `--rp-gstd` via `exp(std/mean)` when `--rp-gstd` is not given. The estimated value is logged as a warning. |
| `--ext-case` | CCA sticking geometry variant (0 or 1). |
| `--tol-ov` | Overlap tolerance for particle sticking. |
| `--n-subcl-perc` | Target fraction of N for PCA subcluster size. |
| `--num-aggregates` | Number of aggregates to generate sequentially. |
| `-f`, `--folder` | Output directory (default: `RESULTS`). |
| `--seed` | Random seed for reproducible generation. |
| `--max-attempts` | Maximum retry attempts per aggregate if generation fails (default: 5). |
| `--config` | Path to a TOML/YAML/JSON config file. Algorithm-tuning options not exposed as flags (retry modes, densification, etc.) are only settable this way - see the [experiments retrospective](experiments.md) for what is available. Explicit flags override the corresponding config value. |
| `-p`, `--plot` | Display the generated aggregate(s) interactively via PyVista. |
| `-v`, `-vv`, `-vvv` | Increase logging verbosity (INFO, DEBUG, TRACE). |
| `--log-file` | Redirect log output to a file. |
| `-h`, `--help` | List all options and their current defaults. |

`--rp-gstd` versus `--rp-std`:

```bash
# --rp-std is estimated into a geometric standard deviation; check -vv output
# for the WARNING that reports the value actually used.
pyfracval -n 200 --df 1.9 --kf 1.2 --rp-g 20 --rp-std 5 -vv

# --rp-gstd, when given explicitly, takes precedence over --rp-std.
pyfracval -n 100 --df 1.8 --rp-gstd 1.3 --rp-std 5
```

Generating multiple aggregates with plots shown afterward:

```bash
pyfracval -n 100 --df 1.7 --kf 1.1 --num-aggregates 3 -p
```

### Streamlit explorer

```bash
pyfracval explore
```

launches a Streamlit app (`pyfracval/app.py`) for interactively browsing
generated aggregates; requires the `plot` dependency group.

## Library usage

```python
import numpy as np
from pathlib import Path
from pyfracval.main_runner import run_simulation
from pyfracval.visualization import plot_particles

sim_config = {
    "N": 128,
    "Df": 1.8,
    "kf": 1.3,
    "rp_g": 10.0,
    "rp_gstd": 1.2,
    "tol_ov": 1e-4,
    "n_subcl_percentage": 0.15,
    "ext_case": 0,
}

output_directory = Path("./my_aggregates")

success, final_coords, final_radii = run_simulation(
    iteration=1,
    sim_config_dict=sim_config,
    output_base_dir=str(output_directory),
    seed=42,
)

if success:
    print(f"Generated {final_coords.shape[0]} particles in {output_directory}")
    center_of_mass = np.mean(final_coords, axis=0)

    plotter = plot_particles(final_coords, final_radii)
    plotter.add_text(
        f"N={final_coords.shape[0]}, Df={sim_config['Df']}, kf={sim_config['kf']}",
        position="upper_left",
    )
    plotter.show()
```

`run_simulation` also accepts `max_runtime_seconds` to bound the worst-case
wall-clock time spent retrying a difficult parameter combination. See the
[API reference](autoapi/pyfracval/main_runner/index) for the full signature.
