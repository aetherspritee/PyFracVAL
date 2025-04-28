# Usage Guide

This guide demonstrates the basic ways to use `pyfracval` to generate fractal aggregates.

## Command-Line Interface (CLI)

The primary way to use `pyfracval` is through its command-line interface.

**Basic Generation:**

To generate a single aggregate with default parameters (check defaults using `-h`):

```bash
pyfracval
```

This will create a `.dat` file in a `RESULTS/` subdirectory.

**Specifying Parameters:**

You can control the aggregate properties using command-line options. For example, to generate an aggregate of 512 particles with Df=1.9 and kf=1.4, using polydisperse primary particles (rp_g=50, rp_gstd=1.25):

```bash
pyfracval -n 512 --df 1.9 --kf 1.4 --rp-g 50 --rp-gstd 1.25
```

**Key Options:**

- `-n`, `--num-particles`: Total number of primary particles.
- `--df`: Target fractal dimension.
- `--kf`: Target fractal prefactor.
- `--rp-g`: Geometric mean radius of primary particles.
- `--rp-gstd`: **Geometric** standard deviation of radii (`>= 1.0`). If provided, **takes precedence** over `--rp-std`. If neither is provided, defaults to [Default Value, e.g., 1.5].
- `--rp-std`: Approximate **arithmetic** standard deviation of radii. Used to _estimate_ `--rp-gstd` via `exp(std/mean)` **only if `--rp-gstd` is not given**. A warning will show the estimated geometric value being used.
- `--tol-ov`: Overlap tolerance (e.g., `1e-5`).
- `--n-subcl-perc`: Target fraction for PCA subcluster size (e.g., `0.1`).
- `--num-aggregates`: Generate multiple aggregates sequentially.
- `-f`, `--folder`: Specify the output directory (default: `RESULTS`).
- `--seed`: Set a specific random seed for reproducibility.
- `-p`, `--plot`: Display the generated aggregate(s) interactively using PyVista (requires PyVista installation).
- `-v`, `-vv`, `-vvv`: Increase logging verbosity (INFO, DEBUG, TRACE).
- `--log-file`: Redirect log output to a file.
- `-h`, `--help`: Show all available options and their defaults.

**Example: Using Arithmetic Standard Deviation**

Generate an aggregate with N=200, Df=1.9, kf=1.2, geometric mean radius 20, and an approximate _arithmetic_ standard deviation of 5. `pyfracval` will estimate the geometric standard deviation and use that.

```bash
pyfracval -n 200 --df 1.9 --kf 1.2 --rp-g 20 --rp-std 5 -vv
```

_(Check the log output for a WARNING indicating the calculated `rp_gstd` value being used)._

**Example: Geometric Standard Deviation Takes Precedence**

If you provide both, `--rp-gstd` will be used:

```bash
# rp_gstd=1.3 will be used, rp-std=5 will be ignored (with a warning)
pyfracval -n 100 --df 1.8 --rp-gstd 1.3 --rp-std 5
```

**Example: Generating Multiple Aggregates with Plotting**

Generate 3 aggregates, each with N=100, Df=1.7, kf=1.1, and show the plots:

```bash
pyfracval -n 100 --df 1.7 --kf 1.1 --num-aggregates 3 -p
```

_(Note: Plots are shown sequentially after all aggregates are generated)._

## Using as a Python Library (Programmatic Usage)

You can also import and use the core simulation function directly in your Python scripts.

```python
import numpy as np
from pathlib import Path
from pyfracval.main_runner import run_simulation
from pyfracval.visualization import plot_particles
import pyvista as pv # Requires installation

# 1. Define Simulation Parameters
sim_config = {
    "N": 128,
    "Df": 1.8,
    "kf": 1.3,
    "rp_g": 10.0,
    "rp_gstd": 1.2,
    "tol_ov": 1e-4,
    "n_subcl_percentage": 0.15,
    "ext_case": 0,
    # "seed": 42 # Optional: for reproducibility
}

# 2. Define Output Directory
output_directory = Path("./my_aggregates")

# 3. Run the Simulation
print("Running simulation...")
success, final_coords, final_radii = run_simulation(
    iteration=1,               # Iteration number for saving
    sim_config_dict=sim_config,
    output_base_dir=str(output_directory)
    # seed=sim_config.get("seed") # Or pass seed directly
)

# 4. Check result and process data
if success and final_coords is not None and final_radii is not None:
    print(f"Simulation successful! Aggregate saved in {output_directory}")
    print(f"Generated {final_coords.shape[0]} particles.")

    # Example: Calculate center of mass
    cm = np.mean(final_coords, axis=0)
    print(f"Center of Mass: {cm}")

    # Example: Plot using pyvista
    print("Generating plot...")
    plotter = plot_particles(final_coords, final_radii)
    plotter.add_text(f"N={final_coords.shape[0]}, Df={sim_config['Df']}, kf={sim_config['kf']}", position="upper_left")
    plotter.show()

elif not success:
    print("Simulation failed. Check logs for errors.")
else:
    print("Simulation reported success but returned None data.")

```

This provides a basic structure for using `pyfracval` programmatically.

<!-- Refer to the [API Reference](./api.md) for details on specific functions and classes. -->
