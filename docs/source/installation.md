# Installation

## Requirements

- Python 3.11 or later.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- [`devenv`](https://devenv.sh/) (Nix-based), for the reproducible development
  environment used by this project. Not required to use PyFracVAL as a
  library.

PyFracVAL is not yet published on PyPI; installation is from source.

## Development environment

The project pins its toolchain (Python interpreter, `uv`, CUDA libraries for
the optional GPU benchmarks) through `devenv.nix`. This is the supported
setup for contributing to PyFracVAL:

```bash
git clone https://github.com/aetherspritee/PyFracVAL.git
cd PyFracVAL
devenv shell
```

`devenv shell` provisions the pinned Python interpreter and installs the
project along with its `test` and `docs` dependency groups. Run commands
inside this shell, or prefix them with `devenv shell --`:

```bash
devenv shell -- uv run pytest
devenv shell -- uv run python pyfracval/main_runner.py
```

Two additional dependency groups are available and installed on demand:

- `plot` (`pyvista`, `streamlit`, `pandas`, `matplotlib`) - interactive 3D
  visualization and the Streamlit exploration app.
- `jax_bench` - the JAX/CUDA benchmark harness described in the
  [GPU acceleration evaluation](gpu_acceleration.md); not synced by default,
  since it pulls several gigabytes of CUDA wheels.

```bash
uv sync --group test --group docs --group plot
```

## Without devenv

`devenv` is not required to use the package. A plain `uv` or `pip` install
from a clone of the repository works as well:

```bash
git clone https://github.com/aetherspritee/PyFracVAL.git
cd PyFracVAL
uv sync
# or: pip install -e .
```

## Verifying the installation

```bash
python -c "import pyfracval; print(pyfracval.__version__)"
```
