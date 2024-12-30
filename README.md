<p align="center">
  <img src="https://github.com/aetherspritee/PyFracVAL/blob/main/.github/logo.png?raw=true" alt="RefIdxDB-Logo"/>
</p>

# PyFracVAL

Implementation of FracVAL by [Morán, J. et al. 2019](https://www.sciencedirect.com/science/article/pii/S0010465519300323?via%3Dihub) in python.
Source files for the original FracVAL Fortran implementation can be found [here](https://data.mendeley.com/datasets/mgf8wdcsfb/1).

## Development

### Using UV

Create an environment (`numba` does not support python 3.13 yet, so please use 3.12 or lower):

```sh
uv venv --python 3.12
```

Sync the packages with the dependencies found in `pyproject.toml`:

```sh
uv sync
```

## TODO

- [ ] Fix PCA issues
  - [x] Instability with arcos arguments, tied to monomer size somehow
  - [ ] low fractal dimensions fail -> This is also happening for original FracVAL
  - [ ] high fractal prefactors fail -> This is also happening for original FracVAL
  - [x] yet to be understood infinite loop occurs sometimes
- [ ] Allow distribution functions for monomer radii
- [ ] Parallelize
- [ ] High fractal dimensions are super slow -> Happens for FracVAL aswell, bit more noticeable for this implementation i feel
