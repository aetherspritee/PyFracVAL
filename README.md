<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!-- Update links with your username and repo name -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url] <!-- Assuming MIT License, change if different -->

<!-- Optional: [![PyPI version](https://img.shields.io/pypi/v/pyfracval.svg?style=for-the-badge)](https://pypi.org/project/pyfracval/) -->
<!-- Optional: [![DOI](https://zenodo.org/badge/DOI/your_doi_here.svg)](https://doi.org/your_doi_here) -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- Make sure the href points to your repo -->
  <a href="https://github.com/aetherspritee/PyFracVAL">
    <!-- Use the raw image link from your current README -->
    <img src="https://github.com/aetherspritee/PyFracVAL/blob/main/.github/logo.png?raw=true" alt="PyFracVAL Logo" width="80" height="80">
  </a>

<h3 align="center">PyFracVAL</h3>

  <p align="center">
    A Python implementation of the FracVAL algorithm for generating 3D fractal-like aggregates with tunable properties (Df, kf) from mono- or polydisperse primary particles using Particle-Cluster and Cluster-Cluster Aggregation. Based on the work of Morán, J. et al. (2019).
    <br />
    <!-- Update the URL to point to your documentation once hosted -->
    <a href="https://[your_docs_host_or_github_pages_url]"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/aetherspritee/PyFracVAL">View Demo</a> --> <!-- Uncomment if you have a demo link -->
    &middot;
    <a href="https://github.com/aetherspritee/PyFracVAL/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/aetherspritee/PyFracVAL/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap--known-issues">Roadmap & Known Issues</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

<!-- Add a screenshot of a generated aggregate if you like -->
<!-- [![PyFracVAL Aggregate][product-screenshot]]([link_to_docs_or_repo]) -->

`PyFracVAL` is a Python implementation of the FracVAL algorithm [1] for generating realistic 3D structures of particle aggregates, often encountered in fields like aerosol science, combustion, materials science, and colloid physics.

This implementation is based on the improved tunable algorithm originally presented in Fortran by Morán et al. [1], which allows for precise control over the resulting aggregate's fractal dimension (Df) and prefactor (kf). This is crucial for studying how these structural parameters influence the physical and optical properties of aggregates.

Key features:

- Generates aggregates from lognormally distributed polydisperse or monodisperse primary particles.
- Uses a hierarchical approach: Particle-Cluster Aggregation (PCA) to form subclusters, followed by Cluster-Cluster Aggregation (CCA).
- Allows specification of target Df (1 < Df < 3) and kf.
- Includes overlap control during aggregation steps.
- Provides a command-line interface for easy generation and options for programmatic use.

The goal is to provide a robust and validated Python alternative to the original Fortran code for researchers needing to generate these complex structures.

[1] J. Morán, A. Fuentes, F. Liu, J. Yon, _FracVAL: An improved tunable algorithm of cluster-cluster aggregation for generation of fractal structures formed by polydisperse primary particles_, Computer Physics Communications 239 (2019) 225–237. https://doi.org/10.1016/j.cpc.2019.01.015

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

This project primarily relies on the Python scientific stack.

- [![Python][Python-shield]][Python-url]
- [![NumPy][NumPy-shield]][NumPy-url]
- [![Numba][Numba-shield]][Numba-url]
- [![Pytest][Pytest-shield]][Pytest-url]
- [![Click][Click-shield]][Click-url]
- [![Pydantic][Pydantic-shield]][Pydantic-url]
- [![PyYAML][PyYAML-shield]][PyYAML-url]

Optional dependencies for plotting/visualization/dev:

- [![PyVista][PyVista-shield]][PyVista-url]
- [![Streamlit][Streamlit-shield]][Streamlit-url]
- [![UV][UV-shield]][UV-url]
- [![Ruff][Ruff-shield]][Ruff-url]
- [![Sphinx][Sphinx-shield]][Sphinx-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Follow these steps to set up `pyfracval` for use or development.

### Prerequisites

- **Python:** Version 3.10 to 3.12 recommended (Numba does not yet support 3.13 as of early 2024). Check with `python --version`.
- **Pip or UV:** A Python package installer. Check with `pip --version` or `uv --version`. [`uv`](https://github.com/astral-sh/uv) is recommended for faster environment management.
- **(Optional) Git:** Required for cloning the repository if installing from source.

### Installation

1.  **From PyPI (Recommended):**
    _(Once published)_

    ```bash
    pip install pyfracval
    # or using uv
    uv pip install pyfracval
    ```

2.  **From Source (for Development):**

    a. Clone the repository:

    ```bash
    git clone https://github.com/aetherspritee/PyFracVAL.git
    cd PyFracVAL
    ```

    b. Create and activate a virtual environment (using UV is recommended):

    ```bash
    # Using UV (recommended)
    uv venv --python 3.12 # Specify compatible Python version
    source .venv/bin/activate # Linux/macOS
    # .\venv\Scripts\activate # Windows

    # Using venv (built-in)
    # python3.12 -m venv .venv # Specify compatible Python version
    # source .venv/bin/activate # Linux/macOS
    # .\venv\Scripts\activate # Windows
    ```

    c. Install in editable mode with development dependencies:

    ```bash
    # Using UV (recommended)
    uv sync -e .[dev,test,docs]

    # Using pip
    # pip install -e .[dev,test,docs]
    ```

    _(Adjust optional dependencies `[dev,test,docs]` based on your `pyproject.toml`)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

Use the command-line interface `pyfracval` to generate aggregates.

**Generate a default aggregate (N=128, Df=2.0, kf=1.0, rp_g=100, rp_gstd=1.5):**

```bash
pyfracval
```

**Generate a specific aggregate (N=256, Df=1.8, kf=1.3, monodisperse):**

```bash
pyfracval -n 256 --df 1.8 --kf 1.3 --rp-gstd 1.0
```

**Generate 5 aggregates with plotting:**

```bash
pyfracval -n 100 --df 1.7 --kf 1.1 --num-aggregates 5 -p
```

**See all options:**

```bash
pyfracval --help
```

_For more detailed examples and programmatic usage, please refer to the [Documentation](https://[your_docs_host_or_github_pages_url])_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP & KNOWN ISSUES -->

## Roadmap & Known Issues

- [ ] **Fix Known Issues:** (Inherited from original FracVAL and implementation details)
  - [ ] Low fractal dimensions (e.g., Df < ~1.7) can fail during PCA, potentially due to geometric constraints in `Gamma_pc` calculation or sticking. _(Mitigation: try slightly higher kf)_.
  - [ ] High fractal dimensions / prefactors (e.g., Df > ~2.1 or high kf) can fail during PCA, potentially due to `Gamma_pc` calculation instability or overlap impossibility. _(Mitigation: try slightly lower kf or Df)_.
  - [ ] High fractal dimensions can be noticeably slow. _(Inherited behavior)_.
- [ ] Allow different distribution functions for monomer radii (e.g., normal).
- [ ] Further parallelization exploration (beyond Numba JIT).
- [ ] Add option for user-defined PCA parameters (Df_pca, kf_pca) via CLI.
- [ ] Publish package to PyPI.
- [ ] Add more examples and potentially tutorials to documentation.
- [ ] Investigate remaining differences compared to original Fortran behavior for edge cases.

See the [open issues](https://github.com/aetherspritee/PyFracVAL/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/YourFeature`)
3.  Install development dependencies (see [Installation](#installation))
4.  Make your changes
5.  Run tests (`pytest`) and linters/formatters (`ruff check .`, `ruff format .`)
6.  Commit your Changes (`git commit -m 'Add some YourFeature'`)
7.  Push to the Branch (`git push origin feature/YourFeature`)
8.  Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/aetherspritee/PyFracVAL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aetherspritee/PyFracVAL" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` file for more information.
_(If you adopted the original's license, change to: GNU General Public License v3. See `LICENSE` file)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CITATION -->

## Citation

If you use `PyFracVAL` in your research, please cite the original FracVAL paper:

```bibtex
@article{Moran2019FracVAL,
    author  = {J. Morán and A. Fuentes and F. Liu and J. Yon},
    title   = {{FracVAL: An improved tunable algorithm of cluster-cluster aggregation for generation of fractal structures formed by polydisperse primary particles}},
    journal = {Computer Physics Communications},
    year    = {2019},
    volume  = {239},
    pages   = {225--237},
    doi     = {10.1016/j.cpc.2019.01.015},
}
```

_(Optional: Add citation for this specific Python implementation if appropriate)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- Based on the original FracVAL algorithm and Fortran code by [Morán et al. (2019)](https://doi.org/10.1016/j.cpc.2019.01.015).
- Uses the algorithm concepts developed by Filippov et al. (2000).
- README template adapted from [Best-README-Template](https://github.com/othneildrew/Best-README-Template).
- Python scientific libraries (NumPy, Numba, etc.).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- Update these links with your username/repo -->

[contributors-shield]: https://img.shields.io/github/contributors/aetherspritee/PyFracVAL.svg?style=for-the-badge
[contributors-url]: https://github.com/aetherspritee/PyFracVAL/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/aetherspritee/PyFracVAL.svg?style=for-the-badge
[forks-url]: https://github.com/aetherspritee/PyFracVAL/network/members
[stars-shield]: https://img.shields.io/github/stars/aetherspritee/PyFracVAL.svg?style=for-the-badge
[stars-url]: https://github.com/aetherspritee/PyFracVAL/stargazers
[issues-shield]: https://img.shields.io/github/issues/aetherspritee/PyFracVAL.svg?style=for-the-badge
[issues-url]: https://github.com/aetherspritee/PyFracVAL/issues
[license-shield]: https://img.shields.io/github/license/aetherspritee/PyFracVAL?style=for-the-badge
[license-url]: https://github.com/aetherspritee/PyFracVAL/blob/main/LICENSE

<!-- [linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555 -->
<!-- [linkedin-url]: https://linkedin.com/in/[your_linkedin_username] -->
<!-- [product-screenshot]: .github/screenshot.png --> <!-- Add path to screenshot if used -->

<!-- Built With Links -->

[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[NumPy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Numba-shield]: https://img.shields.io/badge/Numba-00A3E0?style=for-the-badge&logo=numba&logoColor=white
[Numba-url]: https://numba.pydata.org/
[Pytest-shield]: https://img.shields.io/badge/Pytest-0A9B70?style=for-the-badge&logo=pytest&logoColor=white
[Pytest-url]: https://pytest.org/
[Click-shield]: https://img.shields.io/badge/Click-4D4D4D?style=for-the-badge&logo=python&logoColor=white
[Click-url]: https://click.palletsprojects.com/
[Pydantic-shield]: https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white
[Pydantic-url]: https://docs.pydantic.dev/
[PyYAML-shield]: https://img.shields.io/badge/PyYAML-4B5259?style=for-the-badge&logo=yaml&logoColor=white
[PyYAML-url]: https://pyyaml.org/
[PyVista-shield]: https://img.shields.io/badge/PyVista-6495ED?style=for-the-badge&logo=python&logoColor=white
[PyVista-url]: https://docs.pyvista.org/
[Streamlit-shield]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white
[Streamlit-url]: https://streamlit.io/
[UV-shield]: https://img.shields.io/badge/uv-black?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyBmaWxsPSJjdXJyZW50Q29sb3IiIHZpZXdCb3g9IjAgMCAxNiAxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik04IC4yYTEuMiAxLjIgMCAwIDAtMS4yIDEuMnYxMS4yYTEuMiAxLjIgMCAxIDAgMi40IDBWMS40QTEuMiAxLjIgMCAwIDAgOCAuMm0wIDMuNmExLjIgMS4yIDAgMCAwLTEuMiAxLjJ2NC44YTEuMiAxLjIgMCAxIDAgMi40IDBWNS44YTEuMiAxLjIgMCAwIDAtMS4yLTEuMlptLTYgMy42YTEuMiAxLjIgMCAwIDEgMS4yIDEuMlYxNGEyLjQgMi40IDAgMSAxLTIuNCAwVjkuOGExLjIgMS4yIDAgMCAxIDEuMi0xLjJsMCAwWk0xNCA4LjZhMS4yIDEuMiAwIDAgMC0xLjIgMS4ydjQuMmMwIC43LS41IDEuMi0xLjIgMS4yaC4yYzAtMi4xLTEuMy0zLjgtMy00LjZBNi45IDYuOSAwIDAgMSA4IDEwLjZhNi45IDYuOSAwIDAgMS0yLS40IDQuOCA0LjggMCAwIDEtMy00LjZWNS44YzAtLjcgLjYtMS4yIDEuMi0xLjJoMi40Yy43IDAgMS4yLjUgMS4yIDEuMnYuNmwtMS4zIDNBMy42IDMuNiAwIDAgMSA4IDguNmMyLjEgMCA0LjItLjggNS42LTIuMWRhLjc1LjggMCAwIDEgLjgtLjdjLjQgMCAuOC4zLjggLjdWOS44bDAgMGgxLjJWMTRjMC0uNy0uNS0xLjItMS4yLTEuMlY5LjhjMC0uNy41LTEuMiAxLjItMS4yWiIvPjwvc3ZnPg==
[UV-url]: https://github.com/astral-sh/uv
[Ruff-shield]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[Ruff-url]: https://github.com/astral-sh/ruff
[Sphinx-shield]: https://img.shields.io/badge/Sphinx-181717?style=for-the-badge&logo=sphinx&logoColor=white
[Sphinx-url]: https://www.sphinx-doc.org/en/master/
