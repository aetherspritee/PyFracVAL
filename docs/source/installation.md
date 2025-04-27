# Installation

This page provides instructions on how to install the `pyfracval` package.

## Prerequisites

- Python 3.10 or higher (check your version with `python --version`).
- `pip` (Python package installer)

## Standard Installation

The recommended way to install `pyfracval` is using `pip` from the Python Package Index (PyPI) (once you publish it there):

```bash
pip install pyfracval
```

This will install the package and its required dependencies (like NumPy, Pydantic, etc.).

## Installation from Source (for Development)

If you want to contribute to the project or install the latest development version directly from the source code (e.g., after cloning from GitHub), follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/pyfracval.git
    cd pyfracval
    ```

    _(Replace with your actual repository URL)_

2.  **Create a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage dependencies:

    ```bash
    # Using venv (built-in)
    python -m venv .venv
    source .venv/bin/activate # On Linux/macOS
    # .\venv\Scripts\activate # On Windows

    # Or using conda
    # conda create -n pyfracval-env python=3.11 # Or your preferred version
    # conda activate pyfracval-env
    ```

3.  **Install in Editable Mode:**
    Installing in editable mode (`-e`) links the installed package to your source code, so changes you make are immediately reflected without reinstalling.
    ```bash
    pip install -e .[dev,test,docs]
    ```
    - The `.` refers to the current directory (where `pyproject.toml` is).
    - The `[dev,test,docs]` part installs optional dependencies needed for development, running tests, and building documentation (assuming you define these groups in your `pyproject.toml`). Adjust or remove this part as needed. If you don't have groups defined, you might just do `pip install -e .` and install dev tools separately.

## Checking the Installation

You can verify the installation by importing the package in a Python interpreter:

```python
import pyfracval
# Optional: check version if you have defined __version__
# print(pyfracval.__version__)
```

If no errors occur, the installation was successful.
