"""Core package for PyFracVAL, a fractal aggregate generator."""

import tomllib
from pathlib import Path

__version__ = ""
_authors = ""

package_src = Path(__file__).parent
_pyproject = package_src.parent / "pyproject.toml"
if _pyproject.is_file():
    with open(_pyproject, "rb") as f:
        data = tomllib.load(f)

        if "authors" in data["project"]:
            _authors = ",".join([x["name"] for x in data["project"]["authors"]])
        else:
            raise ValueError("Author not found in pyproject.toml")
