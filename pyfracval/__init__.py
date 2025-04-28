"""Core package for PyFracVAL, a fractal aggregate generator."""

import tomllib
from pathlib import Path

__version__ = ""
_authors = ""

package_src = Path(__file__).parent
# _version = package_src / "VERSION"
_pyproject = package_src.parent / "pyproject.toml"
if _pyproject.is_file():
    with open(_pyproject, "rb") as f:
        data = tomllib.load(f)

        if "version" in data["project"]:
            __version__ = data["project"]["version"]
        else:
            raise ValueError("Version not found in pyproject.toml")

        if "authors" in data["project"]:
            _authors = ",".join([x["name"] for x in data["project"]["authors"]])
        else:
            raise ValueError("Version not found in pyproject.toml")
