"""Core package for PyFracVAL, a fractal aggregate generator."""

import tomllib
from pathlib import Path

__version__ = ""
_authors = ""

project_root = Path(__file__).parent.parent
with open(project_root / "pyproject.toml", "rb") as f:
    data = tomllib.load(f)

    if "version" in data["project"]:
        __version__ = data["project"]["version"]
    else:
        raise ValueError("Version not found in pyproject.toml")

    if "authors" in data["project"]:
        _authors = ",".join([x["name"] for x in data["project"]["authors"]])
    else:
        raise ValueError("Version not found in pyproject.toml")
