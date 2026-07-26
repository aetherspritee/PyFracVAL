# CHANGELOG

## Unreleased

### Highlights
- Refactored core utilities into domain modules (`geometry`, `fractal`, `overlap`, `cca_kernels`, `pca_kernels`) while preserving backward compatibility shims.
- Added configuration adapter (`config_adapter.py`) and environment configuration module (`environments.py`).
- Expanded NumPy-style documentation and integrated Sphinx/BibTeX citations for algorithm background.
- Added local docs verification script and `devenv` docs tasks.
- Added CI workflow to run tests and docs build before release.

### Added
- `scripts/check-docs.sh` for local docs checks (`ci` and `strict` modes).
- `.github/workflows/test.yml` for CI test+docs validation.

### Changed
- Documentation index and usage pages now reference FracVAL literature citations.
- Release workflow now waits for CI validation.

### Fixed
- Sphinx hard doc build error caused by unintended substitution in API docstrings.
- README placeholders for documentation links and outdated setup commands.

## v0.1.0 (2025-04-30)

### Features

- Add semantic versioning
  ([`74afedd`](https://github.com/aetherspritee/PyFracVAL/commit/74afedd44ced843eafb33e7d02d1dd495aff022a))
