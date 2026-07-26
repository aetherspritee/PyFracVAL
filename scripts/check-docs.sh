#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ci}"

if [[ "$MODE" == "strict" ]]; then
  devenv shell -- uv run sphinx-build -b html -W -n --keep-going docs/source docs/build/html
else
  # CI-parity mode: mirrors .github/workflows/docs.yml behavior (HTML build)
  devenv shell -- uv run sphinx-build -b html docs/source docs/build/html
fi

# Optional linkcheck as second argument
if [[ "${2:-}" == "linkcheck" ]]; then
  devenv shell -- uv run sphinx-build -b linkcheck docs/source docs/build/linkcheck
fi

echo "Docs build check complete (mode=$MODE)."
