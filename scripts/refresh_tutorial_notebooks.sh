#!/usr/bin/env bash
# Re-execute tutorial notebooks so MkDocs can render cell outputs (execute: false in mkdocs.yml).
set -euo pipefail
cd "$(dirname "$0")/.."

# Notebook contains UTF-8 text and binary PNG outputs (Windows cp1252 breaks otherwise).
export PYTHONUTF8=1

if command -v uv >/dev/null 2>&1; then
  uv pip install -e ".[hj_reachability,docs]" jupyter nbclient kaleido >/dev/null
else
  pip install -e ".[hj_reachability,docs]" jupyter nbclient kaleido >/dev/null
fi

jupyter execute docs/tutorials/set_builders.ipynb --inplace

echo "Done. Preview with: mkdocs serve"
