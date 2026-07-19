#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

rm -rf build dist
python3 -m build
python3 -m twine check --strict dist/*

printf '%s\n' \
  "Release artifacts were built and validated in dist/." \
  "Before tagging, protect v* tags and verify required main CI is successful." \
  "Ensure the repository PYPI_TOKEN secret contains a current project-scoped token." \
  "Publishing is handled by .github/workflows/release.yml after a v*.*.* tag is pushed."
