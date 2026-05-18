#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen is required to build XML output" >&2
  exit 127
fi

doxygen docs/doxygen/Doxyfile
