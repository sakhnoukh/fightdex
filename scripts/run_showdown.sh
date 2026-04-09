#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "Pokemon-Showdown" ]; then
  echo "Pokemon-Showdown directory not found in repo root."
  echo "Run scripts/setup_showdown.sh first."
  exit 1
fi

cd Pokemon-Showdown
node pokemon-showdown start --no-security
