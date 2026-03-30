#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "Pokemon-Showdown" ]; then
  git clone https://github.com/smogon/pokemon-showdown.git Pokemon-Showdown
fi

cd Pokemon-Showdown
npm install
