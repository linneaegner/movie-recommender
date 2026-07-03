#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZIP="ml-latest-small.zip"
URL="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

if [[ -f ml-latest-small/ratings.csv ]]; then
  echo "Dataset already present in ml-latest-small/"
  exit 0
fi

echo "Downloading MovieLens Small…"
curl -L -o "$ZIP" "$URL"
unzip -q "$ZIP"
rm -f "$ZIP"
echo "Done. Files in ml-latest-small/"
