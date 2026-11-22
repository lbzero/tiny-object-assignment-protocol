#!/usr/bin/env bash
# make_archive.sh --- assemble the Zenodo payload for a release
#
# The git repository deliberately stays small: it carries code, configuration,
# image identifier lists and aggregate result JSONs. The per-ground-truth traces
# (32 files, ~26 MB) are archived separately so that the repository remains
# clonable and the archive remains citable as a single immutable object.
#
# Usage
#   tools/make_archive.sh --traces-a DIR_A --traces-b DIR_B [--out DIR] [--version v1.0.0]
#
#   DIR_A  directory holding the 16 subset-A trace CSVs and their _frames/_meta files
#   DIR_B  same for subset B
#
# Output
#   <out>/ltnet-repro-<version>.tar.gz
#   <out>/SHA256SUMS
#   <out>/ARCHIVE_MANIFEST.txt   file list with sizes, for the Zenodo description
#
set -euo pipefail

VERSION="v1.0.0"
OUT="dist"
A=""
B=""

while [ $# -gt 0 ]; do
  case "$1" in
    --traces-a) A="$2"; shift 2 ;;
    --traces-b) B="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$A" ] && [ -n "$B" ] || { echo "--traces-a and --traces-b are required" >&2; exit 2; }
[ -d "$A" ] || { echo "not a directory: $A" >&2; exit 2; }
[ -d "$B" ] || { echo "not a directory: $B" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="ltnet-repro-${VERSION}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/$NAME"
cp -r "$ROOT/code" "$ROOT/config" "$ROOT/results" "$ROOT/README.md" \
      "$ROOT/requirements.txt" "$ROOT/CITATION.cff" "$ROOT/manifest.csv" \
      "$ROOT/data" "$STAGE/$NAME/" 2>/dev/null || true
mkdir -p "$STAGE/$NAME/traces/subsetA" "$STAGE/$NAME/traces/subsetB"
cp "$A"/*.csv "$A"/*.json "$STAGE/$NAME/traces/subsetA/" 2>/dev/null || true
cp "$B"/*.csv "$B"/*.json "$STAGE/$NAME/traces/subsetB/" 2>/dev/null || true

# Refuse to ship anything that looks like imagery or weights.
if find "$STAGE" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.pt' \) | grep -q .; then
  echo "refusing to archive: found image or weight files" >&2
  find "$STAGE" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.pt' \) >&2
  exit 3
fi

mkdir -p "$OUT"
( cd "$STAGE" && tar czf - "$NAME" ) > "$OUT/${NAME}.tar.gz"
( cd "$OUT" && sha256sum "${NAME}.tar.gz" > SHA256SUMS )
( cd "$STAGE/$NAME" && find . -type f -printf '%10s  %p\n' | sort -k2 ) > "$OUT/ARCHIVE_MANIFEST.txt"

echo "wrote $OUT/${NAME}.tar.gz"
cat "$OUT/SHA256SUMS"
echo "files: $(wc -l < "$OUT/ARCHIVE_MANIFEST.txt")"
