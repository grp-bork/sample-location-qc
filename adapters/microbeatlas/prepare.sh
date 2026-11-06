#!/usr/bin/env bash
# Download the microbeatlas sample metadata and build
# runs/microbeatlas/samples.tsv, the generic sample table (prepare.py).
# Then run: ./run.sh runs/microbeatlas
# Downloads are kept in runs/microbeatlas/download/; delete them to refetch.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:-runs/microbeatlas}
BASE=https://microbeatlas.org/downloads/samples
UA="sample-location-qc (https://github.com/grp-bork/sample-location-qc)"
mkdir -p "$RUN/download"

fetch() {  # url dest [filter]: samples.info is 6 GB, so it is gzipped on the fly
    [ -s "$2" ] && { echo "have $2"; return; }
    curl -fSL --retry 3 -A "$UA" -e https://microbeatlas.org/ "$1" | ${3:-cat} > "$2.part"
    mv "$2.part" "$2"
}
fetch "$BASE/samples.info" "$RUN/download/samples.info.gz" gzip
gzip -t "$RUN/download/samples.info.gz"
fetch "$BASE/samples.info.latlon.parsed.tsv" "$RUN/download/samples.info.latlon.parsed.tsv"

uv run -q adapters/microbeatlas/prepare.py "$RUN/download/samples.info.gz" \
    "$RUN/download/samples.info.latlon.parsed.tsv" "$RUN/samples.tsv"
