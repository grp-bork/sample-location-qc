#!/usr/bin/env bash
# Run the whole check on one run directory.
#   ./run.sh RUN            RUN/samples.tsv (+ optional RUN/study_places.tsv,
#                           else curated/study_places.tsv)
# Settings come from .env (copy .env.example). Service responses are cached in
# $CACHE_DIR (default cache/), shared by all runs; delete it after upgrading
# the metalog-geocoding service.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && { set -a; . ./.env; set +a; }
RUN=${1:?usage: ./run.sh RUN_DIR}
CACHE=${CACHE_DIR:-cache}
[ -s "$RUN/samples.tsv" ] || { echo "no $RUN/samples.tsv" >&2; exit 1; }
mkdir -p "$CACHE"
step() { echo "== $*" >&2; }

step "claims"
uv run -q scripts/build_check_input.py "$RUN"
uv run -q scripts/check_claims.py --input "$RUN/check_input.tsv" --prefix "$CACHE/claims"
step "geocode"
uv run -q scripts/geocode_best.py --input "$RUN/sample_claims.tsv" --prefix "$CACHE/geocoded"
# Place evidence is only looked up for flagged coordinates, so flag, look up,
# and flag again with the evidence.
step "flag (1/2)"
rm -f "$RUN/place_distances.tsv"
uv run -q scripts/flag_coords.py "$RUN" --geo "$CACHE/geocoded.tsv" \
    --verdicts "$CACHE/claims.verdicts.tsv" > "$RUN/flag_coords.1.log" 2>&1
step "place distances"
uv run -q scripts/place_distances.py "$RUN" --cache "$CACHE/place_distance_cache.jsonl" \
    > "$RUN/place_distances.log" 2>&1
step "flag (2/2)"
uv run -q scripts/flag_coords.py "$RUN" --geo "$CACHE/geocoded.tsv" \
    --verdicts "$CACHE/claims.verdicts.tsv" > "$RUN/flag_coords.log" 2>&1
step "export"
uv run -q scripts/export_studies.py "$RUN"
uv run -q scripts/export_corrections.py "$RUN"
step "done: $RUN/geo.samples.tsv, $RUN/studies_flagged.tsv, $RUN/corrections.tsv"
