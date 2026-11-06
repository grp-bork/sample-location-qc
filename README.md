# sample-location-qc

Finds sample coordinates that contradict the sample's own metadata, and
suggests a corrected position where the error is mechanical and the
metadata backs the fix. Written for the microbeatlas sample table, but it
works on any database whose samples have coordinates and an INSDC-style
location text ("Country: detail").

The coordinates are checked with
[metalog-geocoding](https://github.com/grp-bork/metalog-geocoding), which
does reverse geocoding, claim checks, coast distances and place-name
distances.

## What gets flagged

Each sample gets zero or more flags; any flag makes `bogus` true.

| flag | meaning |
|---|---|
| `out_of_range` | latitude/longitude cannot exist |
| `null_island` | exactly (0, 0) |
| `country_mismatch` | the coordinate is in another country than the one claimed, or more than 370 km offshore of it |
| `sea_mismatch` | the coordinate is on land, or in an unrelated sea, for a sea claim |
| `region_mismatch` | the country is right but the stated admin-1 region is more than 25 km away |
| `terrestrial_at_sea` | a soil, gut, plant, etc. sample more than 25 km offshore |
| `autofill` | a spreadsheet drag: a run of five or more coordinates in a study with one constant step |

Tolerances widen with the coordinate's rounding: whole degrees can be up to
78.7 km from the true point. A flag is cleared when the coordinate lies in
a place the metadata names, either the sample's own location text and
`sampling_site` or the places named in the study description. The full
rules and every constant are in the docstring and at the top of
[`scripts/flag_coords.py`](scripts/flag_coords.py).

## What gets corrected

`corrections.tsv` lists a new position only for **mechanical** errors:

- swapped latitude and longitude
- a dropped or flipped sign
- the start coordinate of an autofill run

A fix is listed only when text backs it: the fixed point lands in the
sample's claimed country or sea (`basis` = `claim`), or in a place named
in the study description (`place`), or both. The tool never suggests a
country or region centre. Flagged samples without a correction keep their
coordinate and need a look by hand, or should be hidden.

## Setup

- [uv](https://docs.astral.sh/uv/). Each script declares its own
  dependencies, and `uv run` installs them.
- A running metalog-geocoding service, plus a checkout of the same version
  with its `data/` set up (`scripts/setup_data.py`, see its
  [installation docs](https://github.com/grp-bork/metalog-geocoding/blob/main/docs/installation-and-data.md)).
  `flag_coords.py` reads the Natural Earth and IHO layers and the country
  aliases from that checkout.
- `cp .env.example .env`, then set `METALOG_GEOCODING_URL` and
  `METALOG_GEOCODING_DIR`. `.env` is git-ignored.

## Running on microbeatlas

```sh
adapters/microbeatlas/prepare.sh            # download, build runs/microbeatlas/samples.tsv
./run.sh runs/microbeatlas                  # check, flag, export
uv run scripts/check_expected.py runs/microbeatlas adapters/microbeatlas/expected.tsv
set -a; . ./.env; set +a                    # the report reads METALOG_GEOCODING_DIR
uv run adapters/microbeatlas/report/report_data.py runs/microbeatlas runs/microbeatlas/report.html
```

To compare a new database version with the previous run (or a rule change
with the run before it):

```sh
uv run scripts/compare_runs.py runs/microbeatlas-2026-10 runs/microbeatlas
```

Run times: the adapter takes about 6 min. A first run without a cache makes
about 120k `/check` calls and geocodes about 120k coordinates, which takes a
few hours. Service responses are kept in `cache/` and reused by every later
run, so a new version only sends what is new. Delete `cache/` after
upgrading the service.

## Running on another database

Write an adapter that produces `RUN/samples.tsv`, then run `./run.sh RUN`.
The microbeatlas adapter
([`adapters/microbeatlas/prepare.py`](adapters/microbeatlas/prepare.py)) is
the example to follow.

**`samples.tsv`** (required): tab-separated with a header, one row per
sample with a coordinate, no quoting.

| column | content |
|---|---|
| `sample` | sample id, unique |
| `study` | study/project id; autofill detection and study places work per study |
| `latitude`, `longitude` | decimal degrees **as recorded**. Keep the original text: the number of decimals sets the tolerance |
| `location_field` | which metadata field `location` came from (informational) |
| `location` | INSDC-style location text, `Country: detail` or `Sea: detail`; the part before the first colon is the claim |
| `sampling_site` | free-text site name, if the source has one |
| `scientific_name`, `env_biome`, `env_feature`, `env_material`, `isolation_source` | used to tell marine from terrestrial samples; empty is fine |

**`curated/study_places.tsv`** lists the places named in ENA/SRA study
titles and descriptions, one row per study and place. It is the output of
`scripts/ena_places.py` from
[toponym_ner](https://github.com/grp-bork/toponym_ner), which needs an
LLM server (see its README). Studies match by project or study accession
(`study_accession`, `secondary_study_accession`), so the file serves any
source that uses INSDC accessions. Only rows with `sampled` = `true` are
used, looked up by their `corrected` name. A `RUN/study_places.tsv`, if
present, is used instead of this file.

Flagged studies that are not in the file yet are listed in
`RUN/studies_without_places.txt`. Their fixes can only be backed by the
samples' own location text. To add them, append to the file and rerun:

```sh
python scripts/ena_places.py -o /path/to/sample-location-qc/curated/study_places.tsv \
    -f runs/microbeatlas/studies_without_places.txt   # in a toponym_ner checkout
./run.sh runs/microbeatlas
```

**`curated/claim_map.tsv`** maps claims the checker does not know (site
names, misspellings, territories filed under their sovereign country) to a
name it does. It is shared by all sources. Unplaceable claims show up with
code `unknown_place` in `geo.samples.tsv`; add them here.

## Outputs (in `RUN/`)

| file | content |
|---|---|
| `geo.samples.tsv` | one row per sample: claim, verdict, land/sea/country/admin-1, coast distance, sample type, `flags`, `diagnosis`, suggested fix and its evidence |
| `geo.coords.tsv` | one row per distinct coordinate: geocoding result |
| `corrections.tsv` | the backed fixes, see above |
| `studies_flagged.tsv` | one row per study with flagged samples: counts per flag, fixes, claims vs found |
| `place_distances.tsv` | distances from flagged coordinates and their variants to the named places |
| `*.log` | step logs with summary counts |

## Pipeline

`run.sh` runs these in order. Every script has a docstring with its inputs
and outputs.

1. `build_check_input.py` takes the claim from `location`, applies
   `claim_map.tsv`, and writes the distinct (lat, lon, claim) pairs.
2. `check_claims.py` grades each pair with the service (`POST /check`).
3. `geocode_best.py` gets land/sea, country, admin-1, sea and coast distance
   for every coordinate (`POST /geocode`).
4. `flag_coords.py` applies the flags and diagnoses swaps, sign flips and
   drags.
5. `place_distances.py` measures distances from flagged coordinates and
   their swapped/flipped variants to the named places
   (`POST /place-distance`).
6. `flag_coords.py` runs again, now with the place evidence: it clears
   coordinates that lie in a named place and marks fixes as backed.
7. `export_studies.py` and `export_corrections.py` write the summary tables.
