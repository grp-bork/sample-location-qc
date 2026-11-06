# /// script
# dependencies = ["polars"]
# ///
"""Suggested coordinate corrections: RUN/geo.samples.tsv -> RUN/corrections.tsv.

One row per flagged sample with a suggested position. Only mechanical fixes
are suggested (swap / sign flip, the start of a spreadsheet drag), and only
when the sample's own location text or a place named in the study
description backs the new position (suggest_basis); never a country or
region centre. Flagged samples without a row here keep their coordinate and
need a look by hand (or removal).

Columns: sample, study, latitude, longitude (as recorded), new_latitude,
new_longitude, fix (diagnosis, e.g. lat_sign, swapped, autofill_step(...)x7),
source (transform / autofill), basis (claim / place / claim+place),
uncertainty_km (rounding precision of the recorded coordinate), claim,
supporting_place and supporting_place_km (for basis place), flags.

Usage: uv run export_corrections.py RUN
"""
import sys

import polars as pl

run = sys.argv[1]
s = pl.read_csv(f"{run}/geo.samples.tsv", separator="\t", infer_schema_length=0)
c = (s.filter((pl.col("bogus") == "true") & pl.col("suggest_lat").is_not_null())
      .select(sample="Sample", study="study", latitude="latitude", longitude="longitude",
              new_latitude="suggest_lat", new_longitude="suggest_lon", fix="diagnosis",
              source="suggest_source", basis="suggest_basis", uncertainty_km="suggest_uncertainty_km",
              claim="claim",
              supporting_place=pl.when(pl.col("fix_place_support") == "supported").then(pl.col("fix_place")),
              supporting_place_km=pl.when(pl.col("fix_place_support") == "supported").then(pl.col("fix_place_km")),
              flags="flags")
      .sort("study", "sample"))
c.write_csv(f"{run}/corrections.tsv", separator="\t")
n_bogus = (s["bogus"] == "true").sum()
print(f"{c.height} corrections in {c['study'].n_unique()} studies "
      f"({n_bogus - c.height} flagged samples without one) -> {run}/corrections.tsv", file=sys.stderr)
print(c.group_by("source", "basis").len().sort("len", descending=True), file=sys.stderr)
