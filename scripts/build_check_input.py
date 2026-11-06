# /// script
# requires-python = ">=3.10"
# dependencies = ["polars"]
# ///
"""Build the claim-check input from RUN/samples.tsv.

The claim is the part of a sample's location text before the first colon
("Spain: Mallorca" -> Spain); curated/claim_map.tsv maps names the checker does
not know (site names, misspellings, territories) to one it does.

Outputs:
  RUN/check_input.tsv    one row per distinct (lat, lon, claim): input of check_claims.py
  RUN/sample_claims.tsv  one row per sample with a coordinate
  RUN/claim_counts.tsv   samples per claim

Usage: uv run build_check_input.py RUN [--claim-map FILE]
"""
import argparse
import sys

import polars as pl

MISSING = ["missing", "not collected", "not applicable", "na", "n/a", "none", "unknown",
           "not available", "not provided", "-", "restricted access", "not determined", "null"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run")
ap.add_argument("--claim-map", default="curated/claim_map.tsv")
a = ap.parse_args()

s = (pl.read_csv(f"{a.run}/samples.tsv", separator="\t", infer_schema_length=0,
                 null_values=["", "None"], quote_char=None)
       .filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null()))
lv = pl.col("location").str.to_lowercase()
s = s.with_columns(location=pl.when(lv.is_in(MISSING) | lv.str.contains(r"^(missing|not )"))
                             .then(None).otherwise(pl.col("location")))
s = (s.with_columns(claim=pl.col("location").str.replace(r":.*", "").str.strip_chars())
      .with_columns(claim=pl.when(pl.col("claim") != "").then(pl.col("claim")))
      .select(Sample="sample", latitude="latitude", longitude="longitude",
              name_field=pl.when(pl.col("claim").is_not_null()).then(pl.col("location_field")),
              name_value=pl.when(pl.col("claim").is_not_null()).then(pl.col("location")),
              claim="claim"))
cmap = pl.read_csv(a.claim_map, separator="\t", infer_schema_length=0, quote_char=None)
s = (s.join(cmap, on="claim", how="left")
      .with_columns(claim_raw="claim", claim=pl.coalesce("mapped", "claim"))
      .drop("mapped"))
print("claims remapped:", s["claim"].ne(s["claim_raw"]).sum(), "samples", file=sys.stderr)
print("coord samples:", s.height, " with claim:", s["claim"].is_not_null().sum(), file=sys.stderr)

pairs = (s.filter(pl.col("claim").is_not_null())
          .group_by("latitude", "longitude", "claim").agg(n_samples=pl.len())
          .sort("latitude", "longitude", "claim")
          .with_row_index("id", offset=1))
print("distinct (lat, lon, claim):", pairs.height, " distinct claims:", pairs["claim"].n_unique(), file=sys.stderr)
pairs.select("id", "latitude", "longitude", geographic_location="claim", n_samples="n_samples") \
     .write_csv(f"{a.run}/check_input.tsv", separator="\t")
s.join(pairs.select("id", "latitude", "longitude", "claim"), on=["latitude", "longitude", "claim"], how="left") \
 .write_csv(f"{a.run}/sample_claims.tsv", separator="\t")
(s.filter(pl.col("claim").is_not_null()).group_by("claim").len().sort("len", descending=True)
  .write_csv(f"{a.run}/claim_counts.tsv", separator="\t"))
