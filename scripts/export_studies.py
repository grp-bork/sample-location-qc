# /// script
# dependencies = ["polars"]
# ///
"""Per-study summary of flagged samples: RUN/geo.samples.tsv -> RUN/studies_flagged.tsv.

Usage: uv run export_studies.py RUN
"""
import sys

import polars as pl

FLAGS = ["out_of_range", "null_island", "autofill", "country_mismatch", "sea_mismatch",
         "terrestrial_at_sea", "region_mismatch"]
d = pl.read_csv(f"{sys.argv[1]}/geo.samples.tsv", separator="\t", infer_schema_length=0)
tot = d.group_by("study").agg(n_samples_with_coords=pl.len(),
    n_cleared_by_places=((pl.col("bogus") == "false") & pl.col("place_match").is_not_null()).sum())
b = d.filter(pl.col("bogus") == "true")
agg = b.group_by("study").agg(
    n_flagged=pl.len(),
    n_flagged_coords=pl.struct("latitude", "longitude").n_unique(),
    *[pl.col("flags").str.contains(f).sum().alias(f"n_{f}") for f in FLAGS],
    n_fix_transform=(pl.col("suggest_source") == "transform").sum(),
    n_fix_autofill=(pl.col("suggest_source") == "autofill").sum(),
    n_fix_backed=(pl.col("fix_place_support") == "supported").sum(),
    n_fix_contradicted=(pl.col("fix_place_support") == "original_fits").sum(),
    n_fix_no_match=(pl.col("fix_place_support") == "no_match").sum(),
    claims=pl.col("claim").drop_nulls().unique().sort().str.join("; "),
    found=pl.coalesce("country", "sea").drop_nulls().unique().sort().head(10).str.join("; "),
    diagnoses=pl.col("diagnosis").drop_nulls().str.replace(r"\)x\d+$", ")").unique().sort().str.join("; "),
    fix_places=pl.col("fix_place").filter(pl.col("fix_place_support") == "supported")
                 .unique().sort().head(10).str.join("; "),
    example_sample=pl.col("Sample").sort().first(),
)
out = (agg.join(tot, on="study", how="left")
       .with_columns(frac_flagged=(pl.col("n_flagged") / pl.col("n_samples_with_coords")).round(3),
                     n_fix=pl.col("n_fix_transform") + pl.col("n_fix_autofill"))
       .with_columns(main_flag=pl.concat_list([pl.struct(f=pl.lit(f), n=pl.col(f"n_{f}")) for f in FLAGS])
                     .list.eval(pl.element().sort_by(pl.element().struct.field("n"), descending=True)
                                .first().struct.field("f")).list.first())
       .select("study", "n_samples_with_coords", "n_flagged", "frac_flagged", "n_flagged_coords", "main_flag",
               *[f"n_{f}" for f in FLAGS], "n_fix", "n_fix_transform", "n_fix_autofill",
               "n_fix_backed", "n_fix_contradicted", "n_fix_no_match", "n_cleared_by_places",
               "claims", "found", "diagnoses", "fix_places", "example_sample")
       .sort(["n_flagged", "study"], descending=[True, False]))
out.write_csv(f"{sys.argv[1]}/studies_flagged.tsv", separator="\t")
print(out.height, "studies,", out["n_flagged"].sum(), "flagged samples,", out.width, "columns")
