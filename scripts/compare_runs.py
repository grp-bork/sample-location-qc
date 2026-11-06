# /// script
# dependencies = ["polars"]
# ///
"""Compare two runs (e.g. two database versions, or before / after a rule
change): per-flag totals, and the studies whose flagged count changed most.

Samples are matched by id; a sample in only one run counts as added/removed.

Usage: uv run compare_runs.py OLD_RUN NEW_RUN [--top N]
"""
import argparse

import polars as pl

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("old")
ap.add_argument("new")
ap.add_argument("--top", type=int, default=30)
a = ap.parse_args()
pl.Config.set_tbl_rows(a.top + 10)
pl.Config.set_tbl_width_chars(160)

cols = ["Sample", "study", "bogus", "flags", "suggest_lat"]
o = pl.read_csv(f"{a.old}/geo.samples.tsv", separator="\t", infer_schema_length=0, columns=cols)
n = pl.read_csv(f"{a.new}/geo.samples.tsv", separator="\t", infer_schema_length=0, columns=cols)
FLAGS = ["out_of_range", "null_island", "autofill", "country_mismatch", "sea_mismatch",
         "terrestrial_at_sea", "region_mismatch"]


def totals(d, name):
    return pl.DataFrame({"what": ["samples", "flagged", "with suggestion"] + FLAGS,
                         name: [d.height, (d["bogus"] == "true").sum(), d["suggest_lat"].is_not_null().sum()]
                         + [d["flags"].fill_null("").str.contains(f).sum() for f in FLAGS]})


t = totals(o, "old").join(totals(n, "new"), on="what", maintain_order="left")
print(t.with_columns(delta=pl.col("new") - pl.col("old")))

j = o.join(n, on="Sample", how="full", suffix="_new", coalesce=True)
print("\nsamples only in old:", j["bogus_new"].is_null().sum(), " only in new:", j["bogus"].is_null().sum())
both = j.filter(pl.col("bogus").is_not_null() & pl.col("bogus_new").is_not_null())
print("in both, newly flagged:", ((both["bogus"] == "false") & (both["bogus_new"] == "true")).sum(),
      " no longer flagged:", ((both["bogus"] == "true") & (both["bogus_new"] == "false")).sum(),
      " flags changed:", (both["flags"].fill_null("") != both["flags_new"].fill_null("")).sum())

per = (j.with_columns(study=pl.coalesce("study_new", "study"))
        .group_by("study")
        .agg(old=(pl.col("bogus") == "true").sum(), new=(pl.col("bogus_new") == "true").sum())
        .with_columns(delta=pl.col("new") - pl.col("old"))
        .filter(pl.col("delta") != 0)
        .sort(pl.col("delta").abs(), "study", descending=[True, False]))
print(f"\nstudies with a changed flagged count: {per.height}")
print(per.head(a.top))
