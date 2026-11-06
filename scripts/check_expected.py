# /// script
# dependencies = ["polars"]
# ///
"""Regression check: compare RUN/geo.samples.tsv with hand-checked expectations.

EXPECTED is a TSV with columns sample, bogus (true/false), flags (expected
flags, comma-separated, empty for none; "*" = any), suggest (yes/no/*) and
note (why: the case it guards). Prints every mismatch and exits 1 if there is
one. Samples missing from the run are reported, not failed (a new database
version may drop them).

Usage: uv run check_expected.py RUN EXPECTED
"""
import sys

import polars as pl

run, exp_path = sys.argv[1:3]
s = pl.read_csv(f"{run}/geo.samples.tsv", separator="\t", infer_schema_length=0)
e = pl.read_csv(exp_path, separator="\t", infer_schema_length=0, comment_prefix="#")
j = e.join(s.select(sample="Sample", got_bogus="bogus", got_flags="flags", got_suggest="suggest_lat",
                    diagnosis="diagnosis"), on="sample", how="left")
missing = j.filter(pl.col("got_bogus").is_null())
j = j.filter(pl.col("got_bogus").is_not_null())
bad = []
for r in j.iter_rows(named=True):
    why = []
    if r["bogus"] != r["got_bogus"]:
        why.append(f"bogus {r['got_bogus']}, expected {r['bogus']}")
    if (r["flags"] or "") != "*" and sorted((r["flags"] or "").split(",")) != sorted((r["got_flags"] or "").split(",")):
        why.append(f"flags '{r['got_flags'] or ''}', expected '{r['flags'] or ''}'")
    got_s = "yes" if r["got_suggest"] else "no"
    if (r["suggest"] or "*") != "*" and r["suggest"] != got_s:
        why.append(f"suggestion {got_s}, expected {r['suggest']}")
    if why:
        bad.append(f"{r['sample']}: {'; '.join(why)}  [{r['note']}]")
print(f"{j.height} expected cases checked, {len(bad)} mismatches, {missing.height} not in this run")
for b in bad:
    print("  FAIL", b)
for r in missing.iter_rows(named=True):
    print("  missing", r["sample"], f"[{r['note']}]")
sys.exit(1 if bad else 0)
