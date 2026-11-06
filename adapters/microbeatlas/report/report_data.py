# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "geopandas", "pyarrow", "shapely"]
# ///
"""Build the microbeatlas coordinate report: every number into one JSON,
inlined into template.html.

Usage: uv run report_data.py RUN OUT.html [--total N]
  --total  sample records in samples.info (default: count them in
           RUN/download/samples.info.gz)
Needs METALOG_GEOCODING_DIR (basemap from its Natural Earth layer).
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import polars as pl
import shapely

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--total", type=int)
args = ap.parse_args()
MG = os.environ.get("METALOG_GEOCODING_DIR") or sys.exit("set METALOG_GEOCODING_DIR (see .env.example)")
if args.total is None:
    with gzip.open(f"{args.run}/download/samples.info.gz", "rb") as fh:
        args.total = sum(1 for line in fh if line.startswith(b">"))
r = dict(separator="\t", infer_schema_length=0, quote_char=None)
s = pl.read_csv(f"{args.run}/geo.samples.tsv", **r).with_columns(
    lat=pl.col("latitude").cast(pl.Float64, strict=False),
    lon=pl.col("longitude").cast(pl.Float64, strict=False),
    cd=pl.col("coast_distance_km").cast(pl.Float64, strict=False),
    bogus=pl.col("bogus") == "true",
    flags=pl.col("flags").fill_null(""),
    fix_lat=pl.col("fix_lat").cast(pl.Float64, strict=False),
    fix_lon=pl.col("fix_lon").cast(pl.Float64, strict=False))
# Where the coordinate is, as specific as needed: admin-1 and country on land,
# the sea otherwise.
s = s.with_columns(where=pl.when(pl.col("country").is_not_null())
                   .then(pl.concat_str([pl.col("admin_1"), pl.col("country")], separator=", ", ignore_nulls=True))
                   .otherwise(pl.col("sea")))
D = {}

# --- overview ---------------------------------------------------------------
D["n_samples_total"] = args.total
D["n_with_coords"] = s.height
D["n_with_claim"] = int(s["claim"].is_not_null().sum())
D["n_coords"] = s.select("latitude", "longitude").unique().height
D["n_studies"] = int(s["study"].n_unique())
D["n_bogus"] = int(s["bogus"].sum())
D["n_bogus_studies"] = int(s.filter("bogus")["study"].n_unique())
D["n_coast"] = int(s["cd"].is_not_null().sum())

status = (pl.when(pl.col("bogus")).then(pl.lit("flagged"))
            .when(pl.col("code") == "ok").then(pl.lit("consistent"))
            .when(pl.col("code") == "accepted_exception").then(pl.lit("consistent"))
            .when(pl.col("code") == "not_on_land").then(pl.lit("coastal"))
            .otherwise(pl.lit("unchecked")))
s = s.with_columns(status=status)
D["status"] = dict(s.group_by("status").len().rows())

FLAGS = ["out_of_range", "null_island", "autofill", "country_mismatch", "sea_mismatch",
         "terrestrial_at_sea", "region_mismatch"]
D["flags"] = {f: int(s["flags"].str.contains(rf"\b{f}\b").sum()) for f in FLAGS}
D["flag_studies"] = {f: int(s.filter(pl.col("flags").str.contains(rf"\b{f}\b"))["study"].n_unique())
                     for f in FLAGS}
prim = pl.lit(None, dtype=pl.Utf8)
for f in reversed(FLAGS):
    prim = pl.when(pl.col("flags").str.contains(rf"\b{f}\b")).then(pl.lit(f)).otherwise(prim)
s = s.with_columns(primary=prim)
D["primary"] = dict(s.filter("bogus").group_by("primary").len().rows())

diag = s.filter(pl.col("diagnosis").is_not_null() & (pl.col("diagnosis") != "")).with_columns(
    dkind=pl.col("diagnosis").str.replace(r"\(.*$", ""))
D["diagnosis"] = dict(diag.group_by("dkind").len().rows())
D["n_diagnosed"] = diag.height

# --- map points: one per distinct coordinate and primary flag ---------------
def corrected(kind, la, lo):
    return {"lon_sign": (la, -lo), "lat_sign": (-la, lo), "both_signs": (-la, -lo),
            "swapped": (lo, la), "swapped_lat_sign": (-lo, la),
            "swapped_lon_sign": (lo, -la)}.get(kind)

# distance to what the metadata claims: the region for region mismatches,
# else the claimed country/sea
dist_note = (pl.when(pl.col("primary") == "region_mismatch")
               .then(pl.format("{} is {} km away", pl.col("region_claimed"), pl.col("region_distance_km")))
               .when(pl.col("claim_distance_km").cast(pl.Float64, strict=False) > 0)
               .then(pl.format("{} is {} km away", pl.col("claim"),
                               pl.col("claim_distance_km").cast(pl.Float64, strict=False).round(0).cast(pl.Int64))))
pts = (s.filter(pl.col("bogus") & (pl.col("primary") != "out_of_range") & pl.col("lat").is_not_null())
        .with_columns(dkind=pl.col("diagnosis").str.replace(r"\(.*$", ""), dnote=dist_note)
        .group_by("lat", "lon", "primary")
        .agg(n=pl.len(), st=pl.col("study").n_unique(), study=pl.col("study").mode().first(),
             claim=pl.col("name_value").mode().first(), where=pl.col("where").first(),
             dk=pl.col("dkind").mode().first(), dnote=pl.col("dnote").mode().first(),
             fx=pl.struct("fix_lat", "fix_lon").filter(pl.col("fix_lat").is_not_null()).mode().sort().first(),
             st_type=pl.col("scientific_name").mode().first(),
             sg=pl.struct("suggest_source", "suggest_lat", "suggest_lon", "suggest_uncertainty_km",
                           "fix_place_support", "fix_place", "fix_place_km",
                           "suggest_basis", "claim")
                 .filter(pl.col("suggest_source").is_not_null()).mode().sort().first(),
             ps=pl.col("place_match").filter(pl.col("fix_place_support") == "original_fits").first(),
             nsg=pl.col("suggest_source").is_not_null().sum())
        .sort("n"))
P_ = []
for row in pts.iter_rows(named=True):
    rec = [round(row["lat"], 4), round(row["lon"], 4), FLAGS.index(row["primary"]), row["n"], row["st"],
           row["study"] or "", row["claim"] or "", row["where"] or "", row["dk"] or "", row["dnote"] or "",
           row["st_type"] or ""]
    fx = row["fx"]
    rec.append([round(fx["fix_lat"], 4), round(fx["fix_lon"], 4)] if fx else None)
    sg = row["sg"]
    rec.append([sg["suggest_source"], sg["suggest_lat"], sg["suggest_lon"], sg["suggest_uncertainty_km"],
                sg["fix_place_support"], sg["fix_place"], sg["fix_place_km"],
                sg["suggest_basis"], sg["claim"]]
               if sg and sg["suggest_source"] else None)
    rec.append(row["ps"])
    rec.append(row["nsg"])
    P_.append(rec)
D["points"] = P_
D["point_cols"] = ["lat", "lon", "flag", "n", "studies", "study", "claim", "found", "diagnosis", "distance", "sample_type", "fix", "suggest"]

# --- top studies --------------------------------------------------------------
st = (s.group_by("study").agg(n=pl.len(), bad=pl.col("bogus").sum(),
                               claim=pl.col("name_value").mode().first(),
                               sci=pl.col("scientific_name").mode().first())
        .filter(pl.col("bad") > 0))
sf = (s.filter("bogus").group_by("study", "primary").len().sort("len", descending=True)
        .unique("study", keep="first").select("study", top_flag="primary"))
sd = (diag.group_by("study", "dkind").len().sort("len", descending=True)
          .unique("study", keep="first").select("study", top_diag="dkind", n_diag="len"))
sfound = (s.filter("bogus").with_columns(found=pl.col("where"))
           .group_by("study", "found").len().sort("len", descending=True)
           .unique("study", keep="first").select("study", "found"))
st = (st.join(sf, on="study", how="left").join(sd, on="study", how="left")
        .join(sfound, on="study", how="left").sort("bad", descending=True).head(40))
D["studies"] = [[r["study"], r["n"], r["bad"], r["top_flag"], r["top_diag"] or "", r["n_diag"] or 0,
                 r["claim"] or "", r["found"] or "", r["sci"] or ""] for r in st.iter_rows(named=True)]

# --- claimed vs found (country mismatches) -------------------------------------
cm = (s.filter(pl.col("flags").str.contains("country_mismatch"))
       .with_columns(found=pl.coalesce("country", "sea"))
       .group_by("claim", "found").agg(n=pl.len(), st=pl.col("study").n_unique(),
                                       dk=pl.col("diagnosis").str.replace(r"\(.*$", "").mode().first())
       .sort("n", descending=True).head(15))
D["pairs"] = [[r["claim"], r["found"] or "", r["n"], r["st"], r["dk"] or ""] for r in cm.iter_rows(named=True)]

# --- shared coordinates (one point, many claimed countries) ---------------------
h = (s.filter(pl.col("claim").is_not_null())
      .group_by("latitude", "longitude")
      .agg(st=pl.col("study").n_unique(), cl=pl.col("claim").n_unique(), n=pl.len(),
           bad=pl.col("bogus").sum(), found=pl.coalesce("country", "sea").first(),
           top=pl.col("claim").value_counts(sort=True).head(4).struct.field("claim"))
      .filter(pl.col("cl") >= 5).sort("cl", "n", descending=True).head(12))
D["hotspots"] = [[float(r["latitude"]), float(r["longitude"]), r["cl"], r["st"], r["n"], r["bad"],
                  r["found"] or "", list(r["top"])] for r in h.iter_rows(named=True)]

# --- coast distance histograms by sample type ----------------------------------
edges = [-3000, -1000, -300, -100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100, 300, 1000, 3000]
cdf = s.filter(pl.col("cd").is_not_null())
hist = {}
for t in ["marine", "terrestrial", "unknown"]:
    x = cdf.filter(pl.col("sample_type") == t)["cd"]
    counts = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo_ok = x >= a if a != edges[0] else x > -1e9
        hi_ok = x < b if b != edges[-1] else x < 1e9
        counts.append(int((lo_ok & hi_ok).sum()))
    hist[t] = counts
D["coast_edges"] = edges
D["coast_hist"] = hist
D["coast_type_n"] = {t: int((cdf["sample_type"] == t).sum()) for t in ["marine", "terrestrial", "unknown"]}
D["marine_inland_25"] = int(cdf.filter((pl.col("sample_type") == "marine") & (pl.col("cd") < -25)).height)
D["marine_inland_25_studies"] = int(cdf.filter((pl.col("sample_type") == "marine") & (pl.col("cd") < -25))["study"].n_unique())
D["coast_median"] = {t: float(cdf.filter(pl.col("sample_type") == t)["cd"].median()) for t in ["marine", "terrestrial", "unknown"]}

# --- autofill case: ERP114733 ----------------------------------------------------
a = (s.filter(pl.col("study") == "ERP114733").group_by("lat", "lon")
      .agg(n=pl.len(), af=pl.col("flags").str.contains("autofill").any(),
           found=pl.coalesce("country", "sea").first(), claim=pl.col("name_value").mode().first())
      .sort("lat"))
D["case"] = {"study": "ERP114733", "sample": "ERS3363330",
             "points": [[r["lat"], r["lon"], r["n"], bool(r["af"]), r["found"] or ""] for r in a.iter_rows(named=True)],
             "claim": s.filter(pl.col("Sample") == "ERS3363330")["name_value"][0]}

# --- unplaceable claims ------------------------------------------------------------
u = (s.filter(pl.col("code") == "unknown_place").group_by("claim_raw").len()
      .sort("len", descending=True))
D["unplaceable"] = [[a_, b_] for a_, b_ in u.head(12).rows()]
D["n_unplaceable"] = int(u["len"].sum())
D["n_unplaceable_claims"] = u.height
D["n_remapped"] = int(s.filter(pl.col("claim_raw") != pl.col("claim")).height)
D["n_claims_remapped"] = int(s.filter(pl.col("claim_raw") != pl.col("claim"))["claim_raw"].n_unique())
D["n_out_of_range_coords"] = int(s.filter(pl.col("flags").str.contains("out_of_range")).select("latitude", "longitude").unique().height)

# --- lenient rules: samples that pass only thanks to an allowance -------------
# place rescues (cleared because the coordinate lies in a sampled place) are reported apart
prs = s.filter(~pl.col("bogus") & pl.col("place_match").is_not_null())
D["place_rescue"] = {k: int(prs["lenient"].str.contains(k).sum()) for k in
                     ["country_mismatch", "sea_mismatch", "region_mismatch"]}
D["n_place_rescue"], D["n_place_rescue_studies"] = prs.height, prs["study"].n_unique()
D["place_rescue_top"] = [[r["claim"] or "", r["found"] or "", r["place_match"], r["len"], r["st"]] for r in
    prs.with_columns(found=pl.coalesce("country", "sea")).group_by("claim", "found", "place_match")
       .agg(pl.len(), st=pl.col("study").n_unique()).sort("len", descending=True).head(8).iter_rows(named=True)]
fx_ = s.filter(pl.col("bogus") & pl.col("fix_lat").is_not_null())
D["fix_support"] = {f"{r['suggest_source']}|{r['fix_place_support']}": r["len"] for r in
                    fx_.group_by("suggest_source", "fix_place_support").len().iter_rows(named=True)}
D["n_fix_studies_places"] = int(fx_.filter(~pl.col("fix_place_support").is_in(["no_places"]))["study"].n_unique())
len_ = s.filter(pl.col("lenient").is_not_null() & (pl.col("lenient") != "") & (pl.col("lenient") != '""')
                & ~(~pl.col("bogus") & pl.col("place_match").is_not_null()))
D["lenient"] = {k: int(len_["lenient"].str.contains(k).sum()) for k in
                ["country_mismatch", "sea_mismatch", "region_mismatch", "terrestrial_at_sea"]}
D["n_lenient"] = len_.height
sb = s.filter("bogus").with_columns(u=pl.col("suggest_uncertainty_km").cast(pl.Float64, strict=False))
D["suggest"] = {r["suggest_source"] or "none": [r["len"], r["u"]] for r in
                sb.group_by("suggest_source").agg(pl.len(), u=pl.col("u").median()).iter_rows(named=True)}
D["n_fix_withheld"] = int(s.filter(pl.col("bogus") & (pl.col("fix_place_support") == "original_fits")).height)
sgd = pl.col("suggest_source").is_not_null()
D["n_fix"] = int(s.filter(pl.col("bogus") & sgd).height)
D["n_fix_tas"] = int(s.filter(pl.col("flags").str.contains("terrestrial_at_sea") & sgd).height)
D["n_fix_unbacked"] = int(s.filter(pl.col("bogus") & pl.col("fix_lat").is_not_null() & ~sgd).height)
D["basis"] = dict(s.filter(pl.col("bogus") & sgd).group_by("suggest_basis").len().rows())
D["n_tas"] = int(s["flags"].str.contains("terrestrial_at_sea").sum())
D["n_offshore_far"] = int(s.filter(pl.col("flags").str.contains("country_mismatch") & (pl.col("code") == "not_on_land")).height)

# --- basemap: Natural Earth countries, simplified ------------------------------------
w = gpd.read_parquet(f"{MG}/data/ne_10m_admin_0_countries.parquet")[["geometry"]]
land = shapely.unary_union(w.geometry.values)
# detailed enough to zoom in to a region (~5 km; islands down to ~60 km2)
land = shapely.simplify(land, 0.05, preserve_topology=True)
land = shapely.set_precision(land, 0.01)
geoms = [g for g in getattr(land, "geoms", [land]) if g.area > 0.005]
def ring(c):
    return [[round(x, 2), round(y, 2)] for x, y in c]
mp = [[ring(g.exterior.coords)] + [ring(i.coords) for i in g.interiors] for g in geoms]
D["land"] = {"type": "MultiPolygon", "coordinates": mp}

tpl = (Path(__file__).parent / "template.html").read_text()
assert tpl.count("/*DATA*/") == 1
# "</" inside the JSON would end the script element
Path(args.out).write_text(tpl.replace("/*DATA*/", json.dumps(D, separators=(",", ":")).replace("</", "<\\/")))
print("wrote", args.out, file=sys.stderr)
print({k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 200 else f"<{len(v)}>") for k, v in D.items()})
