# /// script
# requires-python = ">=3.10"
# dependencies = ["polars"]
# ///
"""Convert the microbeatlas downloads into the generic samples.tsv.

Inputs: samples.info.gz (one ">SAMPLE" record per sample, key=value lines)
and samples.info.latlon.parsed.tsv (Sample, LatitudeParsed, LongitudeParsed).
Output: one row per sample with a coordinate, columns as in the README:
  sample, study, latitude, longitude, location_field, location, sampling_site,
  scientific_name, env_biome, env_feature, env_material, isolation_source

location is the first non-missing value among LOCATION_FIELDS (INSDC-style
"Country: detail" fields, in priority order). sampling_site accepts the keys
sampling_site, SamplingSite and name of the sampling site (case, spaces and
underscores ignored). Context fields keep the first value; scientific name and
study keep the last, as the record lists them.

Usage: uv run prepare.py samples.info.gz samples.info.latlon.parsed.tsv OUT.tsv
"""
import gzip
import re
import sys

import polars as pl

LOCATION_FIELDS = [  # priority order; country/sea-level INSDC-style fields only
    "geo_loc_name",
    "geographic location (country and/or sea)",
    "geo loc name",
    "geographic location (country and/or sea,region)",
    "geographic location (country and/or sea, region)",
    "geographic location",
    "Geographic location",
    "Geographic location (country:region,area)",
    "geo: loc: name",
    "country",
]
PRIO = {k: i for i, k in enumerate(LOCATION_FIELDS)}
MISSING = {"missing", "not collected", "not applicable", "na", "n/a", "none", "unknown",
           "not available", "not provided", "-", "restricted access", "not determined", "null"}
SITE_KEYS = {"samplingsite", "nameofthesamplingsite"}
SITE_SKIP = re.compile(r"^(missing|not applicable|not provided|not collected|na|n/a|none|unknown|-)\b", re.I)
CONTEXT = {  # samples.info key -> column; the first one present wins
    "env_biome": "env_biome", "env_broad_scale": "env_biome",
    "env_feature": "env_feature", "env_local_scale": "env_feature",
    "env_material": "env_material", "env_medium": "env_material",
    "isolation_source": "isolation_source", "isolation source": "isolation_source",
}
COLS = ["sample", "study", "location_field", "location", "sampling_site", "scientific_name",
        "env_biome", "env_feature", "env_material", "isolation_source"]


def is_missing(v):
    lv = v.lower()
    return lv in MISSING or lv.startswith("missing") or lv.startswith("not ")


def records(path):
    """Yield one dict per sample record."""
    rec = None
    with gzip.open(path, "rt", errors="replace", newline="\n") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if rec:
                    yield rec
                rec = {"sample": line[1:].strip()}
                continue
            if rec is None or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = re.sub(r"[\t\r]", " ", v)
            if k == "study":
                rec["study"] = v
                continue
            if not k.startswith("sample_"):
                continue
            k = k[7:]
            if k == "SCIENTIFIC_NAME":
                rec["scientific_name"] = v
            elif k in CONTEXT and not rec.get(CONTEXT[k]):
                rec[CONTEXT[k]] = v
            vs = v.strip(" ")
            if not vs:
                continue
            if k in PRIO and not is_missing(vs) and PRIO[k] < rec.get("_prio", len(PRIO)):
                rec.update(_prio=PRIO[k], location_field=k, location=vs)
            if "sampling_site" not in rec and re.sub(r"[\s_]", "", k).lower() in SITE_KEYS \
                    and v.strip() and not SITE_SKIP.match(v.strip()):
                rec["sampling_site"] = v.strip()
    if rec:
        yield rec


def main(info, latlon, out):
    meta = pl.DataFrame([{c: r.get(c) for c in COLS} for r in records(info)],
                        schema={c: pl.Utf8 for c in COLS})
    ll = (pl.read_csv(latlon, separator="\t", infer_schema_length=0, null_values=["", "None"], quote_char=None)
            .filter(pl.col("LatitudeParsed").is_not_null() & pl.col("LongitudeParsed").is_not_null())
            .select(sample="Sample", latitude="LatitudeParsed", longitude="LongitudeParsed"))
    s = ll.join(meta, on="sample", how="left")
    s.select(COLS[:2] + ["latitude", "longitude"] + COLS[2:]).write_csv(out, separator="\t", quote_style="never")
    print(f"{meta.height} records, {s.height} with coordinates, "
          f"{s['location'].is_not_null().sum()} with a location -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
