# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "requests", "pycountry"]
# ///
"""Distances from flagged coordinates (and their swap / sign-flip variants and
autofill bases) to the places named in each study's description.

Inputs: RUN/geo.samples.tsv (bogus rows), RUN/study_places.tsv or else
curated/study_places.tsv (output of ena_places.py from
https://github.com/grp-bork/toponym_ner; rows with sampled == true, `corrected`
name: origin "study"), and each sample's own location detail, the
comma-separated names after the colon of its location and in its
sampling_site (RUN/samples.tsv) (origin "sample"). Calls POST /place-distance
of the metalog-geocoding service (METALOG_GEOCODING_URL, see .env.example);
responses are cached in --cache keyed by (name, country, lat, lon), so reruns
only send what is new.

Outputs: RUN/studies_without_places.txt, flagged studies missing from
study_places.tsv (input for ena_places.py -f), and
RUN/place_distances.tsv, one row per study, original coordinate,
variant and place: the closest candidate's label/source/geometry/area and the
distance in km (null: impossible coordinate or unresolved name, see status).
"""

import argparse
import json
import os
import time

import polars as pl
import pycountry
import requests

UA = "sample-location-qc/0.1 (place-distance batch)"
MAX_COORDS = 64
MAX_PLACES = 300          # per request (service limit 1000)
VARIANTS = {"original": lambda la, lo: (la, lo), "swapped": lambda la, lo: (lo, la),
            "lat_sign": lambda la, lo: (-la, lo), "lon_sign": lambda la, lo: (la, -lo),
            "both_signs": lambda la, lo: (-la, -lo), "swapped_lat_sign": lambda la, lo: (-lo, la),
            "swapped_lon_sign": lambda la, lo: (lo, -la), "swapped_both_signs": lambda la, lo: (-lo, -la)}
COUNTRY_FIX = {"USA": "US", "UK": "GB", "Russia": "RU", "South Korea": "KR", "North Korea": "KP",
               "Iran": "IR", "Vietnam": "VN", "Laos": "LA", "Bolivia": "BO", "Venezuela": "VE",
               "Tanzania": "TZ", "Syria": "SY", "Taiwan": "TW", "Czech Republic": "CZ",
               "Democratic Republic of the Congo": "CD", "Republic of the Congo": "CG",
               "Turkey": "TR", "Moldova": "MD", "Micronesia": "FM", "Brunei": "BN",
               "Cote d'Ivoire": "CI", "Ivory Coast": "CI", "Palestine": "PS", "Kosovo": "XK"}


def iso2(name):
    if name in COUNTRY_FIX:
        return COUNTRY_FIX[name]
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return None


def rnd(x):
    return round(float(x), 6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--url", default=os.environ.get("METALOG_GEOCODING_URL"))
    a = ap.parse_args()
    a.samples, a.places = f"{a.run}/geo.samples.tsv", f"{a.run}/study_places.tsv"
    if not os.path.exists(a.places):
        a.places = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "curated", "study_places.tsv")
    a.sites, a.out = f"{a.run}/samples.tsv", f"{a.run}/place_distances.tsv"

    s_cols = pl.read_csv(a.samples, separator="\t", n_rows=0).columns
    s = (pl.read_csv(a.samples, separator="\t", infer_schema_length=0)
         # flagged rows, plus rows flag_coords.py cleared on this evidence before
         .filter((pl.col("bogus") == "true")
                 | (pl.col("place_match").is_not_null() if "place_match" in s_cols else pl.lit(False))))
    pts = (s.select("study", "latitude", "longitude").unique()
           .with_columns(pl.col("latitude", "longitude").cast(pl.Float64)))
    fixes = (s.filter(pl.col("diagnosis").str.starts_with("autofill") & pl.col("fix_lat").is_not_null())
             .select("study", "latitude", "longitude", suggest_lat="fix_lat", suggest_lon="fix_lon").unique()
             .with_columns(pl.col("latitude", "longitude", "suggest_lat", "suggest_lon").cast(pl.Float64)))
    # one country hint per study: its only claimed country, if it has exactly one
    hint = (s.filter(pl.col("claim_kind") == "country").group_by("study")
            .agg(pl.col("claim").unique()).filter(pl.col("claim").list.len() == 1)
            .with_columns(pl.col("claim").list.first()))
    hint = {r["study"]: iso2(r["claim"]) for r in hint.iter_rows(named=True)}
    print("country hints:", sum(v is not None for v in hint.values()), "of", len(hint),
          "unmapped:", sorted({k for k, v in hint.items() if v is None})[:20])

    # study_places.tsv is the output of toponym_ner's ena_places.py: a study
    # may be named by its project or its study accession
    places, looked_up = {}, set()
    if os.path.exists(a.places):
        sp = pl.read_csv(a.places, separator="\t", infer_schema_length=0, quote_char=None)
        looked_up = set(sp["study_accession"].drop_nulls()) | set(sp["secondary_study_accession"].drop_nulls())
        sp = (sp.filter((pl.col("sampled") == "true") & pl.col("corrected").is_not_null())
                .unpivot(index="corrected", on=["study_accession", "secondary_study_accession"], value_name="study")
                .filter(pl.col("study").is_not_null())
                .select("study", place="corrected").unique())
        for r in sp.iter_rows(named=True):
            places.setdefault(r["study"], []).append(r["place"])
    # flagged studies never run through ena_places.py: input for the next run of it
    todo = sorted(set(pts["study"].drop_nulls()) - looked_up)
    with open(f"{a.run}/studies_without_places.txt", "w") as fh:
        fh.writelines(st + "\n" for st in todo)
    print("flagged studies without a place lookup:", len(todo), f"-> {a.run}/studies_without_places.txt")

    # the sample's own location detail: "Antarctica: Port-aux-Francais, Kerguelen
    # Islands" -> Port-aux-Francais, Kerguelen Islands (hint: the claimed country)
    det = (s.filter(pl.col("name_value").str.contains(":"))
            .select("study", "latitude", "longitude", "claim", "claim_kind",
                    tok=pl.col("name_value").str.splitn(":", 2).struct.field("field_1")
                          .str.replace_all(r"[:;]", ",").str.split(","))
            .explode("tok").with_columns(pl.col("tok").str.strip_chars(" .\"'"))
            .filter(pl.col("tok").str.len_chars() >= 2
                    & ~pl.col("tok").str.contains(r"(?i)^(gaz|n/?a|none|missing|unknown|not [a-z ]+|-?[\d.]+)$"))
            .with_columns(pl.col("latitude", "longitude").cast(pl.Float64)).unique())
    own = {}
    for r in det.iter_rows(named=True):
        c = iso2(r["claim"]) if r["claim_kind"] == "country" and r["claim"] else None
        own.setdefault((r["study"], r["latitude"], r["longitude"]), set()).add((r["tok"], c))
    # the sample's sampling_site (extract_sampling_site.py), split at commas;
    # no country hint, the claimed country is what is in doubt
    if os.path.exists(a.sites):
        sites = (pl.read_csv(a.sites, separator="\t", infer_schema_length=0, quote_char=None)
                   .select(Sample="sample", sampling_site="sampling_site")
                   .filter(pl.col("sampling_site").is_not_null())
                   .join(s.select("Sample", "study", "latitude", "longitude"), on="Sample")
                   .with_columns(tok=pl.col("sampling_site").str.replace_all(r"[;|]", ",").str.split(","))
                   .explode("tok").with_columns(pl.col("tok").str.strip_chars(" .\"'"))
                   .filter((pl.col("tok").str.len_chars() >= 3) & pl.col("tok").str.contains(r"[A-Za-z]{3}")
                           & ~pl.col("tok").str.contains(r"(?i)^(site|station|stn|st|plot|transect|sample|point|reef)\s*[\w-]*\d")
                           & ~pl.col("tok").str.contains(r"(?i)^(missing|unknown|not [a-z ]+|n/?a|none)$"))
                   .with_columns(pl.col("latitude", "longitude").cast(pl.Float64))
                   .select("study", "latitude", "longitude", "tok").unique())
        for r in sites.iter_rows(named=True):
            own.setdefault((r["study"], r["latitude"], r["longitude"]), set()).add((r["tok"], None))
        print("sampling sites:", sites.select("study", "latitude", "longitude").unique().height,
              "coordinates,", sites["tok"].n_unique(), "names")
    print("sample location details:", len(own), "coordinates,", det["tok"].n_unique(), "names")

    rows = []                     # (study, lat, lon, variant, vlat, vlon, place, country, origin)
    for r in pts.iter_rows(named=True):
        st, la, lo = r["study"], r["latitude"], r["longitude"]
        targets = [(p, hint.get(st), "study") for p in places.get(st, [])]
        targets += [(p, c, "sample") for p, c in sorted(own.get((st, la, lo), set()), key=str)]
        for p, c, origin in targets:
            for v, f in VARIANTS.items():
                vla, vlo = f(la, lo)
                rows.append((st, la, lo, v, rnd(vla), rnd(vlo), p, c, origin))
    for r in fixes.iter_rows(named=True):
        st, la, lo = r["study"], r["latitude"], r["longitude"]
        targets = [(p, hint.get(st), "study") for p in places.get(st, [])]
        targets += [(p, c, "sample") for p, c in sorted(own.get((st, la, lo), set()), key=str)]
        for p, c, origin in targets:
            rows.append((st, la, lo, "autofill_base", rnd(r["suggest_lat"]), rnd(r["suggest_lon"]), p, c, origin))
    print("query rows:", len(rows), "studies with places:", len(places),
          "flagged studies with places:", pts.filter(pl.col("study").is_in(list(places)))["study"].n_unique())

    cache = {}
    if os.path.exists(a.cache):
        with open(a.cache) as fh:
            for line in fh:
                e = json.loads(line)
                cache[(e["name"], e["country"], e["lat"], e["lon"])] = e
    need = {}
    for st, la, lo, v, vla, vlo, p, c, origin in rows:
        if (p, c, vla, vlo) not in cache:
            need.setdefault((p, c), set()).add((vla, vlo))
    entries = []
    for (p, c), cs in need.items():
        cs = sorted(cs)
        for i in range(0, len(cs), MAX_COORDS):
            entries.append((p, c, cs[i:i + MAX_COORDS]))
    print("to send:", sum(len(e[2]) for e in entries), "coordinates in", len(entries), "place entries")

    if entries and not a.url:
        raise SystemExit("set METALOG_GEOCODING_URL (see .env.example) or pass --url")
    url = f"{(a.url or '').rstrip('/')}/place-distance"
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Content-Type": "application/json"})
    with open(a.cache, "a") as out:
        pending = entries
        rounds = 0
        while pending and rounds < 50:
            rounds += 1
            again = []
            for i in range(0, len(pending), MAX_PLACES):
                chunk = pending[i:i + MAX_PLACES]
                body = {"places": [{"name": p, "coordinates": [list(x) for x in cs],
                                    **({"country": c} if c else {})} for p, c, cs in chunk]}
                for attempt in range(6):
                    try:
                        resp = sess.post(url, json=body, timeout=900)
                        resp.raise_for_status()
                        break
                    except requests.RequestException as e:
                        print("retry", attempt, e, flush=True)
                        time.sleep(5 * 2 ** attempt)
                else:
                    raise SystemExit("place-distance unreachable")
                for (p, c, cs), res in zip(chunk, resp.json()["places"]):
                    if res["status"] in ("pending", "lookup_failed"):
                        again.append((p, c, cs))
                        continue
                    for k, (la, lo) in enumerate(cs):
                        best = None
                        for cand in res.get("candidates", []):
                            d = cand["distances_m"][k]
                            if d is not None and (best is None or d < best[0]):
                                best = (d, cand)
                        e = {"name": p, "country": c, "lat": la, "lon": lo, "status": res["status"],
                             "interpreted_as": res.get("interpreted_as"),
                             "n_candidates": len(res.get("candidates", [])),
                             "distance_km": None if best is None else round(best[0] / 1000, 3)}
                        if best:
                            cd = best[1]
                            e.update(label=cd["label"], source=cd["source"], kind=cd["kind"],
                                     geometry=cd["geometry"], area_km2=cd["area_km2"], cand_id=cd["id"])
                        cache[(p, c, la, lo)] = e
                        out.write(json.dumps(e) + "\n")
                out.flush()
                time.sleep(0.4)
            print(f"round {rounds}: sent {len(pending)} entries, {len(again)} pending", flush=True)
            pending = again
            if pending:
                time.sleep(10)
    if pending:
        print("WARNING: still pending after retries:", len(pending))

    recs = []
    for st, la, lo, v, vla, vlo, p, c, origin in rows:
        e = cache.get((p, c, vla, vlo), {"status": "pending"})
        recs.append(dict(study=st, latitude=la, longitude=lo, variant=v, var_lat=vla, var_lon=vlo,
                         place=p, origin=origin, country_hint=c, status=e["status"],
                         interpreted_as=e.get("interpreted_as"), label=e.get("label"),
                         source=e.get("source"), kind=e.get("kind"), geometry=e.get("geometry"),
                         area_km2=e.get("area_km2"), distance_km=e.get("distance_km")))
    df = pl.DataFrame(recs, infer_schema_length=None).sort("study", "latitude", "longitude", "variant", "place", "origin")
    df.write_csv(a.out, separator="\t")
    print("wrote", a.out, df.height, "rows")
    print(df.filter(pl.col("variant") == "original").group_by("status").len())


if __name__ == "__main__":
    main()
