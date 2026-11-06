# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "tqdm"]
# ///
"""Geocode distinct coordinates via the metalog geocoding service
(POST /geocode, type "best"): land or sea record plus the signed distance to
the coast (negative on land), whatever the sample claims.

Input: TSV with latitude and longitude columns (RUN/sample_claims.tsv).
Coordinates outside -90..90 / -180..180 are skipped (the service does not
range-check /geocode).

Output (appended per batch, so an interrupted run resumes where it stopped):
  <prefix>.tsv  long format: latitude, longitude, key, value; every
                coordinate gets a "type" row, which is the resume marker;
                coordinates already in it are not sent again, so it doubles
                as a cache across runs (delete it after upgrading the service)

The service URL comes from METALOG_GEOCODING_URL (see .env.example) or --url.

Usage:
  uv run geocode_best.py --input FILE --prefix PREFIX [--batch-size N]
"""

import argparse
import csv
import math
import os
import sys
import time

import requests
from tqdm import tqdm

COLS = ["latitude", "longitude", "key", "value"]


def read_coords(path, lat_col, lon_col):
    coords, skipped = set(), 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)
                    and -90 <= lat <= 90 and -180 <= lon <= 180):
                skipped += 1
                continue
            coords.add((lat, lon))
    return sorted(coords), skipped


def read_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row["key"] == "type":
                    done.add((float(row["latitude"]), float(row["longitude"])))
    return done


def post_with_retry(session, url, payload, timeout, retries=8):
    for attempt in range(retries):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if r.status_code < 500:
                r.raise_for_status()  # 4xx is a caller bug: fail immediately
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            err = str(exc)
        wait = min(2 ** attempt * 5, 300)
        tqdm.write(f"request failed ({err}); retrying in {wait}s")
        time.sleep(wait)
    sys.exit(f"giving up after {retries} attempts: {err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--lat-col", default="latitude")
    ap.add_argument("--lon-col", default="longitude")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--url", default=os.environ.get("METALOG_GEOCODING_URL"))
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    coords, skipped = read_coords(args.input, args.lat_col, args.lon_col)
    out_path = f"{args.prefix}.tsv"
    done = read_done(out_path)
    todo = [c for c in coords if c not in done]
    print(f"{len(coords)} coordinates ({skipped} out of range skipped); "
          f"{len(coords) - len(todo)} already done, {len(todo)} to go", file=sys.stderr)
    if not todo:
        return
    if not args.url:
        sys.exit("set METALOG_GEOCODING_URL (see .env.example) or pass --url")
    base = args.url.rstrip("/")
    session = requests.Session()
    session.headers["User-Agent"] = "sample-location-qc/0.1 (geocode)"

    new = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    fh = open(out_path, "a", newline="")
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    if new:
        w.writerow(COLS)
    try:
        with tqdm(total=len(todo), unit="coord", smoothing=0.05) as bar:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i:i + args.batch_size]
                body = post_with_retry(
                    session, f"{base}/geocode",
                    {"coordinates": [list(c) for c in batch], "type": "best"},
                    args.timeout)
                results = body["results"]
                if len(results) != len(batch):
                    sys.exit(f"expected {len(batch)} results, got {len(results)}")
                for (lat, lon), res in zip(batch, results):
                    rec = res.get("result") or {}
                    if res.get("error"):
                        rec = {"error": res["error"]}
                    # "type" last: a coordinate only counts as done once it is written.
                    rows = [(k, v) for k, v in rec.items() if k != "type" and v is not None]
                    rows.append(("type", rec.get("type", "none")))
                    w.writerows((lat, lon, k, v) for k, v in rows)
                fh.flush()
                bar.update(len(batch))
    finally:
        fh.close()


if __name__ == "__main__":
    main()
