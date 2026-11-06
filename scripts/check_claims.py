# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "tqdm"]
# ///
"""Grade every distinct (lat, lon, claimed country/sea) pair in
RUN/check_input.tsv against the metalog-geocoding service (POST /check).

The input comes from build_check_input.py: columns id, latitude, longitude,
geographic_location (the part of the stated name before the first colon),
n_samples.

Outputs (appended per batch, so an interrupted run resumes where it stopped):
  <prefix>.verdicts.tsv   one row per (latitude, longitude, geographic_location):
                          grade/code/reason/claim_type; pairs already in it are
                          not sent again, so it doubles as a cache across runs
                          (delete it after upgrading the service)
  <prefix>.hierarchy.tsv  location_hierarchy rows (long format)

The service URL comes from METALOG_GEOCODING_URL (see .env.example) or --url.

Usage:
  uv run check_claims.py --input FILE --prefix PREFIX [--batch-size N]
"""

import argparse
import csv
import os
import sys
import time

import requests
from tqdm import tqdm

VERDICT_COLS = ["id", "latitude", "longitude", "geographic_location",
                "grade", "code", "reason", "claim_type"]


def read_input(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def key(row):
    return (row["latitude"], row["longitude"], row["geographic_location"])


def read_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {key(row) for row in csv.DictReader(fh, delimiter="\t")}


def post_with_retry(session, url, payload, timeout, retries=5):
    for attempt in range(retries):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if r.status_code < 500:
                r.raise_for_status()  # 4xx is a caller bug: fail immediately
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except (requests.ConnectionError, requests.Timeout) as exc:
            err = str(exc)
        wait = 2 ** attempt * 5
        tqdm.write(f"request failed ({err}); retrying in {wait}s")
        time.sleep(wait)
    sys.exit(f"giving up after {retries} attempts: {err}")


def open_tsv(path, cols):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                       extrasaction="ignore", lineterminator="\n")
    if new:
        w.writeheader()
    return fh, w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--url", default=os.environ.get("METALOG_GEOCODING_URL"))
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    rows = read_input(args.input)
    verd_path = f"{args.prefix}.verdicts.tsv"
    hier_path = f"{args.prefix}.hierarchy.tsv"
    done = read_done(verd_path)
    todo = [r for r in rows if key(r) not in done]
    print(f"{len(rows)} pairs; {len(rows) - len(todo)} already done, {len(todo)} to go",
          file=sys.stderr)
    if not todo:
        return
    if not args.url:
        sys.exit("set METALOG_GEOCODING_URL (see .env.example) or pass --url")
    base = args.url.rstrip("/")
    session = requests.Session()
    session.headers["User-Agent"] = "sample-location-qc/0.1 (claim check)"
    health = session.get(f"{base}/health", timeout=30).json()
    if health.get("status") != "ok":
        sys.exit(f"service not healthy: {health}")

    verd_fh, verd_w = open_tsv(verd_path, VERDICT_COLS)
    hier_fh = hier_w = None
    try:
        with tqdm(total=len(todo), unit="pair", smoothing=0.05) as bar:
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i:i + args.batch_size]
                coords = sorted({(float(r["latitude"]), float(r["longitude"]))
                                 for r in batch})
                index = {c: k for k, c in enumerate(coords)}
                payload = {
                    "coordinates": [list(c) for c in coords],
                    "samples": [{"id": r["id"],
                                 "coord": index[(float(r["latitude"]),
                                                 float(r["longitude"]))],
                                 "geographic_location": r["geographic_location"]}
                                for r in batch],
                }
                body = post_with_retry(session, f"{base}/check", payload,
                                       args.timeout)
                verdicts = body["verdicts"]
                if len(verdicts) != len(batch):
                    sys.exit(f"expected {len(batch)} verdicts, got {len(verdicts)}")
                hier = body.get("location_hierarchy") or []
                if hier and hier_w is None:
                    hier_fh, hier_w = open_tsv(hier_path, list(hier[0]))
                # Hierarchy first, verdicts second: an id only counts as done
                # once its verdict row is written.
                if hier_w is not None:
                    hier_w.writerows(hier)
                    hier_fh.flush()
                by_id = {r["id"]: r for r in batch}
                for v in verdicts:
                    verd_w.writerow({**by_id[v["id"]], **v})
                verd_fh.flush()
                bar.update(len(batch))
    finally:
        verd_fh.close()
        if hier_fh is not None:
            hier_fh.close()


if __name__ == "__main__":
    main()
