# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "geopandas", "pyarrow", "shapely", "pyproj"]
# ///
"""Merge geocoding results into one coordinate table (distance to coast) and
flag bogus sample coordinates.

Inputs:
  RUN/sample_claims.tsv  from build_check_input.py
  RUN/samples.tsv        sample context (scientific name, env fields, study)
  --geo        long TSV from geocode_best.py (latitude, longitude, key, value)
  --verdicts   one or more verdict TSVs from check_claims.py (matched to samples
               by latitude, longitude and claim)
  --places     RUN/place_distances.tsv from place_distances.py (optional)
  Natural Earth / IHO layers and country aliases from a metalog-geocoding
  checkout: METALOG_GEOCODING_DIR (its data/ set up with scripts/setup_data.py)

Outputs:
  RUN/geo.coords.tsv     one row per distinct coordinate: land/sea, country,
                         admin-1, sea, coast distance (km, negative on land)
  RUN/geo.samples.tsv    one row per sample with a coordinate: claim, verdict,
                         region check, sample type, flags, diagnosis

Flags (any of them makes `bogus` true). Tolerances widen to precision_km, the
largest distance between a rounded coordinate and the true point (78.7 km for
whole degrees, 7.9 km for one decimal, 0.8 km for two):
  out_of_range        latitude/longitude cannot exist
  null_island         exactly (0, 0)
  country_mismatch    recognised country claim, coordinate in another country
                      (checker codes different_location / claim_too_far), or
                      at sea more than OFFSHORE_KM from the claimed country
                      (not_on_land)
  sea_mismatch        recognised sea claim, coordinate on land (not_on_sea) or
                      in a sea more than SEA_NEAR_KM from the claimed one
                      (unrelated_sea; RELATED_SEAS count as the claim; a
                      Southern Ocean claim south of SOUTHERN_OCEAN_LAT, and an
                      adjacent ocean claim inside the IHO Southern Ocean, pass)
  region_mismatch     right country, but the stated admin-1 region is more
                      than REGION_TOL_KM from the coordinate
  terrestrial_at_sea  sample type is terrestrial/host-associated, coordinate
                      is at sea more than SEA_TOL_KM offshore
  autofill            spreadsheet drag: within a study, a run of at least
                      AUTOFILL_MIN distinct coordinates with one constant
                      step, diagonal (|dlat| == |dlon|) or along one axis
                      (then only if some run member contradicts its
                      metadata, since transects look the same); the run's first
                      coordinate is the real one and is not flagged, nor are
                      members less than AUTOFILL_MIN_DEG from it (dragged last
                      decimals; autofill_step is still filled in for those)
`lenient` lists checks a sample passes only thanks to those allowances.
Diagnosis (for bogus rows): the first swap / sign flip that puts the coordinate
within DIAG_TOL_KM of the claimed country or sea (and inside the claimed
admin-1), giving fix_lat/fix_lon; for terrestrial samples at sea without a
usable claim, the only transform that lands on land ("<name>_to_land");
an autofill run instead gets the run's base coordinate as its fix; failing
both, a whole-degree offset from the study's most common coordinate (no fix).
Suggested position (bogus rows): the mechanical fix (suggest_*), only when
text backs it (suggest_basis): "claim", the fix lands in the sample's stated
country / sea (autofill: the base carries the same claim and passes);
"place", it lands in a place the study description names; or both.
Place evidence (--places, from place_distances.py): places the study
description names as sampled (study_places.tsv), and the names in the
sample's own location detail (after the colon of geo_loc_name). fix_place_support says
whether the fix lands in such a place the original coordinate is not in
(supported), only the original is in one (original_fits: then no position
is suggested, the description argues against moving it), neither is
(no_match), or no evidence (no_places, unresolved). A place given only as a
point (a trench, a basin) counts within PLACE_POINT_KM of it; a named country
or state within OFFSHORE_KM for marine samples. A country / sea / region
mismatch whose coordinate lies in a sampled place is cleared (listed in
`lenient`, place in place_match) unless a fix lands in another sampled place;
continents, and the claimed country itself, do not count, and a country claim
is only cleared for a coordinate at sea. Only coordinates
flagged in an earlier run (or cleared this way) are in that file, so rerun
place_distances.py, then this script, after other flag changes.
"""

import argparse
import importlib.util
import os
import re
import sys
import unicodedata

import geopandas as gpd
import polars as pl
import shapely
from pyproj import Geod

REGION_TOL_KM = 25.0
SEA_TOL_KM = 25.0
OFFSHORE_KM = 370.0       # 200 nautical miles: a country claim at sea is fine within its EEZ reach
SEA_NEAR_KM = 50.0        # a neighbouring sea within this distance is not an unrelated sea
DIAG_TOL_KM = 25.0        # a transformed coordinate this close to the claim counts as fitting
PLACE_TOL_KM = 25.0       # ... or this close to a place the study description names as sampled
PLACE_POINT_KM = 100.0    # ... or to its label point, when that is all there is (a trench, a basin)
SOUTHERN_OCEAN_LAT = -40.0  # Subtropical Front, the oceanographers' northern edge (IHO: 60 S)
SOUTHERN_NEIGHBOURS = ["Atlantic Ocean", "South Atlantic Ocean", "Indian Ocean", "Pacific Ocean",
                       "South Pacific Ocean"]
RELATED_SEAS = {          # claim -> IHO seas that also count as that claim
    "Baltic Sea": ["Kattegat", "Skagerrak", "Gulf of Bothnia", "Gulf of Finland", "Gulf of Riga"],
    "Sargasso Sea": ["North Atlantic Ocean"],
    # IHO splits the Arctic and northern Atlantic into marginal seas that
    # submitters still call by the ocean's name
    "Arctic Ocean": ["Barentsz Sea", "Beaufort Sea", "Bering Sea", "Chukchi Sea", "East Siberian Sea",
                     "Greenland Sea", "Kara Sea", "Laptev Sea", "Lincoln Sea", "Norwegian Sea",
                     "The Northwestern Passages", "White Sea", "Baffin Bay"],
    "Atlantic Ocean": ["Baffin Bay", "Davis Strait", "Greenland Sea", "Labrador Sea", "Norwegian Sea"],
    "North Atlantic Ocean": ["Baffin Bay", "Davis Strait", "Greenland Sea", "Labrador Sea", "Norwegian Sea"],
}
RELATED_PAIRS = [f"{c}\t{s}" for c, seas in RELATED_SEAS.items() for s in seas]


def related(claim, sea):
    """The checker placed the point in a sea that counts as the claimed one."""
    return pl.concat_str([claim, sea], separator="\t").is_in(RELATED_PAIRS).fill_null(False)


AUTOFILL_MIN = 5
AUTOFILL_MIN_DEG = 0.01   # ~1 km: smaller drags are fake precision, not wrong places
MG = os.environ.get("METALOG_GEOCODING_DIR") or sys.exit(
    "set METALOG_GEOCODING_DIR to a metalog-geocoding checkout (see .env.example)")
NE = f"{MG}/data"

spec = importlib.util.spec_from_file_location(
    "country_aliases", f"{MG}/metalog_geocoding/country_aliases.py")
country_aliases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(country_aliases)
canonical = country_aliases.canonical

MARINE = re.compile(r"marine|sea ?water|seawater|ocean|coral|sponge|seagrass|kelp|"
                    r"sediment metagenome|hydrothermal|estuar|fish|whale|dolphin|seal\b|"
                    r"oyster|mussel|shrimp|salt marsh|mangrove|beach|coastal|plankton", re.I)
# A gut or faeces without a land host is not terrestrial: eels, amphipods and
# whales have guts too ("gut metagenome", "invertebrate gut metagenome").
TERRESTRIAL = re.compile(r"\b(human|mouse|rat|skin|oral|"
                         r"vagina|lung|blood|milk|cheese|food|plant|leaf|root|rhizosphere|"
                         r"phyllosphere|soil|compost|bioreactor|wastewater|sludge|"
                         r"built environment|indoor|air|insect|chicken|pig|cow|bovine|"
                         r"sheep|goat|dog|cat|honey|wine|beer|fermentation)", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s).casefold()
    return re.sub(r"\s+", " ", s).strip()


SUFFIX = re.compile(r"\s+(province|state|prefecture|region|oblast|county|district|"
                    r"department|governorate|territory)$")


def load_geo(path: str) -> pl.DataFrame:
    long = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    wide = (long.unique(["latitude", "longitude", "key"], keep="last")
                .pivot(on="key", index=["latitude", "longitude"], values="value"))
    keep = ["latitude", "longitude", "type", "continent", "country", "country_code",
            "name_1", "iso_3166_2", "ocean", "sea", "longhurst_province",
            "coast_distance_km", "coast_flag", "coast_landmass_name",
            "coast_landmass_area_km2", "coast_country", "coast_country_code",
            "coast_admin_1", "coast_sea", "is_fallback", "fallback_distance_m", "error"]
    for c in keep:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, pl.Utf8).alias(c))
    return (wide.select(keep)
                .rename({"name_1": "admin_1"})
                .with_columns(pl.col("coast_distance_km").cast(pl.Float64),
                              lat_f=pl.col("latitude").cast(pl.Float64),
                              lon_f=pl.col("longitude").cast(pl.Float64)))


def country_codes(admin0: gpd.GeoDataFrame) -> dict[str, set[str]]:
    """canonical country name -> Natural Earth ADM0_A3 codes."""
    out: dict[str, set[str]] = {}
    for _, r in admin0.iterrows():
        for col in ("NAME", "NAME_LONG", "ADMIN", "GEOUNIT", "BRK_NAME",
                    "FORMAL_EN", "NAME_SORT", "NAME_EN", "NAME_ALT"):
            v = r.get(col)
            if isinstance(v, str) and v:
                out.setdefault(canonical(v), set()).add(r["ADM0_A3"])
    return out


def admin1_aliases(admin1: gpd.GeoDataFrame) -> dict[str, dict[str, set[int]]]:
    """ADM0_A3 -> {normalised region name or code: admin1 row indices}.

    One name can cover several rows: Natural Earth splits Italy into provinces,
    and "Tuscany" (name_alt, region) belongs to all ten Tuscan ones."""
    out: dict[str, dict[str, set[int]]] = {}
    for idx, r in admin1.iterrows():
        d = out.setdefault(r["adm0_a3"], {})
        names = [r.get(c) for c in ("name", "name_en", "gn_name", "woe_name", "region")]
        alt = r.get("name_alt")
        names += alt.split("|") if isinstance(alt, str) else []
        for v in names:
            if isinstance(v, str) and v.strip():
                d.setdefault(norm(v), set()).add(idx)
                d.setdefault(SUFFIX.sub("", norm(v)), set()).add(idx)
        iso = r.get("iso_3166_2")
        for code in (r.get("postal"), iso.split("-")[-1] if isinstance(iso, str) else None):
            if isinstance(code, str) and 2 <= len(code) <= 3:
                d.setdefault("CODE:" + code.upper(), set()).add(idx)
    return out


def region_tokens(name_value: str) -> list[tuple[str, bool]]:
    """Candidate region strings after the country: (text, looks_like_code)."""
    if ":" not in name_value:
        return []
    rest = name_value.split(":", 1)[1]
    parts = [p.strip() for p in re.split(r"[:,;/]", rest) if p.strip()]
    toks = []
    for p in parts:
        if re.fullmatch(r"[A-Z]{2,3}", p):
            toks.append(("CODE:" + p, True))
        else:
            n = norm(p)
            toks += [(n, False), (SUFFIX.sub("", n), False)]
    return toks


def unplaceable(code, reason):
    """The checker could not resolve the claimed name at all (unknown_place;
    older service versions only said so in the reason text)."""
    return ((pl.col(code) == "unknown_place")
            | pl.col(reason).str.contains("has no country called")).fill_null(False)


def place_evidence(s: pl.DataFrame, path: str) -> pl.DataFrame:
    """Compare coordinates with the places the study description names as sampled.

    `path` comes from place_distances.py: distances from each flagged
    coordinate, its swap / sign-flip variants and its autofill base to every
    sampled place of the study. A variant "fits" a place within PLACE_TOL_KM
    or precision_km. Adds place_match (a qualifying place the original
    coordinate fits), fix_place_support, fix_place and fix_place_km (for rows
    with a fix: the place that backs it, else the closest sampled place),
    and clears country/sea/region mismatches whose coordinate fits a sampled
    place (see the module docstring)."""
    new = dict(suggest_basis=pl.Utf8, place_match=pl.Utf8, fix_place_support=pl.Utf8, fix_place=pl.Utf8, fix_place_km=pl.Float64)
    try:
        p = pl.read_csv(path, separator="\t", infer_schema_length=None)
    except FileNotFoundError:
        print(f"no {path}: place evidence skipped", file=sys.stderr)
        return s.with_columns([pl.lit(None, t).alias(c) for c, t in new.items()])
    # per coordinate: were any places named (study or sample), did any resolve
    named = (p.group_by("study", lat_f="latitude", lon_f="longitude")
              .agg(places_named=pl.lit(True), places_resolved=(pl.col("status") == "resolved").any()))
    p = (p.filter(pl.col("distance_km").is_not_null())
          .with_columns(origin=pl.col("origin") if "origin" in p.columns else pl.lit("study"))
          .select("study", lat_f="latitude", lon_f="longitude", variant="variant", place="place", origin="origin",
                  label="label", source="source", kind="kind", geometry="geometry",
                  distance_km="distance_km", area_km2=pl.col("area_km2").cast(pl.Float64)))
    # nested places tie (Valencia, Spain and Spain both at 0 km): name the most specific
    closest = pl.col("label").sort_by("distance_km", "area_km2", "label", nulls_last=True).first()
    rows = (s.filter(pl.col("bogus") & ~pl.col("f_out_of_range") & ~pl.col("f_null_island"))
             .select("Sample", "study", "lat_f", "lon_f", "claim", "precision_km", "f_region_mismatch", "sample_type",
                     variant=pl.when(pl.col("f_autofill")).then(pl.lit("autofill_base"))
                               .otherwise(pl.col("diagnosis").str.replace(r"_to_land$", "")),
                     has_fix=pl.col("fix_lat").is_not_null()))
    # A marine sample is near a named country or state within the offshore
    # allowance of country claims (an Antarctic cruise names "Antarctica").
    tol = pl.max_horizontal(pl.lit(PLACE_TOL_KM), pl.col("precision_km").cast(pl.Float64),
                            pl.when((pl.col("sample_type") == "marine")
                                    & pl.col("source").is_in(["natural_earth_admin0", "natural_earth_admin1"]))
                              .then(pl.lit(OFFSHORE_KM)).otherwise(pl.lit(0.0)),
                            pl.when((pl.col("geometry") == "point") & (pl.col("origin") == "study"))
                              .then(pl.lit(PLACE_POINT_KM)).otherwise(pl.lit(0.0)))
    joined = rows.join(p, on=["study", "lat_f", "lon_f"])
    fit = (joined.filter(pl.col("distance_km") <= tol)
                 .filter((pl.col("variant_right") == "original") | (pl.col("variant_right") == pl.col("variant"))))
    # closest sampled place to the fix, whether or not it counts as backing it
    near = (joined.filter(pl.col("has_fix") & (pl.col("variant_right") == pl.col("variant")))
                  .group_by("Sample").agg(fix_place=closest,
                                          fix_place_km=pl.col("distance_km").min().round(1)))
    # A continent, or the claimed country itself (within tolerance of a border),
    # tells nothing beyond the claim; nor does any country for a region claim.
    canon = {v: canonical(v) for v in pl.concat([fit["claim"], fit["label"]]).drop_nulls().unique()}
    trivial = ((pl.col("kind") == "continent")
               | ((pl.col("source") == "natural_earth_admin0")
                  & (pl.col("f_region_mismatch")
                     | (pl.col("label").replace_strict(canon, default=None)
                        == pl.col("claim").replace_strict(canon, default=None)).fill_null(False))))
    # The sample's own location text backs the coordinate only if the point
    # fits every name in it that resolved to an outline ("Scotland:Glasgow"
    # needs Glasgow). A name known only as a label point may support but not
    # veto: generic site names land anywhere ("Western Shoals, Guam" is also
    # a shoal off the Auckland Islands).
    own_res = (joined.filter((pl.col("origin") == "sample") & (pl.col("variant_right") == "original")
                             & ((pl.col("geometry") != "point") | (pl.col("distance_km") <= tol)))
                     .group_by("Sample").agg(own_n=pl.col("place").n_unique()))
    own_fit = (fit.filter((pl.col("origin") == "sample") & (pl.col("variant_right") == "original"))
                  .group_by("Sample").agg(own_fit=pl.col("place").n_unique()))
    orig = (fit.filter((pl.col("variant_right") == "original") & ~trivial)
               .group_by("Sample").agg(place_match=closest,
                                       own_any=(pl.col("origin") == "sample").any(),
                                       study_any=(pl.col("origin") == "study").any())
               .join(own_res, on="Sample", how="left").join(own_fit, on="Sample", how="left")
               .with_columns(own_match=pl.col("own_any") & (pl.col("own_fit") == pl.col("own_n")))
               .drop("own_any", "own_n", "own_fit"))
    allorig = fit.filter(pl.col("variant_right") == "original").select("Sample", "place").unique()
    fixed = (fit.filter(pl.col("has_fix") & (pl.col("variant_right") == pl.col("variant")))
                .join(allorig, on=["Sample", "place"], how="anti")
                .group_by("Sample").agg(fix_fit=closest,
                                        fix_fit_km=pl.col("distance_km").min().round(1)))
    s = (s.join(orig, on="Sample", how="left").join(named, on=["study", "lat_f", "lon_f"], how="left").join(fixed, on="Sample", how="left")
          .join(near, on="Sample", how="left"))
    # the place that backs the fix, else the closest one
    s = s.with_columns(fix_place=pl.coalesce("fix_fit", "fix_place"),
                       fix_place_km=pl.when(pl.col("fix_fit").is_not_null()).then(pl.col("fix_fit_km"))
                                      .otherwise(pl.col("fix_place_km")))
    s = s.with_columns(fix_place_support=pl.when(pl.col("fix_lat").is_null()).then(None)
                       .when(~pl.col("places_named").fill_null(False)).then(pl.lit("no_places"))
                       .when(~pl.col("places_resolved").fill_null(False)).then(pl.lit("unresolved"))
                       .when(pl.col("fix_fit").is_not_null()).then(pl.lit("supported"))
                       .when(pl.col("place_match").is_not_null()).then(pl.lit("original_fits"))
                       .otherwise(pl.lit("no_match")))
    # Rescue: the coordinate lies in a named place and no fix lands in another.
    # Drags, impossible coordinates and terrestrial samples at sea stay
    # flagged. The sample's own location text can overrule its country or
    # region claim anywhere (Antarctica: ..., Kerguelen Islands). A place from
    # the study description only can at sea (a cruise labelled with a nearby
    # or home country) or for a sea claim: on land, a multi-site study with
    # every coordinate at one of its sites is a copy-paste, not a confirmation.
    rescue = (pl.col("place_match").is_not_null() & pl.col("fix_fit").is_null()
              & ~pl.col("f_autofill") & ~pl.col("f_terrestrial_at_sea")
              & (pl.col("own_match").fill_null(False)
                 | (pl.col("study_any").fill_null(False)
                    & ((pl.col("type") == "sea") | ~(pl.col("f_country_mismatch") | pl.col("f_region_mismatch"))))))
    checks = ["country_mismatch", "sea_mismatch", "region_mismatch"]
    s = s.with_columns(
        lenient=pl.concat_str([pl.col("lenient")] + [pl.when(rescue & pl.col(f"f_{k}")).then(pl.lit(k))
                                                     for k in checks], separator=",", ignore_nulls=True)
                  .str.strip_chars(",").replace("", None),
        **{f"f_{k}": pl.col(f"f_{k}") & ~rescue.fill_null(False) for k in checks})
    fcols = [c for c in s.columns if c.startswith("f_")]
    s = s.with_columns(
        flags=pl.concat_str([pl.when(pl.col(c)).then(pl.lit(c[2:])) for c in fcols],
                            separator=",", ignore_nulls=True).replace("", None),
        bogus=pl.any_horizontal(fcols))
    # A fix is only suggested when text backs it: the sample's own location
    # claim (a swap / sign fix lands in the claimed country or sea by
    # construction; an autofill base must carry the same claim and pass), or a
    # place named in the study description. "<transform>_to_land" only knows
    # the sample is land-based, so it needs the description. If the study
    # names a place the recorded point is in and the fix lands in none, that
    # argues against moving it.
    base_claim = (s.filter(~pl.col("bogus") & pl.col("claim_known"))
                   .select("study", autofill_base_lat="lat_f", autofill_base_lon="lon_f", claim="claim",
                           base_claim_ok=pl.lit(True)).unique())
    s = s.join(base_claim, on=["study", "autofill_base_lat", "autofill_base_lon", "claim"], how="left")
    by_claim = (pl.col("fix_lat").is_not_null() & pl.col("claim_known")
                & pl.when(pl.col("f_autofill")).then(pl.col("base_claim_ok").fill_null(False))
                    .otherwise(~pl.col("diagnosis").str.ends_with("_to_land").fill_null(False)))
    by_place = pl.col("fix_place_support") == "supported"
    s = s.with_columns(suggest_basis=pl.when(pl.col("fix_place_support") == "original_fits").then(None)
                       .when(by_claim & by_place).then(pl.lit("claim+place"))
                       .when(by_claim).then(pl.lit("claim"))
                       .when(by_place).then(pl.lit("place")))
    s = s.with_columns([pl.when(pl.col("suggest_basis").is_null()).then(None).otherwise(pl.col(c)).alias(c)
                        for c in ["suggest_lat", "suggest_lon", "suggest_source", "suggest_uncertainty_km"]])
    unflag = ~pl.col("bogus")
    s = s.drop("fix_fit", "fix_fit_km", "base_claim_ok", "own_match", "study_any", "places_named", "places_resolved")
    return s.with_columns(
        [pl.when(unflag).then(None).otherwise(pl.col(c)).alias(c)
         for c in ["diagnosis", "fix_lat", "fix_lon", "suggest_lat", "suggest_lon", "suggest_source",
                   "suggest_uncertainty_km", "suggest_basis", "fix_place_support", "fix_place", "fix_place_km"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--geo", required=True)
    ap.add_argument("--verdicts", nargs="+", required=True)
    ap.add_argument("--places")
    args = ap.parse_args()
    args.out = f"{args.run}/geo"
    args.places = args.places or f"{args.run}/place_distances.tsv"

    geo = load_geo(args.geo)
    print(f"coordinates geocoded: {geo.height}", file=sys.stderr)

    claims = pl.read_csv(f"{args.run}/sample_claims.tsv", separator="\t", infer_schema_length=0,
                         quote_char=None).rename({"id": "pair_id"})
    ctx = (pl.read_csv(f"{args.run}/samples.tsv", separator="\t", infer_schema_length=0, quote_char=None)
             .select(Sample="sample", scientific_name="scientific_name", env_biome="env_biome",
                     env_feature="env_feature", env_material="env_material",
                     isolation_source="isolation_source", study="study"))
    # Joined by content, not id: ids change whenever the input is rebuilt.
    verd = pl.concat([pl.read_csv(v, separator="\t", infer_schema_length=0, quote_char=None)
                      .select("latitude", "longitude", "geographic_location", "grade", "code", "reason")
                      for v in args.verdicts])
    verd = verd.unique(["latitude", "longitude", "geographic_location"], keep="last")
    vcols = ["grade", "code", "reason"]

    # Prefer the verdict on the claim as written (Svalbard is stricter than
    # Norway); fall back to the claim_map.tsv name only where the checker could
    # not resolve the written name at all (unknown_place).
    s = (claims.join(verd.rename({"geographic_location": "claim"}),
                     on=["latitude", "longitude", "claim"], how="left")
               .join(verd.rename({"geographic_location": "claim_raw"} | {c: f"{c}_raw" for c in vcols}),
                     on=["latitude", "longitude", "claim_raw"], how="left"))
    use_raw = (pl.col("grade_raw").is_not_null()
               & ~unplaceable("code_raw", "reason_raw"))
    s = (s.with_columns([pl.when(use_raw).then(pl.col(f"{c}_raw")).otherwise(pl.col(c)).alias(c)
                         for c in vcols])
          .drop([f"{c}_raw" for c in vcols])
          .join(ctx, on="Sample", how="left")
          .join(geo, on=["latitude", "longitude"], how="left"))
    s = s.with_columns(
        claim_known=pl.col("claim").is_not_null()
                    & ~unplaceable("code", "reason"),
        lat_f=pl.col("latitude").cast(pl.Float64, strict=False),
        lon_f=pl.col("longitude").cast(pl.Float64, strict=False))

    # Sample type from scientific name, env fields and isolation source.
    text = pl.concat_str(["scientific_name", "env_biome", "env_feature", "env_material",
                          "isolation_source"], separator=" | ", ignore_nulls=True)
    s = s.with_columns(sample_type=pl.when(text.str.contains("(?i)" + MARINE.pattern))
                       .then(pl.lit("marine"))
                       .when(text.str.contains("(?i)" + TERRESTRIAL.pattern))
                       .then(pl.lit("terrestrial"))
                       .otherwise(pl.lit("unknown")))

    # --- region check (right country, wrong admin-1?) -----------------------
    admin0 = gpd.read_parquet(f"{NE}/ne_10m_admin_0_countries.parquet")
    admin1 = gpd.read_parquet(f"{NE}/ne_10m_admin_1_states_provinces.parquet")
    codes = country_codes(admin0)
    a1 = admin1_aliases(admin1)
    geod = Geod(ellps="WGS84")

    cand = (s.filter((pl.col("grade") == "ok") & (pl.col("type") == "land")
                     & pl.col("country_code").is_not_null() & pl.col("admin_1").is_not_null()
                     & pl.col("name_value").str.contains(":"))
             .select("name_value", "country_code", "admin_1", "iso_3166_2", "lat_f", "lon_f")
             .unique())
    region_rows = []
    for nv, cc, adm, iso, lat, lon in cand.iter_rows():
        table = a1.get(cc, {})
        hits = sorted(set().union(*(table[t] for t, _ in region_tokens(nv) if t in table)))
        if not hits:
            region_rows.append((nv, lat, lon, None, "uncheckable", None))
            continue
        names = {admin1.at[h, "name"] for h in hits}
        if adm in names or any(admin1.at[h, "iso_3166_2"] == iso for h in hits):
            region_rows.append((nv, lat, lon, adm, "ok", 0.0))
            continue
        p = shapely.Point(lon, lat)
        d, h = min((geod.inv(lon, lat, *shapely.ops.nearest_points(admin1.at[h, "geometry"], p)[0].coords[0])[2] / 1000, h)
                   for h in hits)
        region_rows.append((nv, lat, lon, admin1.at[h, "region"] or admin1.at[h, "name"]
                            if len(hits) > 1 else admin1.at[h, "name"],
                            "mismatch" if d > REGION_TOL_KM else "near_border", round(d, 1)))
    reg = pl.DataFrame(region_rows, orient="row", schema={
        "name_value": pl.Utf8, "lat_f": pl.Float64, "lon_f": pl.Float64,
        "region_claimed": pl.Utf8, "region_check": pl.Utf8, "region_distance_km": pl.Float64})
    s = s.join(reg, on=["name_value", "lat_f", "lon_f"], how="left")

    # --- autofill: constant steps between a study's coordinates -------------
    # Sorted by (lat, lon), a drag down a column is a run of equal steps:
    # diagonal (|dlat| == |dlon|), or along one axis (dlat == 0 or dlon == 0).
    # The coordinate before the run is its base, the one the drag started from.
    pts = (s.filter(pl.col("study").is_not_null() & pl.col("lat_f").abs().le(90) & pl.col("lon_f").abs().le(180))
             .select("study", "lat_f", "lon_f").unique()
             .sort("study", "lat_f", "lon_f")
             .with_columns(dlat=(pl.col("lat_f") - pl.col("lat_f").shift(1).over("study")).round(6),
                           dlon=(pl.col("lon_f") - pl.col("lon_f").shift(1).over("study")).round(6)))
    step_changed = ((pl.col("dlat") != pl.col("dlat").shift(1).over("study"))
                    | (pl.col("dlon") != pl.col("dlon").shift(1).over("study"))).fill_null(True)
    k = pl.int_range(1, pl.len() + 1).over("study", "run")
    pts = (pts.with_columns(run=step_changed.cum_sum().over("study"),
                            prev_lat=pl.col("lat_f").shift(1).over("study"),
                            prev_lon=pl.col("lon_f").shift(1).over("study"))
              .with_columns(run_len=pl.len().over("study", "run"),
                            autofill_base_lat=pl.col("prev_lat").first().over("study", "run"),
                            autofill_base_lon=pl.col("prev_lon").first().over("study", "run"))
              .filter(pl.col("dlat").is_not_null() & ((pl.col("dlat") != 0) | (pl.col("dlon") != 0))
                      & ((pl.col("dlat").abs() == pl.col("dlon").abs())
                         | (pl.col("dlat") == 0) | (pl.col("dlon") == 0))
                      & (pl.col("run_len") >= AUTOFILL_MIN - 1))
              .select("study", "lat_f", "lon_f", "autofill_base_lat", "autofill_base_lon",
                      autofill_step=pl.format("({},{})", pl.col("dlat"), pl.col("dlon")),
                      autofill_run=pl.col("run_len") + 1,
                      autofill_id=pl.col("run"),
                      autofill_diagonal=pl.col("dlat").abs() == pl.col("dlon").abs(),
                      # k-th step from the run's base coordinate, in degrees
                      autofill_offset_deg=(pl.max_horizontal(pl.col("dlat").abs(), pl.col("dlon").abs()) * k).round(6)))
    s = s.join(pts, on=["study", "lat_f", "lon_f"], how="left")

    # --- precision: how far the true point can be from a rounded coordinate --
    # d decimals -> the point is anywhere in a cell of 10^-d degrees; the
    # half-diagonal at the equator is 0.5 * sqrt(2) * 111.32 km * 10^-d.
    def decimals(col):
        frac = pl.col(col).str.extract(r"\.(\d*?)0*$", 1).str.len_chars()
        return pl.when(pl.col(col).str.contains(r"\.")).then(frac).otherwise(0)
    s = s.with_columns(coord_decimals=pl.min_horizontal(decimals("latitude"), decimals("longitude")))
    s = s.with_columns(precision_km=(78.7 * pl.lit(10.0).pow(-pl.col("coord_decimals").cast(pl.Float64))).round(2))

    # --- distance from the coordinate to the claimed country or sea ----------
    a0 = admin0.dissolve("ADM0_A3")["geometry"]
    iho = gpd.read_parquet(f"{NE}/World_Seas_IHO_v3.parquet")[["NAME", "geometry"]]
    geoms: dict[tuple[str, str], object] = {}

    def claim_geom(claim):
        """(kind, geometry) of a claim: a Natural Earth country, or IHO seas
        whose name contains it (Atlantic Ocean -> North/South Atlantic Ocean).
        RELATED_SEAS are matched by the sea name instead (see related())."""
        c = canonical(claim)
        ids = [a for a in codes.get(c, ()) if a in a0.index]
        if ids:
            key = ("country", c)
            if key not in geoms:
                geoms[key] = shapely.union_all(a0.loc[ids].values)
            return key[0], geoms[key]
        # a claim that is no IHO sea itself (Sargasso Sea) takes its related seas
        names = [n for n in iho["NAME"] if claim.casefold() in n.casefold()] or RELATED_SEAS.get(claim, [])
        if not names:
            return None, None
        key = ("sea", claim)
        if key not in geoms:
            geoms[key] = shapely.union_all(iho[iho["NAME"].isin(names)].geometry.values)
        return key[0], geoms[key]

    def dist_km(geom, lats, lons):
        """Geodesic km from each point to the planar-nearest point of geom (0 inside)."""
        line = shapely.shortest_line(shapely.points(lons, lats), geom)
        end = shapely.get_coordinates(line)[1::2]
        return [d / 1000 for d in geod.inv(lons, lats, end[:, 0], end[:, 1])[2]]

    need = (s.filter(pl.col("claim_known") & pl.col("code").is_in(
                ["different_location", "claim_too_far", "not_on_land", "not_on_sea", "unrelated_sea"]))
             .select("claim", "lat_f", "lon_f").unique())
    drows = []
    for (claim,), grp in need.group_by("claim"):
        kind, g = claim_geom(claim)
        if g is None:
            continue
        lats, lons = grp["lat_f"].to_list(), grp["lon_f"].to_list()
        drows += [(claim, la, lo, kind, round(d, 1)) for la, lo, d in zip(lats, lons, dist_km(g, lats, lons))]
    cdist = pl.DataFrame(drows, orient="row", schema={"claim": pl.Utf8, "lat_f": pl.Float64, "lon_f": pl.Float64,
                                                      "claim_kind": pl.Utf8, "claim_distance_km": pl.Float64})
    s = s.join(cdist, on=["claim", "lat_f", "lon_f"], how="left")
    # The checker names the distance itself (0.2.0+); prefer it, since its
    # territory polygons can differ from the Natural Earth union used above.
    said = pl.coalesce(
        pl.col("reason").str.extract(r"nearest land called '.*' is ([\d,.]+) km away", 1),
        pl.col("reason").str.extract(r"is ([\d,.]+) km from the nearest land belonging to", 1),
    ).str.replace_all(",", "").cast(pl.Float64, strict=False)
    s = s.with_columns(claim_distance_km=pl.coalesce(said, pl.col("claim_distance_km")))

    # A sample whose metadata names a sea was taken at sea, whatever its
    # scientific name suggests ("invertebrate gut metagenome", Mariana Trench).
    iho_names = [n.casefold() for n in iho["NAME"]]
    sea_claims = [c for c in s["claim"].drop_nulls().unique().to_list()
                  if not codes.get(canonical(c)) and (c in RELATED_SEAS or any(c.casefold() in n for n in iho_names))]
    s = s.with_columns(sea_claim=pl.col("claim_known") & pl.col("claim").is_in(sea_claims))

    # --- flags -------------------------------------------------------------
    # Each check has a base tolerance; a coarsely rounded coordinate widens it
    # to its precision_km. "lenient" lists the checks that only pass because
    # of that widening or of RELATED_SEAS / the offshore allowance.
    code, prec, dist = pl.col("code"), pl.col("precision_km"), pl.col("claim_distance_km")
    strict = dict(
        country_mismatch=pl.col("claim_known") & (
            (code.is_in(["different_location", "claim_too_far"]) & ~unplaceable("code", "reason"))
            | ((code == "not_on_land") & (dist > OFFSHORE_KM))),
        sea_mismatch=pl.col("claim_known") & code.is_in(["not_on_sea", "unrelated_sea"]),
        region_mismatch=pl.col("region_check") == "mismatch",
        terrestrial_at_sea=(pl.col("sample_type") == "terrestrial") & (pl.col("type") == "sea") & ~pl.col("sea_claim")
                           & (pl.col("coast_distance_km") > SEA_TOL_KM),
    )
    lenient = dict(
        country_mismatch=pl.col("claim_known") & (
            (code.is_in(["different_location", "claim_too_far"]) & ~unplaceable("code", "reason")
             & ~(dist <= prec).fill_null(False))
            | ((code == "not_on_land") & (dist > pl.max_horizontal(pl.lit(OFFSHORE_KM), prec)))),
        sea_mismatch=pl.col("claim_known") & (
            ((code == "unrelated_sea") & ~(dist <= pl.max_horizontal(pl.lit(SEA_NEAR_KM), prec)).fill_null(False)
             & ~((pl.col("claim") == "Southern Ocean") & (pl.col("lat_f") <= SOUTHERN_OCEAN_LAT))
             & ~((pl.col("sea") == "Southern Ocean") & pl.col("claim").is_in(SOUTHERN_NEIGHBOURS))
             & ~related(pl.col("claim"), pl.col("sea")))
            | ((code == "not_on_sea") & ~(dist <= prec).fill_null(False))),
        region_mismatch=(pl.col("region_check") == "mismatch") & (pl.col("region_distance_km") > prec),
        terrestrial_at_sea=(pl.col("sample_type") == "terrestrial") & (pl.col("type") == "sea") & ~pl.col("sea_claim")
                           & (pl.col("coast_distance_km") > pl.max_horizontal(pl.lit(SEA_TOL_KM), prec)),
    )
    s = s.with_columns(
        f_out_of_range=code == "coordinate_out_of_range",
        f_null_island=(pl.col("lat_f") == 0) & (pl.col("lon_f") == 0),
        **{f"f_{k}": v for k, v in lenient.items()},
    )
    # A one-axis run can be a real transect (a cruise along a parallel), so it
    # only counts as a drag when some of its points contradict their metadata.
    contra = (pl.col("f_country_mismatch") | pl.col("f_sea_mismatch") | pl.col("f_region_mismatch")
              | pl.col("f_terrestrial_at_sea")).fill_null(False)
    s = s.with_columns(f_autofill=(pl.col("autofill_offset_deg") >= AUTOFILL_MIN_DEG)
                       & (pl.col("autofill_diagonal") | contra.any().over("study", "autofill_id")))
    fcols = [c for c in s.columns if c.startswith("f_")]
    s = s.with_columns([pl.col(c).fill_null(False) for c in fcols])
    s = s.with_columns(
        flags=pl.concat_str([pl.when(pl.col(c)).then(pl.lit(c[2:])) for c in fcols],
                            separator=",", ignore_nulls=True),
        lenient=pl.concat_str([pl.when(v.fill_null(False) & ~pl.col(f"f_{k}")).then(pl.lit(k))
                               for k, v in strict.items()], separator=",", ignore_nulls=True),
        bogus=pl.any_horizontal(fcols))

    # --- diagnosis: does a swapped / sign-flipped coordinate fit? -------------
    # Target: the claimed country or sea (within DIAG_TOL_KM or precision_km;
    # and the claimed admin-1 when there is one); without a usable claim, a
    # terrestrial sample at sea is diagnosed when exactly one transform puts
    # it on land ("<transform>_to_land").
    bad = (s.filter(pl.col("bogus") & ~pl.col("f_out_of_range") & ~pl.col("f_null_island"))
             .with_columns(claim_t=pl.when(pl.col("claim_known")).then(pl.col("claim")))
             .with_columns(mm=pl.col("f_country_mismatch") | pl.col("f_sea_mismatch") | pl.col("f_region_mismatch"))
             .select("claim_t", "region_claimed", "lat_f", "lon_f", "precision_km", "f_terrestrial_at_sea", "mm")
             .unique())
    transforms = {"swapped": lambda la, lo: (lo, la), "lat_sign": lambda la, lo: (-la, lo),
                  "lon_sign": lambda la, lo: (la, -lo), "both_signs": lambda la, lo: (-la, -lo),
                  "swapped_lat_sign": lambda la, lo: (-lo, la),
                  "swapped_lon_sign": lambda la, lo: (lo, -la)}
    trows = []
    for claim, rclaim, la, lo, pk, tas, mm in bad.iter_rows():
        # a transform must explain the flag: the point contradicts the claim,
        # or a land-based sample sits at sea (then the result must be on land)
        if not (mm or tas):
            continue
        for k, (name, f) in enumerate(transforms.items()):
            tla, tlo = f(la, lo)
            if -90 <= tla <= 90 and -180 <= tlo <= 180 and (tla, tlo) != (la, lo):
                trows.append((claim, rclaim, la, lo, name, k, tla, tlo, max(DIAG_TOL_KM, pk or 0), bool(tas)))
    tcols = ["claim_t", "region_claimed", "lat_f", "lon_f", "diagnosis", "order", "tlat", "tlon", "tol", "tas"]
    t = pl.DataFrame(trows, orient="row", schema=dict(zip(tcols, [pl.Utf8, pl.Utf8, pl.Float64, pl.Float64,
                                                                pl.Utf8, pl.Int64, pl.Float64, pl.Float64, pl.Float64,
                                                                pl.Boolean])))
    land = shapely.union_all(a0.values)
    shapely.prepare(land)
    t = t.with_columns(on_land=pl.Series(shapely.contains_xy(land, t["tlon"].to_numpy(), t["tlat"].to_numpy()),
                                         dtype=pl.Boolean))
    fits = []
    # claimed country or sea
    for (claim,), grp in t.filter(pl.col("claim_t").is_not_null()).group_by("claim_t"):
        kind, g = claim_geom(claim)
        if g is None and claim not in RELATED_SEAS:
            continue
        near = (pl.Series(dist_km(g, grp["tlat"].to_list(), grp["tlon"].to_list())) <= grp["tol"]
                if g is not None else pl.Series([False] * grp.height))
        if claim in RELATED_SEAS:
            # landing inside a related sea counts like landing in the claim
            pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grp["tlon"], grp["tlat"]), crs=iho.crs)
            hit = set(gpd.sjoin(pts, iho[iho["NAME"].isin(RELATED_SEAS[claim])], predicate="within").index)
            near = near | pl.Series([i in hit for i in range(grp.height)])
        fits.append(grp.with_columns(fit=near & (~pl.col("tas") | pl.col("on_land"))))
    # no usable claim: on land at all
    nc = t.filter(pl.col("claim_t").is_null())
    if nc.height:
        nc = nc.with_columns(fit=pl.col("on_land"))
        one = nc.group_by("lat_f", "lon_f").agg(nfit=pl.col("fit").sum())
        nc = (nc.join(one, on=["lat_f", "lon_f"]).with_columns(fit=pl.col("fit") & (pl.col("nfit") == 1),
                                                              diagnosis=pl.col("diagnosis") + "_to_land")
                .drop("nfit"))
        fits.append(nc)
    diag = pl.DataFrame(schema={"claim_t": pl.Utf8, "region_claimed": pl.Utf8, "lat_f": pl.Float64,
                                "lon_f": pl.Float64, "diagnosis": pl.Utf8, "fix_lat": pl.Float64,
                                "fix_lon": pl.Float64})
    if fits:
        cand = pl.concat(fits).filter("fit").drop("tas", "on_land")
        # the claimed admin-1, when there is one, must hold the transformed point
        rc = cand.filter(pl.col("region_claimed").is_not_null())
        if rc.height:
            pg = gpd.GeoDataFrame(rc.to_pandas(), geometry=gpd.points_from_xy(rc["tlon"], rc["tlat"]), crs=admin1.crs)
            j1 = gpd.sjoin(pg, admin1[["name", "region", "geometry"]], how="left", predicate="within")
            okr = {(r.claim_t, r.region_claimed, r.lat_f, r.lon_f, r.diagnosis)
                   for r in j1.itertuples() if r.region_claimed in (r.name, r.region)}
            keep = [(a, b, c, d, e) in okr for a, b, c, d, e in
                    rc.select("claim_t", "region_claimed", "lat_f", "lon_f", "diagnosis").iter_rows()]
            cand = pl.concat([cand.filter(pl.col("region_claimed").is_null()), rc.filter(pl.Series(keep))])
        diag = (cand.sort("order").unique(["claim_t", "region_claimed", "lat_f", "lon_f"], keep="first",
                                           maintain_order=True)
                    .select("claim_t", "region_claimed", "lat_f", "lon_f", "diagnosis",
                            fix_lat=pl.col("tlat"), fix_lon=pl.col("tlon")))
    s = (s.with_columns(claim_t=pl.when(pl.col("bogus") & pl.col("claim_known")).then(pl.col("claim")))
          .join(diag, on=["claim_t", "region_claimed", "lat_f", "lon_f"], how="left", nulls_equal=True)
          .drop("claim_t"))
    s = s.with_columns([pl.when(pl.col("bogus")).then(pl.col(c)).alias(c) for c in ("diagnosis", "fix_lat", "fix_lon")])
    # a drag explains the whole run; its fix is the base coordinate, provided
    # the base itself passes every check (a drag from 1,1 fixes nothing)
    base_ok = (s.group_by("study", "lat_f", "lon_f").agg(base_ok=~pl.col("bogus").any())
                .rename({"lat_f": "autofill_base_lat", "lon_f": "autofill_base_lon"}))
    s = s.join(base_ok, on=["study", "autofill_base_lat", "autofill_base_lon"], how="left")
    af = pl.col("f_autofill")
    s = s.with_columns(diagnosis=pl.when(af).then(pl.format("autofill_step{}x{}", pl.col("autofill_step"),
                                                            pl.col("autofill_run")))
                                   .otherwise(pl.col("diagnosis")))
    ok = pl.col("base_ok").fill_null(False)
    s = s.with_columns(
        fix_lat=pl.when(af).then(pl.when(ok).then(pl.col("autofill_base_lat").round(7))).otherwise(pl.col("fix_lat")),
        fix_lon=pl.when(af).then(pl.when(ok).then(pl.col("autofill_base_lon").round(7))).otherwise(pl.col("fix_lon")))

    # Study-level: offset from the study's most common coordinate.
    in_range = pl.col("lat_f").abs().le(90) & pl.col("lon_f").abs().le(180)
    mode = (s.filter(~pl.col("bogus") & pl.col("study").is_not_null() & in_range)
              .group_by("study", "lat_f", "lon_f").len()
              .sort(["len", "lat_f", "lon_f"], descending=[True, False, False])
              .unique("study", keep="first", maintain_order=True)
              .filter(pl.col("len") >= 3)
              .select("study", study_lat="lat_f", study_lon="lon_f", study_n="len"))
    s = s.join(mode, on="study", how="left").with_columns(
        dlat=(pl.col("lat_f") - pl.col("study_lat")).round(6),
        dlon=(pl.col("lon_f") - pl.col("study_lon")).round(6))
    whole = ((pl.col("dlat") == pl.col("dlat").round(0)) & (pl.col("dlon") == pl.col("dlon").round(0))
             & ((pl.col("dlat") != 0) | (pl.col("dlon") != 0)))
    s = s.with_columns(diagnosis=pl.when(pl.col("diagnosis").is_not_null()).then(pl.col("diagnosis"))
                       .when(pl.col("bogus") & ~pl.col("f_out_of_range") & whole)
                       .then(pl.format("study_offset({},{})", pl.col("dlat"), pl.col("dlon")))
                       .otherwise(None))

    # --- suggested position for flagged samples -------------------------------
    # Only mechanical fixes (swap / sign flip, autofill base); no centroids.
    s = s.with_columns(
        suggest_source=pl.when(pl.col("fix_lat").is_null()).then(None)
                         .when(pl.col("f_autofill")).then(pl.lit("autofill"))
                         .otherwise(pl.lit("transform")),
        suggest_lat=pl.col("fix_lat"), suggest_lon=pl.col("fix_lon"),
        suggest_uncertainty_km=pl.when(pl.col("fix_lat").is_not_null()).then(pl.col("precision_km")))

    s = place_evidence(s, args.places)

    # --- outputs ------------------------------------------------------------
    n_samples =claims.group_by("latitude", "longitude").agg(n_samples=pl.len())
    geo.drop("lat_f", "lon_f").join(n_samples, on=["latitude", "longitude"], how="left") \
       .sort(pl.col("latitude").cast(pl.Float64), pl.col("longitude").cast(pl.Float64)) \
       .write_csv(f"{args.out}.coords.tsv", separator="\t")
    out_cols = ["Sample", "latitude", "longitude", "study", "name_field", "name_value", "claim_raw", "claim",
                "claim_known", "grade", "code", "reason", "type", "country", "admin_1", "sea",
                "coast_distance_km", "coast_landmass_name", "region_claimed", "region_check",
                "region_distance_km", "sample_type", "scientific_name", "coord_decimals", "precision_km",
                "claim_kind", "claim_distance_km", "bogus", "flags", "lenient",
                "diagnosis", "fix_lat", "fix_lon", "suggest_lat", "suggest_lon", "suggest_source",
                "suggest_uncertainty_km", "suggest_basis", "fix_place_support", "fix_place", "fix_place_km", "place_match", "autofill_step", "autofill_run", "autofill_offset_deg", "study_lat", "study_lon", "study_n"]
    s.select(out_cols).sort("Sample").write_csv(f"{args.out}.samples.tsv", separator="\t")

    print(f"samples: {s.height}, with geocode: {s['type'].is_not_null().sum()}, "
          f"with verdict: {s['grade'].is_not_null().sum()}", file=sys.stderr)
    print(s.select([pl.col(c).sum().alias(c[2:]) for c in fcols] + [pl.col("bogus").sum()]),
          file=sys.stderr)
    print(s.filter(pl.col("bogus"))["diagnosis"].value_counts(sort=True).head(12), file=sys.stderr)
    print(s["region_check"].value_counts(sort=True), file=sys.stderr)
    print(s["sample_type"].value_counts(sort=True), file=sys.stderr)


if __name__ == "__main__":
    main()
