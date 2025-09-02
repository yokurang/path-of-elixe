# bulk_collect_categories.py
# Collect sample training data across all categories × {corrupted, not}
# - status="any"
# - base_currency="Exalted Orb"
# - league="Standard"
# - realm="poe2" (kept for Trade2 API compatibility)
#
# Persists FX cache locally to avoid requerying every slice and across runs.
# Outputs Parquet files next to this script (or outdir if provided),
# and logs step-by-step progress via logger "poe.bulk".

from __future__ import annotations

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.insert(0, project_root)

import asyncio
import copy
import logging
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import aiohttp

from src.recorder.constants import CATEGORIES

# ---- Import your existing pipeline bits
from take_home_project_poe.misc.poe2_currency_recorder_web_scrape import get_currency_cache, Server
from take_home_project_poe.misc.market_item_listener_bak import ( 
    setup_logging,
    options_from_config,
    headers_from_config,
    payload_from_config,
    search,
    records_to_dataframe,
    PriceConverter,
)

# Dedicated logger for this bulk runner
log_bulk = logging.getLogger("poe.bulk")
log_bulk.addHandler(logging.NullHandler())


# -------------------------
# FX cache persistence utils
# -------------------------

def _resolve_server_enum(name: str) -> Server:
    for s in Server:
        if s.value == name:
            return s
    # Default to Standard if unknown
    return Server.STANDARD

def _fx_cache_path(cache_dir: Path, base_server: str, base_currency: str) -> Path:
    safe_cur = base_currency.replace(" ", "_")
    return cache_dir / f"fx_cache_{base_server}_{safe_cur}.pkl"

def _save_fx_cache_to_disk(cache: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    log_bulk.info("FX cache persisted -> %s (bytes=%d)", path, path.stat().st_size)

def _load_fx_cache_from_disk(path: Path) -> Any:
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return cache

def _file_age(path: Path) -> timedelta:
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime

def build_or_load_fx_cache(
    *,
    base_server: str,
    base_currency: str,
    cache_dir: Path,
    log_level: str = "INFO",
    ttl_hours: int = 12,
    force_refresh: bool = False,
    allow_stale_on_error: bool = True,
) -> Tuple[Optional[Any], str, bool]:
    """
    Returns (fx_cache_or_None, source_label, is_stale).

    Strategy:
      1) If cache file exists and is fresh (<= ttl), load and return ("disk-fresh", False).
      2) Else try live fetch -> save -> return ("live", False).
      3) If live fetch fails and disk exists and allow_stale_on_error, load stale -> ("disk-stale", True).
      4) Else return (None, "none", True) -> caller may run without FX.
    """
    setup_logging(log_level)
    path = _fx_cache_path(cache_dir, base_server, base_currency)
    ttl = timedelta(hours=ttl_hours)

    # 1) Fresh disk cache
    if path.exists() and not force_refresh:
        age = _file_age(path)
        if age <= ttl:
            try:
                cache = _load_fx_cache_from_disk(path)
                log_bulk.info("FX cache loaded from disk (fresh) <- %s | age=%s", path, age)
                return cache, "disk-fresh", False
            except Exception as e:
                log_bulk.warning("Failed to load fresh FX cache from disk (%s): %s; will try live fetch.", path, e)

    # 2) Live fetch
    try:
        server_enum = _resolve_server_enum(base_server)
        log_bulk.info("Fetching FX cache live for server=%s (will persist to %s)", server_enum.value, path)
        cache = get_currency_cache([server_enum], log_level=log_level)
        _save_fx_cache_to_disk(cache, path)
        return cache, "live", False
    except Exception as e:
        log_bulk.error("Live FX fetch failed: %s", e)

    # 3) Stale disk cache fallback
    if allow_stale_on_error and path.exists():
        try:
            cache = _load_fx_cache_from_disk(path)
            age = _file_age(path)
            log_bulk.warning("Using STALE FX cache from disk <- %s | age=%s", path, age)
            return cache, "disk-stale", True
        except Exception as e:
            log_bulk.error("Failed to load stale FX cache after live fetch failure: %s", e)

    # 4) No FX available
    log_bulk.error("No FX cache available; proceeding WITHOUT conversion.")
    return None, "none", True


def _make_cfg_with_filters(
    *,
    base_cfg: Dict[str, Any],
    category_opt: str,
    corrupted: bool,
) -> Dict[str, Any]:
    """Return a deep-copied config with category + corrupted option set."""
    cfg = copy.deepcopy(base_cfg)

    # Ensure filters skeleton
    filters_obj = cfg.get("filters") or {}
    type_filters = (filters_obj.get("type_filters") or {})
    type_filters_filters = (type_filters.get("filters") or {})

    # Category selector
    type_filters_filters["category"] = {"option": category_opt}
    type_filters["filters"] = type_filters_filters
    type_filters["disabled"] = False
    filters_obj["type_filters"] = type_filters

    # Corrupted flag
    misc_filters = (filters_obj.get("misc_filters") or {})
    mf_filters = (misc_filters.get("filters") or {})
    mf_filters["corrupted"] = {"option": "true" if corrupted else "false"}
    misc_filters["filters"] = mf_filters
    misc_filters["disabled"] = False
    filters_obj["misc_filters"] = misc_filters

    cfg["filters"] = filters_obj

    # Status: ANY
    cfg["status"] = {"option": "any"}

    return cfg


async def _search_once_with_cfg(
    cfg: Dict[str, Any],
    *,
    fx_cache: Optional[Any] = None,
    skip_fx: bool = False,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Run one Trade2 search using an in-memory cfg and return a DataFrame (with FX-converted base price if available).
    - If fx_cache is provided, it will be used to construct PriceConverter.
    - If fx_cache is None and skip_fx is False, will attempt to fetch (legacy behavior).
    - If fx_cache is None and skip_fx is True, runs without conversion (amount_in_base=None).
    """
    setup_logging(log_level)

    # These loggers are inside your existing pipeline
    logging.getLogger("poe.trade2.search")
    logging.getLogger("poe.trade")

    log_bulk.debug("Resolving options/headers/payload from cfg...")
    opts = options_from_config(cfg)
    headers = headers_from_config(cfg)
    payload = payload_from_config(cfg)

    converter = None
    if fx_cache is not None:
        log_bulk.info(
            "Using provided FX cache | base_currency=%s base_server=%s realm=%s league=%s",
            opts["base_currency"], opts["base_server"].value, opts["realm"], opts["league"]
        )
        converter = PriceConverter(fx_cache, opts["base_server"], opts["base_currency"])
    elif not skip_fx:
        # Legacy behavior (not used in this bulk flow, since we pre-warm at run start)
        log_bulk.info(
            "No FX cache provided; fetching live | base_currency=%s base_server=%s realm=%s league=%s",
            opts["base_currency"], opts["base_server"].value, opts["realm"], opts["league"]
        )
        from take_home_project_poe.misc.poe2_currency_recorder_web_scrape import get_currency_cache as _get_cc
        fx_cache = _get_cc([opts["base_server"]], log_level=log_level)
        converter = PriceConverter(fx_cache, opts["base_server"], opts["base_currency"])
    else:
        log_bulk.warning("Running WITHOUT FX conversion for this slice (skip_fx=True).")

    total_timeout = aiohttp.ClientTimeout(total=opts["timeout"])
    async with aiohttp.ClientSession(timeout=total_timeout) as session:
        log_bulk.info("Issuing Trade2 search...")
        records = await search(
            session,
            realm=opts["realm"],
            league=opts["league"],
            filters=payload,
            headers=headers,
            timeout=opts["timeout"],
            max_retries=opts["max_retries"],
            post_request_pause=opts["post_request_pause"],
            converter=converter,   # converter can be None -> will skip conversion in record builder
        )

    df = records_to_dataframe(records)
    log_bulk.info("Search returned %d rows × %d cols", *df.shape)
    return df


def run_bulk_sampling_over_categories(
    categories: List[str] = CATEGORIES,
    corrupted_options: Tuple[bool, bool] = (False, True),
    *,
    base_currency: str = "Exalted Orb",
    base_server: str = "Standard",
    realm: str = "poe2",
    league: str = "Standard",
    timeout: int = 30,
    max_retries: int = 5,
    post_request_pause: float = 1.0,
    log_level: str = "INFO",
    outdir: Optional[Path] = None,
    fx_ttl_hours: int = 12,
    fx_force_refresh: bool = False,
) -> None:
    """
    For each (category, corrupted):
      - Build config with status='any'
      - Pre-warm FX once and persist to disk; reuse across slices and runs
      - Run search through the Trade2 pipeline
      - Save per-slice Parquet and a combined Parquet
      - Log summary with total combos, succeeded, failed, and failed names
      - Log progress as: category i/len(categories) and slice j/total_combos
    """
    setup_logging(log_level)
    log_bulk.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    base_cfg: Dict[str, Any] = {
        "realm": realm,
        "league": league,
        "timeout": timeout,
        "max_retries": max_retries,
        "post_request_pause": post_request_pause,
        "status": {"option": "any"},
        "stats": [{"type": "and", "filters": [], "disabled": False}],
        "filters": {"type_filters": {"filters": {}, "disabled": False}},
        "sort": {"price": "asc"},
        "base_currency": base_currency,
        "base_server": base_server,
    }

    here = Path(__file__).resolve().parent
    outdir = Path(outdir) if outdir else here
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Pre-warm / load FX once per bulk run
    fx_cache, fx_source, fx_is_stale = build_or_load_fx_cache(
        base_server=base_server,
        base_currency=base_currency,
        cache_dir=outdir,
        log_level=log_level,
        ttl_hours=fx_ttl_hours,
        force_refresh=fx_force_refresh,
        allow_stale_on_error=True,
    )
    skip_fx = fx_cache is None
    if not skip_fx:
        log_bulk.info("FX ready (source=%s, stale=%s).", fx_source, fx_is_stale)
    else:
        log_bulk.warning("FX unavailable (source=%s); conversion will be skipped for all slices.", fx_source)

    ts = time.strftime("%Y%m%d-%H%M%S")
    total_combos = len(categories) * len(corrupted_options)

    log_bulk.info(
        "Bulk run starting | categories=%d corrupted_options=%s TOTAL_COMBOS=%d outdir=%s timestamp=%s",
        len(categories), list(corrupted_options), total_combos, str(outdir), ts
    )
    log_bulk.info(
        "Global options | base_currency=%s base_server=%s realm=%s league=%s timeout=%s max_retries=%s pause=%.2f",
        base_currency, base_server, realm, league, timeout, max_retries, post_request_pause
    )

    all_dfs: List[pd.DataFrame] = []
    total_slices = 0
    succeeded_slices = 0
    failed_names: List[str] = []
    succeeded_names: List[str] = []

    for cat_idx, cat in enumerate(categories, start=1):
        cat_rows_sum = 0
        cat_failed = 0
        for corr in corrupted_options:
            total_slices += 1
            name_tag = f"{cat}:{'corrupted' if corr else 'uncorrupted'}"
            log_bulk.info(
                "---- Slice start: %s | category %d/%d | slice %d/%d ----",
                name_tag, cat_idx, len(categories), total_slices, total_combos
            )
            try:
                cfg = _make_cfg_with_filters(base_cfg=base_cfg, category_opt=cat, corrupted=corr)
                log_bulk.debug(
                    "Config constructed for slice: %s",
                    {k: cfg.get(k) for k in ("realm", "league", "status", "base_currency", "base_server")}
                )

                df = asyncio.run(_search_once_with_cfg(cfg, fx_cache=fx_cache, skip_fx=skip_fx, log_level=log_level))

                # Add labels BEFORE saving
                df.insert(0, "label", cat)
                df.insert(1, "bulk_corrupted", corr)
                df.insert(2, "base_currency", base_currency)
                log_bulk.info("Annotated DataFrame | shape=%s", tuple(df.shape))

                # Save per-slice parquet
                safe_cat = cat.replace(".", "_")
                tag = f"{safe_cat}_{'corrupted' if corr else 'uncorrupted'}"
                parquet_path = outdir / "training_data" / f"poe2_{ts}_{tag}.parquet"
                log_bulk.info(
                    "Preparing to save slice %s | rows=%d cols=%d -> %s",
                    name_tag, len(df), df.shape[1], parquet_path
                )
                df.to_parquet(parquet_path, index=False)
                log_bulk.info("Saved per-slice Parquet -> %s", parquet_path)

                all_dfs.append(df)
                succeeded_slices += 1
                succeeded_names.append(name_tag)
                cat_rows_sum += len(df)

            except Exception as e:
                failed_names.append(name_tag)
                cat_failed += 1
                log_bulk.exception("Slice FAILED for %s: %s", name_tag, e)

            log_bulk.info("---- Slice end: %s ----", name_tag)

        # Per-category summary after both corruption states
        log_bulk.info(
            "Category summary: %s | category %d/%d | total_rows=%d | failed_slices=%d/%d",
            cat, cat_idx, len(categories), cat_rows_sum, cat_failed, len(corrupted_options)
        )

    failed_count = total_slices - succeeded_slices
    log_bulk.info(
        "Slices finished | TOTAL=%d (expected=%d) SUCCEEDED=%d FAILED=%d",
        total_slices, total_combos, succeeded_slices, failed_count
    )
    if failed_names:
        log_bulk.warning("Failed slice names (%d): %s", len(failed_names), ", ".join(failed_names))
    if succeeded_names:
        log_bulk.info("Succeeded slice names (%d): %s", len(succeeded_names), ", ".join(succeeded_names))

    # Save combined parquet if we have any successful slices
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = outdir / f"poe2_{ts}_combined.parquet"
        combined.to_parquet(combined_path, index=False)
        log_bulk.info("Saved combined Parquet -> %s | combined shape=%s", combined_path, tuple(combined.shape))
    else:
        log_bulk.warning("No successful slices; combined Parquet will not be written.")


if __name__ == "__main__":
    run_bulk_sampling_over_categories(
        categories=CATEGORIES,
        corrupted_options=(False, True),
        base_currency="Exalted Orb",
        base_server="Standard",
        realm="poe2",
        league="Standard",
        timeout=30,
        max_retries=5,
        post_request_pause=1.0,
        log_level="INFO",
        outdir=None,          # or Path("data_out")
        fx_ttl_hours=12,      # cache considered fresh up to 12h
        fx_force_refresh=False,
    )
