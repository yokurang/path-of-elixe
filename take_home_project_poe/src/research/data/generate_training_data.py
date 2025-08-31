# bulk_collect_categories.py
# Collect sample training data across all categories × {corrupted, not}
# - status="any"
# - base_currency="Exalted Orb"
# - league="Standard"
# - realm="poe2" (kept for Trade2 API compatibility)
#
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import aiohttp

from src.data.constants import CATEGORIES

# ---- Import your existing pipeline bits
from src.data.market_currency_listener import get_currency_cache
from src.data.market_item_listener import ( 
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


async def _search_once_with_cfg(cfg: Dict[str, Any], log_level: str = "INFO") -> pd.DataFrame:
    """
    Run one Trade2 search using an in-memory cfg and return a DataFrame (with FX-converted base price).
    Uses your existing market_listener pipeline; this function just wires it together.
    """
    # Ensure overall logging is configured (respects your setup_logging)
    setup_logging(log_level)

    # These loggers are inside your existing pipeline
    log_search = logging.getLogger("poe.trade2.search")
    log_fetch = logging.getLogger("poe.trade")

    log_bulk.debug("Resolving options/headers/payload from cfg...")
    opts = options_from_config(cfg)
    headers = headers_from_config(cfg)
    payload = payload_from_config(cfg)

    log_bulk.info(
        "Preparing FX cache and converter | base_currency=%s base_server=%s realm=%s league=%s",
        opts["base_currency"], opts["base_server"].value, opts["realm"], opts["league"]
    )
    fx_cache = get_currency_cache([opts["base_server"]], log_level=log_level)
    converter = PriceConverter(fx_cache, opts["base_server"], opts["base_currency"])

    log_bulk.info(
        "Starting aiohttp session | timeout=%s max_retries=%s post_request_pause=%.2fs",
        opts["timeout"], opts["max_retries"], opts["post_request_pause"]
    )
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
            converter=converter,
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
) -> None:
    """
    For each (category, corrupted):
      - Build config with status='any'
      - Run search through the Trade2 pipeline
      - Save per-slice Parquet and a combined Parquet
    """
    # Set up logging once for the bulk runner
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
        # If you want to inject cookies directly here, uncomment:
        # "poe_sessid": "...",
        # "cf_clearance": "...",
    }

    here = Path(__file__).resolve().parent
    outdir = Path(outdir) if outdir else here
    outdir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    log_bulk.info(
        "Bulk run starting | categories=%d corrupted_options=%s outdir=%s timestamp=%s",
        len(categories), list(corrupted_options), str(outdir), ts
    )
    log_bulk.info(
        "Global options | base_currency=%s base_server=%s realm=%s league=%s timeout=%s max_retries=%s pause=%.2f",
        base_currency, base_server, realm, league, timeout, max_retries, post_request_pause
    )

    all_dfs: List[pd.DataFrame] = []
    total_slices = 0
    succeeded_slices = 0

    for cat in categories:
        for corr in corrupted_options:
            total_slices += 1
            log_bulk.info("---- Slice start: category=%s | corrupted=%s ----", cat, corr)
            try:
                cfg = _make_cfg_with_filters(base_cfg=base_cfg, category_opt=cat, corrupted=corr)
                log_bulk.debug("Config constructed for slice: %s", {k: cfg.get(k) for k in ("realm", "league", "status", "base_currency", "base_server")})

                df = asyncio.run(_search_once_with_cfg(cfg, log_level=log_level))

                # Add label and corruption flag BEFORE saving
                df.insert(0, "label", cat)
                df.insert(1, "bulk_corrupted", corr)
                log_bulk.info("Annotated DataFrame | shape=%s", tuple(df.shape))

                # Save per-slice parquet
                safe_cat = cat.replace(".", "_")
                tag = f"{safe_cat}_{'corrupted' if corr else 'uncorrupted'}"
                parquet_path = outdir / f"poe2_{ts}_{tag}.parquet"
                df.to_parquet(parquet_path, index=False)
                log_bulk.info("Saved per-slice Parquet -> %s", parquet_path)

                all_dfs.append(df)
                succeeded_slices += 1

            except Exception as e:
                log_bulk.exception("Slice failed for category=%s corrupted=%s: %s", cat, corr, e)

            log_bulk.info("---- Slice end: category=%s | corrupted=%s ----", cat, corr)

    log_bulk.info("Slices finished | total=%d succeeded=%d failed=%d", total_slices, succeeded_slices, total_slices - succeeded_slices)

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
        outdir=None,
    )
