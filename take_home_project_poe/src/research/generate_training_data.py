# file: src/preprocessor/generate_training_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import re

import pandas as pd

# Updated recorder imports - use the new paginated function
from src.recorder.poe_trade2_recorder import (
    search_to_dataframe_with_limit,  # NEW: Use paginated search
    build_basic_payload,
)
from src.recorder.constants import ITEM_TYPES, ITEM_RARITIES

# =============================================================================
# Paths / constants
# =============================================================================
TRAINING_DIR = Path("new/training_data")
OVERALL_DIR = TRAINING_DIR / "overall"
OUTPUT_LOG = Path("training.data.log")

# LEAGUE = "Standard"
LEAGUE = "Rise%20of%20the%20Abyssal"
BASE_CURRENCY = "Exalted Orb"  # assignment requires Exalted Orb as base
PER_COMBO_DEFAULT = 7000       # target unique rows/ids per combo; ~100 per page

# Root logger + file handler
_root = logging.getLogger()
if not _root.handlers:
    _root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    _root.addHandler(ch)
    try:
        OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(OUTPUT_LOG, encoding="utf-8")
        fh.setFormatter(fmt)
        _root.addHandler(fh)
    except Exception:
        # If file handler fails, continue with console-only logging
        pass

log = logging.getLogger("poe.training.collector")

# =============================================================================
# Updated pagination helper with better logging
# =============================================================================
def _paged_collect_with_pagination(
    *,
    base_payload: Dict[str, Any],
    base_currency: str,
    league: str,
    realm: Optional[str],
    recorder_log_level: str,
    category_key: str,              # namespaced key for per-category de-dupe
    seen_by_category: Dict[str, Set[str]],
    per_combo: int,
) -> pd.DataFrame:
    """
    Use smart pagination with comprehensive caching:
    1. Category-level cache prevents re-scraping across combos
    2. Request-level cache prevents infinite loops within pagination
    """
    seen = seen_by_category.setdefault(category_key, set())
    need_new = per_combo
    
    log.info("Starting smart pagination for %s: need=%d new items, category_cache_size=%d", 
             category_key, need_new, len(seen))
    
    try:
        # STEP 1: Make a small request to get total_available count
        small_payload = dict(base_payload)
        small_payload["offset"] = 0  # Ensure we start from beginning
        
        df_sample, sample_info = search_to_dataframe_with_limit(
            payload=small_payload,
            base_currency=base_currency,
            max_results=100,  # Just get first page to learn total_available
            league=league,
            realm=realm,
            save_csv=None,
            log_level=recorder_log_level,
        )
        
        total_available = sample_info.total_items_available
        log.info("Discovery complete for %s:", category_key)
        log.info("  Total available items: %d", total_available)
        
        # STEP 2: Calculate smart limit based on what we haven't seen yet
        # Check how many items from sample are actually new to us
        if not df_sample.empty and "id" in df_sample.columns:
            sample_ids = df_sample["id"].astype(str)
            sample_new_mask = ~sample_ids.isin(seen)
            sample_new_count = sample_new_mask.sum()
            log.info("  Sample analysis: %d total, %d new, %d already seen", 
                     len(df_sample), sample_new_count, len(df_sample) - sample_new_count)
        else:
            sample_new_count = 0
        
        # If we already have enough new items in the sample, we might be done
        if sample_new_count >= need_new:
            log.info("Sample has sufficient new items (%d >= %d needed)", sample_new_count, need_new)
            max_to_request = 100
        else:
            # Calculate smart limit: consider deduplication rate from sample
            if len(df_sample) > 0:
                dedup_rate = sample_new_count / len(df_sample)  # Fraction that are new
                estimated_needed = min(
                    total_available,
                    int(need_new / max(dedup_rate, 0.1))  # Account for dedup, but don't go crazy
                )
            else:
                estimated_needed = min(total_available, need_new * 2)
            
            max_to_request = min(
                total_available,  # Can't request more than exists
                estimated_needed  # Smart estimate based on dedup rate
            )
        
        log.info("Smart limits for %s:", category_key)
        log.info("  Need new: %d", need_new)
        log.info("  Total available: %d", total_available)
        log.info("  Will request: %d", max_to_request)
        
        # STEP 3: If we already have the sample data and it's sufficient, use it
        if max_to_request <= 100 and not df_sample.empty:
            log.info("Using sample data (sufficient for request)")
            df_all = df_sample
            pagination_info = sample_info
        else:
            # STEP 4: Make the full request with smart limit
            df_all, pagination_info = search_to_dataframe_with_limit(
                payload=base_payload,
                base_currency=base_currency,
                max_results=max_to_request,  # Smart limit!
                league=league,
                realm=realm,
                save_csv=None,
                log_level=recorder_log_level,
            )
        
        # Log final pagination results
        log.info("Pagination complete for %s:", category_key)
        log.info("  Total available items: %d", pagination_info.total_items_available)
        log.info("  Total pages available: %d", pagination_info.total_pages_available) 
        log.info("  Pages scraped: %d", pagination_info.pages_scraped)
        log.info("  IDs collected: %d", pagination_info.ids_collected)
        log.info("  IDs successfully fetched: %d", pagination_info.ids_successfully_fetched)
        log.info("  Max results limit used: %d", pagination_info.max_results_limit)
        
        if df_all.empty or "id" not in df_all.columns:
            log.warning("No valid data returned for %s", category_key)
            return pd.DataFrame()
        
        # STEP 5: Apply cross-combo deduplication using category cache
        ids_series = df_all["id"].astype(str)
        new_mask = ~ids_series.isin(seen)
        df_new = df_all.loc[new_mask].copy()
        
        log.info("Cross-combo deduplication for %s:", category_key)
        log.info("  Raw fetched: %d items", len(df_all))
        log.info("  After category deduplication: %d new items", len(df_new))
        log.info("  Already seen in category: %d items filtered", len(df_all) - len(df_new))
        log.info("  Category cache now has: %d total IDs", len(seen))
        
        # STEP 6: Take only what we need
        if len(df_new) > per_combo:
            df_new = df_new.head(per_combo)
            log.info("  Capped to target: %d items", len(df_new))
        
        # STEP 7: Update category-level cache
        new_ids = df_new["id"].astype(str).tolist()
        seen.update(new_ids)
        
        log.info("Final result for %s: collected %d new items, category_cache_size=%d", 
                 category_key, len(df_new), len(seen))
        
        return df_new
        
    except Exception as e:
        log.error("Smart pagination failed for %s: %s", category_key, e)
        return pd.DataFrame()


def _paged_collect_manual(
    *,
    base_payload: Dict[str, Any],
    base_currency: str,
    league: str,
    realm: Optional[str],
    recorder_log_level: str,
    category_key: str,              # namespaced key for per-category de-dupe
    seen_by_category: Dict[str, Set[str]],
    per_combo: int,
    page_step: Optional[int] = 100, # how far to bump offset per loop (≈ API page size)
    max_pages: Optional[int] = None,
) -> pd.DataFrame:
    """
    ENHANCED manual pagination with detailed offset and ID logging.
    Keep this as backup option or for fine-grained control.
    """
    collected_parts: List[pd.DataFrame] = []
    seen = seen_by_category.setdefault(category_key, set())

    offset = 0
    pages = 0
    total_new = 0
    processed_ids: Set[str] = set()           # unique ids we've *processed* this combo
    last_ids_set: Optional[Set[str]] = None   # to detect identical pages (stall guard)

    log.info("Starting manual pagination for %s: target=%d, already_seen=%d", 
             category_key, per_combo, len(seen))

    while True:
        if max_pages is not None and pages >= max_pages:
            log.info("Reached max_pages=%d at offset=%d for %s", max_pages, offset, category_key)
            break

        payload = dict(base_payload)
        payload["offset"] = offset

        # Enhanced logging: show what we're about to request
        log.info("=== PAGE REQUEST for %s ===", category_key)
        log.info("  Offset: %d", offset)
        log.info("  Page number: %d", pages + 1)
        log.info("  Progress: %d/%d items collected so far", total_new, per_combo)

        try:
            # Use single-page search with explicit offset
            df_page = search_to_dataframe_with_limit(
                payload=payload,
                base_currency=base_currency,
                max_results=100,  # Single page worth
                league=league,
                realm=realm,
                save_csv=None,
                log_level=recorder_log_level,
            )[0]  # Get just the DataFrame, ignore pagination info for single page
        except Exception as e:
            log.warning("Page fetch failed at offset=%d for %s: %s", offset, category_key, e)
            break

        pages += 1

        if df_page is None or df_page.empty:
            log.info("Empty page at offset=%d for %s — stopping.", offset, category_key)
            break
        if "id" not in df_page.columns:
            log.warning("No 'id' column in page at offset=%d for %s; stopping.", offset, category_key)
            break

        # Enhanced logging: show what we received
        log.info("=== PAGE RESPONSE for %s ===", category_key)
        log.info("  Received: %d items", len(df_page))
        log.info("  Offset was: %d", offset)

        # Page id set (unique within the page)
        ids_series = df_page["id"].astype(str)
        ids_set = set(ids_series.unique())
        unique_ids_this_page = len(ids_set)

        log.info("  Unique IDs in this page: %d", unique_ids_this_page)
        log.info("  Sample IDs: %s", list(ids_set)[:3] if ids_set else "[]")

        # 🔒 Stall guard #1: identical page content as last time
        if last_ids_set is not None and ids_set == last_ids_set:
            log.info("Same page repeated at offset=%d for %s — stopping.", offset, category_key)
            break
        last_ids_set = ids_set

        # Track per-combo processed ids (unique)
        before_proc = len(processed_ids)
        processed_ids.update(ids_set)
        added_proc = len(processed_ids) - before_proc

        # Keep only rows NEW to this category (cross-combo de-dup)
        new_mask = ~ids_series.isin(seen)
        df_new = df_page.loc[new_mask].copy()

        # Enhanced logging: show deduplication results
        log.info("=== DEDUPLICATION for %s ===", category_key)
        log.info("  Already seen in category: %d items filtered out", len(df_page) - len(df_new))
        log.info("  New items for category: %d", len(df_new))
        log.info("  New unique IDs processed this combo: %d", added_proc)

        # 🔒 Stall guard #2: no progress (no new processed ids AND no new rows to collect)
        if added_proc == 0 and df_new.empty:
            log.info("No new ids to process or collect at offset=%d for %s — stopping.", offset, category_key)
            break

        # Cap what we *collect* to what's still needed
        need_collect = per_combo - total_new
        if need_collect <= 0:
            log.info("Already collected enough items (%d/%d) for %s", total_new, per_combo, category_key)
            df_new = df_new.iloc[0:0]
        elif len(df_new) > need_collect:
            log.info("Capping collection: need %d more, got %d new, taking first %d", 
                     need_collect, len(df_new), need_collect)
            df_new = df_new.head(need_collect)

        # Update category-level seen set with the rows we're actually keeping
        if not df_new.empty:
            for iid in df_new["id"].astype(str):
                seen.add(iid)
            collected_parts.append(df_new)
            total_new += len(df_new)

        # Enhanced progress log
        pct_collect = (total_new / per_combo) * 100
        pct_proc = (len(processed_ids) / per_combo) * 100
        log.info("=== PROGRESS SUMMARY for %s ===", category_key)
        log.info("  Offset: %d | Page: %d", offset, pages)
        log.info("  Collected: %d/%d (%.1f%%)", total_new, per_combo, pct_collect)
        log.info("  Processed unique IDs: %d/%d (%.1f%%)", len(processed_ids), per_combo, pct_proc)
        log.info("  Total seen in category: %d", len(seen))

        # Stop when we've processed enough unique ids or collected enough new rows
        if len(processed_ids) >= per_combo:
            log.info("Processed target per_combo=%d unique ids for %s; stopping.", per_combo, category_key)
            break
        if total_new >= per_combo:
            log.info("Collected target per_combo=%d new rows for %s; stopping.", per_combo, category_key)
            break

        # Optional end-of-results guard: if we got a short page (typ. < API page size), likely no more
        if page_step is not None and len(df_page) < page_step:
            log.info("Got a short page (%d < %d) for %s — likely end of results. Stopping.", 
                     len(df_page), page_step, category_key)
            break

        # Advance offset for next loop
        step = page_step if page_step is not None else len(df_page)
        offset += step
        log.info("Advancing to next page: offset %d -> %d (+%d)", offset - step, offset, step)

    log.info("=== COLLECTION COMPLETE for %s ===", category_key)
    log.info("  Total pages scraped: %d", pages)
    log.info("  Final items collected: %d", total_new)
    log.info("  Unique IDs processed: %d", len(processed_ids))

    if collected_parts:
        result = pd.concat(collected_parts, ignore_index=True)
        log.info("  Final DataFrame: %d rows", len(result))
        return result
    return pd.DataFrame()


# =============================================================================
# Small helpers
# =============================================================================
def _slug_u(s: str) -> str:
    """lowercase, non-alnum -> underscore; trim leading/trailing underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _ensure_dirs() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    OVERALL_DIR.mkdir(parents=True, exist_ok=True)


def _add_corrupted(payload: Dict[str, Any], is_corrupted: bool) -> None:
    q = payload.setdefault("query", {})
    filters = q.setdefault("filters", {})
    misc = filters.setdefault("misc_filters", {"filters": {}, "disabled": False})
    misc.setdefault("filters", {})["corrupted"] = {"option": "true" if is_corrupted else "false"}


def _select_unique_for_category(
    df: pd.DataFrame,
    category_key: str,
    seen_by_category: Dict[str, Set[str]],
    per_combo: int,
) -> pd.DataFrame:
    """
    Take up to `per_combo` rows from df whose ids were not seen for this category_key.
    (Ensures uniqueness inside each category across all passes.)
    """
    if df.empty or "id" not in df.columns:
        return pd.DataFrame()
    seen = seen_by_category.setdefault(category_key, set())
    ids = df["id"].astype(str)
    take = df.loc[~ids.isin(seen)].head(per_combo).copy()
    for iid in take["id"].astype(str):
        seen.add(iid)
    return take


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        log.info("No rows to save -> %s (skipped)", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("Saved %d rows -> %s", len(df), path)


# ---- De-dupe key + warm-load -------------------------------------------------
def _seen_key(category_key: str, league: str, realm: Optional[str]) -> str:
    """Namespace the seen set by market and category to avoid cross-market collisions."""
    return f"{_league_fs(league)}|{(realm or '').strip()}|{category_key}"


def _warm_seen_from_overall(
    seen_by_category: Dict[str, Set[str]],
    *,
    seen_key: str,
    category_key: str,
    league: str,
) -> None:
    """Pre-populate seen set from existing overall parquet (if present)."""
    overall_name = make_overall_filename(category_key=category_key, league=league)
    overall_path = OVERALL_DIR / overall_name
    if not overall_path.exists():
        return
    try:
        df = pd.read_parquet(overall_path, columns=["id"])
        ids = df["id"].astype(str).tolist()
        seen_by_category.setdefault(seen_key, set()).update(ids)
        log.info("Warm-loaded %d seen ids from %s", len(ids), overall_path)
    except Exception as e:
        log.warning("Failed to warm-load overall parquet %s: %s", overall_path, e)


# =============================================================================
# Naming helpers (deterministic, round-trippable without regex)
# =============================================================================
def _cat_to_fs(category_key: str) -> str:
    """'armour.helmet' -> 'armour_helmet'"""
    return _slug_u(category_key.replace(".", "_"))


def _league_fs(league: str) -> str:
    """'Standard' -> 'standard'"""
    return _slug_u(league)


def make_combo_filename(
    *,
    category_key: str,
    rarity: Optional[str],
    corrupted: Optional[bool],
    league: str,
) -> str:
    cat = _cat_to_fs(category_key)
    rar = _slug_u(rarity) if rarity is not None else "any"
    # Use requested wording: "corrupted" vs "not_corrupted" (or "any")
    if corrupted is True:
        corr = "corrupted"
    elif corrupted is False:
        corr = "not_corrupted"
    else:
        corr = "any"
    lg = _league_fs(league)
    # Example: armour_helmet_rarity_unique_corrupted_not_corrupted_standard.parquet
    return f"{cat}_rarity_{rar}_corrupted_{corr}_{lg}.parquet"


def make_overall_filename(*, category_key: str, league: str) -> str:
    cat = _cat_to_fs(category_key)
    lg = _league_fs(league)
    # Example: armour_helmet_overall_standard.parquet
    return f"{cat}_overall_{lg}.parquet"


# =============================================================================
# Core: generate training data (progressive saving) with pagination choice
# =============================================================================
def generate_training_data(
    *,
    per_combo: int = PER_COMBO_DEFAULT,
    base_currency: str = BASE_CURRENCY,
    league: str = LEAGUE,
    realm: Optional[str] = None,               # if None, recorder will read realm from config
    recorder_log_level: str = "INFO",          # Reduced default verbosity
    categories: Optional[List[str]] = None,    # e.g. ["weapon.wand"] to limit scope
    use_auto_pagination: bool = True,          # NEW: Choose pagination method
) -> pd.DataFrame:
    """
    Progressive writer - now only does rarity combinations (no corrupted pass):
      • For each category: iterate rarities (normal, magic, rare, unique)
      • Save each combo parquet immediately
      • Save category-level overall parquet
      • At the end, save a manifest

    Uniqueness is enforced per category across all rarity combinations.
    """
    _ensure_dirs()

    # Set logging levels
    try:
        lvl = getattr(logging, recorder_log_level.upper(), logging.INFO)
        logging.getLogger("poe.trade2.fetch").setLevel(lvl)
        logging.getLogger("poe.trade2.search").setLevel(lvl)
    except Exception:
        pass

    # Validate category subset
    cat_keys = list(ITEM_TYPES.keys()) if not categories else categories
    invalid = [c for c in cat_keys if c not in ITEM_TYPES]
    if invalid:
        raise ValueError(f"Unknown category keys: {invalid}")

    seen_by_category: Dict[str, Set[str]] = {}
    manifest_rows: List[Dict[str, Any]] = []

    total_categories = len(cat_keys)
    per_cat_combos = len(ITEM_RARITIES)   # Only rarity combos now
    global_total = total_categories * per_cat_combos
    gidx = 0

    pagination_method = "auto-pagination" if use_auto_pagination else "manual-pagination"
    log.info(
        "Starting generation: categories=%d, rarity_combos=%d, "
        "league=%s, base=%s, per_combo=%d, method=%s",
        total_categories, per_cat_combos, league, base_currency, per_combo, pagination_method
    )

    for cidx, cat_key in enumerate(cat_keys, start=1):
        log.info("========== CATEGORY %d/%d: %s ==========", cidx, total_categories, cat_key)
        per_cat_parts: List[pd.DataFrame] = []

        # Build namespaced seen key & warm-load from existing overall parquet
        seen_key = _seen_key(cat_key, league, realm)
        _warm_seen_from_overall(seen_by_category, seen_key=seen_key, category_key=cat_key, league=league)

        # ---------------- RARITY COMBINATIONS ----------------
        for ridx, rar_key in enumerate(ITEM_RARITIES.keys(), start=1):
            gidx += 1
            log.info("[%d/%d] %s + %s", gidx, global_total, cat_key, rar_key)

            base_payload = build_basic_payload(
                category_key=cat_key,
                rarity_key=rar_key,
                status_option="securable",
                sort_key="price",
                sort_dir="asc",
            )

            # Choose pagination method
            if use_auto_pagination:
                df_take = _paged_collect_with_pagination(
                    base_payload=base_payload,
                    base_currency=base_currency,
                    league=league,
                    realm=realm,
                    recorder_log_level=recorder_log_level,
                    category_key=seen_key,
                    seen_by_category=seen_by_category,
                    per_combo=per_combo,
                )
            else:
                df_take = _paged_collect_manual(
                    base_payload=base_payload,
                    base_currency=base_currency,
                    league=league,
                    realm=realm,
                    recorder_log_level=recorder_log_level,
                    category_key=seen_key,
                    seen_by_category=seen_by_category,
                    per_combo=per_combo,
                    page_step=100,
                    max_pages=None,
                )

            combo_name = make_combo_filename(category_key=cat_key, rarity=rar_key, corrupted=None, league=league)
            combo_path = TRAINING_DIR / combo_name
            _save_parquet(df_take, combo_path)
            log.info("Saved %d rows -> %s", len(df_take), combo_path.name)

            if not df_take.empty:
                per_cat_parts.append(df_take)
                manifest_rows.append({
                    "phase": "cat×rarity",
                    "category": cat_key,
                    "rarity": rar_key,
                    "corrupted": None,
                    "rows": len(df_take),
                    "file": str(combo_path),
                })

        # ---------------- OVERALL per category ----------------
        if per_cat_parts:
            df_cat = pd.concat(per_cat_parts, ignore_index=True)
            if "id" in df_cat.columns:
                df_cat = df_cat.drop_duplicates(subset=["id"])
            overall_name = make_overall_filename(category_key=cat_key, league=league)
            overall_path = OVERALL_DIR / overall_name
            _save_parquet(df_cat, overall_path)
            log.info("Category overall: %d unique rows -> %s", len(df_cat), overall_path.name)
            manifest_rows.append({
                "phase": "overall",
                "category": cat_key,
                "rarity": None,
                "corrupted": None,
                "rows": len(df_cat),
                "file": str(overall_path),
            })
        else:
            log.info("No rows collected for category=%s", cat_key)

    # Final manifest
    manifest = pd.DataFrame(manifest_rows)
    man_path = OVERALL_DIR / f"manifest_{_league_fs(league)}.parquet"
    if not manifest.empty:
        manifest.to_parquet(man_path, index=False)
        log.info("Manifest saved: %d entries -> %s", len(manifest), man_path.name)
    else:
        log.warning("Manifest is empty")
    return manifest


# =============================================================================
# main
# =============================================================================
def main() -> None:
    """
    Run a full Standard-league training snapshot in Exalted Orbs with pagination.
    """
    try:
        generate_training_data(
            per_combo=PER_COMBO_DEFAULT,
            base_currency=BASE_CURRENCY,   # Exalted Orb for the assignment
            league=LEAGUE,                 # Standard
            realm=None,                    # let recorder fall back to config for realm (poe2)
            recorder_log_level="DEBUG",    # show per-id GET lines from poe.trade2.fetch
            categories=None,               # or e.g. ["weapon.wand"] to limit scope
            use_auto_pagination=True,      # NEW: Use efficient built-in pagination
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Exiting gracefully.")


if __name__ == "__main__":
    main()