# file: src/preprocessor/generate_training_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import re

import pandas as pd

# Recorder imports
from src.recorder.poe_trade2_recorder import (
    search_to_dataframe,
    build_basic_payload,
)
from src.recorder.constants import ITEM_TYPES, ITEM_RARITIES

# =============================================================================
# Paths / constants
# =============================================================================
TRAINING_DIR = Path("securable/training_data")
OVERALL_DIR = TRAINING_DIR / "overall"
OUTPUT_LOG = Path("output_generate_training_data.log")

LEAGUE = "Standard"
# LEAGUE = "Rise%20of%20the%20Abyssal"
BASE_CURRENCY = "Exalted Orb"  # assignment requires Exalted Orb as base
PER_COMBO_DEFAULT = 3000       # target unique rows/ids per combo; ~100 per page

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
# Pagination helper
# =============================================================================
def _paged_collect(
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
    Walk paginated results by bumping `offset` until we hit one of:
      • processed `per_combo` unique ids for this combo, OR
      • collected `per_combo` new-to-category rows, OR
      • no progress (same page / no new ids), OR
      • empty page / missing 'id' column / max_pages.

    De-dupe across combos is enforced via `seen_by_category[category_key]`.
    """
    collected_parts: List[pd.DataFrame] = []
    seen = seen_by_category.setdefault(category_key, set())

    offset = 0
    pages = 0
    total_new = 0
    processed_ids: Set[str] = set()           # unique ids we've *processed* this combo
    last_ids_set: Optional[Set[str]] = None   # to detect identical pages (stall guard)

    while True:
        if max_pages is not None and pages >= max_pages:
            log.info("Reached max_pages=%d at offset=%d", max_pages, offset)
            break

        payload = dict(base_payload)
        payload["offset"] = offset

        try:
            df_page = search_to_dataframe(
                payload=payload,
                base_currency=base_currency,
                league=league,
                realm=realm,
                save_csv=None,
                log_level=recorder_log_level,
            )
        except Exception as e:
            log.warning("Page fetch failed at offset=%d: %s", offset, e)
            break

        pages += 1

        if df_page is None or df_page.empty:
            log.info("Empty page at offset=%d — stopping.", offset)
            break
        if "id" not in df_page.columns:
            log.warning("No 'id' column in page at offset=%d; stopping.", offset)
            break

        # Page id set (unique within the page)
        ids_series = df_page["id"].astype(str)
        ids_set = set(ids_series.unique())

        # 🔒 Stall guard #1: identical page content as last time
        if last_ids_set is not None and ids_set == last_ids_set:
            log.info("Same page repeated at offset=%d — stopping.", offset)
            break
        last_ids_set = ids_set

        # Track per-combo processed ids (unique)
        before_proc = len(processed_ids)
        processed_ids.update(ids_set)
        added_proc = len(processed_ids) - before_proc

        # Keep only rows NEW to this category (cross-combo de-dup)
        new_mask = ~ids_series.isin(seen)
        df_new = df_page.loc[new_mask].copy()

        # 🔒 Stall guard #2: no progress (no new processed ids AND no new rows to collect)
        if added_proc == 0 and df_new.empty:
            log.info("No new ids to process or collect at offset=%d — stopping.", offset)
            break

        # Cap what we *collect* to what's still needed
        need_collect = per_combo - total_new
        if need_collect <= 0:
            df_new = df_new.iloc[0:0]
        elif len(df_new) > need_collect:
            df_new = df_new.head(need_collect)

        # Update category-level seen set with the rows we're actually keeping
        if not df_new.empty:
            for iid in df_new["id"].astype(str):
                seen.add(iid)
            collected_parts.append(df_new)
            total_new += len(df_new)

        # Progress log
        pct_collect = (total_new / per_combo) * 100
        pct_proc = (len(processed_ids) / per_combo) * 100
        log.info(
            "Progress [cat=%s | offset=%d | page=%d]: collected %d/%d (%.1f%%), processed %d/%d (%.1f%%)",
            category_key, offset, pages, total_new, per_combo, pct_collect, len(processed_ids), per_combo, pct_proc
        )

        # Stop when we've processed enough unique ids or collected enough new rows
        if len(processed_ids) >= per_combo:
            log.info("Processed target per_combo=%d unique ids; stopping.", per_combo)
            break
        if total_new >= per_combo:
            log.info("Collected target per_combo=%d new rows; stopping.", per_combo)
            break

        # Optional end-of-results guard: if we got a short page (typ. < API page size), likely no more
        if page_step is not None and len(df_page) < page_step:
            log.info("Got a short page (%d < %d) — likely end of results. Stopping.", len(df_page), page_step)
            break

        # Advance offset for next loop
        step = page_step if page_step is not None else len(df_page)
        offset += step

    if collected_parts:
        return pd.concat(collected_parts, ignore_index=True)
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
# Core: generate training data (progressive saving)
# =============================================================================
def generate_training_data(
    *,
    per_combo: int = PER_COMBO_DEFAULT,
    base_currency: str = BASE_CURRENCY,
    league: str = LEAGUE,
    realm: Optional[str] = None,               # if None, recorder will read realm from config
    recorder_log_level: str = "DEBUG",         # DEBUG shows per-id GET lines from recorder
    categories: Optional[List[str]] = None,    # e.g. ["weapon.wand"] to limit scope
) -> pd.DataFrame:
    """
    Progressive writer:
      • For each category:
          - PASS 1: iterate rarities, save each combo parquet immediately
          - PASS 2: iterate corrupted={true,false}, save each combo parquet immediately
          - OVERALL: immediately combine/dedupe and save category-level parquet
      • At the very end, save a manifest.

    Uniqueness is enforced per category across both passes using a per-category ID cache.
    """
    _ensure_dirs()

    # Ensure recorder sub-loggers show per-id fetch lines if asked
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
    per_cat_pass1 = len(ITEM_RARITIES)   # rarity combos
    per_cat_pass2 = 2                    # corrupted true/false
    global_total = total_categories * (per_cat_pass1 + per_cat_pass2)
    gidx = 0

    log.info(
        "Starting generation: categories=%d, combos_per_category=%d (rarities=%d + corrupted=%d), "
        "league=%s, base=%s, per_combo=%d",
        total_categories, per_cat_pass1 + per_cat_pass2, per_cat_pass1, per_cat_pass2,
        league, base_currency, per_combo
    )

    for cidx, cat_key in enumerate(cat_keys, start=1):
        log.info("========== CATEGORY %d/%d: %s ==========", cidx, total_categories, cat_key)
        per_cat_parts: List[pd.DataFrame] = []

        # Build a namespaced seen key & warm-load from existing overall parquet
        seen_key = _seen_key(cat_key, league, realm)
        _warm_seen_from_overall(seen_by_category, seen_key=seen_key, category_key=cat_key, league=league)

        # ---------------- PASS 1: category × rarity ----------------
        for ridx, rar_key in enumerate(ITEM_RARITIES.keys(), start=1):
            gidx += 1
            combo_title = f"cat={cat_key} | rarity={rar_key} | corrupted=any | league={league} | base={base_currency}"
            log.info("[global %d/%d][cat %d/%d][pass1 %d/%d] START %s",
                     gidx, global_total, cidx, total_categories, ridx, per_cat_pass1, combo_title)

            base_payload = build_basic_payload(
                category_key=cat_key,
                rarity_key=rar_key,
                status_option="securable",
                sort_key="price",
                sort_dir="asc",
            )

            df_take = _paged_collect(
                base_payload=base_payload,
                base_currency=base_currency,
                league=league,
                realm=realm,
                recorder_log_level=recorder_log_level,
                category_key=seen_key,          # namespaced dedupe key
                seen_by_category=seen_by_category,
                per_combo=per_combo,
                page_step=100,                  # keep offset aligned with API pages
                max_pages=None,
            )

            combo_name = make_combo_filename(category_key=cat_key, rarity=rar_key, corrupted=None, league=league)
            combo_path = TRAINING_DIR / combo_name
            _save_parquet(df_take, combo_path)
            log.info("DONE combo -> %s (rows_saved=%d)", combo_path, len(df_take))

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

        # ---------------- PASS 2: category × corrupted ----------------
        for p2idx, is_corr in enumerate((True, False), start=1):
            gidx += 1
            corr_label = "corrupted" if is_corr else "not_corrupted"
            combo_title = f"cat={cat_key} | rarity=any | corrupted={corr_label} | league={league} | base={base_currency}"
            log.info("[global %d/%d][cat %d/%d][pass2 %d/%d] START %s",
                     gidx, global_total, cidx, total_categories, p2idx, per_cat_pass2, combo_title)

            base_payload = build_basic_payload(
                category_key=cat_key,
                rarity_key=None,
                status_option="securable",
                sort_key="price",
                sort_dir="asc",
            )
            _add_corrupted(base_payload, is_corr)

            df_take = _paged_collect(
                base_payload=base_payload,
                base_currency=base_currency,
                league=league,
                realm=realm,
                recorder_log_level=recorder_log_level,
                category_key=seen_key,          # namespaced dedupe key
                seen_by_category=seen_by_category,
                per_combo=per_combo,
                page_step=100,
                max_pages=None,
            )

            combo_name = make_combo_filename(category_key=cat_key, rarity=None, corrupted=is_corr, league=league)
            combo_path = TRAINING_DIR / combo_name
            _save_parquet(df_take, combo_path)
            log.info("DONE combo -> %s (rows_saved=%d)", combo_path, len(df_take))

            if not df_take.empty:
                per_cat_parts.append(df_take)
                manifest_rows.append({
                    "phase": "cat×corrupted",
                    "category": cat_key,
                    "rarity": None,
                    "corrupted": is_corr,
                    "rows": len(df_take),
                    "file": str(combo_path),
                })

        # ---------------- OVERALL per exact category (progressive save) ----------------
        if per_cat_parts:
            df_cat = pd.concat(per_cat_parts, ignore_index=True)
            if "id" in df_cat.columns:
                df_cat = df_cat.drop_duplicates(subset=["id"])
            overall_name = make_overall_filename(category_key=cat_key, league=league)
            overall_path = OVERALL_DIR / overall_name
            _save_parquet(df_cat, overall_path)
            log.info("DONE category overall -> %s (rows_saved=%d)", overall_path, len(df_cat))
            manifest_rows.append({
                "phase": "overall",
                "category": cat_key,
                "rarity": None,
                "corrupted": None,
                "rows": len(df_cat),
                "file": str(overall_path),
            })
        else:
            log.info("No rows collected for category=%s; overall parquet skipped.", cat_key)

    # final manifest
    manifest = pd.DataFrame(manifest_rows)
    man_path = OVERALL_DIR / f"manifest_{_league_fs(league)}.parquet"
    if not manifest.empty:
        manifest.to_parquet(man_path, index=False)
        log.info("Manifest saved → %s (rows=%d)", man_path, len(manifest))
    else:
        log.warning("Manifest is empty — no files written.")
    return manifest


# =============================================================================
# main
# =============================================================================
def main() -> None:
    """
    Run a full Standard-league training snapshot in Exalted Orbs.
    """
    try:
        generate_training_data(
            per_combo=PER_COMBO_DEFAULT,
            base_currency=BASE_CURRENCY,   # Exalted Orb for the assignment
            league=LEAGUE,                 # Standard
            realm=None,                    # let recorder fall back to config for realm (poe2)
            recorder_log_level="DEBUG",    # show per-id GET lines from poe.trade2.fetch
            categories=None,               # or e.g. ["weapon.wand"] to limit scope
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Exiting gracefully.")


if __name__ == "__main__":
    main()