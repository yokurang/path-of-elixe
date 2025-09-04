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
BASE_CURRENCY = "Exalted Orb"     # assignment requires Exalted Orb as base
PER_COMBO_DEFAULT = 100            # ≤50 rows saved per combo (recorder returns up to 100 per search)

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
    rarity: Optional[str],         # None -> "any"
    corrupted: Optional[bool],     # None -> "any"
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

        # ---------------- PASS 1: category × rarity (progressive save) ----------------
        for ridx, rar_key in enumerate(ITEM_RARITIES.keys(), start=1):
            gidx += 1
            combo_title = f"cat={cat_key} | rarity={rar_key} | corrupted=any | league={league} | base={base_currency}"
            log.info("[global %d/%d][cat %d/%d][pass1 %d/%d] START %s",
                     gidx, global_total, cidx, total_categories, ridx, per_cat_pass1, combo_title)

            payload = build_basic_payload(
                category_key=cat_key,
                rarity_key=rar_key,
                status_option="any",
                sort_key="price",
                sort_dir="asc",
            )
            try:
                df_all = search_to_dataframe(
                    payload=payload,
                    base_currency=base_currency,
                    league=league,
                    realm=realm,
                    save_csv=None,
                    log_level=recorder_log_level,
                )
            except Exception as e:
                log.warning("Search failed: %s (%s)", combo_title, e)
                continue

            # Combo diagnostics
            seen = seen_by_category.setdefault(cat_key, set())
            ids = df_all["id"].astype(str) if ("id" in df_all.columns and not df_all.empty) else pd.Series([], dtype=str)
            new_mask = ~ids.isin(seen)
            new_available = int(new_mask.sum())
            dupes = int((~new_mask).sum())
            take_n = min(per_combo, new_available)
            log.info("Combo results: total=%d, new_available=%d, duplicates=%d, taking=%d",
                     len(df_all), new_available, dupes, take_n)

            df_take = _select_unique_for_category(df_all, cat_key, seen_by_category, per_combo)
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

        # ---------------- PASS 2: category × corrupted (progressive save) ----------------
        for p2idx, is_corr in enumerate((True, False), start=1):
            gidx += 1
            corr_label = "corrupted" if is_corr else "not_corrupted"
            combo_title = f"cat={cat_key} | rarity=any | corrupted={corr_label} | league={league} | base={base_currency}"
            log.info("[global %d/%d][cat %d/%d][pass2 %d/%d] START %s",
                     gidx, global_total, cidx, total_categories, p2idx, per_cat_pass2, combo_title)

            payload = build_basic_payload(
                category_key=cat_key,
                rarity_key=None,
                status_option="any",
                sort_key="price",
                sort_dir="asc",
            )
            _add_corrupted(payload, is_corr)

            try:
                df_all = search_to_dataframe(
                    payload=payload,
                    base_currency=base_currency,
                    league=league,
                    realm=realm,
                    save_csv=None,
                    log_level=recorder_log_level,
                )
            except Exception as e:
                log.warning("Search failed: %s (%s)", combo_title, e)
                continue

            # Combo diagnostics
            seen = seen_by_category.setdefault(cat_key, set())
            ids = df_all["id"].astype(str) if ("id" in df_all.columns and not df_all.empty) else pd.Series([], dtype=str)
            new_mask = ~ids.isin(seen)
            new_available = int(new_mask.sum())
            dupes = int((~new_mask).sum())
            take_n = min(per_combo, new_available)
            log.info("Combo results: total=%d, new_available=%d, duplicates=%d, taking=%d",
                     len(df_all), new_available, dupes, take_n)

            df_take = _select_unique_for_category(df_all, cat_key, seen_by_category, per_combo)
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
    (Recorder uses MAX_RESULTS=100 per search; we save ≤50 unique per combo.)
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
