from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging
import re
import pandas as pd
import traceback

from src.recorder.poe_trade2_recorder import (
    search_to_dataframe_with_limit,
    build_basic_payload,
)
from src.recorder.constants import ITEM_TYPES, ITEM_RARITIES

from src.recorder.poe_currency_recorder import FXCache

TRAINING_DIR = Path("train/training_data")
DEBUGGING_DIR = Path("debug/training_data")
OVERALL_DIR = TRAINING_DIR / "overall"
OUTPUT_LOG = Path("logs/training.data.log")

LEAGUE = "Rise%20of%20the%20Abyssal"
BASE_CURRENCY = "Exalted Orb"
PER_COMBO_DEFAULT = 5000

_root = logging.getLogger()
if not _root.handlers:
    _root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    _root.addHandler(ch)
    try:
        OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(OUTPUT_LOG, encoding="utf-8")
        fh.setFormatter(fmt)
        _root.addHandler(fh)
    except Exception as e:
        print("Failed to set up file logging:", e)

log = logging.getLogger("poe.training.collector")
log.setLevel(logging.DEBUG)


# help name files
def _slug_u(s: str) -> str:
    clean = s.replace("%20", " ")
    return re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")


def _ensure_dirs() -> None:
    for p in [TRAINING_DIR, OVERALL_DIR]:
        try:
            p.mkdir(parents=True, exist_ok=True)
            log.debug("Ensured directory exists: %s", p)
        except Exception as e:
            log.error("Failed to create directory %s: %s", p, e)
            raise


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        if df.empty:
            log.warning("No rows to save -> %s (skipped)", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        log.info("Saved %d rows -> %s", len(df), path)
    except Exception as e:
        log.error("Failed to save parquet %s: %s", path, e)
        log.debug(traceback.format_exc())


# combine item x rarity files into one overall file for ease of loading collective data
def make_combo_filename(*, category_key: str, rarity: Optional[str],
                        league: str) -> str:
    cat = _slug_u(category_key.replace(".", "_"))
    rar = _slug_u(rarity) if rarity else "any"
    lg = _slug_u(league)
    return f"{cat}_rarity_{rar}_{lg}.parquet"


def make_overall_filename(*, category_key: str, league: str) -> str:
    cat = _slug_u(category_key.replace(".", "_"))
    lg = _slug_u(league)
    return f"{cat}_overall_{lg}.parquet"


# core
def _collect_for_combo(
    *,
    base_payload: Dict[str, Any],
    base_currency: str,
    league: str,
    realm: Optional[str],
    recorder_log_level: str,
    category_key: str,
    rarity_key: str,
    seen: Set[str],
    per_combo: int,
) -> pd.DataFrame:
    """
    Fetch up to `per_combo` securable rows for one (category x rarity).

    Logging clarifies:
      - Raw: number of rows returned by the API (no filtering yet).
      - New: how many of those rows are not already in `seen`.
      - Taking: how many new rows we actually keep (limited by per_combo).
      - dropped_seen: rows skipped because they were already seen.
      - dropped_cap: rows skipped because they exceeded per_combo.
      - dups_in_df: duplicates already inside this API response itself.
    """
    log.debug("Starting _collect_for_combo: %s x %s", category_key, rarity_key)

    try:
        # Ask for more than per_combo to increase chance of finding unseen rows
        df, info = search_to_dataframe_with_limit(
            payload=base_payload,
            base_currency=base_currency,
            max_results=per_combo * 2,
            league=league,
            realm=realm,
            log_level=recorder_log_level,
        )
    except Exception as e:
        log.error("API fetch failed for %s x %s: %s", category_key, rarity_key,
                  e)
        log.debug(traceback.format_exc())
        return pd.DataFrame()

    if df.empty or "id" not in df.columns:
        log.warning("No usable rows (empty or missing 'id') for %s x %s",
                    category_key, rarity_key)
        return pd.DataFrame()

    # keeping track of ids so far
    ids = df["id"].astype(str)
    total_raw = len(df)  # API rows before any filtering
    new_mask = ~ids.isin(seen)  # keep only not-yet-seen IDs
    new_total = int(new_mask.sum())  # how many are new
    take = min(per_combo, new_total)  # cap at per_combo
    dropped_seen = total_raw - new_total  # already-seen rows
    dropped_cap = new_total - take  # rows beyond cap
    dups_in_df = int(ids.duplicated().sum())  # duplicates inside this batch

    # keep the new, capped rows
    df_new = df.loc[new_mask].head(per_combo).copy()
    seen.update(df_new["id"].astype(str).tolist())

    log.info(
        "[%s x %s] Raw=%d; new=%d; taking=%d; dropped_seen=%d; dropped_cap=%d; dups_in_df=%d",
        category_key,
        rarity_key,
        total_raw,
        new_total,
        take,
        dropped_seen,
        dropped_cap,
        dups_in_df,
    )

    return df_new


def generate_training_data(
    *,
    per_combo: int = PER_COMBO_DEFAULT,
    base_currency: str = BASE_CURRENCY,
    league: str = LEAGUE,
    realm: Optional[str] = None,
    recorder_log_level: str = "DEBUG",
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    _ensure_dirs()

    cat_keys = list(ITEM_TYPES.keys()) if not categories else categories
    invalid = [c for c in cat_keys if c not in ITEM_TYPES]
    if invalid:
        log.error("Invalid category keys: %s", invalid)
        raise ValueError(f"Unknown category keys: {invalid}")

    seen_by_category: Dict[str, Set[str]] = {}
    manifest_rows: List[Dict[str, Any]] = []
    total_combos = len(cat_keys) * len(ITEM_RARITIES)

    log.info(
        "=== Starting training data collection ===\n"
        "  Total categories: %d\n"
        "  Total rarity combos: %d\n"
        "  Target per combo: %d\n"
        "  League: %s\n"
        "  Base currency: %s",
        len(cat_keys),
        len(ITEM_RARITIES),
        per_combo,
        league,
        base_currency,
    )

    gidx = 0
    for cat_key in cat_keys:  # repeat over every item category
        log.info("=== CATEGORY START: %s ===", cat_key)
        seen = seen_by_category.setdefault(cat_key, set())
        per_cat_parts: List[pd.DataFrame] = []

        for rar_key in ITEM_RARITIES.keys():  # repeat for every rarity category
            gidx += 1
            log.info("[%d/%d] Collecting %s x %s", gidx, total_combos, cat_key,
                     rar_key)

            try:
                payload = build_basic_payload(
                    category_key=cat_key,
                    rarity_key=rar_key,
                    status_option="securable",
                    sort_key="price",
                    sort_dir="asc",
                )
            except Exception as e:
                log.error("Failed to build payload for %s x %s: %s", cat_key,
                          rar_key, e)
                log.debug(traceback.format_exc())
                continue

            df_take = _collect_for_combo(
                base_payload=payload,
                base_currency=base_currency,
                league=league,
                realm=realm,
                recorder_log_level=recorder_log_level,
                category_key=cat_key,
                rarity_key=rar_key,
                seen=seen,
                per_combo=per_combo,
            )

            combo_name = make_combo_filename(category_key=cat_key,
                                             rarity=rar_key,
                                             league=league)
            # combo_path = TRAINING_DIR / combo_name
            combo_path = DEBUGGING_DIR / combo_name
            _save_parquet(df_take, combo_path)

            if not df_take.empty:
                per_cat_parts.append(df_take)
                manifest_rows.append({
                    "phase": "category_x_rarity",
                    "category": cat_key,
                    "rarity": rar_key,
                    "rows": len(df_take),
                    "file": str(combo_path),
                })
            else:
                log.warning("No new rows collected for %s x %s", cat_key,
                            rar_key)

        if per_cat_parts:
            try:
                df_cat = pd.concat(per_cat_parts, ignore_index=True)
                if "id" in df_cat.columns:
                    before = len(df_cat)
                    df_cat = df_cat.drop_duplicates(subset=["id"])
                    log.debug(
                        "Dropped %d duplicates in overall file for %s",
                        before - len(df_cat),
                        cat_key,
                    )
                overall_name = make_overall_filename(category_key=cat_key,
                                                     league=league)
                overall_path = OVERALL_DIR / overall_name
                _save_parquet(df_cat, overall_path)
                manifest_rows.append({
                    "phase": "overall",
                    "category": cat_key,
                    "rows": len(df_cat),
                    "file": str(overall_path),
                })
            except Exception as e:
                log.error("Failed to build overall file for %s: %s", cat_key, e)
                log.debug(traceback.format_exc())

        log.info("=== CATEGORY END: %s ===", cat_key)

    manifest = pd.DataFrame(manifest_rows)
    man_path = OVERALL_DIR / f"manifest_{_slug_u(league)}.parquet"
    try:
        if not manifest.empty:
            manifest.to_parquet(man_path, index=False)
            log.info("Manifest saved: %d entries -> %s", len(manifest),
                     man_path)
        else:
            log.warning("Manifest is empty — nothing collected")
    except Exception as e:
        log.error("Failed to save manifest: %s", e)
        log.debug(traceback.format_exc())

    return manifest


def main() -> None:
    try:
        # === FULL RUN ===
        # generate_training_data(
        #     per_combo=PER_COMBO_DEFAULT,
        #     base_currency=BASE_CURRENCY,
        #     league=LEAGUE,
        #     realm=None,
        #     recorder_log_level="DEBUG",
        #     categories=None,
        # )

        # === TEST RUN ===
        generate_training_data(
            per_combo=5,
            base_currency=BASE_CURRENCY,
            league=LEAGUE,
            realm=None,
            recorder_log_level="DEBUG",
            categories=["weapon.wand"],
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user. Exiting.")
    except Exception as e:
        log.error("Fatal error in main: %s", e)
        log.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
