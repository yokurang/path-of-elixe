"""
DEPRECATED
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable
import math
import re
import json

import pandas as pd
import numpy as np

from utils import find_project_root

# -----------------------------------------------------------------------------
# Regex & constants
# -----------------------------------------------------------------------------
RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*$")
PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
TIER_NUM_RE = re.compile(r"^\s*[SPsp]?(\d+)\s*$")  # "S10" / "P4" -> "10" / "4"

# Known hashes you may want explicitly (extend as needed)
IMPORTANT_MOD_HASHES = {
    "stat_3489782002": "max_energy_shield",
    "stat_803737631": "accuracy",
    "stat_2482852589": "increased_energy_shield_pct",
    "stat_3299347043": "max_life",
    "stat_3917489142": "item_rarity_pct",
    "stat_3261801346": "dexterity",
    "stat_4080418644": "strength",
    "stat_1050105434": "max_mana",
    "stat_587431675": "critical_chance_pct",
    "stat_3336890334": "lightning_damage",
    "stat_709508406": "fire_damage",
    "stat_3885405204": "skill_level",
    "stat_1028592286": "spirit",
    "stat_691932474": "accuracy_rating",
    "stat_1509134228": "physical_damage_pct",
    "stat_3695891184": "attack_speed_pct",
    "stat_821021828": "life_on_kill",
    "stat_1263695895": "light_radius",
    "stat_55876295": "life_leech",
    "stat_669069897": "mana_leech",
    "stat_2694482655": "increased_damage",
    "stat_3639275092": "reduced_requirements",
}

LEVEL_KEYWORDS = ["lightning", "fire", "cold", "chaos", "bow", "spell"]

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _to_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        return float(str(x).replace(",", "").strip())
    except (ValueError, TypeError):
        return np.nan

def _safe_parse_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (json.JSONDecodeError, ValueError):
            return x
    return x

def _normalize_text(s: Any) -> str:
    s = re.sub(r"[\[\]\|]", "", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _extract_property_value(values: Any) -> Optional[str]:
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return None

def _parse_numeric_avg(text: Any) -> float:
    if text is None:
        return 0.0
    s = str(text).strip()
    m = RANGE_RE.match(s)
    if m:
        lo = _to_float(m.group(1)); hi = _to_float(m.group(2))
        if math.isnan(lo) or math.isnan(hi):
            return 0.0
        return (lo + hi) / 2.0
    m = PERCENT_RE.search(s)
    if m:
        v = _to_float(m.group(1))
        return 0.0 if math.isnan(v) else v  # leave as "percent units"; caller can rescale
    m = NUM_RE.search(s)
    if m:
        v = _to_float(m.group(0))
        return 0.0 if math.isnan(v) else v
    return 0.0

def _avg_or_first_number(text: str) -> float:
    m = RANGE_RE.search(text)
    if m:
        lo = _to_float(m.group(1)); hi = _to_float(m.group(2))
        if not math.isnan(lo) and not math.isnan(hi):
            return (lo + hi) / 2.0
    nums = [ _to_float(n) for n in NUM_RE.findall(text) ]
    nums = [n for n in nums if not math.isnan(n)]
    return float(nums[0]) if nums else 0.0

def _coalesce_extended(raw_ext: Any) -> List[dict]:
    """
    Normalize 'extended' into a list[dict]. Handles dict, json string,
    list of json strings, or list of dicts.
    """
    ext = raw_ext
    if isinstance(ext, str):
        ext = _safe_parse_json(ext)
    if isinstance(ext, dict):
        return [ext]
    if isinstance(ext, list):
        out: List[dict] = []
        for item in ext:
            if isinstance(item, str):
                obj = _safe_parse_json(item)
                if isinstance(obj, dict):
                    out.append(obj)
            elif isinstance(item, dict):
                out.append(item)
        return out
    return []

# -----------------------------------------------------------------------------
# Extractors
# -----------------------------------------------------------------------------
def extract_base_attributes(row: pd.Series) -> Dict[str, float]:
    """
    NOTE: We intentionally avoid emitting boolean flags as features.
    Only continuous basics here.
    """
    a: Dict[str, float] = {}
    a["ilvl"] = _to_float(row.get("ilvl", 0))
    sockets = _safe_parse_json(row.get("sockets", []))
    a["socket_count"] = float(len(sockets)) if isinstance(sockets, list) else 0.0
    return a

def extract_requirements(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {"req_level": 0.0, "req_str": 0.0, "req_dex": 0.0, "req_int": 0.0}
    reqs = _safe_parse_json(row.get("requirements", row.get("weapon_requirements", [])))
    if isinstance(reqs, list):
        for req in reqs:
            if not isinstance(req, dict):
                continue
            name = _normalize_text(req.get("name", ""))
            val = _parse_numeric_avg(_extract_property_value(req.get("values", [])))
            if "level" in name:
                out["req_level"] = val
            elif name.endswith("strength") or name.endswith("str"):
                out["req_str"] = val
            elif name.endswith("dexterity") or name.endswith("dex"):
                out["req_dex"] = val
            elif name.endswith("intelligence") or name.endswith("int"):
                out["req_int"] = val
    return out

def extract_weapon_properties(row: pd.Series) -> Dict[str, float]:
    props: Dict[str, float] = {}
    arr = _safe_parse_json(row.get("properties", []))
    if isinstance(arr, list):
        for prop in arr:
            if not isinstance(prop, dict):
                continue
            name = _normalize_text(prop.get("name", ""))
            val = _parse_numeric_avg(_extract_property_value(prop.get("values", [])))
            if "physical damage" in name:
                props["phys_damage"] = val
            elif "chaos damage" in name:
                props["chaos_damage"] = val
            elif "cold damage" in name:
                props["cold_damage"] = val
            elif "fire damage" in name:
                props["fire_damage"] = val
            elif "lightning damage" in name:
                props["lightning_damage"] = val
            elif "critical" in name and "chance" in name:
                props["crit_chance"] = val
            elif "attacks per second" in name:
                props["attack_speed"] = val
            elif "quality" in name:
                props["quality"] = val
            elif "armor" in name or "armour" in name:
                props["armor"] = val
            elif "evasion" in name:
                props["evasion"] = val
            elif "energy shield" in name:
                props["energy_shield"] = val
    return props

# -----------------------------------------------------------------------------
# Mods (implicit & explicit treated as attributes)
# -----------------------------------------------------------------------------
def extract_mod_attributes(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {
        "implicit_mod_count": 0.0,
        "explicit_mod_count": 0.0,
        # recognized aggregates
        "mod_add_fire_avg": 0.0,
        "mod_add_cold_avg": 0.0,
        "mod_add_lightning_avg": 0.0,
        "mod_add_chaos_avg": 0.0,
        "mod_inc_attack_speed_pct": 0.0,  # fraction (0.12 = 12%)
        "mod_inc_crit_chance_pct": 0.0,
        "mod_inc_phys_pct": 0.0,
        "mod_accuracy": 0.0,
        # “others” buckets
        "other_mods_avg_value": 0.0,
        "other_mods_avg_level": 0.0,
        "other_mods_count": 0.0,
        # highest explicit peaks
        "highest_explicit_tier_num": 0.0,
        "highest_explicit_level": 0.0,
    }
    for kw in LEVEL_KEYWORDS:
        out[f"mod_plus_levels_{kw}"] = 0.0

    other_vals: List[float] = []
    other_lvls: List[float] = []

    ext_list = _coalesce_extended(row.get("extended", {}))
    all_implicit: List[dict] = []
    all_explicit: List[dict] = []

    for ext in ext_list:
        mods = ext.get("mods", {})
        if isinstance(mods, dict):
            impl = mods.get("implicit", []) or []
            expl = mods.get("explicit", []) or []
            if isinstance(impl, list):
                all_implicit.extend([m for m in impl if isinstance(m, dict)])
            if isinstance(expl, list):
                all_explicit.extend([m for m in expl if isinstance(m, dict)])

    out["implicit_mod_count"] = float(len(all_implicit))
    out["explicit_mod_count"] = float(len(all_explicit))

    # explicit tier/level peaks
    max_tier_num = 0.0
    max_level = 0.0
    for m in all_explicit:
        tier = str(m.get("tier", "") or "")
        mlevel = _to_float(m.get("level", 0)) or 0.0
        mtm = TIER_NUM_RE.match(tier)
        if mtm:
            tnum = _to_float(mtm.group(1))
            if not math.isnan(tnum):
                max_tier_num = max(max_tier_num, float(tnum))
        max_level = max(max_level, float(mlevel))
    out["highest_explicit_tier_num"] = float(max_tier_num)
    out["highest_explicit_level"] = float(max_level)

    def consume_mod_line(mod: dict):
        nonlocal other_vals, other_lvls
        name = _normalize_text(mod.get("name", ""))
        level = _to_float(mod.get("level", 1)) or 1.0
        mags = mod.get("magnitudes", [])
        used_structured = False

        if isinstance(mags, list) and mags:
            for mag in mags:
                if not isinstance(mag, dict):
                    continue
                h = str(mag.get("hash", "")).replace("implicit.", "").replace("explicit.", "")
                lo = _to_float(mag.get("min", 0)); hi = _to_float(mag.get("max", 0))
                v = (lo + hi) / 2.0 if (not math.isnan(lo) and not math.isnan(hi)) else 0.0

                if h in IMPORTANT_MOD_HASHES:
                    k = IMPORTANT_MOD_HASHES[h]
                    out[k] = out.get(k, 0.0) + v
                    used_structured = True
                else:
                    if v != 0.0:
                        other_vals.append(v)
                        other_lvls.append(level)
                        used_structured = True

        if not used_structured and name:
            # adds X-Y elemental
            if "adds" in name and "damage" in name:
                if "fire" in name:
                    out["mod_add_fire_avg"] += _avg_or_first_number(name); return
                if "cold" in name:
                    out["mod_add_cold_avg"] += _avg_or_first_number(name); return
                if "lightning" in name:
                    out["mod_add_lightning_avg"] += _avg_or_first_number(name); return
                if "chaos" in name:
                    out["mod_add_chaos_avg"] += _avg_or_first_number(name); return

            # % increases
            if "attack speed" in name and "%" in name:
                m = PERCENT_RE.search(name)
                if m: out["mod_inc_attack_speed_pct"] += (_to_float(m.group(1)) / 100.0); return
            if ("critical chance" in name or "critical strike chance" in name) and "%" in name:
                m = PERCENT_RE.search(name)
                if m: out["mod_inc_crit_chance_pct"] += (_to_float(m.group(1)) / 100.0); return
            if "physical" in name and "damage" in name and "% increased" in name:
                m = PERCENT_RE.search(name)
                if m: out["mod_inc_phys_pct"] += (_to_float(m.group(1)) / 100.0); return

            # accuracy
            if "accuracy" in name:
                out["mod_accuracy"] += _avg_or_first_number(name); return

            # +X to level of ...
            if "to level" in name:
                lvl_val = _avg_or_first_number(name)
                for kw in LEVEL_KEYWORDS:
                    if kw in name:
                        out[f"mod_plus_levels_{kw}"] += lvl_val
                        return

            # fallback → others
            v = _avg_or_first_number(name)
            if v != 0.0:
                other_vals.append(v); other_lvls.append(level)

    for m in all_implicit:
        consume_mod_line(m)
    for m in all_explicit:
        consume_mod_line(m)

    if other_vals:
        out["other_mods_avg_value"] = float(np.mean(other_vals))
        out["other_mods_avg_level"] = float(np.mean(other_lvls)) if other_lvls else 0.0
        out["other_mods_count"] = float(len(other_vals))

    return out

# -----------------------------------------------------------------------------
# Per-channel DPS only (no totals to avoid multicollinearity)
# -----------------------------------------------------------------------------
def derive_channel_dps(feats: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    aps = feats.get("attack_speed", 0.0) or 0.0

    def d(name: str) -> float:
        return float(feats.get(name, 0.0) or 0.0)

    if aps > 0:
        out["pdps"] = aps * d("phys_damage")
        out["fdps"] = aps * d("fire_damage")
        out["cdps"] = aps * d("cold_damage")
        out["ldps"] = aps * d("lightning_damage")
        out["chaos_dps"] = aps * d("chaos_damage")
    else:
        out["pdps"] = out["fdps"] = out["cdps"] = out["ldps"] = out["chaos_dps"] = 0.0
    return out

# -----------------------------------------------------------------------------
# Main feature engineering
# -----------------------------------------------------------------------------
def make_item_features(
    df: pd.DataFrame,
    compute_dps: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    rows: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        f: Dict[str, float] = {}

        # base & requirements (full vector)
        f.update(extract_base_attributes(row))
        f.update(extract_requirements(row))

        # properties (avg damages, aps, etc.)
        f.update(extract_weapon_properties(row))

        # mods (implicit+explicit attributes, plus explicit tier/level peaks)
        f.update(extract_mod_attributes(row))

        # per-channel dps only; then drop aps & raw channels to avoid perfect collinearity
        if compute_dps:
            dps = derive_channel_dps(f)
            f.update(dps)
            for k in ("attack_speed", "phys_damage", "fire_damage", "cold_damage", "lightning_damage", "chaos_damage"):
                f.pop(k, None)

        rows.append(f)

    features = pd.DataFrame(rows, index=df.index)

    # ---- Optionality (continuous replacement for boolean constraints) ----
    # helper booleans from raw df (NOT added to the feature set)
    corrupted_s = df["corrupted"].astype(bool) if "corrupted" in df.columns else pd.Series(False, index=df.index)
    unmod_s = df["unmodifiable"].astype(bool) if "unmodifiable" in df.columns else pd.Series(False, index=df.index)
    craftable = (~corrupted_s) & (~unmod_s)  # helper only

    # per-category cap95 of explicit_mod_count
    meta_cat = df["category"].astype(str) if "category" in df.columns else pd.Series("unknown", index=df.index)
    if "explicit_mod_count" not in features.columns:
        features["explicit_mod_count"] = 0.0  # safety

    cap95_by_cat = (
        features.groupby(meta_cat)["explicit_mod_count"]
        .quantile(0.95)
        .reindex(meta_cat.unique())
    )
    cap_map = meta_cat.map(cap95_by_cat.to_dict()).fillna(0.0)

    open_slots_est = craftable.astype(float) * np.maximum(0.0, cap_map.values - features["explicit_mod_count"].values)
    features["open_slots_est"] = open_slots_est

    # targets
    targets = pd.DataFrame(index=df.index)
    targets["price"] = pd.to_numeric(df.get("price_amount_in_base", 0), errors="coerce").fillna(0.0)
    targets["log_price"] = np.log1p(targets["price"].clip(lower=0.0))

    # metadata
    metadata = pd.DataFrame(index=df.index)
    for col in ["id", "league", "indexed", "rarity", "category", "base_type", "type_line", "name"]:
        if col in df.columns:
            metadata[col] = df[col].astype(str)

    # numeric cleanup
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0.0, inplace=True)

    # ensure no total DPS columns sneak in from upstream
    totals_to_drop = ["dps", "edps", "calc_edps", "calc_total_dps", "total_dps"]
    features.drop(columns=[c for c in totals_to_drop if c in features.columns], inplace=True, errors="ignore")

    # remove zero-variance columns
    var = features.var()
    zero_var = var[var <= 1e-12].index
    if len(zero_var) > 0:
        features.drop(columns=list(zero_var), inplace=True, errors="ignore")

    return features, targets, metadata

# -----------------------------------------------------------------------------
# Category split helper (bottom-up per category; uniques as their own category)
# -----------------------------------------------------------------------------
def create_category_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    models: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    if "category" not in metadata.columns:
        return models

    cats = metadata["category"].astype(str)
    is_unique = (metadata.get("rarity", "").astype(str).str.lower() == "unique")

    for cat in sorted(cats.dropna().unique()):
        mask = (cats == cat)
        if (mask & ~is_unique).any():
            models[cat] = (features[mask & ~is_unique].copy(), targets[mask & ~is_unique].copy())
        if (mask & is_unique).any():
            names = metadata.loc[mask & is_unique, "type_line"].astype(str).dropna().unique()
            for nm in names:
                sel = mask & is_unique & (metadata["type_line"].astype(str) == nm)
                if sel.sum() >= 5:
                    models[f"unique_{nm}"] = (features[sel].copy(), targets[sel].copy())
    return models

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_and_process_item_data(
    file_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
    **feature_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        root = Path(base_path).expanduser().resolve() if base_path else find_project_root()
        path = (root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception:
        try:
            df = pd.read_parquet(path, engine="fastparquet")
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet: {e}")

    return make_item_features(df, **feature_kwargs)

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_features(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
    print(f"Price range: {targets['price'].min():.2f} - {targets['price'].max():.2f}")
    print(f"Mean price: {targets['price'].mean():.2f}, Median: {targets['price'].median():.2f}")

    key_feats = [
        "pdps", "fdps", "cdps", "ldps", "chaos_dps",
        "open_slots_est",
        "highest_explicit_tier_num", "highest_explicit_level",
        "req_level", "req_str", "req_dex", "req_int",
        "implicit_mod_count", "explicit_mod_count",
        "other_mods_avg_value", "other_mods_avg_level",
    ]
    print("\nkey features present:")
    for k in key_feats:
        if k in features.columns:
            nz = int((features[k] != 0).sum())
            mean_val = float(features.loc[features[k] != 0, k].mean()) if nz else 0.0
            print(f"  {k}: {nz} nonzero, avg={mean_val:.3f}")

    # sanity: totals should be gone
    forbidden = {"dps", "edps", "calc_edps", "calc_total_dps", "total_dps"}
    present_forbidden = sorted([c for c in forbidden if c in features.columns])
    if present_forbidden:
        print(f"\nWARNING: drop totals to avoid multicollinearity: {present_forbidden}")

# Example
if __name__ == "__main__":
    X, y, meta = load_and_process_item_data(
        "training_data/overall/weapon_bow_overall_standard.parquet",
        compute_dps=True,
    )
    validate_features(X, y)
    buckets = create_category_models(X, y, meta)
    print(f"\ncreated {len(buckets)} category models:")
    for name, (fx, ty) in buckets.items():
        print(f"  {name}: {len(fx)} rows")


def split_by_rarity(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    copy: bool = True,
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split into craftables (Normal/Magic/Rare) vs Unique.
    Returns: (craft_X, craft_y, craft_meta), (uniq_X, uniq_y, uniq_meta)
    """
    if "rarity" in metadata.columns:
        rarity = metadata["rarity"].astype(str).str.lower()
        is_unique = rarity.eq("unique")
    else:
        # if no rarity column, treat everything as craftable
        is_unique = pd.Series(False, index=metadata.index)

    is_craft = ~is_unique

    def pick(mask: pd.Series):
        if copy:
            return (features.loc[mask].copy(),
                    targets.loc[mask].copy(),
                    metadata.loc[mask].copy())
        return (features.loc[mask], targets.loc[mask], metadata.loc[mask])

    return pick(is_craft), pick(is_unique)


def load_and_process_item_data_split(
    file_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
    **feature_kwargs
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Convenience wrapper: run feature engineering, then split by rarity.
    Returns: (craft_X, craft_y, craft_meta), (uniq_X, uniq_y, uniq_meta)
    """
    X, y, meta = load_and_process_item_data(
        file_path, base_path=base_path, **feature_kwargs
    )
    return split_by_rarity(X, y, meta)
