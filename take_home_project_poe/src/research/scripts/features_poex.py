# features_poex.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable, Callable
import math
import re
import json

import pandas as pd
import numpy as np

from .utils import find_project_root

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

LEVEL_KEYWORDS = ["lightning", "fire", "cold", "chaos", "spell"]

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
        return 0.0 if math.isnan(v) else v
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
    nums = [_to_float(n) for n in NUM_RE.findall(text)]
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

def _to_bool01(v: Any) -> float:
    """Robust boolean → {0.0, 1.0}."""
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and not math.isnan(v):
        return 1.0 if v != 0 else 0.0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return 1.0
        if s in {"false", "f", "no", "n", "0"}:
            return 0.0
    return 0.0

# -----------------------------------------------------------------------------
# Parsers: base / sockets / requirements / properties / mods / flags
# -----------------------------------------------------------------------------
def parse_base_attributes(row: pd.Series) -> Dict[str, float]:
    a: Dict[str, float] = {}
    a["ilvl"] = _to_float(row.get("ilvl", 0))
    return a

def parse_sockets(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    sockets = _safe_parse_json(row.get("sockets", []))
    if isinstance(sockets, list):
        out["socket_count"] = float(len(sockets))
        # try:
        #     out["rune_group_count"] = float(len({d.get("group") for d in sockets if isinstance(d, dict) and "group" in d}))
        # except Exception:
        #     out["rune_group_count"] = 0.0
    else:
        out["socket_count"] = 0.0
        # out["rune_group_count"] = 0.0
    return out

def parse_requirements(row: pd.Series) -> Dict[str, float]:
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

def _is_damage_prop(name: str, elem: str) -> bool:
    name = _normalize_text(name)
    return (f"{elem} damage" in name)

def parse_weapon_properties(row: pd.Series) -> Dict[str, float]:
    """
    Parse property blocks into unified numeric channels.
    Supports attack and cast weapons; we compute a single 'rate_per_s'.
    """
    props: Dict[str, float] = {
        "phys_damage": 0.0,
        "chaos_damage": 0.0,
        "cold_damage": 0.0,
        "fire_damage": 0.0,
        "lightning_damage": 0.0,
        "crit_chance": 0.0,
        "quality": 0.0,
        "rate_per_s": 0.0,  # attacks/casts per second (unified)
    }
    arr = _safe_parse_json(row.get("properties", []))
    if not isinstance(arr, list):
        return props

    for prop in arr:
        if not isinstance(prop, dict):
            continue
        name = _normalize_text(prop.get("name", ""))
        val = _parse_numeric_avg(_extract_property_value(prop.get("values", [])))

        if _is_damage_prop(name, "physical"):
            props["phys_damage"] = val
        elif _is_damage_prop(name, "chaos"):
            props["chaos_damage"] = val
        elif _is_damage_prop(name, "cold"):
            props["cold_damage"] = val
        elif _is_damage_prop(name, "fire"):
            props["fire_damage"] = val
        elif _is_damage_prop(name, "lightning"):
            props["lightning_damage"] = val
        elif "attacks per second" in name or "casts per second" in name:
            props["rate_per_s"] = val
        elif "critical" in name and "chance" in name:
            props["crit_chance"] = val
        elif "quality" in name:
            props["quality"] = val
        elif "armor" in name or "armour" in name:
            props["armor"] = val
        elif "evasion" in name:
            props["evasion"] = val
        elif "energy shield" in name:
            props["energy_shield"] = val

    props["attack_speed"] = props["rate_per_s"]
    return props

def parse_mod_attributes(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {
        "implicit_mod_count": 0.0,
        "explicit_mod_count": 0.0,
        "mod_add_fire_avg": 0.0,
        "mod_add_cold_avg": 0.0,
        "mod_add_lightning_avg": 0.0,
        "mod_add_chaos_avg": 0.0,
        "mod_inc_attack_speed_pct": 0.0,
        "mod_inc_crit_chance_pct": 0.0,
        "mod_inc_phys_pct": 0.0,
        "mod_accuracy": 0.0,
        "other_mods_avg_value": 0.0,
        "other_mods_avg_level": 0.0,
        "other_mods_count": 0.0,
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
            if "adds" in name and "damage" in name:
                if "fire" in name:
                    out["mod_add_fire_avg"] += _avg_or_first_number(name); return
                if "cold" in name:
                    out["mod_add_cold_avg"] += _avg_or_first_number(name); return
                if "lightning" in name:
                    out["mod_add_lightning_avg"] += _avg_or_first_number(name); return
                if "chaos" in name:
                    out["mod_add_chaos_avg"] += _avg_or_first_number(name); return

            if "attack speed" in name and "%" in name:
                m = PERCENT_RE.search(name)
                if m: out["mod_inc_attack_speed_pct"] += (_to_float(m.group(1)) / 100.0); return
            if ("critical chance" in name or "critical strike chance" in name) and "%" in name:
                m = PERCENT_RE.search(name)
                if m: out["mod_inc_crit_chance_pct"] += (_to_float(m.group(1)) / 100.0); return
            if "physical" in name and "damage" in name and "% increased" in name:
                m = PERCENT_RE.search(name)
                if m: out["mod_inc_phys_pct"] += (_to_float(m.group(1)) / 100.0); return

            if "accuracy" in name:
                out["mod_accuracy"] += _avg_or_first_number(name); return

            if "to level" in name:
                lvl_val = _avg_or_first_number(name)
                for kw in LEVEL_KEYWORDS:
                    if kw in name:
                        out[f"mod_plus_levels_{kw}"] += lvl_val
                        return

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

def parse_boolean_flags(row: pd.Series) -> Dict[str, float]:
    """
    One-hot (0/1) boolean flags as numeric features.
    """
    return {
        "is_corrupted": _to_bool01(row.get("corrupted", False)),
        "is_unmodifiable": _to_bool01(row.get("unmodifiable", False)),
        "is_duplicated": _to_bool01(row.get("duplicated", False)),
    }

# -----------------------------------------------------------------------------
# Derived: DPS channels (always computed)
# -----------------------------------------------------------------------------
def derive_channel_dps(feats: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    rps = float(feats.get("rate_per_s", feats.get("attack_speed", 0.0)) or 0.0)

    def d(name: str) -> float:
        return float(feats.get(name, 0.0) or 0.0)

    if rps > 0:
        out["pdps"] = rps * d("phys_damage")
        out["fdps"] = rps * d("fire_damage")
        out["cdps"] = rps * d("cold_damage")
        out["ldps"] = rps * d("lightning_damage")
        out["chaos_dps"] = rps * d("chaos_damage")
    else:
        out["pdps"] = out["fdps"] = out["cdps"] = out["ldps"] = out["chaos_dps"] = 0.0
    return out

# -----------------------------------------------------------------------------
# Optional augmentation (now: cleanup only; no feature creation)
# -----------------------------------------------------------------------------
def augment_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # proactively drop disallowed/legacy engineered columns if present
    DROP_COLS = [
        "log1p_pdps",
        "log1p_chaos_dps",
        "reqdex_x_pdps",
        "log1p_open_slots_est",
        "log1p_req_level",
        "log1p_req_dex",
        "log1p_highest_explicit_tier_num",
        "log1p_highest_explicit_level",
        "log1p_implicit_mod_count",
        "log1p_explicit_mod_count",
        "log1p_other_mods_avg_value",
        "log1p_other_mods_count",
        "log1p_crit_chance",
        "log1p_quality",
        "log1p_ilvl",
        "pdps_hinge_1",
        "pdps_hinge_2",
        "pdps_hinge_3",
        "pdps_x_crit",
        "chaos_x_crit",
        "pdps_x_open",
        "chaos_x_open",
    ]
    X.drop(columns=[c for c in DROP_COLS if c in X.columns], inplace=True, errors="ignore")

    # clean infinities/nans
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)

    # prune zero-variance columns
    var = X.var()
    zero_var = var[var <= 1e-12].index
    if len(zero_var) > 0:
        X.drop(columns=list(zero_var), inplace=True, errors="ignore")

    return X

# -----------------------------------------------------------------------------
# Feature engineering (base + DPS always)
# -----------------------------------------------------------------------------
def make_item_features(
    df: pd.DataFrame,
    *,
    custom_hooks: Optional[List[Callable[[pd.Series, Dict[str, float]], None]]] = None,
    augment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    rows: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        f: Dict[str, float] = {}

        f.update(parse_base_attributes(row))
        f.update(parse_sockets(row))
        f.update(parse_requirements(row))
        f.update(parse_weapon_properties(row))
        f.update(parse_mod_attributes(row))
        f.update(parse_boolean_flags(row))

        if custom_hooks:
            for hook in custom_hooks:
                try:
                    hook(row, f)
                except Exception:
                    pass

        dps = derive_channel_dps(f)
        f.update(dps)

        for k in ("attack_speed", "rate_per_s",
                  "phys_damage", "fire_damage", "cold_damage", "lightning_damage", "chaos_damage"):
            f.pop(k, None)

        rows.append(f)

    features = pd.DataFrame(rows, index=df.index)

    # ---- Optionality proxy using raw booleans on df (robust to missing cols) ----
    corrupted_s = df["corrupted"].astype(bool) if "corrupted" in df.columns else pd.Series(False, index=df.index)
    unmod_s = df["unmodifiable"].astype(bool) if "unmodifiable" in df.columns else pd.Series(False, index=df.index)
    craftable = (~corrupted_s) & (~unmod_s)

    meta_cat = df["category"].astype(str) if "category" in df.columns else pd.Series("unknown", index=df.index)
    if "explicit_mod_count" not in features.columns:
        features["explicit_mod_count"] = 0.0

    cap95_by_cat = (
        features.groupby(meta_cat)["explicit_mod_count"]
        .quantile(0.95)
        .reindex(meta_cat.unique())
    )
    cap_map = meta_cat.map(cap95_by_cat.to_dict()).fillna(0.0)

    open_slots_est = craftable.astype(float) * np.maximum(0.0, cap_map.values - features["explicit_mod_count"].values)
    features["open_slots_est"] = open_slots_est

    # Targets
    targets = pd.DataFrame(index=df.index)
    targets["price"] = pd.to_numeric(df.get("price_amount_in_base", 0), errors="coerce").fillna(0.0)
    targets["log_price"] = np.log1p(targets["price"].clip(lower=0.0))

    # Metadata - THIS IS WHERE NAME MAPPING SHOULD GO
    metadata = pd.DataFrame(index=df.index)
    for col in ["id", "league", "indexed", "rarity", "category", "base_type", "type_line", "name"]:
        if col in df.columns:
            metadata[col] = df[col].astype(str)

    # Add unique item name mapping to metadata, not features
    base_type_to_name = {
        'Attuned Wand': 'Lifesprig',
        'Bone Wand': 'Sanguine Diviner',
        'Acrid Wand': 'Cursecarver',
        'Volatile Wand': "Enezun's Charge",
        'Withered Wand': 'The Wicked Quill'
    }
    
    # Create unique_name column in metadata
    if 'base_type' in df.columns:
        metadata['unique_name'] = df['base_type'].apply(
            lambda x: base_type_to_name.get(x, x if x != '' else 'Unknown')
        )
    else:
        metadata['unique_name'] = 'Unknown'

    # Clean features DataFrame (should only contain numeric data)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0.0, inplace=True)

    # Drop any columns that might have slipped in
    totals_to_drop = ["dps", "edps", "calc_edps", "calc_total_dps", "total_dps", "pdps_aug"]
    features.drop(columns=[c for c in totals_to_drop if c in features.columns], inplace=True, errors="ignore")

    # Only compute variance on numeric columns
    numeric_features = features.select_dtypes(include=[np.number])
    var = numeric_features.var()
    zero_var = var[var <= 1e-12].index
    if len(zero_var) > 0:
        features.drop(columns=list(zero_var), inplace=True, errors="ignore")

    if augment:
        features = augment_features(features)

    return features, targets, metadata


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
        "is_corrupted", "is_unmodifiable", "is_duplicated",
    ]
    print("\nkey features present:")
    for k in key_feats:
        if k in features.columns:
            nz = int((features[k] != 0).sum())
            mean_val = float(features.loc[features[k] != 0, k].mean()) if nz else 0.0
            print(f"  {k}: {nz} nonzero, avg={mean_val:.3f}")

    forbidden = {"dps", "edps", "calc_edps", "calc_total_dps", "total_dps"}
    present_forbidden = sorted([c for c in forbidden if c in features.columns])
    if present_forbidden:
        print(f"\nWARNING: drop totals to avoid multicollinearity: {present_forbidden}")

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def load_raw_table(file_path: str | Path, *, base_path: Optional[str | Path] = None) -> pd.DataFrame:
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
    return df

# -----------------------------------------------------------------------------
# Splitters & bucket builders
# -----------------------------------------------------------------------------
def split_by_rarity(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    copy: bool = True,
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    if "rarity" in metadata.columns:
        rarity = metadata["rarity"].astype(str).str.lower()
        is_unique = rarity.eq("unique")
    else:
        is_unique = pd.Series(False, index=metadata.index)
    is_craft = ~is_unique

    def pick(mask: pd.Series):
        if copy:
            return (features.loc[mask].copy(),
                    targets.loc[mask].copy(),
                    metadata.loc[mask].copy())
        return (features.loc[mask], targets.loc[mask], metadata.loc[mask])

    return pick(is_craft), pick(is_unique)

def create_craftable_category_buckets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    if "category" not in metadata.columns:
        return out

    rarity = metadata.get("rarity", "").astype(str).str.lower()
    cats = metadata["category"].astype(str).fillna("unknown")

    for cat in sorted(cats.unique()):
        mask = (cats == cat) & (rarity != "unique")
        if mask.any():
            out[cat] = (
                features.loc[mask].copy(),
                targets.loc[mask].copy(),
                metadata.loc[mask].copy(),
            )
    return out

def create_unique_item_buckets(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    if "rarity" not in metadata.columns or "type_line" not in metadata.columns:
        return out

    rarity = metadata["rarity"].astype(str).str.lower()
    type_line = metadata["type_line"].astype(str)

    mask_u = rarity.eq("unique")
    for nm in sorted(type_line[mask_u].dropna().unique()):
        sel = mask_u & (type_line == nm)
        if sel.any():
            out[f"unique_{nm}"] = (
                features.loc[sel].copy(),
                targets.loc[sel].copy(),
                metadata.loc[sel].copy(),
            )
    return out

def create_unique_item_buckets_by_name(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create buckets for unique items grouped by their unique name.
    Uses the 'unique_name' column from metadata for grouping.
    """
    out: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    
    if "rarity" not in metadata.columns or "unique_name" not in metadata.columns:
        return out

    rarity = metadata["rarity"].astype(str).str.lower()
    unique_name = metadata["unique_name"].astype(str)

    # Only process unique items
    mask_unique = rarity.eq("unique")
    
    for name in sorted(unique_name[mask_unique].dropna().unique()):
        if name == 'Unknown':
            continue  # Skip unknown items
            
        sel = mask_unique & (unique_name == name)
        if sel.any():
            out[name] = (
                features.loc[sel].copy(),
                targets.loc[sel].copy(),
                metadata.loc[sel].copy(),
            )
    return out

# -----------------------------------------------------------------------------
# Simple one-call interface
# -----------------------------------------------------------------------------
def prepare_feature_sets_from_raw(
    file_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
    custom_hooks: Optional[List[Callable[[pd.Series, Dict[str, float]], None]]] = None,
    augment: bool = True,
    group_uniques_by_name: bool = False,
) -> Dict[str, Any]:
    """
    Returns:
        {
            "all": (X, y, meta),
            "craft_all": (craft_X, craft_y, craft_meta),
            "unique_all": (uniq_X, uniq_y, uniq_meta),
            "craft_by_category": {cat: (Xc, yc, mc), ...},
            "unique_by_item": {"unique_<type_line>": (Xu, yu, mu), ...},
            "unique_by_name": {name: (Xu, yu, mu), ...},  # New option
        }
    """
    df = load_raw_table(file_path, base_path=base_path)
    X, y, meta = make_item_features(df, custom_hooks=custom_hooks, augment=augment)

    (craft_X, craft_y, craft_meta), (uniq_X, uniq_y, uniq_meta) = split_by_rarity(X, y, meta)

    craft_buckets = create_craftable_category_buckets(craft_X, craft_y, craft_meta)
    unique_buckets = create_unique_item_buckets(uniq_X, uniq_y, uniq_meta)
    
    result = {
        "all": (X, y, meta),
        "craft_all": (craft_X, craft_y, craft_meta),
        "unique_all": (uniq_X, uniq_y, uniq_meta),
        "craft_by_category": craft_buckets,
        "unique_by_item": unique_buckets,
    }
    
    # Add unique grouping by name if requested
    if group_uniques_by_name:
        unique_by_name = create_unique_item_buckets_by_name(uniq_X, uniq_y, uniq_meta)
        result["unique_by_name"] = unique_by_name
    
    return result

def prepare_unique_item_features_by_name(df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Alternative approach: Group by unique names BEFORE feature engineering.
    This ensures each unique item type gets its own feature set.
    """
    # First create the name mapping
    base_type_to_name = {
        'Attuned Wand': 'Lifesprig',
        'Bone Wand': 'Sanguine Diviner',
        'Acrid Wand': 'Cursecarver',
        'Volatile Wand': "Enezun's Charge",
        'Withered Wand': 'The Wicked Quill'
    }
    
    # Add unique_name column for grouping
    if 'base_type' in df.columns:
        df['unique_name'] = df['base_type'].apply(
            lambda x: base_type_to_name.get(x, x if x != '' else 'Unknown')
        )
    else:
        df['unique_name'] = 'Unknown'
    
    # Filter to only unique rarity items
    if 'rarity' in df.columns:
        unique_df = df[df['rarity'].str.lower() == 'unique'].copy()
    else:
        unique_df = df.copy()
    
    # Group by unique_name and create feature sets for each
    unique_item_buckets = {}
    
    for name, group_df in unique_df.groupby('unique_name'):
        if name == 'Unknown':
            continue
            
        try:
            # Create features for this specific unique item
            features, targets, metadata = make_item_features(group_df, augment=True)
            unique_item_buckets[name] = (features, targets, metadata)
        except Exception as e:
            print(f"Warning: Failed to create features for {name}: {e}")
            continue
    
    return unique_item_buckets
# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    res = prepare_feature_sets_from_raw(
        "training_data/overall/weapon_bow_overall_standard.parquet",
        augment=True,
    )
    Xc, yc, mc = res["craft_all"]
    Xu, yu, mu = res["unique_all"]
    print(f"Craftables: {len(Xc)} rows, {Xc.shape[1]} features")
    print(f"Uniques   : {len(Xu)} rows, {Xu.shape[1]} features")
    print(f"Craftable buckets: {len(res['craft_by_category'])}")
    print(f"Unique buckets   : {len(res['unique_by_item'])}")
