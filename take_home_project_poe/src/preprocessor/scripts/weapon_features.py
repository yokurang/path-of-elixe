from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import math
import re
import json

import pandas as pd
import numpy as np

from utils import find_project_root

# -----------------------------------------------------------------------------
# Constants and Regexes
# -----------------------------------------------------------------------------
RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*$")
PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def _to_float(x: Any) -> float:
    """Convert to float with fallback to NaN."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        return float(str(x).replace(",", "").strip())
    except (ValueError, TypeError):
        return np.nan

def _safe_parse_json(x: Any) -> Any:
    """Parse JSON strings; return original value if parsing fails."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (json.JSONDecodeError, ValueError):
            return x
    return x

def _normalize_text(s: str) -> str:
    """Normalize text by removing PoE markup and standardizing whitespace."""
    s = re.sub(r"[\[\]\|]", "", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _extract_property_value(values: Any) -> Optional[str]:
    """Extract value from PoE property values structure like [['31-47', 0]]."""
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)
    return None

def _parse_numeric_range(s: Any) -> Tuple[float, float, float]:
    """
    Parse numeric ranges, percentages, or single values.
    Returns (min, max, average).
    Percentages are converted to fractions (e.g., 5% -> 0.05).
    """
    if s is None:
        return (np.nan, np.nan, np.nan)
    
    text = str(s).strip()
    
    # Try range pattern first (e.g., "31-47")
    range_match = RANGE_RE.match(text)
    if range_match:
        min_val = _to_float(range_match.group(1))
        max_val = _to_float(range_match.group(2))
        avg_val = (min_val + max_val) / 2.0
        return (min_val, max_val, avg_val)
    
    # Try percentage pattern (e.g., "5.00%")
    percent_match = PERCENT_RE.search(text)
    if percent_match:
        val = _to_float(percent_match.group(1)) / 100.0
        return (val, val, val)
    
    # Try single number
    num_match = NUM_RE.search(text)
    if num_match:
        val = _to_float(num_match.group(0))
        return (val, val, val)
    
    return (np.nan, np.nan, np.nan)

# -----------------------------------------------------------------------------
# Property Extractors
# -----------------------------------------------------------------------------
def _extract_weapon_properties(properties: Any) -> Dict[str, float]:
    """Extract damage, crit chance, and APS from weapon properties."""
    result = {}
    properties = _safe_parse_json(properties)
    
    if not isinstance(properties, list):
        return result
    
    def _normalize_prop_name(name: str) -> str:
        name = _normalize_text(name)
        # Standardize critical hit chance naming
        return name.replace("critical hit chance", "critical chance")
    
    for prop in properties:
        if not isinstance(prop, dict):
            continue
            
        prop_name = _normalize_prop_name(prop.get("name", ""))
        values = prop.get("values", [])
        value_text = _extract_property_value(values)
        
        if "physical damage" in prop_name:
            min_val, max_val, avg_val = _parse_numeric_range(value_text)
            result.update({
                "phys_min": min_val, "phys_max": max_val, "phys_avg": avg_val
            })
        elif "chaos damage" in prop_name:
            min_val, max_val, avg_val = _parse_numeric_range(value_text)
            result.update({
                "chaos_min": min_val, "chaos_max": max_val, "chaos_avg": avg_val
            })
        elif "cold damage" in prop_name:
            min_val, max_val, avg_val = _parse_numeric_range(value_text)
            result.update({
                "cold_min": min_val, "cold_max": max_val, "cold_avg": avg_val
            })
        elif "fire damage" in prop_name:
            min_val, max_val, avg_val = _parse_numeric_range(value_text)
            result.update({
                "fire_min": min_val, "fire_max": max_val, "fire_avg": avg_val
            })
        elif "lightning damage" in prop_name:
            min_val, max_val, avg_val = _parse_numeric_range(value_text)
            result.update({
                "lightning_min": min_val, "lightning_max": max_val, "lightning_avg": avg_val
            })
        elif "critical chance" in prop_name:
            _, _, result["crit_chance"] = _parse_numeric_range(value_text)
        elif "attacks per second" in prop_name:
            _, _, result["aps"] = _parse_numeric_range(value_text)
        elif "quality" in prop_name:
            _, _, result["quality"] = _parse_numeric_range(value_text)
        elif "spirit" in prop_name:
            _, _, result["spirit"] = _parse_numeric_range(value_text)
    
    return result

def _extract_requirements(requirements: Any) -> Dict[str, float]:
    """Extract level and attribute requirements."""
    result = {
        "req_level": np.nan, "req_str": np.nan, 
        "req_dex": np.nan, "req_int": np.nan
    }
    
    requirements = _safe_parse_json(requirements)
    if not isinstance(requirements, list):
        return result
    
    def _find_requirement_by_suffix(suffix: str) -> Optional[float]:
        for req in requirements:
            if not isinstance(req, dict):
                continue
            name = _normalize_text(req.get("name", ""))
            if name.endswith(suffix):
                value_text = _extract_property_value(req.get("values", []))
                return _parse_numeric_range(value_text)[2]  # Return average
        return None
    
    # Map requirement types to their possible name suffixes
    req_mappings = [
        ("req_level", ["level"]),
        ("req_str", ["strength", "str"]),
        ("req_dex", ["dexterity", "dex"]),
        ("req_int", ["intelligence", "int"]),
    ]
    
    for req_key, suffixes in req_mappings:
        for suffix in suffixes:
            value = _find_requirement_by_suffix(suffix)
            if value is not None:
                result[req_key] = value
                break
    
    return result

def _extract_socket_info(sockets: Any) -> Tuple[int, int]:
    """Extract total sockets and rune sockets count."""
    sockets = _safe_parse_json(sockets)
    if not isinstance(sockets, list):
        return (0, 0)
    
    total_sockets = len(sockets)
    rune_sockets = 0
    
    for socket in sockets:
        if isinstance(socket, dict):
            socket_type = str(socket.get("type", "")).lower()
        elif isinstance(socket, str):
            socket_type = socket.lower()
        else:
            continue
            
        if socket_type == "rune":
            rune_sockets += 1
    
    return (total_sockets, rune_sockets)

def _extract_extended_stats(extended: Any) -> Dict[str, float]:
    """Extract DPS and augmentation stats from extended field."""
    result = {
        "dps": np.nan, "pdps": np.nan, "edps": np.nan,
        "dps_aug": 0.0, "pdps_aug": 0.0, "edps_aug": 0.0,
    }
    
    extended = _safe_parse_json(extended)
    if not isinstance(extended, dict):
        return result
    
    # Extract DPS values
    for stat in ("dps", "pdps", "edps"):
        if stat in extended:
            result[stat] = _to_float(extended[stat])
    
    # Extract augmentation flags
    for key, value in extended.items():
        if isinstance(key, str) and key.endswith("_aug"):
            result[key] = 1.0 if bool(value) else 0.0
    
    # Extract other numeric fields (excluding complex nested structures)
    excluded_keys = {"mods", "hashes"}
    for key, value in extended.items():
        if key in result or key in excluded_keys:
            continue
        numeric_val = _to_float(value)
        if not math.isnan(numeric_val):
            result[key] = numeric_val
    
    return result

def _extract_mod_features(implicit_mods: Any, explicit_mods: Any, extended: Any) -> Dict[str, float]:
    """Extract mod count and key mod presence flags."""
    result = {
        "n_implicit_mods": 0.0,
        "n_explicit_mods": 0.0,
        "has_added_fire": 0.0,
        "has_added_cold": 0.0,
        "has_added_lightning": 0.0,
        "has_added_chaos": 0.0,
        "has_inc_phys_pct": 0.0,
        "has_inc_attack_speed_pct": 0.0,
        "has_inc_crit_chance_pct": 0.0,
        "has_accuracy": 0.0,
    }
    
    def _extract_mod_names(mods: Any) -> List[str]:
        mods = _safe_parse_json(mods)
        mod_names = []
        
        if isinstance(mods, list):
            for mod in mods:
                if isinstance(mod, dict) and "name" in mod:
                    mod_names.append(_normalize_text(mod["name"]))
                else:
                    mod_names.append(_normalize_text(str(mod)))
        
        return mod_names
    
    # Extract mod names
    implicit_names = _extract_mod_names(implicit_mods)
    explicit_names = _extract_mod_names(explicit_mods)
    
    # Try to get mods from extended if primary sources are empty
    extended_data = _safe_parse_json(extended)
    if isinstance(extended_data, dict) and "mods" in extended_data:
        mods_data = extended_data["mods"]
        if isinstance(mods_data, dict):
            if not implicit_names and "implicit" in mods_data:
                implicit_names = _extract_mod_names(mods_data["implicit"])
            if not explicit_names and "explicit" in mods_data:
                explicit_names = _extract_mod_names(mods_data["explicit"])
    
    result["n_implicit_mods"] = float(len(implicit_names))
    result["n_explicit_mods"] = float(len(explicit_names))
    
    # Check for specific mod types
    all_mod_names = implicit_names + explicit_names
    for mod_name in all_mod_names:
        if "adds" in mod_name and "fire" in mod_name:
            result["has_added_fire"] = 1.0
        if "adds" in mod_name and "cold" in mod_name:
            result["has_added_cold"] = 1.0
        if "adds" in mod_name and "lightning" in mod_name:
            result["has_added_lightning"] = 1.0
        if "adds" in mod_name and "chaos" in mod_name:
            result["has_added_chaos"] = 1.0
        if "% increased" in mod_name and "physical" in mod_name:
            result["has_inc_phys_pct"] = 1.0
        if "% increased" in mod_name and "attack speed" in mod_name:
            result["has_inc_attack_speed_pct"] = 1.0
        if "% increased" in mod_name and ("critical chance" in mod_name or "critical strike chance" in mod_name):
            result["has_inc_crit_chance_pct"] = 1.0
        if "accuracy" in mod_name:
            result["has_accuracy"] = 1.0
    
    return result

# -----------------------------------------------------------------------------
# DPS Computation
# -----------------------------------------------------------------------------
def _compute_channel_dps(features: pd.DataFrame) -> pd.DataFrame:
    """Compute DPS for each damage channel using DPS = APS × average_damage."""
    result = features.copy()
    
    def _safe_numeric(col: str) -> pd.Series:
        return pd.to_numeric(result.get(col, 0.0), errors="coerce").fillna(0.0)
    
    aps = _safe_numeric("aps")
    
    # Compute per-channel DPS
    result["pdps"] = aps * _safe_numeric("phys_avg")
    result["fdps"] = aps * _safe_numeric("fire_avg")
    result["cdps"] = aps * _safe_numeric("cold_avg")
    result["ldps"] = aps * _safe_numeric("lightning_avg")
    result["chaos_dps"] = aps * _safe_numeric("chaos_avg")
    
    return result

# -----------------------------------------------------------------------------
# Feature Engineering Helpers
# -----------------------------------------------------------------------------
def _safe_drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop columns that exist in the dataframe."""
    existing_cols = [col for col in columns if col in df.columns]
    return df.drop(columns=existing_cols) if existing_cols else df

def _create_count_dummies(
    series: pd.Series,
    prefix: str,
    drop_first: bool = True,
    max_value: Optional[int] = None
) -> pd.DataFrame:
    """Create one-hot encoding for count data with optional clipping."""
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    
    if max_value is not None:
        numeric_series = numeric_series.clip(lower=0, upper=max_value)
    
    dummies = pd.get_dummies(numeric_series, prefix=prefix)
    
    if drop_first and len(dummies.columns) > 1:
        # Drop the smallest category (typically 0) as baseline
        baseline_col = dummies.columns.sort_values()[0]
        dummies = dummies.drop(columns=baseline_col)
    
    return dummies.astype(float)

# -----------------------------------------------------------------------------
# Main Feature Engineering Function
# -----------------------------------------------------------------------------
def make_weapon_features(
    df: pd.DataFrame,
    *,
    one_hot_rarity: bool = True,
    drop_raw_damage: bool = True,
    drop_aps: bool = True,
    drop_extended_dps: bool = True,
    keep_mod_flags: bool = False,
    sockets_one_hot: bool = True,
    sockets_max_cap: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform PoE2 weapon data into ML features.
    
    Args:
        df: Raw weapon data DataFrame
        one_hot_rarity: Create rarity dummy variables
        drop_raw_damage: Remove min/max/avg damage columns after computing DPS
        drop_aps: Remove attacks per second (redundant with DPS)
        drop_extended_dps: Remove precomputed DPS from extended stats
        keep_mod_flags: Keep has_* mod indicator flags
        sockets_one_hot: One-hot encode socket counts
        sockets_max_cap: Maximum sockets for one-hot encoding
    
    Returns:
        Tuple of (features, targets, metadata) DataFrames
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Initialize result DataFrames
    features = pd.DataFrame(index=df.index)
    targets = pd.DataFrame(index=df.index)
    metadata = pd.DataFrame(index=df.index)
    
    # --- Extract Targets ---
    targets["price_in_base"] = pd.to_numeric(
        df.get("price_amount_in_base", np.nan), errors="coerce"
    )
    targets["log_price"] = np.log1p(targets["price_in_base"].clip(lower=0))
    
    # --- Extract Metadata ---
    meta_columns = [
        "id", "league", "indexed", "rarity", "category", 
        "base_type", "type_line", "ilvl"
    ]
    for col in meta_columns:
        if col in df.columns:
            metadata[col] = df[col]
    
    # --- Extract Basic Features ---
    features["ilvl"] = pd.to_numeric(df.get("ilvl", np.nan), errors="coerce")
    
    # Boolean flags as numeric
    boolean_cols = ["verified", "identified", "corrupted", "duplicated", "unmodifiable", "support"]
    for col in boolean_cols:
        if col in df.columns:
            features[col] = df[col].astype(bool).astype(float)
    
    # --- Socket Features ---
    socket_data = df.get("sockets", [None] * len(df))
    socket_counts = [_extract_socket_info(sockets) for sockets in socket_data]
    total_sockets, rune_sockets = zip(*socket_counts)
    
    features["n_sockets"] = list(total_sockets)
    features["n_rune_sockets"] = list(rune_sockets)
    
    if sockets_one_hot:
        socket_dummies = _create_count_dummies(
            features["n_sockets"], 
            prefix="sockets", 
            drop_first=True, 
            max_value=sockets_max_cap
        )
        features = pd.concat([features, socket_dummies], axis=1)
        features = _safe_drop_columns(features, ["n_sockets"])
    
    # Always drop redundant rune socket features
    features = _safe_drop_columns(features, ["n_rune_sockets"])
    
    # --- Requirement Features ---
    req_source = df.get("requirements", df.get("weapon_requirements", [None] * len(df)))
    req_data = [_extract_requirements(reqs) for reqs in req_source]
    req_df = pd.DataFrame(req_data, index=df.index)
    
    # Derive primary requirement as max of str/dex/int
    stat_reqs = req_df[["req_str", "req_dex", "req_int"]]
    req_df["req_primary"] = stat_reqs.max(axis=1, skipna=True)
    
    # Add primary requirement only
    features["req_primary"] = req_df["req_primary"]
    
    # --- Weapon Property Features ---
    property_data = [_extract_weapon_properties(props) for props in df.get("properties", [None] * len(df))]
    prop_df = pd.DataFrame(property_data, index=df.index)
    features = pd.concat([features, prop_df], axis=1)
    
    # --- Extended Stats ---
    extended_data = [_extract_extended_stats(ext) for ext in df.get("extended", [None] * len(df))]
    ext_df = pd.DataFrame(extended_data, index=df.index)
    features = pd.concat([features, ext_df], axis=1)
    
    # Consolidate augmentation flags
    aug_columns = [col for col in features.columns if col.endswith("_aug")]
    if aug_columns:
        features["has_augmentation"] = (features[aug_columns].max(axis=1) > 0).astype(float)
        features = _safe_drop_columns(features, aug_columns)
    
    if drop_extended_dps:
        features = _safe_drop_columns(features, ["dps", "pdps", "edps"])
    
    # --- Mod Features ---
    mod_data = [
        _extract_mod_features(impl, expl, ext) 
        for impl, expl, ext in zip(
            df.get("implicit_mods", [None] * len(df)),
            df.get("explicit_mods", [None] * len(df)),
            df.get("extended", [None] * len(df))
        )
    ]
    mod_df = pd.DataFrame(mod_data, index=df.index)
    features = pd.concat([features, mod_df], axis=1)
    
    if not keep_mod_flags:
        has_columns = [col for col in features.columns if col.startswith("has_")]
        features = _safe_drop_columns(features, has_columns)
    
    # --- Compute Per-Channel DPS ---
    features = _compute_channel_dps(features)
    
    # --- Cleanup Raw Damage Columns ---
    if drop_raw_damage:
        damage_columns = [
            "phys_min", "phys_max", "phys_avg",
            "fire_min", "fire_max", "fire_avg", 
            "cold_min", "cold_max", "cold_avg",
            "lightning_min", "lightning_max", "lightning_avg",
            "chaos_min", "chaos_max", "chaos_avg",
        ]
        features = _safe_drop_columns(features, damage_columns)
    
    if drop_aps:
        features = _safe_drop_columns(features, ["aps"])
    
    # --- Rarity One-Hot Encoding ---
    if one_hot_rarity:
        rarity_series = df.get("rarity", "unknown").fillna("unknown").astype(str).str.lower()
        rarity_dummies = pd.get_dummies(rarity_series, prefix="rarity")
        features = pd.concat([features, rarity_dummies], axis=1)
    
    # Remove any category dummies that might exist
    cat_columns = [col for col in features.columns if col.startswith("cat_")]
    features = _safe_drop_columns(features, cat_columns)
    
    # --- Final Data Cleaning ---
    # Ensure all features are numeric
    for col in features.columns:
        # if features[col].dtype == np.bool: 
        #     features[col] = features[col].astype(int)

        if features[col].dtype == object:
            features[col] = pd.to_numeric(features[col], errors="coerce")
    
    # Handle infinite values and NaNs
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0.0, inplace=True)
    targets.replace([np.inf, -np.inf], np.nan, inplace=True)
    targets.fillna(0.0, inplace=True)
    
    return features, targets, metadata

def load_and_process_weapon_data(
    file_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
    **feature_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load parquet file and process into weapon features.
    
    Args:
        file_path: Path to parquet file
        base_path: Base directory for relative paths
        **feature_kwargs: Arguments passed to make_weapon_features
    
    Returns:
        Tuple of (features, targets, metadata) DataFrames
    """
    # Resolve file path
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        root = Path(base_path).expanduser().resolve() if base_path else find_project_root()
        path = (root / path).resolve()
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load parquet with engine fallback
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception as pyarrow_error:
        try:
            df = pd.read_parquet(path, engine="fastparquet")
        except Exception as fastparquet_error:
            raise RuntimeError(
                "Failed to read parquet file with available engines. "
                "Please install pyarrow or fastparquet.\n"
                f"PyArrow error: {pyarrow_error}\n"
                f"FastParquet error: {fastparquet_error}"
            ) from None
    
    return make_weapon_features(df, **feature_kwargs)

def validate_features(features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
    """Validate processed features and targets for common issues."""
    validation_results = {}
    
    # Check shapes
    validation_results["features_shape"] = features.shape
    validation_results["targets_shape"] = targets.shape
    
    # Check for NaN and infinite values
    features_array = features.to_numpy()
    targets_array = targets.to_numpy()
    
    validation_results["features_nan_count"] = int(pd.isna(features_array).sum())
    validation_results["targets_nan_count"] = int(pd.isna(targets_array).sum())
    validation_results["features_inf_count"] = int(np.isinf(features_array).sum())
    validation_results["targets_inf_count"] = int(np.isinf(targets_array).sum())
    
    # Identify problematic columns
    features_nan_cols = features.columns[features.isna().any()].tolist()
    targets_nan_cols = targets.columns[targets.isna().any()].tolist()
    
    validation_results["features_nan_columns"] = features_nan_cols
    validation_results["targets_nan_columns"] = targets_nan_cols
    
    # Print summary
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
    print(f"NaN values - Features: {validation_results['features_nan_count']}, Targets: {validation_results['targets_nan_count']}")
    print(f"Infinite values - Features: {validation_results['features_inf_count']}, Targets: {validation_results['targets_inf_count']}")
    
    if features_nan_cols:
        cols_preview = features_nan_cols[:10]
        suffix = "..." if len(features_nan_cols) > 10 else ""
        print(f"Features with NaN ({len(features_nan_cols)}): {cols_preview}{suffix}")
    
    if targets_nan_cols:
        print(f"Targets with NaN: {targets_nan_cols}")
    
    return validation_results