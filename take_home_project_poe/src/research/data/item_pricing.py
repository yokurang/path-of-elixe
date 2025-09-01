# item_pricing.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast
import json
import math
import re

import numpy as np
import pandas as pd


# ---------------------------
# Dataclass (your schema)
# ---------------------------
@dataclass
class ItemData:
    """Structure for storing item data for ML training"""
    item_id: str
    timestamp: str
    league: str
    name: str
    base_type: str
    rarity: str
    item_level: int
    price_amount: Optional[float]
    price_currency: Optional[str]
    explicit_mods: List[str]
    mod_magnitudes: List[Dict]        # [{slot, name, tier, level, magnitudes:[{hash,min,max}]}...]
    implicit_mods: List[str]
    corrupted: bool
    type_line: str
    duplicated: bool
    unmodifiable: bool
    category: str

    # Convenience
    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------
# Parsing & normalization utils
# ---------------------------
def _jsonish_to_obj(x: Any) -> Any:
    """Parse strings like '[]'/'{}'/JSON/python-literal → objects. Return as-is on failure."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "[]":
            return []
        if s == "{}":
            return {}
        if s == "":
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(s)
            except Exception:
                pass
    return x

def _ensure_list(x: Any) -> List[Any]:
    obj = _jsonish_to_obj(x)
    if obj is None:
        return []
    return obj if isinstance(obj, list) else [obj]

def tier_num(t: Optional[str]) -> Optional[int]:
    """Parse 'S1','S2','P4' => 1,2,4. Lower is better (higher tier)."""
    if not t or not isinstance(t, str):
        return None
    m = re.search(r"(\d+)", t)
    return int(m.group(1)) if m else None

def tier_better_or_equal(candidate: Optional[str], target: Optional[str]) -> bool:
    """Return True if candidate tier is as good or better than target (numerically <=)."""
    tn_c = tier_num(candidate)
    tn_t = tier_num(target)
    if tn_c is None or tn_t is None:
        # if we can't compare, be permissive so we don't throw away potential comps
        return True
    return tn_c <= tn_t


# ---------------------------
# Category inference (heuristic)
# ---------------------------
_CATEGORY_PATTERNS = [
    (re.compile(r"^armour\.shield$", re.I), "shield"),
    (re.compile(r"^armour\.helmet$", re.I), "helmet"),
    (re.compile(r"^armour\.boots$", re.I), "boots"),
    (re.compile(r"^armour\.gloves$", re.I), "gloves"),
    (re.compile(r"^weapon\.", re.I), "weapon"),
    (re.compile(r"^accessory\.amulet$", re.I), "amulet"),
    (re.compile(r"^accessory\.ring$", re.I), "ring"),
    (re.compile(r"^accessory\.belt$", re.I), "belt"),
]

_PROPERTIES_TO_CATEGORY = {
    "Amulet": "amulet",
    "Ring": "ring",
    "Belt": "belt",
    "[Shield]": "shield",
    "[Bow]": "weapon",
    "[Wand]": "weapon",
    "[Staff]": "weapon",
}

_BASE_HINTS = [
    (re.compile(r"\bAmulet\b", re.I), "amulet"),
    (re.compile(r"\bRing\b", re.I), "ring"),
    (re.compile(r"\bBelt\b", re.I), "belt"),
    (re.compile(r"\bShield\b", re.I), "shield"),
    (re.compile(r"\bHelmet|Tiara|Mask\b", re.I), "helmet"),
    (re.compile(r"\bBoots\b", re.I), "boots"),
    (re.compile(r"\bGloves\b", re.I), "gloves"),
    (re.compile(r"\bSword|Axe|Mace|Dagger|Bow|Wand|Staff|Sceptre|Claw\b", re.I), "weapon"),
]

def infer_category(row: pd.Series) -> str:
    label = str(row.get("label") or "")
    for pat, cat in _CATEGORY_PATTERNS:
        if pat.search(label):
            return cat
    props = _ensure_list(row.get("properties"))
    if props:
        first = props[0]
        if isinstance(first, dict):
            nm = str(first.get("name") or "")
            if nm in _PROPERTIES_TO_CATEGORY:
                return _PROPERTIES_TO_CATEGORY[nm]
        else:
            s = str(first)
            if s in _PROPERTIES_TO_CATEGORY:
                return _PROPERTIES_TO_CATEGORY[s]
    bt = str(row.get("base_type") or row.get("type_line") or "")
    nm = str(row.get("name") or "")
    for text in (bt, nm):
        for pat, cat in _BASE_HINTS:
            if pat.search(text):
                return cat
    # fallback
    return (str(row.get("category") or "unknown")).lower()


# ---------------------------
# Mod extraction
# ---------------------------
def extract_mods_from_extended(ext: Any) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Returns (explicit_mods_text, implicit_mods_text, mod_magnitudes_records)
    mod_magnitudes_records elements like:
      {"slot":"explicit","name":"Rotund","tier":"P5","level":46,
       "magnitudes":[{"hash":"explicit.stat_...","min":"85","max":"99"}]}
    """
    ext = _jsonish_to_obj(ext) or {}
    out_exp: List[str] = []
    out_imp: List[str] = []
    magrecs: List[Dict] = []

    mods = ext.get("mods") or {}
    for slot in ("explicit", "implicit"):
        arr = _ensure_list(mods.get(slot))
        for m in arr:
            if not isinstance(m, dict):
                continue
            name = m.get("name") or ""
            tier = m.get("tier")
            level = m.get("level")
            mags = _ensure_list(m.get("magnitudes"))
            # Build human text if possible
            text = name if name else ""
            if text:
                (out_exp if slot == "explicit" else out_imp).append(text)
            # Record magnitude structure
            magrecs.append({
                "slot": slot,
                "name": name,
                "tier": tier,
                "level": level,
                "magnitudes": mags,
            })

    return out_exp, out_imp, magrecs


def extract_mods_strings(row: pd.Series) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Prefer 'extended.mods' (authoritative tiers/magnitudes). If unavailable,
    fall back to string arrays 'explicit_mods'/'implicit_mods'.
    """
    exp_s, imp_s, mags = extract_mods_from_extended(row.get("extended"))
    if not exp_s and isinstance(row.get("explicit_mods"), (list, str)):
        exp_s = _ensure_list(row.get("explicit_mods"))
    if not imp_s and isinstance(row.get("implicit_mods"), (list, str)):
        imp_s = _ensure_list(row.get("implicit_mods"))
    return exp_s, imp_s, mags


# ---------------------------
# Row conversion
# ---------------------------
ESSENTIAL_COLS = [
    "id", "indexed", "league", "name", "base_type", "type_line", "rarity",
    "ilvl", "price_amount_in_base", "price_currency_in_base",
    "explicit_mods", "implicit_mods", "extended",
    "corrupted", "duplicated", "unmodifiable", "properties", "label", "category",
]

def to_itemdata(row: pd.Series) -> ItemData:
    exp, imp, mags = extract_mods_strings(row)
    return ItemData(
        item_id=str(row.get("id")),
        timestamp=str(row.get("indexed") or ""),
        league=str(row.get("league") or ""),
        name=str(row.get("name") or ""),
        base_type=str(row.get("base_type") or ""),
        rarity=str(row.get("rarity") or ""),
        item_level=int(row.get("ilvl") or 0),
        price_amount=float(row.get("price_amount_in_base")) if pd.notna(row.get("price_amount_in_base")) else None,
        price_currency=str(row.get("price_currency_in_base") or None),
        explicit_mods=[str(x) for x in exp],
        mod_magnitudes=mags,
        implicit_mods=[str(x) for x in imp],
        corrupted=bool(row.get("corrupted")),
        type_line=str(row.get("type_line") or ""),
        duplicated=bool(row.get("duplicated")),
        unmodifiable=bool(row.get("unmodifiable")),
        category=infer_category(row),
    )


def load_items_from_parquet(path: Path | str) -> List[ItemData]:
    df = pd.read_parquet(path)
    for c in ESSENTIAL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    # normalize json-ish
    for col in ("extended", "explicit_mods", "implicit_mods", "properties"):
        if col in df.columns:
            df[col] = df[col].map(_jsonish_to_obj)
    return [to_itemdata(r) for _, r in df.iterrows()]


# ---------------------------
# Comparable-search pricing
# ---------------------------
def _mods_index(magrecs: List[Dict]) -> Dict[str, Dict]:
    """
    Map mod name -> best (highest) tier & approx max magnitude seen on the item.
    Assumes lower numeric tier is better (1 better than 4).
    """
    idx: Dict[str, Dict] = {}
    for m in magrecs:
        name = m.get("name") or ""
        tr = m.get("tier")
        # pick best tier (lowest number)
        old = idx.get(name)
        if old is None or (tier_num(tr) or 999) < (tier_num(old["tier"]) or 999):
            maxmag = None
            mags = m.get("magnitudes") or []
            # try to capture a representative 'max' if present
            for mg in mags:
                try:
                    mx = float(mg.get("max"))
                    maxmag = mx if (maxmag is None or mx > maxmag) else maxmag
                except Exception:
                    pass
            idx[name] = {"tier": tr, "max": maxmag}
    return idx


def _candidate_has_mod_at_least(candidate_idx: Dict[str, Dict], name: str, need: Dict) -> bool:
    """Check candidate has mod 'name' with tier >= needed (better or equal) and magnitude >= needed if available."""
    got = candidate_idx.get(name)
    if not got:
        return False
    if not tier_better_or_equal(got.get("tier"), need.get("tier")):
        return False
    need_max = need.get("max")
    if need_max is not None and got.get("max") is not None:
        return got["max"] >= need_max - 1e-9
    return True


def _top_explicit_mod_names(item: ItemData, k: int = 3) -> List[str]:
    """Pick top-K explicit mod names by tier quality (S1/P1 better than S4/P4)."""
    idx = _mods_index(item.mod_magnitudes)
    # sort by tier number ascending; stable if None
    ordered = sorted(idx.items(), key=lambda kv: (tier_num(kv[1]["tier"]) or 999, kv[0]))
    names = [n for n, _ in ordered if n]  # drop empty names
    return names[:k] if k > 0 else names


def _row_to_itemindex(row: pd.Series) -> Dict[str, Dict]:
    """Build mod index for a pool row."""
    _, _, mags = extract_mods_strings(row)
    return _mods_index(mags)


def _filter_by_base(df: pd.DataFrame, item: ItemData) -> pd.DataFrame:
    out = df[df["league"] == item.league]
    # Normalize types for safety
    out = out[out["base_type"].fillna("") == item.base_type]
    return out


def comparable_listings(pool_df: pd.DataFrame, item: ItemData) -> pd.DataFrame:
    """
    Implements your rarity-based search logic:
      - Normal: base type only
      - Rare/Magic: base type + top-tier explicit mods (progressive relaxation)
      - Unique: same base + mods at least as strong (tier/magnitude)
    """
    base_pool = _filter_by_base(pool_df, item)

    # Quick exits
    if item.rarity.lower() == "normal":
        return base_pool[base_pool["rarity"].str.lower() == "normal"]

    if item.rarity.lower() == "unique":
        pool = base_pool[base_pool["rarity"].str.lower() == "unique"].copy()
        # Prefer exact unique identification via type_line/name when present
        if item.type_line:
            pool = pool[pool["type_line"] == item.type_line]
        if not pool.empty and item.name:
            pool2 = pool[pool["name"] == item.name]
            if not pool2.empty:
                pool = pool2

        # Keep comps with equal-or-better mods
        need_idx = _mods_index(item.mod_magnitudes)
        mask = []
        for _, r in pool.iterrows():
            cand_idx = _row_to_itemindex(r)
            ok = all(_candidate_has_mod_at_least(cand_idx, n, need) for n, need in need_idx.items() if n)
            mask.append(ok)
        comps = pool[pd.Series(mask, index=pool.index)]
        return comps if not comps.empty else pool

    # Rare/Magic (same loop; magic just has fewer explicit mods)
    pool = base_pool[base_pool["rarity"].str.lower().isin(["rare", "magic"])].copy()
    # choose top-K mods; start specific, then relax
    all_top = _top_explicit_mod_names(item, k=3)  # tune k
    need_idx = _mods_index(item.mod_magnitudes)

    for k in range(len(all_top), -1, -1):
        want = set(all_top[:k])
        if not want:
            # last resort: base only (already filtered)
            return pool
        mask = []
        for _, r in pool.iterrows():
            cand_idx = _row_to_itemindex(r)
            ok = True
            for n in want:
                need = need_idx.get(n)
                if not need or not _candidate_has_mod_at_least(cand_idx, n, need):
                    ok = False
                    break
            mask.append(ok)
        comps = pool[pd.Series(mask, index=pool.index)]
        if not comps.empty:
            return comps

    return pool  # fallback


def estimate_price_floor(pool_df: pd.DataFrame, item: ItemData, floor_quantile: float = 0.25) -> Optional[float]:
    comps = comparable_listings(pool_df, item)
    if comps.empty:
        return None
    prices = comps["price_amount_in_base"].astype(float)
    prices = prices[prices > 0]
    if prices.empty:
        return None
    return float(np.quantile(prices, floor_quantile))


# ---------------------------
# Public preprocessing API
# ---------------------------
def parquet_to_items(path: Path | str) -> List[ItemData]:
    return load_items_from_parquet(path)

def items_to_dataframe(items: List[ItemData]) -> pd.DataFrame:
    return pd.DataFrame([it.to_row() for it in items])


# ---------------------------
# Example wiring
# ---------------------------
"""
from pathlib import Path
import pandas as pd

# 1) Load your raw market listings as a pool of comps
pool_df = pd.read_parquet("training_data/poe2_all_listings.parquet")

# 2) Convert one file into ItemData rows for training or pricing
items = parquet_to_items("training_data/poe2_20250901-010915_accessory_amulet_corrupted.parquet")

# 3) Get a price floor estimate for each item (using its rarity logic)
for it in items[:5]:
    floor = estimate_price_floor(pool_df, it, floor_quantile=0.25)
    print(it.item_id, it.rarity, it.base_type, "→ floor:", floor)
"""
