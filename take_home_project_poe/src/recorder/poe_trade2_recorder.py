import asyncio
import json
import logging
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import yaml

from main import setup_logging
from src.recorder.constants import (
    # URLs & paths
    POE_BASE_URL,
    TRADE2_SEARCH_URL,
    TRADE2_FETCH_URL,
    CONFIG_PATH,
    COOKIES_PATH,
    CURRENCY_CACHE_PATH,
    # Currency helpers
    CURRENCY_MAP,
    FULL_TO_SHORT_CURRENCY_MAP,
    SHORT_TO_FULL_CURRENCY_MAP,
    # Optional helpers for building category payloads
    ITEM_TYPES,
    ITEM_RARITIES,
)

# For FX cache compatibility mapping
from src.recorder.poe_currency_recorder import Quote, FXCache  # noqa: F401

# ---------------------------
# Tunables / Constants
# ---------------------------
MAX_RESULTS = 100
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 5
POLITE_PAUSE = 1.5  # short sleep between requests to be nice to API

HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": POE_BASE_URL,
    "Priority": "u=1, i",
    "Referer": POE_BASE_URL,  # overwritten per request
    "Sec-CH-UA": "\"Not;A=Brand\";v=\"99\", \"Brave\";v=\"139\", \"Chromium\";v=\"139\"",
    "Sec-CH-UA-Arch": "\"x86\"",
    "Sec-CH-UA-Bitness": "\"64\"",
    "Sec-CH-UA-Full-Version-List": "\"Not;A=Brand\";v=\"99.0.0.0\", \"Brave\";v=\"139.0.0.0\", \"Chromium\";v=\"139.0.0.0\"",
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Model": "",
    "Sec-CH-UA-Platform": "\"Windows\"",
    "Sec-CH-UA-Platform-Version": "\"19.0.0\"",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-GPC": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

log_search = logging.getLogger("poe.trade2.search")
log_fetch = logging.getLogger("poe.trade2.fetch")
for lg in (log_search, log_fetch):
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())


def _retry_after_seconds(resp: aiohttp.ClientResponse) -> Optional[float]:
    ra_raw = resp.headers.get("Retry-After")
    if ra_raw:
        s = ra_raw.strip()
        try:
            return max(0.0, float(s))
        except Exception:
            pass
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception as e:
            raise ValueError(f"_retry_after_seconds failed: {e}")
    x_reset = resp.headers.get("X-RateLimit-Reset")
    if x_reset:
        try:
            reset_ts = float(x_reset)
            now_ts = datetime.now(timezone.utc).timestamp()
            return max(0.0, reset_ts - now_ts)
        except Exception as e:
            raise ValueError(f"_retry_after_seconds failed: {e}")
    return None


def _expo_backoff(attempt: int, base: float = 0.5, cap: float = 12.0) -> float:
    raw = min(cap, base * (2 ** (attempt - 1)))
    return max(0.0, raw * (1.0 + random.uniform(-0.15, 0.15)))


def _build_search_url(*, realm: str, league: str) -> str:
    from urllib.parse import quote
    return TRADE2_SEARCH_URL.format(realm=realm, league=quote(league, safe=""))


def _build_referer(*, realm: str, league: str) -> str:
    from urllib.parse import quote
    return f"{POE_BASE_URL}/trade2/search/{realm}/{quote(league, safe='')}"


def _masked_cookies_for_log(obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj or {})
    if "POESESSID" in out:
        out["POESESSID"] = out["POESESSID"][:6] + "…"
    if "cf_clearance" in out:
        out["cf_clearance"] = out["cf_clearance"][:6] + "…"
    return out


def _summarize_payload(payload: Dict[str, Any]) -> str:
    q = payload.get("query", {})
    status = (q.get("status") or {}).get("option")
    sort = payload.get("sort")
    filters = q.get("filters") or {}
    type_filters = (filters.get("type_filters") or {}).get("filters") or {}
    misc_filters = (filters.get("misc_filters") or {}).get("filters") or {}
    cat = (type_filters.get("category") or {}).get("option")
    rarity = (misc_filters.get("rarity") or {}).get("option")
    return f"status={status}, sort={sort}, category={cat}, rarity={rarity}"


# ---------------------------
# Config & cookies
# ---------------------------
def load_league_realm(path: Path = CONFIG_PATH) -> Tuple[str, str]:
    """Read only league/realm from config.yaml, with sane defaults."""
    league = "Standard"
    realm = "poe2"
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        league = str(cfg.get("league") or league)
        realm = str(cfg.get("realm") or realm)
    except FileNotFoundError:
        log_search.info("CONFIG not found (%s); using defaults realm=%s league=%s", path, realm, league)
    return league, realm


def load_cookies(path: Path = COOKIES_PATH) -> Dict[str, str]:
    """Read cookies from poe_cookies_config.json -> {'cookies': {...}}"""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        cookies = {str(k): str(v) for k, v in (raw.get("cookies") or {}).items()}
        log_search.info("Loaded cookies: %s", _masked_cookies_for_log(cookies))
        return cookies
    except FileNotFoundError:
        log_search.warning("Cookies file not found: %s", path)
        return {}
    except Exception as e:
        log_search.error("Failed to read cookies file %s: %s", path, e)
        return {}


def headers_with_cookies(base: Dict[str, str], cookies: Dict[str, str]) -> Dict[str, str]:
    h = dict(base)
    parts = []
    if cookies.get("POESESSID"):
        parts.append(f"POESESSID={cookies['POESESSID']}")
    if cookies.get("cf_clearance"):
        parts.append(f"cf_clearance={cookies['cf_clearance']}")
    if parts:
        h["Cookie"] = "; ".join(parts)
    return h


@dataclass(frozen=True)
class _FXView:
    pair_rates_full: Dict[Tuple[str, str], float]
    pair_rates_short: Dict[Tuple[str, str], float]
    full_map: Dict[str, str]   # full_name -> short
    short_map: Dict[str, str]  # short -> full


def _extract_fx_view(obj: Any) -> _FXView:
    """
    Accepts either:
      - an FXCache instance,
      - any FXCache-like object with matching attributes,
      - or a dict with similarly named keys.
    """
    # Exact class match first
    try:
        from src.recorder.poe_currency_recorder import FXCache as _RealFXCache  # type: ignore
        if isinstance(obj, _RealFXCache):
            return _FXView(
                pair_rates_full=dict(getattr(obj, "pair_rates_full") or {}),
                pair_rates_short=dict(getattr(obj, "pair_rates_short") or {}),
                full_map=dict(getattr(obj, "full_map") or {}),
                short_map=dict(getattr(obj, "short_map") or {}),
            )
    except Exception:
        pass

    # FXCache-like object (attribute duck-typing)
    needed = ("pair_rates_full", "pair_rates_short", "full_map", "short_map")
    if all(hasattr(obj, k) for k in needed):
        return _FXView(
            pair_rates_full=dict(getattr(obj, "pair_rates_full") or {}),
            pair_rates_short=dict(getattr(obj, "pair_rates_short") or {}),
            full_map=dict(getattr(obj, "full_map") or {}),
            short_map=dict(getattr(obj, "short_map") or {}),
        )

    # Dict payload
    if isinstance(obj, dict):
        return _FXView(
            pair_rates_full=dict(obj.get("pair_rates_full") or {}),
            pair_rates_short=dict(obj.get("pair_rates_short") or {}),
            full_map=dict(obj.get("full_map") or {}),
            short_map=dict(obj.get("short_map") or {}),
        )

    raise ValueError("Unrecognized FX cache format")


def load_fx_cache_or_raise(path: Path = CURRENCY_CACHE_PATH) -> _FXView:
    """
    Load cache/currency_fx.bak.pkl robustly. First try normal pickle; if that fails
    due to a class-path mismatch, use a compatibility unpickler that remaps any
    historical 'FXCache'/'Quote' to the real classes from src.recorder.poe_currency_recorder.
    """
    from src.recorder.poe_currency_recorder import FXCache as _RealFXCache, Quote as _RealQuote  # type: ignore

    def _load_with_compat(fp):
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if name == "FXCache" and (module.startswith("src.recorder.") or module == "__main__"):
                    return _RealFXCache
                if name == "Quote" and (module.startswith("src.recorder.") or module == "__main__"):
                    return _RealQuote
                return super().find_class(module, name)
        return _CompatUnpickler(fp).load()

    try:
        with path.open("rb") as f:
            try:
                raw = pickle.load(f)  # fast path
            except Exception:
                f.seek(0)
                raw = _load_with_compat(f)  # remap classes if needed
        fxv = _extract_fx_view(raw)
        if not fxv.pair_rates_full and not fxv.pair_rates_short:
            raise ValueError("empty FX cache")

        # Optional: log a bit about the cache
        ts = getattr(raw, "ts", None)
        ts_str = datetime.fromtimestamp(ts).isoformat() if isinstance(ts, (int, float)) else "unknown"
        log_search.info("FX cache loaded: pairs_full=%d, pairs_short=%d, ts=%s",
                        len(fxv.pair_rates_full), len(fxv.pair_rates_short), ts_str)
        return fxv
    except Exception as e:
        raise ValueError(f"Exchange rates not available (failed to load {path}): {e}")


# ---------------------------
# Currency conversion
# ---------------------------
def _canon_full_currency(token: str) -> str:
    if not token:
        raise ValueError("empty currency token")
    t = str(token).strip()
    # Already canonical full
    if t in FULL_TO_SHORT_CURRENCY_MAP:
        return t
    # Short code -> full
    if t in SHORT_TO_FULL_CURRENCY_MAP:
        return SHORT_TO_FULL_CURRENCY_MAP[t]
    # Alias map (lowercased)
    full = CURRENCY_MAP.get(t.lower())
    if full:
        return full
    # Final: case-insensitive match to full names
    low = t.lower()
    for full_name in FULL_TO_SHORT_CURRENCY_MAP.keys():
        if full_name.lower() == low:
            return full_name
    raise ValueError(f"unknown currency: {token!r}")


def _lookup_rate(fx: _FXView, src_full: str, dst_full: str) -> Optional[float]:
    if src_full == dst_full:
        return 1.0
    r = fx.pair_rates_full.get((src_full, dst_full))
    if r:
        return r
    sa = FULL_TO_SHORT_CURRENCY_MAP.get(src_full)
    sb = FULL_TO_SHORT_CURRENCY_MAP.get(dst_full)
    if sa and sb:
        r2 = fx.pair_rates_short.get((sa, sb))
        if r2:
            return r2
    rinv = fx.pair_rates_full.get((dst_full, src_full))
    if rinv and rinv > 0:
        return 1.0 / rinv
    if sa and sb:
        rinv2 = fx.pair_rates_short.get((sb, sa))
        if rinv2 and rinv2 > 0:
            return 1.0 / rinv2
    return None


class PriceConverter:
    """Convert listing prices to a base currency using the FX cache matrix."""
    def __init__(self, fx: _FXView, base_currency: str):
        self.fx = fx
        self.base_full = _canon_full_currency(base_currency)

    def convert(self, amount: Optional[float], currency: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
        if amount is None or currency is None:
            raise ValueError(f"PriceConverter.convert failed; amount={amount}, currency={currency}")
        src_full = _canon_full_currency(currency)
        rate = _lookup_rate(self.fx, src_full, self.base_full)
        if rate and rate > 0:
            return float(amount) * rate, rate
        raise ValueError("Exchange rates not available")


# ---------------------------
# Domain records
# ---------------------------
@dataclass(frozen=True)
class Trade2Price:
    amount_original: Optional[float]
    currency_original: Optional[str]
    ptype: Optional[str]
    amount_in_base: Optional[float]
    currency_in_base: Optional[str]
    rate_to_base: Optional[float]


@dataclass(frozen=True)
class Trade2ListingRecord:
    # Listing / identity (keep these required)
    id: str
    league: Optional[str]
    realm: Optional[str]
    indexed: Optional[str]
    seller: Optional[str]
    price: Trade2Price
    fee: Optional[int]

    # Core item meta (make Optionals default to None, bools default False)
    verified: Optional[bool] = None
    rarity: Optional[str] = None
    base_type: Optional[str] = None
    type_line: Optional[str] = None
    name: Optional[str] = None
    ilvl: Optional[int] = None                 # coalesced later
    identified: Optional[bool] = None
    corrupted: bool = False
    duplicated: bool = False
    unmodifiable: bool = False
    category: Optional[str] = None
    frame_type: Optional[int] = None

    # influence flags + bag
    elder: bool = False
    shaper: bool = False
    searing: bool = False
    tangled: bool = False
    influences: Dict[str, Any] = field(default_factory=dict)

    # Stack / misc
    stack_size: Optional[int] = None
    max_stack_size: Optional[int] = None
    support: Optional[bool] = None

    # PoE2 gems / skills
    gem_sockets: List[str] = field(default_factory=list)
    gem_tabs: List[Dict[str, Any]] = field(default_factory=list)
    gem_background: Optional[str] = None
    gem_skill: Optional[str] = None

    # Text blocks
    sec_descr_text: Optional[str] = None
    descr_text: Optional[str] = None
    flavour_text: List[str] = field(default_factory=list)
    flavour_text_note: Optional[str] = None
    prophecy_text: Optional[str] = None

    # Raw blocks
    sockets: List[Dict[str, Any]] = field(default_factory=list)
    socketed_items: List[Dict[str, Any]] = field(default_factory=list)
    properties: List[Dict[str, Any]] = field(default_factory=list)
    notable_properties: List[Dict[str, Any]] = field(default_factory=list)
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    weapon_requirements: List[Dict[str, Any]] = field(default_factory=list)
    support_gem_requirements: List[Dict[str, Any]] = field(default_factory=list)
    additional_properties: List[Dict[str, Any]] = field(default_factory=list)
    next_level_requirements: List[Dict[str, Any]] = field(default_factory=list)
    granted_skills: List[Dict[str, Any]] = field(default_factory=list)
    extended: Dict[str, Any] = field(default_factory=dict)

    # Mod arrays
    implicit_mods: List[str] = field(default_factory=list)
    explicit_mods: List[str] = field(default_factory=list)
    crafted_mods: List[str] = field(default_factory=list)
    fractured_mods: List[str] = field(default_factory=list)
    crucible_mods: List[str] = field(default_factory=list)
    cosmetic_mods: List[str] = field(default_factory=list)
    veiled_mods: List[str] = field(default_factory=list)
    rune_mods: List[str] = field(default_factory=list)
    desecrated_mods: List[str] = field(default_factory=list)
    desecrated: bool = False

    # Extras (kept for completeness)
    utility_mods: List[str] = field(default_factory=list)
    enchant_mods: List[str] = field(default_factory=list)
    ultimatum_mods: List[Dict[str, Any]] = field(default_factory=list)
    logbook_mods: List[Dict[str, Any]] = field(default_factory=list)
    scourge_mods: List[str] = field(default_factory=list)
    scourged: Dict[str, Any] = field(default_factory=dict)
    crucible: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _category_from_properties(props: List[Dict[str, Any]]) -> Optional[str]:
        if not props:
            return None
        name = (props[0] or {}).get("name")
        if not name:
            return None
        s = str(name)
        return (s[1:-1] if s.startswith('[') and s.endswith(']') else s) if s else None

    @staticmethod
    def _coalesce_ilvl(item: Dict[str, Any]) -> Optional[int]:
        """Prefer `ilvl`; fall back to `itemLevel` (string or int)."""
        v = item.get("ilvl", None)
        if v is None:
            v = item.get("itemLevel", None)
        if v is None:
            return None
        try:
            return int(str(v))
        except Exception:
            return None

    @staticmethod
    def _price(listing_price: Dict[str, Any], converter: Optional["PriceConverter"]) -> "Trade2Price":
        amt = listing_price.get("amount")
        cur_raw = listing_price.get("currency")
        typ = listing_price.get("type")
        try:
            amt_f = float(str(amt)) if amt is not None else None
        except Exception:
            amt_f = None

        base_amt = None
        base_cur = converter.base_full if converter else None
        rate = None
        cur_norm = None

        if cur_raw is not None:
            try:
                cur_norm = _canon_full_currency(cur_raw)
            except Exception:
                cur_norm = str(cur_raw)

        if converter and amt_f is not None and cur_norm:
            base_amt, rate = converter.convert(amt_f, cur_norm)

        return Trade2Price(
            amount_original=amt_f,
            currency_original=cur_norm,
            ptype=typ,
            amount_in_base=base_amt,
            currency_in_base=base_cur,
            rate_to_base=rate,
        )

    @classmethod
    def from_api(cls, listing: Dict[str, Any], converter: Optional["PriceConverter"]) -> "Trade2ListingRecord":
        item = listing.get("item") or {}
        lst = listing.get("listing") or {}
        acc = (lst.get("account") or {})
        props: List[Dict[str, Any]] = item.get("properties") or []

        price_obj = cls._price(lst.get("price") or {}, converter)
        cat = cls._category_from_properties(props)

        infl = item.get("influences") or {}
        frame_type = item.get("frameType")
        elder = bool(item.get("elder", False))
        shaper = bool(item.get("shaper", False))
        searing = bool(item.get("searing", False))
        tangled = bool(item.get("tangled", False))

        return cls(
            id=listing.get("id"),
            league=item.get("league"),
            realm=item.get("realm"),
            indexed=lst.get("indexed"),
            seller=acc.get("name"),
            price=price_obj,
            fee=lst.get("fee"),

            verified=item.get("verified"),
            rarity=item.get("rarity"),
            base_type=item.get("baseType"),
            type_line=item.get("typeLine"),
            name=item.get("name"),
            ilvl=cls._coalesce_ilvl(item),
            identified=item.get("identified"),
            corrupted=bool(item.get("corrupted", False)),
            duplicated=bool(item.get("duplicated", False)),
            unmodifiable=bool(item.get("unmodifiable", False)),
            category=cat,
            frame_type=frame_type,

            elder=elder, shaper=shaper, searing=searing, tangled=tangled,
            influences=infl,

            stack_size=item.get("stackSize"),
            max_stack_size=item.get("maxStackSize"),
            support=item.get("support"),

            gem_sockets=item.get("gemSockets") or [],
            gem_tabs=item.get("gemTabs") or [],
            gem_background=item.get("gemBackground"),
            gem_skill=item.get("gemSkill"),

            sec_descr_text=item.get("secDescrText"),
            descr_text=item.get("descrText"),
            flavour_text=item.get("flavourText") or [],
            flavour_text_note=item.get("flavourTextNote"),
            prophecy_text=item.get("prophecyText"),

            sockets=item.get("sockets") or [],
            socketed_items=item.get("socketedItems") or [],
            properties=props,
            notable_properties=item.get("notableProperties") or [],
            requirements=item.get("requirements") or [],
            weapon_requirements=item.get("weaponRequirements") or [],
            support_gem_requirements=item.get("supportGemRequirements") or [],
            additional_properties=item.get("additionalProperties") or [],
            next_level_requirements=item.get("nextLevelRequirements") or [],
            granted_skills=item.get("grantedSkills") or [],
            extended=item.get("extended") or {},

            implicit_mods=item.get("implicitMods") or [],
            explicit_mods=item.get("explicitMods") or [],
            crafted_mods=item.get("craftedMods") or [],
            fractured_mods=item.get("fracturedMods") or [],
            crucible_mods=item.get("crucibleMods") or [],
            cosmetic_mods=item.get("cosmeticMods") or [],
            veiled_mods=item.get("veiledMods") or [],
            rune_mods=item.get("runeMods") or [],
            desecrated_mods=item.get("desecratedMods") or [],
            desecrated=bool(item.get("desecrated", False)),

            utility_mods=item.get("utilityMods") or [],
            enchant_mods=item.get("enchantMods") or [],
            ultimatum_mods=item.get("ultimatumMods") or [],
            logbook_mods=item.get("logbookMods") or [],
            scourge_mods=item.get("scourgeMods") or [],
            scourged=item.get("scourged") or {},
            crucible=item.get("crucible") or {},
        )

    def to_row(self) -> Dict[str, Any]:
        J = lambda x: json.dumps(x, ensure_ascii=False)
        return {
            "id": self.id,
            "league": self.league,
            "realm": self.realm,
            "indexed": self.indexed,
            "seller": self.seller,
            "fee": self.fee,
            "price_amount_original": self.price.amount_original,
            "price_currency_original": self.price.currency_original,
            "price_type": self.price.ptype,
            "price_amount_in_base": self.price.amount_in_base,
            "price_currency_in_base": self.price.currency_in_base,
            "price_rate_to_base": self.price.rate_to_base,
            "verified": self.verified,
            "rarity": self.rarity,
            "base_type": self.base_type,
            "type_line": self.type_line,
            "name": self.name,
            "ilvl": self.ilvl,
            "identified": self.identified,
            "corrupted": self.corrupted,
            "duplicated": self.duplicated,
            "unmodifiable": self.unmodifiable,
            "category": self.category,
            "frame_type": self.frame_type,
            "elder": self.elder,
            "shaper": self.shaper,
            "searing": self.searing,
            "tangled": self.tangled,
            "influences": J(self.influences),
            "stack_size": self.stack_size,
            "max_stack_size": self.max_stack_size,
            "support": self.support,
            "gem_sockets": J(self.gem_sockets),
            "gem_tabs": J(self.gem_tabs),
            "gem_background": self.gem_background,
            "gem_skill": self.gem_skill,
            "sec_descr_text": self.sec_descr_text,
            "descr_text": self.descr_text,
            "flavour_text": J(self.flavour_text),
            "flavour_text_note": self.flavour_text_note,
            "prophecy_text": self.prophecy_text,
            "sockets": J(self.sockets),
            "socketed_items": J(self.socketed_items),
            "properties": J(self.properties),
            "notable_properties": J(self.notable_properties),
            "requirements": J(self.requirements),
            "weapon_requirements": J(self.weapon_requirements),
            "support_gem_requirements": J(self.support_gem_requirements),
            "additional_properties": J(self.additional_properties),
            "next_level_requirements": J(self.next_level_requirements),
            "granted_skills": J(self.granted_skills),
            "extended": J(self.extended),
            "implicit_mods": J(self.implicit_mods),
            "explicit_mods": J(self.explicit_mods),
            "crafted_mods": J(self.crafted_mods),
            "fractured_mods": J(self.fractured_mods),
            "crucible_mods": J(self.crucible_mods),
            "cosmetic_mods": J(self.cosmetic_mods),
            "veiled_mods": J(self.veiled_mods),
            "rune_mods": J(self.rune_mods),
            "desecrated_mods": J(self.desecrated_mods),
            "desecrated": self.desecrated,
            "utility_mods": J(self.utility_mods),
            "enchant_mods": J(self.enchant_mods),
            "ultimatum_mods": J(self.ultimatum_mods),
            "logbook_mods": J(self.logbook_mods),
            "scourge_mods": J(self.scourge_mods),
            "scourged": J(self.scourged),
            "crucible": J(self.crucible),
        }


# ---------------------------
# Network calls
# ---------------------------
async def post_trade2_search(
    session: aiohttp.ClientSession,
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    polite_pause: float,
) -> Any | None:
    attempt = 0
    consec_429 = 0
    while True:
        attempt += 1
        log_search.info("[1/2] POST search -> %s (attempt %d) | %s", url, attempt, _summarize_payload(payload))
        resp = await session.post(url, json=payload, headers=headers, timeout=timeout)
        async with resp:
            rate_rem = resp.headers.get("X-RateLimit-Remaining")
            rate_used = resp.headers.get("X-RateLimit-Used")
            log_search.info("Search response: status=%s, rate_remaining=%s, rate_used=%s",
                            resp.status, rate_rem, rate_used)
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError as cre:
                if cre.status == 403:
                    body = (await resp.text())[:300]
                    raise RuntimeError(f"403 Forbidden from trade2 search. Check cookies in {COOKIES_PATH}. Body~{body!r}")
                if resp.status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                    ra = _retry_after_seconds(resp)
                    if ra is not None:
                        log_search.warning("Search HTTP %s; Retry-After=%.2fs (attempt %d/%d)",
                                           resp.status, ra, attempt, max_retries)
                        await asyncio.sleep(ra)
                    else:
                        delay = _expo_backoff(attempt)
                        log_search.warning("Search HTTP %s; backoff=%.2fs (attempt %d/%d)",
                                           resp.status, delay, attempt, max_retries)
                        await asyncio.sleep(delay)
                    consec_429 = consec_429 + 1 if resp.status == 429 else 0
                    if consec_429 >= 3:
                        cool = 30.0
                        log_search.warning("Search received %d consecutive 429s; cool down %.0fs", consec_429, cool)
                        await asyncio.sleep(cool)
                        consec_429 = 0
                    if polite_pause:
                        await asyncio.sleep(polite_pause)
                    continue
                body_snip = (await resp.text())[:300]
                raise RuntimeError(f"trade2 search failed status={resp.status} body~{body_snip!r}")
            data = await resp.json(content_type=None)
        if polite_pause:
            await asyncio.sleep(polite_pause)
        return data


async def fetch_listings_one_by_one(
    session: aiohttp.ClientSession,
    *,
    search_id: str,
    ids: List[str],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    polite_pause: float,
    converter: Optional[PriceConverter],
) -> List[Trade2ListingRecord]:
    if not ids:
        return []

    out: List[Trade2ListingRecord] = []
    total = len(ids)
    log_fetch.info("[2/2] Fetching %d listings by id (one-by-one)", total)

    for idx, iid in enumerate(ids, start=1):
        url = TRADE2_FETCH_URL.format(ids=iid, search_id=str(search_id))
        attempt = 0
        while True:
            attempt += 1
            log_fetch.info("[fetch %d/%d] GET %s (attempt %d)", idx, total, url, attempt)
            resp = await session.get(url, headers=headers, timeout=timeout)
            async with resp:
                try:
                    resp.raise_for_status()
                except aiohttp.ClientResponseError as cre:
                    if cre.status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                        ra = _retry_after_seconds(resp)
                        if ra is not None:
                            log_fetch.info("[id=%s] HTTP %s; Retry-After=%.2fs (attempt %d/%d)",
                                           iid, resp.status, ra, attempt, max_retries)
                            await asyncio.sleep(ra)
                        else:
                            delay = _expo_backoff(attempt)
                            log_fetch.info("[id=%s] HTTP %s; backoff=%.2fs (attempt %d/%d)",
                                           iid, resp.status, delay, attempt, max_retries)
                            await asyncio.sleep(delay)
                            if polite_pause:
                                await asyncio.sleep(polite_pause)
                        continue
                    body = (await resp.text())[:300]
                    log_fetch.warning("[id=%s] HTTP %s (abort this id). body~%r", iid, resp.status, body)
                    break

                try:
                    data = await resp.json(content_type=None)
                except Exception as e:
                    log_fetch.info("[id=%s] Invalid JSON: %s", iid, e)
                    break

            if polite_pause:
                await asyncio.sleep(polite_pause)

            results = (data or {}).get("result") or []
            raw = results[0] if results else None
            if not raw or not isinstance(raw, dict):
                log_fetch.info("[id=%s] No result (possibly stale)", iid)
                break

            try:
                rec = Trade2ListingRecord.from_api(raw, converter)
                out.append(rec)
            except Exception as e:
                log_fetch.info("[id=%s] Parse error: %s", iid, e)
            break

    log_fetch.info("Fetch complete: parsed=%d / requested=%d", len(out), total)
    return out


# ---------------------------
# Public workflow
# ---------------------------
def records_to_dataframe(records: List[Trade2ListingRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.to_row() for r in records])


async def _search_async(
    *,
    payload: Dict[str, Any],
    base_currency: str,
    league: Optional[str],
    realm: Optional[str],
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    polite_pause: float = POLITE_PAUSE,
) -> List[Trade2ListingRecord]:
    """
    SINGLE-PAGE search:
      - Respects caller-provided `payload["offset"]` (defaults to 0 if missing).
      - Does not loop or paginate internally.
      - Caps to MAX_RESULTS if set > 0.
    """
    # 0) Resolve league/realm
    if not (league and realm):
        cfg_league, cfg_realm = load_league_realm(CONFIG_PATH)
        league = league or cfg_league
        realm = realm or cfg_realm
    log_search.info(
        "Search start (single page): realm=%s, league=%s, base_currency=%s, MAX_RESULTS=%d",
        realm, league, base_currency, MAX_RESULTS
    )

    # 1) Load FX cache + converter
    fx = load_fx_cache_or_raise(CURRENCY_CACHE_PATH)
    converter = PriceConverter(fx, base_currency)

    # 2) Cookies & headers
    cookies = load_cookies(COOKIES_PATH)
    headers = headers_with_cookies(HEADERS, cookies)
    headers["Referer"] = _build_referer(realm=realm, league=league)

    # 3) Session
    jar = aiohttp.CookieJar(unsafe=True)
    if cookies:
        jar.update_cookies(cookies)

    # 4) Single POST search (respect existing offset; default to 0)
    url = _build_search_url(realm=realm, league=league)
    total_timeout = aiohttp.ClientTimeout(total=timeout)
    payload = dict(payload or {})
    payload.setdefault("offset", 0)

    async with aiohttp.ClientSession(timeout=total_timeout, cookie_jar=jar) as session:
        data = await post_trade2_search(
            session,
            url=url,
            payload=payload,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            polite_pause=polite_pause,
        )

        if not isinstance(data, dict):
            return []
        if "error" in data:
            msg = (data.get("error") or {}).get("message", "unknown error")
            raise RuntimeError(f"trade2 search error: {msg}")

        search_id = data.get("id") or ""
        ids_page = list(data.get("result") or [])
        total_matches = data.get("total")
        complexity = data.get("complexity")

        log_search.info(
            "Search page: offset=%d, got=%d, total=%s, complexity=%s",
            payload.get("offset", 0), len(ids_page), total_matches, complexity
        )

        if not ids_page:
            return []

        # Cap to MAX_RESULTS if configured
        if MAX_RESULTS and MAX_RESULTS > 0 and len(ids_page) > MAX_RESULTS:
            log_search.info("Limiting processed IDs from %d to %d (MAX_RESULTS)", len(ids_page), MAX_RESULTS)
            ids_page = ids_page[:MAX_RESULTS]

        # 5) Fetch listing details for this page
        records = await fetch_listings_one_by_one(
            session,
            search_id=search_id,
            ids=ids_page,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            polite_pause=polite_pause,
            converter=converter,
        )
        failures = len(ids_page) - len(records)
        log_search.info(
            "Single-page fetch finished: success=%d, failed=%d, requested=%d",
            len(records), failures, len(ids_page)
        )
        return records


def search_to_dataframe(
    *,
    payload: Optional[Dict[str, Any]] = None,
    base_currency: str = "Exalted Orb",
    league: Optional[str] = None,
    realm: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    polite_pause: float = POLITE_PAUSE,
    save_csv: Optional[str] = None,
    log_level: str = "INFO",
    # If payload is None, these help build a basic one:
    category_key: Optional[str] = None,
    rarity_key: Optional[str] = None,
    status_option: str = "securable", # instant-buyouts
    sort_key: str = "price",
    sort_dir: str = "asc",
) -> pd.DataFrame:
    """
    Run a Trade2 search (single page) and return a DataFrame of listings
    with prices converted to `base_currency`.
    """
    setup_logging(log_level)

    # Build a payload if not provided
    effective_payload = payload or build_basic_payload(
        category_key=category_key,
        rarity_key=rarity_key,
        status_option=status_option,
        sort_key=sort_key,
        sort_dir=sort_dir,
    )
    if payload is None:
        log_search.info("No payload provided; built basic payload: %s", _summarize_payload(effective_payload))
    else:
        log_search.info("Using provided payload: %s", _summarize_payload(effective_payload))

    async def _run() -> pd.DataFrame:
        records = await _search_async(
            payload=effective_payload,
            base_currency=base_currency,
            league=league,
            realm=realm,
            timeout=timeout,
            max_retries=max_retries,
            polite_pause=polite_pause,
        )
        df = records_to_dataframe(records)
        if save_csv:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv, index=False)
            log_fetch.info("Saved CSV -> %s", save_csv)
        return df

    return asyncio.run(_run())


# ---------------------------
# Helper: basic payload builder (for training scrapes)
# ---------------------------
def build_basic_payload(
    *,
    category_key: Optional[str] = None,
    rarity_key: Optional[str] = None,
    status_option: str = "securable",  # "online" or "any", or "securable" for instant buyouts
    sort_key: str = "price",
    sort_dir: str = "asc",  # "asc"/"desc"
) -> Dict[str, Any]:
    """
    Construct a minimal Trade2 payload focused on category/rarity.
      - category_key: one of ITEM_TYPES keys (e.g., "weapon", "armour.chest", ...)
      - rarity_key: one of ITEM_RARITIES keys (e.g., "unique", "rare", ...)
    """
    if category_key and category_key not in ITEM_TYPES:
        raise ValueError(f"Unknown category key {category_key!r}")
    if rarity_key and rarity_key not in ITEM_RARITIES:
        raise ValueError(f"Unknown rarity key {rarity_key!r}")

    stats = [{"type": "and", "filters": [], "disabled": False}]
    type_filters: Dict[str, Any] = {"filters": {}, "disabled": False}
    misc_filters: Dict[str, Any] = {"filters": {}, "disabled": False}

    if category_key:
        type_filters["filters"]["category"] = {"option": category_key}
    if rarity_key:
        misc_filters["filters"]["rarity"] = {"option": rarity_key}

    filters = {"type_filters": type_filters}
    if misc_filters["filters"]:
        filters["misc_filters"] = misc_filters

    payload = {
        "query": {
            "status": {"option": status_option},
            "stats": stats,
            "filters": filters,
        },
        "sort": {sort_key: sort_dir},
        "offset": 0,  # caller can override this per page
    }
    return payload


# ---------------------------
# __main__ demo
# ---------------------------
if __name__ == "__main__":
    # Example: scrape ANY Weapon (single page at offset 0), convert to Divine Orb prices,
    # using league/realm from config.yaml and cookies from poe_cookies_config.json.
    setup_logging("INFO")
    try:
        df = search_to_dataframe(
            payload=None,                 # use auto-built payload below
            category_key="weapon",        # ignored if payload is provided
            rarity_key=None,
            status_option="securable",
            base_currency="Divine Orb",
            league=None,                  # read from config.yaml
            realm=None,                   # read from config.yaml
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_MAX_RETRIES,
            polite_pause=POLITE_PAUSE,
            save_csv=None,
            log_level="INFO",
        )
        print(df.head())
        print(f"rows={len(df)}")
    except Exception as e:
        log_search.exception("Run failed: %s", e)
        raise
