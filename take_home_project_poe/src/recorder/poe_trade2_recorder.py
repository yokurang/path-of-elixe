import asyncio
import json
import logging
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator

import aiohttp
import pandas as pd
import yaml

from src.research.scripts.utils import setup_logging
from src.recorder.constants import (
    POE_BASE_URL,
    TRADE2_SEARCH_URL,
    TRADE2_FETCH_URL,
    CONFIG_PATH,
    COOKIES_PATH,
    CURRENCY_CACHE_PATH,
    CURRENCY_MAP,
    FULL_TO_SHORT_CURRENCY_MAP,
    SHORT_TO_FULL_CURRENCY_MAP,
    ITEM_TYPES,
    ITEM_RARITIES,
)

from src.recorder.poe_currency_recorder import FXCache

MAX_RESULTS = 150
ITEMS_PER_PAGE = 100
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 5
POLITE_PAUSE = 1.5

HEADERS = {
    "Accept":
        "*/*",
    "Accept-Encoding":
        "gzip, deflate, br, zstd",
    "Accept-Language":
        "en-US,en;q=0.9",
    "Content-Type":
        "application/json",
    "Origin":
        POE_BASE_URL,
    "Priority":
        "u=1, i",
    "Referer":
        POE_BASE_URL,  # overwritten per request
    "Sec-CH-UA":
        "\"Not;A=Brand\";v=\"99\", \"Brave\";v=\"139\", \"Chromium\";v=\"139\"",
    "Sec-CH-UA-Arch":
        "\"x86\"",
    "Sec-CH-UA-Bitness":
        "\"64\"",
    "Sec-CH-UA-Full-Version-List":
        "\"Not;A=Brand\";v=\"99.0.0.0\", \"Brave\";v=\"139.0.0.0\", \"Chromium\";v=\"139.0.0.0\"",
    "Sec-CH-UA-Mobile":
        "?0",
    "Sec-CH-UA-Model":
        "",
    "Sec-CH-UA-Platform":
        "\"Windows\"",
    "Sec-CH-UA-Platform-Version":
        "\"19.0.0\"",
    "Sec-Fetch-Dest":
        "empty",
    "Sec-Fetch-Mode":
        "cors",
    "Sec-Fetch-Site":
        "same-origin",
    "Sec-GPC":
        "1",
    "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "X-Requested-With":
        "XMLHttpRequest",
}

log = logging.getLogger("poe.trade2")
if not log.handlers:
    log.addHandler(logging.NullHandler())


# checks retry-after header when to request again
def _retry_after_seconds(resp: aiohttp.ClientResponse) -> Optional[float]:
    ra_raw = resp.headers.get("Retry-After")
    if ra_raw:
        try:
            return max(0.0, float(ra_raw.strip()))
        except Exception:
            try:
                dt = parsedate_to_datetime(ra_raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(0.0,
                           (dt - datetime.now(timezone.utc)).total_seconds())
            except Exception:
                return None
    return None


def _expo_backoff(attempt: int, base: float = 0.5, cap: float = 12.0) -> float:
    raw = min(cap, base * (2**(attempt - 1)))
    return raw * (1.0 + random.uniform(-0.15, 0.15))


def _build_search_url(realm: str, league: str) -> str:
    from urllib.parse import quote
    return TRADE2_SEARCH_URL.format(realm=realm, league=quote(league, safe=""))


def _build_referer(realm: str, league: str) -> str:
    from urllib.parse import quote
    return f"{POE_BASE_URL}/trade2/search/{realm}/{quote(league, safe='')}"


def load_league_realm(path: Path = CONFIG_PATH) -> Tuple[str, str]:
    league, realm = "Rise of the Abyssal", "poe2"
    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("league") or league), str(cfg.get("realm") or realm)
    except FileNotFoundError:
        return league, realm


def load_cookies(path: Path = COOKIES_PATH) -> Dict[str, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        return {str(k): str(v) for k, v in (raw.get("cookies") or {}).items()}
    except FileNotFoundError:
        return {}


def headers_with_cookies(base: Dict[str, str],
                         cookies: Dict[str, str]) -> Dict[str, str]:
    h = dict(base)
    parts = []
    if cookies.get("POESESSID"):
        parts.append(f"POESESSID={cookies['POESESSID']}")
    if cookies.get("cf_clearance"):
        parts.append(f"cf_clearance={cookies['cf_clearance']}")
    if parts:
        h["Cookie"] = "; ".join(parts)
    return h


# FX + Conversion
# snapshot of all currency FX exchange data
@dataclass(frozen=True)
class _FXView:
    pair_rates_full: Dict[Tuple[str, str], float]
    pair_rates_short: Dict[Tuple[str, str], float]
    full_map: Dict[str, str]
    short_map: Dict[str, str]


def _extract_fx_view(obj: Any) -> _FXView:
    return _FXView(
        pair_rates_full=dict(getattr(obj, "pair_rates_full", {})),
        pair_rates_short=dict(getattr(obj, "pair_rates_short", {})),
        full_map=dict(getattr(obj, "full_map", {})),
        short_map=dict(getattr(obj, "short_map", {})),
    )


def load_fx_cache_or_raise(path: Path = CURRENCY_CACHE_PATH) -> _FXView:
    with path.open("rb") as f:
        raw = pickle.load(f)
    fxv = _extract_fx_view(raw)
    if not fxv.pair_rates_full and not fxv.pair_rates_short:
        raise ValueError("empty FX cache")
    return fxv


def _canon_full_currency(token: str) -> str:
    if token in FULL_TO_SHORT_CURRENCY_MAP:
        return token
    if token in SHORT_TO_FULL_CURRENCY_MAP:
        return SHORT_TO_FULL_CURRENCY_MAP[token]
    full = CURRENCY_MAP.get(token.lower())
    if full:
        return full
    for full_name in FULL_TO_SHORT_CURRENCY_MAP.keys():
        if full_name.lower() == token.lower():
            return full_name
    raise ValueError(f"unknown currency: {token!r}")


def _lookup_rate(fx: _FXView, src: str, dst: str) -> Optional[float]:
    if src == dst:
        return 1.0
    return (fx.pair_rates_full.get((src, dst)) or fx.pair_rates_short.get(
        (FULL_TO_SHORT_CURRENCY_MAP.get(src),
         FULL_TO_SHORT_CURRENCY_MAP.get(dst))) or
            (1.0 / fx.pair_rates_full[(dst, src)]) if
            (dst, src) in fx.pair_rates_full else None)


class PriceConverter:

    def __init__(self, fx: _FXView, base_currency: str):
        self.fx = fx
        self.base_full = _canon_full_currency(base_currency)

    def convert(self, amount: float, currency: str) -> Tuple[float, float]:
        src = _canon_full_currency(currency)
        rate = _lookup_rate(self.fx, src, self.base_full)
        if not rate:
            raise ValueError(
                f"Conversion rate not found: cannot convert from '{src}' to '{self.base_full}'"
            )
        return amount * rate, rate


# Data Models


# source is from https://www.pathofexile.com/trade2/search/poe2/Rise%20of%20the%20Abyssal API response
@dataclass(frozen=True)
class Trade2Price:
    amount_original: Optional[float]
    currency_original: Optional[str]
    ptype: Optional[str]
    amount_in_base: Optional[float]
    currency_in_base: Optional[str]
    rate_to_base: Optional[float]


# source: https://www.pathofexile.com/developer/docs/reference#type-Item
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
    ilvl: Optional[int] = None  # coalesced later
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

    # from inspection, first name element of properties columns seems to be the item category
    @staticmethod
    def _category_from_properties(props: List[Dict[str, Any]]) -> Optional[str]:
        if not props:
            return None
        name = (props[0] or {}).get("name")
        if not name:
            return None
        s = str(name)
        return (s[1:-1]
                if s.startswith('[') and s.endswith(']') else s) if s else None

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
    def _price(listing_price: Dict[str, Any],
               converter: Optional["PriceConverter"]) -> "Trade2Price":
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
    def from_api(
            cls, listing: Dict[str, Any],
            converter: Optional["PriceConverter"]) -> "Trade2ListingRecord":
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
            elder=elder,
            shaper=shaper,
            searing=searing,
            tangled=tangled,
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


# Streaming
async def stream_search_results(
    *,
    payload: Dict[str, Any],
    base_currency: str,
    max_results: int,
    league: Optional[str] = None,
    realm: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    polite_pause: float = POLITE_PAUSE,
) -> AsyncGenerator[Trade2ListingRecord, None]:
    # here we try to yield as soon as each record is parsed to avoid large RAM spikes (my poor comp)
    if not (league and realm):
        league, realm = load_league_realm(CONFIG_PATH)
    log.info(
        "=== Starting streaming search for league=%s, realm=%s, target=%d ===",
        league, realm, max_results)

    fx = load_fx_cache_or_raise(CURRENCY_CACHE_PATH)
    converter = PriceConverter(fx, base_currency)

    cookies = load_cookies(COOKIES_PATH)
    headers = headers_with_cookies(HEADERS, cookies)
    headers["Referer"] = _build_referer(realm, league)
    jar = aiohttp.CookieJar(unsafe=True)
    if cookies:
        jar.update_cookies(cookies)

    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            cookie_jar=jar,
    ) as session:
        url = _build_search_url(realm, league)
        offset, seen, collected, page_num = 0, set(), 0, 1

        while collected < max_results:
            log.info("[Page %d] Requesting offset=%d (collected=%d/%d)",
                     page_num, offset, collected, max_results)

            data = None
            for attempt in range(1, max_retries + 1):  # retry logic
                try:
                    resp = await session.post(url,
                                              json={
                                                  **payload, "offset": offset
                                              },
                                              headers=headers,
                                              timeout=timeout)
                    async with resp:
                        if resp.status == 429:  # rate limit
                            ra = _retry_after_seconds(resp)
                            delay = ra if ra is not None else _expo_backoff(
                                attempt)
                            log.warning(
                                "[Page %d] Rate limited on page fetch, waiting %.2fs",
                                page_num, delay)
                            await asyncio.sleep(delay)
                            continue
                        resp.raise_for_status()
                        data = await resp.json(content_type=None)
                    break
                except Exception as e:
                    delay = _expo_backoff(attempt)
                    log.warning(
                        "[Page %d] Error fetching page offset=%d attempt %d: %s. Retry in %.2fs",
                        page_num, offset, attempt, e, delay)
                    await asyncio.sleep(delay)
            else:
                log.error("[Page %d] All retries failed for page offset=%d",
                          page_num, offset)
                break

            ids = list((data or {}).get("result") or [])
            log.info("[Page %d] Returned %d IDs", page_num, len(ids))
            if not ids:
                # Short/empty page is a natural end-of-results signal
                break

            page_success, page_fail = 0, 0
            # Fetch-and-yield each listing immediately for streaming behavior
            for iid in ids:
                if iid in seen:
                    # Avoid re-fetching duplicates (server-side overlap can happen)
                    log.debug("[Page %d] Skipping duplicate ID=%s", page_num,
                              iid)
                    continue
                seen.add(iid)

                rec = await _fetch_single_listing(
                    session=session,
                    iid=iid,
                    search_id=data.get(
                        "id"),  # tie fetches to the search session
                    headers=headers,
                    timeout=timeout,
                    max_retries=max_retries,
                    polite_pause=polite_pause,
                    converter=converter,
                    page_num=page_num,
                )
                if rec:
                    collected += 1
                    page_success += 1
                    log.info("[Page %d] Parsed ID=%s (collected %d/%d)",
                             page_num, iid, collected, max_results)
                    # yield early for real-time consumption
                    yield rec
                    if collected >= max_results:  # reached desired limit
                        break
                else:
                    page_fail += 1

            log.info(
                "[Page %d] Summary: success=%d, failed=%d, total_collected=%d/%d",
                page_num, page_success, page_fail, collected, max_results)

            # Stop if server gave us a short page (end) or we reached target
            if len(ids) < ITEMS_PER_PAGE or collected >= max_results:
                log.info("[Page %d] Final page reached (collected=%d/%d)",
                         page_num, collected, max_results)
                break

            # Advance by the server’s fixed page size (explicit, predictable)
            offset += ITEMS_PER_PAGE
            page_num += 1

    log.info("=== Completed streaming search. Total collected=%d/%d ===",
             collected, max_results)


async def _fetch_single_listing(
    session: aiohttp.ClientSession,
    iid: str,
    search_id: str,
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    polite_pause: float,
    converter: PriceConverter,
    page_num: int,
) -> Optional[Trade2ListingRecord]:
    # Use the Trade2 fetch endpoint; this variant POSTs the search_id in JSON
    url = TRADE2_FETCH_URL.format(ids=iid)
    payload = {"query": search_id}

    # Per-ID retry loop (isolated so a single bad ID doesn’t kill the page)
    for attempt in range(1, max_retries + 1):
        try:
            log.debug("[Page %d] Fetching ID=%s (attempt %d)", page_num, iid,
                      attempt)
            resp = await session.post(url,
                                      json=payload,
                                      headers=headers,
                                      timeout=timeout)
            async with resp:
                if resp.status == 429:  # rate limit
                    ra = _retry_after_seconds(resp)
                    delay = ra if ra is not None else _expo_backoff(attempt)
                    log.warning(
                        "[Page %d] Rate limited on ID=%s, waiting %.2fs",
                        page_num, iid, delay)
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            break
        except Exception as e:
            delay = _expo_backoff(attempt)
            log.warning(
                "[Page %d] Error fetching ID=%s attempt %d: %s. "
                "Retry in %.2fs", page_num, iid, attempt, e, delay)
            await asyncio.sleep(delay)
    else:
        # Give up this ID after max_retries; continue with others
        log.error("[Page %d] All retries failed for ID=%s", page_num, iid)
        return None

    if polite_pause:
        await asyncio.sleep(polite_pause)

    results = (data or {}).get("result") or []
    if not results:
        # Robust to stale/missing IDs (can happen if listings are removed quickly)
        log.warning("[Page %d] Empty result block for ID=%s", page_num, iid)
        return None

    try:
        # Parse into a stable record shape; price conversion happens here
        return Trade2ListingRecord.from_api(results[0], converter)
    except Exception as e:
        # Defensive parse guard: do not crash the stream on a single malformed item
        log.error("[Page %d] Parse error for ID=%s: %s", page_num, iid, e)
        return None


# Public API (streaming)
def records_to_dataframe(records: List[Trade2ListingRecord]) -> pd.DataFrame:
    # Simple helper to collect into a tabular artifact
    return pd.DataFrame([r.to_row() for r in records])


def search_to_dataframe_with_limit(
    payload: Dict[str, Any],
    base_currency: str = "Exalted Orb",
    max_results: int = MAX_RESULTS,
    league: Optional[str] = None,
    realm: Optional[str] = None,
    log_level: str = "INFO",
) -> Tuple[pd.DataFrame, int]:
    """
    Streaming wrapper: collect results into a DataFrame.
    Returns (DataFrame, count_of_records).
    """

    async def _run():
        rows = []
        async for rec in stream_search_results(payload=payload,
                                               base_currency=base_currency,
                                               max_results=max_results,
                                               league=league,
                                               realm=realm):
            rows.append(rec.to_row())
        return pd.DataFrame(rows), len(rows)

    return asyncio.run(_run())


def search_to_dataframe(**kwargs) -> pd.DataFrame:
    df, _ = search_to_dataframe_with_limit(max_results=ITEMS_PER_PAGE, **kwargs)
    return df


def build_basic_payload(category_key=None,
                        rarity_key=None,
                        status_option="securable",
                        sort_key="price",
                        sort_dir="asc"):
    # build basic and overwrite if provided arguments
    if category_key and category_key not in ITEM_TYPES:
        raise ValueError(f"Unknown category {category_key}")
    if rarity_key and rarity_key not in ITEM_RARITIES:
        raise ValueError(f"Unknown rarity {rarity_key}")
    filters = {"type_filters": {"filters": {}, "disabled": False}}
    if category_key:
        filters["type_filters"]["filters"]["category"] = {
            "option": category_key
        }
    if rarity_key:
        filters["misc_filters"] = {
            "filters": {
                "rarity": {
                    "option": rarity_key
                }
            },
            "disabled": False,
        }
    return {
        "query": {
            "status": {
                "option": status_option
            },
            "stats": [{
                "type": "and",
                "filters": []
            }],
            "filters": filters,
        },
        "sort": {
            sort_key: sort_dir
        },
        "offset": 0,
    }


# (demo + tests)
if __name__ == "__main__":
    setup_logging("INFO")

    # === Test: Paginated streaming search ===
    try:
        payload = build_basic_payload(category_key="weapon.wand",
                                      rarity_key="rare")
        df, count = search_to_dataframe_with_limit(payload,
                                                   base_currency="Divine Orb",
                                                   max_results=150,
                                                   league="Standard")
        print("\n=== Test: Paginated Streaming Search ===")
        print(f"Fetched {count} rows")
        print(df.head())
    except Exception as e:
        log.exception("Streaming search failed: %s", e)

    # === Test: Single ID fetch into Trade2ListingRecord ===
    try:
        print("\n=== Test: Single Listing Fetch ===")

        test_id = "bca84485f9c09b118b2c8b8c6fa478a1d104ec63b95cad3310325b0c9ee89ab1"
        test_search_id = "LW0dQv7in"  # must come from a real search response!

        async def _test_single():
            league, realm = load_league_realm()
            fx = load_fx_cache_or_raise(CURRENCY_CACHE_PATH)
            converter = PriceConverter(fx, "Divine Orb")
            cookies = load_cookies(COOKIES_PATH)
            headers = headers_with_cookies(HEADERS, cookies)
            headers["Referer"] = _build_referer(realm, league)
            jar = aiohttp.CookieJar(unsafe=True)
            if cookies:
                jar.update_cookies(cookies)

            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT),
                    cookie_jar=jar,
            ) as session:
                rec = await _fetch_single_listing(
                    session=session,
                    iid=test_id,
                    search_id=test_search_id,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT,
                    max_retries=DEFAULT_MAX_RETRIES,
                    polite_pause=POLITE_PAUSE,
                    converter=converter,
                )
                return rec

        record = asyncio.run(_test_single())
        if record:
            print("Fetched single record successfully:")
            print(json.dumps(record.to_row(), indent=2, ensure_ascii=False))
        else:
            print("Single record fetch failed or returned None.")
    except Exception as e:
        log.exception("Single record test failed: %s", e)
