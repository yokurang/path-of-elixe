import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

from main import setup_logging

from take_home_project_poe.misc.poe2_currency_recorder_web_scrape import (
    get_currency_cache,
    get_rate as get_fx_rate,
    Server,
)

from src.recorder.constants import CURRENCY_MAP

import aiohttp
import pandas as pd
import yaml

POE_BASE_URL = "https://www.pathofexile.com"
TRADE2_SEARCH_URL = POE_BASE_URL + "/api/trade2/search/{realm}/{league}"
TRADE2_FETCH_URL = POE_BASE_URL + "/api/trade2/fetch/{ids}?query={search_id}"
CONFIG_PATH = "config.yaml"
COOKIES_PATH = "poe_cookies_config.json"

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
        "https://www.pathofexile.com",
    "Priority":
        "u=1, i",
    "Referer":
        "https://www.pathofexile.com",  # will be overridden per search
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

log_search = logging.getLogger("poe.trade2.search")
log_search.addHandler(logging.NullHandler())
log_fetch = logging.getLogger("poe.trade")
log_fetch.addHandler(logging.NullHandler())

# price conversion class

class PriceConverter:
    """
    Converts listing prices to a base currency using the cached currency matrix.
    Applies a single .get() mapping from Trade API codes/labels to AOEAH canonical names.
    """
    def __init__(self, cache, server: Server, base_currency: str):
        self.cache = cache
        self.server = server
        # Apply the same mapping to base currency (in case someone passes "ex" etc.)
        self.base_currency = CURRENCY_MAP.get(
            str(base_currency).strip().lower(),
            base_currency,
        )

    def convert(self, amount: Optional[float], currency: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
        """
        Returns (amount_in_base, rate_to_base). If conversion fails, both are None.
        """
        if amount is None or not currency:
            raise ValueError(f"PriceConverter.convert failed; amount={amount}, currency={currency}")

        src = CURRENCY_MAP.get(str(currency).strip().lower(), currency)
        dst = self.base_currency

        if src == dst:
            return amount, 1.0

        rate = get_fx_rate(self.cache, self.server, src, dst)
        if rate and rate > 0:
            converted = amount * rate
            log_fetch.debug("FX convert: %.6f %s -> %.6f %s (rate=%.6f, server=%s)",
                            amount, src, converted, dst, rate, self.server.value)
            return converted, rate

        log_fetch.debug("FX missing: cannot convert %s -> %s (server=%s)", src, dst, self.server.value)
        raise ValueError("FX missing: cannot convert %s -> %s (server=%s)", src, dst, self.server.value)

# helper functions
def build_search_url(*, realm: str, league: str) -> str:
    from urllib.parse import quote
    return TRADE2_SEARCH_URL.format(realm=realm, league=quote(league, safe=""))

def referer_for(*, realm: str, league: str) -> str:
    from urllib.parse import quote
    return f"https://www.pathofexile.com/trade2/search/{realm}/{quote(league, safe='')}"

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

def _masked_cfg_for_log(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if "poe_sessid" in out:
        out["poe_sessid"] = out["poe_sessid"]
    if "cf_clearance" in out:
        out["cf_clearance"] = out["cf_clearance"]
    return out

# load configs from yaml file
def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    log_search.info("Loading config from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    log_search.info("Config loaded (masked): %s", json.dumps(_masked_cfg_for_log(cfg), ensure_ascii=False, sort_keys=True))
    return cfg

def headers_from_config(cfg: Dict[str, Any]) -> Dict[str, str]:
    log_search.info("Building headers from config (cookies masked in logs)")
    h = dict(HEADERS)
    cookie_parts = []
    if cfg.get("poe_sessid"):
        cookie_parts.append(f"POESESSID={cfg['poe_sessid']}")
    if cfg.get("cf_clearance"):
        cookie_parts.append(f"cf_clearance={cfg['cf_clearance']}")
    if cookie_parts:
        h["Cookie"] = "; ".join(cookie_parts)
        log_search.info("Cookie present: POESESSID=%s, cf_clearance=%s",
                        cfg.get("poe_sessid"),
                        cfg.get("cf_clearance"))
    else:
        log_search.info("No Cookie set (anonymous/unauthenticated)")
    return h

def payload_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    log_search.info("Constructing search payload from config")
    status_obj = cfg.get("status", {"option": "online"})
    if isinstance(status_obj, str):
        status_obj = {"option": status_obj}
    stats_list = cfg.get("stats") or [{"type": "and", "filters": [], "disabled": False}]
    filters_obj = cfg.get("filters") or {"type_filters": {"filters": {}, "disabled": False}}
    if "type_filters" not in filters_obj:
        filters_obj = {"type_filters": {"filters": {}, "disabled": False}}
    sort_obj = cfg.get("sort") or {"price": "asc"}

    payload = {"query": {"status": status_obj, "stats": stats_list, "filters": filters_obj}, "sort": sort_obj}
    q = payload["query"]
    log_search.info("Payload summary: status=%s sort=%s stats_blocks=%d has_type_filters=%s",
                    q.get("status"), payload.get("sort"), len(q.get("stats") or []),
                    "type_filters" in (q.get("filters") or {}))
    return payload

def _parse_server(name: Optional[str]) -> Server:
    if not name:
        return Server.STANDARD

    for s in Server:
        if name == s.value:
            return s

    raise ValueError(
        f"Unknown server {name!r}. Valid values: "
        + ", ".join(repr(s.value) for s in Server)
    )

def options_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    opts = {
        "realm": cfg.get("realm", "poe2"),
        "league": cfg.get("league", "Rise of the Abyssal"),
        "timeout": cfg.get("timeout", 30),
        "max_retries": cfg.get("max_retries", 5),
        "post_request_pause": float(cfg.get("post_request_pause", 1.0)),
        # base currency + server for conversion
        "base_currency": cfg.get("base_currency", "Exalted Orb"),
        # Keep Server.STANDARD exactly "Standard" so it matches Trade search
        "base_server": _parse_server(cfg.get("base_server", "Standard")),
    }
    log_search.info("Resolved options: %s", json.dumps({**opts, "base_server": opts["base_server"].value}, ensure_ascii=False, sort_keys=True))
    return opts

@dataclass(frozen=True)
class Trade2SearchOutcome:
    ok: bool
    search_id: Optional[str]
    ids: List[str]
    total: Optional[int]
    complexity: Optional[int]
    truncated: bool
    message: Optional[str]

def _process_search_json(data: Dict[str, Any]) -> Trade2SearchOutcome:
    log_search.debug("Processing search JSON")
    if isinstance(data, dict) and "error" in data:
        err = data.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else None
        if msg and "complex" in str(msg).lower():
            log_search.error("Query rejected as too complex: %s", msg)
        return Trade2SearchOutcome(False, None, [], None, None, False, str(msg) if msg is not None else None)

    search_id = data.get("id")
    ids = data.get("result", []) or []
    total = data.get("total")
    complexity = data.get("complexity")
    truncated = bool(total is not None and total >= 10_000)
    if truncated:
        log_search.warning("Search returned >=10,000 matches (truncated); consider narrowing filters")
    log_search.info("Parsed search outcome: search_id=%s ids=%d total=%s complexity=%s truncated=%s",
                    search_id, len(ids), total, complexity, truncated)
    return Trade2SearchOutcome(True, search_id, ids, total, complexity, truncated, None)

async def post_trade2_search(
    session: aiohttp.ClientSession,
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    post_request_pause: float,
) -> Trade2SearchOutcome:
    attempt = 0
    consec_429 = 0
    while True:
        attempt += 1
        log_search.info("POST %s (attempt %d)", url, attempt)
        log_search.debug("Request payload: %s", json.dumps(payload, ensure_ascii=False))
        resp = await session.post(url, json=payload, headers=headers, timeout=timeout)
        async with resp:
            rate_rem = resp.headers.get("X-RateLimit-Remaining")
            rate_used = resp.headers.get("X-RateLimit-Used")
            log_search.info("HTTP status=%s, X-RateLimit-Remaining=%s, X-RateLimit-Used=%s",
                            resp.status, rate_rem, rate_used)
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError:
                if resp.status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                    log_search.warning(f"Response Header from POE2 trade2={resp.headers}")
                    ra = _retry_after_seconds(resp)
                    ra_hdr = resp.headers.get("Retry-After")
                    if ra is not None:
                        log_search.warning("search HTTP %s; Retry-After header=%r -> waiting %.2fs (attempt %d/%d)",
                                           resp.status, ra_hdr, ra, attempt, max_retries)
                        await asyncio.sleep(ra)
                    else:
                        delay = _expo_backoff(attempt)
                        log_search.warning("search HTTP %s; Retry-After missing; backoff=%.2fs (attempt %d/%d)",
                                           resp.status, delay, attempt, max_retries)
                        await asyncio.sleep(delay)
                        await asyncio.sleep(post_request_pause)
                    consec_429 = consec_429 + 1 if resp.status == 429 else 0
                    if consec_429 >= 3:
                        cool = 30.0
                        log_search.warning("search received %d consecutive 429s; cooling down for %.0fs", consec_429, cool)
                        await asyncio.sleep(cool)
                        consec_429 = 0
                    continue

                body_snip = (await resp.text())[:300]
                log_search.error("search non-retriable or exhausted; status=%s, body~%r", resp.status, body_snip)
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=f"{resp.reason}: {body_snip}",
                    headers=resp.headers,
                )

            data = await resp.json(content_type=None)
            log_search.debug("search response JSON received")

        if post_request_pause:
            log_search.debug("Polite pause after request: %.2fs", post_request_pause)
            await asyncio.sleep(post_request_pause)
        return _process_search_json(data)

@dataclass(frozen=True)
class Trade2Price:
    # original listing price
    amount_original: Optional[float]
    currency_original: Optional[str]
    ptype: Optional[str]
    # converted to base currency
    amount_in_base: Optional[float]
    currency_in_base: Optional[str]
    rate_to_base: Optional[float] # fx used for original->base
    

@dataclass(frozen=True)
class Trade2ListingRecord:
    # Listing / identity
    id: str
    league: Optional[str]
    realm: Optional[str]
    indexed: Optional[str]
    seller: Optional[str]
    price: Trade2Price
    fee: Optional[int] # from listing.fee

    # Item meta
    verified: Optional[bool]
    rarity: Optional[str]
    base_type: Optional[str]
    type_line: Optional[str]
    name: Optional[str]
    ilvl: Optional[int]
    identified: Optional[bool]
    corrupted: bool
    duplicated: bool
    unmodifiable: bool
    category: Optional[str] # derived from properties[0].name

    # Stack info
    stack_size: Optional[int]
    max_stack_size: Optional[int]

    # Gem-related extras
    support: Optional[bool]
    gem_sockets: List[str]
    weapon_requirements: List[Dict[str, Any]]
    gem_tabs: List[Dict[str, Any]]
    gem_background: Optional[str]
    gem_skill: Optional[str]
    sec_descr_text: Optional[str]
    flavour_text: List[str]
    descr_text: Optional[str]

    # Raw blocks
    sockets: List[Dict[str, Any]]
    properties: List[Dict[str, Any]]
    requirements: List[Dict[str, Any]]
    granted_skills: List[Dict[str, Any]]
    socketed_items: List[Dict[str, Any]]
    extended: Dict[str, Any]

    # Mods & flags
    rune_mods: List[str]
    desecrated_mods: List[str]
    implicit_mods: List[str]
    explicit_mods: List[str]
    veiled_mods: List[str]
    desecrated: bool


    @staticmethod
    def _price(p: Dict[str, Any], converter: Optional["PriceConverter"]) -> "Trade2Price":
        amt = p.get("amount")
        cur_raw = p.get("currency")
        typ = p.get("type")
        try:
            amt_f = float(str(amt)) if amt is not None else None
        except Exception:
            amt_f = None

        base_amt = None
        rate = None
        base_cur = converter.base_currency if converter else None
        cur = (CURRENCY_MAP.get(str(cur_raw).strip().lower(), cur_raw)
               if converter else cur_raw)

        if converter and amt_f is not None and cur:
            base_amt, rate = converter.convert(amt_f, cur)

        return Trade2Price(
            amount_original=amt_f,
            currency_original=cur,
            ptype=typ,
            amount_in_base=base_amt,
            currency_in_base=base_cur,
            rate_to_base=rate,
        )

    @staticmethod
    def _category_from_properties(props: List[Dict[str, Any]]) -> Optional[str]:
        """
        Use the first property's name as the category.
        Examples:
          {"name": "[Crossbow]"} -> "Crossbow"
          {"name": "[Evasion|Evasion Rating]"} -> "Evasion"
          {"name": "Waystone"} -> "Waystone"
        """
        if not props:
            return None
        raw = (props[0] or {}).get("name", "")
        cat = re.sub(r"[\[\]]", "", str(raw)).split("|")[0].strip()
        return cat or None

    @classmethod
    def from_api(cls, listing: Dict[str, Any], converter: Optional["PriceConverter"]) -> "Trade2ListingRecord":
        item = listing.get("item") or {}
        lst = listing.get("listing") or {}
        acc = (lst.get("account") or {})

        price_obj = cls._price(lst.get("price") or {}, converter)

        props: List[Dict[str, Any]] = item.get("properties") or []
        category = cls._category_from_properties(props)

        return cls(
            # listing/identity
            id=listing.get("id"),
            league=item.get("league"),
            realm=item.get("realm"),
            indexed=lst.get("indexed"),
            seller=acc.get("name"),
            price=price_obj,
            fee=lst.get("fee"),

            # meta
            verified=item.get("verified"),
            rarity=item.get("rarity"),
            base_type=item.get("baseType"),
            type_line=item.get("typeLine"),
            name=item.get("name"),
            ilvl=item.get("ilvl"),
            identified=item.get("identified"),
            corrupted=bool(item.get("corrupted", False)),
            duplicated=bool(item.get("duplicated", False)),
            unmodifiable=bool(item.get("unmodifiable", False)),
            category=category,

            # stack info
            stack_size=item.get("stackSize"),
            max_stack_size=item.get("maxStackSize"),

            # gem-related
            support=item.get("support"),
            gem_sockets=item.get("gemSockets") or [],
            weapon_requirements=item.get("weaponRequirements") or [],
            gem_tabs=item.get("gemTabs") or [],
            gem_background=item.get("gemBackground"),
            gem_skill=item.get("gemSkill"),
            sec_descr_text=item.get("secDescrText"),
            flavour_text=item.get("flavourText") or [],
            descr_text=item.get("descrText"),

            # raw blocks
            sockets=item.get("sockets") or [],
            properties=props,
            requirements=item.get("requirements") or [],
            granted_skills=item.get("grantedSkills") or [],
            socketed_items=item.get("socketedItems") or [],
            extended=item.get("extended") or {},

            # mods & flags
            rune_mods=item.get("runeMods") or [],
            desecrated_mods=item.get("desecratedMods") or [],
            implicit_mods=item.get("implicitMods") or [],
            explicit_mods=item.get("explicitMods") or [],
            veiled_mods=item.get("veiledMods") or [],
            desecrated=bool(item.get("desecrated", False)),
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "league": self.league,
            "realm": self.realm,
            "indexed": self.indexed,
            "seller": self.seller,
            # listing extras
            "fee": self.fee,
            # price
            "price_amount_original": self.price.amount_original,
            "price_currency_original": self.price.currency_original,
            "price_type": self.price.ptype,
            "price_amount_in_base": self.price.amount_in_base,
            "price_currency_in_base": self.price.currency_in_base,
            "price_rate_to_base": self.price.rate_to_base,
            # meta
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
            # stack
            "stack_size": self.stack_size,
            "max_stack_size": self.max_stack_size,
            # gem-related
            "support": self.support,
            "gem_sockets": json.dumps(self.gem_sockets, ensure_ascii=False),
            "weapon_requirements": json.dumps(self.weapon_requirements, ensure_ascii=False),
            "gem_tabs": json.dumps(self.gem_tabs, ensure_ascii=False),
            "gem_background": self.gem_background,
            "gem_skill": self.gem_skill,
            "sec_descr_text": self.sec_descr_text,
            "flavour_text": json.dumps(self.flavour_text, ensure_ascii=False),
            "descr_text": self.descr_text,
            # raw blocks
            "sockets": json.dumps(self.sockets, ensure_ascii=False),
            "properties": json.dumps(self.properties, ensure_ascii=False),
            "requirements": json.dumps(self.requirements, ensure_ascii=False),
            "granted_skills": json.dumps(self.granted_skills, ensure_ascii=False),
            "socketed_items": json.dumps(self.socketed_items, ensure_ascii=False),
            "extended": json.dumps(self.extended, ensure_ascii=False),
            # mods & flags
            "rune_mods": json.dumps(self.rune_mods, ensure_ascii=False),
            "desecrated_mods": json.dumps(self.desecrated_mods, ensure_ascii=False),
            "implicit_mods": json.dumps(self.implicit_mods, ensure_ascii=False),
            "explicit_mods": json.dumps(self.explicit_mods, ensure_ascii=False),
            "veiled_mods": json.dumps(self.veiled_mods, ensure_ascii=False),
            "desecrated": self.desecrated,
        }


async def fetch_listings_batch(
    session: aiohttp.ClientSession,
    search_id: str,
    ids: List[str],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    post_request_pause: float,
    converter: Optional[PriceConverter],
) -> List[Trade2ListingRecord]:
    """
    Fetch listings ONE BY ONE to avoid 'Invalid query' from overly long id lists.
    Includes FX conversion to base currency.
    """
    if not ids:
        log_fetch.info("No ids to fetch; returning empty list")
        return []

    out: List[Trade2ListingRecord] = []
    for idx, iid in enumerate(ids, start=1):
        url = TRADE2_FETCH_URL.format(ids=iid, search_id=search_id)
        attempt = 0
        while True:
            attempt += 1
            log_fetch.info("[id=%s %d/%d] GET %s (attempt %d)", iid, idx, len(ids), url, attempt)
            resp = await session.get(url, headers=headers, timeout=timeout)
            async with resp:
                rate_rem = resp.headers.get("X-RateLimit-Remaining")
                rate_used = resp.headers.get("X-RateLimit-Used")
                ra_hdr = resp.headers.get("Retry-After")
                log_fetch.info("[id=%s] HTTP status=%s, X-RateLimit-Remaining=%s, X-RateLimit-Used=%s, Retry-After=%r",
                               iid, resp.status, rate_rem, rate_used, ra_hdr)
                try:
                    resp.raise_for_status()
                except aiohttp.ClientResponseError:
                    if resp.status in (429, 500, 502, 503, 504) and attempt <= max_retries:
                        log_search.warning(f"Response Header from POE2 trade2={resp.headers}")
                        ra = _retry_after_seconds(resp)
                        if ra is not None:
                            log_fetch.warning("[id=%s] HTTP %s; Retry-After -> wait %.2fs (attempt %d/%d)",
                                              iid, resp.status, ra, attempt, max_retries)
                            await asyncio.sleep(ra)
                        else:
                            delay = _expo_backoff(attempt)
                            log_fetch.warning("[id=%s] HTTP %s; Retry-After missing; backoff=%.2fs (attempt %d/%d)",
                                              iid, resp.status, delay, attempt, max_retries)
                            await asyncio.sleep(delay)
                            if post_request_pause:
                                await asyncio.sleep(post_request_pause)
                        continue

                    body = (await resp.text())
                    log_fetch.error("[id=%s] FAIL: HTTP %s %r", iid, resp.status, body)
                    break

                try:
                    data = await resp.json(content_type=None)
                except Exception as e:
                    log_fetch.error("[id=%s] FAIL: invalid JSON: %s", iid, e)
                    if post_request_pause:
                        await asyncio.sleep(post_request_pause)
                    break

            if post_request_pause:
                log_fetch.debug("[id=%s] Polite pause after request: %.2fs", iid, post_request_pause)
                await asyncio.sleep(post_request_pause)

            results = (data or {}).get("result") or []
            raw = results[0] if results else None
            if not raw or not isinstance(raw, dict):
                log_fetch.info("[id=%s] FAIL: no result (possibly stale)", iid)
                break

            try:
                rec = Trade2ListingRecord.from_api(raw, converter)
                log_fetch.info(
                    "[id=%s] SUCCESS: seller=%s price_raw=%s %s price_base=%s %s name=%s base=%s ilvl=%s",
                    rec.id,
                    rec.seller,
                    rec.price.amount_original,
                    rec.price.currency_original,
                    rec.price.amount_in_base,
                    rec.price.currency_in_base,
                    (rec.name or rec.type_line or "").strip(),
                    (rec.base_type).strip() or "",
                    rec.ilvl,
                )
                out.append(rec)
            except Exception as e:
                log_fetch.warning("[id=%s] FAIL: parse error: %s", iid, e)
            break

    log_fetch.info("Parsed %d/%d listings successfully", len(out), len(ids))
    return out

async def search(
    session: aiohttp.ClientSession,
    *,
    realm: str,
    league: str,
    filters: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    post_request_pause: float,
    converter: Optional[PriceConverter],
) -> List[Trade2ListingRecord]:
    log_search.info("Preparing search; realm=%s, league=%s", realm, league)

    headers = dict(headers)
    headers["Referer"] = referer_for(realm=realm, league=league)
    log_search.debug("Effective headers keys: %s", list(headers.keys()))

    url = build_search_url(realm=realm, league=league)
    log_search.info("Posting search to %s", url)

    outcome = await post_trade2_search(
        session,
        url=url,
        payload=filters,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        post_request_pause=post_request_pause,
    )

    if not outcome.ok:
        msg = outcome.message or "Unknown error"
        log_search.error("Search failed: %s", msg)
        raise RuntimeError(f"trade2 search failed: {msg}")

    ids = list(outcome.ids or [])
    log_search.info("Search OK: ids=%d total=%s complexity=%s truncated=%s search_id=%s",
                    len(ids), outcome.total, outcome.complexity, outcome.truncated, outcome.search_id)
    if not ids:
        log_search.info("No ids returned by search; nothing to fetch")
        return []

    results = await fetch_listings_batch(
        session,
        search_id=outcome.search_id or "",
        ids=ids,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        post_request_pause=post_request_pause,
        converter=converter,
    )

    log_fetch.info("Fetched %d/%d listings (search_id=%s)", len(results), len(ids), outcome.search_id or "")
    return results

def records_to_dataframe(records: List[Trade2ListingRecord]) -> pd.DataFrame:
    log_fetch.info("Converting %d records into DataFrame", len(records))
    df = pd.DataFrame([r.to_row() for r in records])
    log_fetch.info("DataFrame created with shape %s", tuple(df.shape))
    return df

def search_to_dataframe(
    config_path: str = "config.yaml",
    save_csv: Optional[str] = None,
    log_level: str = "INFO",
) -> pd.DataFrame:
    """
    Loads config, builds FX cache, runs search, converts prices to base currency,
    returns DataFrame with both raw and base prices.
    """
    setup_logging(log_level)

    async def _run() -> pd.DataFrame:
        cfg = load_config(config_path)
        opts = options_from_config(cfg)
        headers = headers_from_config(cfg)
        payload = payload_from_config(cfg)

        # Build currency cache for the chosen server/base currency
        log_search.info("Building currency cache for server=%s (base=%s)",
                        opts["base_server"].value, opts["base_currency"])
        fx_cache = get_currency_cache([opts["base_server"]], log_level=log_level)
        converter = PriceConverter(fx_cache, opts["base_server"], opts["base_currency"])

        log_search.info("Initializing aiohttp session with total timeout=%ss", opts["timeout"])
        total_timeout = aiohttp.ClientTimeout(total=opts["timeout"])
        async with aiohttp.ClientSession(timeout=total_timeout) as session:
            log_search.info("Starting search workflow")
            records = await search(
                session,
                realm=opts["realm"],
                league=opts["league"],
                filters=payload,
                headers=headers,
                timeout=opts["timeout"],
                max_retries=opts["max_retries"],
                post_request_pause=opts["post_request_pause"],
                converter=converter,
            )

        df = records_to_dataframe(records)
        if save_csv:
            df.to_csv(save_csv, index=False)
            log_fetch.info("Saved CSV -> %s", save_csv)
        log_fetch.info("search_to_dataframe complete")
        return df

    return asyncio.run(_run())


if __name__ == "__main__":
    df = search_to_dataframe("config.yaml", save_csv="results.csv", log_level="INFO")
    print(f"df head=\n{df.head()}")
    print(f"len df={len(df)}")