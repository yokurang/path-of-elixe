import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import yaml

POE_BASE_URL = "https://www.pathofexile.com"
TRADE2_SEARCH_URL = POE_BASE_URL + "/api/trade2/search/{realm}/{league}"
TRADE2_FETCH_URL = POE_BASE_URL + "/api/trade2/fetch/{ids}?query={search_id}"
CONFIG_PATH = "config.yaml"

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


def setup_logging(level: str = "INFO") -> str:
    """
    Initialize logging to console + logs/log_<YYYY-MM-DD-HH-MM>.txt.
    Returns the path to the log file.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = datetime.now().strftime("logs/log_%Y-%m-%d-%H-%M.txt")

    # If logging already configured, don't reconfigure; just add a file handler.
    root = logging.getLogger()
    has_handlers = bool(root.handlers)

    lvl = getattr(logging, level.upper(), logging.INFO)
    if not has_handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8")
            ],
        )
    else:
        # Ensure our file handler exists
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(lvl)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.setLevel(lvl)

    logging.info("Logging initialized; file=%s", log_path)
    return log_path


# helper functions
def build_search_url(*, realm: str, league: str) -> str:
    from urllib.parse import quote
    return TRADE2_SEARCH_URL.format(realm=realm, league=quote(league, safe=""))


def referer_for(*, realm: str, league: str) -> str:
    from urllib.parse import quote
    return f"https://www.pathofexile.com/trade2/search/{realm}/{quote(league, safe='')}"


def _retry_after_seconds(resp: aiohttp.ClientResponse) -> Optional[float]:
    """
    Parse Retry-After seconds or HTTP-date; fallback to X-RateLimit-Reset (epoch).
    """
    ra_raw = resp.headers.get("Retry-After")
    if ra_raw:
        s = ra_raw.strip()
        # numeric seconds?
        try:
            return max(0.0, float(s))
        except Exception:
            pass
        # HTTP-date?
        try:
            dt = parsedate_to_datetime(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            pass

    # fallback epoch
    x_reset = resp.headers.get("X-RateLimit-Reset")
    if x_reset:
        try:
            reset_ts = float(x_reset)
            now_ts = datetime.now(timezone.utc).timestamp()
            return max(0.0, reset_ts - now_ts)
        except Exception:
            pass
    return None


def _expo_backoff(attempt: int, base: float = 0.5, cap: float = 12.0) -> float:
    """Exponential backoff with ±15% jitter."""
    raw = min(cap, base * (2**(attempt - 1)))
    return max(0.0, raw * (1.0 + random.uniform(-0.15, 0.15)))


def _mask_secret(val: Optional[str]) -> Optional[str]:
    if not val or not isinstance(val, str):
        return val
    if len(val) <= 6:
        return "***"
    return f"{val[:3]}…{val[-3:]}"


def _masked_cfg_for_log(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if "poe_sessid" in out:
        out["poe_sessid"] = _mask_secret(out["poe_sessid"])
    if "cf_clearance" in out:
        out["cf_clearance"] = _mask_secret(out["cf_clearance"])
    return out


# get configs from yaml
def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    log_search.info("Loading config from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    log_search.info(
        "Config loaded (masked): %s",
        json.dumps(_masked_cfg_for_log(cfg), ensure_ascii=False,
                   sort_keys=True))
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
                        _mask_secret(cfg.get("poe_sessid")),
                        _mask_secret(cfg.get("cf_clearance")))
    else:
        log_search.info("No Cookie set (anonymous/unauthenticated)")
    return h


def payload_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    log_search.info("Constructing search payload from config")
    status_obj = cfg.get("status", {"option": "online"})
    if isinstance(status_obj, str):
        status_obj = {"option": status_obj}
    stats_list = cfg.get("stats") or [{
        "type": "and",
        "filters": [],
        "disabled": False
    }]
    filters_obj = cfg.get("filters") or {
        "type_filters": {
            "filters": {},
            "disabled": False
        }
    }
    if "type_filters" not in filters_obj:
        filters_obj = {"type_filters": {"filters": {}, "disabled": False}}
    sort_obj = cfg.get("sort") or {"price": "asc"}

    payload = {
        "query": {
            "status": status_obj,
            "stats": stats_list,
            "filters": filters_obj
        },
        "sort": sort_obj
    }
    q = payload["query"]
    log_search.info(
        "Payload summary: status=%s sort=%s stats_blocks=%d has_type_filters=%s",
        q.get("status"),
        payload.get("sort"),
        len(q.get("stats") or []),
        "type_filters" in (q.get("filters") or {}),
    )
    return payload


def options_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    opts = {
        "realm": cfg.get("realm", "poe2"),
        "league": cfg.get("league", "Rise of the Abyssal"),
        "timeout": cfg.get("timeout", 30),
        "max_retries": cfg.get("max_retries", 5),
        "post_request_pause": float(cfg.get("post_request_pause", 1.0)),
    }
    log_search.info("Resolved options: %s",
                    json.dumps(opts, ensure_ascii=False, sort_keys=True))
    return opts


# data types for poe trade queries and responses
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
        return Trade2SearchOutcome(False, None, [], None, None, False,
                                   str(msg) if msg is not None else None)

    search_id = data.get("id")
    ids = data.get("result", []) or []
    total = data.get("total")
    complexity = data.get("complexity")
    truncated = bool(total is not None and total >= 10_000)
    if truncated:
        log_search.warning(
            "Search returned >=10,000 matches (truncated); consider narrowing filters"
        )
    log_search.info(
        "Parsed search outcome: search_id=%s ids=%d total=%s complexity=%s truncated=%s",
        search_id, len(ids), total, complexity, truncated)
    return Trade2SearchOutcome(True, search_id, ids, total, complexity,
                               truncated, None)


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
    """POST /api/trade2/search with respectful retries and detailed logs."""
    attempt = 0
    consec_429 = 0
    while True:
        attempt += 1
        log_search.info("POST %s (attempt %d)", url, attempt)
        log_search.debug("Request payload: %s",
                         json.dumps(payload, ensure_ascii=False))
        resp = await session.post(url,
                                  json=payload,
                                  headers=headers,
                                  timeout=timeout)
        async with resp:
            rate_rem = resp.headers.get("X-RateLimit-Remaining")
            rate_used = resp.headers.get("X-RateLimit-Used")
            log_search.info(
                "HTTP status=%s, X-RateLimit-Remaining=%s, X-RateLimit-Used=%s",
                resp.status, rate_rem, rate_used)
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError:
                if resp.status in (429, 500, 502, 503,
                                   504) and attempt <= max_retries:
                    log_search.warning(f"Response Header from POE2 trade2={resp.headers}")
                    ra = _retry_after_seconds(resp)
                    ra_hdr = resp.headers.get("Retry-After")
                    if ra is not None:
                        log_search.warning(
                            "search HTTP %s; Retry-After header=%r -> waiting %.2fs (attempt %d/%d)",
                            resp.status, ra_hdr, ra, attempt, max_retries)
                        await asyncio.sleep(ra)
                    else:
                        delay = _expo_backoff(attempt)
                        log_search.warning(
                            "search HTTP %s; Retry-After missing; backoff=%.2fs (attempt %d/%d)",
                            resp.status, delay, attempt, max_retries)
                        await asyncio.sleep(delay)
                        await asyncio.sleep(post_request_pause)
                    consec_429 = consec_429 + 1 if resp.status == 429 else 0
                    if consec_429 >= 3:
                        cool = 30.0
                        log_search.warning(
                            "search received %d consecutive 429s; cooling down for %.0fs",
                            consec_429, cool)
                        await asyncio.sleep(cool)
                        consec_429 = 0
                    continue

                body_snip = (await resp.text())[:300]
                log_search.error(
                    "search non-retriable or exhausted; status=%s, body~%r",
                    resp.status, body_snip)
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
            log_search.debug("Polite pause after request: %.2fs",
                             post_request_pause)
            await asyncio.sleep(post_request_pause)
        return _process_search_json(data)


@dataclass(frozen=True)
class Trade2Price:
    amount: Optional[float]
    currency: Optional[str]
    ptype: Optional[str]


@dataclass(frozen=True)
class Trade2ListingRecord:
    id: str
    league: Optional[str]
    realm: Optional[str]
    indexed: Optional[str]
    seller: Optional[str]
    price: Trade2Price
    verified: Optional[bool]
    rarity: Optional[str]
    base_type: Optional[str]
    type_line: Optional[str]
    name: Optional[str]
    ilvl: Optional[int]
    identified: Optional[bool]
    corrupted: Optional[bool]
    category: Optional[str]
    dps: Optional[float]
    pdps: Optional[float]
    edps: Optional[float]
    aps: Optional[float]
    crit_chance: Optional[float]
    properties_raw: Tuple[Tuple[str, Tuple[Tuple[Any, Any], ...], Optional[int],
                                Optional[int]], ...]
    requirements_raw: Tuple[Tuple[str, Tuple[Tuple[Any, Any], ...],
                                  Optional[int], Optional[int]], ...]
    req_level: Optional[int]
    req_str: Optional[int]
    req_dex: Optional[int]
    req_int: Optional[int]
    granted_skills: Tuple[str, ...]
    skill_hashes: Tuple[str, ...]
    mods_implicit: Tuple[str, ...]
    mods_explicit: Tuple[str, ...]
    hashes_implicit: Tuple[str, ...]
    hashes_explicit: Tuple[str, ...]

    # parsing helpers
    @staticmethod
    def _num(x: Any) -> Optional[float]:
        try:
            return float(str(x).replace("%", ""))
        except Exception:
            return None

    @staticmethod
    def _to_int(x: Any) -> Optional[int]:
        try:
            return int(str(x))
        except Exception:
            return None

    @staticmethod
    def _price(p: Dict[str, Any]) -> "Trade2Price":
        amt = p.get("amount")
        return Trade2Price(
            amount=Trade2ListingRecord._num(amt) if amt is not None else None,
            currency=p.get("currency"),
            ptype=p.get("type"),
        )

    @staticmethod
    def _category(icat: Any) -> Optional[str]:
        if isinstance(icat, str):
            return icat
        if isinstance(icat, list) and icat:
            return icat[0]
        if isinstance(icat, dict):
            for k, v in icat.items():
                if v:
                    return k
        return None

    @staticmethod
    def _norm(s: str) -> str:
        return s.replace("[", "").replace("]", "").lower()

    @classmethod
    def _aps_crit(cls, props: Any) -> Tuple[Optional[float], Optional[float]]:
        aps = crit = None
        for p in props or []:
            name = cls._norm(p.get("name", ""))
            vals = p.get("values") or []
            v0 = vals[0][0] if vals and vals[0] else None
            if "attacks per second" in name:
                aps = aps or cls._num(v0)
            elif "critical" in name and "chance" in name:
                crit = crit or cls._num(v0)
        return aps, crit

    @staticmethod
    def _freeze_prop_like(
        seq: Any
    ) -> Tuple[Tuple[str, Tuple[Tuple[Any, Any], ...], Optional[int],
                     Optional[int]], ...]:
        out: List[Tuple[str, Tuple[Tuple[Any, Any], ...], Optional[int],
                        Optional[int]]] = []
        for obj in seq or []:
            vals = tuple(
                ((v[0] if isinstance(v, list) else v),
                 (v[1] if isinstance(v, list) and len(v) > 1 else None))
                for v in (obj.get("values") or []))
            out.append((obj.get("name"), vals, obj.get("displayMode"),
                        obj.get("type")))
        return tuple(out)

    @classmethod
    def _req_ints(
        cls, reqs: Any
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        lvl = s = d = i = None
        for r in reqs or []:
            name = cls._norm(r.get("name", ""))
            vals = r.get("values") or []
            v0 = vals[0][0] if vals and vals[0] else None
            if "level" in name:
                lvl = cls._to_int(v0)
            elif "strength" in name or name == "str":
                s = cls._to_int(v0)
            elif "dexterity" in name or name == "dex":
                d = cls._to_int(v0)
            elif "intelligence" in name or name == "int":
                i = cls._to_int(v0)
        return lvl, s, d, i

    @staticmethod
    def _skill_names_and_hashes(
            item: Dict[str, Any]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        names: List[str] = []
        for s in item.get("grantedSkills") or []:
            txt = s["values"][0][0] if s.get("values") else s.get("name", "")
            m = re.match(r"Level\s+(\d+)\s+(.+)", str(txt))
            names.append(f"{m.group(2)} ({m.group(1)})" if m else str(txt))
        # hashes flattened from extended.hashes.skill separately
        return tuple(names), tuple()

    @staticmethod
    def _hashes(lst: Any) -> Tuple[str, ...]:
        return tuple(h[0] for h in (lst or []))

    @staticmethod
    def _jsonify(o: Any) -> Any:
        if isinstance(o, tuple):
            return [Trade2ListingRecord._jsonify(x) for x in o]
        if isinstance(o, dict):
            return {k: Trade2ListingRecord._jsonify(v) for k, v in o.items()}
        return o

    @classmethod
    def from_api(cls, listing: Dict[str, Any]) -> "Trade2ListingRecord":
        item = listing.get("item") or {}
        lst = listing.get("listing") or {}
        acc = lst.get("account") or {}
        ext = item.get("extended") or {}
        hashes = ext.get("hashes", {}) or {}

        props = item.get("properties")
        reqs = item.get("requirements")
        aps, crit = cls._aps_crit(props)
        lvl, rs, rd, ri = cls._req_ints(reqs)
        skill_names, _ = cls._skill_names_and_hashes(item)

        rec = cls(
            id=listing.get("id"),
            league=item.get("league"),
            realm=item.get("realm"),
            indexed=lst.get("indexed"),
            seller=acc.get("name"),
            price=cls._price(lst.get("price") or {}),
            verified=item.get("verified"),
            rarity=item.get("rarity"),
            base_type=item.get("baseType"),
            type_line=item.get("typeLine"),
            name=item.get("name"),
            ilvl=item.get("ilvl"),
            identified=item.get("identified"),
            corrupted=item.get("corrupted"),
            category=cls._category(item.get("category")),
            dps=cls._num(ext.get("dps")),
            pdps=cls._num(ext.get("pdps")),
            edps=cls._num(ext.get("edps")),
            aps=aps,
            crit_chance=crit,
            properties_raw=cls._freeze_prop_like(props),
            requirements_raw=cls._freeze_prop_like(reqs),
            req_level=lvl,
            req_str=rs,
            req_dex=rd,
            req_int=ri,
            granted_skills=tuple(skill_names),
            skill_hashes=tuple(h[0] for h in (hashes.get("skill") or [])),
            mods_implicit=tuple(item.get("implicitMods") or []),
            mods_explicit=tuple(item.get("explicitMods") or []),
            hashes_implicit=tuple(h[0] for h in (hashes.get("implicit") or [])),
            hashes_explicit=tuple(h[0] for h in (hashes.get("explicit") or [])),
        )
        return rec

    # ---- Row serialization ----
    def to_row(self) -> Dict[str, Any]:
        return {
            "id":
                self.id,
            "league":
                self.league,
            "realm":
                self.realm,
            "indexed":
                self.indexed,
            "seller":
                self.seller,
            "price_amount":
                self.price.amount,
            "price_currency":
                self.price.currency,
            "price_type":
                self.price.ptype,
            "verified":
                self.verified,
            "rarity":
                self.rarity,
            "base_type":
                self.base_type,
            "type_line":
                self.type_line,
            "name":
                self.name,
            "ilvl":
                self.ilvl,
            "identified":
                self.identified,
            "corrupted":
                self.corrupted,
            "category":
                self.category,
            "dps":
                self.dps,
            "pdps":
                self.pdps,
            "edps":
                self.edps,
            "aps":
                self.aps,
            "crit_chance":
                self.crit_chance,
            "properties_raw":
                json.dumps(self._jsonify(self.properties_raw),
                           ensure_ascii=False),
            "requirements_raw":
                json.dumps(self._jsonify(self.requirements_raw),
                           ensure_ascii=False),
            "req_level":
                self.req_level,
            "req_str":
                self.req_str,
            "req_dex":
                self.req_dex,
            "req_int":
                self.req_int,
            "granted_skills":
                "|".join(self.granted_skills),
            "skill_hashes":
                "|".join(self.skill_hashes),
            "mods_implicit":
                "|".join(self.mods_implicit),
            "mods_explicit":
                "|".join(self.mods_explicit),
            "hashes_implicit":
                "|".join(self.hashes_implicit),
            "hashes_explicit":
                "|".join(self.hashes_explicit),
        }


async def fetch_listings_batch(
    session: aiohttp.ClientSession,
    *,
    search_id: str,
    ids: List[str],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    post_request_pause: float,
) -> List[Trade2ListingRecord]:
    """
    Fetch listings ONE BY ONE to avoid 'Invalid query' from overly long id lists.
    Logs retry behavior, Retry-After seconds, and per-id success/failure.
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

                    body_snip = (await resp.text())[:300]
                    log_fetch.error("[id=%s] FAIL: HTTP %s %r", iid, resp.status, body_snip)
                    break  # move on to next id

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
                rec = Trade2ListingRecord.from_api(raw)
                log_fetch.info(
                    "[id=%s] SUCCESS: seller=%s price=%s %s name=%s base=%s ilvl=%s",
                    rec.id,
                    rec.seller,
                    rec.price.amount,
                    rec.price.currency,
                    (rec.name or rec.type_line or "").strip(),
                    (rec.base_type or "").strip(),
                    rec.ilvl,
                )
                out.append(rec)
            except Exception as e:
                log_fetch.warning("[id=%s] FAIL: parse error: %s", iid, e)
            break  # done with this id

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
) -> List[Trade2ListingRecord]:
    """
    Run trade2 search then fetch all listings in a single batched request.
    """
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
    log_search.info(
        "Search OK: ids=%d total=%s complexity=%s truncated=%s search_id=%s",
        len(ids), outcome.total, outcome.complexity, outcome.truncated,
        outcome.search_id)
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
    )

    log_fetch.info("Fetched %d/%d listings (search_id=%s)", len(results),
                   len(ids), outcome.search_id or "")
    return results


def records_to_dataframe(records: List[Trade2ListingRecord]) -> pd.DataFrame:
    """Convert parsed listing records into a pandas DataFrame."""
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
    One-call convenience:
      - Sets up logging (console + file)
      - Loads config (masked logging)
      - Builds headers/payload/options with summaries
      - Runs polite trade2 search + single batched fetch
      - Converts to DataFrame (records_to_dataframe)
      - Optionally saves to CSV
      - Returns the DataFrame
    """
    # Ensure logging is configured for both console and file
    setup_logging(log_level)

    async def _run() -> pd.DataFrame:
        cfg = load_config(config_path)
        opts = options_from_config(cfg)
        headers = headers_from_config(cfg)
        payload = payload_from_config(cfg)

        log_search.info("Initializing aiohttp session with total timeout=%ss",
                        opts["timeout"])
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
            )

        df = records_to_dataframe(records)
        if save_csv:
            df.to_csv(save_csv, index=False)
            log_fetch.info("Saved CSV -> %s", save_csv)
        log_fetch.info("search_to_dataframe complete")
        return df

    return asyncio.run(_run())

df = search_to_dataframe("config.yaml", save_csv="results.csv", log_level="INFO")
print(f"df head={df.head()}")
print(f"len df={len(df)}")