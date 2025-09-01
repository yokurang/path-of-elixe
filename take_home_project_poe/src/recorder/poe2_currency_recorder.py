from __future__ import annotations

import asyncio
import json
import math
import os
import pickle
import time
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import aiohttp
import yaml

from src.recorder.constants import (
    FULL_TO_SHORT_CURRENCY_MAP, # dict[str full_name] -> str short_code
    SHORT_TO_FULL_CURRENCY_MAP, # dict[str short_code] -> str full_name
    POE_BASE_URL,              
    TRADE2_FETCH_URL, 
    CONFIG_PATH,
    COOKIES_PATH,
    CURRENCY_CACHE_PATH,
)

CACHE_TTL_HOURS = 12
MAX_RESULTS_PER_CCY = 300
MAX_RETRIES = 5
TIMEOUT = 30
POST_REQUEST_PAUSE = 1.1    
FETCH_REQUEST_PAUSE = 0.5    
VWAP_PRICE_ROUND_DP = 6        

# Robust aggregation
IQR_K = 1.5
MIN_N_FOR_TRIM = 8
RATE_MIN = 1e-12
RATE_MAX = 1e12

# Exchange constraints
EXCHANGE_STATUS_OPTION = "any"

# Base headers
HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": POE_BASE_URL,
    "Priority": "u=1, i",
    "Referer": POE_BASE_URL,
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

log = logging.getLogger("poe.currency.fx")
if not log.handlers:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class Quote:
    base_full: str
    quote_full: str
    unit_price: float # quote per 1 base
    weight: float = 1.0 # base-weight proxy (stock * base amount per offer)

@dataclass(frozen=True)
class AggRate:
    rate: float # VWAP price (quote per 1 base)
    weight: float # total weight used in VWAP (sum of base units: stock * base_amount)

@dataclass
class FXCache:
    ts: float
    league: str
    realm: str
    short_map: Dict[str, str]
    full_map: Dict[str, str]
    direct_quotes: List[Quote] # aggregated pairs as Quote(base, quote, rate)
    pair_rates_full: Dict[Tuple[str, str], float] # direct VWAP + inverses + identity
    pair_rates_short: Dict[Tuple[str, str], float] # direct VWAP + inverses + identity

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

def _expo_backoff(attempt: int, base: float = 0.5, cap: float = 15.0) -> float:
    raw = min(cap, base * (2 ** (attempt - 1)))
    return max(0.0, raw * (1.0 + random.uniform(-0.15, 0.15)))

def _now() -> float:
    return time.time()

def _is_stale(ts: float, ttl_hours: int) -> bool:
    return (_now() - ts) > (ttl_hours * 3600)

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        default_config = {"realm": "poe2", "league": "Rise of the Abyssal"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(default_config, f)
        log.info("Created default config at %s", path)
        return default_config
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _resolve_currency_name(token: str) -> str:
    if token in FULL_TO_SHORT_CURRENCY_MAP:
        return token
    if token in SHORT_TO_FULL_CURRENCY_MAP:
        return SHORT_TO_FULL_CURRENCY_MAP[token]
    t = (token or "").strip().lower()
    for full in FULL_TO_SHORT_CURRENCY_MAP.keys():
        if full.lower() == t:
            return full
    for short, full in SHORT_TO_FULL_CURRENCY_MAP.items():
        if short.lower() == t:
            return full
    raise ValueError(f"Unknown currency: {token!r}")

def _exchange_api_url(realm: str, league: str) -> str:
    return f"{POE_BASE_URL}/api/trade2/exchange/{realm}/{league}"

def _create_exchange_payload(base_short: str) -> Dict[str, Any]:
    # Query: sellers who will trade for our BASE (what they want from us).
    # have=[] means any; want=[base] means they want base from us.
    return {
        "query": {
            "status": {"option": EXCHANGE_STATUS_OPTION},  # "any"
            "have": [],
            "want": [base_short],
        },
        "sort": {"have": "asc"},
    }

# returns (search_id, ids, preloaded_listings)
async def _search_exchange(
    session: aiohttp.ClientSession,
    realm: str,
    league: str,
    base_short: str,
    headers: Dict[str, str],
    progress_info: str = ""
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    url = _exchange_api_url(realm, league)
    request_headers = dict(headers)
    request_headers["Referer"] = f"{POE_BASE_URL}/trade2/exchange/{realm}/{league}"

    attempt = 0
    while True:
        attempt += 1
        log.debug("%s EXCHANGE POST BASE=%s (want=%s) status=%s attempt=%d url=%s",
                  progress_info, SHORT_TO_FULL_CURRENCY_MAP.get(base_short, base_short),
                  base_short, EXCHANGE_STATUS_OPTION, attempt, url)
        try:
            payload = _create_exchange_payload(base_short)

            async with session.post(url, json=payload,
                                    headers=request_headers, timeout=TIMEOUT) as resp:

                if resp.status == 429:
                    ra = _retry_after_seconds(resp)
                    if ra is not None:
                        delay = min(ra + 2.0, 60.0)
                        log.warning("%s 429 Too Many Requests; sleeping %.1fs (Retry-After derived)",
                                    progress_info, delay)
                        await asyncio.sleep(delay)
                        continue

                try:
                    resp.raise_for_status()
                except aiohttp.ClientResponseError as http_error:
                    # 403 is almost always missing/expired cookies (POESESSID/cf_clearance)
                    if http_error.status == 403:
                        body = (await resp.text())[:300]
                        log.error("%s HTTP 403 Forbidden: PoE trade blocked the request. "
                                  "Ensure cookies are present/valid in %s (POESESSID, cf_clearance). "
                                  "Body(first300)=%r", progress_info, COOKIES_PATH, body)
                        raise  # bubble up; do not retry
                    if resp.status in (429, 500, 502, 503, 504) and attempt <= MAX_RETRIES:
                        ra = _retry_after_seconds(resp)
                        delay = ra if ra is not None else _expo_backoff(attempt)
                        log.warning("%s HTTP %s; backoff %.1fs (attempt %d/%d)",
                                    progress_info, resp.status, delay, attempt, MAX_RETRIES)
                        await asyncio.sleep(delay)
                        continue
                    body = (await resp.text())[:300]
                    log.error("%s HTTP %s; aborting. Body(first300)=%r",
                              progress_info, resp.status, body)
                    raise RuntimeError(f"exchange search failed {resp.status}: {body}") from http_error

                try:
                    data = await resp.json(content_type=None)
                except Exception as json_error:
                    if attempt <= MAX_RETRIES:
                        delay = _expo_backoff(attempt)
                        log.warning("%s JSON parse error %s; retry in %.1fs (attempt %d/%d)",
                                    progress_info, type(json_error).__name__, delay, attempt, MAX_RETRIES)
                        await asyncio.sleep(delay)
                        continue
                    log.error("%s JSON parse failed permanently: %s",
                              progress_info, json_error)
                    raise RuntimeError("exchange search JSON parse failed") from json_error

                if not isinstance(data, dict):
                    if attempt <= MAX_RETRIES:
                        delay = _expo_backoff(attempt)
                        log.warning("%s Invalid response root type=%s; retry in %.1fs",
                                    progress_info, type(data), delay)
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError("exchange search returned invalid data structure")

                search_id = str(data.get("id") or "")
                if not search_id:
                    if attempt <= MAX_RETRIES:
                        delay = _expo_backoff(attempt)
                        log.warning("%s Missing search_id; retry in %.1fs", progress_info, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError("exchange search returned no id")

                result_data = data.get("result")
                preloaded_listings: List[Dict[str, Any]] = []
                ids: List[str] = []

                if isinstance(result_data, dict):
                    vals = list(result_data.values())
                    if vals and isinstance(vals[0], dict) and "listing" in vals[0]:
                        preloaded_listings = vals[:MAX_RESULTS_PER_CCY]
                        ids = [v.get("id") for v in preloaded_listings
                               if isinstance(v, dict) and v.get("id")]
                        log.info("%s EXCHANGE OK BASE=%s (want=%s) → preloaded=%d (capped=%d)",
                                 progress_info, SHORT_TO_FULL_CURRENCY_MAP.get(base_short, base_short),
                                 base_short, len(preloaded_listings), MAX_RESULTS_PER_CCY)
                    else:
                        ids = list(result_data.keys())[:MAX_RESULTS_PER_CCY]
                        log.info("%s EXCHANGE OK BASE=%s (want=%s) → ids=%d (capped=%d)",
                                 progress_info, SHORT_TO_FULL_CURRENCY_MAP.get(base_short, base_short),
                                 base_short, len(ids), MAX_RESULTS_PER_CCY)

                elif isinstance(result_data, list):
                    if result_data and isinstance(result_data[0], dict) and "listing" in result_data[0]:
                        preloaded_listings = result_data[:MAX_RESULTS_PER_CCY]
                        ids = [v.get("id") for v in preloaded_listings
                               if isinstance(v, dict) and v.get("id")]
                        log.info("%s EXCHANGE OK BASE=%s (want=%s) → preloaded=%d (capped=%d)",
                                 progress_info, SHORT_TO_FULL_CURRENCY_MAP.get(base_short, base_short),
                                 base_short, len(preloaded_listings), MAX_RESULTS_PER_CCY)
                    else:
                        ids = result_data[:MAX_RESULTS_PER_CCY]
                        log.info("%s EXCHANGE OK BASE=%s (want=%s) → ids=%d (capped=%d)",
                                 progress_info, SHORT_TO_FULL_CURRENCY_MAP.get(base_short, base_short),
                                 base_short, len(ids), MAX_RESULTS_PER_CCY)
                else:
                    log.warning("%s Unexpected result type=%s; treating as empty",
                                progress_info, type(result_data))

                if POST_REQUEST_PAUSE > 0:
                    await asyncio.sleep(POST_REQUEST_PAUSE)

                return search_id, ids, preloaded_listings

        except Exception as e:
            # Do not retry on explicit 403
            if isinstance(e, aiohttp.ClientResponseError) and getattr(e, "status", None) == 403:
                raise
            if attempt <= MAX_RETRIES:
                delay = _expo_backoff(attempt)
                log.warning("%s _search_exchange error %s: %s; retry in %.1fs (attempt %d/%d)",
                            progress_info, type(e).__name__, e, delay, attempt, MAX_RETRIES)
                await asyncio.sleep(delay)
                continue
            log.error("%s _search_exchange giving up after %d attempts; last error: %s: %s",
                      progress_info, attempt, type(e).__name__, e)
            raise


async def _fetch_listings_batched(
    session: aiohttp.ClientSession,
    search_id: str,
    ids: List[str],
    headers: Dict[str, str],
    base_full: str,
    progress_info: str = ""
) -> List[Dict[str, Any]]:
    """Fetch listings in batches instead of one-by-one."""
    results: List[Dict[str, Any]] = []
    if not ids:
        return results

    batch_size = min(10, max(1, len(ids)))
    total_batches = (len(ids) + batch_size - 1) // batch_size

    for batch_start in range(0, len(ids), batch_size):
        batch_ids = ids[batch_start:batch_start + batch_size]
        batch_num = (batch_start // batch_size) + 1

        ids_param = ",".join(batch_ids)
        url = TRADE2_FETCH_URL.format(ids=ids_param, search_id=search_id)

        batch_progress = f"{progress_info} batch [{batch_num}/{total_batches}]"
        log.debug("%s Fetching batch URL: %s", batch_progress, url)

        attempt = 0
        while True:
            attempt += 1
            try:
                async with session.get(url, headers=headers, timeout=TIMEOUT) as resp:
                    if resp.status == 429:
                        ra = _retry_after_seconds(resp)
                        if ra is not None:
                            delay = min(ra + 1.0, 30.0)
                            log.warning("%s Rate limited (429), waiting %.1fs", batch_progress, delay)
                            await asyncio.sleep(delay)
                            continue

                    try:
                        resp.raise_for_status()
                    except aiohttp.ClientResponseError:
                        if resp.status in (429, 500, 502, 503, 504) and attempt <= MAX_RETRIES:
                            ra = _retry_after_seconds(resp)
                            delay = ra if ra is not None else _expo_backoff(attempt)
                            log.warning("%s HTTP %s, backoff %.1fs (attempt %d/%d)",
                                        batch_progress, resp.status, delay, attempt, MAX_RETRIES)
                            await asyncio.sleep(delay)
                            continue
                        text = (await resp.text())[:200]
                        log.error("%s HTTP error %s: %r", batch_progress, resp.status, text)
                        break

                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        if attempt <= MAX_RETRIES:
                            delay = _expo_backoff(attempt)
                            log.warning("%s JSON decode error, retry in %.1fs", batch_progress, delay)
                            await asyncio.sleep(delay)
                            continue
                        log.error("%s JSON decode failed after retries", batch_progress)
                        break

                    result = data.get("result")
                    if isinstance(result, dict):
                        results.extend(result.values())
                    elif isinstance(result, list):
                        results.extend(result)
                    else:
                        log.error("%s Unexpected result type: %s", batch_progress, type(result))
                        break

                    if FETCH_REQUEST_PAUSE > 0:
                        await asyncio.sleep(FETCH_REQUEST_PAUSE)
                    break

            except Exception as e:
                if attempt <= MAX_RETRIES:
                    delay = _expo_backoff(attempt)
                    log.warning("%s error %s, retry in %.1fs (attempt %d/%d)",
                                batch_progress, e, delay, attempt, MAX_RETRIES)
                    await asyncio.sleep(delay)
                    continue
                log.error("%s giving up after %d attempts (%s)", batch_progress, attempt, e)
                break

    log.info("%s FETCH BASE=%s: collected %d listings from %d ids (%d batches)",
             progress_info, base_full, len(results), len(ids), total_batches)
    return results

# Parsing (handle both orientations, always weight in BASE units)
def _extract_quotes_from_listing(listing: Dict[str, Any], base_full: str) -> List[Tuple[str, float, float]]:
    """
    Extract (QUOTE_full, unit_price, weight) from a listing's offers.

    Two valid orientations for BASE:
      A) exchange.currency == BASE  (you PAY BASE, receive QUOTE as item)
         unit_price = item.amount / exchange.amount     # QUOTE per 1 BASE
         weight     = stock * exchange.amount           # in BASE units

      B) item.currency == BASE      (you GET BASE, pay QUOTE as exchange)
         unit_price = exchange.amount / item.amount     # QUOTE per 1 BASE
         weight     = stock * item.amount               # in BASE units
    """
    out: List[Tuple[str, float, float]] = []
    if not isinstance(listing, dict):
        return out

    offers = listing.get("listing", {}).get("offers", [])
    if not isinstance(offers, list):
        return out

    for offer in offers:
        if not isinstance(offer, dict):
            continue

        exch = offer.get("exchange", {}) or {}
        item = offer.get("item", {}) or {}

        pay_currency = exch.get("currency")
        pay_amount = exch.get("amount")
        get_currency = item.get("currency")
        get_amount = item.get("amount")
        stock = item.get("stock", 1)

        if not all([pay_currency, pay_amount, get_currency, get_amount]):
            continue

        try:
            pay_amount_f = float(pay_amount)
            get_amount_f = float(get_amount)
            stock_f = float(stock) if stock is not None else 1.0
        except (ValueError, TypeError):
            continue

        if pay_amount_f <= 0 or get_amount_f <= 0 or stock_f <= 0:
            continue

        try:
            pay_currency_full = _resolve_currency_name(str(pay_currency))
            get_currency_full = _resolve_currency_name(str(get_currency))
        except ValueError:
            continue

        # Orientation A: BASE is exchange.currency (you pay BASE)
        if pay_currency_full == base_full:
            quote_full = get_currency_full
            unit_price = get_amount_f / pay_amount_f  # QUOTE per 1 BASE
            weight = stock_f * pay_amount_f           # weight in BASE units
            log.debug(
                "PARSING MATCH [A] BASE=%s QUOTE=%s unit=%.6f weight=%.3f "
                "(YOU PAY: %s x %.6f, YOU GET: %s x %.6f, stock=%.3f)",
                base_full, quote_full, unit_price, weight,
                pay_currency_full, pay_amount_f, get_currency_full, get_amount_f, stock_f
            )
            out.append((quote_full, unit_price, weight))
            continue

        # Orientation B: BASE is item.currency (you get BASE)
        if get_currency_full == base_full:
            quote_full = pay_currency_full
            unit_price = pay_amount_f / get_amount_f  # QUOTE per 1 BASE
            weight = stock_f * get_amount_f           # weight in BASE units
            log.debug(
                "PARSING MATCH [B] BASE=%s QUOTE=%s unit=%.6f weight=%.3f "
                "(YOU GET: %s x %.6f, YOU PAY: %s x %.6f, stock=%.3f)",
                base_full, quote_full, unit_price, weight,
                get_currency_full, get_amount_f, pay_currency_full, pay_amount_f, stock_f
            )
            out.append((quote_full, unit_price, weight))
            continue

        # Neither side is our BASE; skip
        continue

    return out

def _listings_to_quotes_direct(base_full: str, listings: List[Dict[str, Any]]) -> List[Quote]:
    """Parse listings into normalized BASE→QUOTE quotes."""
    quotes: List[Quote] = []
    skipped = 0

    for listing in listings:
        if not isinstance(listing, dict):
            skipped += 1
            continue

        extracted = _extract_quotes_from_listing(listing, base_full)
        if not extracted:
            skipped += 1
            continue

        for quote_currency_full, unit_price, weight in extracted:
            if quote_currency_full == base_full:
                continue  # ignore identity here; add 1:1 at the end

            if not (RATE_MIN <= unit_price <= RATE_MAX and math.isfinite(unit_price)):
                skipped += 1
                continue

            if weight <= 0 or not math.isfinite(weight):
                skipped += 1
                continue

            quotes.append(Quote(
                base_full=base_full,
                quote_full=quote_currency_full,
                unit_price=unit_price,
                weight=weight
            ))

    log.info("PARSE BASE=%s (direct): %d quotes extracted (skipped=%d)", base_full, len(quotes), skipped)
    return quotes

# Aggregation (weighted VWAP)
def _trim_outliers(values: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return values
    vals = sorted(v for v in values if RATE_MIN <= v <= RATE_MAX and math.isfinite(v))
    if len(vals) < max(MIN_N_FOR_TRIM, 1):
        return vals
    q1_idx = int(0.25 * (len(vals) - 1))
    q3_idx = int(0.75 * (len(vals) - 1))
    q1, q3 = vals[q1_idx], vals[q3_idx]
    iqr = max(0.0, q3 - q1)
    lo_f, hi_f = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    return [v for v in vals if lo_f <= v <= hi_f]

def _aggregate_direct(quotes: List[Quote]) -> Dict[Tuple[str, str], AggRate]:
    """
    VWAP per (BASE, QUOTE). Bucket on rounded unit_price to collapse float noise.
    Each bucket weighted by BASE units (stock * base_amount).
    """
    buckets: Dict[Tuple[str, str], Dict[float, Tuple[float, float]]] = {}
    # (base,quote) -> { rounded_price : (sum(price*weight), sum(weight)) }

    for q in quotes:
        if q.base_full == q.quote_full:
            continue
        if not (q.unit_price > 0 and math.isfinite(q.unit_price)):
            continue
        if not (q.weight > 0 and math.isfinite(q.weight)):
            continue
        pair = (q.base_full, q.quote_full)
        p = round(q.unit_price, VWAP_PRICE_ROUND_DP)
        swp, sw = buckets.setdefault(pair, {}).get(p, (0.0, 0.0))
        buckets[pair][p] = (swp + p * q.weight, sw + q.weight)

    direct: Dict[Tuple[str, str], AggRate] = {}
    for pair, price_map in buckets.items():
        prices = list(price_map.keys())
        kept = _trim_outliers(prices) if len(prices) >= MIN_N_FOR_TRIM else prices
        total_value = 0.0
        total_weight = 0.0
        for p in kept:
            swp, sw = price_map[p]
            total_value += swp
            total_weight += sw
        if total_weight > 0:
            rate = total_value / total_weight
            direct[pair] = AggRate(rate=rate, weight=total_weight)
            log.info("FINAL RATE BASE=%s → QUOTE=%s: %.6f (buckets=%d, total_weight=%.1f)",
                     pair[0], pair[1], rate, len(kept), total_weight)
    log.info("AGGREGATION: %d direct BASE→QUOTE pairs from %d samples", len(direct), len(quotes))
    return direct

def _agg_to_plain_rates(agg: Dict[Tuple[str, str], AggRate]) -> Dict[Tuple[str, str], float]:
    return {(a, b): ar.rate for (a, b), ar in agg.items()}

def _safe_inverse(r: float) -> Optional[float]:
    if not math.isfinite(r) or r <= 0:
        return None
    inv = 1.0 / r
    if inv < RATE_MIN:
        inv = RATE_MIN
    elif inv > RATE_MAX:
        inv = RATE_MAX
    return inv

def _add_missing_inverses(rates: Dict[Tuple[str, str], float]) -> int:
    added = 0
    for (a, b), r in list(rates.items()):
        if (b, a) in rates:
            continue
        inv = _safe_inverse(r)
        if inv is None:
            continue
        rates[(b, a)] = inv
        added += 1
        log.info("INVERSE ADDED: BASE=%s → QUOTE=%s: %.6f (from %s→%s %.6f)",
                 b, a, inv, a, b, r)
    return added

def direct_to_quotes(direct: Dict[Tuple[str, str], float]) -> List[Quote]:
    # Represent aggregated pairs as Quote entries for display/caching (weight=1.0 here)
    return [Quote(base_full=a, quote_full=b, unit_price=r) for (a, b), r in direct.items()]

# Cookies / headers
def _load_cookies() -> Dict[str, str]:
    try:
        with COOKIES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        obj = raw.get("cookies", {})
        return {str(k): str(v) for k, v in obj.items()}
    except FileNotFoundError:
        log.warning("Cookie file %s not found - using anonymous session", COOKIES_PATH)
        return {}
    except json.JSONDecodeError as e:
        log.error("Invalid JSON in %s: %s", COOKIES_PATH, e)
        return {}

def _build_headers_with_cookies() -> Dict[str, str]:
    headers = dict(HEADERS)
    cookies = _load_cookies()
    parts = []
    if cookies.get("POESESSID"):
        parts.append(f"POESESSID={cookies['POESESSID']}")
    if cookies.get("cf_clearance"):
        parts.append(f"cf_clearance={cookies['cf_clearance']}")
    if parts:
        headers["Cookie"] = "; ".join(parts)
        log.info("Auth session detected (POESESSID=%s..., cf_clearance=%s...)",
                 cookies.get("POESESSID", "")[:12], cookies.get("cf_clearance", "")[:12])
    else:
        log.warning("Anonymous session: no authentication cookies")
    return headers

# Public API (direct VWAP + inverses; no graph)
async def build_currency_fx(force_refresh: bool = False,
                            ttl_hours: int = CACHE_TTL_HOURS) -> FXCache:
    """Collect exchange quotes for every currency (status='any'), aggregate per BASE→QUOTE, add inverses, cache."""
    # Cache check
    if CURRENCY_CACHE_PATH.exists() and not force_refresh:
        with CURRENCY_CACHE_PATH.open("rb") as f:
            cached: FXCache = pickle.load(f)
        if not _is_stale(cached.ts, ttl_hours):
            log.info("CACHE HIT: %s (pairs=%d)",
                     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cached.ts)),
                     len(cached.pair_rates_full))
            return cached
        log.info("CACHE STALE → rebuilding")

    # Load config
    cfg = _load_yaml(CONFIG_PATH)
    realm = str(cfg.get("realm") or "poe2")
    league = str(cfg.get("league") or "Rise of the Abyssal")

    # Currency universe
    all_full = list(FULL_TO_SHORT_CURRENCY_MAP.keys())
    total_currencies = len(all_full)

    log.info("FX BUILD: realm=%s league=%s currencies=%d (status=%s)",
             realm, league, total_currencies, EXCHANGE_STATUS_OPTION)

    headers = _build_headers_with_cookies()

    # Use a cookie jar in addition to the explicit Cookie header to satisfy Cloudflare.
    cookie_dict = _load_cookies()
    jar = aiohttp.CookieJar(unsafe=True)
    if cookie_dict:
        jar.update_cookies(cookie_dict)
        log.debug("Cookie jar primed with: %s", ", ".join(sorted(cookie_dict.keys())))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT * 2),
                                     cookie_jar=jar) as session:
        all_quotes: List[Quote] = []
        bases_without_quotes: List[str] = []

        for i, base_full in enumerate(all_full, 1):
            base_short = FULL_TO_SHORT_CURRENCY_MAP[base_full]
            progress = f"[{i:2d}/{total_currencies:2d}]"
            log.info("%s START BASE=%s (%s): querying offers (want=%s, status=%s)",
                     progress, base_full, base_short, base_short, EXCHANGE_STATUS_OPTION)

            all_listings: List[Dict[str, Any]] = []

            try:
                search_id, ids, preloaded = await _search_exchange(
                    session, realm, league, base_short, headers, progress
                )

                pre_ct = len(preloaded)
                id_ct = len(ids)

                if pre_ct:
                    all_listings.extend(preloaded)
                    log.info("%s LOAD listings for BASE=%s: preloaded=%d (from exchange), ids=%d",
                             progress, base_full, pre_ct, id_ct)
                elif id_ct:
                    fetched = await _fetch_listings_batched(
                        session, search_id, ids, headers, base_full, progress
                    )
                    all_listings.extend(fetched)
                    log.info("%s LOAD listings for BASE=%s: fetched=%d via /fetch, ids_in=%d",
                             progress, base_full, len(fetched), id_ct)
                else:
                    log.info("%s LOAD listings for BASE=%s: none returned (preloaded=0, ids=0)",
                             progress, base_full)

                direct_quotes = _listings_to_quotes_direct(base_full, all_listings)
                q_ct = len(direct_quotes)

                if q_ct == 0:
                    log.info("%s NO QUOTES for BASE=%s (%s)",
                             progress, base_full, base_short)
                    bases_without_quotes.append(base_full)
                else:
                    covered_quotes = sorted(set(q.quote_full for q in direct_quotes))
                    log.info("%s PARSE ok: BASE=%s (%s) → quotes=%d, distinct_QUOTES=%d",
                             progress, base_full, base_short, q_ct, len(covered_quotes))
                    all_quotes.extend(direct_quotes)

            except Exception as e:
                log.exception("%s COLLECT BASE=%s (%s) failed: %s",
                              progress, base_full, base_short, e)
                bases_without_quotes.append(base_full)

    log.info("COLLECTION SUMMARY: currencies=%d, total_samples=%d, bases_failed_or_empty=%d",
             total_currencies, len(all_quotes), len(bases_without_quotes))

    if not all_quotes:
        raise RuntimeError("No exchange quotes collected — check cookies or rate limits")

    log.info("AGGREGATION: computing VWAP across %d samples…", len(all_quotes))
    agg_direct = _aggregate_direct(all_quotes)
    direct_rates_full: Dict[Tuple[str, str], float] = _agg_to_plain_rates(agg_direct)

    inv_added = _add_missing_inverses(direct_rates_full)
    log.info("INVERSE COMPLETION: added %d inverse pairs", inv_added)

    # Add identity pairs explicitly (1:1)
    for c_full in all_full:
        direct_rates_full[(c_full, c_full)] = 1.0

    # Map full→short for convenience
    pair_rates_short: Dict[Tuple[str, str], float] = {}
    for (fa, fb), r in direct_rates_full.items():
        sa = FULL_TO_SHORT_CURRENCY_MAP.get(fa)
        sb = FULL_TO_SHORT_CURRENCY_MAP.get(fb)
        if sa and sb:
            pair_rates_short[(sa, sb)] = r

    cache = FXCache(
        ts=_now(),
        league=league,
        realm=realm,
        short_map=dict(SHORT_TO_FULL_CURRENCY_MAP),
        full_map=dict(FULL_TO_SHORT_CURRENCY_MAP),
        direct_quotes=sorted(direct_to_quotes(direct_rates_full), key=lambda q: (q.base_full, q.quote_full)),
        pair_rates_full=direct_rates_full,
        pair_rates_short=pair_rates_short,
    )

    CURRENCY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CURRENCY_CACHE_PATH.open("wb") as f:
        pickle.dump(cache, f)

    log.info("CACHE SAVED: %s (pairs=%d, timestamp=%s)",
             CURRENCY_CACHE_PATH,
             len(direct_rates_full),
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cache.ts)))
    return cache


def get_currency_fx(force_refresh: bool = False,
                    ttl_hours: int = CACHE_TTL_HOURS) -> FXCache:
    if CURRENCY_CACHE_PATH.exists() and not force_refresh:
        with CURRENCY_CACHE_PATH.open("rb") as f:
            cached: FXCache = pickle.load(f)
        if not _is_stale(cached.ts, ttl_hours):
            return cached
    return asyncio.run(build_currency_fx(force_refresh=force_refresh, ttl_hours=ttl_hours))

def get_rate(from_currency: str, to_currency: str, use_short_names: bool = True) -> Optional[float]:
    if use_short_names and from_currency == to_currency:
        return 1.0
    if not use_short_names and from_currency == to_currency:
        return 1.0
    fx = get_currency_fx()
    if use_short_names:
        return fx.pair_rates_short.get((from_currency, to_currency))
    else:
        return fx.pair_rates_full.get((from_currency, to_currency))

def get_all_rates_for_currency(currency: str, use_short_names: bool = True) -> Dict[str, float]:
    fx = get_currency_fx()
    rates = fx.pair_rates_short if use_short_names else fx.pair_rates_full
    out: Dict[str, float] = {}
    for (frm, to), r in rates.items():
        if frm == currency:
            out[to] = r
    # guarantee identity
    if currency not in out:
        out[currency] = 1.0
    return out

def print_fx_summary():
    fx = get_currency_fx()
    log.info("\n" + "=" * 80)
    log.info("POE CURRENCY EXCHANGE RATES (%s - %s)  status=%s",
             fx.league, fx.realm, EXCHANGE_STATUS_OPTION)
    log.info("Cache timestamp: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fx.ts)))
    log.info("=" * 80)
    log.info(f"{'BASE':<10} {'QUOTE':<10} {'Rate':<18} Description")
    log.info("-" * 80)
    for (fs, ts), r in sorted(fx.pair_rates_short.items()):
        ff = fx.short_map.get(fs, fs)
        tf = fx.short_map.get(ts, ts)
        log.info(f"{fs:<10} {ts:<10} {r:<18.6f} 1 {ff} = {r:.6f} {tf}")
    log.info("Totals: %d pairs (includes inverses & identity). Stored entries=%d",
             len(fx.pair_rates_short), len(fx.direct_quotes))

if __name__ == "__main__":
    try:
        fx = get_currency_fx()
        print_fx_summary()

        log.info("\n" + "=" * 80)
        log.info("DIRECT EXCHANGE RATE MATRIX (BASE → QUOTE)")
        log.info("=" * 80)
        for base in sorted(SHORT_TO_FULL_CURRENCY_MAP.keys()):
            base_full = SHORT_TO_FULL_CURRENCY_MAP[base]
            rates = get_all_rates_for_currency(base)
            if rates:
                log.info(f"\nFrom {base} ({base_full}):")
                for to_curr, rate in sorted(rates.items()):
                    to_full = SHORT_TO_FULL_CURRENCY_MAP.get(to_curr, to_curr)
                    log.info(f"  BASE={base} → QUOTE={to_curr:<10} {rate:>12.6f} ({to_full})")
            else:
                log.info(f"\nFrom {base} ({base_full}): No rates available")

        log.info("\n" + "=" * 50)
        log.info("MARKET DEPTH (DIRECT + INVERSES)")
        log.info("=" * 50)
        total_pairs = len(fx.pair_rates_short)
        n = len(SHORT_TO_FULL_CURRENCY_MAP)
        max_possible = n * n # includes identity pairs
        coverage = total_pairs / max_possible if max_possible > 0 else 0
        stored_pairs = len(fx.direct_quotes)
        log.info("Market coverage: %d/%d pairs (%.1f%%)", total_pairs, max_possible, 100 * coverage)
        log.info("Stored entries (includes direct VWAP, inverses, identity): %d", stored_pairs)

    except Exception as e:
        log.exception("Fatal error: %s", e)
        raise
