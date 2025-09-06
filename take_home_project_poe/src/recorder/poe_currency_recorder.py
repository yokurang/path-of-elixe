import argparse
import asyncio
import json
import logging
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from src.recorder.constants import (
    FULL_TO_SHORT_CURRENCY_MAP,
    SHORT_TO_FULL_CURRENCY_MAP,
    POE_BASE_URL,
    TRADE2_FETCH_URL,
    COOKIES_PATH,
    CURRENCY_CACHE_PATH,
)

# Defaults
MAX_RESULTS_PER_PAIR = 70
TIMEOUT_S = 30
MAX_RETRIES = 5
TTL_HOURS = 1080
POLITE_PAUSE_S = 2.0

FILTER_OUTLIERS_DEFAULT = True
MIN_QUOTES_FOR_FILTER = 5
TRIM_LOW_PCT_DEFAULT = 5.0
TRIM_HIGH_PCT_DEFAULT = 95.0
MAD_Z_MAX_DEFAULT = 3.5

HEADERS_BASE = {
    "Accept":
        "application/json",
    "Content-Type":
        "application/json",
    "Origin":
        POE_BASE_URL,
    "Referer":
        POE_BASE_URL,
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/139.0.0.0 Safari/537.36"),
}

log = logging.getLogger("poe.currency.fx")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# class that specifies how the fx currency rate dict shall be cached in a pickle file
@dataclass(frozen=True)
class FXCache:
    ts: float
    league: str
    realm: str
    pair_rates_full: Dict[Tuple[str, str], float]
    pair_rates_short: Dict[Tuple[str, str], float]
    short_map: Dict[str, str]
    full_map: Dict[str, str]


# helper functions
def _now() -> float:
    return time.time()


def _is_stale(ts: float, ttl_h: int) -> bool:
    return (_now() - ts) > (ttl_h * 3600)


def _load_cookies() -> Dict[str, str]:
    try:
        raw = json.loads(Path(COOKIES_PATH).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (raw.get("cookies") or {}).items()}
    except Exception:
        return {}


def _headers_with_cookies() -> Dict[str, str]:
    h = dict(HEADERS_BASE)
    cookies = _load_cookies()
    parts = [
        f"{k}={v}" for k, v in cookies.items()
        if k in ("POESESSID", "cf_clearance")
    ]
    if parts:
        h["Cookie"] = "; ".join(parts)
    return h


def _retry_after_seconds(resp: aiohttp.ClientResponse) -> Optional[float]:
    ra_raw = resp.headers.get("Retry-After")
    if ra_raw:
        try:
            return max(0.0, float(ra_raw.strip()))
        except:
            pass
        try:
            dt = parsedate_to_datetime(ra_raw.strip())
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except:
            return None
    x_reset = resp.headers.get("X-RateLimit-Reset")
    if x_reset:
        try:
            return max(0.0,
                       float(x_reset) - datetime.now(timezone.utc).timestamp())
        except:
            return None
    return None


def _expo_backoff(attempt: int, base: float = 0.5, cap: float = 12.0) -> float:
    return min(cap, base *
               (2**(attempt - 1))) * (1.0 + random.uniform(-0.15, 0.15))


def _quantile(sorted_vals: List[float], q: float) -> float:
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo  # fractional part if position of the quantile is not an integer
    return float(
        sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac
    )  # weighted average of the quantile's value from the closest integers between pos


def _percentile_bounds(vals: List[float], low_pct: float,
                       high_pct: float) -> Tuple[float, float]:
    s = sorted(vals)
    return _quantile(s, low_pct / 100.0), _quantile(
        s, high_pct / 100.0
    )  # get percentile bounds on the dataset for first pass of filtering


def _payload_pair(base: str, quote: str) -> Dict[str, Any]:
    return {
        "query": {
            "status": {
                "option": "any"
            },
            "have": [quote],
            "want": [base]
        },
        "sort": {
            "have": "asc"
        }
    }


# post fetch and retry logic
async def _post(session: aiohttp.ClientSession, url: str,
                payload: Dict[str, Any]) -> Dict[str, Any]:
    pretty_payload = json.dumps(payload, indent=2, sort_keys=True)
    log.debug("[search] POST %s payload=%s", url, pretty_payload)
    base = ",".join(payload.get("query", {}).get("want", []))
    quote = ",".join(payload.get("query", {}).get("have", []))
    for attempt in range(1, MAX_RETRIES + 1):
        resp = await session.post(url,
                                  json=payload,
                                  headers=_headers_with_cookies(),
                                  timeout=TIMEOUT_S)
        async with resp:
            rate_rem, rate_used = resp.headers.get(
                "X-RateLimit-Remaining"), resp.headers.get("X-RateLimit-Used")
            log.info(
                "[search] %s/%s status=%s, rate_remaining=%s, rate_used=%s",
                base, quote, resp.status, rate_rem, rate_used)

            if resp.status == 403:
                raise RuntimeError("403 Forbidden: check cookies")

            if resp.status in (429, 500, 502, 503, 504):
                ra = _retry_after_seconds(resp) or _expo_backoff(attempt)
                log.warning("[search] HTTP %s; retry in %.2fs (attempt %d)",
                            resp.status, ra, attempt)
                await asyncio.sleep(ra)
                continue  # next attempt

            resp.raise_for_status()
            data = await resp.json(content_type=None)
            await asyncio.sleep(POLITE_PAUSE_S)
            return data

    raise RuntimeError("Max retries exceeded for POST")


async def _fetch_id(session: aiohttp.ClientSession, search_id: str,
                    iid: str) -> Optional[Dict[str, Any]]:
    url = TRADE2_FETCH_URL.format(ids=iid, search_id=search_id)
    for attempt in range(1, MAX_RETRIES + 1):
        resp = await session.get(url,
                                 headers=_headers_with_cookies(),
                                 timeout=TIMEOUT_S)
        async with resp:
            rate_rem, rate_used = resp.headers.get(
                "X-RateLimit-Remaining"), resp.headers.get("X-RateLimit-Used")
            log.debug(
                "[fetch id=%s] status=%s, rate_remaining=%s, rate_used=%s", iid,
                resp.status, rate_rem, rate_used)
            if resp.status in (429, 500, 502, 503, 504):
                ra = _retry_after_seconds(resp) or _expo_backoff(attempt)
                log.warning(
                    "[fetch id=%s] HTTP %s; retry in %.2fs (attempt %d)", iid,
                    resp.status, ra, attempt)
                await asyncio.sleep(ra)
                continue
            try:
                return (await resp.json(content_type=None)).get(
                    "result", [None]
                )[0]  # as of 2025-09-02 the API response dict has a "result" key
            except:
                return None
    return None


# parse prices from API. We need to go from "listing" -> offers and that array has "exchange" (buying/quote) and "item" (selling/base) keys; collection of asks
def _collect_prices(
        base: str, quote: str,
        listings: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    obs = []
    for listing in (l.get("listing") or {} for l in listings or []):
        for offer in listing.get("offers", []):
            item, exch = offer.get("item") or {}, offer.get("exchange") or {}
            if item.get("currency") == base and exch.get("currency") == quote:
                try:
                    base_amt, quote_amt = float(item["amount"]), float(
                        exch["amount"])
                    if base_amt > 0 and quote_amt > 0:
                        obs.append((quote_amt / base_amt, base_amt))
                except:
                    pass
    return obs


# outlier filtering strategy: 1) percentile bounds into 2) MAD z-score filtering
def _filter_outliers(
        obs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(obs) < MIN_QUOTES_FOR_FILTER:
        return obs  # if not enough quotes, return observations (no filtering)
    prices = [p for p, _ in obs]
    lo, hi = _percentile_bounds(prices, TRIM_LOW_PCT_DEFAULT,
                                TRIM_HIGH_PCT_DEFAULT)
    trimmed = [(p, a) for (p, a) in obs if lo <= p <= hi
              ] or obs  # first pass of filtering using percentile bounds
    m, mad = median([p for p, _ in trimmed]), median([
        abs(p - median([p for p, _ in trimmed])) for p, _ in trimmed
    ])  # get median absolute differences of each price point
    return [
        (p, a)
        for (p, a) in trimmed
        if mad == 0 or abs(0.6745 * (p - m) / mad) <= MAD_Z_MAX_DEFAULT
    ] or trimmed  # scale residuals into robust z-scores and filter out outliers; else return trimmed observations


def _vwap(obs: List[Tuple[float, float]]) -> Optional[float]:
    num, den = sum(p * a for p, a in obs), sum(
        a for _, a in obs)  # sum of volume * price / sum of volume
    return num / den if den else None


async def build_currency_fx(realm: str, league: str, top_n: int,
                            force_refresh: bool) -> FXCache:
    if CURRENCY_CACHE_PATH.exists() and not force_refresh:
        cached: FXCache = pickle.loads(Path(CURRENCY_CACHE_PATH).read_bytes())
        if not _is_stale(cached.ts, TTL_HOURS):
            return cached

    jar = aiohttp.CookieJar(unsafe=True)
    cookies = _load_cookies()
    if cookies:
        jar.update_cookies(cookies)

    pair_rates_full, pair_rates_short = {}, {}
    all_full = list(FULL_TO_SHORT_CURRENCY_MAP.keys())

    async with aiohttp.ClientSession(cookie_jar=jar) as session:
        exch_url = f"{POE_BASE_URL}/api/trade2/exchange/{realm}/{league}"

        # Identity pairs
        for f in all_full:
            fs = FULL_TO_SHORT_CURRENCY_MAP[f]
            pair_rates_full[(f, f)] = pair_rates_short[(fs, fs)] = 1.0

        # All pairs
        for base_full in all_full:
            base_short = FULL_TO_SHORT_CURRENCY_MAP[base_full]
            for quote_full in all_full:
                if quote_full == base_full:
                    continue
                quote_short = FULL_TO_SHORT_CURRENCY_MAP[quote_full]

                data = await _post(session, exch_url,
                                   _payload_pair(base_short, quote_short))
                search_id, result = str(data.get("id") or
                                        ""), data.get("result")

                listings = []
                if isinstance(result, dict):
                    listings.extend(result.values())
                elif isinstance(result, list):
                    for iid in result[:top_n]:
                        r = await _fetch_id(session, search_id, iid)
                        if r:
                            listings.append(r)

                obs = _collect_prices(base_short, quote_short, listings)
                obs = _filter_outliers(obs) if FILTER_OUTLIERS_DEFAULT else obs
                rate = _vwap(obs)

                if rate:
                    pair_rates_full[(base_full, quote_full)] = rate
                    pair_rates_short[(base_short, quote_short)] = rate
                    prices_only = [p for p, _ in obs]
                    m = median(prices_only) if prices_only else None
                    log.info(
                        "%s/%s: n=%d VWAP=%.6f median=%s",
                        base_short,
                        quote_short,
                        len(prices_only),
                        rate,
                        (f"{m:.6f}" if m is not None else "NA"),
                    )
                else:
                    log.info("%s/%s: no valid prices", base_short, quote_short)

                if POLITE_PAUSE_S:
                    await asyncio.sleep(POLITE_PAUSE_S)

    # cache results as pickle
    cache = FXCache(
        ts=_now(),
        league=league,
        realm=realm,
        pair_rates_full=pair_rates_full,
        pair_rates_short=pair_rates_short,
        short_map=dict(SHORT_TO_FULL_CURRENCY_MAP),
        full_map=dict(FULL_TO_SHORT_CURRENCY_MAP),
    )
    Path(CURRENCY_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CURRENCY_CACHE_PATH).write_bytes(pickle.dumps(cache))
    return cache


def get_currency_fx(realm: str, league: str, top_n: int,
                    force_refresh: bool) -> FXCache:
    return asyncio.run(build_currency_fx(realm, league, top_n, force_refresh))


def print_fx_summary(realm: str, league: str, top_n: int):
    fx = get_currency_fx(realm, league, top_n, False)
    log.info("FX SNAPSHOT: %s / %s @ %s", fx.league, fx.realm,
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fx.ts)))
    for (fs, ts), r in sorted(fx.pair_rates_short.items()):
        if fs != ts:
            log.info(f"{fs:<12} -> {ts:<12} {r:>12.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realm", default="poe2")
    ap.add_argument("--league", default="Standard")
    ap.add_argument("--top", type=int, default=MAX_RESULTS_PER_PAIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    fx = get_currency_fx(args.realm, args.league, args.top, args.force)
    print_fx_summary(args.realm, args.league, args.top)
