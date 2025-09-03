# src/recorder/poe_currency_recorder.py
from __future__ import annotations

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
import yaml
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from src.recorder.constants import (
    FULL_TO_SHORT_CURRENCY_MAP,  # {"Orb of Alchemy": "alch", ...}
    SHORT_TO_FULL_CURRENCY_MAP,  # {"alch": "Orb of Alchemy", ...}
    POE_BASE_URL,  # "https://www.pathofexile.com"
    TRADE2_FETCH_URL,  # f"{POE_BASE_URL}/api/trade2/fetch/{ids}?query={search_id}"
    CONFIG_PATH,  # e.g. src/recorder/config.yaml
    COOKIES_PATH,  # e.g. src/recorder/cookies.json
    CURRENCY_CACHE_PATH,  # e.g. .cache/poe_fx.pkl
)

DEFAULT_REALM = "poe2"
DEFAULT_LEAGUE = "Standard"  # match the screenshot/site unless overridden

MAX_RESULTS_PER_PAIR = 20  # top-N listings to consider
TIMEOUT_S = 30
MAX_RETRIES = 5  # align with the reference snippet
TTL_HOURS = 12

# polite pacing + 429 cooldown
POLITE_PAUSE_S = 1.0
CONSEC_429_THRESHOLD = 3
CONSEC_429_COOLDOWN_S = 30.0

# Robust outlier filter (applied to prices only; no volume/stock used)
FILTER_OUTLIERS_DEFAULT = True
MIN_QUOTES_FOR_FILTER = 5
TRIM_LOW_PCT_DEFAULT = 5.0  # two-sided trim: keep [5%, 95%]
TRIM_HIGH_PCT_DEFAULT = 95.0
MAD_Z_MAX_DEFAULT = 3.5  # robust z cutoff post-trim

HEADERS_BASE = {
    "Accept":
        "application/json",
    "Content-Type":
        "application/json",
    "Origin":
        POE_BASE_URL,
    "Referer":
        POE_BASE_URL,
    "User-Agent":
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
         "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"),
}

log = logging.getLogger("poe.currency.fx")
if not log.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@dataclass(frozen=True)
class Quote:
    """
    Minimal metadata kept for clarity (not used in averaging).
    Orientation: you pay QUOTE to get BASE.
    """
    base_full: str  # what you receive
    quote_full: str  # what you pay


@dataclass
class FXCache:
    ts: float
    league: str
    realm: str
    pair_rates_full: Dict[Tuple[str, str],
                          float]  # ("Orb of Alchemy","Orb of Annulment")->rate
    pair_rates_short: Dict[Tuple[str, str], float]  # ("alch","annul")->rate
    short_map: Dict[str, str]
    full_map: Dict[str, str]


def _now() -> float:
    return time.time()


def _is_stale(ts: float, ttl_h: int) -> bool:
    return (_now() - ts) > (ttl_h * 3600)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        cfg = {"realm": DEFAULT_REALM, "league": DEFAULT_LEAGUE}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        log.info("Created default config at %s", path)
        return cfg
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_cookies() -> Dict[str, str]:
    try:
        raw = json.loads(Path(COOKIES_PATH).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in (raw.get("cookies") or {}).items()}
    except Exception:
        return {}


def _headers_with_cookies() -> Dict[str, str]:
    h = dict(HEADERS_BASE)
    cookies = _load_cookies()
    parts = []
    if cookies.get("POESESSID"):
        parts.append(f"POESESSID={cookies['POESESSID']}")
    if cookies.get("cf_clearance"):
        parts.append(f"cf_clearance={cookies['cf_clearance']}")
    if parts:
        h["Cookie"] = "; ".join(parts)
        log.info("Auth session detected")
    else:
        log.warning("Anonymous session: no authentication cookies")
    return h


def _exchange_url(realm: str, league: str) -> str:
    return f"{POE_BASE_URL}/api/trade2/exchange/{realm}/{league}"


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
    raw = min(cap, base * (2**(attempt - 1)))
    return max(0.0, raw * (1.0 + random.uniform(-0.15, 0.15)))


def _quantile(sorted_vals: List[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])
    q = min(max(q, 0.0), 1.0)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _percentile_bounds(vals: List[float], low_pct: float,
                       high_pct: float) -> Tuple[float, float]:
    s = sorted(vals)
    return _quantile(s, low_pct / 100.0), _quantile(s, high_pct / 100.0)


def _payload_pair(base_short: str, quote_short: str) -> Dict[str, Any]:
    """
    >>> have = what you pay  (QUOTE)
    >>> want = what you get  (BASE)
    Example: base='alch', quote='annul' ⇒ have=['annul'], want=['alch'].
    """
    return {
        "query": {
            "status": {
                "option": "any"
            },
            "have": [quote_short],
            "want": [base_short]
        },
        "sort": {
            "have": "asc"
        },
    }


async def _post_exchange_search(
    session: aiohttp.ClientSession,
    *,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    polite_pause: float,
) -> Dict[str, Any]:
    attempt = 0
    consec_429 = 0
    while True:
        attempt += 1
        log.info("[search] POST %s (attempt %d)", url, attempt)
        resp = await session.post(url,
                                  json=payload,
                                  headers=headers,
                                  timeout=timeout)
        async with resp:
            rate_rem = resp.headers.get("X-RateLimit-Remaining")
            rate_used = resp.headers.get("X-RateLimit-Used")
            log.info("[search] status=%s, rate_remaining=%s, rate_used=%s",
                     resp.status, rate_rem, rate_used)
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError as cre:
                if cre.status == 403:
                    body = (await resp.text())[:300]
                    raise RuntimeError(
                        f"403 Forbidden from trade2/exchange. Check cookies in {COOKIES_PATH}. Body~{body!r}"
                    )
                if resp.status in (429, 500, 502, 503,
                                   504) and attempt <= max_retries:
                    ra = _retry_after_seconds(resp)
                    if ra is not None:
                        log.warning(
                            "[search] HTTP %s; Retry-After=%.2fs (attempt %d/%d)",
                            resp.status, ra, attempt, max_retries)
                        await asyncio.sleep(ra)
                    else:
                        delay = _expo_backoff(attempt)
                        log.warning(
                            "[search] HTTP %s; backoff=%.2fs (attempt %d/%d)",
                            resp.status, delay, attempt, max_retries)
                        await asyncio.sleep(delay)
                    consec_429 = consec_429 + 1 if resp.status == 429 else 0
                    if consec_429 >= CONSEC_429_THRESHOLD:
                        log.warning(
                            "[search] %d consecutive 429s; cooldown %.0fs",
                            consec_429, CONSEC_429_COOLDOWN_S)
                        await asyncio.sleep(CONSEC_429_COOLDOWN_S)
                        consec_429 = 0
                    if polite_pause:
                        await asyncio.sleep(polite_pause)
                    continue
                body_snip = (await resp.text())[:300]
                raise RuntimeError(
                    f"trade2 exchange search failed status={resp.status} body~{body_snip!r}"
                )
            data = await resp.json(content_type=None)
        if polite_pause:
            await asyncio.sleep(polite_pause)
        if not isinstance(data, dict):
            raise RuntimeError("exchange POST returned non-dict")
        return data


async def _fetch_ids_one_by_one(
    session: aiohttp.ClientSession,
    *,
    search_id: str,
    ids: List[str],
    headers: Dict[str, str],
    timeout: int,
    max_retries: int,
    polite_pause: float,
) -> List[Dict[str, Any]]:
    """Rate-limit aware fetch of each id separately (no batching)."""
    if not ids:
        return []
    out: List[Dict[str, Any]] = []
    total = len(ids)
    log.info("[fetch] starting one-by-one fetch of %d ids", total)

    consec_429 = 0
    for idx, iid in enumerate(ids, start=1):
        url = TRADE2_FETCH_URL.format(ids=iid, search_id=str(search_id))
        attempt = 0
        while True:
            attempt += 1
            log.debug("[fetch %d/%d] GET %s (attempt %d)", idx, total, url,
                      attempt)
            resp = await session.get(url, headers=headers, timeout=timeout)
            async with resp:
                try:
                    resp.raise_for_status()
                except aiohttp.ClientResponseError:
                    if resp.status in (429, 500, 502, 503,
                                       504) and attempt <= max_retries:
                        ra = _retry_after_seconds(resp)
                        if ra is not None:
                            log.info(
                                "[fetch id=%s] HTTP %s; Retry-After=%.2fs (attempt %d/%d)",
                                iid, resp.status, ra, attempt, max_retries)
                            await asyncio.sleep(ra)
                        else:
                            delay = _expo_backoff(attempt)
                            log.info(
                                "[fetch id=%s] HTTP %s; backoff=%.2fs (attempt %d/%d)",
                                iid, resp.status, delay, attempt, max_retries)
                            await asyncio.sleep(delay)
                        consec_429 = consec_429 + 1 if resp.status == 429 else 0
                        if consec_429 >= CONSEC_429_THRESHOLD:
                            log.warning(
                                "[fetch] %d consecutive 429s; cooldown %.0fs",
                                consec_429, CONSEC_429_COOLDOWN_S)
                            await asyncio.sleep(CONSEC_429_COOLDOWN_S)
                            consec_429 = 0
                        if polite_pause:
                            await asyncio.sleep(polite_pause)
                        continue
                    body = (await resp.text())[:300]
                    log.warning("[fetch id=%s] HTTP %s (skip). body~%r", iid,
                                resp.status, body)
                    break

                try:
                    data = await resp.json(content_type=None)
                except Exception as e:
                    log.info("[fetch id=%s] Invalid JSON: %s", iid, e)
                    break

            if polite_pause:
                await asyncio.sleep(polite_pause)

            results = (data or {}).get("result") or []
            raw = results[0] if results else None
            if not raw or not isinstance(raw, dict):
                log.info("[fetch id=%s] No result (possibly stale)", iid)
                break

            out.append(raw)
            break  # success; next id
    log.info("[fetch] complete: got %d/%d listings", len(out), total)
    return out


# ---------------------------------------------------------------------
# Price extraction, outlier filtering, and frequency-weighted mean
# ---------------------------------------------------------------------
def _collect_unit_prices(base_short: str, quote_short: str,
                         listings: List[Dict[str, Any]]) -> List[float]:
    """
    Collect per-offer prices as QUOTE per 1 BASE.
    Each offer/listing contributes **one** observation (frequency weighting).
    """
    prices: List[float] = []
    for listing_data in listings or []:
        listing = (listing_data.get("listing") or {})
        for offer in listing.get("offers", []):
            item = offer.get("item") or {}
            exch = offer.get("exchange") or {}
            if item.get("currency") != base_short or exch.get(
                    "currency") != quote_short:
                continue
            try:
                base_amt = float(item["amount"])
                quote_amt = float(exch["amount"])
            except (KeyError, TypeError, ValueError):
                continue
            if base_amt <= 0 or not math.isfinite(
                    base_amt) or not math.isfinite(quote_amt):
                continue
            price = quote_amt / base_amt
            if price > 0 and math.isfinite(price):
                prices.append(price)
    return prices


def _filter_outliers_prices(
    prices: List[float],
    *,
    trim_low_pct: float = TRIM_LOW_PCT_DEFAULT,
    trim_high_pct: float = TRIM_HIGH_PCT_DEFAULT,
    mad_z_max: float = MAD_Z_MAX_DEFAULT,
) -> List[float]:
    if len(prices) < MIN_QUOTES_FOR_FILTER:
        return prices[:]
    # 1) percentile trim
    lo, hi = _percentile_bounds(prices, trim_low_pct, trim_high_pct)
    trimmed = [p for p in prices if lo <= p <= hi]
    if len(trimmed) < MIN_QUOTES_FOR_FILTER:
        trimmed = prices[:]
    # 2) MAD-based robust z-score about the median
    m = median(trimmed)
    abs_dev = [abs(x - m) for x in trimmed]
    mad = median(abs_dev) if abs_dev else 0.0
    if mad == 0:
        return trimmed

    def rob_z(x: float) -> float:
        return 0.6745 * (x - m) / mad

    filtered = [x for x in trimmed if abs(rob_z(x)) <= mad_z_max]
    if len(filtered) < max(3, MIN_QUOTES_FOR_FILTER // 2):
        return trimmed
    return filtered


def _freq_weighted_mean(prices: List[float]) -> Optional[float]:
    """Frequency-weighted average (each listing counts once)."""
    if not prices:
        return None
    s = 0.0
    n = 0
    for p in prices:
        if p > 0 and math.isfinite(p):
            s += p
            n += 1
    if n == 0:
        return None
    return s / n


async def build_currency_fx(
    *,
    realm: Optional[str] = None,
    league: Optional[str] = None,
    top_n: int = MAX_RESULTS_PER_PAIR,
    force_refresh: bool = False,
    ttl_hours: int = TTL_HOURS,
    filter_outliers: bool = FILTER_OUTLIERS_DEFAULT,
) -> FXCache:
    # serve cache if fresh and not forcing
    if CURRENCY_CACHE_PATH.exists() and not force_refresh:
        cached: FXCache = pickle.loads(Path(CURRENCY_CACHE_PATH).read_bytes())
        if not _is_stale(cached.ts, ttl_hours):
            return cached

    cfg = _load_yaml(Path(CONFIG_PATH))
    realm = realm or str(cfg.get("realm") or DEFAULT_REALM)
    league = league or str(cfg.get("league") or DEFAULT_LEAGUE)

    all_full = list(FULL_TO_SHORT_CURRENCY_MAP.keys())
    log.info(
        "Building FX: realm=%s league=%s currencies=%d pairs=%d (freq-weighted mean, filter_outliers=%s)",
        realm,
        league,
        len(all_full),
        len(all_full) * (len(all_full) - 1),
        filter_outliers,
    )

    headers = _headers_with_cookies()
    jar = aiohttp.CookieJar(unsafe=True)
    cookies = _load_cookies()
    if cookies:
        jar.update_cookies(cookies)

    pair_rates_full: Dict[Tuple[str, str], float] = {}
    pair_rates_short: Dict[Tuple[str, str], float] = {}

    total_timeout = aiohttp.ClientTimeout(total=TIMEOUT_S * 2)
    async with aiohttp.ClientSession(timeout=total_timeout,
                                     cookie_jar=jar) as session:
        exch_url = _exchange_url(realm, league)
        log.info("Exchange URL: %s", exch_url)

        # Identity pairs
        for f in all_full:
            fs = FULL_TO_SHORT_CURRENCY_MAP[f]
            pair_rates_full[(f, f)] = 1.0
            pair_rates_short[(fs, fs)] = 1.0

        # All ordered pairs (base <- quote)
        for base_full in all_full:
            base_short = FULL_TO_SHORT_CURRENCY_MAP[base_full]
            for quote_full in all_full:
                if quote_full == base_full:
                    continue
                quote_short = FULL_TO_SHORT_CURRENCY_MAP[quote_full]

                payload = _payload_pair(base_short, quote_short)

                # 1) POST exchange search (rate-limit aware)
                data = await _post_exchange_search(
                    session,
                    url=exch_url,
                    payload=payload,
                    headers=headers,
                    timeout=TIMEOUT_S,
                    max_retries=MAX_RETRIES,
                    polite_pause=POLITE_PAUSE_S,
                )

                # Interpret result
                result = data.get("result")
                search_id = str(data.get("id") or "")
                listings: List[Dict[str, Any]] = []
                ids: List[str] = []

                if isinstance(result, dict):
                    listings.extend(list(result.values()))
                elif isinstance(result, list):
                    ids = result[:top_n]
                else:
                    log.debug("Unknown 'result' type for %s→%s: %r", base_short,
                              quote_short,
                              type(result).__name__)

                # 2) If ids were returned, fetch them one by one (rate-limit aware)
                if ids:
                    fetched = await _fetch_ids_one_by_one(
                        session,
                        search_id=search_id,
                        ids=ids,
                        headers=headers,
                        timeout=TIMEOUT_S,
                        max_retries=MAX_RETRIES,
                        polite_pause=POLITE_PAUSE_S,
                    )
                    listings.extend(fetched)

                # 3) Collect prices (each offer counts once)
                prices = _collect_unit_prices(base_short, quote_short, listings)
                if filter_outliers and prices:
                    prices = _filter_outliers_prices(
                        prices,
                        trim_low_pct=TRIM_LOW_PCT_DEFAULT,
                        trim_high_pct=TRIM_HIGH_PCT_DEFAULT,
                        mad_z_max=MAD_Z_MAX_DEFAULT,
                    )

                rate = _freq_weighted_mean(prices)
                if rate is not None:
                    pair_rates_full[(base_full, quote_full)] = rate
                    pair_rates_short[(base_short, quote_short)] = rate
                    if log.isEnabledFor(logging.INFO):
                        m = median(prices) if prices else None
                        log.info("%s→%s: n=%d mean=%.6f median=%s", base_short,
                                 quote_short, len(prices), rate,
                                 (f"{m:.6f}" if m is not None else "NA"))
                else:
                    log.info("%s→%s: 0 valid prices", base_short, quote_short)

                # be polite between pairs
                if POLITE_PAUSE_S:
                    await asyncio.sleep(POLITE_PAUSE_S)

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


def get_currency_fx(
    *,
    realm: Optional[str] = None,
    league: Optional[str] = None,
    top_n: int = MAX_RESULTS_PER_PAIR,
    force_refresh: bool = False,
    ttl_hours: int = TTL_HOURS,
) -> FXCache:
    if CURRENCY_CACHE_PATH.exists() and not force_refresh:
        cached: FXCache = pickle.loads(Path(CURRENCY_CACHE_PATH).read_bytes())
        if not _is_stale(cached.ts, ttl_hours):
            return cached
    return asyncio.run(
        build_currency_fx(
            realm=realm,
            league=league,
            top_n=top_n,
            force_refresh=force_refresh,
            ttl_hours=ttl_hours,
        ))


def get_rate(
    frm: str,
    to: str,
    use_short: bool = True,
    realm: Optional[str] = None,
    league: Optional[str] = None,
    top_n: int = MAX_RESULTS_PER_PAIR,
) -> Optional[float]:
    """
    Get the frequency-weighted average price for frm->to where price = to per 1 frm.
    (frm is BASE, to is QUOTE; i.e., how many 'to' you pay for 1 'frm')
    """
    if frm == to:
        return 1.0
    fx = get_currency_fx(realm=realm, league=league, top_n=top_n)
    table = fx.pair_rates_short if use_short else fx.pair_rates_full
    return table.get((frm, to))


def print_fx_summary(realm: Optional[str] = None,
                     league: Optional[str] = None,
                     top_n: int = MAX_RESULTS_PER_PAIR):
    fx = get_currency_fx(realm=realm, league=league, top_n=top_n)
    log.info("FX SNAPSHOT: %s / %s @ %s", fx.league, fx.realm,
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fx.ts)))
    for (fs, ts), r in sorted(fx.pair_rates_short.items()):
        if fs != ts:
            ff = fx.short_map.get(fs, fs)
            tf = fx.short_map.get(ts, ts)
            log.info(
                f"{fs:<12} -> {ts:<12} {r:>12.6f}   (1 {ff} = {r:.6f} {tf})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--realm", default=None)
    ap.add_argument("--league", default=None)  # e.g. "Standard"
    ap.add_argument("--top", type=int, default=MAX_RESULTS_PER_PAIR)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--no-filter",
        action="store_true",
        help="disable robust outlier filtering (still dedup/parse-sane)")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for h in logging.getLogger().handlers:
            h.setLevel(logging.DEBUG)

    try:
        fx = get_currency_fx(
            realm=args.realm,
            league=args.league,
            top_n=args.top,
            force_refresh=args.force,
            ttl_hours=TTL_HOURS,
        )
        print_fx_summary(realm=args.realm, league=args.league, top_n=args.top)
    except Exception as e:
        log.exception("Fatal error: %s", e)
        raise
