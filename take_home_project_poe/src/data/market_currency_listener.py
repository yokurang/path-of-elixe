import enum
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from main import setup_logging
import requests
from bs4 import BeautifulSoup

import csv
from pathlib import Path

URL = "https://www.aoeah.com/poe-2-currency/exchange-rates"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
}

log_listener = logging.getLogger("poe.currency.listener")


class Server(enum.Enum):
    STANDARD = "Standard"
    RISE_SC = "Rise of the Abyssal SC"
    RISE_HC = "Rise of the Abyssal HC"
    HARDCORE = "Hardcore"


@dataclass(frozen=True)
class PairRate:
    server: Server
    base: str
    quote: str
    rate: float # base -> quote
    updated_ts: float # epoch seconds


def fetch_html() -> str:
    """Download the AOEAH exchange matrix page."""
    log_listener.debug("Fetching HTML from %s", URL)
    resp = requests.get(URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text


def _find_server_table(soup: BeautifulSoup, server: Server):
    """Find the DIV/TABLE for a specific server tab."""
    tab = soup.find("div",
                    class_="poecur-tabdiv",
                    attrs={"data-server": server.value})
    if not tab:
        raise ValueError(f"No table found for server {server.value}")
    table = tab.find("table")
    if not table:
        raise ValueError(f"No <table> under {server.value}")
    return table


def _label_from_th(th) -> str:
    txt = th.get_text(strip=True) or ""
    if txt:
        return txt
    img = th.find("img")
    if img:
        return (img.get("title") or img.get("alt") or "").strip()
    for attr in ("data-name", "data-title", "title", "aria-label"):
        if th.has_attr(attr) and th[attr].strip():
            return th[attr].strip()
    return ""


def _label_from_td(td) -> str:
    txt = td.get_text(strip=True) or ""
    if txt:
        return txt
    img = td.find("img")
    if img:
        return (img.get("title") or img.get("alt") or "").strip()
    for attr in ("data-name", "data-title", "title", "aria-label"):
        if td.has_attr(attr) and td[attr].strip():
            return td[attr].strip()
    return ""


def _parse_header_names(table) -> List[str]:
    """
    Extract column headers (quote currencies). We skip the first TH because it
    is the row label column.
    """
    headers: List[str] = []
    thead = table.find("thead")
    if thead:
        ths = thead.find_all("th")
        if ths and len(ths) > 1:
            for th in ths[1:]:
                headers.append(_label_from_th(th))

    headers = [h for h in (headers or []) if h]
    return headers

def _parse_compact_number(s: str) -> float:
    """
    Parse numbers like:
      '108.7K', '1.2M', '3B', '12,345', '1 234', '1.5k', '833'
    Returns float, raises ValueError if impossible.
    """
    if not s:
        raise ValueError("Empty number")

    s = s.strip().replace(",", "").replace(" ", "")
    if not s:
        raise ValueError("Empty number after stripping")

    # Handle trailing '+' (e.g. '1K+')
    if s.endswith("+"):
        s = s[:-1]

    # Suffix map
    mult = 1.0
    if s[-1] in ("K", "k", "M", "m", "B", "b"):
        suffix = s[-1].upper()
        s_num = s[:-1]
        if suffix == "K":
            mult = 1e3
        elif suffix == "M":
            mult = 1e6
        elif suffix == "B":
            mult = 1e9
        s = s_num

    # Some sites might inject thin spaces etc. already removed above
    val = float(s)
    return val * mult


def _parse_ratio_cell(cell_text: str) -> Optional[Tuple[float, float]]:
    """
    Parse strings like '833:1' or '108.7K:1' into (833.0, 1.0).
    Returns None if not a valid positive ratio.
    """
    s = (cell_text or "").strip()
    if not s or s in {"-", "—", "–", "N/A", "NA"}:
        return None
    if ":" not in s:
        raise ValueError("Failed to parse ratio; delimiter ':' is not found.")
    a, b = s.split(":", 1)
    try:
        x = _parse_compact_number(a.strip())
        y = _parse_compact_number(b.strip())
        if x <= 0 or y <= 0:
            raise ValueError(f"Failed to parse ratios; either x: {x} or y: {y} is <= 0")
        return (x, y)
    except Exception as e:
        raise ValueError(f"Failed to parse ratio: {e}")


def parse_matrix_to_pairs(html: str,
                          server: Server) -> Dict[Tuple[str, str], float]:
    """
    Parse the full exchange matrix for a server and produce base->quote rates.

    Orientation:
      'a:b' in (row R, column C) means 'a units of COLUMN (C) == b units of ROW (R)'.
      Therefore:
        1 R = a/b C   -> rate[R→C] = a / b
        1 C = b/a R   -> rate[C→R] = b / a
    """
    soup = BeautifulSoup(html, "html.parser")
    table = _find_server_table(soup, server)
    cols = _parse_header_names(table)

    log_listener.info("Server=%s parsed %d column headers: %s", server.value,
                      len(cols), cols[:8])

    out: Dict[Tuple[str, str], float] = {}
    body_rows = table.select("tbody tr")
    log_listener.debug("Server=%s found %d data rows", server.value,
                       len(body_rows))

    for r_idx, tr in enumerate(body_rows):
        tds = tr.find_all("td")
        if not tds:
            continue

        row_name = _label_from_td(tds[0]) or "Unknown"
        if row_name == "Unknown":
            img = tds[0].find("img")
            if img:
                row_name = (img.get("title") or img.get("alt") or
                            "Unknown").strip()

        for col_name, td in zip(cols, tds[1:]):
            raw = td.get_text(strip=True)
            ratio = _parse_ratio_cell(raw)
            if not ratio:
                continue

            a, b = ratio  # a col = b row
            try:
                rate_rc = a / b
                rate_cr = b / a
            except Exception:
                continue

            out[(row_name, col_name)] = rate_rc
            out[(col_name, row_name)] = rate_cr

            log_listener.debug(
                "Computed pair: [%s -> %s] = %.6f ; [%s -> %s] = %.6f",
                row_name, col_name, rate_rc, col_name, row_name, rate_cr)

        log_listener.debug("Row %d parsed for base=%s", r_idx, row_name)

    log_listener.info("Server=%s built %d directed pair entries", server.value,
                      len(out))
    return out


def get_currency_cache(
    servers: Optional[List[Server]] = None,
    *,
    log_level: str = "INFO",
) -> Dict[Tuple[Server, str, str], PairRate]:
    """
    Fetch once and return the full pairwise exchange-rate cache for the given servers.
    """
    setup_logging(log_level)

    servers = servers or [Server.STANDARD]
    log_listener.info("Fetching currency tables for servers=%s",
                      [s.value for s in servers])

    html = fetch_html()
    now_ts = time.time()
    cache: Dict[Tuple[Server, str, str], PairRate] = {}

    for server in servers:
        try:
            pair_map = parse_matrix_to_pairs(html, server)
            count_before = len(cache)
            for (base, quote), rate in pair_map.items():
                cache[(server, base, quote)] = PairRate(server, base, quote,
                                                        rate, now_ts)
                log_listener.debug("Cached rate: %s -> %s @ %.6f (%s)", base,
                                   quote, rate, server.value)
            added = len(cache) - count_before
            log_listener.info("Server=%s stored pair entries=%d", server.value,
                              added)
        except Exception as e:
            log_listener.warning("Server=%s parse failed: %s", server.value, e)

    log_listener.info("Cache built; total pair entries=%d", len(cache))
    return cache


def get_rate(
    cache: Dict[Tuple[Server, str, str], PairRate],
    server: Server,
    from_currency: str,
    to_currency: str,
) -> Optional[float]:
    """
    Convenience accessor: read a rate from a returned cache.
    If not found, tries inverse.
    """
    base = (from_currency or "").strip()
    quote = (to_currency or "").strip()
    pr = cache.get((server, base, quote))
    if pr:
        log_listener.info("Lookup direct: %s -> %s = %.6f (%s)", base, quote,
                          pr.rate, server.value)
        return pr.rate
    inv = cache.get((server, quote, base))
    if inv and inv.rate > 0:
        rate = 1.0 / inv.rate
        log_listener.info("Lookup inverse: %s -> %s = %.6f (%s)", base, quote,
                          rate, server.value)
        return rate
    log_listener.warning("Lookup missing: %s -> %s (%s)", base, quote,
                         server.value)
    return None

# for debugging
def save_cache_to_csv(
    cache: Dict[Tuple[Server, str, str], PairRate],
    path: str | Path = "aoeah_rates.csv",
) -> Path:
    """
    Save the directed pairwise rates to a CSV with columns:
      server, base, quote, rate, updated_ts
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(
        (
            pr.server.value,
            pr.base,
            pr.quote,
            f"{pr.rate:.12g}", # avoid scientific noise, keep precision
            f"{int(pr.updated_ts)}",
        )
        for pr in cache.values()
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["server", "base", "quote", "rate", "updated_ts"])
        w.writerows(rows)

    log_listener.info("Wrote %d pair rows to %s", len(rows), path)
    return path


def save_currency_list_to_csv(
    cache: Dict[Tuple[Server, str, str], PairRate],
    path: str | Path = "aoeah_currencies.csv",
) -> Path:
    """
    Save a de-duplicated list of currency names for inspection.
    Columns: currency
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    currencies = sorted(
        {pr.base for pr in cache.values()} | {pr.quote for pr in cache.values()}
    )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["currency"])
        for c in currencies:
            w.writerow([c])

    log_listener.info("Wrote %d currencies to %s", len(currencies), path)
    return path

# debug
# cache = get_currency_cache([Server.STANDARD], log_level="INFO")
# print("Chaos -> Divine =",
#       get_rate(cache, Server.STANDARD, "Chaos Orb", "Divine Orb"))

# write CSVs for inspection
# save_cache_to_csv(cache, "aoeah_rates.csv")
# save_currency_list_to_csv(cache, "aoeah_currencies.csv")
