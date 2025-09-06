"""
Beautifulsoup scraper to get a list of currencies available from the poe2 website 
https://www.pathofexile.com/trade2/exchange/poe2/Standard
"""

import json
import requests

LEAGUE = "Standard"
ENDPOINTS = [
    f"https://www.pathofexile.com/api/trade2/data/static?league={LEAGUE}",
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/123.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# KEEP IT SIMPLE; DON'T COLLECT WHAT SEES TO BE IRRELEVANT(?) CURRENCIES
EXCLUDE_PREFIXES = ("greater-", "perfect-", 'lesser-'
                   )  # drop Greater/Perfect variants
EXCLUDE_SUFFIXES = ("shard",)  # don't care about shards


def fetch_static_json():
    last_err = None
    for url in ENDPOINTS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to fetch static data: {last_err}")


def build_currency_map(data: dict) -> dict[str, str]:
    """
    From PoE static data, take ONLY the section labeled 'Currency' and
    map each entry's id -> text, excluding variant ids.
    """
    result = {}
    for section in data.get("result", []):
        # Sections look like: {"id": "Currency", "label": "Currency", "entries": [...]}
        if section.get("label") != "Currency":
            continue
        for entry in section.get("entries", []):
            cid = entry.get("id")
            text = entry.get("text")
            if not cid or not text:
                continue
            if cid.startswith(EXCLUDE_PREFIXES):
                continue
            if cid.endswith(EXCLUDE_SUFFIXES):
                continue
            result[cid] = text
        break  # we found Currency; no need to continue
    if not result:
        raise RuntimeError("No 'Currency' entries found in static data.")
    return dict(sorted(result.items(), key=lambda kv: kv[0]))


def main():
    data = fetch_static_json()
    mapping = build_currency_map(data)

    print("SHORT_TO_FULL_CURRENCY_MAP =")
    print(json.dumps(mapping, ensure_ascii=False, indent=2))

    # note to self: this overwrite existing maps all the time
    with open("short_to_full_currency_map.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
