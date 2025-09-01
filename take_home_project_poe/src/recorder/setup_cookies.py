#!/usr/bin/env python3
"""
POE Cookie Setup (simple)

- Prompts for POESESSID (required) and cf_clearance (optional)
- Saves to poe_cookies_config.json
- Uses setup_logging from main.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from main import setup_logging

OUT_FILE = "poe_cookies_config.json"

def save_cookies_to_config(cookies: dict, config_file: str = "poe_config.json") -> None:
    cfg = {
        "cookies": cookies,
        "last_updated": datetime.now().isoformat(),
    }
    Path(config_file).write_text(json.dumps(cfg, indent=2))
    logging.info("Cookies saved to %s", Path(config_file).resolve())


def interactive_cookie_setup() -> dict | None:
    logging.info("=== POE2 Cookie Setup ===")
    try:
        poesessid = input("POESESSID (required): ").strip()
        if not poesessid:
            logging.error("POESESSID cannot be empty.")
            return None

        cf = input("cf_clearance (optional): ").strip()

        cookies = {"POESESSID": poesessid}
        if cf:
            cookies["cf_clearance"] = cf
        return cookies

    except KeyboardInterrupt:
        logging.warning("Cancelled by user.")
        return None


def main() -> int:
    setup_logging("INFO")

    cookies = interactive_cookie_setup()
    if not cookies:
        logging.error("Cookie setup failed.")
        return 1

    save_cookies_to_config(cookies, OUT_FILE)
    logging.info("Ready to scrape with cookies: %s", list(cookies.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
