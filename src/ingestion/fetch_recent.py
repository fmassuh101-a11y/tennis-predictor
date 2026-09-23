"""
Fetch 2025 and 2026 ATP match results from the ATP Tour website.
This fills the gap that Sackmann hasn't updated yet.
Saves to data/raw/live/atp_results_YYYY.csv
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

LIVE_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/live")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_flashscore_results(year: int) -> pd.DataFrame | None:
    """
    Fetch ATP results from flashscore for a given year.
    NOTE: Flashscore requires JS rendering for full data.
    This is a placeholder — use the API-based approach below.
    """
    print(f"Flashscore scraping requires JS rendering. Use fetch_via_api() instead.")
    return None


def fetch_via_api_sports(year: int, api_key: str) -> pd.DataFrame | None:
    """
    Fetch ATP results from api-sports.io (free tier: 100 req/day).
    Get a free key at: https://rapidapi.com/api-sports/api/tennis-live-data
    """
    url = "https://tennis-live-data.p.rapidapi.com/results/atp"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "tennis-live-data.p.rapidapi.com"
    }
    params = {"season_id": str(year)}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                df = pd.DataFrame(results)
                return df
        print(f"  API returned status {r.status_code}")
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def fetch_tennis_abstract_2025() -> pd.DataFrame | None:
    """
    Tennis Abstract (Jeff Sackmann's site) sometimes has 2025 data
    available as downloadable CSVs.
    Check: http://www.tennisabstract.com/cgi-bin/player.cgi
    """
    # Tennis Abstract doesn't have bulk CSV downloads
    # but the match charting project has some recent data
    urls = [
        "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/charting-m-matches.csv",
    ]

    dfs = []
    for url in urls:
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text), low_memory=False)
                dfs.append(df)
                print(f"  Downloaded match charting data: {len(df)} records")
        except Exception as e:
            print(f"  Error: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def check_sackmann_2025() -> bool:
    """Check if Sackmann has added 2025 ATP data yet."""
    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2025.csv"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        lines = r.text.strip().split("\n")
        print(f"  2025 ATP file exists! {len(lines)-1} matches")
        # Save it
        os.makedirs(
            os.path.join(os.path.dirname(__file__), "../../data/raw/sackmann/atp"),
            exist_ok=True
        )
        dest = os.path.join(
            os.path.dirname(__file__),
            "../../data/raw/sackmann/atp/atp_matches_2025.csv"
        )
        with open(dest, "w") as f:
            f.write(r.text)
        return True
    else:
        print(f"  2025 ATP file not yet on Sackmann GitHub (HTTP {r.status_code})")
        return False


def check_sackmann_2026() -> bool:
    """Check if Sackmann has added 2026 ATP data yet."""
    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2026.csv"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        lines = r.text.strip().split("\n")
        print(f"  2026 ATP file exists! {len(lines)-1} matches")
        dest = os.path.join(
            os.path.dirname(__file__),
            "../../data/raw/sackmann/atp/atp_matches_2026.csv"
        )
        with open(dest, "w") as f:
            f.write(r.text)
        return True
    else:
        print(f"  2026 ATP file not yet on Sackmann GitHub (HTTP {r.status_code})")
        return False


if __name__ == "__main__":
    os.makedirs(LIVE_DIR, exist_ok=True)
    print("=== Checking for recent ATP data ===\n")

    print("Checking Sackmann GitHub for 2025 data...")
    has_2025 = check_sackmann_2025()

    print("\nChecking Sackmann GitHub for 2026 data...")
    has_2026 = check_sackmann_2026()

    print("\nDownloading Match Charting Project (recent detailed stats)...")
    mc_df = fetch_tennis_abstract_2025()
    if mc_df is not None:
        out = os.path.join(LIVE_DIR, "match_charting.csv")
        mc_df.to_csv(out, index=False)
        print(f"  Saved to {out}")

    print("\n=== Summary ===")
    print(f"Sackmann 2025 file: {'FOUND' if has_2025 else 'NOT YET AVAILABLE'}")
    print(f"Sackmann 2026 file: {'FOUND' if has_2026 else 'NOT YET AVAILABLE'}")
    print("\nFor 2025-2026 live data, you have two options:")
    print("  1. Get a free API key from https://rapidapi.com (tennis-live-data)")
    print("  2. tennis-data.co.uk odds files (includes results for 2024/2025)")
