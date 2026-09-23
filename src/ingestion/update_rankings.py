"""
Fetch and save current ATP rankings from ESPN.
Run this script periodically to keep rankings up to date.

Usage:
    python -m src.ingestion.update_rankings

Saves to: data/raw/current_atp_rankings.csv
Then run: python -m src.models.player_state  (to rebuild player_state.pkl)
"""

import os
import csv
import requests
from datetime import date

RANKINGS_PATH = os.path.join(
    os.path.dirname(__file__), "../../data/raw/current_atp_rankings.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_rankings_from_espn() -> list[dict]:
    """
    Fetch current ATP rankings from ESPN API (free, no key required).
    Returns list of {rank, player_name, points}.
    """
    url = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/rankings"

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  ESPN API error: {r.status_code}")
            return []

        data = r.json()
        rankings_data = data.get("rankings", [{}])[0].get("ranks", [])

        results = []
        for entry in rankings_data:
            athlete = entry.get("athlete", {})
            name = athlete.get("displayName", "")
            rank = entry.get("current", 0)
            points = entry.get("points", 0)
            if name and rank:
                results.append({
                    "rank": rank,
                    "player_name": name,
                    "points": points,
                    "ranking_date": date.today().isoformat(),
                })

        print(f"  ESPN: {len(results)} players fetched")
        return results

    except Exception as e:
        print(f"  ESPN API error: {e}")
        return []


def save_rankings(rankings: list[dict]):
    os.makedirs(os.path.dirname(RANKINGS_PATH), exist_ok=True)
    with open(RANKINGS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "player_name", "points", "ranking_date"])
        writer.writeheader()
        writer.writerows(rankings)
    print(f"  Saved {len(rankings)} rankings → {RANKINGS_PATH}")


def load_current_rankings() -> dict:
    """
    Load saved rankings as {player_name: {rank, points, ranking_date}}.
    Returns empty dict if file not found.
    """
    if not os.path.exists(RANKINGS_PATH):
        return {}
    import pandas as pd
    df = pd.read_csv(RANKINGS_PATH)
    result = {}
    for _, row in df.iterrows():
        result[str(row["player_name"]).strip()] = {
            "rank": int(row["rank"]),
            "rank_points": float(row.get("points", 0) or 0),
            "ranking_date": str(row.get("ranking_date", "")),
        }
    return result


def run_update():
    print("Fetching current ATP rankings...")
    rankings = fetch_rankings_from_espn()

    if not rankings:
        print("  Could not fetch live rankings. Keeping existing file.")
        return

    save_rankings(rankings)
    print(f"\nTop 10:")
    for r in rankings[:10]:
        print(f"  {r['rank']:3}. {r['player_name']:30} {r['points']:,} pts")


if __name__ == "__main__":
    run_update()
