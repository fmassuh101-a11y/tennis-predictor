"""
Daily prediction runner.
Fetches today's ATP matches from multiple sources, generates predictions,
flags value bets, and saves a daily report.

Run every day: python -m src.agents.daily_runner
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, date

from src.models.predict import predict_match, load_state
from src.ingestion.update_rankings import run_update as refresh_rankings

LIVE_DIR   = os.path.join(os.path.dirname(__file__), "../../data/raw/live")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "../../data/reports")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────
# 1. TODAY'S MATCH FETCHERS
# ─────────────────────────────────────────────

def fetch_matches_from_odds_api() -> list[dict]:
    """
    Fetch today's ATP matches from The Odds API (free tier: 500 req/month).
    Get key at: https://the-odds-api.com  (free, no credit card)
    Returns list of {player1, player2, surface, p1_odds, p2_odds, tournament}
    """
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("  No ODDS_API_KEY set. Skipping Odds API.")
        return []

    url = "https://api.the-odds-api.com/v4/sports/tennis_atp/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            print(f"  Odds API error: {r.status_code}")
            return []
        data = r.json()
        matches = []
        for game in data:
            teams = game.get("home_team", ""), game.get("away_team", "")
            bookmakers = game.get("bookmakers", [])
            odds = {}
            for bm in bookmakers:
                if bm["key"] in ("pinnacle", "betfair", "bet365"):
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "h2h":
                            for o in mkt.get("outcomes", []):
                                odds[o["name"]] = o["price"]
                    break
            if not odds and bookmakers:
                for mkt in bookmakers[0].get("markets", []):
                    if mkt["key"] == "h2h":
                        for o in mkt.get("outcomes", []):
                            odds[o["name"]] = o["price"]
            if len(teams) == 2 and teams[0] in odds and teams[1] in odds:
                matches.append({
                    "player1": teams[0],
                    "player2": teams[1],
                    "p1_odds": odds[teams[0]],
                    "p2_odds": odds[teams[1]],
                    "tournament": game.get("sport_title", "ATP"),
                    "surface": "Hard",  # will be enriched if known
                    "source": "odds_api",
                })
        print(f"  Odds API: {len(matches)} matches found")
        return matches
    except Exception as e:
        print(f"  Odds API error: {e}")
        return []


def fetch_matches_from_sofascore() -> list[dict]:
    """
    Fetch today's ATP matches from SofaScore API (unofficial but stable).
    No API key needed.
    """
    today = date.today().strftime("%Y-%m-%d")
    url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{today}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  SofaScore: HTTP {r.status_code}")
            return []
        data = r.json()
        events = data.get("events", [])
        matches = []
        for event in events:
            # Filter ATP only
            tournament = event.get("tournament", {})
            category = tournament.get("category", {}).get("name", "")
            if "ATP" not in category and "Challenger" not in category:
                continue

            home = event.get("homeTeam", {}).get("name", "")
            away = event.get("awayTeam", {}).get("name", "")
            if not home or not away:
                continue

            # Surface from tournament ground type
            ground = event.get("groundType", "Hard")
            surface_map = {"CLAY": "Clay", "GRASS": "Grass",
                           "HARD": "Hard", "CARPET": "Carpet"}
            surface = surface_map.get(str(ground).upper(), "Hard")

            # Round
            round_name = event.get("roundInfo", {}).get("name", "R32")

            # Level
            t_name = tournament.get("name", "")
            if any(x in t_name for x in ["Grand Slam", "Australian", "Roland", "Wimbledon", "US Open"]):
                level = "G"
            elif any(x in t_name for x in ["Masters", "ATP 1000"]):
                level = "M"
            elif "Challenger" in t_name:
                level = "C"
            else:
                level = "A"

            matches.append({
                "player1": home,
                "player2": away,
                "p1_odds": None,
                "p2_odds": None,
                "tournament": t_name,
                "surface": surface,
                "level": level,
                "round": round_name,
                "source": "sofascore",
                "event_id": event.get("id"),
            })

        print(f"  SofaScore: {len(matches)} ATP/Challenger matches found")
        return matches
    except Exception as e:
        print(f"  SofaScore error: {e}")
        return []


def fetch_odds_from_tennis_data_today() -> dict:
    """
    Check tennis-data.co.uk for any recent 2026 updates.
    Returns dict of {(winner_norm, loser_norm): {odds}} for recent matches.
    """
    try:
        odds_path = os.path.join(
            os.path.dirname(__file__), "../../data/raw/odds/odds_2026.xlsx"
        )
        if not os.path.exists(odds_path):
            return {}
        df = pd.read_excel(odds_path, engine="openpyxl")
        recent = df.tail(50)  # last 50 matches
        result = {}
        for _, row in recent.iterrows():
            key = (str(row.get("Winner", "")).strip().lower(),
                   str(row.get("Loser", "")).strip().lower())
            result[key] = {
                "b365_w": row.get("B365W"),
                "b365_l": row.get("B365L"),
                "avg_w": row.get("AvgW"),
                "avg_l": row.get("AvgL"),
            }
        return result
    except Exception as e:
        print(f"  tennis-data.co.uk today: {e}")
        return {}


# ─────────────────────────────────────────────
# 2. MANUAL MATCH INPUT
# ─────────────────────────────────────────────

def run_manual_predictions(matches: list[dict]) -> list[dict]:
    """
    Run predictions on a manually provided list of matches.
    Each match: {player1, player2, surface, p1_odds, p2_odds, level, best_of, round}
    """
    results = []
    for m in matches:
        print(f"\nPredicting: {m['player1']} vs {m['player2']} ({m.get('surface','Hard')})")
        r = predict_match(
            player1=m["player1"],
            player2=m["player2"],
            surface=m.get("surface", "Hard"),
            p1_odds=m.get("p1_odds"),
            p2_odds=m.get("p2_odds"),
            level=m.get("level", "A"),
            best_of=m.get("best_of", 3),
            round_name=m.get("round", "R32"),
        )
        r["tournament"] = m.get("tournament", "Unknown")
        results.append(r)
        time.sleep(0.1)
    return results


# ─────────────────────────────────────────────
# 3. REPORT GENERATION
# ─────────────────────────────────────────────

def generate_report(predictions: list[dict]) -> str:
    today_str = date.today().strftime("%Y-%m-%d")
    os.makedirs(REPORT_DIR, exist_ok=True)

    value_bets = [p for p in predictions if p.get("has_value_bet") and "error" not in p]
    all_preds  = [p for p in predictions if "error" not in p]

    lines = [
        f"TENNIS PREDICTOR — DAILY REPORT",
        f"Date: {today_str}",
        f"Matches analyzed: {len(all_preds)}",
        f"Value bets flagged: {len(value_bets)}",
        "=" * 60,
    ]

    if value_bets:
        lines.append("\n★ VALUE BETS ★")
        for p in sorted(value_bets, key=lambda x: max(b["edge"] for b in x["value_bets"]), reverse=True):
            for b in p["value_bets"]:
                lines.append(
                    f"  BET: {b['bet_on']:25} @ {b['odds']:.2f}  |  "
                    f"Edge: {b['edge_pct']}  |  "
                    f"Model: {b['model_prob']*100:.1f}%  |  "
                    f"Market: {b['market_prob']*100:.1f}%"
                )
                lines.append(f"       Match: {p['match']} on {p['surface']}")
        lines.append("")

    lines.append("ALL PREDICTIONS")
    lines.append("-" * 60)
    for p in all_preds:
        winner_mark = "★" if p.get("has_value_bet") else " "
        lines.append(
            f"{winner_mark} {p['predicted_winner']:25} "
            f"({p['confidence']*100:.0f}%)  |  "
            f"{p['match']}  |  {p['surface']}"
        )

    report = "\n".join(lines)

    # Save as text and JSON
    txt_path  = os.path.join(REPORT_DIR, f"report_{today_str}.txt")
    json_path = os.path.join(REPORT_DIR, f"report_{today_str}.json")

    with open(txt_path, "w") as f:
        f.write(report)
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"\nReport saved: {txt_path}")
    return report


# ─────────────────────────────────────────────
# 4. MAIN DAILY RUN
# ─────────────────────────────────────────────

def run_daily(manual_matches: list[dict] = None):
    """
    Full daily pipeline:
    1. Fetch today's matches (auto + manual)
    2. Predict each match
    3. Flag value bets
    4. Save report
    """
    print(f"\n{'='*60}")
    print(f"TENNIS PREDICTOR — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Refresh ATP rankings before predictions
    print("\nRefreshing ATP rankings...")
    refresh_rankings()

    all_matches = []

    # Auto-fetch from SofaScore
    print("\nFetching today's matches...")
    sofascore = fetch_matches_from_sofascore()
    all_matches.extend(sofascore)

    # Auto-fetch odds from The Odds API
    odds_matches = fetch_matches_from_odds_api()
    # Merge odds into sofascore matches where names match
    # (basic: just add separately if no match found)
    all_matches.extend([m for m in odds_matches if m not in all_matches])

    # Manual override
    if manual_matches:
        print(f"\nAdding {len(manual_matches)} manual matches...")
        all_matches = manual_matches + all_matches  # manual takes priority

    if not all_matches:
        print("\nNo matches found. Use manual_matches parameter.")
        print("Example:")
        print('  run_daily([{"player1": "Carlos Alcaraz", "player2": "Jannik Sinner",')
        print('              "surface": "Clay", "p1_odds": 1.85, "p2_odds": 2.00}])')
        return []

    print(f"\nTotal matches to predict: {len(all_matches)}")
    predictions = run_manual_predictions(all_matches)

    # Report
    report = generate_report(predictions)
    print("\n" + report)

    return predictions


if __name__ == "__main__":
    # Example: run with some manual matches for today
    today_matches = [
        {
            "player1": "Carlos Alcaraz",
            "player2": "Jannik Sinner",
            "surface": "Hard",
            "p1_odds": 1.90,
            "p2_odds": 1.95,
            "level": "G",
            "best_of": 5,
            "round": "SF",
            "tournament": "Australian Open",
        },
    ]
    run_daily(manual_matches=today_matches)
