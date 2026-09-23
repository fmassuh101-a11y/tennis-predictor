"""
Sofascore unofficial API client.
Fetches real-time match stats: aces, double faults, 1st serve %, BP conversion, etc.

Usage:
  python -m src.ingestion.sofascore
  from src.ingestion.sofascore import get_event_stats, search_player_events
"""

import ssl
import json
import time
import urllib.request
from typing import Optional

# Sofascore blocks default Python UA — use a browser-like one
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}

# SSL context that doesn't verify — needed because macOS Python
# often lacks the right cert bundle for third-party sites
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

BASE = "https://api.sofascore.com/api/v1"


def _get(url: str, retries: int = 3) -> Optional[dict]:
    """Make a GET request, return parsed JSON or None."""
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [sofascore] Error fetching {url}: {e}")
                return None


def get_event_stats(event_id: int) -> Optional[dict]:
    """
    Fetch full statistics for a match event.

    Returns dict with stats like:
      aces, double_faults, first_serve_percentage,
      first_serve_points_won, second_serve_points_won,
      break_point_conversions, service_games_won, etc.
    """
    data = _get(f"{BASE}/event/{event_id}/statistics")
    if not data or "statistics" not in data:
        return None

    # Flatten into a simple dict keyed by stat name
    result = {"home": {}, "away": {}}
    for group in data["statistics"]:
        for item in group.get("statisticsItems", []):
            name = item.get("name", "").lower().replace(" ", "_")
            home_val = item.get("homeValue")
            away_val = item.get("awayValue")
            result["home"][name] = home_val
            result["away"][name] = away_val
    return result


def search_tennis_events(date_str: str) -> list:
    """
    Get all tennis events on a given date (format: YYYY-MM-DD).
    Returns list of event dicts with id, home/away team names, tournament info.
    """
    data = _get(f"{BASE}/sport/tennis/scheduled-events/{date_str}")
    if not data or "events" not in data:
        return []
    return data["events"]


def get_today_atp_events(date_str: str) -> list:
    """
    Filter today's events to ATP/WTA singles matches.
    Returns list of dicts: {event_id, player1, player2, tournament, round, score}
    """
    events = search_tennis_events(date_str)
    matches = []
    for ev in events:
        cat = ev.get("tournament", {}).get("category", {}).get("name", "")
        sport_event_name = ev.get("tournament", {}).get("name", "")
        # Filter ATP/WTA main tour + challengers
        if not any(x in cat.upper() for x in ["ATP", "WTA", "CHALLENGER"]):
            continue

        home = ev.get("homeTeam", {}).get("name", "Unknown")
        away = ev.get("awayTeam", {}).get("name", "Unknown")
        status = ev.get("status", {}).get("description", "")

        matches.append({
            "event_id": ev["id"],
            "player1": home,
            "player2": away,
            "tournament": sport_event_name,
            "round": ev.get("roundInfo", {}).get("name", ""),
            "status": status,
            "start_timestamp": ev.get("startTimestamp"),
        })
    return matches


def fetch_player_recent_stats(event_ids: list, player_name: str) -> dict:
    """
    Given a list of recent event IDs for a player,
    compute their rolling serve/return stats from Sofascore.

    Returns aggregated stats dict.
    """
    stats_list = []
    for eid in event_ids[-10:]:  # last 10 matches
        stats = get_event_stats(eid)
        if not stats:
            continue
        # Determine if player is home or away
        for side in ["home", "away"]:
            stats_list.append(stats[side])

    if not stats_list:
        return {}

    # Average numeric stats
    agg = {}
    for stat_dict in stats_list:
        for k, v in stat_dict.items():
            if isinstance(v, (int, float)):
                agg.setdefault(k, []).append(v)

    return {k: sum(v) / len(v) for k, v in agg.items()}


if __name__ == "__main__":
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    print(f"Fetching ATP/WTA events for {today}...")
    events = get_today_atp_events(today)
    print(f"Found {len(events)} matches")
    for ev in events[:5]:
        print(f"  [{ev['event_id']}] {ev['player1']} vs {ev['player2']} — {ev['tournament']} {ev['round']} ({ev['status']})")
        if ev["status"] in ("Ended", "Finished"):
            stats = get_event_stats(ev["event_id"])
            if stats:
                print(f"    Home aces: {stats['home'].get('aces')} | Away aces: {stats['away'].get('aces')}")
