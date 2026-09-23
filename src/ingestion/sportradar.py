"""
Sportradar Tennis API v3 client.
API key: read from the SPORTRADAR_API_KEY environment variable (see .env.example).
Never hardcode it here — this file is source, and source gets published.
Trial: 1,000 calls, QPS=1, expires 2026-04-14

Key endpoints used:
  - /schedules/{date}/summaries.json   — all matches on a date + stats
  - /competitions.json                 — list all competitions
  - /competitions/{id}/seasons.json    — seasons for a competition
  - /rankings/atp/singles.json         — live ATP rankings
  - /rankings/wta/singles.json         — live WTA rankings
  - /competitors/{id}/summaries.json   — recent matches for a player
"""

import os
import time
import json
import requests

try:  # keys live in .env, which is gitignored; the app still runs without it
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except ImportError:
    pass
from datetime import datetime, timezone, timedelta
from typing import Optional

SR_KEY = os.environ.get("SPORTRADAR_API_KEY", "")
BASE   = "https://api.sportradar.com/tennis/trial/v3/en"
CHILE_OFFSET = -3   # Chile summer time (CLST) = UTC-3

_last_call = 0.0


def _get(path: str, params: dict = None, retries: int = 3) -> Optional[dict]:
    """Rate-limited GET: enforces QPS=1."""
    if not SR_KEY:
        raise RuntimeError(
            "SPORTRADAR_API_KEY is not set. Put it in .env at the project root "
            "or export it before running."
        )
    global _last_call
    gap = time.time() - _last_call
    if gap < 1.05:
        time.sleep(1.05 - gap)

    url = f"{BASE}/{path}"
    p   = {"api_key": SR_KEY, **(params or {})}
    for attempt in range(retries):
        _last_call = time.time()
        try:
            r = requests.get(url, params=p, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt + 1)
                continue
            print(f"  [sportradar] {r.status_code} {path}")
            return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print(f"  [sportradar] error {path}: {e}")
    return None


# ─── Helpers ─────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_chile(dt: datetime) -> datetime:
    """Convert a UTC datetime to Chilean time (UTC-3 summer / UTC-4 winter).
    March–April: Chile uses summer time (CLST = UTC-3).
    """
    return dt.astimezone(timezone(timedelta(hours=CHILE_OFFSET)))


def parse_sr_time(iso: str) -> Optional[datetime]:
    """Parse Sportradar ISO timestamp → UTC datetime."""
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return None


def chile_str(dt: datetime) -> str:
    """Format as 'HH:MM' in Chilean time."""
    return to_chile(dt).strftime("%H:%M")


# ─── Primary endpoints ────────────────────────────────────────────────────────

def daily_summaries(date_str: str) -> list:
    """All match summaries for a given date (YYYY-MM-DD)."""
    data = _get(f"schedules/{date_str}/summaries.json")
    return data.get("summaries", []) if data else []


def atp_rankings() -> list:
    """Current ATP singles rankings (top ~500)."""
    data = _get("rankings/atp/singles.json")
    return data.get("rankings", []) if data else []


def wta_rankings() -> list:
    """Current WTA singles rankings (top ~500)."""
    data = _get("rankings/wta/singles.json")
    return data.get("rankings", []) if data else []


def competition_seasons(comp_id: str) -> list:
    data = _get(f"competitions/{comp_id}/seasons.json")
    return data.get("seasons", []) if data else []


def season_summaries(season_id: str) -> list:
    """All match results for a given season."""
    data = _get(f"seasons/{season_id}/summaries.json")
    return data.get("summaries", []) if data else []


def competitor_profile(comp_id: str) -> dict:
    data = _get(f"competitors/{comp_id}/profile.json")
    return data or {}


def competitor_summaries(comp_id: str) -> list:
    """Recent match history for a player."""
    data = _get(f"competitors/{comp_id}/summaries.json")
    return data.get("summaries", []) if data else []


# ─── Today's matches (Sportradar authoritative) ───────────────────────────────

def get_todays_atp_wta(hours_ahead: float = 5.0) -> list:
    """
    Fetch today's ATP/WTA/Challenger matches starting within the next
    `hours_ahead` hours from now (UTC), plus any currently live.

    Returns list of dicts ready for prediction:
      {sr_id, p1_name, p2_name, p1_sr_id, p2_sr_id,
       start_utc, start_chile, surface, category, competition,
       best_of, round_name, status, is_live}
    """
    today     = utc_now().date().isoformat()
    summaries = daily_summaries(today)
    now_utc   = utc_now()
    cutoff    = now_utc + timedelta(hours=hours_ahead)
    window_start = now_utc - timedelta(hours=2)   # also show live matches

    WANTED_CATS = {"ATP", "WTA", "Grand Slam", "Challenger", "WTA 125K"}

    matches = []
    for s in summaries:
        se    = s.get("sport_event", {})
        ctx   = se.get("sport_event_context", {})
        cat   = ctx.get("category", {}).get("name", "")
        comp  = ctx.get("competition", {})
        mode  = ctx.get("mode", {})
        rnd   = ctx.get("round", {})
        status_obj = s.get("sport_event_status", {})

        if cat not in WANTED_CATS:
            continue

        players = se.get("competitors", [])
        if len(players) < 2:
            continue

        # Only singles
        if comp.get("type", "") != "singles":
            continue

        start_iso = se.get("start_time", "")
        start_dt  = parse_sr_time(start_iso)
        if start_dt is None:
            continue

        status = status_obj.get("status", "")
        is_live = status in ("live", "started", "1st_set", "2nd_set", "3rd_set", "4th_set", "5th_set")

        # Time filter
        if not is_live:
            if start_dt < window_start or start_dt > cutoff:
                continue

        p1_id  = players[0].get("id", "")
        p2_id  = players[1].get("id", "")
        p1_name = players[0].get("name", "")
        p2_name = players[1].get("name", "")

        # Reverse "Last, First" → "First Last"
        def clean(n):
            if "," in n:
                parts = n.split(",", 1)
                return parts[1].strip() + " " + parts[0].strip()
            return n

        comp_name = comp.get("name", "")
        gender = comp.get("gender", "")

        # Surface detection from competition name
        surface = _surface_from_name(comp_name)
        best_of = mode.get("best_of", 3)
        round_name = _parse_round(rnd.get("name", ""))

        matches.append({
            "sr_id":       se.get("id", ""),
            "p1_name":     clean(p1_name),
            "p2_name":     clean(p2_name),
            "p1_sr_id":    p1_id,
            "p2_sr_id":    p2_id,
            "start_utc":   start_dt.isoformat(),
            "start_chile": chile_str(start_dt),
            "start_utc_str": start_dt.strftime("%H:%M"),
            "surface":     surface,
            "category":    cat,
            "competition": comp_name,
            "gender":      gender,
            "best_of":     best_of,
            "round_name":  round_name,
            "status":      status,
            "is_live":     is_live,
        })

    # Sort: live first, then by start time
    matches.sort(key=lambda x: (0 if x["is_live"] else 1, x["start_utc"]))
    return matches


# ─── Stats extraction ─────────────────────────────────────────────────────────

def extract_match_stats(summary: dict) -> dict:
    """
    Extract serve/return stats from a Sportradar summary.
    Returns dict: {home: {...stats}, away: {...stats}, winner_qualifier, scores}
    """
    se_status = summary.get("sport_event_status", {})
    stats_obj = summary.get("statistics", {})
    competitors = stats_obj.get("totals", {}).get("competitors", [])

    result = {}
    for comp in competitors:
        q   = comp.get("qualifier", "")   # "home" or "away"
        raw = comp.get("statistics", {})
        svc_pts = raw.get("service_points_won", 0) + raw.get("service_points_lost", 0)
        first_in = raw.get("first_serve_successful", 0)
        first_won = raw.get("first_serve_points_won", 0)
        second_won = raw.get("second_serve_points_won", 0)
        second_in  = raw.get("second_serve_successful", 0)
        total_bp   = raw.get("total_breakpoints", 0)
        bp_won     = raw.get("breakpoints_won", 0)         # bp opportunities converted (return)
        aces       = raw.get("aces", 0)
        dfs        = raw.get("double_faults", 0)

        result[q] = {
            "sr_id":     comp.get("id", ""),
            "name":      comp.get("name", ""),
            "aces":      aces,
            "double_faults": dfs,
            # Raw counts (maps directly to matches table w_svpt, w_1stIn, etc.)
            "svpt":           svc_pts if svc_pts > 0 else None,
            "first_in":       first_in if first_in > 0 else None,
            "first_won":      first_won if first_in > 0 else None,
            "second_in":      second_in if second_in > 0 else None,
            "second_won":     second_won if second_in > 0 else None,
            # Percentages (for overlay)
            "first_serve_in_pct":   first_in / svc_pts if svc_pts > 0 else None,
            "first_serve_won_pct":  first_won / first_in if first_in > 0 else None,
            "second_serve_won_pct": second_won / second_in if second_in > 0 else None,
            "bp_conv_pct":  bp_won / total_bp if total_bp > 0 else None,
            "bp_face":      total_bp,
            "bp_converted": bp_won,
            "games_won":    raw.get("games_won", 0),
            "points_won":   raw.get("points_won", 0),
            "service_points": svc_pts,
            "ace_rate":     aces / svc_pts if svc_pts > 0 else None,
            # Bonus stats
            "tiebreaks_won":     raw.get("tiebreaks_won", 0),
            "max_games_in_row":  raw.get("max_games_in_a_row", 0),
            "max_points_in_row": raw.get("max_points_in_a_row", 0),
        }

    winner_q = "home" if se_status.get("winner") == "home_team" else (
               "away" if se_status.get("winner") == "away_team" else None)
    result["winner_qualifier"] = winner_q
    result["scores"] = se_status.get("period_scores", [])
    return result


# ─── Utilities ────────────────────────────────────────────────────────────────

CLAY_KW  = [
    "clay", "monte", "madrid", "rome", "paris", "hamburg", "roland",
    "rio", "estoril", "barcelona", "buenos aires", "marrakech", "casablanca",
    # South America = almost always clay
    "santiago", "chile", "asuncion", "paraguay", "colombia",
    "bogota", "lima", "peru", "ecuador", "quito", "cordoba", "argentina",
    "brazil", "brasil", "sao paulo", "montevideo", "uruguay",
    # Other clay
    "münchen", "munich", "geneva", "hamburg", "budapest", "istanbul",
    "morocc", "kigali", "africa",
]
GRASS_KW = ["wimbledon", "grass", "halle", "queens", "hertogenbosch",
            "eastbourne", "nottingham", "birmingham", "s-hertogenbosch"]
HARD_KW  = [
    "hard", "indian wells", "miami", "australian open", "us open",
    "flushing", "dubai", "doha", "cincinnati", "montreal", "toronto",
    "beijing", "shanghai", "tokyo", "sydney", "brisbane",
    "phoenix", "memphis", "delray", "dallas", "houston",
    "winston", "acapulco", "monterrey", "morelos", "mexico",
    "istanbul", "nur-sultan", "astana", "tashkent", "seoul",
    "antalya", "hersonissos",
]

def _surface_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in GRASS_KW): return "Grass"
    if any(k in n for k in HARD_KW):  return "Hard"
    if any(k in n for k in CLAY_KW):  return "Clay"
    return "Hard"  # default


ROUND_MAP = {
    "final": "F", "semi": "SF", "quarter": "QF",
    "round of 16": "R16", "round of 32": "R32",
    "round of 64": "R64", "round of 128": "R128",
    "1st round": "R64", "2nd round": "R32",
    "3rd round": "R16", "4th round": "QF",
    "qualifying": "Q",
}

def _parse_round(r: str) -> str:
    r = (r or "").lower()
    for k, v in ROUND_MAP.items():
        if k in r:
            return v
    return "R32"


if __name__ == "__main__":
    from datetime import date
    print(f"Fetching today's ATP/WTA matches (within 5h, Chile time = UTC-3)...")
    matches = get_todays_atp_wta(hours_ahead=5.0)
    now_chile = chile_str(utc_now())
    print(f"Current Chile time: {now_chile}")
    print(f"Found {len(matches)} upcoming/live matches\n")
    for m in matches:
        live_tag = " ⚡LIVE" if m["is_live"] else ""
        print(f"  {m['start_chile']} CL | {m['p1_name']} vs {m['p2_name']} | {m['competition']} | {m['surface']}{live_tag}")
