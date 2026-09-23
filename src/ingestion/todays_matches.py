"""
Fetch today's tennis matches and run predictions.

Primary schedule source: Sportradar (accurate UTC times, clean player names).
Odds source: api-tennis.com (bookmaker odds via existing key).

Only shows matches starting within the next 5 hours (Chile time = UTC-3)
plus any currently live matches.

Run: python -m src.ingestion.todays_matches
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except ImportError:
    pass

import sys, json, requests
from datetime import date, datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.sportradar import (
    get_todays_atp_wta, utc_now, chile_str, CHILE_OFFSET
)

# api-tennis.com (odds only)
KEY_FILE = os.path.join(os.path.dirname(__file__), "../../data/api_tennis_key.txt")
BASE_URL  = "https://api.api-tennis.com/tennis/"

# api-tennis event types → (tour, level, best_of)
ODDS_EVENT_TYPES = {
    "265": ("ATP", "A", 3),
    "275": ("ATP", "G", 5),
    "281": ("ATP", "C", 3),
    "266": ("WTA", "A", 3),
    "272": ("WTA", "A", 3),
    "271": ("WTA", "C", 3),
}


def get_api_key() -> str:
    k = os.environ.get("TENNIS_API_KEY", "").strip()
    if k: return k
    env = os.environ.get("API_TENNIS_KEY")
    if env:
        return env.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f: return f.read().strip()
    return ""


def fetch_odds_map(api_key: str, today: str) -> dict:
    """
    Fetch all odds for today from api-tennis.com.
    Returns dict: normalized_player_name_lower → {p1_odds, p2_odds, bookmaker, event_key}
    """
    odds = {}
    for etype in ODDS_EVENT_TYPES:
        try:
            r = requests.get(BASE_URL, params={
                "method": "get_fixtures", "APIkey": api_key,
                "date_start": today, "date_stop": today,
                "event_type_key": etype,
            }, timeout=15)
            if r.status_code != 200: continue
            for e in r.json().get("result", []):
                p1 = (e.get("event_first_player") or "").strip()
                p2 = (e.get("event_second_player") or "").strip()
                ekey = e.get("event_key")
                if p1 and p2 and ekey:
                    odds[f"{_norm(p1)}|{_norm(p2)}"] = ekey
                    odds[f"{_norm(p2)}|{_norm(p1)}"] = ekey
        except Exception:
            pass
    return odds


def fetch_single_odds(api_key: str, event_key: int) -> tuple:
    """Return (p1_odds, p2_odds, bookmaker) for an event key."""
    try:
        r = requests.get(BASE_URL, params={
            "method": "get_odds", "APIkey": api_key, "event_key": event_key
        }, timeout=10)
        if r.status_code != 200: return None, None, None
        result = r.json().get("result", {}).get(str(event_key), {})
        ha = result.get("Home/Away", {})
        if not ha: return None, None, None
        home_book = ha.get("Home", {})
        away_book = ha.get("Away", {})
        if not home_book or not away_book: return None, None, None
        for bookie in ["Pncl", "1xBet", "bet365", "WilliamHill"]:
            if bookie in home_book and bookie in away_book:
                return float(home_book[bookie]), float(away_book[bookie]), bookie
        bookie = list(home_book.keys())[0]
        return float(home_book[bookie]), float(away_book[bookie]), bookie
    except Exception:
        return None, None, None


def _norm(name: str) -> str:
    """Normalize for exact matching."""
    return name.lower().strip()


def _last(name: str) -> str:
    """Extract last name for fuzzy last-name matching.
    'D. Medvedev' → 'medvedev'  |  'Daniil Medvedev' → 'medvedev'
    'G. A. Olivieri' → 'olivieri'
    """
    parts = name.strip().split()
    # Find last part that isn't an initial (len > 1 and no dot)
    for p in reversed(parts):
        clean = p.rstrip(".")
        if len(clean) > 1:
            return clean.lower()
    return parts[-1].lower() if parts else ""


# Level mapping from Sportradar category / competition name
def _level(category: str, comp_name: str, gender: str) -> str:
    cn = comp_name.lower()
    cat = category.lower()
    if "grand slam" in cat or any(x in cn for x in ["australian open","us open","french open","wimbledon","roland"]):
        return "G"
    if "atp" in cat:
        if any(x in cn for x in ["masters","1000","indian wells","miami","monte","madrid","rome","paris","canada","cincinnati","shanghai","beijing"]):
            return "M"
        if "challenger" in cn or "challenger" in cat: return "C"
        return "A"
    if "wta" in cat:
        if any(x in cn for x in ["1000","indian wells","miami","madrid","rome","dubai","doha","beijing","singapore","montreal","toronto","cincinnati","wuhan"]):
            return "M"
        if "125" in cn or "challenger" in cn: return "C"
        return "A"
    if "challenger" in cat or "challenger" in cn: return "C"
    return "A"


def fetch_schedule_from_api_tennis(api_key: str, today: str) -> list:
    """Fallback: build schedule from api-tennis fixtures when Sportradar has nothing."""
    matches = []
    for etype, (tour, level, best_of) in ODDS_EVENT_TYPES.items():
        try:
            r = requests.get(BASE_URL, params={
                "method": "get_fixtures", "APIkey": api_key,
                "date_start": today, "date_stop": today,
                "event_type_key": etype,
            }, timeout=15)
            if r.status_code != 200:
                continue
            for e in r.json().get("result", []):
                p1 = (e.get("event_first_player") or "").strip()
                p2 = (e.get("event_second_player") or "").strip()
                league = (e.get("league_name") or "").strip()
                game_time = e.get("event_game_time") or ""
                if not p1 or not p2:
                    continue
                # Parse time
                try:
                    start_dt = datetime.strptime(f"{today} {game_time}", "%Y-%m-%d %H:%M")
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                    chile_time = (start_dt + timedelta(hours=CHILE_OFFSET)).strftime("%H:%M")
                    start_utc_str = start_dt.strftime("%H:%M")
                except Exception:
                    chile_time = game_time
                    start_utc_str = game_time
                # Surface guess from league name
                surf = "Hard"
                ln = league.lower()
                if "clay" in ln or "roland" in ln or "barcelona" in ln or "madrid" in ln:
                    surf = "Clay"
                elif "grass" in ln or "wimbledon" in ln or "halle" in ln or "queens" in ln:
                    surf = "Grass"
                matches.append({
                    "sr_id": str(e.get("event_key", "")),
                    "p1_name": p1, "p2_name": p2,
                    "p1_sr_id": "", "p2_sr_id": "",
                    "start_chile": chile_time,
                    "start_utc_str": start_utc_str,
                    "surface": surf,
                    "category": tour,
                    "competition": league,
                    "best_of": best_of,
                    "round_name": "R32",
                    "status": "not_started",
                    "is_live": False,
                    "gender": "women" if tour == "WTA" else "men",
                    "event_key": e.get("event_key"),
                })
        except Exception:
            pass
    return matches


def run(date_str: str = None, hours_ahead: float = 5.0):
    today = date_str or date.today().isoformat()
    now_utc = utc_now()
    chile_now = chile_str(now_utc)

    print(f"\n{'='*65}")
    print(f"  Today's Tennis Predictions — {today}")
    print(f"  Current time: {chile_now} CL (UTC-3)  |  Showing next {hours_ahead:.0f}h")
    print(f"{'='*65}")

    # 1. Get match schedule from Sportradar
    schedule = get_todays_atp_wta(hours_ahead=hours_ahead)
    if not schedule:
        print("  Sportradar empty — falling back to api-tennis fixtures...")
        api_key = get_api_key()
        if api_key:
            schedule = fetch_schedule_from_api_tennis(api_key, today)
        if not schedule:
            print("  No upcoming matches found.")
            return []
        print(f"  Found {len(schedule)} matches from api-tennis.")

    # 2. Get odds lookup from api-tennis.com
    api_key = get_api_key()
    odds_lookup = {}
    all_events_raw = []
    if api_key:
        print(f"  Fetching odds from api-tennis.com...")
        try:
            for etype in ODDS_EVENT_TYPES:
                r = requests.get(BASE_URL, params={
                    "method": "get_fixtures", "APIkey": api_key,
                    "date_start": today, "date_stop": today,
                    "event_type_key": etype,
                }, timeout=15)
                if r.status_code == 200:
                    all_events_raw.extend(r.json().get("result", []))
            # Build lookup: by full-name AND by last-name pair (both orderings)
            for e in all_events_raw:
                p1 = (e.get("event_first_player") or "").strip()
                p2 = (e.get("event_second_player") or "").strip()
                ekey = e.get("event_key")
                if p1 and p2 and ekey:
                    odds_lookup[f"{_norm(p1)}|{_norm(p2)}"] = ekey
                    odds_lookup[f"{_norm(p2)}|{_norm(p1)}"] = ekey
                    # Last-name fallback
                    l1, l2 = _last(p1), _last(p2)
                    odds_lookup[f"L:{l1}|{l2}"] = ekey
                    odds_lookup[f"L:{l2}|{l1}"] = ekey
        except Exception as ex:
            print(f"  Odds fetch error: {ex}")

    # 3. Run predictions
    from src.models.predict import predict_match
    from src.models.predict_wta import predict_wta_match

    all_results = []
    value_bets  = []

    for m in schedule:
        p1_sr = m["p1_name"]
        p2_sr = m["p2_name"]
        gender = m.get("gender", "")
        cat    = m.get("category", "")
        comp   = m.get("competition", "")
        surf   = m["surface"]
        level  = _level(cat, comp, gender)
        best_of = m.get("best_of", 3)
        rnd    = m.get("round_name", "R32")
        tour   = "WTA" if gender == "women" or "WTA" in cat else "ATP"

        # Try to find odds
        p1_odds = p2_odds = bookie = None
        key1 = f"{_norm(p1_sr)}|{_norm(p2_sr)}"
        key2 = f"{_norm(p2_sr)}|{_norm(p1_sr)}"
        lkey = f"L:{_last(p1_sr)}|{_last(p2_sr)}"
        ekey = odds_lookup.get(key1) or odds_lookup.get(key2) or odds_lookup.get(lkey)
        if ekey and api_key:
            p1_odds, p2_odds, bookie = fetch_single_odds(api_key, ekey)

        # Run prediction
        try:
            if tour == "WTA":
                r = predict_wta_match(
                    p1_sr, p2_sr, surf, level=level,
                    best_of=best_of, round_name=rnd, verbose=False,
                    p1_odds=p1_odds, p2_odds=p2_odds
                )
            else:
                r = predict_match(
                    p1_sr, p2_sr, surf, level=level,
                    best_of=best_of, round_name=rnd, verbose=False,
                    p1_odds=p1_odds, p2_odds=p2_odds
                )
        except Exception as ex:
            r = {"error": str(ex)}

        if "error" in r:
            continue

        # Enrich with schedule info
        r["event_time"]   = m["start_chile"]          # Chilean time HH:MM
        r["event_time_utc"] = m["start_utc_str"]       # UTC HH:MM
        r["tournament"]   = comp
        r["tour"]         = tour
        r["is_live"]      = m["is_live"]
        r["sr_id"]        = m["sr_id"]
        r["bookmaker"]    = bookie

        all_results.append(r)

        odds_str = f" [{bookie}: {p1_odds}/{p2_odds}]" if p1_odds else " [no odds]"
        live_tag = " ⚡LIVE" if m["is_live"] else ""
        if r.get("value_bets"):
            value_bets.extend(r["value_bets"])
            b = r["value_bets"][0]
            print(f"  ★ {m['start_chile']}CL {r['p1_name']} vs {r['p2_name']}{odds_str}{live_tag}"
                  f" → BET {b['bet_on']} @ {b['odds']} | Edge {b['edge_pct']}")
        else:
            winner = r.get("predicted_winner", "?")
            conf   = r.get("confidence", 0) * 100
            print(f"    {m['start_chile']}CL {r['p1_name']} vs {r['p2_name']}{odds_str}{live_tag}"
                  f" → {winner} {conf:.0f}%")

    print(f"\n  Matches predicted: {len(all_results)}  |  Value bets: {len(value_bets)}")

    # Save for dashboard
    out = {
        "date": today,
        "generated_at": now_utc.isoformat(),
        "generated_chile": chile_now,
        "hours_ahead": hours_ahead,
        "predictions": all_results,
        "value_bets":  value_bets,
    }
    out_path = os.path.join(os.path.dirname(__file__), "../../data/todays_picks.json")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  Saved → data/todays_picks.json")
    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date",  default=None, help="YYYY-MM-DD")
    p.add_argument("--hours", default=5.0,  type=float, help="hours ahead window")
    args = p.parse_args()
    run(args.date, args.hours)
