"""
Load ATP/WTA/Challenger match data from api-tennis.com

Sign up at: https://api-tennis.com  (~$40/month)
After subscribing, set your API key:
    export TENNIS_API_KEY="your_key_here"
  OR put it in data/api_tennis_key.txt

This script:
1. Fetches ALL 2025 + 2026 ATP Challenger match results (bulk backfill)
2. Also fetches WTA Challenger (ITF W100/W125) if desired
3. Resolves player names to Sackmann format and inserts into SQLite
4. Can be run daily to pick up new results

Run:
    python -m src.ingestion.load_api_tennis            # full backfill from 2025-01-01
    python -m src.ingestion.load_api_tennis --days 7   # last 7 days only

Event type keys (api-tennis.com):
    265 = ATP Singles (main tour)
    267 = ATP Doubles
    275 = Grand Slam Singles
    277 = Grand Slam Doubles
    279 = WTA Singles (main tour)
    281 = ATP Challenger Singles  ← main target
    282 = ATP Challenger Doubles
    283 = WTA 125 Singles
    285 = ITF Men Singles
    287 = ITF Women Singles
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except ImportError:
    pass
import sys
import sqlite3
import argparse
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from difflib import get_close_matches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

DB_PATH     = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
API_KEY_FILE = os.path.join(os.path.dirname(__file__), "../../data/api_tennis_key.txt")
BASE_URL    = "https://api.api-tennis.com/tennis/"
HEADERS     = {"User-Agent": "Mozilla/5.0"}

# Which event types to import
# Change these to add/remove tour levels
TARGET_EVENT_TYPES = {
    "281": ("atp_challenger",    "C"),  # ATP Challenger Singles
    "265": ("tennis_api_atp",    "A"),  # ATP main tour (backup if tennis-data.co.uk gaps)
    "275": ("tennis_api_slam",   "G"),  # Grand Slams
    "279": ("tennis_api_wta",    "A"),  # WTA main tour (backup)
    "283": ("tennis_api_wta125", "C"),  # WTA 125
}

SURFACE_MAP = {
    "Hard": "Hard", "Clay": "Clay", "Grass": "Grass",
    "Carpet": "Carpet", "Indoor Hard": "Hard", "Outdoor Hard": "Hard",
    "Indoor Clay": "Clay", "Outdoor Clay": "Clay",
}

ROUND_MAP = {
    "Final": "F", "Semi-Final": "SF", "Quarter-Final": "QF",
    "Round of 16": "R16", "Round of 32": "R32",
    "Round of 64": "R64", "Round of 128": "R128",
    "Round Robin": "RR", "1/4": "QF", "1/2": "SF", "1/1": "F",
}


def get_api_key() -> str:
    # 1. Environment variable
    key = os.environ.get("TENNIS_API_KEY", "").strip()
    if key:
        return key
    # 2. Key file
    env = os.environ.get("API_TENNIS_KEY")
    if env:
        return env.strip()
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    return ""


def fetch_events(api_key: str, event_type: str, date_start: str, date_stop: str) -> list:
    """Fetch completed match results for a given event type and date range."""
    params = {
        "method": "get_fixtures",
        "APIkey": api_key,
        "date_start": date_start,
        "date_stop": date_stop,
        "event_type_key": event_type,
    }
    try:
        r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  API HTTP {r.status_code}")
            return []
        data = r.json()
        if data.get("error") != "0":
            print(f"  API error: {data.get('result')}")
            return []
        events = data.get("result", [])
        # Keep only finished matches
        finished = [e for e in events if e.get("event_status") in ("Finished", "After Extra Time", "FT")]
        return finished
    except Exception as e:
        print(f"  Fetch error: {e}")
        return []


def build_name_lookup(conn) -> dict:
    """Build {(lastname_lower, initial_lower): full_name} from DB."""
    df = pd.read_sql_query("""
        SELECT DISTINCT winner_name FROM matches WHERE winner_name IS NOT NULL
        UNION SELECT DISTINCT loser_name FROM matches WHERE loser_name IS NOT NULL
    """, conn)
    lookup = {}
    for name in df.iloc[:, 0]:
        parts = str(name).strip().split()
        if len(parts) < 2:
            continue
        initial = parts[0][0].lower()
        last_only = parts[-1].lower()
        key1 = (last_only, initial)
        if key1 not in lookup:
            lookup[key1] = name
        if len(parts) > 2:
            compound = " ".join(parts[1:]).lower()
            key2 = (compound, initial)
            if key2 not in lookup:
                lookup[key2] = name
    return lookup


def resolve_player_name(api_name: str, lookup: dict, cache: dict) -> str:
    """
    api-tennis.com gives names as 'Lastname, Firstname' or 'Firstname Lastname'.
    Try to match to our canonical full-name format.
    """
    if api_name in cache:
        return cache[api_name]

    name = str(api_name).strip()

    # Format 1: "Lastname, Firstname" → convert to "Firstname Lastname"
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name_fl = f"{parts[1]} {parts[0]}"
    else:
        name_fl = name

    # Try exact match (case-insensitive)
    name_lower = name_fl.lower()
    for k, v in lookup.items():
        if v.lower() == name_lower:
            cache[api_name] = v
            return v

    # Try (last, initial) key
    parts = name_fl.split()
    if len(parts) >= 2:
        initial = parts[0][0].lower()
        last_only = parts[-1].lower()
        key = (last_only, initial)
        if key in lookup:
            cache[api_name] = lookup[key]
            return lookup[key]
        if len(parts) > 2:
            compound = " ".join(parts[1:]).lower()
            key2 = (compound, initial)
            if key2 in lookup:
                cache[api_name] = lookup[key2]
                return lookup[key2]

    # Fuzzy match on the full name
    all_names = [v for v in lookup.values()]
    matches = get_close_matches(name_fl.lower(), [n.lower() for n in all_names], n=1, cutoff=0.80)
    if matches:
        for n in all_names:
            if n.lower() == matches[0]:
                cache[api_name] = n
                return n

    # Not found — store as-is (new player)
    # We'll insert them with their API name and add to lookup for future
    canonical = name_fl
    lookup[(parts[-1].lower() if parts else "?", parts[0][0].lower() if parts else "?")] = canonical
    cache[api_name] = canonical
    return canonical


def events_to_rows(events: list, source: str, level: str, lookup: dict) -> pd.DataFrame:
    cache = {}
    rows = []
    for e in events:
        try:
            home = e.get("event_home_team", "")   # Player 1
            away = e.get("event_away_team", "")   # Player 2
            result = e.get("event_final_result", "")  # e.g. "2 - 0" (sets)

            if not home or not away or not result:
                continue

            # Determine winner from result
            parts = result.strip().split(" - ") if " - " in result else result.split("-")
            if len(parts) != 2:
                continue
            try:
                h_sets = int(parts[0].strip())
                a_sets = int(parts[1].strip())
            except ValueError:
                continue

            if h_sets > a_sets:
                winner_td = home
                loser_td = away
            elif a_sets > h_sets:
                winner_td = away
                loser_td = home
            else:
                continue  # draw / walkover / unclear

            winner_name = resolve_player_name(winner_td, lookup, cache)
            loser_name  = resolve_player_name(loser_td,  lookup, cache)

            # Parse date
            dt_str = e.get("event_date", "")
            try:
                dt = datetime.strptime(dt_str[:10], "%Y-%m-%d")
                tourney_date = int(dt.strftime("%Y%m%d"))
            except Exception:
                continue

            surface_raw = e.get("event_ground", "") or "Hard"
            surface = SURFACE_MAP.get(surface_raw.strip(), "Hard")
            round_raw = e.get("event_status_info", "") or e.get("event_round", "") or "R32"
            round_name = ROUND_MAP.get(round_raw.strip(), "R32")
            tourney_name = e.get("league_name", "") or e.get("tournament_name", "")
            winner_rank = e.get("home_position") if winner_td == home else e.get("away_position")
            loser_rank  = e.get("away_position") if winner_td == home else e.get("home_position")

            rows.append({
                "tourney_date":  tourney_date,
                "tourney_name":  str(tourney_name).strip(),
                "surface":       surface,
                "tourney_level": level,
                "winner_name":   winner_name,
                "loser_name":    loser_name,
                "winner_rank":   winner_rank,
                "loser_rank":    loser_rank,
                "round":         round_name,
                "best_of":       3,
                "source":        source,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def insert_rows(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    if df.empty:
        return 0
    existing = pd.read_sql_query(f"""
        SELECT tourney_date, winner_name, loser_name FROM matches
        WHERE source IN ({','.join('?' * df['source'].nunique())})
    """, conn, params=list(df['source'].unique()))
    existing_keys = set(
        zip(existing["tourney_date"].astype(int),
            existing["winner_name"].astype(str),
            existing["loser_name"].astype(str))
    )
    new = df[~df.apply(
        lambda r: (int(r["tourney_date"]), str(r["winner_name"]), str(r["loser_name"])) in existing_keys,
        axis=1
    )]
    if new.empty:
        return 0
    cursor = conn.execute("PRAGMA table_info(matches)")
    db_cols = {row[1] for row in cursor.fetchall()}
    insert_cols = [c for c in new.columns if c in db_cols]
    new[insert_cols].to_sql("matches", conn, if_exists="append", index=False)
    conn.commit()
    return len(new)


def run(days: int = None, event_types: list = None):
    api_key = get_api_key()
    if not api_key:
        print("\n" + "="*60)
        print("  NO API KEY FOUND")
        print("  Sign up at: https://api-tennis.com")
        print("  Then: echo 'YOUR_KEY' > data/api_tennis_key.txt")
        print("  Or:   export TENNIS_API_KEY=your_key")
        print("="*60 + "\n")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Building name lookup...")
    lookup = build_name_lookup(conn)
    print(f"  {len(lookup):,} known player name keys")

    if days:
        date_start = (date.today() - timedelta(days=days)).isoformat()
    else:
        date_start = "2025-01-01"  # Full 2025 backfill
    date_stop = date.today().isoformat()
    print(f"Fetching matches: {date_start} → {date_stop}")

    if event_types is None:
        event_types = list(TARGET_EVENT_TYPES.keys())

    total_inserted = 0
    for etype in event_types:
        source, level = TARGET_EVENT_TYPES[etype]
        label = source.replace("_", " ").upper()
        print(f"\n  [{label}] event_type={etype}...")

        # Fetch in 90-day chunks to avoid API limits
        start = datetime.strptime(date_start, "%Y-%m-%d")
        stop  = datetime.strptime(date_stop, "%Y-%m-%d")
        chunk_days = 90
        all_events = []

        current = start
        while current <= stop:
            chunk_end = min(current + timedelta(days=chunk_days), stop)
            events = fetch_events(api_key, etype,
                                  current.strftime("%Y-%m-%d"),
                                  chunk_end.strftime("%Y-%m-%d"))
            all_events.extend(events)
            print(f"    {current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}: {len(events)} events")
            current = chunk_end + timedelta(days=1)

        print(f"  Total events fetched: {len(all_events)}")
        df = events_to_rows(all_events, source, level, lookup)
        print(f"  Converted: {len(df)} rows")

        n = insert_rows(df, conn)
        total_inserted += n
        print(f"  Inserted {n} new matches (source={source})")

    conn.close()
    print(f"\nTotal new matches inserted: {total_inserted}")
    print("\nNext step: rebuild player states")
    print("  python -m src.models.player_state")
    print("  python -m src.models.player_state_wta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="Only fetch last N days (default: full 2025 backfill)")
    parser.add_argument("--types", nargs="+", default=None,
                        help="Event type keys to fetch (default: all in TARGET_EVENT_TYPES)")
    args = parser.parse_args()
    run(days=args.days, event_types=args.types)
