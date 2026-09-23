"""
Pull competitor profiles from Sportradar for all players in our DB.
Profiles contain: nationality, date_of_birth, height, hand, rankings history.

Most valuable: we use SR player IDs from match data to get profiles
for players appearing in 2025-2026 matches (ATP/WTA active players).

Run AFTER load_sportradar_history.py completes.
Run: python -m src.ingestion.load_sportradar_profiles
Estimated calls: ~300 top players × 1 call = 300 calls
"""

import os, sys, json, sqlite3, time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.sportradar import _get

DB_PATH     = os.path.join(os.path.dirname(__file__), "../../data/sportradar_matches.db")
PROFILE_DB  = os.path.join(os.path.dirname(__file__), "../../data/sportradar_profiles.db")


def init_profile_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS profiles (
        sr_id           TEXT PRIMARY KEY,
        name            TEXT,
        nationality     TEXT,
        date_of_birth   TEXT,
        height          INTEGER,
        hand            TEXT,
        gender          TEXT,
        current_rank    INTEGER,
        highest_rank    INTEGER,
        profile_json    TEXT,
        fetched_at      TEXT
    )""")
    conn.commit()


def fetch_profile(sr_id: str) -> dict:
    data = _get(f"competitors/{sr_id}/profile.json")
    return data or {}


def extract_profile_row(sr_id: str, data: dict) -> dict:
    comp = data.get("competitor", {})
    info = comp.get("info", {})
    rankings = data.get("competitor_rankings", [])

    current_rank = None
    highest_rank = None
    for rk in rankings:
        if rk.get("movement_type") == "current":
            current_rank = rk.get("rank")
        if rk.get("movement_type") == "highest":
            highest_rank = rk.get("rank")

    return {
        "sr_id":        sr_id,
        "name":         comp.get("name", ""),
        "nationality":  comp.get("country_code", ""),
        "date_of_birth": info.get("date_of_birth", ""),
        "height":       info.get("height"),
        "hand":         info.get("dominant_hand", ""),
        "gender":       comp.get("gender", ""),
        "current_rank": current_rank,
        "highest_rank": highest_rank,
        "profile_json": json.dumps(data),
        "fetched_at":   date.today().isoformat(),
    }


def run(min_matches: int = 3, max_players: int = 400):
    """
    Pull profiles for players who appear in at least min_matches in our DB.
    Focus on players with 2026 data (most relevant for current predictions).
    """
    # Connect to match DB to get player list
    match_conn = sqlite3.connect(DB_PATH)

    # Get players with most 2026 appearances first
    query = """
    SELECT sr_id, name, COUNT(*) as n_matches
    FROM (
        SELECT home_sr_id AS sr_id, home_name AS name FROM matches WHERE match_date >= '2025-06-01'
        UNION ALL
        SELECT away_sr_id AS sr_id, away_name AS name FROM matches WHERE match_date >= '2025-06-01'
    )
    WHERE sr_id != ''
    GROUP BY sr_id
    HAVING n_matches >= ?
    ORDER BY n_matches DESC
    LIMIT ?
    """
    players = match_conn.execute(query, (min_matches, max_players)).fetchall()
    match_conn.close()

    print(f"Players to fetch profiles for: {len(players)}")

    # Connect to profile DB
    prof_conn = sqlite3.connect(PROFILE_DB)
    init_profile_db(prof_conn)

    # Skip already fetched
    done = set(r[0] for r in prof_conn.execute("SELECT sr_id FROM profiles").fetchall())
    print(f"Already fetched: {len(done)}")

    fetched = 0
    errors  = 0

    for sr_id, name, n in players:
        if sr_id in done:
            continue

        data = fetch_profile(sr_id)
        if not data:
            errors += 1
            continue

        row = extract_profile_row(sr_id, data)
        cols = list(row.keys())
        ph   = ",".join("?" * len(cols))
        prof_conn.execute(
            f"INSERT OR REPLACE INTO profiles ({','.join(cols)}) VALUES ({ph})",
            [row[c] for c in cols]
        )
        prof_conn.commit()
        fetched += 1

        if fetched % 20 == 0:
            print(f"  {fetched} profiles fetched | {errors} errors")

    prof_conn.close()
    print(f"\nDone. Profiles fetched: {fetched} | Errors: {errors}")
    print(f"DB: {PROFILE_DB}")


if __name__ == "__main__":
    run()
