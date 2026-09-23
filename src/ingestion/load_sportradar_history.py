"""
Pull 2025-2026 match history from Sportradar and store in SQLite.

Strategy: use daily summaries for every day in 2025-2026 (up to today).
Each summary includes: match result, scores, and full serve/return statistics.

This fills our biggest gap: 2025 ATP/WTA main tour + challenger results
with rich stats (aces, DFs, 1st serve %, BP conversion, etc.).

Run: python -m src.ingestion.load_sportradar_history
Estimated calls: ~450 days × 1 call/day = 450 calls (within 1,000 quota)
"""

import os, sys, json, sqlite3
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.sportradar import (
    daily_summaries, extract_match_stats, _surface_from_name, _parse_round
)

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/sportradar_matches.db")

WANTED_CATS = {"ATP", "WTA", "Grand Slam", "Challenger", "WTA 125K"}

# Level mapping
def _level(cat: str, comp_name: str) -> str:
    cn   = comp_name.lower()
    cat_ = cat.lower()
    if any(x in cn for x in ["australian open","us open","french open","wimbledon","roland garros"]):
        return "G"
    if "grand slam" in cat_: return "G"
    if "atp" in cat_:
        if any(x in cn for x in ["masters","1000","indian wells","miami","rome","madrid",
                                   "paris","montreal","toronto","cincinnati","shanghai","beijing"]):
            return "M"
        if "challenger" in cn: return "C"
        return "A"
    if "wta" in cat_:
        if any(x in cn for x in ["1000","indian wells","miami","madrid","rome","dubai",
                                   "beijing","montreal","toronto","cincinnati"]):
            return "M"
        if "125" in cn: return "C"
        return "A"
    if "challenger" in cat_ or "challenger" in cn: return "C"
    return "A"


def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        sr_event_id      TEXT PRIMARY KEY,
        match_date       TEXT,
        category         TEXT,
        competition      TEXT,
        season_id        TEXT,
        surface          TEXT,
        level            TEXT,
        round_name       TEXT,
        gender           TEXT,
        best_of          INTEGER,
        home_sr_id       TEXT,
        home_name        TEXT,
        away_sr_id       TEXT,
        away_name        TEXT,
        winner           TEXT,     -- 'home' or 'away'
        home_score       INTEGER,
        away_score       INTEGER,
        sets_detail      TEXT,     -- JSON
        -- Serve stats (home) — counts
        home_aces        INTEGER,
        home_dfs         INTEGER,
        home_svpt        INTEGER,
        home_1stIn       INTEGER,
        home_1stWon      INTEGER,
        home_2ndIn       INTEGER,
        home_2ndWon      INTEGER,
        home_bp_face     INTEGER,
        home_bp_conv     INTEGER,
        -- Serve stats (home) — pct
        home_first_in_pct    REAL,
        home_first_won_pct   REAL,
        home_second_won_pct  REAL,
        home_bp_conv_pct     REAL,
        home_ace_rate        REAL,
        home_tiebreaks_won   INTEGER,
        -- Serve stats (away) — counts
        away_aces        INTEGER,
        away_dfs         INTEGER,
        away_svpt        INTEGER,
        away_1stIn       INTEGER,
        away_1stWon      INTEGER,
        away_2ndIn       INTEGER,
        away_2ndWon      INTEGER,
        away_bp_face     INTEGER,
        away_bp_conv     INTEGER,
        -- Serve stats (away) — pct
        away_first_in_pct    REAL,
        away_first_won_pct   REAL,
        away_second_won_pct  REAL,
        away_bp_conv_pct     REAL,
        away_ace_rate        REAL,
        away_tiebreaks_won   INTEGER,
        ingested_at      TEXT
    )""")
    conn.commit()


def process_summary(s: dict, match_date: str) -> dict:
    """Extract a match row from a Sportradar summary dict."""
    se     = s.get("sport_event", {})
    ctx    = se.get("sport_event_context", {})
    cat    = ctx.get("category", {}).get("name", "")
    comp   = ctx.get("competition", {})
    mode   = ctx.get("mode", {})
    rnd    = ctx.get("round", {})
    status = s.get("sport_event_status", {})

    if cat not in WANTED_CATS:
        return None
    if comp.get("type", "") != "singles":
        return None
    if status.get("status") not in ("closed", "ended"):
        return None

    players = se.get("competitors", [])
    if len(players) < 2:
        return None

    comp_name = comp.get("name", "")
    season    = ctx.get("season", {})
    gender    = comp.get("gender", "")
    best_of   = mode.get("best_of", 3)

    stats = extract_match_stats(s)

    home = stats.get("home", {})
    away = stats.get("away", {})

    winner_q = stats.get("winner_qualifier")
    if winner_q is None:
        # Sportradar uses winner_id (competitor SR id) instead of home/away for some feeds
        winner_id = status.get("winner_id") or status.get("winner")
        if winner_id:
            if winner_id == players[0].get("id"):
                winner_q = "home"
            elif winner_id == players[1].get("id"):
                winner_q = "away"
            elif winner_id in ("home_team", "home"):
                winner_q = "home"
            elif winner_id in ("away_team", "away"):
                winner_q = "away"
    if winner_q is None:
        return None

    return {
        "sr_event_id":  se.get("id", ""),
        "match_date":   match_date,
        "category":     cat,
        "competition":  comp_name,
        "season_id":    season.get("id", ""),
        "surface":      _surface_from_name(comp_name),
        "level":        _level(cat, comp_name),
        "round_name":   _parse_round(rnd.get("name", "")),
        "gender":       gender,
        "best_of":      best_of,
        "home_sr_id":   players[0].get("id", ""),
        "home_name":    players[0].get("name", ""),
        "away_sr_id":   players[1].get("id", ""),
        "away_name":    players[1].get("name", ""),
        "winner":       winner_q,
        "home_score":   status.get("home_score", 0),
        "away_score":   status.get("away_score", 0),
        "sets_detail":  json.dumps(stats.get("scores", [])),
        # Home stats — counts
        "home_aces":           home.get("aces"),
        "home_dfs":            home.get("double_faults"),
        "home_svpt":           home.get("svpt"),
        "home_1stIn":          home.get("first_in"),
        "home_1stWon":         home.get("first_won"),
        "home_2ndIn":          home.get("second_in"),
        "home_2ndWon":         home.get("second_won"),
        "home_bp_face":        home.get("bp_face"),
        "home_bp_conv":        home.get("bp_converted"),
        # Home stats — pct
        "home_first_in_pct":   home.get("first_serve_in_pct"),
        "home_first_won_pct":  home.get("first_serve_won_pct"),
        "home_second_won_pct": home.get("second_serve_won_pct"),
        "home_bp_conv_pct":    home.get("bp_conv_pct"),
        "home_ace_rate":       home.get("ace_rate"),
        "home_tiebreaks_won":  home.get("tiebreaks_won"),
        # Away stats — counts
        "away_aces":           away.get("aces"),
        "away_dfs":            away.get("double_faults"),
        "away_svpt":           away.get("svpt"),
        "away_1stIn":          away.get("first_in"),
        "away_1stWon":         away.get("first_won"),
        "away_2ndIn":          away.get("second_in"),
        "away_2ndWon":         away.get("second_won"),
        "away_bp_face":        away.get("bp_face"),
        "away_bp_conv":        away.get("bp_converted"),
        # Away stats — pct
        "away_first_in_pct":   away.get("first_serve_in_pct"),
        "away_first_won_pct":  away.get("first_serve_won_pct"),
        "away_second_won_pct": away.get("second_serve_won_pct"),
        "away_bp_conv_pct":    away.get("bp_conv_pct"),
        "away_ace_rate":       away.get("ace_rate"),
        "away_tiebreaks_won":  away.get("tiebreaks_won"),
        "ingested_at":         date.today().isoformat(),
    }


def run(start_date: str = "2025-01-01", end_date: str = None):
    end   = date.today() if end_date is None else date.fromisoformat(end_date)
    start = date.fromisoformat(start_date)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Find already-ingested dates to skip
    done_dates = set(r[0] for r in conn.execute("SELECT DISTINCT match_date FROM matches").fetchall())
    print(f"Already ingested: {len(done_dates)} dates")

    current = start
    total_new = 0
    total_skipped = 0
    calls = 0

    while current <= end:
        d = current.isoformat()
        current += timedelta(days=1)

        if d in done_dates:
            total_skipped += 1
            continue

        summaries = daily_summaries(d)
        calls += 1

        if not summaries:
            continue

        rows = []
        for s in summaries:
            row = process_summary(s, d)
            if row and row["sr_event_id"]:
                rows.append(row)

        if rows:
            cols = list(rows[0].keys())
            placeholders = ",".join("?" * len(cols))
            col_str = ",".join(cols)
            conn.executemany(
                f"INSERT OR IGNORE INTO matches ({col_str}) VALUES ({placeholders})",
                [tuple(r[c] for c in cols) for r in rows]
            )
            conn.commit()
            total_new += len(rows)

        if calls % 20 == 0:
            print(f"  [{d}] {calls} API calls | {total_new} matches ingested so far")

    conn.close()
    print(f"\nDone. API calls used: {calls} | New matches: {total_new} | Dates skipped: {total_skipped}")
    print(f"DB: {DB_PATH}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end",   default=None)
    args = p.parse_args()
    run(args.start, args.end)
