"""
Pull FULL match history from Sportradar — ALL categories including ITF.
Unlike load_sportradar_history.py, this:
  - Includes ITF Men, ITF Women, Juniors, Wheelchair
  - Re-pulls dates already in DB (to catch new categories)
  - Stores in sportradar_full.db (separate from main sportradar_matches.db)

Run when new API key available (needs ~500+ calls for full 2025-2026):
  python -m src.ingestion.load_sportradar_full_history --start 2025-01-01
  python -m src.ingestion.load_sportradar_full_history --start 2024-01-01 --db data/sportradar_2024.db
"""

import os, sys, json, sqlite3
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.sportradar import (
    daily_summaries, extract_match_stats, _surface_from_name, _parse_round
)
from src.ingestion.load_sportradar_history import init_db, process_summary

# All categories — no filter
ALL_CATS = None  # None = accept everything

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/sportradar_full.db")


def _level(cat, comp_name):
    cn = comp_name.lower()
    cat_ = cat.lower()
    if any(x in cn for x in ["australian open", "us open", "french open", "wimbledon", "roland garros"]):
        return "G"
    if "grand slam" in cat_: return "G"
    if "atp" in cat_:
        if any(x in cn for x in ["masters", "1000", "indian wells", "miami", "rome", "madrid",
                                   "paris", "montreal", "toronto", "cincinnati", "shanghai"]):
            return "M"
        if "challenger" in cn: return "C"
        return "A"
    if "wta" in cat_:
        if any(x in cn for x in ["1000", "indian wells", "miami", "madrid", "rome", "dubai",
                                   "beijing", "montreal", "toronto", "cincinnati"]):
            return "M"
        if "125" in cn: return "C"
        return "A"
    if "challenger" in cat_ or "challenger" in cn: return "C"
    if "itf" in cat_: return "F"
    if "junior" in cat_: return "J"
    return "A"


def process_summary_all(s: dict, match_date: str) -> dict:
    """Like process_summary but accepts ALL categories."""
    se = s.get("sport_event", {})
    ctx = se.get("sport_event_context", {})
    cat = ctx.get("category", {}).get("name", "")
    comp = ctx.get("competition", {})
    mode = ctx.get("mode", {})
    rnd = ctx.get("round", {})
    status = s.get("sport_event_status", {})

    if status.get("status") not in ("closed", "ended"):
        return None
    if comp.get("type", "") != "singles":
        return None

    players = se.get("competitors", [])
    if len(players) < 2:
        return None

    comp_name = comp.get("name", "")
    season = ctx.get("season", {})
    gender = comp.get("gender", "")
    best_of = mode.get("best_of", 3)

    stats = extract_match_stats(s)
    home = stats.get("home", {})
    away = stats.get("away", {})

    winner_q = stats.get("winner_qualifier")
    if winner_q is None:
        winner_id = status.get("winner_id") or status.get("winner")
        if winner_id:
            if winner_id == players[0].get("id"): winner_q = "home"
            elif winner_id == players[1].get("id"): winner_q = "away"
            elif winner_id in ("home_team", "home"): winner_q = "home"
            elif winner_id in ("away_team", "away"): winner_q = "away"
    if winner_q is None:
        return None

    return {
        "sr_event_id": se.get("id", ""),
        "match_date": match_date,
        "category": cat,
        "competition": comp_name,
        "season_id": season.get("id", ""),
        "surface": _surface_from_name(comp_name),
        "level": _level(cat, comp_name),
        "round_name": _parse_round(rnd.get("name", "")),
        "gender": gender,
        "best_of": best_of,
        "home_sr_id": players[0].get("id", ""),
        "home_name": players[0].get("name", ""),
        "away_sr_id": players[1].get("id", ""),
        "away_name": players[1].get("name", ""),
        "winner": winner_q,
        "home_score": status.get("home_score", 0),
        "away_score": status.get("away_score", 0),
        "sets_detail": json.dumps(stats.get("scores", [])),
        "home_aces": home.get("aces"), "home_dfs": home.get("double_faults"),
        "home_first_in_pct": home.get("first_serve_in_pct"),
        "home_first_won_pct": home.get("first_serve_won_pct"),
        "home_second_won_pct": home.get("second_serve_won_pct"),
        "home_bp_conv_pct": home.get("bp_conv_pct"),
        "home_bp_face": home.get("bp_face"), "home_bp_conv": home.get("bp_converted"),
        "home_ace_rate": home.get("ace_rate"),
        "away_aces": away.get("aces"), "away_dfs": away.get("double_faults"),
        "away_first_in_pct": away.get("first_serve_in_pct"),
        "away_first_won_pct": away.get("first_serve_won_pct"),
        "away_second_won_pct": away.get("second_serve_won_pct"),
        "away_bp_conv_pct": away.get("bp_conv_pct"),
        "away_bp_face": away.get("bp_face"), "away_bp_conv": away.get("bp_converted"),
        "away_ace_rate": away.get("ace_rate"),
        "ingested_at": date.today().isoformat(),
    }


def run(start_date: str = "2025-01-01", end_date: str = None, db_path: str = None):
    db = db_path or DB_PATH
    end = date.today() if end_date is None else date.fromisoformat(end_date)
    start = date.fromisoformat(start_date)

    conn = sqlite3.connect(db)
    init_db(conn)

    # Only skip dates where we got a full result (not re-pulling partial)
    done_dates = set(r[0] for r in conn.execute(
        "SELECT DISTINCT match_date FROM matches"
    ).fetchall())
    print(f"DB: {db}")
    print(f"Already ingested: {len(done_dates)} dates | Range: {start} → {end}")

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

        if summaries:
            rows = []
            for s in summaries:
                row = process_summary_all(s, d)
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
            print(f"  [{d}] {calls} API calls | {total_new} matches so far")

    conn.close()

    total = sqlite3.connect(db).execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    by_cat = sqlite3.connect(db).execute(
        "SELECT category, COUNT(*) FROM matches GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    print(f"\nDone. Calls: {calls} | New: {total_new} | Skipped: {total_skipped}")
    print(f"Total in {db}: {total:,}")
    print("By category:")
    for c in by_cat: print(f"  {c[0]}: {c[1]:,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--db", default=None, help="Custom DB path")
    args = p.parse_args()
    run(args.start, args.end, args.db)
