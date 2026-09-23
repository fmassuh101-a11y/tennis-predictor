"""
Pull recent match summaries for each player via competitor summaries endpoint.
Adds matches not captured by daily summaries (different tournaments, exhibitions, etc.)
Each call returns up to 30 recent matches WITH serve stats for that player.

Run: python -m src.ingestion.load_sportradar_competitor_summaries
Estimated calls: up to 260 (top 260 players by match count)
"""

import os, sys, json, sqlite3, time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.sportradar import _get, extract_match_stats, _surface_from_name, _parse_round

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/sportradar_matches.db")

WANTED_CATS = {"ATP", "WTA", "Grand Slam", "Challenger", "WTA 125K", "ITF Men", "ITF Women"}


def _level(cat, comp_name):
    cn = comp_name.lower()
    cat_ = cat.lower()
    if any(x in cn for x in ["australian open", "us open", "french open", "wimbledon", "roland garros"]):
        return "G"
    if "grand slam" in cat_: return "G"
    if "atp" in cat_:
        if any(x in cn for x in ["masters", "1000", "indian wells", "miami", "rome", "madrid",
                                   "paris", "montreal", "toronto", "cincinnati", "shanghai", "beijing"]):
            return "M"
        if "challenger" in cn: return "C"
        return "A"
    if "wta" in cat_:
        if any(x in cn for x in ["1000", "indian wells", "miami", "madrid", "rome", "dubai",
                                   "beijing", "montreal", "toronto", "cincinnati", "wuhan"]):
            return "M"
        if "125" in cn: return "C"
        return "A"
    if "challenger" in cat_ or "challenger" in cn: return "C"
    if "itf" in cat_: return "F"  # ITF level
    return "A"


def process_competitor_summary(s: dict) -> dict:
    se = s.get("sport_event", {})
    ctx = se.get("sport_event_context", {})
    cat = ctx.get("category", {}).get("name", "")
    comp = ctx.get("competition", {})
    mode = ctx.get("mode", {})
    rnd = ctx.get("round", {})
    status = s.get("sport_event_status", {})

    if status.get("status") not in ("closed", "ended"):
        return None

    players = se.get("competitors", [])
    if len(players) < 2:
        return None

    if comp.get("type", "") != "singles":
        return None

    comp_name = comp.get("name", "")
    gender = comp.get("gender", "")
    best_of = mode.get("best_of", 3)
    season = ctx.get("season", {})

    # Get match date from scheduled field
    scheduled = se.get("scheduled", "")
    if scheduled:
        match_date = scheduled[:10]
    else:
        return None

    from src.ingestion.load_sportradar_history import process_summary
    return process_summary(s, match_date)


def run(max_players: int = 260):
    conn = sqlite3.connect(DB_PATH)

    # Get SR IDs for top N players
    players = conn.execute("""
        SELECT sr_id, name, COUNT(*) n
        FROM (
            SELECT home_sr_id AS sr_id, home_name AS name FROM matches
            UNION ALL
            SELECT away_sr_id, away_name FROM matches
        )
        WHERE sr_id != ''
        GROUP BY sr_id
        ORDER BY n DESC
        LIMIT ?
    """, (max_players,)).fetchall()

    existing = set(r[0] for r in conn.execute("SELECT sr_event_id FROM matches").fetchall())
    print(f"Pulling competitor summaries for {len(players)} players")
    print(f"Existing matches in DB: {len(existing):,}")

    fetched_total = 0
    new_total = 0
    errors = 0

    for i, (sr_id, name, _) in enumerate(players):
        data = _get(f"competitors/{sr_id}/summaries.json")
        if not data:
            errors += 1
            continue

        summaries = data.get("summaries", [])
        new_rows = []

        from src.ingestion.load_sportradar_history import process_summary
        for s in summaries:
            se = s.get("sport_event", {})
            scheduled = se.get("scheduled", "")
            if not scheduled:
                continue
            match_date = scheduled[:10]

            row = process_summary(s, match_date)
            if row and row["sr_event_id"] and row["sr_event_id"] not in existing:
                new_rows.append(row)
                existing.add(row["sr_event_id"])

        if new_rows:
            cols = list(new_rows[0].keys())
            placeholders = ",".join("?" * len(cols))
            col_str = ",".join(cols)
            conn.executemany(
                f"INSERT OR IGNORE INTO matches ({col_str}) VALUES ({placeholders})",
                [tuple(r[c] for c in cols) for r in new_rows]
            )
            conn.commit()
            new_total += len(new_rows)

        fetched_total += 1
        if fetched_total % 20 == 0:
            print(f"  {fetched_total}/{len(players)} players | {new_total} new matches | {errors} errors")

    conn.close()

    total = sqlite3.connect(DB_PATH).execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"\nDone. Players processed: {fetched_total} | New matches: {new_total} | Errors: {errors}")
    print(f"Total in sportradar_matches.db: {total:,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=260)
    args = p.parse_args()
    run(args.max)
