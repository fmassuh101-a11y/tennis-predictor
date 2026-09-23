"""
Pull point-by-point timeline for every match in sportradar_matches.db.
Stores in data/sportradar_timelines.db for later feature computation.

Each timeline gives per-point data:
  - Who served, who won the point, score state
  - Computed features: serve win %, BP conv under pressure, tiebreak rate, momentum

Run with new API key (needs ~20k calls for all 2025-2026 matches, prioritize by level):
  python -m src.ingestion.load_sportradar_timelines --level G M A    # Grand Slams + Masters + ATP 500
  python -m src.ingestion.load_sportradar_timelines --level all       # Everything

Timeline features computed per player per match:
  - svpt_won / svpt_total → serve win %
  - bp_saved / bp_faced → BP save %
  - bp_won / bp_opp → BP conversion %
  - tb_pts_won / tb_pts_total → tiebreak win %
  - pressure_pts_won: BP/set point situations
"""

import os, sys, json, sqlite3
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.sportradar import _get

MATCH_DB    = os.path.join(os.path.dirname(__file__), "../../data/sportradar_matches.db")
TIMELINE_DB = os.path.join(os.path.dirname(__file__), "../../data/sportradar_timelines.db")


def init_timeline_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS timelines (
        sr_event_id     TEXT PRIMARY KEY,
        match_date      TEXT,
        home_sr_id      TEXT,
        away_sr_id      TEXT,
        -- Computed serve stats from timeline
        home_svpt_won   INTEGER,
        home_svpt_total INTEGER,
        home_bp_faced   INTEGER,
        home_bp_saved   INTEGER,
        home_bp_opp     INTEGER,  -- break point opportunities on return
        home_bp_won     INTEGER,
        home_tb_pts_won INTEGER,
        home_tb_pts_total INTEGER,
        away_svpt_won   INTEGER,
        away_svpt_total INTEGER,
        away_bp_faced   INTEGER,
        away_bp_saved   INTEGER,
        away_bp_opp     INTEGER,
        away_bp_won     INTEGER,
        away_tb_pts_won INTEGER,
        away_tb_pts_total INTEGER,
        -- Raw timeline JSON (for future use)
        timeline_json   TEXT,
        fetched_at      TEXT
    )""")
    conn.commit()


def parse_timeline(events: list, home_sr_id: str, away_sr_id: str) -> dict:
    """Extract serve/break/tiebreak stats from raw timeline events."""
    home = {"svpt_won": 0, "svpt_total": 0, "bp_faced": 0, "bp_saved": 0,
            "bp_opp": 0, "bp_won": 0, "tb_pts_won": 0, "tb_pts_total": 0}
    away = {"svpt_won": 0, "svpt_total": 0, "bp_faced": 0, "bp_saved": 0,
            "bp_opp": 0, "bp_won": 0, "tb_pts_won": 0, "tb_pts_total": 0}

    for e in events:
        if e.get("type") != "point":
            continue

        server = e.get("server")      # "home" or "away"
        winner = e.get("competitor")  # who won the point
        h_score = e.get("home_score", 0)
        a_score = e.get("away_score", 0)

        if server not in ("home", "away"):
            continue

        srv = home if server == "home" else away
        ret = away if server == "home" else home
        srv_won = (winner == server)

        # Service point
        srv["svpt_total"] += 1
        if srv_won:
            srv["svpt_won"] += 1

        # Detect if this was a break point situation:
        # A BP exists when returner is at game point on server's serve
        # In tennis scoring: deuce=40-40. BP = returner leads (e.g. returner at AD or 40-30 with server losing)
        # Sportradar scores use: 0=0, 15=1, 30=2, 40=3, AD=4 (sometimes)
        # We detect BP as: returner_score > server_score (simple heuristic)
        ret_score = a_score if server == "home" else h_score
        srv_score = h_score if server == "home" else a_score

        is_bp = (ret_score > srv_score) and (ret_score >= 40 or (ret_score >= 3))
        if is_bp:
            srv["bp_faced"] += 1
            ret["bp_opp"]   += 1
            if srv_won:
                srv["bp_saved"] += 1
            else:
                ret["bp_won"] += 1

        # Tiebreak detection (when game score is 6-6)
        # We approximate: if both scores reach a high value, it's a tiebreak
        # (In tiebreaks, points go 1,2,3... not 15,30,40)
        # Heuristic: treat as tiebreak if point scores > 40
        if h_score > 40 or a_score > 40:
            srv["tb_pts_total"] += 1
            ret["tb_pts_total"] += 1  # both play tiebreak
            if srv_won:
                srv["tb_pts_won"] += 1
            else:
                ret["tb_pts_won"] += 1

    return {"home": home, "away": away}


def run(levels: list = None, max_matches: int = None):
    """
    Pull timelines.
    levels: list like ['G', 'M', 'A'] — filter by match level. None = all.
    """
    match_conn = sqlite3.connect(MATCH_DB)
    tl_conn    = sqlite3.connect(TIMELINE_DB)
    init_timeline_db(tl_conn)

    # Already fetched
    done = set(r[0] for r in tl_conn.execute("SELECT sr_event_id FROM timelines").fetchall())
    print(f"Already fetched: {len(done)} timelines")

    # Build match list
    query = "SELECT sr_event_id, match_date, home_sr_id, away_sr_id, level FROM matches WHERE winner IS NOT NULL"
    if levels and levels != ["all"]:
        ph = ",".join("?" * len(levels))
        query += f" AND level IN ({ph})"
        matches = match_conn.execute(query, levels).fetchall()
    else:
        matches = match_conn.execute(query).fetchall()

    if max_matches:
        matches = matches[:max_matches]

    print(f"Matches to fetch timelines for: {len(matches):,}")
    match_conn.close()

    fetched = 0
    errors  = 0
    empty   = 0

    for eid, mdate, home_id, away_id, level in matches:
        if eid in done:
            continue

        data = _get(f"sport_events/{eid}/timeline.json")
        if not data:
            errors += 1
            continue

        events = data.get("timeline", [])
        if not events:
            empty += 1
            # Still mark as done so we don't retry
            tl_conn.execute(
                "INSERT OR IGNORE INTO timelines (sr_event_id, match_date, fetched_at) VALUES (?,?,?)",
                (eid, mdate, date.today().isoformat())
            )
            tl_conn.commit()
            continue

        stats = parse_timeline(events, home_id, away_id)
        h = stats["home"]
        a = stats["away"]

        tl_conn.execute("""
            INSERT OR REPLACE INTO timelines (
                sr_event_id, match_date, home_sr_id, away_sr_id,
                home_svpt_won, home_svpt_total, home_bp_faced, home_bp_saved,
                home_bp_opp, home_bp_won, home_tb_pts_won, home_tb_pts_total,
                away_svpt_won, away_svpt_total, away_bp_faced, away_bp_saved,
                away_bp_opp, away_bp_won, away_tb_pts_won, away_tb_pts_total,
                timeline_json, fetched_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            eid, mdate, home_id, away_id,
            h["svpt_won"], h["svpt_total"], h["bp_faced"], h["bp_saved"],
            h["bp_opp"], h["bp_won"], h["tb_pts_won"], h["tb_pts_total"],
            a["svpt_won"], a["svpt_total"], a["bp_faced"], a["bp_saved"],
            a["bp_opp"], a["bp_won"], a["tb_pts_won"], a["tb_pts_total"],
            json.dumps(events),
            date.today().isoformat()
        ))
        tl_conn.commit()
        fetched += 1
        done.add(eid)

        if fetched % 50 == 0:
            total = tl_conn.execute("SELECT COUNT(*) FROM timelines").fetchone()[0]
            print(f"  {fetched} fetched | {errors} errors | {empty} empty | total: {total}")

    tl_conn.close()
    print(f"\nDone. Fetched: {fetched} | Errors: {errors} | Empty: {empty}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--level", nargs="+", default=["G", "M"],
                   help="Match levels to pull (G=GrandSlam M=Masters A=ATP C=Challenger all=everything)")
    p.add_argument("--max", type=int, default=None, help="Max matches to process")
    args = p.parse_args()
    run(levels=args.level, max_matches=args.max)
