"""
Convert Sportradar match history DB → our main SQLite DB format.

Sportradar provides:
  - Match results (winner, scores) for 2025-2026
  - Serve/return stats (aces, DFs, 1st serve %, BP) for ~50% of 2025 + all 2026

This script converts and imports that data into our main DB so it feeds
the ELO chain and feature pipeline.

Run after load_sportradar_history.py completes.
Run: python -m src.ingestion.load_sportradar_to_db
"""

import os, sys, json, sqlite3
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

SR_DB   = os.path.join(os.path.dirname(__file__), "../../data/sportradar_matches.db")
MAIN_DB = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")


def name_sr_to_normal(name: str) -> str:
    """
    Convert Sportradar 'Last, First' → 'First Last'.
    'Sinner, Jannik' → 'Jannik Sinner'
    'Sinner' → 'Sinner'
    """
    if "," in name:
        parts = name.split(",", 1)
        return parts[1].strip() + " " + parts[0].strip()
    return name.strip()


def init_main_schema(conn):
    """Ensure sportradar_raw table exists in main DB."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sportradar_raw (
        sr_event_id     TEXT PRIMARY KEY,
        match_date      TEXT,
        winner_name     TEXT,
        loser_name      TEXT,
        winner_sr_id    TEXT,
        loser_sr_id     TEXT,
        surface         TEXT,
        level           TEXT,
        round_name      TEXT,
        category        TEXT,
        competition     TEXT,
        gender          TEXT,
        best_of         INTEGER,
        winner_score    INTEGER,
        loser_score     INTEGER,
        sets_detail     TEXT,
        -- Winner serve stats (counts)
        w_aces          INTEGER,
        w_dfs           INTEGER,
        w_svpt          INTEGER,
        w_1stIn         INTEGER,
        w_1stWon        INTEGER,
        w_2ndIn         INTEGER,
        w_2ndWon        INTEGER,
        w_bp_face       INTEGER,
        w_bp_saved      INTEGER,
        -- Winner serve stats (pct)
        w_first_in_pct  REAL,
        w_first_won_pct REAL,
        w_second_won_pct REAL,
        w_bp_save_pct   REAL,
        w_ace_rate      REAL,
        -- Loser serve stats (counts)
        l_aces          INTEGER,
        l_dfs           INTEGER,
        l_svpt          INTEGER,
        l_1stIn         INTEGER,
        l_1stWon        INTEGER,
        l_2ndIn         INTEGER,
        l_2ndWon        INTEGER,
        l_bp_face       INTEGER,
        l_bp_saved      INTEGER,
        -- Loser serve stats (pct)
        l_first_in_pct  REAL,
        l_first_won_pct REAL,
        l_second_won_pct REAL,
        l_bp_save_pct   REAL,
        l_ace_rate      REAL
    )""")
    conn.commit()


def run():
    sr_conn   = sqlite3.connect(SR_DB)
    sr_conn.row_factory = sqlite3.Row

    if not os.path.exists(MAIN_DB):
        print(f"Main DB not found at {MAIN_DB}")
        print("Creating it...")

    main_conn = sqlite3.connect(MAIN_DB)
    init_main_schema(main_conn)

    # Check what's already loaded
    done = set(r[0] for r in main_conn.execute("SELECT sr_event_id FROM sportradar_raw").fetchall())
    print(f"Already in main DB: {len(done)} matches")

    rows = sr_conn.execute("""
        SELECT * FROM matches
        WHERE winner IS NOT NULL
        ORDER BY match_date
    """).fetchall()
    sr_conn.close()

    print(f"Sportradar matches to process: {len(rows)}")

    inserted = 0
    skipped  = 0

    for r in rows:
        eid = r["sr_event_id"]
        if eid in done:
            skipped += 1
            continue

        winner_q = r["winner"]   # "home" or "away"
        if winner_q == "home":
            w_name  = name_sr_to_normal(r["home_name"])
            l_name  = name_sr_to_normal(r["away_name"])
            w_sr_id = r["home_sr_id"]
            l_sr_id = r["away_sr_id"]
            w_score = r["home_score"]
            l_score = r["away_score"]
            w_aces    = r["home_aces"];  l_aces    = r["away_aces"]
            w_dfs     = r["home_dfs"];   l_dfs     = r["away_dfs"]
            w_fin     = r["home_first_in_pct"];  l_fin  = r["away_first_in_pct"]
            w_fwon    = r["home_first_won_pct"]; l_fwon = r["away_first_won_pct"]
            w_swon    = r["home_second_won_pct"];l_swon = r["away_second_won_pct"]
            w_ace_r   = r["home_ace_rate"];  l_ace_r = r["away_ace_rate"]
            w_bp_face = r["home_bp_face"];   l_bp_face = r["away_bp_face"]
        else:
            w_name  = name_sr_to_normal(r["away_name"])
            l_name  = name_sr_to_normal(r["home_name"])
            w_sr_id = r["away_sr_id"]
            l_sr_id = r["home_sr_id"]
            w_score = r["away_score"]
            l_score = r["home_score"]
            w_aces    = r["away_aces"];  l_aces  = r["home_aces"]
            w_dfs     = r["away_dfs"];   l_dfs   = r["home_dfs"]
            w_fin     = r["away_first_in_pct"];  l_fin  = r["home_first_in_pct"]
            w_fwon    = r["away_first_won_pct"]; l_fwon = r["home_first_won_pct"]
            w_swon    = r["away_second_won_pct"];l_swon = r["home_second_won_pct"]
            w_ace_r   = r["away_ace_rate"];  l_ace_r = r["home_ace_rate"]
            w_bp_face = r["away_bp_face"];   l_bp_face = r["home_bp_face"]

        # Raw counts — may be None if match was ingested before schema update
        def _get_count(prefix, field):
            try: return r[f"{prefix}_{field}"]
            except Exception: return None

        # BP save pct (still computed from conv pct)
        away_bp_conv = r["away_bp_conv_pct"] if winner_q == "home" else r["home_bp_conv_pct"]
        home_bp_conv = r["home_bp_conv_pct"] if winner_q == "home" else r["away_bp_conv_pct"]
        w_bp_save = (1 - away_bp_conv) if away_bp_conv is not None else None
        l_bp_save = (1 - home_bp_conv) if home_bp_conv is not None else None

        # BP saved count (winner saved = winner_bp_face - (loser_bp_conv from raw))
        w_bpf = w_bp_face
        l_bpf = l_bp_face
        w_bps = round(w_bp_save * w_bpf) if (w_bp_save is not None and w_bpf) else None
        l_bps = round(l_bp_save * l_bpf) if (l_bp_save is not None and l_bpf) else None

        if winner_q == "home":
            w_svpt = _get_count("home", "svpt")
            w_1in  = _get_count("home", "1stIn")
            w_1won = _get_count("home", "1stWon")
            w_2won = _get_count("home", "2ndWon")
            l_svpt = _get_count("away", "svpt")
            l_1in  = _get_count("away", "1stIn")
            l_1won = _get_count("away", "1stWon")
            l_2won = _get_count("away", "2ndWon")
        else:
            w_svpt = _get_count("away", "svpt")
            w_1in  = _get_count("away", "1stIn")
            w_1won = _get_count("away", "1stWon")
            w_2won = _get_count("away", "2ndWon")
            l_svpt = _get_count("home", "svpt")
            l_1in  = _get_count("home", "1stIn")
            l_1won = _get_count("home", "1stWon")
            l_2won = _get_count("home", "2ndWon")

        main_conn.execute("""
            INSERT OR REPLACE INTO sportradar_raw (
                sr_event_id, match_date, winner_name, loser_name,
                winner_sr_id, loser_sr_id,
                surface, level, round_name, category, competition, gender, best_of,
                winner_score, loser_score, sets_detail,
                w_aces, w_dfs, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
                w_bp_face, w_bp_saved,
                w_first_in_pct, w_first_won_pct, w_second_won_pct,
                w_bp_save_pct, w_ace_rate,
                l_aces, l_dfs, l_svpt, l_1stIn, l_1stWon, l_2ndWon,
                l_bp_face, l_bp_saved,
                l_first_in_pct, l_first_won_pct, l_second_won_pct,
                l_bp_save_pct, l_ace_rate
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            eid, r["match_date"], w_name, l_name, w_sr_id, l_sr_id,
            r["surface"], r["level"], r["round_name"],
            r["category"], r["competition"], r["gender"], r["best_of"],
            w_score, l_score, r["sets_detail"],
            w_aces, w_dfs, w_svpt, w_1in, w_1won, w_2won,
            w_bpf, w_bps,
            w_fin, w_fwon, w_swon, w_bp_save, w_ace_r,
            l_aces, l_dfs, l_svpt, l_1in, l_1won, l_2won,
            l_bpf, l_bps,
            l_fin, l_fwon, l_swon, l_bp_save, l_ace_r,
        ))
        inserted += 1

    main_conn.commit()
    main_conn.close()

    print(f"\nInserted: {inserted:,} | Skipped (already done): {skipped:,}")

    # Print summary
    conn = sqlite3.connect(MAIN_DB)
    total = conn.execute("SELECT COUNT(*) FROM sportradar_raw").fetchone()[0]
    with_stats = conn.execute("SELECT COUNT(*) FROM sportradar_raw WHERE w_aces IS NOT NULL").fetchone()[0]
    by_yr = conn.execute("SELECT SUBSTR(match_date,1,4) yr, COUNT(*) FROM sportradar_raw GROUP BY yr").fetchall()
    by_cat = conn.execute("SELECT category, COUNT(*) FROM sportradar_raw GROUP BY category").fetchall()
    conn.close()

    print(f"\nTotal in sportradar_raw: {total:,} | With stats: {with_stats:,}")
    for y in by_yr: print(f"  {y[0]}: {y[1]:,}")
    print("By category:")
    for c in by_cat: print(f"  {c[0]}: {c[1]:,}")


if __name__ == "__main__":
    run()
