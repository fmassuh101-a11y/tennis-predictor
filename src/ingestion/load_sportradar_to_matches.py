"""
Insert Sportradar matches not yet in main matches table.
Also UPDATE existing sportradar rows to fill in count stats from new columns.

Run: python -m src.ingestion.load_sportradar_to_matches
"""

import os, sys, sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

MAIN_DB = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")


def run():
    conn = sqlite3.connect(MAIN_DB)

    # Check existing row count
    before = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Matches before: {before:,}")

    conn.execute("""
        INSERT OR IGNORE INTO matches (
            tourney_date, tourney_name, surface, tourney_level,
            winner_name, loser_name,
            score, round, best_of, source,
            w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon,
            w_bpFaced, w_bpSaved,
            l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon,
            l_bpFaced, l_bpSaved
        )
        SELECT
            REPLACE(sr.match_date, '-', '') AS tourney_date,
            sr.competition                  AS tourney_name,
            sr.surface,
            sr.level                        AS tourney_level,
            sr.winner_name,
            sr.loser_name,
            COALESCE(sr.winner_score || '-' || sr.loser_score, '') AS score,
            sr.round_name                   AS round,
            sr.best_of,
            'sportradar'                    AS source,
            -- Winner serve counts
            sr.w_aces                       AS w_ace,
            sr.w_dfs                        AS w_df,
            sr.w_svpt                       AS w_svpt,
            sr.w_1stIn                      AS w_1stIn,
            sr.w_1stWon                     AS w_1stWon,
            sr.w_2ndWon                     AS w_2ndWon,
            sr.w_bp_face                    AS w_bpFaced,
            sr.w_bp_saved                   AS w_bpSaved,
            -- Loser serve counts
            sr.l_aces                       AS l_ace,
            sr.l_dfs                        AS l_df,
            sr.l_svpt                       AS l_svpt,
            sr.l_1stIn                      AS l_1stIn,
            sr.l_1stWon                     AS l_1stWon,
            sr.l_2ndWon                     AS l_2ndWon,
            sr.l_bp_face                    AS l_bpFaced,
            sr.l_bp_saved                   AS l_bpSaved
        FROM sportradar_raw sr
        WHERE NOT EXISTS (
            SELECT 1 FROM matches m
            WHERE LOWER(m.winner_name) = LOWER(sr.winner_name)
              AND LOWER(m.loser_name)  = LOWER(sr.loser_name)
              AND m.tourney_date = REPLACE(sr.match_date, '-', '')
        )
    """)
    conn.commit()

    after = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"Matches after:  {after:,}")
    print(f"Inserted:       {after - before:,} new matches")

    # Verify by year/source
    res = conn.execute("""
        SELECT SUBSTR(tourney_date,1,4) yr, source, COUNT(*)
        FROM matches
        WHERE source = 'sportradar'
        GROUP BY yr
        ORDER BY yr
    """).fetchall()
    print("\nSportradar matches added by year:")
    for r in res:
        print(f"  {r[0]}: {r[2]:,}")

    conn.close()


if __name__ == "__main__":
    run()
