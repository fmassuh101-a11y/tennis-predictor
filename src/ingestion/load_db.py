"""
Load all downloaded Sackmann CSVs into a unified SQLite database.
Creates: data/tennis.db with tables: matches, players, rankings
"""

import os
import glob
import sqlite3
import pandas as pd
from tqdm import tqdm

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
ATP_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/sackmann/atp")
ITF_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/sackmann/itf")

# These are the columns we care about — ignore the rest for now
MATCH_COLUMNS = {
    "tourney_id": str,
    "tourney_name": str,
    "surface": str,
    "draw_size": "Int64",
    "tourney_level": str,       # G=GrandSlam, M=Masters, A=ATP, C=Challenger, S=Satellite/ITF
    "tourney_date": str,
    "match_num": "Int64",
    "winner_id": "Int64",
    "winner_seed": "Int64",
    "winner_entry": str,
    "winner_name": str,
    "winner_hand": str,
    "winner_ht": "Int64",
    "winner_ioc": str,
    "winner_age": float,
    "winner_rank": "Int64",
    "winner_rank_points": "Int64",
    "loser_id": "Int64",
    "loser_seed": "Int64",
    "loser_entry": str,
    "loser_name": str,
    "loser_hand": str,
    "loser_ht": "Int64",
    "loser_ioc": str,
    "loser_age": float,
    "loser_rank": "Int64",
    "loser_rank_points": "Int64",
    "score": str,
    "best_of": "Int64",
    "round": str,
    "minutes": "Int64",
    # Serve stats - winner
    "w_ace": "Int64",
    "w_df": "Int64",
    "w_svpt": "Int64",
    "w_1stIn": "Int64",
    "w_1stWon": "Int64",
    "w_2ndWon": "Int64",
    "w_SvGms": "Int64",
    "w_bpSaved": "Int64",
    "w_bpFaced": "Int64",
    # Serve stats - loser
    "l_ace": "Int64",
    "l_df": "Int64",
    "l_svpt": "Int64",
    "l_1stIn": "Int64",
    "l_1stWon": "Int64",
    "l_2ndWon": "Int64",
    "l_SvGms": "Int64",
    "l_bpSaved": "Int64",
    "l_bpFaced": "Int64",
}


def normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep only known columns, fill missing ones with NA, add source tag."""
    for col, dtype in MATCH_COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.NA
        else:
            try:
                df[col] = df[col].astype(dtype)
            except Exception:
                df[col] = pd.NA

    df = df[list(MATCH_COLUMNS.keys())].copy()
    df["source"] = source
    return df


def load_atp_matches(conn: sqlite3.Connection):
    """Load all ATP main tour + challenger match files."""
    files = sorted(glob.glob(os.path.join(ATP_DIR, "atp_matches_*.csv")))
    print(f"\nFound {len(files)} ATP match files")

    all_dfs = []
    for fpath in tqdm(files, desc="Loading ATP CSVs"):
        try:
            df = pd.read_csv(fpath, low_memory=False)
            fname = os.path.basename(fpath)
            source = "atp_challenger" if "qual_chall" in fname else "atp_main"
            df = normalize_columns(df, source)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_sql("matches", conn, if_exists="append", index=False)
        print(f"Loaded {len(combined):,} ATP matches into database")


def load_itf_matches(conn: sqlite3.Connection):
    """Load all ITF men's match files."""
    files = sorted(glob.glob(os.path.join(ITF_DIR, "itf_men_*.csv")))
    print(f"\nFound {len(files)} ITF match files")

    all_dfs = []
    for fpath in tqdm(files, desc="Loading ITF CSVs"):
        try:
            df = pd.read_csv(fpath, low_memory=False)
            df = normalize_columns(df, "itf")
            all_dfs.append(df)
        except Exception as e:
            print(f"  Error loading {fpath}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_sql("matches", conn, if_exists="append", index=False)
        print(f"Loaded {len(combined):,} ITF matches into database")


def load_players(conn: sqlite3.Connection):
    """Load player index."""
    fpath = os.path.join(ATP_DIR, "atp_players.csv")
    if not os.path.exists(fpath):
        print("Player file not found, skipping.")
        return
    df = pd.read_csv(fpath, low_memory=False)
    df.to_sql("players", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df):,} players")


def create_indexes(conn: sqlite3.Connection):
    """Add indexes to speed up queries."""
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(tourney_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_loser ON matches(loser_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_surface ON matches(surface)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_source ON matches(source)")
    conn.commit()
    print("Indexes created.")


def build_database():
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)

    # Start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    load_players(conn)
    load_atp_matches(conn)
    load_itf_matches(conn)
    create_indexes(conn)

    # Summary
    cursor = conn.cursor()
    cursor.execute("SELECT source, COUNT(*) FROM matches GROUP BY source")
    print("\n=== Database Summary ===")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} matches")
    cursor.execute("SELECT COUNT(*) FROM matches")
    total = cursor.fetchone()[0]
    print(f"  TOTAL: {total:,} matches")

    conn.close()
    db_size = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"\nDatabase saved to: {DB_PATH} ({db_size:.1f} MB)")


if __name__ == "__main__":
    build_database()
