"""
Download and load ATP Futures (ITF M25/M15) matches from Jeff Sackmann's tennis_atp repo.
These are the lowest tier ATP matches — essential for building accurate ELO history
for players before they graduate to challenger/main tour level.

Files: atp_matches_futures_YYYY.csv (1991-2024)
~18,000 matches per year → ~600,000 total matches not currently in DB.

Run: python -m src.ingestion.load_futures
"""

import os
import sqlite3
import requests
import pandas as pd
from io import StringIO

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/sackmann/atp")
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"

YEARS = list(range(1991, 2025))  # 1991-2024 (2025 not published yet)

LEVEL_MAP = {
    "F": "F",   # ITF Futures
    "C": "C",   # Challenger (some qualify files also have these)
    "A": "A",   # ATP (rare in futures files)
}


def download_file(year: int) -> pd.DataFrame | None:
    """Download a futures CSV for a given year, use cache if available."""
    fname = f"atp_matches_futures_{year}.csv"
    local = os.path.join(RAW_DIR, fname)

    if os.path.exists(local):
        try:
            df = pd.read_csv(local, low_memory=False)
            return df
        except Exception:
            pass

    url = f"{BASE_URL}/{fname}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 100:
            os.makedirs(RAW_DIR, exist_ok=True)
            with open(local, "wb") as f:
                f.write(r.content)
            df = pd.read_csv(StringIO(r.text), low_memory=False)
            return df
        else:
            return None
    except Exception as e:
        print(f"  Error downloading {year}: {e}")
        return None


def normalize_row(row: pd.Series, year: int) -> dict | None:
    """Convert a Sackmann futures row to our DB schema."""
    winner = str(row.get("winner_name", "")).strip()
    loser  = str(row.get("loser_name", "")).strip()
    if not winner or not loser or winner == "nan" or loser == "nan":
        return None

    def safe_int(v):
        try: return int(float(v))
        except: return None

    def safe_float(v):
        try: return float(v)
        except: return None

    def safe_str(v):
        s = str(v).strip()
        return s if s and s != "nan" else None

    tourney_date = safe_str(row.get("tourney_date"))
    if tourney_date is None:
        return None

    return {
        "tourney_id":         safe_str(row.get("tourney_id")),
        "tourney_name":       safe_str(row.get("tourney_name")),
        "surface":            safe_str(row.get("surface")),
        "draw_size":          safe_int(row.get("draw_size")),
        "tourney_level":      safe_str(row.get("tourney_level")) or "F",
        "tourney_date":       tourney_date,
        "match_num":          safe_int(row.get("match_num")),
        "winner_id":          safe_int(row.get("winner_id")),
        "winner_seed":        safe_int(row.get("winner_seed")),
        "winner_entry":       safe_str(row.get("winner_entry")),
        "winner_name":        winner,
        "winner_hand":        safe_str(row.get("winner_hand")),
        "winner_ht":          safe_int(row.get("winner_ht")),
        "winner_ioc":         safe_str(row.get("winner_ioc")),
        "winner_age":         safe_float(row.get("winner_age")),
        "winner_rank":        safe_int(row.get("winner_rank")),
        "winner_rank_points": safe_int(row.get("winner_rank_points")),
        "loser_id":           safe_int(row.get("loser_id")),
        "loser_seed":         safe_int(row.get("loser_seed")),
        "loser_entry":        safe_str(row.get("loser_entry")),
        "loser_name":         loser,
        "loser_hand":         safe_str(row.get("loser_hand")),
        "loser_ht":           safe_int(row.get("loser_ht")),
        "loser_ioc":          safe_str(row.get("loser_ioc")),
        "loser_age":          safe_float(row.get("loser_age")),
        "loser_rank":         safe_int(row.get("loser_rank")),
        "loser_rank_points":  safe_int(row.get("loser_rank_points")),
        "score":              safe_str(row.get("score")),
        "best_of":            safe_int(row.get("best_of")) or 3,
        "round":              safe_str(row.get("round")),
        "minutes":            safe_int(row.get("minutes")),
        "w_ace":              safe_int(row.get("w_ace")),
        "w_df":               safe_int(row.get("w_df")),
        "w_svpt":             safe_int(row.get("w_svpt")),
        "w_1stIn":            safe_int(row.get("w_1stIn")),
        "w_1stWon":           safe_int(row.get("w_1stWon")),
        "w_2ndWon":           safe_int(row.get("w_2ndWon")),
        "w_SvGms":            safe_int(row.get("w_SvGms")),
        "w_bpSaved":          safe_int(row.get("w_bpSaved")),
        "w_bpFaced":          safe_int(row.get("w_bpFaced")),
        "l_ace":              safe_int(row.get("l_ace")),
        "l_df":               safe_int(row.get("l_df")),
        "l_svpt":             safe_int(row.get("l_svpt")),
        "l_1stIn":            safe_int(row.get("l_1stIn")),
        "l_1stWon":           safe_int(row.get("l_1stWon")),
        "l_2ndWon":           safe_int(row.get("l_2ndWon")),
        "l_SvGms":            safe_int(row.get("l_SvGms")),
        "l_bpSaved":          safe_int(row.get("l_bpSaved")),
        "l_bpFaced":          safe_int(row.get("l_bpFaced")),
        "source":             "atp_futures",
    }


def load_futures(min_year: int = 1991):
    conn = sqlite3.connect(DB_PATH)

    # Check what's already loaded
    existing = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM matches WHERE source='atp_futures'", conn
    ).iloc[0]["n"]
    print(f"Already loaded: {existing:,} ATP futures matches")

    if existing > 100000:
        print("Already have substantial futures data. Use --force to reload.")
        conn.close()
        return

    # Clear and reload
    conn.execute("DELETE FROM matches WHERE source='atp_futures'")
    conn.commit()

    total_inserted = 0
    for year in YEARS:
        if year < min_year:
            continue
        df = download_file(year)
        if df is None or len(df) == 0:
            print(f"  {year}: not found")
            continue

        rows = []
        for _, row in df.iterrows():
            r = normalize_row(row, year)
            if r:
                rows.append(r)

        if not rows:
            print(f"  {year}: 0 valid rows")
            continue

        cols = list(rows[0].keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        vals = [[r.get(c) for c in cols] for r in rows]

        conn.executemany(
            f"INSERT OR IGNORE INTO matches ({col_str}) VALUES ({placeholders})", vals
        )
        conn.commit()
        total_inserted += len(rows)
        print(f"  {year}: +{len(rows):,} futures matches")

    conn.close()
    print(f"\nTotal ATP futures loaded: {total_inserted:,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--min-year", type=int, default=1991)
    args = p.parse_args()
    load_futures(min_year=args.min_year)
