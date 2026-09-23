"""
Load ATP 2025 competition data from NoiseQuotient/ATP-Match-Data-Scraper-2025.
Source: https://github.com/NoiseQuotient/ATP-Match-Data-Scraper-2025
File:   data/raw/ATP_2025_Competitions.xlsx

Contains 3,268 ATP matches (Dec 2024 - Sep 2025) with:
  - Full player names (no abbreviation needed)
  - Serve stats: aces, 1st serve %, BP saved, etc.
  - Rank, age, height, hand
  - Level (G=Grand Slam, M=Masters, 500/250=A)

Loaded as source='atp_scraper_2025' to avoid conflict with existing sources.
Run: python -m src.ingestion.load_atp2025_scraper
"""

import os
import re
import sqlite3
import numpy as np
import pandas as pd

DB_PATH   = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
FILE_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/ATP_2025_Competitions.xlsx")
SOURCE    = "atp_scraper_2025"

LEVEL_MAP = {"G": "G", "M": "M", "500/250": "A", "Unknown": "A"}

ROUND_MAP = {
    "Final": "F", "Semifinal": "SF", "Quarterfinal": "QF",
    "Round of 16": "R16", "Round of 32": "R32", "Round of 64": "R64",
    "Round of 128": "R128", "Round Robin": "RR",
    "1st Round": "R64", "2nd Round": "R32", "3rd Round": "R16",
}


def parse_pct(s) -> float | None:
    """Parse '69% (37/54)' or '0.48' → float or None."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    m = re.search(r"(\d+)/(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a / b if b > 0 else None
    try:
        return float(s.rstrip("%")) / 100
    except Exception:
        return None


def parse_int_stat(s) -> float | None:
    """Parse '10.0' or '10' or '10 (...)' → float or None."""
    if pd.isna(s):
        return None
    try:
        return float(str(s).split()[0])
    except Exception:
        return None


def load_file() -> pd.DataFrame:
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Not found: {FILE_PATH}\nDownload from: "
                                "https://github.com/NoiseQuotient/ATP-Match-Data-Scraper-2025")
    df = pd.read_excel(FILE_PATH, engine="openpyxl")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Winner", "Loser"])
    df = df[df["Winner"].notna() & df["Loser"].notna()]
    print(f"Loaded {len(df):,} rows from {os.path.basename(FILE_PATH)}")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        dt = row["Date"]
        tourney_date = int(dt.strftime("%Y%m%d"))

        winner = str(row["Winner"]).strip()
        loser  = str(row["Loser"]).strip()
        if not winner or not loser or winner == loser:
            continue

        level    = LEVEL_MAP.get(str(row.get("Level", "")).strip(), "A")
        surface  = str(row.get("Surface", "Hard")).strip().capitalize()
        round_td = str(row.get("Round", "")).strip()
        round_c  = ROUND_MAP.get(round_td, "R32")
        best_of  = int(row.get("BestOf", 3) or 3) if "BestOf" in df.columns else 3

        # Figure out which player is P1 and which is P2 to get per-player serve stats
        p1_is_winner = str(row.get("Player1 Name", "")).strip() == winner

        def p(col_w, col_l):
            """Return (winner_val, loser_val) from P1/P2 columns."""
            w_col = col_w if p1_is_winner else col_l
            l_col = col_l if p1_is_winner else col_w
            return row.get(w_col), row.get(l_col)

        # Serve stats
        w_aces_raw, l_aces_raw = p("Aces P1", "Aces P2")
        w_ace = parse_int_stat(w_aces_raw)
        l_ace = parse_int_stat(l_aces_raw)

        w_df_raw, l_df_raw = p("Double Faults P1", "Double Faults P2")
        w_df_val = parse_int_stat(w_df_raw)
        l_df_val = parse_int_stat(l_df_raw)

        # 1stIn: parse "48% (54/112)" to get numerator/denominator
        w_1st_raw, l_1st_raw = p("1st Serve P1", "1st Serve P2")
        def parse_fraction(s):
            if pd.isna(s): return None, None
            m = re.search(r"\((\d+)/(\d+)\)", str(s))
            return (int(m.group(1)), int(m.group(2))) if m else (None, None)

        w_1stIn, w_svpt = parse_fraction(w_1st_raw)
        l_1stIn, l_svpt = parse_fraction(l_1st_raw)

        w_1stWon_raw, l_1stWon_raw = p("1st Serve Points Won P1", "1st Serve Points Won P2")
        w_1stWon_n, _ = parse_fraction(w_1stWon_raw)
        l_1stWon_n, _ = parse_fraction(l_1stWon_raw)

        w_bpSaved_raw, l_bpSaved_raw = p("Break Points Saved P1", "Break Points Saved P2")
        w_bpSaved_n, w_bpFaced_n = parse_fraction(w_bpSaved_raw)
        l_bpSaved_n, l_bpFaced_n = parse_fraction(l_bpSaved_raw)

        w_SvGms_raw, l_SvGms_raw = p("Service Games Played P1", "Service Games Played P2")
        w_SvGms = parse_int_stat(w_SvGms_raw)
        l_SvGms = parse_int_stat(l_SvGms_raw)

        # Rank
        w_rank_raw, l_rank_raw = p("P1 Rank", "P2 Rank")
        try: w_rank = float(w_rank_raw) if pd.notna(w_rank_raw) else None
        except: w_rank = None
        try: l_rank = float(l_rank_raw) if pd.notna(l_rank_raw) else None
        except: l_rank = None

        rows.append({
            "tourney_date":   tourney_date,
            "tourney_name":   str(row.get("Tournament", "")).strip(),
            "surface":        surface,
            "tourney_level":  level,
            "winner_name":    winner,
            "loser_name":     loser,
            "winner_rank":    w_rank,
            "loser_rank":     l_rank,
            "score":          str(row.get("Score", "") or "").strip(),
            "round":          round_c,
            "best_of":        best_of,
            "w_ace":          w_ace,
            "l_ace":          l_ace,
            "w_df":           w_df_val,
            "l_df":           l_df_val,
            "w_svpt":         w_svpt,
            "l_svpt":         l_svpt,
            "w_1stIn":        w_1stIn,
            "l_1stIn":        l_1stIn,
            "w_1stWon":       w_1stWon_n,
            "l_1stWon":       l_1stWon_n,
            "w_bpSaved":      w_bpSaved_n,
            "w_bpFaced":      w_bpFaced_n,
            "l_bpSaved":      l_bpSaved_n,
            "l_bpFaced":      l_bpFaced_n,
            "w_SvGms":        w_SvGms,
            "l_SvGms":        l_SvGms,
            "source":         SOURCE,
        })

    result = pd.DataFrame(rows)
    print(f"Transformed: {len(result):,} rows with serve stats")
    return result


def insert_into_db(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    existing = pd.read_sql_query(
        f"SELECT tourney_date, winner_name, loser_name FROM matches WHERE source='{SOURCE}'",
        conn
    )
    existing_keys = set(zip(
        existing["tourney_date"].astype(int),
        existing["winner_name"].astype(str),
        existing["loser_name"].astype(str)
    ))

    new_rows = df[~df.apply(
        lambda r: (int(r["tourney_date"]), str(r["winner_name"]), str(r["loser_name"])) in existing_keys,
        axis=1
    )].copy()

    if len(new_rows) == 0:
        print("  No new rows — already up to date.")
        return 0

    cursor = conn.execute("PRAGMA table_info(matches)")
    db_cols = {row[1] for row in cursor.fetchall()}
    insert_cols = [c for c in new_rows.columns if c in db_cols]
    new_rows[insert_cols].to_sql("matches", conn, if_exists="append", index=False)
    conn.commit()
    return len(new_rows)


def run():
    df_raw = load_file()
    df = transform(df_raw)

    conn = sqlite3.connect(DB_PATH)
    n = insert_into_db(df, conn)
    print(f"Inserted {n:,} new matches as source='{SOURCE}'")

    total = pd.read_sql_query(
        f"SELECT COUNT(*) as n FROM matches WHERE source='{SOURCE}'", conn
    ).iloc[0, 0]
    print(f"Total in DB for {SOURCE}: {total:,}")
    conn.close()


if __name__ == "__main__":
    run()
