"""
Load 2025/2026 ATP match results from tennis-data.co.uk Excel files.
These files have winner/loser, surface, date, ranking, set scores, and odds
but no serve stats (ace, df, 1stIn etc.).

Pipeline:
1. Read all 2024-12-29 onward matches from odds_2025.xlsx + odds_2026.xlsx
2. Map "Lastname I." format → "Firstname Lastname" (using Sackmann player DB)
3. Insert into SQLite as source='tennis_data_recent'
4. player_state.py then reads these when rebuilding state

Run: python -m src.ingestion.load_recent
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from difflib import get_close_matches

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
ODDS_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/odds")

# Map tennis-data round names to Sackmann codes
ROUND_MAP = {
    "1st Round":     "R64",
    "2nd Round":     "R32",
    "3rd Round":     "R16",
    "4th Round":     "QF",
    "Quarterfinals": "QF",
    "Semifinals":    "SF",
    "The Final":     "F",
    "Round Robin":   "RR",
}

# Map Series to tourney_level
LEVEL_MAP = {
    "Grand Slam":    "G",
    "Masters 1000":  "M",
    "Masters Cup":   "F",   # year-end finals
    "ATP500":        "A",
    "ATP250":        "A",
}


def parse_td_name(td_name: str) -> tuple[str, str]:
    """
    Parse tennis-data 'Lastname I.' or 'Lastname I.I.' format.
    Returns (lastname_lower, initial_lower) using the FIRST initial only.
    e.g. "Alcaraz C."       → ("alcaraz", "c")
         "Auger-Aliassime F." → ("auger-aliassime", "f")
         "De Minaur A."     → ("de minaur", "a")
         "Bautista Agut R." → ("bautista agut", "r")
         "Cerundolo J.M."   → ("cerundolo", "j")   ← multi-initial: only first counts
    """
    td_name = str(td_name).strip()
    # Remove trailing dot
    td_name = td_name.rstrip(".")
    parts = td_name.split()
    if not parts:
        return ("", "")

    # Find the first part that is a single letter (or looks like an initial)
    # Walk from the right, stop when we hit a clearly non-initial part
    initial_idx = None
    for i in range(len(parts) - 1, -1, -1):
        p = parts[i].rstrip(".")
        # Accept: single letter, or "J.M" style (first char is letter, rest are dots/letters <= 3 chars)
        if len(p) == 1 and p.isalpha():
            initial_idx = i
            break
        elif len(p) <= 3 and p[0].isalpha() and all(c.isalpha() or c == "." for c in p):
            # Could be "J.M" or "F.C" — treat first letter as initial
            initial_idx = i
            break

    if initial_idx is not None and initial_idx > 0:
        initial = parts[initial_idx][0].lower()
        last = " ".join(parts[:initial_idx]).lower()
    else:
        # No clear initial found — treat last word as surname
        initial = parts[0][0].lower()
        last = parts[-1].lower()

    return (last, initial)


def build_name_lookup(conn) -> dict:
    """
    Build lookup {(lastname, initial): sackmann_full_name}
    from players already in the matches table.

    For compound last names like "Alex De Minaur", we add multiple keys:
      ("minaur", "a")       — last word only
      ("de minaur", "a")    — everything after first word (matches tennis-data format)
    """
    df = pd.read_sql_query("""
        SELECT DISTINCT winner_name FROM matches WHERE winner_name IS NOT NULL
        UNION
        SELECT DISTINCT loser_name  FROM matches WHERE loser_name  IS NOT NULL
    """, conn)

    lookup = {}
    for name in df.iloc[:, 0]:
        parts = str(name).strip().split()
        if len(parts) < 2:
            continue
        initial = parts[0][0].lower()

        # Key 1: just the last word ("minaur" for "Alex De Minaur")
        last_only = parts[-1].lower()
        key1 = (last_only, initial)
        if key1 not in lookup:
            lookup[key1] = name

        # Key 2: everything after first name ("de minaur" for "Alex De Minaur")
        # This matches the tennis-data compound-surname format
        if len(parts) > 2:
            compound_last = " ".join(parts[1:]).lower()
            key2 = (compound_last, initial)
            if key2 not in lookup:
                lookup[key2] = name

    return lookup


def resolve_name(td_name: str, lookup: dict, cache: dict) -> str | None:
    """
    Resolve a tennis-data name to Sackmann full name.
    Uses exact lookup first, then fuzzy fallback on last name.
    """
    if td_name in cache:
        return cache[td_name]

    last, initial = parse_td_name(td_name)
    if not last:
        return None

    # Exact key match
    key = (last, initial)
    if key in lookup:
        cache[td_name] = lookup[key]
        return lookup[key]

    # Fuzzy on last name only
    all_lasts = list({k[0] for k in lookup})
    close = get_close_matches(last, all_lasts, n=1, cutoff=0.85)
    if close:
        fuzzy_last = close[0]
        # Among all matches for this last name, try to match initial
        candidates = [lookup[(l, i)] for (l, i) in lookup if l == fuzzy_last]
        if len(candidates) == 1:
            cache[td_name] = candidates[0]
            return candidates[0]
        # Multiple players with same last name — require initial match
        for (l, i) in lookup:
            if l == fuzzy_last and i == initial:
                cache[td_name] = lookup[(l, i)]
                return lookup[(l, i)]

    cache[td_name] = None
    return None


def build_score_string(row) -> str:
    """Build a score string from set scores like '6-4 6-3' or '6-4 3-6 7-6'."""
    parts = []
    for i in range(1, 6):
        w = row.get(f"W{i}")
        l = row.get(f"L{i}")
        if pd.notna(w) and pd.notna(l):
            parts.append(f"{int(w)}-{int(l)}")
        else:
            break
    return " ".join(parts) if parts else ""


def load_recent_matches() -> pd.DataFrame:
    """Load all 2025/2026 matches from tennis-data Excel files."""
    dfs = []
    for year in [2025, 2026]:
        path = os.path.join(ODDS_DIR, f"odds_{year}.xlsx")
        if not os.path.exists(path):
            print(f"  Skipping {year} — file not found")
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df["_source_year"] = year
        dfs.append(df)

    if not dfs:
        print("No recent files found.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Only keep completed matches
    combined = combined[combined.get("Comment", pd.Series(["Completed"] * len(combined))) == "Completed"]
    combined = combined.dropna(subset=["Winner", "Loser", "Date"])
    print(f"Loaded {len(combined):,} completed matches from 2025/2026 files")
    return combined


def transform_to_db_format(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Convert tennis-data rows to Sackmann-compatible DB schema."""
    cache = {}
    rows = []
    unresolved = set()

    for _, row in df.iterrows():
        w_td = str(row["Winner"]).strip()
        l_td = str(row["Loser"]).strip()

        w_name = resolve_name(w_td, lookup, cache)
        l_name = resolve_name(l_td, lookup, cache)

        if w_name is None or l_name is None:
            unresolved.add(w_td if w_name is None else l_td)
            continue

        dt = pd.to_datetime(row["Date"], errors="coerce")
        if pd.isna(dt):
            continue

        tourney_date = int(dt.strftime("%Y%m%d"))
        surface = str(row.get("Surface", "Hard")).strip().capitalize()
        series = str(row.get("Series", "ATP250")).strip()
        level = LEVEL_MAP.get(series, "A")
        round_name = ROUND_MAP.get(str(row.get("Round", "")), "R32")
        best_of = int(row.get("Best of", 3) or 3)
        score = build_score_string(row)

        rows.append({
            "tourney_date": tourney_date,
            "tourney_name": str(row.get("Tournament", "")).strip(),
            "surface": surface,
            "tourney_level": level,
            "winner_name": w_name,
            "loser_name":  l_name,
            "winner_rank": row.get("WRank"),
            "loser_rank":  row.get("LRank"),
            "winner_rank_points": row.get("WPts"),
            "loser_rank_points":  row.get("LPts"),
            "score":   score,
            "round":   round_name,
            "best_of": best_of,
            "source":  "tennis_data_recent",
            # Odds
            "odds_b365_winner": row.get("B365W"),
            "odds_b365_loser":  row.get("B365L"),
            "odds_avg_winner":  row.get("AvgW"),
            "odds_avg_loser":   row.get("AvgL"),
            "odds_max_winner":  row.get("MaxW"),
            "odds_max_loser":   row.get("MaxL"),
        })

    if unresolved:
        print(f"  Could not resolve {len(unresolved)} player names "
              f"(e.g. {list(unresolved)[:5]})")

    return pd.DataFrame(rows)


def insert_into_db(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    """Insert rows into matches table, skipping duplicates."""
    # Get existing (tourney_date, winner_name, loser_name) combos to avoid dupes
    existing = pd.read_sql_query("""
        SELECT tourney_date, winner_name, loser_name FROM matches
        WHERE source = 'tennis_data_recent'
    """, conn)
    existing_keys = set(
        zip(existing["tourney_date"].astype(int),
            existing["winner_name"].astype(str),
            existing["loser_name"].astype(str))
    )

    new_rows = df[~df.apply(
        lambda r: (r["tourney_date"], r["winner_name"], r["loser_name"]) in existing_keys,
        axis=1
    )]

    if len(new_rows) == 0:
        print("  No new rows to insert (all already in DB).")
        return 0

    # Get columns that exist in the matches table
    cursor = conn.execute("PRAGMA table_info(matches)")
    db_cols = {row[1] for row in cursor.fetchall()}

    insert_cols = [c for c in new_rows.columns if c in db_cols]
    new_rows = new_rows[insert_cols]
    new_rows.to_sql("matches", conn, if_exists="append", index=False)
    conn.commit()
    return len(new_rows)


def run():
    conn = sqlite3.connect(DB_PATH)

    print("Building name lookup from existing DB...")
    lookup = build_name_lookup(conn)
    print(f"  {len(lookup):,} known player name keys")

    print("\nLoading recent match files...")
    raw = load_recent_matches()
    if raw.empty:
        conn.close()
        return

    print("\nMapping player names and transforming...")
    transformed = transform_to_db_format(raw, lookup)
    print(f"  {len(transformed):,} / {len(raw):,} matches successfully resolved")

    print("\nInserting into database...")
    n = insert_into_db(transformed, conn)
    print(f"  Inserted {n:,} new matches")

    # Summary
    total = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM matches WHERE source='tennis_data_recent'", conn
    ).iloc[0, 0]
    print(f"  Total recent matches in DB: {total:,}")

    conn.close()
    print("\nDone. Now run: python -m src.models.player_state")


if __name__ == "__main__":
    run()
