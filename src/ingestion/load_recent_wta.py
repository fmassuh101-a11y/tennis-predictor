"""
Load 2025/2026 WTA match results from tennis-data.co.uk Excel files.
Inserts into SQLite as source='tennis_data_wta_recent'.
player_state_wta.py already reads this source when rebuilding state.

Run: python -m src.ingestion.load_recent_wta
"""

import os
import sqlite3
import pandas as pd
from difflib import get_close_matches

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
ODDS_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/odds")

ROUND_MAP = {
    "1st Round":     "R64",
    "2nd Round":     "R32",
    "3rd Round":     "R16",
    "4th Round":     "QF",
    "Quarterfinals": "QF",
    "Semifinals":    "SF",
    "The Final":     "F",
    "Final":         "F",
    "Round Robin":   "RR",
}

TIER_MAP = {
    "Grand Slam":          "G",
    "WTA1000":             "M",
    "Tour Championships":  "F",
    "WTA500":              "A",
    "WTA250":              "A",
    "WTA125":              "C",
}


def parse_td_name(td_name: str) -> tuple:
    td_name = str(td_name).strip().rstrip(".")
    parts = td_name.split()
    if not parts:
        return ("", "")
    initial_idx = None
    for i in range(len(parts) - 1, -1, -1):
        p = parts[i].rstrip(".")
        if len(p) == 1 and p.isalpha():
            initial_idx = i
            break
        elif len(p) <= 3 and p[0].isalpha() and all(c.isalpha() or c == "." for c in p):
            initial_idx = i
            break
    if initial_idx is not None and initial_idx > 0:
        initial = parts[initial_idx][0].lower()
        last = " ".join(parts[:initial_idx]).lower()
    else:
        initial = parts[0][0].lower()
        last = parts[-1].lower()
    return (last, initial)


def build_wta_name_lookup(conn) -> dict:
    df = pd.read_sql_query("""
        SELECT DISTINCT winner_name FROM matches WHERE source='wta_main' AND winner_name IS NOT NULL
        UNION
        SELECT DISTINCT loser_name  FROM matches WHERE source='wta_main' AND loser_name  IS NOT NULL
    """, conn)
    lookup = {}
    for name in df.iloc[:, 0]:
        parts = str(name).strip().split()
        if len(parts) < 2:
            continue
        initial = parts[0][0].lower()
        last_only = parts[-1].lower()
        key1 = (last_only, initial)
        if key1 not in lookup:
            lookup[key1] = name
        if len(parts) > 2:
            compound_last = " ".join(parts[1:]).lower()
            key2 = (compound_last, initial)
            if key2 not in lookup:
                lookup[key2] = name
    return lookup


def resolve_name(td_name: str, lookup: dict, cache: dict):
    if td_name in cache:
        return cache[td_name]
    last, initial = parse_td_name(td_name)
    if not last:
        return None
    key = (last, initial)
    if key in lookup:
        cache[td_name] = lookup[key]
        return lookup[key]
    all_lasts = list({k[0] for k in lookup})
    close = get_close_matches(last, all_lasts, n=1, cutoff=0.85)
    if close:
        fuzzy_last = close[0]
        candidates = [lookup[(l, i)] for (l, i) in lookup if l == fuzzy_last]
        if len(candidates) == 1:
            cache[td_name] = candidates[0]
            return candidates[0]
        for (l, i) in lookup:
            if l == fuzzy_last and i == initial:
                cache[td_name] = lookup[(l, i)]
                return lookup[(l, i)]
    cache[td_name] = None
    return None


def build_score_string(row) -> str:
    parts = []
    for i in range(1, 4):
        w = row.get(f"W{i}")
        l = row.get(f"L{i}")
        if pd.notna(w) and pd.notna(l):
            parts.append(f"{int(w)}-{int(l)}")
        else:
            break
    return " ".join(parts) if parts else ""


def load_wta_recent_matches() -> pd.DataFrame:
    dfs = []
    for year, suffix in [(2025, "wta_2025"), (2026, "wta_2026")]:
        path = os.path.join(ODDS_DIR, f"odds_{suffix}.xlsx")
        if not os.path.exists(path):
            print(f"  Skipping {year} — {path} not found")
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df["_source_year"] = year
        dfs.append(df)
        print(f"  Loaded {len(df)} rows from {path}")
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[
        combined.get("Comment", pd.Series(["Completed"] * len(combined))) == "Completed"
    ]
    combined = combined.dropna(subset=["Winner", "Loser", "Date"])
    print(f"  {len(combined):,} completed WTA matches total")
    return combined


def transform_wta(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
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
        tier = str(row.get("Tier", "WTA250")).strip()
        level = TIER_MAP.get(tier, "A")
        round_name = ROUND_MAP.get(str(row.get("Round", "")), "R32")
        best_of = int(row.get("Best of", 3) or 3)
        score = build_score_string(row)
        rows.append({
            "tourney_date": tourney_date,
            "tourney_name": str(row.get("Tournament", "")).strip(),
            "surface": surface,
            "tourney_level": level,
            "winner_name": w_name,
            "loser_name": l_name,
            "winner_rank": row.get("WRank"),
            "loser_rank": row.get("LRank"),
            "winner_rank_points": row.get("WPts"),
            "loser_rank_points": row.get("LPts"),
            "score": score,
            "round": round_name,
            "best_of": best_of,
            "source": "tennis_data_wta_recent",
            "odds_b365_winner": row.get("B365W"),
            "odds_b365_loser": row.get("B365L"),
            "odds_avg_winner": row.get("AvgW"),
            "odds_avg_loser": row.get("AvgL"),
            "odds_max_winner": row.get("MaxW"),
            "odds_max_loser": row.get("MaxL"),
        })
    if unresolved:
        print(f"  Could not resolve {len(unresolved)} WTA names (e.g. {list(unresolved)[:5]})")
    return pd.DataFrame(rows)


def insert_into_db(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    existing = pd.read_sql_query("""
        SELECT tourney_date, winner_name, loser_name FROM matches
        WHERE source = 'tennis_data_wta_recent'
    """, conn)
    existing_keys = set(
        zip(existing["tourney_date"].astype(int),
            existing["winner_name"].astype(str),
            existing["loser_name"].astype(str))
    )
    new_rows = df[~df.apply(
        lambda r: (int(r["tourney_date"]), str(r["winner_name"]), str(r["loser_name"])) in existing_keys,
        axis=1
    )]
    if len(new_rows) == 0:
        print("  No new rows to insert.")
        return 0
    cursor = conn.execute("PRAGMA table_info(matches)")
    db_cols = {row[1] for row in cursor.fetchall()}
    insert_cols = [c for c in new_rows.columns if c in db_cols]
    new_rows[insert_cols].to_sql("matches", conn, if_exists="append", index=False)
    conn.commit()
    return len(new_rows)


def run():
    conn = sqlite3.connect(DB_PATH)
    print("Building WTA name lookup from wta_main source...")
    lookup = build_wta_name_lookup(conn)
    print(f"  {len(lookup):,} known WTA player name keys")

    print("\nLoading WTA recent files...")
    raw = load_wta_recent_matches()
    if raw.empty:
        conn.close()
        return

    print("\nMapping names and transforming...")
    transformed = transform_wta(raw, lookup)
    print(f"  {len(transformed):,} / {len(raw):,} matches resolved")

    print("\nInserting into database...")
    n = insert_into_db(transformed, conn)
    print(f"  Inserted {n:,} new WTA matches")

    total = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM matches WHERE source='tennis_data_wta_recent'", conn
    ).iloc[0, 0]
    print(f"  Total WTA recent matches in DB: {total:,}")
    conn.close()
    print("\nDone. Now run: python -m src.models.player_state_wta")


if __name__ == "__main__":
    run()
