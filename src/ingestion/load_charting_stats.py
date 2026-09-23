"""
Load Jeff Sackmann Match Charting Project stats into player state.

Downloads serve/return/winner stats for ATP and WTA players from:
  https://github.com/JeffSackmann/tennis_MatchChartingProject

Computes per-player averages (last 20 charted matches) and enriches
player_state.pkl and wta_player_state.pkl with real serve/return stats.

Run:
    python -m src.ingestion.load_charting_stats
"""

import os
import sys
import pickle
import requests
import io
import pandas as pd
import numpy as np
from difflib import get_close_matches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

ATP_STATS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/charting-m-stats-Overview.csv"
WTA_STATS_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/charting-w-stats-Overview.csv"

CHARTING_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/charting")

ATP_STATE_PATH = os.path.join(os.path.dirname(__file__), "../../data/player_state.pkl")
WTA_STATE_PATH = os.path.join(os.path.dirname(__file__), "../../data/wta_player_state.pkl")

NUM_COLS = [
    "serve_pts", "aces", "dfs", "first_in", "first_won",
    "second_in", "second_won", "bk_pts", "bp_saved",
    "return_pts", "return_pts_won", "winners", "unforced"
]


def fetch_charting_stats(url: str) -> pd.DataFrame:
    fname = url.split("/")[-1]
    local_path = os.path.join(CHARTING_DIR, fname)
    print(f"  Fetching: {fname}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        os.makedirs(CHARTING_DIR, exist_ok=True)
        with open(local_path, "w") as f:
            f.write(r.text)
        content = r.text
    except Exception as e:
        print(f"  Download failed ({e}), using cached version...")
        if os.path.exists(local_path):
            with open(local_path) as f:
                content = f.read()
        else:
            raise
    df = pd.read_csv(io.StringIO(content))
    df = df[df["set"] == "Total"].copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_player_charting(df: pd.DataFrame, last_n: int = 20) -> dict:
    """
    For each player, compute serve/return rate averages over last N charted matches.
    Returns dict: player_name_lower -> stats_dict
    """
    result = {}
    for player, grp in df.groupby("player"):
        grp = grp.tail(last_n)
        sp = grp["serve_pts"].sum()
        if sp < 10:
            continue
        fi = grp["first_in"].sum()
        fw = grp["first_won"].sum()
        si = grp["second_in"].sum()
        sw = grp["second_won"].sum()
        bkp = grp["bk_pts"].sum()
        bps = grp["bp_saved"].sum()
        rp = grp["return_pts"].sum()
        rpw = grp["return_pts_won"].sum()

        stats = {
            "charting_first_serve_pct":  fi / sp if sp > 0 else np.nan,
            "charting_first_won_pct":    fw / fi if fi > 0 else np.nan,
            "charting_second_won_pct":   sw / si if si > 0 else np.nan,
            "charting_bp_save_pct":      bps / bkp if bkp > 0 else np.nan,
            "charting_return_won_pct":   rpw / rp if rp > 0 else np.nan,
            "charting_aces_per_match":   grp["aces"].mean(),
            "charting_df_per_match":     grp["dfs"].mean(),
            "charting_winners_per_match":grp["winners"].mean(),
            "charting_unforced_per_match":grp["unforced"].mean(),
            "charting_matches":          len(grp),
        }
        result[player.lower().strip()] = stats
    return result


def fuzzy_match(name: str, lookup_keys: list, cache: dict) -> str | None:
    """Find best matching player name from charting data."""
    name_l = name.lower().strip()
    if name_l in cache:
        return cache[name_l]
    # Exact
    if name_l in lookup_keys:
        cache[name_l] = name_l
        return name_l
    # Last name match
    parts = name_l.split()
    last = parts[-1] if parts else ""
    for k in lookup_keys:
        if k.endswith(last) and len(last) > 3:
            cache[name_l] = k
            return k
    # Fuzzy
    matches = get_close_matches(name_l, lookup_keys, n=1, cutoff=0.75)
    if matches:
        cache[name_l] = matches[0]
        return matches[0]
    cache[name_l] = None
    return None


def enrich_state(state_path: str, charting: dict, label: str):
    """Load player state, add charting stats, save back."""
    print(f"\n  Enriching {label} player state...")
    with open(state_path, "rb") as f:
        state = pickle.load(f)

    summaries = state["summaries"]
    lookup_keys = list(charting.keys())
    cache = {}
    enriched = 0

    for name, profile in summaries.items():
        match_key = fuzzy_match(name, lookup_keys, cache)
        if match_key and match_key in charting:
            stats = charting[match_key]
            profile.update(stats)
            enriched += 1

    print(f"  Enriched {enriched}/{len(summaries)} players with charting stats")

    # Save back
    with open(state_path, "wb") as f:
        pickle.dump(state, f)
    print(f"  Saved → {state_path}")


def run():
    print("\n" + "="*60)
    print("  Loading Sackmann Match Charting Stats")
    print("="*60)

    # ATP
    atp_df = fetch_charting_stats(ATP_STATS_URL)
    atp_charting = compute_player_charting(atp_df)
    print(f"  ATP players with charting data: {len(atp_charting)}")
    enrich_state(ATP_STATE_PATH, atp_charting, "ATP")

    # WTA
    wta_df = fetch_charting_stats(WTA_STATS_URL)
    wta_charting = compute_player_charting(wta_df)
    print(f"  WTA players with charting data: {len(wta_charting)}")
    enrich_state(WTA_STATE_PATH, wta_charting, "WTA")

    print("\n  Done. Player states now include real serve/return stats.")
    print("  Next: retrain models to use these features.")
    print("="*60)


if __name__ == "__main__":
    run()
