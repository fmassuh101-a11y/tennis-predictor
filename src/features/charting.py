"""
MatchCharting clutch stats — break point performance under pressure.

Sources: data/raw/charting/charting-{m,w}-stats-KeyPoints{Serve,Return}.csv

For each player we compute rolling (last 20 charted matches):
  - clutch_bp_save_rate:   % of break points SAVED while serving
  - clutch_bp_conv_rate:   % of break points WON while returning

These are NaN if the player has < 3 charted matches (XGBoost handles NaN fine).

Output: dict mapping player_name_lower → sorted list of (date_str, bp_save, bp_conv)
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict

CHARTING_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/charting")


def _normalise(name: str) -> str:
    return str(name).strip().lower()


def _load_bp_stats(serve_path: str, return_path: str) -> pd.DataFrame:
    """
    Load BP rows from serve + return charting files.
    Returns DataFrame with columns: date_str, player, bp_save_rate, bp_conv_rate
    """
    srv = pd.read_csv(serve_path)
    ret = pd.read_csv(return_path)

    # Break-point rows only
    # Serve file uses "BP" (break points server faces)
    # Return file uses "BPO" (break point opportunities for the returner)
    srv_bp = srv[srv["row"] == "BP"].copy()
    ret_bp = ret[ret["row"] == "BPO"].copy()

    # Extract date from match_id (first 8 chars = YYYYMMDD)
    for df in [srv_bp, ret_bp]:
        df["date_str"] = df["match_id"].str[:8]

    # Per player per match: BP save rate (server saving break points)
    srv_agg = (
        srv_bp.groupby(["date_str", "match_id", "player"])
        .agg(bp_pts=("pts", "sum"), bp_saved=("pts_won", "sum"))
        .reset_index()
    )
    srv_agg["bp_save_rate"] = np.where(
        srv_agg["bp_pts"] > 0,
        srv_agg["bp_saved"] / srv_agg["bp_pts"],
        np.nan,
    )

    # Per player per match: BP conversion rate (returner winning break points)
    ret_agg = (
        ret_bp.groupby(["date_str", "match_id", "player"])
        .agg(bp_pts=("pts", "sum"), bp_won=("pts_won", "sum"))
        .reset_index()
    )
    ret_agg["bp_conv_rate"] = np.where(
        ret_agg["bp_pts"] > 0,
        ret_agg["bp_won"] / ret_agg["bp_pts"],
        np.nan,
    )

    # Merge on player + match
    merged = srv_agg[["date_str", "match_id", "player", "bp_save_rate"]].merge(
        ret_agg[["date_str", "match_id", "player", "bp_conv_rate"]],
        on=["date_str", "match_id", "player"],
        how="outer",
    )
    merged["player_lower"] = merged["player"].apply(_normalise)
    merged = merged.sort_values("date_str").reset_index(drop=True)
    return merged


def build_clutch_lookup(gender: str = "m") -> dict:
    """
    Build per-player rolling clutch stats lookup.

    Returns dict: player_name_lower → list of {"date": str, "bp_save": float, "bp_conv": float}
    sorted by date ascending (pre-match values so no leakage).
    """
    serve_path  = os.path.join(CHARTING_DIR, f"charting-{gender}-stats-KeyPointsServe.csv")
    return_path = os.path.join(CHARTING_DIR, f"charting-{gender}-stats-KeyPointsReturn.csv")

    if not os.path.exists(serve_path):
        return {}

    df = _load_bp_stats(serve_path, return_path)

    # Build per-player history sorted chronologically
    history: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        p = row["player_lower"]
        history[p].append({
            "date": row["date_str"],
            "bp_save": row.get("bp_save_rate"),
            "bp_conv": row.get("bp_conv_rate"),
        })

    # Sort by date just in case
    for p in history:
        history[p].sort(key=lambda x: x["date"])

    return dict(history)


def get_rolling_clutch(history: list, before_date: str, window: int = 20) -> dict:
    """
    Given a player's charting history and a match date,
    return their rolling clutch stats computed from matches BEFORE that date.
    """
    past = [h for h in history if h["date"] < before_date]
    if len(past) < 3:
        return {"bp_save": np.nan, "bp_conv": np.nan}

    window_data = past[-window:]
    save_vals = [h["bp_save"] for h in window_data if h["bp_save"] is not None and not (isinstance(h["bp_save"], float) and np.isnan(h["bp_save"]))]
    conv_vals = [h["bp_conv"] for h in window_data if h["bp_conv"] is not None and not (isinstance(h["bp_conv"], float) and np.isnan(h["bp_conv"]))]

    return {
        "bp_save": float(np.mean(save_vals)) if save_vals else np.nan,
        "bp_conv": float(np.mean(conv_vals)) if conv_vals else np.nan,
    }


def build_serve_return_lookup(gender: str = "m") -> dict:
    """
    Build per-player rolling serve/return stats from the Overview charting file.
    Returns dict: player_name_lower → list of (date_str, stats_dict) sorted by date.
    """
    path = os.path.join(CHARTING_DIR, f"charting-{gender}-stats-Overview.csv")
    if not os.path.exists(path):
        return {}

    num_cols = ["serve_pts", "aces", "dfs", "first_in", "first_won",
                "second_in", "second_won", "bk_pts", "bp_saved",
                "return_pts", "return_pts_won", "winners", "unforced"]

    df_c = pd.read_csv(path)
    df_c = df_c[df_c["set"] == "Total"].copy()
    for c in num_cols:
        df_c[c] = pd.to_numeric(df_c[c], errors="coerce")

    df_c["date_str"] = df_c["match_id"].str[:8]
    df_c = df_c.sort_values("date_str")

    history = defaultdict(list)
    for _, row in df_c.iterrows():
        p = _normalise(row["player"])
        sp = row["serve_pts"]
        fi = row["first_in"]
        fw = row["first_won"]
        si = row["second_in"]
        sw = row["second_won"]
        bkp = row["bk_pts"]
        bps = row["bp_saved"]
        rp = row["return_pts"]
        rpw = row["return_pts_won"]
        history[p].append({
            "date": row["date_str"],
            "sp": sp, "fi": fi, "fw": fw, "si": si, "sw": sw,
            "bkp": bkp, "bps": bps, "rp": rp, "rpw": rpw,
            "aces": row["aces"], "dfs": row["dfs"],
            "winners": row["winners"], "unforced": row["unforced"],
        })

    return dict(history)


def get_rolling_serve_return(history: list, before_date: str, window: int = 20) -> dict:
    """Compute rolling serve/return stats from matches BEFORE before_date."""
    past = [h for h in history if h["date"] < before_date]
    if len(past) < 3:
        return {}
    w = past[-window:]

    def safe_rate(num_key, den_key):
        n = sum(h[num_key] for h in w if pd.notna(h.get(num_key)))
        d = sum(h[den_key] for h in w if pd.notna(h.get(den_key)))
        return n / d if d > 0 else np.nan

    def safe_avg(key):
        vals = [h[key] for h in w if pd.notna(h.get(key))]
        return float(np.mean(vals)) if vals else np.nan

    return {
        "charting_first_serve_pct":    safe_rate("fi", "sp"),
        "charting_first_won_pct":      safe_rate("fw", "fi"),
        "charting_second_won_pct":     safe_rate("sw", "si"),
        "charting_bp_save_pct":        safe_rate("bps", "bkp"),
        "charting_return_won_pct":     safe_rate("rpw", "rp"),
        "charting_aces_per_match":     safe_avg("aces"),
        "charting_df_per_match":       safe_avg("dfs"),
        "charting_winners_per_match":  safe_avg("winners"),
        "charting_unforced_per_match": safe_avg("unforced"),
    }


def add_charting_serve_features(df: pd.DataFrame, gender: str = "m") -> pd.DataFrame:
    """
    Add winner/loser charting serve+return features to match DataFrame.
    Temporal safe: only uses charted matches before current match date.
    """
    print(f"Loading charting serve/return data (gender={gender})...")
    lookup = build_serve_return_lookup(gender)
    if not lookup:
        print("  No charting Overview data found — skipping.")
        return df

    print(f"  {len(lookup)} players with charting serve/return history")

    charting_cols = [
        "charting_first_serve_pct", "charting_first_won_pct", "charting_second_won_pct",
        "charting_bp_save_pct", "charting_return_won_pct",
        "charting_aces_per_match", "charting_df_per_match",
        "charting_winners_per_match", "charting_unforced_per_match",
    ]
    for col in charting_cols:
        df[f"winner_{col}"] = np.nan
        df[f"loser_{col}"] = np.nan

    df = df.sort_values("tourney_date").reset_index(drop=True)

    for idx, row in df.iterrows():
        date_str = str(row["tourney_date"])[:8]
        w = _normalise(row["winner_name"])
        l = _normalise(row["loser_name"])

        if w in lookup:
            stats = get_rolling_serve_return(lookup[w], date_str)
            for col, val in stats.items():
                df.at[idx, f"winner_{col}"] = val

        if l in lookup:
            stats = get_rolling_serve_return(lookup[l], date_str)
            for col, val in stats.items():
                df.at[idx, f"loser_{col}"] = val

    non_null = df["winner_charting_first_serve_pct"].notna().sum()
    print(f"  Charting serve stats: {non_null:,}/{len(df):,} matches ({non_null/len(df)*100:.1f}%)")
    return df


def add_clutch_features(df: pd.DataFrame, gender: str = "m") -> pd.DataFrame:
    """
    Add winner_clutch_bp_save, winner_clutch_bp_conv,
        loser_clutch_bp_save,  loser_clutch_bp_conv
    to the match DataFrame using MatchCharting data.
    No leakage: only uses matches charted before the current match date.
    """
    print(f"Loading MatchCharting clutch data (gender={gender})...")
    lookup = build_clutch_lookup(gender)
    if not lookup:
        print("  No MatchCharting data found — skipping clutch features.")
        return df

    matched_players = len(lookup)
    print(f"  {matched_players} players with charting history")

    df = df.sort_values("tourney_date").reset_index(drop=True)

    for col in ["winner_clutch_bp_save", "winner_clutch_bp_conv",
                "loser_clutch_bp_save",  "loser_clutch_bp_conv"]:
        df[col] = np.nan

    for idx, row in df.iterrows():
        date_str = str(row["tourney_date"])[:8]
        w = _normalise(row["winner_name"])
        l = _normalise(row["loser_name"])

        if w in lookup:
            stats = get_rolling_clutch(lookup[w], date_str)
            df.at[idx, "winner_clutch_bp_save"] = stats["bp_save"]
            df.at[idx, "winner_clutch_bp_conv"] = stats["bp_conv"]

        if l in lookup:
            stats = get_rolling_clutch(lookup[l], date_str)
            df.at[idx, "loser_clutch_bp_save"] = stats["bp_save"]
            df.at[idx, "loser_clutch_bp_conv"] = stats["bp_conv"]

    non_null = df["winner_clutch_bp_save"].notna().sum()
    print(f"  Clutch BP save populated for {non_null:,}/{len(df):,} matches ({non_null/len(df)*100:.1f}%)")
    return df
