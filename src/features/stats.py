"""
Rolling serve/return statistics per player.
Only computed where Sackmann has match stats (mainly post-1991 ATP main tour).
No guessing — if stats don't exist for a player, the column stays NaN.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


STAT_WINDOW = 20  # rolling window: last 20 matches with available stats


def _safe_pct(num, denom):
    """Return num/denom or NaN if denom is 0 or NaN."""
    if pd.isna(num) or pd.isna(denom) or denom == 0:
        return np.nan
    return num / denom


def compute_match_stats(row: pd.Series, prefix: str) -> dict:
    """Extract per-match serve stats for winner (w_) or loser (l_)."""
    p = "w" if prefix == "winner" else "l"
    svpt = row.get(f"{p}_svpt")
    first_in = row.get(f"{p}_1stIn")
    first_won = row.get(f"{p}_1stWon")
    second_won = row.get(f"{p}_2ndWon")
    ace = row.get(f"{p}_ace")
    df_count = row.get(f"{p}_df")
    bp_saved = row.get(f"{p}_bpSaved")
    bp_faced = row.get(f"{p}_bpFaced")

    return {
        "1st_in_pct": _safe_pct(first_in, svpt),
        "1st_won_pct": _safe_pct(first_won, first_in),
        "2nd_won_pct": _safe_pct(second_won, svpt - (first_in or 0) if svpt and first_in else None, ),
        "ace_pct": _safe_pct(ace, svpt),
        "df_pct": _safe_pct(df_count, svpt),
        "bp_save_pct": _safe_pct(bp_saved, bp_faced),
    }


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute rolling averages of serve stats for both players
    using only prior matches. Adds columns like:
        winner_serve_1st_in_pct, loser_serve_bp_save_pct, etc.
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    stat_keys = ["1st_in_pct", "1st_won_pct", "2nd_won_pct",
                 "ace_pct", "df_pct", "bp_save_pct"]

    # Rolling buffers per player
    serve_history: dict[str, list[dict]] = defaultdict(list)

    # Initialize output columns
    for role in ["winner", "loser"]:
        for k in stat_keys:
            df[f"{role}_serve_{k}"] = np.nan

    for idx, row in df.iterrows():
        w = str(row["winner_name"]).strip().lower()
        l = str(row["loser_name"]).strip().lower()

        # Write pre-match rolling averages
        for player, role in [(w, "winner"), (l, "loser")]:
            hist = serve_history[player]
            if hist:
                window = hist[-STAT_WINDOW:]
                for k in stat_keys:
                    vals = [h[k] for h in window if not np.isnan(h.get(k, np.nan))]
                    if vals:
                        df.at[idx, f"{role}_serve_{k}"] = np.mean(vals)

        # Compute current match stats and add to history AFTER recording
        w_stats = compute_match_stats(row, "winner")
        l_stats = compute_match_stats(row, "loser")
        serve_history[w].append(w_stats)
        serve_history[l].append(l_stats)

    return df
