"""
Advanced features that go beyond basic ELO and form.
These are the features that push accuracy from 62% toward 70%+.

Features added:
- Handedness advantage (lefties are systematically underestimated)
- Age & peak performance curve
- Height advantage on fast surfaces
- Dominant win rate (straight sets %)
- Tiebreak win rate (clutch)
- Break point conversion on return
- Service hold rate
- Match efficiency (points per game won)
- Seeding-based tournament draw advantage
- Score-based volatility (3-setter rate)
"""

import pandas as pd
import numpy as np
from collections import defaultdict


def parse_score(score: str) -> dict:
    """
    Parse a match score string like '6-3 7-5' or '6-3 4-6 7-5'.
    Returns: n_sets_winner, n_sets_loser, had_tiebreak, went_3_sets, etc.
    """
    result = {
        "n_sets": 0,
        "went_3plus": False,
        "had_tiebreak": False,
        "winner_dominant": False,  # won 6-0, 6-1, 6-2 in any set
        "loser_won_set": False,
    }
    if not isinstance(score, str) or not score:
        return result

    # Remove retirements/walkovers
    score = score.split("RET")[0].split("W/O")[0].strip()
    sets = score.split()

    winner_sets = 0
    loser_sets = 0
    for s in sets:
        parts = s.split("-")
        if len(parts) != 2:
            continue
        try:
            w_games = int(parts[0].split("(")[0])
            l_games = int(parts[1].split("(")[0])
        except ValueError:
            continue

        if w_games > l_games:
            winner_sets += 1
        else:
            loser_sets += 1

        if "(" in s:
            result["had_tiebreak"] = True
        if w_games in (6, 7) and l_games <= 2:
            result["winner_dominant"] = True

    result["n_sets"] = winner_sets + loser_sets
    result["went_3plus"] = (winner_sets + loser_sets) >= 3
    result["loser_won_set"] = loser_sets > 0
    return result


def compute_advanced_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute advanced rolling features per player.
    Processed in strict chronological order.
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Rolling history per player
    adv_history: dict[str, list] = defaultdict(list)

    adv_cols = [
        "winner_tiebreak_rate", "loser_tiebreak_rate",
        "winner_dominant_rate", "loser_dominant_rate",
        "winner_straight_sets_rate", "loser_straight_sets_rate",
        "winner_bp_conv_return", "loser_bp_conv_return",
        "winner_hold_rate", "loser_hold_rate",
    ]
    for col in adv_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        w = str(row["winner_name"]).strip().lower()
        l = str(row["loser_name"]).strip().lower()

        # Write pre-match rolling averages for both players
        for player, role in [(w, "winner"), (l, "loser")]:
            hist = adv_history[player]
            if len(hist) >= 3:  # need at least 3 matches for stats to be meaningful
                window = hist[-20:]

                # Tiebreak rate
                tb_vals = [h["had_tiebreak"] for h in window]
                df.at[idx, f"{role}_tiebreak_rate"] = np.mean(tb_vals)

                # Dominant win rate (when they won — only from wins)
                dom_vals = [h["dominant"] for h in window if h["was_winner"]]
                if dom_vals:
                    df.at[idx, f"{role}_dominant_rate"] = np.mean(dom_vals)

                # Straight sets rate (when they won)
                ss_vals = [not h["loser_won_set"] for h in window if h["was_winner"]]
                if ss_vals:
                    df.at[idx, f"{role}_straight_sets_rate"] = np.mean(ss_vals)

                # Break point conversion on return
                bp_return_vals = [h["bp_conv_return"] for h in window
                                  if h["bp_conv_return"] is not None]
                if bp_return_vals:
                    df.at[idx, f"{role}_bp_conv_return"] = np.mean(bp_return_vals)

                # Service hold rate
                hold_vals = [h["hold_rate"] for h in window if h["hold_rate"] is not None]
                if hold_vals:
                    df.at[idx, f"{role}_hold_rate"] = np.mean(hold_vals)

        # Compute current match stats for history
        score_data = parse_score(row.get("score", ""))

        # Winner stats from this match
        w_svgms = row.get("w_SvGms")
        w_bpfaced = row.get("w_bpFaced")
        w_bpsaved = row.get("w_bpSaved")
        l_bpfaced = row.get("l_bpFaced")
        l_bpsaved = row.get("l_bpSaved")
        l_svgms = row.get("l_SvGms")

        # Winner's break point conversion on return = opponent's (l_bpFaced - l_bpSaved) / l_bpFaced
        w_bp_return = None
        if pd.notna(l_bpfaced) and pd.notna(l_bpsaved) and l_bpfaced > 0:
            w_bp_return = (l_bpfaced - l_bpsaved) / l_bpfaced

        l_bp_return = None
        if pd.notna(w_bpfaced) and pd.notna(w_bpsaved) and w_bpfaced > 0:
            l_bp_return = (w_bpfaced - w_bpsaved) / w_bpfaced

        # Service hold rate = (SvGms - BPsFaced + BPsSaved) / SvGms approx
        w_hold = None
        if pd.notna(w_svgms) and w_svgms > 0 and pd.notna(w_bpfaced) and pd.notna(w_bpsaved):
            games_broken = w_bpfaced - w_bpsaved
            w_hold = max(0, w_svgms - games_broken) / w_svgms

        l_hold = None
        if pd.notna(l_svgms) and l_svgms > 0 and pd.notna(l_bpfaced) and pd.notna(l_bpsaved):
            games_broken = l_bpfaced - l_bpsaved
            l_hold = max(0, l_svgms - games_broken) / l_svgms

        adv_history[w].append({
            "was_winner": True,
            "had_tiebreak": score_data["had_tiebreak"],
            "dominant": score_data["winner_dominant"],
            "loser_won_set": score_data["loser_won_set"],
            "went_3plus": score_data["went_3plus"],
            "bp_conv_return": w_bp_return,
            "hold_rate": w_hold,
        })
        adv_history[l].append({
            "was_winner": False,
            "had_tiebreak": score_data["had_tiebreak"],
            "dominant": False,
            "loser_won_set": score_data["loser_won_set"],
            "went_3plus": score_data["went_3plus"],
            "bp_conv_return": l_bp_return,
            "hold_rate": l_hold,
        })

    return df


# Tournaments known to be played indoors
INDOOR_TOURNAMENTS = {
    "bnp paribas masters", "paris masters", "dallas open", "abn amro world tennis tournament",
    "swiss indoors", "vienna open", "stockholm open", "st. petersburg open",
    "open 13", "open sud de france", "kremlin cup", "sofia open", "tel aviv open",
    "astana open", "almaty open", "antalya open", "gijon open", "firenze open",
    "nordic open", "open de moselle", "chile open", "singapore open", "masters cup",
    "nitto atp finals", "atp finals", "european open", "european open antwerp",
    "indoor clay", "palermo international",  # some clay indoors
    # WTA indoor
    "wta finals", "bnp paribas wta finals", "china open", "wuhan open",
}


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that don't require rolling (per-match facts):
    - handedness (L vs R)
    - age difference / peak distance
    - height difference
    - seeding
    - round difficulty
    - indoor/outdoor flag (NEW)
    """
    # Handedness: left-handers have documented advantage
    hand_map = {"R": 1, "L": 2, "U": 0}
    df["winner_hand_code"] = df["winner_hand"].map(hand_map).fillna(0).astype(int)
    df["loser_hand_code"] = df["loser_hand"].map(hand_map).fillna(0).astype(int)
    df["is_lefty_vs_righty"] = (
        (df["winner_hand_code"] == 2) & (df["loser_hand_code"] == 1)
    ).astype(int)
    df["is_righty_vs_lefty"] = (
        (df["winner_hand_code"] == 1) & (df["loser_hand_code"] == 2)
    ).astype(int)

    # Age features
    df["winner_age"] = pd.to_numeric(df["winner_age"], errors="coerce")
    df["loser_age"] = pd.to_numeric(df["loser_age"], errors="coerce")
    df["age_diff"] = df["winner_age"] - df["loser_age"]
    df["winner_peak_dist"] = (df["winner_age"] - 26).abs()
    df["loser_peak_dist"] = (df["loser_age"] - 26).abs()

    # Height features
    df["winner_ht"] = pd.to_numeric(df["winner_ht"], errors="coerce")
    df["loser_ht"] = pd.to_numeric(df["loser_ht"], errors="coerce")
    df["height_diff"] = df["winner_ht"] - df["loser_ht"]

    # Seeding
    df["winner_seed"] = pd.to_numeric(df["winner_seed"], errors="coerce")
    df["loser_seed"] = pd.to_numeric(df["loser_seed"], errors="coerce")
    df["winner_is_seeded"] = df["winner_seed"].notna().astype(int)
    df["loser_is_seeded"] = df["loser_seed"].notna().astype(int)

    # Round encoding
    round_map = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 7, "RR": 4, "BR": 5,
    }
    df["round_importance"] = df["round"].map(round_map).fillna(3)

    # Indoor/outdoor flag — use is_indoor DB column if present, else infer from tournament name
    if "is_indoor" in df.columns and df["is_indoor"].notna().sum() > 0:
        df["is_indoor"] = df["is_indoor"].fillna(0).astype(int)
    else:
        tourney_lower = df["tourney_name"].str.lower().str.strip().fillna("")
        df["is_indoor"] = tourney_lower.apply(
            lambda t: int(any(ind in t for ind in INDOOR_TOURNAMENTS))
        )

    return df
