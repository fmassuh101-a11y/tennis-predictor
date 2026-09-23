"""
Extract features from point-by-point data using correct Sackmann encoding:
  S = server wins point
  R = receiver wins point
  D = double fault (receiver wins)
  A = ace (server wins)
  ; = game separator
  . = set separator (also ends previous game)

From this we compute per player:
  - Ace rate (A / serve points)
  - Double fault rate (D / serve points)
  - First serve win % approximation
  - Service games won rate
  - Return games won rate
  - Tiebreak win rate (most critical — clutch metric)
  - Average rally: S+R length proxy
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

PBP_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/sackmann/pbp")
PBP_STATS_PATH = os.path.join(os.path.dirname(__file__), "../../data/features/pbp_player_stats.csv")

# Only use ATP + Challenger files (not WTA for our model)
USE_PATTERNS = [
    "pbp_matches_atp_main_*.csv",
    "pbp_matches_atp_qual_*.csv",
    "pbp_matches_ch_main_*.csv",
    "pbp_matches_ch_qual_*.csv",
    "pbp_matches_fu_main_*.csv",
    "pbp_matches_itf_main_*.csv",
]


def parse_match_pbp(pbp_str: str, server1: str, server2: str, score: str) -> dict:
    """
    Parse a full match PBP string. Returns per-player stats dict.

    Server alternates every game. server1 serves game 1, server2 game 2, etc.
    Tiebreak detected from score (any 7-6 or game with tiebreak notation).
    """
    stats = {
        server1: {"sv_pts": 0, "sv_wins": 0, "rv_pts": 0, "rv_wins": 0,
                  "aces": 0, "dfs": 0, "sv_games": 0, "sv_games_won": 0,
                  "tb_pts_won": 0, "tb_pts_total": 0, "tb_games_won": 0, "tb_games_total": 0},
        server2: {"sv_pts": 0, "sv_wins": 0, "rv_pts": 0, "rv_wins": 0,
                  "aces": 0, "dfs": 0, "sv_games": 0, "sv_games_won": 0,
                  "tb_pts_won": 0, "tb_pts_total": 0, "tb_games_won": 0, "tb_games_total": 0},
    }

    if not isinstance(pbp_str, str) or not pbp_str:
        return stats

    # Detect if match had tiebreaks from score
    score_str = str(score) if score else ""

    # Split into games by ; and .
    flat = pbp_str.replace(".", ";")
    games = flat.split(";")

    game_idx = 0  # tracks which game we're in (used to detect tiebreaks)
    games_in_set = 0
    set_server_idx = 0  # 0 = server1 starts each set

    for game_str in games:
        game_str = game_str.strip()
        if not game_str:
            games_in_set += 1
            if "." in pbp_str:  # set boundary
                games_in_set = 0
            game_idx += 1
            continue

        # Determine server for this game
        # server1 serves odd games (1,3,5...) in each set, starting game 0
        server_idx = (game_idx + set_server_idx) % 2
        current_server = server1 if server_idx == 0 else server2
        current_returner = server2 if server_idx == 0 else server1

        # Detect tiebreak: usually game 13 (index 12) in a set (6-6)
        is_tiebreak = (games_in_set == 12)  # 13th game = tiebreak

        sv = stats[current_server]
        rv = stats[current_returner]

        sv_wins_game = 0
        rv_wins_game = 0

        for ch in game_str:
            if ch == "S":      # server wins (including serve winner)
                sv["sv_pts"] += 1; rv["rv_pts"] += 1
                sv["sv_wins"] += 1; rv_wins_game += 0; sv_wins_game += 1
                if is_tiebreak:
                    sv["tb_pts_won"] += 1; sv["tb_pts_total"] += 1
                    rv["tb_pts_total"] += 1
            elif ch == "A":    # ace
                sv["sv_pts"] += 1; rv["rv_pts"] += 1
                sv["sv_wins"] += 1; sv["aces"] += 1; sv_wins_game += 1
                if is_tiebreak:
                    sv["tb_pts_won"] += 1; sv["tb_pts_total"] += 1
                    rv["tb_pts_total"] += 1
            elif ch == "D":    # double fault — receiver wins
                sv["sv_pts"] += 1; rv["rv_pts"] += 1
                rv["rv_wins"] += 1; sv["dfs"] += 1; rv_wins_game += 1
                if is_tiebreak:
                    rv["tb_pts_won"] += 1; rv["tb_pts_total"] += 1
                    sv["tb_pts_total"] += 1
            elif ch == "R":    # receiver wins point
                sv["sv_pts"] += 1; rv["rv_pts"] += 1
                rv["rv_wins"] += 1; rv_wins_game += 1
                if is_tiebreak:
                    rv["tb_pts_won"] += 1; rv["tb_pts_total"] += 1
                    sv["tb_pts_total"] += 1

        # Track service games won/lost
        if sv_wins_game > rv_wins_game:
            sv["sv_games_won"] += 1
            if is_tiebreak:
                sv["tb_games_won"] += 1
                rv["tb_games_total"] += 1
                sv["tb_games_total"] += 1
        elif rv_wins_game > sv_wins_game:
            if is_tiebreak:
                rv["tb_games_won"] += 1
                rv["tb_games_total"] += 1
                sv["tb_games_total"] += 1

        sv["sv_games"] += 1
        game_idx += 1
        games_in_set += 1

    return stats


def compute_pbp_player_stats() -> pd.DataFrame:
    """Process all PBP files and compute per-player career stats."""

    all_files = []
    for pattern in USE_PATTERNS:
        all_files.extend(glob.glob(os.path.join(PBP_DIR, pattern)))

    print(f"Processing {len(all_files)} PBP files...")

    all_dfs = []
    for fpath in all_files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
            if len(df) > 0 and "pbp" in df.columns:
                all_dfs.append(df)
        except Exception as e:
            print(f"  Error {fpath}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    pbp_df = pd.concat(all_dfs, ignore_index=True)
    pbp_df = pbp_df.dropna(subset=["server1", "server2"])
    print(f"Total PBP records with data: {len(pbp_df):,}")

    # Aggregate per player
    agg = defaultdict(lambda: {
        "sv_pts": 0, "sv_wins": 0, "rv_pts": 0, "rv_wins": 0,
        "aces": 0, "dfs": 0, "sv_games": 0, "sv_games_won": 0,
        "tb_pts_won": 0, "tb_pts_total": 0,
        "tb_games_won": 0, "tb_games_total": 0, "n_matches": 0,
    })

    for _, row in pbp_df.iterrows():
        s1 = str(row["server1"]).strip()
        s2 = str(row["server2"]).strip()
        pbp = row.get("pbp", "")
        score = row.get("score", "")

        match_stats = parse_match_pbp(pbp, s1, s2, score)

        for player in [s1, s2]:
            ms = match_stats[player]
            ag = agg[player]
            for k in ms:
                ag[k] += ms[k]
            ag["n_matches"] += 1

    # Convert to DataFrame with rates
    rows = []
    for player, s in agg.items():
        if s["n_matches"] < 2:
            continue

        def safe_rate(num, den):
            return round(num / den, 4) if den > 0 else None

        rows.append({
            "player_name": player,
            "pbp_n_matches": s["n_matches"],
            "pbp_ace_rate": safe_rate(s["aces"], s["sv_pts"]),
            "pbp_df_rate": safe_rate(s["dfs"], s["sv_pts"]),
            "pbp_sv_win_rate": safe_rate(s["sv_wins"], s["sv_pts"]),
            "pbp_rv_win_rate": safe_rate(s["rv_wins"], s["rv_pts"]),
            "pbp_sv_game_hold_rate": safe_rate(s["sv_games_won"], s["sv_games"]),
            "pbp_tb_pt_win_rate": safe_rate(s["tb_pts_won"], s["tb_pts_total"]),
            "pbp_tb_game_win_rate": safe_rate(s["tb_games_won"], s["tb_games_total"]),
        })

    result = pd.DataFrame(rows)

    # Rename tb_win_rate to match old column name for compatibility
    result["pbp_tb_win_rate"] = result["pbp_tb_game_win_rate"]
    result["pbp_tb_matches"] = result["pbp_tb_game_win_rate"].notna().astype(int) * result["pbp_n_matches"]

    result.to_csv(PBP_STATS_PATH, index=False)
    print(f"PBP player stats saved: {len(result)} players → {PBP_STATS_PATH}")
    return result


if __name__ == "__main__":
    df = compute_pbp_player_stats()
    print(f"\nColumns: {list(df.columns)}")
    best = df[df["pbp_n_matches"] >= 10].sort_values("pbp_tb_pt_win_rate", ascending=False).head(20)
    print("\nBest tiebreak point win rate (min 10 matches):")
    print(best[["player_name", "pbp_tb_pt_win_rate", "pbp_sv_win_rate",
                "pbp_ace_rate", "pbp_n_matches"]].to_string(index=False))
