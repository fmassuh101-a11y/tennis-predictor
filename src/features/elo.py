"""
Surface-specific ELO ratings computed from match history.
Processes matches in strict chronological order — no data leakage.

Surfaces tracked: Hard, Clay, Grass, Carpet (+ overall)
"""

import pandas as pd
import numpy as np
from collections import defaultdict


INITIAL_ELO = 1500
K_FACTOR = 32        # standard K for tennis
K_NEW_PLAYER = 40    # higher K for first 30 matches (learns faster)
NEW_PLAYER_THRESHOLD = 30

SURFACES = ["Hard", "Clay", "Grass", "Carpet", "overall"]


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that player A beats player B given their ELOs."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def k_factor(n_matches: int) -> float:
    return K_NEW_PLAYER if n_matches < NEW_PLAYER_THRESHOLD else K_FACTOR


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a sorted (by date) match DataFrame, compute ELO ratings
    for both players at the time of each match (BEFORE the match is played).

    Returns df with added columns:
        winner_elo_{surface}, loser_elo_{surface}  (pre-match ELO)
        winner_elo_overall, loser_elo_overall
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # Track current ELO per player per surface
    elo: dict[str, dict[str, float]] = defaultdict(lambda: {s: INITIAL_ELO for s in SURFACES})
    match_count: dict[str, int] = defaultdict(int)

    # Output columns
    for s in SURFACES:
        df[f"winner_elo_{s.lower()}"] = np.nan
        df[f"loser_elo_{s.lower()}"] = np.nan

    for idx, row in df.iterrows():
        # Always use normalized name — IDs differ across sources and would break the ELO chain
        w = str(row["winner_name"]).strip().lower()
        l = str(row["loser_name"]).strip().lower()
        surface = str(row.get("surface", "")).strip().capitalize()
        if surface not in SURFACES:
            surface = "overall"

        # Record pre-match ELOs
        for s in SURFACES:
            df.at[idx, f"winner_elo_{s.lower()}"] = elo[w][s]
            df.at[idx, f"loser_elo_{s.lower()}"] = elo[l][s]

        # Update ELO after the match
        for s in [surface, "overall"]:
            w_elo = elo[w][s]
            l_elo = elo[l][s]
            exp_w = expected_score(w_elo, l_elo)
            exp_l = 1.0 - exp_w

            k_w = k_factor(match_count[w])
            k_l = k_factor(match_count[l])

            elo[w][s] = w_elo + k_w * (1.0 - exp_w)
            elo[l][s] = l_elo + k_l * (0.0 - exp_l)

        match_count[w] += 1
        match_count[l] += 1

    return df


def build_elo_table(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Run through all matches and return the CURRENT ELO ratings for all players.
    Used to get live ratings for predicting today's matches.
    """
    df = df.sort_values("tourney_date")
    elo: dict[str, dict[str, float]] = defaultdict(lambda: {s: INITIAL_ELO for s in SURFACES})
    match_count: dict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        w = str(row["winner_id"]) if pd.notna(row.get("winner_id")) else row["winner_name"]
        l = str(row["loser_id"]) if pd.notna(row.get("loser_id")) else row["loser_name"]
        surface = str(row.get("surface", "")).strip().capitalize()
        if surface not in SURFACES:
            surface = "overall"

        for s in [surface, "overall"]:
            w_elo = elo[w][s]
            l_elo = elo[l][s]
            exp_w = expected_score(w_elo, l_elo)
            exp_l = 1.0 - exp_w
            k_w = k_factor(match_count[w])
            k_l = k_factor(match_count[l])
            elo[w][s] = w_elo + k_w * (1.0 - exp_w)
            elo[l][s] = l_elo + k_l * (0.0 - exp_l)

        match_count[w] += 1
        match_count[l] += 1

    return dict(elo)
