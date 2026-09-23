"""
Main feature pipeline.
Loads matches from DB, computes all features, outputs training-ready matrix.

CRITICAL DESIGN DECISION:
  In Sackmann's data, the winner is ALWAYS player1. If we train on this directly,
  the model learns "player1 always wins" — useless. So we randomize which player
  is listed as p1/p2 and set the target accordingly.

Output format:
  One row per match. Features are expressed as (p1 - p2) differences.
  Target: 1 if p1 won, 0 if p2 won.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.elo import compute_elo
from src.features.form import compute_rolling_features, compute_rank_features
from src.features.stats import compute_rolling_stats
from src.features.advanced import compute_advanced_rolling, add_static_features
from src.features.pbp_stats import compute_pbp_player_stats
from src.features.charting import add_clutch_features, add_charting_serve_features

PBP_STATS_PATH = os.path.join(os.path.dirname(__file__), "../../data/features/pbp_player_stats.csv")

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../../data/features/feature_matrix.parquet")
ODDS_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/odds/odds_combined.csv")


def load_matches(min_year: int = 1990, sources: list[str] = None) -> pd.DataFrame:
    """Load matches from SQLite. Filter by year and source. Deduplicates across sources."""
    conn = sqlite3.connect(DB_PATH)

    if sources is None:
        # All ATP sources — ordered by data quality (best first for dedup)
        sources = [
            "atp_futures",        # ITF/Futures (1991-2024) — for ELO chain, filtered from training rows
            "atp_main",           # Sackmann gold standard (1967-2024)
            "atp_scraper_2025",   # NoiseQuotient scraper: 3,233 matches + serve stats (Dec2024-Sep2025)
            "tml_2025",           # TML-Database 2025 — rich serve stats
            "tml_2026",           # TML-Database 2026
            "tennis_data_recent", # tennis-data.co.uk 2024-2026
            "atp_challenger",     # api-tennis.com challengers (all years)
            "tennis_api_atp",     # api-tennis.com ATP main 2025-2026
            "sportradar",         # Sportradar 2025-2026: ATP + Challenger (new matches only)
        ]

    placeholders = ",".join("?" * len(sources))
    query = f"""
        SELECT * FROM matches
        WHERE CAST(SUBSTR(tourney_date, 1, 4) AS INTEGER) >= ?
          AND source IN ({placeholders})
          AND winner_name IS NOT NULL
          AND loser_name IS NOT NULL
        ORDER BY tourney_date ASC
    """
    df = pd.read_sql_query(query, conn, params=[min_year] + sources)
    conn.close()
    print(f"Loaded {len(df):,} matches from database (year >= {min_year}, before dedup)")

    # Deduplicate: same winner+loser+date = same match from multiple sources.
    # Score each row by non-null serve stat count to keep the richest record.
    serve_stat_cols = ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
                       "w_bpSaved", "w_bpFaced", "l_ace", "l_df"]
    existing_stat_cols = [c for c in serve_stat_cols if c in df.columns]
    if existing_stat_cols:
        df["_stat_score"] = df[existing_stat_cols].notna().sum(axis=1)
    else:
        df["_stat_score"] = 0

    # Source priority (lower index = preferred when stats are equal)
    source_priority = {s: i for i, s in enumerate(sources)}
    df["_src_priority"] = df["source"].map(source_priority).fillna(99)

    # Sort: best stat coverage first, then preferred source, then keep first
    df = df.sort_values(["_stat_score", "_src_priority"], ascending=[False, True])
    dedup_key = (
        df["winner_name"].str.lower().str.strip() + "|" +
        df["loser_name"].str.lower().str.strip() + "|" +
        df["tourney_date"].astype(str)
    )
    before = len(df)
    df = df[~dedup_key.duplicated(keep="first")].copy()
    df = df.drop(columns=["_stat_score", "_src_priority"])
    df = df.sort_values("tourney_date").reset_index(drop=True)
    print(f"After dedup: {len(df):,} matches (removed {before - len(df):,} duplicates)")
    return df


def load_odds() -> pd.DataFrame:
    """Load odds data for joining onto matches."""
    df = pd.read_csv(ODDS_PATH, low_memory=False)
    # Normalize names for joining
    df["winner_name_norm"] = df["winner_name"].str.strip().str.lower()
    df["loser_name_norm"] = df["loser_name"].str.strip().str.lower()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def normalize_name(name: str) -> str:
    """
    Normalize player names to 'lastname initial' format for joining.
    Handles both:
      - "Mayer F."        → "mayer f"   (odds format: lastname firstname_initial)
      - "Tennys Sandgren" → "sandgren t" (sackmann format: firstname lastname)
    """
    if pd.isna(name):
        return ""
    name = str(name).strip().rstrip(".")
    parts = name.split()
    if not parts:
        return ""
    # Detect odds format: "Lastname F" (last part is single letter or initial)
    if len(parts) >= 2 and len(parts[-1].rstrip(".")) == 1:
        # Already "Lastname Initial" — just lowercase
        return f"{parts[0].lower()} {parts[-1].lower().rstrip('.')}"
    # Sackmann format: "Firstname Lastname" or "First Middle Last"
    # Use last word as surname, first letter of first word as initial
    last = parts[-1].lower()
    initial = parts[0][0].lower()
    return f"{last} {initial}"


def attach_odds(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Join odds data onto matches using normalized name + date.
    Only attaches where we have a confident match — no guessing.
    """
    matches["match_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    matches["winner_name_norm"] = matches["winner_name"].apply(normalize_name)
    matches["loser_name_norm"] = matches["loser_name"].apply(normalize_name)

    odds["match_date"] = pd.to_datetime(odds["match_date"], errors="coerce")
    odds["year"] = odds["match_date"].dt.year
    odds["month"] = odds["match_date"].dt.month
    odds["winner_name_norm"] = odds["winner_name"].apply(normalize_name)
    odds["loser_name_norm"] = odds["loser_name"].apply(normalize_name)

    # Sackmann tourney_date is tournament START date — extract year+month for joining
    matches["year"] = matches["match_date"].dt.year
    matches["month"] = matches["match_date"].dt.month

    odds_cols = ["winner_name_norm", "loser_name_norm", "year", "month",
                 "odds_b365_winner", "odds_b365_loser",
                 "odds_ps_winner", "odds_ps_loser",
                 "odds_max_winner", "odds_max_loser",
                 "odds_avg_winner", "odds_avg_loser"]
    odds_sub = odds[[c for c in odds_cols if c in odds.columns]].drop_duplicates(
        subset=["winner_name_norm", "loser_name_norm", "year", "month"]
    ).copy()

    merged = matches.merge(odds_sub, on=["winner_name_norm", "loser_name_norm", "year", "month"],
                           how="left")
    n_with_odds = merged["odds_b365_winner"].notna().sum()
    print(f"Odds attached: {n_with_odds:,} / {len(merged):,} matches have odds data")
    return merged


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature computations in order."""
    print("Adding static features (hand, age, height, seeding)...")
    df = add_static_features(df)

    print("Computing ELO ratings...")
    df = compute_elo(df)

    print("Computing form, H2H, fatigue features...")
    df = compute_rolling_features(df)

    print("Computing ranking features...")
    df = compute_rank_features(df)

    print("Computing rolling serve/return stats...")
    df = compute_rolling_stats(df)

    print("Computing advanced features (tiebreak, dominant wins, hold rate)...")
    df = compute_advanced_rolling(df)

    print("Adding MatchCharting serve/return stats...")
    df = add_charting_serve_features(df, gender="m")

    print("Adding MatchCharting clutch stats (BP save/convert under pressure)...")
    df = add_clutch_features(df, gender="m")

    # Attach PBP tiebreak win rates
    if os.path.exists(PBP_STATS_PATH):
        print("Attaching PBP stats (ace, serve win, tiebreak, hold rate)...")
        pbp_cols = ["player_name", "pbp_tb_win_rate", "pbp_ace_rate",
                    "pbp_sv_win_rate", "pbp_rv_win_rate", "pbp_sv_game_hold_rate",
                    "pbp_tb_pt_win_rate", "pbp_n_matches"]
        pbp = pd.read_csv(PBP_STATS_PATH)
        pbp = pbp[[c for c in pbp_cols if c in pbp.columns]]
        pbp = pbp[pbp["pbp_n_matches"] >= 5]   # only reliable stats
        df = df.merge(pbp.rename(columns={c: f"winner_{c}" if c != "player_name" else "winner_name"
                                          for c in pbp.columns}),
                      on="winner_name", how="left")
        df = df.merge(pbp.rename(columns={c: f"loser_{c}" if c != "player_name" else "loser_name"
                                          for c in pbp.columns}),
                      on="loser_name", how="left")
        n_w = df["winner_pbp_tb_win_rate"].notna().sum()
        print(f"  PBP stats attached: {n_w:,} / {len(df):,} matches")

    return df


def build_training_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the enriched match DataFrame into a ML-ready feature matrix.

    For each match, randomly assign winner/loser as p1/p2.
    Express all features as p1 - p2 differences.
    Target: p1_won (1 or 0).

    NOTE: ITF/Futures matches (source=atp_futures, tourney_level=F) are used
    for ELO chain computation but excluded from the training matrix — the model
    should learn ATP/challenger-level patterns only.
    """
    # Filter: only include ATP-level matches in training (not raw ITF futures)
    train_df = df[df["source"] != "atp_futures"].copy()
    print(f"Training rows after excluding ITF futures: {len(train_df):,} "
          f"(excluded {len(df)-len(train_df):,} ITF rows used only for ELO)")
    df = train_df

    np.random.seed(42)
    n = len(df)
    # Randomly decide if winner is p1 (True) or p2 (False)
    p1_is_winner = np.random.choice([True, False], size=n)

    rows = []
    feature_pairs = [
        # (winner_col, loser_col, feature_name)
        ("winner_elo_overall", "loser_elo_overall", "elo_overall"),
        ("winner_elo_hard", "loser_elo_hard", "elo_hard"),
        ("winner_elo_clay", "loser_elo_clay", "elo_clay"),
        ("winner_elo_grass", "loser_elo_grass", "elo_grass"),
        ("winner_rank", "loser_rank", "rank"),
        ("winner_rank_points", "loser_rank_points", "rank_points"),
        ("winner_form_overall", "loser_form_overall", "form_overall"),
        ("winner_form_surface", "loser_form_surface", "form_surface"),
        ("winner_days_rest", "loser_days_rest", "days_rest"),
        ("winner_matches_30d", "loser_matches_30d", "matches_30d"),
        ("winner_serve_1st_in_pct", "loser_serve_1st_in_pct", "serve_1st_in"),
        ("winner_serve_1st_won_pct", "loser_serve_1st_won_pct", "serve_1st_won"),
        ("winner_serve_bp_save_pct", "loser_serve_bp_save_pct", "serve_bp_save"),
        ("winner_serve_ace_pct", "loser_serve_ace_pct", "serve_ace"),
        # Advanced features
        ("winner_tiebreak_rate", "loser_tiebreak_rate", "tiebreak_rate"),
        ("winner_dominant_rate", "loser_dominant_rate", "dominant_rate"),
        ("winner_straight_sets_rate", "loser_straight_sets_rate", "straight_sets_rate"),
        ("winner_bp_conv_return", "loser_bp_conv_return", "bp_conv_return"),
        ("winner_hold_rate", "loser_hold_rate", "hold_rate"),
        ("winner_peak_dist", "loser_peak_dist", "peak_dist"),
        ("winner_age", "loser_age", "age"),
        ("winner_ht", "loser_ht", "height"),
        ("winner_pbp_tb_win_rate", "loser_pbp_tb_win_rate", "pbp_tb_rate"),
        ("winner_pbp_ace_rate", "loser_pbp_ace_rate", "pbp_ace_rate"),
        ("winner_pbp_sv_win_rate", "loser_pbp_sv_win_rate", "pbp_sv_win_rate"),
        ("winner_pbp_rv_win_rate", "loser_pbp_rv_win_rate", "pbp_rv_win_rate"),
        ("winner_pbp_sv_game_hold_rate", "loser_pbp_sv_game_hold_rate", "pbp_hold_rate"),
        ("winner_pbp_tb_pt_win_rate", "loser_pbp_tb_pt_win_rate", "pbp_tb_pt_rate"),
        # NEW: momentum, indoor form, level form
        ("winner_win_streak", "loser_win_streak", "win_streak"),
        ("winner_form_trend", "loser_form_trend", "form_trend"),
        ("winner_form_last5_surface", "loser_form_last5_surface", "form_last5_surface"),
        ("winner_level_form", "loser_level_form", "level_form"),
        # MatchCharting clutch: break point performance under pressure
        ("winner_clutch_bp_save", "loser_clutch_bp_save", "clutch_bp_save"),
        ("winner_clutch_bp_conv", "loser_clutch_bp_conv", "clutch_bp_conv"),
        # Strength of schedule: avg ELO of recent opponents
        ("winner_avg_opp_elo", "loser_avg_opp_elo", "avg_opp_elo"),
        # Charting serve/return stats (Sackmann Match Charting Project)
        ("winner_charting_first_serve_pct", "loser_charting_first_serve_pct", "charting_first_serve_pct"),
        ("winner_charting_first_won_pct",   "loser_charting_first_won_pct",   "charting_first_won_pct"),
        ("winner_charting_second_won_pct",  "loser_charting_second_won_pct",  "charting_second_won_pct"),
        ("winner_charting_bp_save_pct",     "loser_charting_bp_save_pct",     "charting_bp_save_pct"),
        ("winner_charting_return_won_pct",  "loser_charting_return_won_pct",  "charting_return_won_pct"),
        ("winner_charting_aces_per_match",  "loser_charting_aces_per_match",  "charting_aces_per_match"),
        ("winner_charting_unforced_per_match", "loser_charting_unforced_per_match", "charting_unforced_per_match"),
    ]

    for i, row in enumerate(df.itertuples()):
        p1_wins = bool(p1_is_winner[i])
        feat = {"p1_won": int(p1_wins)}

        # Surface one-hot
        surface = str(getattr(row, "surface", "")).strip()
        for s in ["Hard", "Clay", "Grass", "Carpet"]:
            feat[f"surface_{s.lower()}"] = int(surface == s)

        # Tournament level
        level = str(getattr(row, "tourney_level", ""))
        for lvl, code in [("grand_slam", "G"), ("masters", "M"), ("atp", "A"), ("challenger", "C")]:
            feat[f"level_{lvl}"] = int(level == code)

        # Best of 3 vs 5
        best_of = getattr(row, "best_of", None)
        feat["best_of_5"] = int(best_of == 5) if pd.notna(best_of) else 0

        # Difference features: p1 - p2
        for w_col, l_col, fname in feature_pairs:
            w_val = getattr(row, w_col, np.nan)
            l_val = getattr(row, l_col, np.nan)
            if p1_wins:
                p1_val, p2_val = w_val, l_val
            else:
                p1_val, p2_val = l_val, w_val
            feat[f"diff_{fname}"] = (
                p1_val - p2_val
                if pd.notna(p1_val) and pd.notna(p2_val)
                else np.nan
            )

        # H2H — p1 advantage
        h2h_w = getattr(row, "h2h_winner_wins", 0) or 0
        h2h_l = getattr(row, "h2h_loser_wins", 0) or 0
        h2h_ws = getattr(row, "h2h_surface_winner_wins", 0) or 0
        h2h_ls = getattr(row, "h2h_surface_loser_wins", 0) or 0
        if p1_wins:
            feat["h2h_p1_advantage"] = h2h_w - h2h_l
            feat["h2h_surface_p1_advantage"] = h2h_ws - h2h_ls
        else:
            feat["h2h_p1_advantage"] = h2h_l - h2h_w
            feat["h2h_surface_p1_advantage"] = h2h_ls - h2h_ws

        # Raw odds (if available) — market implied probability
        if p1_wins:
            p1_odds = getattr(row, "odds_avg_winner", np.nan)
            p2_odds = getattr(row, "odds_avg_loser", np.nan)
        else:
            p1_odds = getattr(row, "odds_avg_loser", np.nan)
            p2_odds = getattr(row, "odds_avg_winner", np.nan)

        feat["p1_odds"] = p1_odds if pd.notna(p1_odds) else np.nan
        feat["p2_odds"] = p2_odds if pd.notna(p2_odds) else np.nan

        # Implied probability from odds (removes bookmaker margin)
        if pd.notna(p1_odds) and pd.notna(p2_odds) and p1_odds > 1 and p2_odds > 1:
            raw_p1 = 1.0 / p1_odds
            raw_p2 = 1.0 / p2_odds
            total = raw_p1 + raw_p2
            feat["market_implied_p1"] = raw_p1 / total  # normalized
        else:
            feat["market_implied_p1"] = np.nan

        # Metadata (not used as features)
        # Handedness (boolean flags — not a difference, categorical)
        w_hand = getattr(row, "winner_hand_code", 0)
        l_hand = getattr(row, "loser_hand_code", 0)
        if p1_wins:
            feat["p1_lefty"] = int(w_hand == 2)
            feat["p2_lefty"] = int(l_hand == 2)
        else:
            feat["p1_lefty"] = int(l_hand == 2)
            feat["p2_lefty"] = int(w_hand == 2)

        # Indoor/outdoor flag
        feat["is_indoor"] = int(getattr(row, "is_indoor", 0) or 0)

        # Round importance
        feat["round_importance"] = getattr(row, "round_importance", 3)

        # Seeded flag
        w_seed = getattr(row, "winner_is_seeded", 0)
        l_seed = getattr(row, "loser_is_seeded", 0)
        if p1_wins:
            feat["p1_seeded"] = w_seed
            feat["p2_seeded"] = l_seed
        else:
            feat["p1_seeded"] = l_seed
            feat["p2_seeded"] = w_seed

        feat["tourney_date"] = getattr(row, "tourney_date", None)
        feat["tourney_name"] = getattr(row, "tourney_name", None)
        feat["source"] = getattr(row, "source", None)
        feat["winner_name"] = getattr(row, "winner_name", None)
        feat["loser_name"] = getattr(row, "loser_name", None)

        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"\nFeature matrix: {len(result):,} rows x {len(result.columns)} columns")
    print(f"  Rows with odds data: {result['market_implied_p1'].notna().sum():,}")
    print(f"  Target balance: {result['p1_won'].mean():.3f} (should be ~0.5)")
    return result


def run_pipeline(min_year: int = 1991, include_odds: bool = True):
    """Full pipeline: load → features → matrix → save."""
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

    # Load
    matches = load_matches(min_year=min_year)

    # Attach odds
    if include_odds and os.path.exists(ODDS_PATH):
        odds = load_odds()
        matches = attach_odds(matches, odds)

    # Feature engineering
    matches = build_features(matches)

    # Build training matrix
    matrix = build_training_matrix(matches)

    # Save
    matrix.to_parquet(FEATURES_PATH, index=False)
    print(f"\nSaved feature matrix to {FEATURES_PATH}")
    return matrix


if __name__ == "__main__":
    matrix = run_pipeline(min_year=1991)  # includes all ATP sources + dedup
    print("\nFeature columns:")
    for col in sorted(matrix.columns):
        non_null = matrix[col].notna().sum()
        print(f"  {col}: {non_null:,} non-null ({100*non_null/len(matrix):.1f}%)")
