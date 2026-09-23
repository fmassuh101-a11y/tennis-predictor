"""
WTA feature pipeline — mirrors the ATP pipeline exactly but for women's tennis.
Sources: wta_main (Sackmann 2000-2024) + tennis_data_wta_recent (2025/2026 if available)

Output: data/features/feature_matrix_wta.parquet
"""

import os
import sqlite3
import numpy as np
import pandas as pd

from src.features.elo import compute_elo
from src.features.form import compute_rolling_features, compute_rank_features
from src.features.stats import compute_rolling_stats
from src.features.advanced import compute_advanced_rolling, add_static_features
from src.features.charting import add_clutch_features, add_charting_serve_features

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../../data/features/feature_matrix_wta.parquet")


def load_wta_matches(min_year: int = 2000) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    sources = ["wta_qual_itf", "wta_main"]  # ITF first for ELO chain

    # Include recent WTA data if loaded
    cursor = conn.execute("SELECT COUNT(*) FROM matches WHERE source='tennis_data_wta_recent'")
    if cursor.fetchone()[0] > 0:
        sources.append("tennis_data_wta_recent")

    # Include Sportradar WTA matches (2025-2026, women's ATP + WTA 125K)
    cursor = conn.execute("SELECT COUNT(*) FROM matches WHERE source='sportradar' AND tourney_level IN ('A','M','G','C')")
    if cursor.fetchone()[0] > 0:
        sources.append("sportradar")

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
    print(f"Loaded {len(df):,} WTA matches (year >= {min_year}, before dedup)")

    # Deduplicate across sources
    serve_stat_cols = ["w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
                       "w_bpSaved", "w_bpFaced", "l_ace", "l_df"]
    existing_stat_cols = [c for c in serve_stat_cols if c in df.columns]
    df["_stat_score"] = df[existing_stat_cols].notna().sum(axis=1) if existing_stat_cols else 0
    source_priority = {s: i for i, s in enumerate(sources)}
    df["_src_priority"] = df["source"].map(source_priority).fillna(99)
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
    print(f"After dedup: {len(df):,} WTA matches (removed {before - len(df):,} duplicates)")
    return df


def build_wta_training_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Same logic as ATP build_training_matrix.
    ITF/qual matches (wta_qual_itf) used for ELO only, excluded from training rows.
    """
    # Filter: exclude ITF/qual matches from training matrix
    train_df = df[df["source"] != "wta_qual_itf"].copy()
    print(f"WTA training rows after excluding ITF/qual: {len(train_df):,} "
          f"(excluded {len(df)-len(train_df):,} ITF rows used only for ELO)")
    df = train_df

    np.random.seed(42)
    n = len(df)
    p1_is_winner = np.random.choice([True, False], size=n)

    rows = []
    feature_pairs = [
        ("winner_elo_overall", "loser_elo_overall", "elo_overall"),
        ("winner_elo_hard",    "loser_elo_hard",    "elo_hard"),
        ("winner_elo_clay",    "loser_elo_clay",    "elo_clay"),
        ("winner_elo_grass",   "loser_elo_grass",   "elo_grass"),
        ("winner_rank",        "loser_rank",        "rank"),
        ("winner_rank_points", "loser_rank_points", "rank_points"),
        ("winner_form_overall","loser_form_overall","form_overall"),
        ("winner_form_surface","loser_form_surface","form_surface"),
        ("winner_days_rest",   "loser_days_rest",   "days_rest"),
        ("winner_matches_30d", "loser_matches_30d", "matches_30d"),
        ("winner_serve_1st_in_pct", "loser_serve_1st_in_pct", "serve_1st_in"),
        ("winner_serve_1st_won_pct","loser_serve_1st_won_pct","serve_1st_won"),
        ("winner_serve_bp_save_pct","loser_serve_bp_save_pct","serve_bp_save"),
        ("winner_serve_ace_pct",    "loser_serve_ace_pct",    "serve_ace"),
        ("winner_tiebreak_rate",   "loser_tiebreak_rate",    "tiebreak_rate"),
        ("winner_dominant_rate",   "loser_dominant_rate",    "dominant_rate"),
        ("winner_straight_sets_rate","loser_straight_sets_rate","straight_sets_rate"),
        ("winner_bp_conv_return",  "loser_bp_conv_return",   "bp_conv_return"),
        ("winner_hold_rate",       "loser_hold_rate",        "hold_rate"),
        ("winner_peak_dist",       "loser_peak_dist",        "peak_dist"),
        ("winner_age",             "loser_age",              "age"),
        ("winner_ht",              "loser_ht",               "height"),
        # NEW: momentum, indoor form, level form
        ("winner_win_streak",         "loser_win_streak",         "win_streak"),
        ("winner_form_trend",         "loser_form_trend",         "form_trend"),
        ("winner_form_last5_surface", "loser_form_last5_surface", "form_last5_surface"),
        ("winner_level_form",         "loser_level_form",         "level_form"),
        # MatchCharting clutch: break point performance under pressure
        ("winner_clutch_bp_save", "loser_clutch_bp_save", "clutch_bp_save"),
        ("winner_clutch_bp_conv", "loser_clutch_bp_conv", "clutch_bp_conv"),
        # Charting serve/return stats
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

        surface = str(getattr(row, "surface", "")).strip()
        for s in ["Hard", "Clay", "Grass", "Carpet"]:
            feat[f"surface_{s.lower()}"] = int(surface == s)

        level = str(getattr(row, "tourney_level", ""))
        for lvl, code in [("grand_slam","G"),("masters","M"),("wta","A"),("challenger","C")]:
            feat[f"level_{lvl}"] = int(level == code)

        best_of = getattr(row, "best_of", None)
        feat["best_of_5"] = int(best_of == 5) if pd.notna(best_of) else 0

        for w_col, l_col, fname in feature_pairs:
            w_val = getattr(row, w_col, np.nan)
            l_val = getattr(row, l_col, np.nan)
            p1_val, p2_val = (w_val, l_val) if p1_wins else (l_val, w_val)
            feat[f"diff_{fname}"] = (
                p1_val - p2_val
                if pd.notna(p1_val) and pd.notna(p2_val)
                else np.nan
            )

        h2h_w  = getattr(row, "h2h_winner_wins", 0) or 0
        h2h_l  = getattr(row, "h2h_loser_wins", 0) or 0
        h2h_ws = getattr(row, "h2h_surface_winner_wins", 0) or 0
        h2h_ls = getattr(row, "h2h_surface_loser_wins", 0) or 0
        if p1_wins:
            feat["h2h_p1_advantage"] = h2h_w - h2h_l
            feat["h2h_surface_p1_advantage"] = h2h_ws - h2h_ls
        else:
            feat["h2h_p1_advantage"] = h2h_l - h2h_w
            feat["h2h_surface_p1_advantage"] = h2h_ls - h2h_ws

        w_hand = getattr(row, "winner_hand_code", 0)
        l_hand = getattr(row, "loser_hand_code", 0)
        if p1_wins:
            feat["p1_lefty"] = int(w_hand == 2)
            feat["p2_lefty"] = int(l_hand == 2)
        else:
            feat["p1_lefty"] = int(l_hand == 2)
            feat["p2_lefty"] = int(w_hand == 2)

        feat["is_indoor"] = int(getattr(row, "is_indoor", 0) or 0)
        feat["round_importance"] = getattr(row, "round_importance", 3)
        w_seed = getattr(row, "winner_is_seeded", 0)
        l_seed = getattr(row, "loser_is_seeded", 0)
        if p1_wins:
            feat["p1_seeded"] = w_seed; feat["p2_seeded"] = l_seed
        else:
            feat["p1_seeded"] = l_seed; feat["p2_seeded"] = w_seed

        feat["tourney_date"] = getattr(row, "tourney_date", None)
        feat["tourney_name"] = getattr(row, "tourney_name", None)
        feat["source"]       = getattr(row, "source", None)
        feat["winner_name"]  = getattr(row, "winner_name", None)
        feat["loser_name"]   = getattr(row, "loser_name", None)
        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"\nWTA Feature matrix: {len(result):,} rows x {len(result.columns)} columns")
    print(f"  Target balance: {result['p1_won'].mean():.3f} (should be ~0.5)")
    return result


def run_wta_pipeline(min_year: int = 2003):
    """Full WTA pipeline: load → features → matrix → save."""
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

    matches = load_wta_matches(min_year=min_year)

    print("Adding static features...")
    matches = add_static_features(matches)
    print("Computing ELO...")
    matches = compute_elo(matches)
    print("Computing form, H2H, fatigue...")
    matches = compute_rolling_features(matches)
    print("Computing ranking features...")
    matches = compute_rank_features(matches)
    print("Computing rolling serve/return stats...")
    matches = compute_rolling_stats(matches)
    print("Computing advanced features...")
    matches = compute_advanced_rolling(matches)
    print("Adding MatchCharting clutch stats...")
    matches = add_charting_serve_features(matches, gender="w")
    matches = add_clutch_features(matches, gender="w")

    matrix = build_wta_training_matrix(matches)
    matrix.to_parquet(FEATURES_PATH, index=False)
    print(f"\nSaved WTA feature matrix to {FEATURES_PATH}")
    return matrix


if __name__ == "__main__":
    run_wta_pipeline(min_year=2003)
