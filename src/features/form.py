"""
Form, fatigue, H2H, and ranking features.
All computed using only past matches — strict temporal ordering.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute (using only prior matches):
    - winner/loser recent form (win% last 20 overall, last 20 on surface)
    - winner/loser fatigue (days since last match, matches in last 30 days)
    - H2H record (overall and on surface)
    - H2H last 5 meetings on surface

    Returns df with new feature columns added.
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # History per player: list of (date, opponent_id, surface, won)
    history: dict[str, list] = defaultdict(list)

    # New columns
    feature_cols = [
        "winner_form_overall", "loser_form_overall",
        "winner_form_surface", "loser_form_surface",
        "winner_days_rest", "loser_days_rest",
        "winner_matches_30d", "loser_matches_30d",
        "h2h_winner_wins", "h2h_loser_wins",
        "h2h_surface_winner_wins", "h2h_surface_loser_wins",
        # NEW: win streak, form trend, recent surface form
        "winner_win_streak", "loser_win_streak",
        "winner_form_trend", "loser_form_trend",
        "winner_form_last5_surface", "loser_form_last5_surface",
        "winner_level_form", "loser_level_form",
        # NEW: strength of schedule (avg ELO of last 10 opponents)
        "winner_avg_opp_elo", "loser_avg_opp_elo",
    ]
    for col in feature_cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        w = str(row["winner_name"]).strip().lower()
        l = str(row["loser_name"]).strip().lower()
        surface = str(row.get("surface", "")).strip().capitalize()
        level = str(row.get("tourney_level", "")).strip()

        try:
            match_date = pd.to_datetime(str(row["tourney_date"]), format="%Y%m%d")
        except Exception:
            try:
                match_date = pd.to_datetime(row["tourney_date"])
            except Exception:
                history[w].append((None, l, surface, True, level))
                history[l].append((None, w, surface, False, level))
                continue

        # --- Form features ---
        for player, is_winner in [(w, True), (l, False)]:
            prefix = "winner" if is_winner else "loser"
            past = history[player]

            if past:
                # Last 20 overall
                last20 = past[-20:]
                wins20 = sum(1 for p in last20 if p[3])
                form_overall = wins20 / len(last20)
                df.at[idx, f"{prefix}_form_overall"] = form_overall

                # Last 20 on same surface
                last20_surf = [p for p in past if p[2] == surface][-20:]
                if last20_surf:
                    wins_surf = sum(1 for p in last20_surf if p[3])
                    df.at[idx, f"{prefix}_form_surface"] = wins_surf / len(last20_surf)

                # Last 5 on same surface (more recent signal)
                last5_surf = [p for p in past if p[2] == surface][-5:]
                if last5_surf:
                    wins5_surf = sum(1 for p in last5_surf if p[3])
                    df.at[idx, f"{prefix}_form_last5_surface"] = wins5_surf / len(last5_surf)

                # Form trend: last 5 vs last 20 (positive = hot, negative = cold)
                last5 = past[-5:]
                if len(last5) >= 3 and len(last20) >= 10:
                    form5 = sum(1 for p in last5 if p[3]) / len(last5)
                    form_trend = form5 - form_overall  # +ve = improving, -ve = declining
                    df.at[idx, f"{prefix}_form_trend"] = form_trend

                # Win streak: consecutive wins coming in (capped at 20)
                streak = 0
                for p in reversed(past):
                    if p[3]:
                        streak += 1
                    else:
                        break
                df.at[idx, f"{prefix}_win_streak"] = min(streak, 20)

                # Level-specific form (how do they do at this tournament level: G/M/A/C)
                level_hist = [p for p in past if len(p) > 4 and p[4] == level][-15:]
                if len(level_hist) >= 3:
                    level_wins = sum(1 for p in level_hist if p[3])
                    df.at[idx, f"{prefix}_level_form"] = level_wins / len(level_hist)

                # Avg opponent ELO (strength of schedule, last 10 matches)
                opp_elos = [p[5] for p in past[-10:] if len(p) > 5 and p[5] is not None]
                if opp_elos:
                    df.at[idx, f"{prefix}_avg_opp_elo"] = float(np.mean(opp_elos))

                # Days since last match
                dated = [p for p in past if p[0] is not None]
                if dated:
                    last_date = dated[-1][0]
                    days_rest = (match_date - last_date).days
                    df.at[idx, f"{prefix}_days_rest"] = days_rest

                    # Matches in last 30 days
                    cutoff = match_date - timedelta(days=30)
                    recent = sum(1 for p in dated if p[0] >= cutoff)
                    df.at[idx, f"{prefix}_matches_30d"] = recent

        # --- H2H features ---
        all_meetings = [p for p in history[w] if p[1] == l]
        df.at[idx, "h2h_winner_wins"] = sum(1 for p in all_meetings if p[3])
        df.at[idx, "h2h_loser_wins"] = sum(1 for p in all_meetings if not p[3])

        surf_meetings = [p for p in all_meetings if p[2] == surface]
        df.at[idx, "h2h_surface_winner_wins"] = sum(1 for p in surf_meetings if p[3])
        df.at[idx, "h2h_surface_loser_wins"] = sum(1 for p in surf_meetings if not p[3])

        # Update history AFTER recording features (no leakage)
        # Store opponent's pre-match ELO as 6th element for avg_opp_elo feature
        w_elo = row.get("winner_elo_overall")
        l_elo = row.get("loser_elo_overall")
        w_elo = float(w_elo) if w_elo is not None and not (isinstance(w_elo, float) and np.isnan(w_elo)) else None
        l_elo = float(l_elo) if l_elo is not None and not (isinstance(l_elo, float) and np.isnan(l_elo)) else None
        history[w].append((match_date, l, surface, True, level, l_elo))
        history[l].append((match_date, w, surface, False, level, w_elo))

    return df


def compute_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking difference and ratio features."""
    df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
    df["rank_ratio"] = df["winner_rank"] / df["loser_rank"].replace(0, np.nan)
    df["rank_points_diff"] = df["winner_rank_points"] - df["loser_rank_points"]
    return df
