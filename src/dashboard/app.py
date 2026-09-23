"""
Tennis Predictor Dashboard
Run with: streamlit run src/dashboard/app.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.predict import predict_match, load_state, find_player

# WTA model (loaded only if available)
WTA_MODEL_AVAILABLE = os.path.exists(
    os.path.join(os.path.dirname(__file__), "../../models/xgb_wta_calibrated.pkl")
)
if WTA_MODEL_AVAILABLE:
    from src.models.predict_wta import predict_wta_match, load_wta_state

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tennis Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .value-bet { background: #1a3a1a; border-left: 4px solid #00cc44; padding: 12px; border-radius: 4px; margin: 8px 0; }
    .no-bet    { background: #1a1a2e; border-left: 4px solid #444; padding: 12px; border-radius: 4px; margin: 8px 0; }
    .metric-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }
    .player-stat { font-size: 0.85em; color: #8b949e; }
    h1 { color: #e6edf3; }
    .stMetric label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def get_state():
    return load_state()

@st.cache_data
def get_player_names():
    state = get_state()
    summaries = state["summaries"]
    from datetime import datetime
    players = []
    for name, s in summaries.items():
        rank = s.get("rank") or 9999
        last = s.get("last_match_date")
        # Days since last match — recently active players sort first
        if last:
            try:
                days_ago = (datetime.now() - last).days
            except Exception:
                days_ago = 9999
        else:
            days_ago = 9999
        players.append((name, days_ago, rank))
    # Sort: active recently first, then by rank
    players.sort(key=lambda x: (x[1], x[2]))
    return [p[0] for p in players[:5000]]

@st.cache_data
def load_reports():
    report_dir = os.path.join(os.path.dirname(__file__), "../../data/reports")
    reports = []
    if os.path.exists(report_dir):
        for f in sorted(os.listdir(report_dir)):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(report_dir, f)) as fp:
                        data = json.load(fp)
                    reports.extend(data)
                except Exception:
                    pass
    return reports

@st.cache_data
def load_feature_matrix():
    path = os.path.join(os.path.dirname(__file__), "../../data/features/feature_matrix.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🎾 Tennis Predictor")
    st.caption(f"Built: {datetime.now().strftime('%Y-%m-%d')}")

    page = st.radio("Navigation", [
        "📅 Today's Picks",
        "🎯 Predict a Match (ATP)",
        "🎯 Predict a Match (WTA)",
        "📊 Model Performance",
        "👤 Player Profile",
            ])

    st.divider()
    st.caption("ATP Model: XGBoost + Calibrated")
    st.caption("ATP Data: 820k matches | 1991-2026")
    st.caption("ATP: 67.65% accuracy · AUC 0.742  |  WTA: 61.36% · AUC 0.677  |  832,242 matches · 59 features")
    st.caption("✓ ATP challengers + futures: full ELO chain")
    st.caption("✓ ATP main tour: data through Mar 2026")
    if WTA_MODEL_AVAILABLE:
        st.caption("WTA Model: 530k matches | 2003-2026")
        st.caption("WTA Test accuracy: 69.7%")
        st.caption("✓ WTA qual/ITF: full ELO chain through Mar 2026")


# ─────────────────────────────────────────────
# Page: Predict a Match
# ─────────────────────────────────────────────
if page == "🎯 Predict a Match (ATP)":
    st.title("Match Predictor")
    st.caption("Enter match details to get a calibrated win probability.")

    all_players = get_player_names()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Player 1")
        p1 = st.selectbox("Player 1 (type to search)", all_players, index=0, key="p1")

    with col2:
        st.subheader("Player 2")
        p2 = st.selectbox("Player 2 (type to search)", all_players, index=1, key="p2")

    use_odds = st.checkbox("Show advanced options", value=False, key="use_odds")
    if use_odds:
        oc1, oc2 = st.columns(2)
        with oc1:
            p1_odds = st.number_input("Player 1 Odds", min_value=1.01, max_value=50.0,
                                       value=1.80, step=0.05, key="p1o")
        with oc2:
            p2_odds = st.number_input("Player 2 Odds", min_value=1.01, max_value=50.0,
                                       value=2.00, step=0.05, key="p2o")
    else:
        p1_odds = None
        p2_odds = None
        st.info("💡 The model returns a calibrated probability, not just a winner.")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])
    with col4:
        level = st.selectbox("Level", ["A", "C", "M", "G"],
                             format_func=lambda x: {"A":"ATP250/500","C":"Challenger","M":"Masters","G":"Grand Slam"}[x])
    with col5:
        best_of = st.selectbox("Best of", [3, 5])
    with col6:
        round_name = st.selectbox("Round", ["R128","R64","R32","R16","QF","SF","F","RR"])

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        min_edge = st.slider("Min edge to flag bet (%)", 3, 15, 5) / 100
    with col_filter2:
        odds_range = st.slider("Odds range for bets", 1.2, 6.0, (1.5, 3.5))

    if st.button("🎯 Predict", type="primary", use_container_width=True):
        if p1 == p2:
            st.error("Please select two different players.")
        else:
            with st.spinner("Computing prediction..."):
                result = predict_match(
                    p1, p2, surface,
                    p1_odds=p1_odds, p2_odds=p2_odds,
                    level=level, best_of=best_of, round_name=round_name,
                    min_odds=odds_range[0], max_odds=odds_range[1],
                    min_edge=min_edge, verbose=False,
                )

            if "error" in result:
                st.error(result["error"])
            else:
                # Stale data warning
                p1_rest = result.get("p1_rest_days") or 0
                p2_rest = result.get("p2_rest_days") or 0
                stale_players = []
                if p1_rest > 200:
                    stale_players.append(f"{result['p1_name']} ({p1_rest}d ago)")
                if p2_rest > 200:
                    stale_players.append(f"{result['p2_name']} ({p2_rest}d ago)")
                if stale_players:
                    st.warning(f"⚠ Stale data: {', '.join(stale_players)} — last match in DB from 2024 (no 2025 challenger data available). Prediction is based on historical stats only.")

                # Winner banner
                winner = result["predicted_winner"]
                conf = result["confidence"] * 100
                color = "#00cc44" if result["has_value_bet"] else "#4a9eff"

                st.markdown(f"""
                <div style="background: {color}22; border: 2px solid {color};
                     border-radius: 8px; padding: 20px; text-align: center; margin: 16px 0;">
                    <h2 style="margin:0; color: {color}">🏆 {winner}</h2>
                    <p style="margin:4px 0; color: #e6edf3; font-size: 1.2em">{conf:.1f}% confidence</p>
                </div>
                """, unsafe_allow_html=True)

                # Player comparison
                c1, c2 = st.columns(2)
                for col, pname, prob, elo, rank, form, rest, this_odds in [
                    (c1, result["p1_name"], result["prob_p1"],
                     result["p1_elo"], result["p1_rank"], result["p1_form"],
                     result["p1_rest_days"], p1_odds),
                    (c2, result["p2_name"], result["prob_p2"],
                     result["p2_elo"], result["p2_rank"], result["p2_form"],
                     result["p2_rest_days"], p2_odds),
                ]:
                    with col:
                        odds_label = f"Odds: {this_odds}" if this_odds else None
                        st.metric(pname, f"{prob*100:.1f}%", odds_label)
                        st.progress(prob)
                        st.caption(f"ELO: {elo:.0f} | Rank: {rank} | "
                                   f"Form: {form*100:.0f}% | Rest: {rest}d")

                # Value bet
                st.divider()
                if result["value_bets"]:
                    for b in result["value_bets"]:
                        st.markdown(f"""
                        <div class="value-bet">
                            <strong>✦ Model favours {b['bet_on']}</strong><br>
                            Model: <strong>{b['model_prob']*100:.1f}%</strong> &nbsp;|&nbsp;
                            Market: <strong>{b['market_prob']*100:.1f}%</strong> &nbsp;|&nbsp;
                            Edge: <strong style="color:#00cc44">{b['edge_pct']}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                elif not use_odds:
                    st.markdown('<div class="no-bet">📊 Prediction complete.</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-bet">Prediction complete.</div>',
                                unsafe_allow_html=True)

                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["prob_p1"] * 100,
                    title={"text": f"{result['p1_name']} win probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#4a9eff"},
                        "steps": [
                            {"range": [0, 40], "color": "#3d1a1a"},
                            {"range": [40, 60], "color": "#1a1a3d"},
                            {"range": [60, 100], "color": "#1a3d1a"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
                    }
                ))
                fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e6edf3")
                st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Page: Predict a Match (WTA)
# ─────────────────────────────────────────────
elif page == "🎯 Predict a Match (WTA)":
    st.title("WTA Match Predictor")
    st.caption("Women's tennis predictions. Same model architecture as ATP, trained on WTA data.")

    if not WTA_MODEL_AVAILABLE:
        st.warning("WTA model not yet trained. Run: python -m src.models.train_wta")
    else:
        @st.cache_data
        def get_wta_player_names():
            state = load_wta_state()
            summaries = state["summaries"]
            players = []
            for name, s in summaries.items():
                rank = s.get("rank") or 9999
                last = s.get("last_match_date")
                days_ago = (datetime.now() - last).days if last else 9999
                players.append((name, days_ago, rank))
            players.sort(key=lambda x: (x[1], x[2]))
            return [p[0] for p in players[:3000]]

        wta_players = get_wta_player_names()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Player 1")
            p1 = st.selectbox("Player 1", wta_players, index=0, key="wta_p1")
        with col2:
            st.subheader("Player 2")
            p2 = st.selectbox("Player 2", wta_players, index=1, key="wta_p2")

        use_wta_odds = st.checkbox("Show advanced options", value=False, key="wta_use_odds")
        if use_wta_odds:
            oc1, oc2 = st.columns(2)
            with oc1:
                p1_odds = st.number_input("Player 1 Odds", min_value=1.01, max_value=50.0,
                                           value=1.80, step=0.05, key="wta_p1o")
            with oc2:
                p2_odds = st.number_input("Player 2 Odds", min_value=1.01, max_value=50.0,
                                           value=2.00, step=0.05, key="wta_p2o")
        else:
            p1_odds = None
            p2_odds = None
            st.info("💡 The model returns a calibrated probability, not just a winner.")

        col3, col4, col5, col6 = st.columns(4)
        with col3:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="wta_surf")
        with col4:
            level = st.selectbox("Level", ["A", "M", "G"],
                                 format_func=lambda x: {"A":"WTA250/500","M":"WTA1000","G":"Grand Slam"}[x],
                                 key="wta_lvl")
        with col5:
            best_of = st.selectbox("Best of", [3], key="wta_bo")
        with col6:
            round_name = st.selectbox("Round", ["R64","R32","R16","QF","SF","F"], key="wta_rnd")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            min_edge = st.slider("Min edge (%)", 3, 15, 5, key="wta_edge") / 100
        with col_f2:
            odds_range = st.slider("Odds range", 1.2, 6.0, (1.5, 3.5), key="wta_odds")

        if st.button("🎯 Predict WTA", type="primary", use_container_width=True):
            if p1 == p2:
                st.error("Please select two different players.")
            else:
                with st.spinner("Computing WTA prediction..."):
                    result = predict_wta_match(
                        p1, p2, surface,
                        p1_odds=p1_odds, p2_odds=p2_odds,
                        level=level, best_of=best_of, round_name=round_name,
                        min_odds=odds_range[0], max_odds=odds_range[1],
                        min_edge=min_edge, verbose=False,
                    )

                if "error" in result:
                    st.error(result["error"])
                else:
                    p1_rest = result.get("p1_rest_days") or 0
                    p2_rest = result.get("p2_rest_days") or 0
                    stale = []
                    if p1_rest > 200:
                        stale.append(f"{result['p1_name']} ({p1_rest}d)")
                    if p2_rest > 200:
                        stale.append(f"{result['p2_name']} ({p2_rest}d)")
                    if stale:
                        st.warning(f"⚠ Stale data: {', '.join(stale)} — based on 2024 data only.")

                    winner = result["predicted_winner"]
                    conf   = result["confidence"] * 100
                    color  = "#00cc44" if result["has_value_bet"] else "#cc44aa"

                    st.markdown(f"""
                    <div style="background: {color}22; border: 2px solid {color};
                         border-radius: 8px; padding: 20px; text-align: center; margin: 16px 0;">
                        <h2 style="margin:0; color: {color}">🏆 {winner}</h2>
                        <p style="margin:4px 0; color: #e6edf3; font-size: 1.2em">{conf:.1f}% confidence</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    for col, pname, prob, elo, rank, form, rest, this_odds in [
                        (c1, result["p1_name"], result["prob_p1"],
                         result["p1_elo"], result["p1_rank"], result["p1_form"],
                         result["p1_rest_days"], p1_odds),
                        (c2, result["p2_name"], result["prob_p2"],
                         result["p2_elo"], result["p2_rank"], result["p2_form"],
                         result["p2_rest_days"], p2_odds),
                    ]:
                        with col:
                            odds_label = f"Odds: {this_odds}" if this_odds else None
                            st.metric(pname, f"{prob*100:.1f}%", odds_label)
                            st.progress(prob)
                            st.caption(f"ELO: {elo:.0f} | Rank: {rank} | "
                                       f"Form: {form*100:.0f}% | Rest: {rest}d")

                    st.divider()
                    if result["value_bets"]:
                        for b in result["value_bets"]:
                            st.markdown(f"""
                            <div class="value-bet">
                                <strong>✦ Model favours {b['bet_on']}</strong><br>
                                Model: <strong>{b['model_prob']*100:.1f}%</strong> &nbsp;|&nbsp;
                                Market: <strong>{b['market_prob']*100:.1f}%</strong> &nbsp;|&nbsp;
                                Edge: <strong style="color:#00cc44">{b['edge_pct']}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                    elif not use_wta_odds:
                        st.markdown('<div class="no-bet">📊 Prediction complete.</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="no-bet">Prediction complete.</div>',
                                    unsafe_allow_html=True)

                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["prob_p1"] * 100,
                        title={"text": f"{result['p1_name']} win probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#cc44aa"},
                            "steps": [
                                {"range": [0, 40], "color": "#3d1a1a"},
                                {"range": [40, 60], "color": "#1a1a3d"},
                                {"range": [60, 100], "color": "#1a3d1a"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
                        }
                    ))
                    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)",
                                      font_color="#e6edf3")
                    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# Page: Model Performance
# ─────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("Model Performance")

    model_path = os.path.join(os.path.dirname(__file__), "../../models/xgb_calibrated.pkl")
    if not os.path.exists(model_path):
        st.warning("Model not yet trained. Run: python -m src.models.train")
    else:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        metrics = model_data.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy (Test)", f"{metrics.get('accuracy', 0)*100:.1f}%")
        col2.metric("AUC", f"{metrics.get('auc', 0):.3f}")
        col3.metric("Log Loss", f"{metrics.get('log_loss', 0):.3f}")
        col4.metric("Brier Score", f"{metrics.get('brier', 0):.3f}")

        st.divider()
        st.subheader("Feature Importance")

        feat_cols = model_data.get("feature_cols", [])
        # Get importances from the underlying XGBoost estimator
        try:
            base_model = model_data["model"].calibrated_classifiers_[0].estimator
            imps = base_model.feature_importances_
        except Exception:
            try:
                imps = model_data["model"].estimators_[0].feature_importances_
            except Exception:
                imps = np.zeros(len(feat_cols))

        imp_df = pd.DataFrame({"feature": feat_cols, "importance": imps})
        imp_df = imp_df.sort_values("importance", ascending=True).tail(20)

        fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale="blues")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e6edf3", height=600, showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Historical data coverage
    st.divider()
    st.subheader("Data Coverage")
    matrix = load_feature_matrix()
    if matrix is not None:
        matrix["year"] = matrix["tourney_date"].astype(str).str[:4].astype(int)
        by_year = matrix.groupby("year").size().reset_index(name="matches")
        fig2 = px.bar(by_year, x="year", y="matches", title="Matches per Year",
                      color="matches", color_continuous_scale="blues")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                           font_color="#e6edf3", coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────
# Page: Player Profile
# ─────────────────────────────────────────────
elif page == "👤 Player Profile":
    st.title("Player Profile")

    all_players = get_player_names()
    player_name = st.selectbox("Select Player", all_players)

    state = get_state()
    _, profile = find_player(player_name, state["summaries"])

    if profile is None:
        st.error("Player not found.")
    else:
        col1, col2, col3 = st.columns(3)
        rank_val = profile.get("rank", "N/A")
        rank_date = profile.get("ranking_date", "")
        rank_label = f"ATP Rank ({rank_date})" if rank_date else "ATP Rank"
        col1.metric(rank_label, rank_val)
        col2.metric("Overall ELO", f"{profile['elo']['overall']:.0f}")
        col3.metric("Overall Form", f"{(profile.get('form_overall') or 0)*100:.0f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Days Rest", profile.get("days_rest", "N/A"))
        col5.metric("Age", f"{profile.get('age', 'N/A'):.1f}" if profile.get("age") else "N/A")
        col6.metric("Hand", "Left 🤚" if str(profile.get("hand","")).upper() == "L" else "Right ✋")

        st.divider()
        st.subheader("ELO by Surface")
        elo = profile.get("elo", {})
        elo_data = {k: v for k, v in elo.items() if k != "overall"}
        fig = go.Figure(go.Bar(
            x=list(elo_data.keys()), y=list(elo_data.values()),
            marker_color=["#4a9eff", "#cc7722", "#22cc55", "#8844cc"]
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(13,17,23,1)", font_color="#e6edf3")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Form by Surface")
        form_surf = profile.get("form_by_surface", {})
        cols = st.columns(4)
        for i, (surf, frm) in enumerate(form_surf.items()):
            if frm is not None:
                cols[i].metric(surf, f"{frm*100:.0f}%")


# ─────────────────────────────────────────────
# Page: Today's Picks (auto from API)
# ─────────────────────────────────────────────
elif page == "📅 Today's Picks":
    import time as _time
    from datetime import timedelta as _td, timezone as _tz

    # Chilean time (UTC-3 summer)
    _CHILE = _tz(_td(hours=-3))
    _now_chile = datetime.now(_CHILE)
    _now_utc   = datetime.now(_tz.utc)

    st.title("📅 Today's Picks")
    st.caption(f"🇨🇱 Chile time: **{_now_chile.strftime('%H:%M')}** &nbsp;|&nbsp; "
               f"Only showing matches starting within the next 5 hours")

    picks_path = os.path.join(os.path.dirname(__file__), "../../data/todays_picks.json")

    # ── Cooldown logic: 1 hour between refreshes ─────────────────────────
    COOLDOWN_MIN = 60
    can_refresh  = True
    age_min      = 9999
    if os.path.exists(picks_path):
        age_min = (_time.time() - os.path.getmtime(picks_path)) / 60
        can_refresh = age_min >= COOLDOWN_MIN

    col_refresh, col_info = st.columns([1, 3])
    with col_refresh:
        if can_refresh:
            if st.button("🔄 Refresh Matches", type="primary", use_container_width=True):
                with st.spinner("Fetching upcoming matches..."):
                    try:
                        from src.ingestion.todays_matches import run as fetch_today
                        fetch_today()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            mins_left = int(COOLDOWN_MIN - age_min)
            st.button(f"⏳ Next refresh in {mins_left}m", disabled=True, use_container_width=True)

    with col_info:
        if os.path.exists(picks_path):
            age_str = f"{int(age_min)}m ago" if age_min < 60 else f"{age_min/60:.1f}h ago"
            if not can_refresh:
                st.caption(f"Last updated {age_str}. Refresh available after {int(COOLDOWN_MIN - age_min)} more minutes.")
            elif age_min < 9999:
                st.warning(f"⚠ Picks are {age_str} old — ready to refresh!")
        else:
            st.caption("Click Refresh to fetch today's schedule.")

    if not os.path.exists(picks_path):
        st.info("Click **Refresh Matches** to fetch today's schedule.")
    else:
        with open(picks_path) as f:
            picks_data = json.load(f)

        predictions = picks_data.get("predictions", [])

        # Check if picks are from today
        picks_date = picks_data.get("date", "")
        today_str  = date.today().isoformat()
        if picks_date != today_str:
            st.warning(f"⚠ These picks are from {picks_date}, not today ({today_str}). Click **Refresh** to get today's matches.")

        # All predictions are already time-filtered (within 5h) by todays_matches.py
        # Sort: live first, then by event_time, then by confidence
        def _sort_key(p):
            live = 0 if p.get("is_live") else 1
            t = p.get("event_time", "99:99")
            return (live, t)

        upcoming_sorted = sorted(predictions, key=_sort_key)

        if not upcoming_sorted:
            st.warning("No upcoming matches found. Click Refresh to update.")
        else:
            st.caption(f"{len(upcoming_sorted)} upcoming matches today across all tournaments")

            # ── TOP 5 PICKS ──────────────────────────────────────────────
            st.markdown("## 🏆 TOP 5 PICKS TODAY")
            st.caption("Upcoming matches with the model prediction for each.")

            def est_odds(conf):
                """Estimate bookmaker odds from model probability (with ~10% bookmaker margin)."""
                if conf <= 0: return 99.0
                fair = 1.0 / conf
                return round(fair * 0.90, 2)

            def real_or_est_odds(p):
                """Use real p1 odds if available, otherwise estimate from model confidence."""
                winner = p.get("predicted_winner","")
                p1n = p.get("p1_name","")
                if winner == p1n:
                    real = p.get("p1_odds")
                else:
                    real = p.get("p2_odds")
                if real and float(real) > 1.0:
                    return float(real), True
                return est_odds(p.get("confidence", 0.5)), False

            # Score each match: prefer 55-72% confidence (maps to odds ~1.40-1.65)
            # Penalize very high confidence (85%+) because bookmakers pay 1.05-1.12 — not worth it
            # Also filter stale data and require fresh player data
            def pick_score(p):
                conf = p.get("confidence", 0)
                is_stale = not p.get("data_reliable", True) or (p.get("p1_rest_days") or 0) > 200 or (p.get("p2_rest_days") or 0) > 200
                if is_stale: return -1
                odds = est_odds(conf)
                if odds < 1.35: return -1   # bookmaker will pay less than 1.35 — skip
                # Sweet spot: odds 1.40-2.20, confidence 55-72%
                if 1.38 <= odds <= 2.20:
                    return conf + 0.5   # boost these
                return conf

            candidates = [p for p in upcoming_sorted
                          if not ((p.get("p1_rest_days") or 0) > 200 or (p.get("p2_rest_days") or 0) > 200)]
            top5 = sorted(candidates, key=pick_score, reverse=True)[:5]
            if not top5:
                top5 = upcoming_sorted[:5]

            for i, p in enumerate(top5, 1):
                winner  = p.get("predicted_winner","?")
                loser   = p.get("p2_name","?") if winner == p.get("p1_name") else p.get("p1_name","?")
                conf    = p.get("confidence",0)*100
                odds, is_real = real_or_est_odds(p)
                bookie  = p.get("bookmaker") or ""
                tour    = p.get("tournament","?")
                surf    = p.get("surface","?")
                time_s  = p.get("event_time","?")    # Chilean time HH:MM
                is_live = p.get("is_live", False)
                has_bet = p.get("has_value_bet", False)
                is_stale = not p.get("data_reliable", True) or (p.get("p1_rest_days") or 0) > 200 or (p.get("p2_rest_days") or 0) > 200

                odds_label = f"{bookie}: {odds:.2f}" if is_real else f"Est: ~{odds:.2f}"
                live_badge = ' <span style="background:#ff3333;color:#fff;padding:2px 7px;border-radius:4px;font-size:0.75em;font-weight:bold;">⚡ LIVE</span>' if is_live else ""
                time_label = f"{time_s} CL{' · LIVE' if is_live else ''} · {surf}"

                if has_bet:
                    bg, border, label = "#0d2b0d", "#00cc44", "✅ HIGH CONFIDENCE"
                elif odds >= 1.50 and conf >= 55:
                    bg, border, label = "#0d2b0d", "#00cc44", "✅ BET — GOOD ODDS"
                elif odds >= 1.35:
                    bg, border, label = "#0d1a2b", "#4a9eff", "🔵 CONSIDER — DECENT ODDS"
                else:
                    bg, border, label = "#2b1a00", "#ff9900", "⚠ LOW ODDS — PROBABLY NOT WORTH IT"

                if has_bet:
                    b = p["value_bets"][0]
                    bet_player = b["bet_on"]
                    bet_odds   = b["odds"]
                    edge_pct   = b["edge_pct"]
                    model_pct  = round(b["model_prob"] * 100)
                    mkt_pct    = round(b["market_prob"] * 100)
                    opponent   = loser if bet_player == winner else winner
                    extra_html = (
                        f'<div style="margin-top:6px;color:#8b949e;font-size:0.85em;">'
                        f'Model: <strong style="color:#ccc">{winner}</strong> wins ({conf:.0f}%)'
                        f' &nbsp;|&nbsp; Market gives {bet_player}: {mkt_pct}%'
                        f' &nbsp;→&nbsp; Model says {model_pct}%</div>'
                        f'<div style="margin-top:10px;">'
                        f'<span style="background:#00cc44;color:#000;padding:5px 14px;border-radius:5px;font-weight:bold;font-size:1.05em;">'
                        f'💰 BET: {bet_player} @ {bet_odds} &nbsp;|&nbsp; Edge: {edge_pct} vs {bookie}'
                        f'</span></div>'
                    )
                    main_html = (
                        f'<strong style="color:#00cc44;font-size:1.4em;">{bet_player}</strong>{live_badge}'
                        f'<span style="color:#8b949e"> vs </span>'
                        f'<strong>{opponent}</strong>'
                    )
                elif odds < 1.35:
                    extra_html = (
                        f'<div style="margin-top:6px;font-size:1.6em;font-weight:bold;color:{border}">{conf:.0f}%'
                        f'<span style="font-size:0.6em;color:#8b949e;font-weight:normal;"> confidence</span></div>'
                        f'<div style="margin-top:6px;color:#ff6666;font-size:0.9em;">⛔ Odds {odds_label} — payout too low, skip</div>'
                    )
                    main_html = f'<strong style="color:{border}">{winner}</strong>{live_badge}<span style="color:#8b949e"> beats </span><strong>{loser}</strong>'
                else:
                    extra_html = (
                        f'<div style="margin-top:6px;font-size:1.6em;font-weight:bold;color:{border}">{conf:.0f}%'
                        f'<span style="font-size:0.6em;color:#8b949e;font-weight:normal;"> confidence</span></div>'
                        f'<div style="margin-top:6px;color:#8b949e;font-size:0.9em;">Odds: <strong style="color:#fff">{odds_label}</strong></div>'
                    )
                    main_html = f'<strong style="color:{border}">{winner}</strong>{live_badge}<span style="color:#8b949e"> beats </span><strong>{loser}</strong>'

                stale_div = '<div style="margin-top:4px;color:#ff9900;font-size:0.8em;">⚠ Data may be from 2024 — lower confidence</div>' if is_stale else ""

                st.markdown(
                    f'<div style="background:{bg};border:2px solid {border};border-radius:10px;padding:16px 20px;margin:8px 0;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span style="color:{border};font-weight:bold;font-size:1.1em;">#{i} &nbsp; {label}</span>'
                    f'<span style="color:#8b949e;font-size:0.9em;">{time_label}</span>'
                    f'</div>'
                    f'<div style="margin-top:10px;font-size:1.3em;">{main_html}</div>'
                    f'{extra_html}{stale_div}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── ALL MATCHES ──────────────────────────────────────────────
            st.divider()
            st.markdown("## All Matches Today")
            st.caption("Est. odds = what bookmakers likely pay (~10% margin). Only bet if your app shows similar or better odds.")

            col_h1, col_h2, col_h3, col_h4, col_h5, col_h6 = st.columns([2,2,1,1,1,2])
            col_h1.markdown("**Match**"); col_h2.markdown("**Pick**")
            col_h3.markdown("**Conf**"); col_h4.markdown("**Est. Odds**")
            col_h5.markdown("**Time**"); col_h6.markdown("**Verdict**")
            st.divider()

            for p in upcoming_sorted:
                winner = p.get("predicted_winner","?")
                loser  = p.get("p2_name","?") if winner == p.get("p1_name") else p.get("p1_name","?")
                p1n    = p.get("p1_name","?")
                p2n    = p.get("p2_name","?")
                conf   = p.get("confidence",0)*100
                odds, is_real = real_or_est_odds(p)
                bookie = p.get("bookmaker") or ""
                tour   = p.get("tournament","?")
                time_s   = p.get("event_time","?")       # Chilean HH:MM
                is_live  = p.get("is_live", False)
                is_stale = not p.get("data_reliable", True) or (p.get("p1_rest_days") or 0) > 200 or (p.get("p2_rest_days") or 0) > 200
                has_bet = p.get("has_value_bet", False)
                odds_display = f"{odds:.2f}" if is_real else f"~{odds:.2f}"
                time_display = f"⚡{time_s}" if is_live else time_s

                if is_stale:
                    conf_color = "#ff9900"
                    verdict = "⚠ STALE"
                elif has_bet:
                    conf_color = "#00cc44"
                    verdict = "✅ HIGH CONFIDENCE"
                elif odds < 1.35:
                    conf_color = "#555555"
                    verdict = "⛔ LOW ODDS"
                elif odds >= 1.50 and conf >= 55:
                    conf_color = "#00cc44"
                    verdict = "✅ BET"
                elif odds >= 1.38:
                    conf_color = "#4a9eff"
                    verdict = "🔵 CONSIDER"
                else:
                    conf_color = "#666666"
                    verdict = "⚪ SKIP"

                c1, c2, c3, c4, c5, c6 = st.columns([2,2,1,1,1,2])
                c1.write(f"{p1n} vs {p2n}")
                c2.markdown(f'<span style="color:{conf_color}"><strong>{winner}</strong></span>', unsafe_allow_html=True)
                c3.markdown(f'<span style="color:{conf_color}"><strong>{conf:.0f}%</strong></span>', unsafe_allow_html=True)
                c4.markdown(f'<span style="color:{conf_color}"><strong>{odds_display}</strong></span>', unsafe_allow_html=True)
                c5.write(time_display)
                c6.write(f"{verdict} · {tour}")

            # ── ODDS CHECKER ──────────────────────────────────────────────
            st.divider()
            st.markdown("## 💰 Odds Value Checker")
            st.caption("Enter the odds you see on your bookmaker (1xbet, Bettano, 1win...) — the bot tells you if there's value.")

            match_labels = [
                f"{p.get('p1_name','?')} vs {p.get('p2_name','?')}  [{p.get('tournament','?')}]"
                for p in upcoming_sorted
            ]
            selected_idx = st.selectbox("Select a match", range(len(match_labels)),
                                        format_func=lambda i: match_labels[i], key="odds_checker_match")
            selected = upcoming_sorted[selected_idx]
            p1n_sel = selected.get("p1_name", "Player 1")
            p2n_sel = selected.get("p2_name", "Player 2")
            model_p1 = selected.get("prob_p1", 0.5)
            model_p2 = selected.get("prob_p2", 0.5)

            st.markdown(f"**Model says:** {p1n_sel} {model_p1*100:.1f}% &nbsp;|&nbsp; {p2n_sel} {model_p2*100:.1f}%")

            oc1, oc2 = st.columns(2)
            with oc1:
                odds_p1 = st.number_input(f"Odds for **{p1n_sel}**", min_value=1.01, max_value=50.0,
                                          value=2.00, step=0.05, format="%.2f", key="odds_p1_input")
            with oc2:
                odds_p2 = st.number_input(f"Odds for **{p2n_sel}**", min_value=1.01, max_value=50.0,
                                          value=1.80, step=0.05, format="%.2f", key="odds_p2_input")

            if st.button("🔍 Check Value", type="primary", key="check_odds_btn"):
                implied_p1_raw = 1.0 / odds_p1
                implied_p2_raw = 1.0 / odds_p2
                total_implied  = implied_p1_raw + implied_p2_raw
                market_p1 = implied_p1_raw / total_implied
                market_p2 = implied_p2_raw / total_implied
                margin    = (total_implied - 1.0) * 100

                edge_p1 = model_p1 - market_p1
                edge_p2 = model_p2 - market_p2

                st.markdown(f"**Bookmaker margin:** {margin:.1f}% &nbsp;|&nbsp; "
                            f"Market implied: {p1n_sel} {market_p1*100:.1f}% / {p2n_sel} {market_p2*100:.1f}%")
                st.divider()

                for player, odds_val, model_prob, edge, market_prob in [
                    (p1n_sel, odds_p1, model_p1, edge_p1, market_p1),
                    (p2n_sel, odds_p2, model_p2, edge_p2, market_p2),
                ]:
                    if edge >= 0.05 and 1.5 <= odds_val <= 4.0:
                        st.markdown(f"""
                        <div style="background:#0d2b0d; border:2px solid #00cc44; border-radius:10px; padding:16px; margin:6px 0;">
                            <div style="font-size:1.2em; font-weight:bold; color:#00cc44">✅ BET ON {player.upper()} @ {odds_val}</div>
                            <div style="margin-top:8px;">
                                Edge: <strong style="color:#00cc44">+{edge*100:.1f}%</strong>
                                &nbsp;|&nbsp; Model: {model_prob*100:.1f}%
                                &nbsp;|&nbsp; Market implies: {market_prob*100:.1f}%
                            </div>
                            <div style="color:#8b949e; font-size:0.85em; margin-top:4px;">
                                The model gives {player} a {model_prob*100:.1f}% chance but the bookmaker
                                is only pricing in {market_prob*100:.1f}%. That's a {edge*100:.1f}% edge — worth betting.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif edge >= 0.02:
                        st.markdown(f"""
                        <div style="background:#0d1a2b; border:2px solid #4a9eff; border-radius:10px; padding:16px; margin:6px 0;">
                            <div style="font-size:1.1em; font-weight:bold; color:#4a9eff">🔵 SMALL EDGE — {player} @ {odds_val}</div>
                            <div style="margin-top:8px;">
                                Edge: <strong style="color:#4a9eff">+{edge*100:.1f}%</strong>
                                &nbsp;|&nbsp; Model: {model_prob*100:.1f}%  vs  Market: {market_prob*100:.1f}%
                            </div>
                            <div style="color:#8b949e; font-size:0.85em; margin-top:4px;">Edge exists but small. Only bet if you're confident.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        edge_str = f"{edge*100:+.1f}%"
                        st.markdown(f"""
                        <div style="background:#1a1a1a; border:2px solid #444; border-radius:10px; padding:16px; margin:6px 0;">
                            <div style="font-size:1.1em; color:#888">⚪ NO VALUE — {player} @ {odds_val}</div>
                            <div style="margin-top:8px; color:#666;">
                                Edge: {edge_str} &nbsp;|&nbsp; Model: {model_prob*100:.1f}%  vs  Market: {market_prob*100:.1f}%
                            </div>
                            <div style="color:#555; font-size:0.85em; margin-top:4px;">The bookmaker is pricing this correctly or worse than model. Skip.</div>
                        </div>
                        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Page: Today's Bets
# ─────────────────────────────────────────────
elif page == "📋 Today's Bets":
    st.title("Today's Bet Report")
    st.caption("Add today's matches manually or run the daily runner.")

    # Show past reports
    reports = load_reports()
    value_bets = [r for r in reports if r.get("has_value_bet") and "error" not in r]

    if not value_bets:
        st.info("No reports yet. Use the **Predict a Match** page to generate predictions, "
                "or run:\n```\npython -m src.agents.daily_runner\n```")

        st.subheader("Quick Multi-Match Entry")
        st.markdown("Paste matches below (one per line: `Player1 vs Player2, Surface, P1odds, P2odds`):")
        sample = "Carlos Alcaraz vs Jannik Sinner, Hard, 2.10, 1.75\nNovak Djokovic vs Daniil Medvedev, Hard, 1.80, 2.05"
        batch_input = st.text_area("Matches", value=sample, height=120)

        if st.button("Run Batch Prediction", type="primary"):
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            results = []
            for line in lines:
                try:
                    parts = [p.strip() for p in line.split(",")]
                    players = parts[0].split(" vs ")
                    p1, p2 = players[0].strip(), players[1].strip()
                    surf = parts[1] if len(parts) > 1 else "Hard"
                    p1o = float(parts[2]) if len(parts) > 2 else None
                    p2o = float(parts[3]) if len(parts) > 3 else None
                    r = predict_match(p1, p2, surf, p1_odds=p1o, p2_odds=p2o, verbose=False)
                    results.append(r)
                except Exception as e:
                    results.append({"error": str(e), "match": line})

            for r in results:
                if "error" in r:
                    st.error(f"{r['match']}: {r['error']}")
                    continue
                has_bet = r.get("has_value_bet")
                emoji = "✦" if has_bet else "·"
                conf_color = "#00cc44" if has_bet else "#4a9eff"
                st.markdown(f"""
                <div class="{'value-bet' if has_bet else 'no-bet'}">
                    <strong>{emoji} {r['predicted_winner']}</strong>
                    ({r['confidence']*100:.0f}%) — {r['match']} on {r['surface']}
                    {('<br><strong style="color:#00cc44">VALUE BET: ' +
                     ' | '.join(b['bet_on'] + ' @ ' + str(b['odds']) + ' (edge ' + b['edge_pct'] + ')'
                                for b in r['value_bets']) + '</strong>')
                     if has_bet else ''}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success(f"Found {len(value_bets)} value bets in report history")
        for bet in value_bets:
            for b in bet["value_bets"]:
                st.markdown(f"""
                <div class="value-bet">
                    <strong>✦ {b['bet_on']} @ {b['odds']}</strong>
                    &nbsp;|&nbsp; Edge: <strong style="color:#00cc44">{b['edge_pct']}</strong>
                    &nbsp;|&nbsp; Match: {bet['match']} ({bet['surface']})
                </div>
                """, unsafe_allow_html=True)
