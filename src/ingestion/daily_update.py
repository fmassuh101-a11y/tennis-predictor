"""
Daily update pipeline — keeps the bot fresh.

Runs in order:
1. Download latest ATP rankings from ESPN
2. Download latest WTA rankings from ESPN
3. Download any new ATP main-tour matches from tennis-data.co.uk (2025 + 2026 files)
4. Download any new WTA matches from tennis-data.co.uk (2025w + 2026w files)
5. Rebuild ATP player state (ELO, form, rest days)
6. Rebuild WTA player state
7. Check if Sackmann has published new 2025 challenger data

Run: python -m src.ingestion.daily_update

After this completes, restart the dashboard (streamlit run src/dashboard/app.py) to clear cache.
"""

import os
import sys
import csv
import sqlite3
import requests
import pandas as pd
from datetime import date, datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

RAW_DIR   = os.path.join(os.path.dirname(__file__), "../../data/raw")
ODDS_DIR  = os.path.join(RAW_DIR, "odds")
DB_PATH   = os.path.join(os.path.dirname(__file__), "../../data/tennis.db")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

ESPN_ATP_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/atp/rankings"
ESPN_WTA_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/wta/rankings"

ATP_RANKINGS_PATH = os.path.join(RAW_DIR, "current_atp_rankings.csv")
WTA_RANKINGS_PATH = os.path.join(RAW_DIR, "current_wta_rankings.csv")

TODAY = date.today().isoformat()


# ─── Step 1+2: Rankings ───────────────────────────────────────────────────────

def fetch_espn_rankings(url: str, label: str) -> list:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  {label}: ESPN HTTP {r.status_code}")
            return []
        data = r.json()
        ranks = data.get("rankings", [{}])[0].get("ranks", [])
        results = []
        for entry in ranks:
            athlete = entry.get("athlete", {})
            name = athlete.get("displayName", "")
            rank = entry.get("current", 0)
            points = entry.get("points", 0)
            if name and rank:
                results.append({"rank": rank, "player_name": name,
                                 "points": points, "ranking_date": TODAY})
        print(f"  {label}: {len(results)} players fetched")
        return results
    except Exception as e:
        print(f"  {label}: error — {e}")
        return []


def save_csv(rankings: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "player_name", "points", "ranking_date"])
        writer.writeheader()
        writer.writerows(rankings)


# ─── Step 3+4: Download odds files ───────────────────────────────────────────

TENNIS_DATA_FILES = {
    "odds_2025.xlsx":     "http://www.tennis-data.co.uk/2025/2025.xlsx",
    "odds_2026.xlsx":     "http://www.tennis-data.co.uk/2026/2026.xlsx",
    "odds_wta_2025.xlsx": "http://www.tennis-data.co.uk/2025w/2025.xlsx",
    "odds_wta_2026.xlsx": "http://www.tennis-data.co.uk/2026w/2026.xlsx",
}


def download_odds_files():
    os.makedirs(ODDS_DIR, exist_ok=True)
    for filename, url in TENNIS_DATA_FILES.items():
        path = os.path.join(ODDS_DIR, filename)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(path, "wb") as f:
                    f.write(r.content)
                print(f"  Updated {filename} ({len(r.content)//1024}KB)")
            else:
                print(f"  {filename}: HTTP {r.status_code} or empty response")
        except Exception as e:
            print(f"  {filename}: error — {e}")


# ─── Step 5+6: Rebuild player states ─────────────────────────────────────────

def fetch_api_tennis_yesterday():
    """Fetch yesterday's results from api-tennis.com (ATP Challenger + main tour)"""
    key_path = os.path.join(RAW_DIR, "..", "data", "api_tennis_key.txt")
    # Try a few locations
    for p in [os.path.join(os.path.dirname(__file__), "../../data/api_tennis_key.txt"),
              "data/api_tennis_key.txt"]:
        if os.path.exists(p):
            key_path = p
            break
    else:
        print("  No api-tennis.com key found — skipping challenger update")
        return

    from datetime import timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    today     = date.today().isoformat()

    import sqlite3, pandas as pd
    key = open(key_path).read().strip()
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "../../data/tennis.db"))
    cursor = conn.execute("PRAGMA table_info(matches)")
    db_cols = {row[1] for row in cursor.fetchall()}

    from difflib import get_close_matches as gcm
    df_names = pd.read_sql_query("""
        SELECT DISTINCT winner_name FROM matches WHERE winner_name IS NOT NULL
        UNION SELECT DISTINCT loser_name FROM matches WHERE loser_name IS NOT NULL
    """, conn)
    lookup = {}
    for name in df_names.iloc[:,0]:
        parts = str(name).strip().split()
        if len(parts) < 2: continue
        i = parts[0][0].lower()
        lookup[(parts[-1].lower(), i)] = name
        if len(parts) > 2:
            lookup[(" ".join(parts[1:]).lower(), i)] = name

    def res(n, cache={}):
        if n in cache: return cache[n]
        s = n.strip().rstrip(".")
        parts = s.split()
        if not parts: return n
        if len(parts[0].rstrip(".")) <= 2:
            init = parts[0][0].lower(); last = " ".join(parts[1:]).lower()
        else:
            last = parts[-1].lower(); init = parts[0][0].lower() if len(parts) > 1 else ""
        k = (last, init)
        if k in lookup: cache[n] = lookup[k]; return lookup[k]
        for (l,i),v in lookup.items():
            if l == last: cache[n] = v; return v
        cache[n] = n; return n

    CLAY_KW = ["clay","monte","madrid","rome","paris","hamburg","roland","rio","estoril",
               "barcelona","buenos aires","marrakech","kigali","hersonissos","lima","casablanca"]
    GRASS_KW = ["wimbledon","grass","halle","queens","hertogenbosch","eastbourne"]

    def surf(name):
        n = (name or "").lower()
        if any(c in n for c in CLAY_KW): return "Clay"
        if any(g in n for g in GRASS_KW): return "Grass"
        return "Hard"

    ROUND_MAP2 = {"final":"F","semi-final":"SF","quarter-final":"QF","round of 16":"R16",
                  "round of 32":"R32","round of 64":"R64","1st round":"R64","2nd round":"R32"}

    def rnd(r):
        r = (r or "").lower()
        for k,v in ROUND_MAP2.items():
            if k in r: return v
        return "R32"

    total = 0
    for etype, (src, lvl) in [("281",("atp_challenger","C")),("265",("tennis_api_atp","A"))]:
        r = requests.get("https://api.api-tennis.com/tennis/", params={
            "method":"get_fixtures","APIkey":key,
            "date_start":yesterday,"date_stop":today,"event_type_key":etype
        }, timeout=20)
        if r.status_code != 200: continue
        events = [e for e in r.json().get("result",[]) if e.get("event_status")=="Finished"]
        rows = []
        for e in events:
            p1 = e.get("event_first_player",""); p2 = e.get("event_second_player","")
            ws = e.get("event_winner",""); result = e.get("event_final_result","")
            try: s1,s2 = map(int,result.split(" - ")[:2])
            except: continue
            if ws=="First Player": w,l = p1,p2
            elif ws=="Second Player": w,l = p2,p1
            elif s1>s2: w,l = p1,p2
            elif s2>s1: w,l = p2,p1
            else: continue
            try:
                td = int(datetime.strptime(e["event_date"][:10],"%Y-%m-%d").strftime("%Y%m%d"))
            except: continue
            tname = e.get("tournament_name","")
            rows.append({"tourney_date":td,"tourney_name":tname,"surface":surf(tname),
                         "tourney_level":lvl,"winner_name":res(w),"loser_name":res(l),
                         "round":rnd(e.get("tournament_round","")),"best_of":3,"source":src})
        if rows:
            existing = pd.read_sql_query(f"SELECT tourney_date,winner_name,loser_name FROM matches WHERE source=?",
                                          conn, params=[src])
            ex_keys = set(zip(existing["tourney_date"].astype(int),
                              existing["winner_name"].astype(str),existing["loser_name"].astype(str)))
            new = [r for r in rows if (r["tourney_date"],r["winner_name"],r["loser_name"]) not in ex_keys]
            if new:
                pd.DataFrame(new)[[c for c in pd.DataFrame(new).columns if c in db_cols]].to_sql(
                    "matches", conn, if_exists="append", index=False)
                conn.commit()
                total += len(new)
                print(f"  api-tennis [{src}]: +{len(new)} new matches")
    conn.close()
    if total:
        print(f"  Total new matches from api-tennis.com: {total}")


def rebuild_atp_state():
    from src.ingestion.load_recent import run as load_atp
    from src.models.player_state import build_player_state
    print("\n  Loading new ATP main-tour matches into DB...")
    load_atp()
    print("  Fetching yesterday's challenger results from api-tennis.com...")
    fetch_api_tennis_yesterday()
    print("  Rebuilding ATP player state...")
    build_player_state()
    print("  ATP player state rebuilt.")


def rebuild_wta_state():
    from src.ingestion.load_recent_wta import run as load_wta
    from src.models.player_state_wta import build_wta_player_state
    print("\n  Loading new WTA matches into DB...")
    load_wta()
    print("  Rebuilding WTA player state...")
    build_wta_player_state()
    print("  WTA player state rebuilt.")


# ─── Step 7: Check for Sackmann 2025 challenger data ─────────────────────────

def check_sackmann_2025():
    url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_2025.csv"
    try:
        r = requests.head(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            print("\n  ✓ Sackmann 2025 challenger data IS NOW AVAILABLE!")
            print("  Run: python -m src.ingestion.sackmann  (to load it)")
            return True
        else:
            print("\n  Sackmann 2025 challenger file: not yet published (HTTP 404). Challenger data still from Nov 2024.")
            return False
    except Exception:
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print(f"\n{'='*60}")
    print(f"  Tennis Predictor Daily Update — {TODAY}")
    print(f"{'='*60}")

    print("\n[1/4] Fetching ATP rankings...")
    atp_ranks = fetch_espn_rankings(ESPN_ATP_URL, "ATP")
    if atp_ranks:
        save_csv(atp_ranks, ATP_RANKINGS_PATH)

    print("\n[2/4] Fetching WTA rankings...")
    wta_ranks = fetch_espn_rankings(ESPN_WTA_URL, "WTA")
    if wta_ranks:
        save_csv(wta_ranks, WTA_RANKINGS_PATH)

    print("\n[3/4] Downloading latest match files from tennis-data.co.uk...")
    download_odds_files()

    print("\n[4/5] Rebuilding player states...")
    try:
        rebuild_atp_state()
    except Exception as e:
        print(f"  ATP state error: {e}")
    try:
        rebuild_wta_state()
    except Exception as e:
        print(f"  WTA state error: {e}")

    print("\n[4.5/5] Enriching player states with Sackmann charting stats...")
    try:
        from src.ingestion.load_charting_stats import run as load_charting
        load_charting()
    except Exception as e:
        print(f"  Charting stats error: {e}")

    check_sackmann_2025()

    print("\n[5/5] Generating today's picks...")
    try:
        from src.ingestion.todays_matches import run as fetch_today
        fetch_today()
        print("  Today's picks saved to data/todays_picks.json")
    except Exception as e:
        print(f"  Today's picks error: {e}")

    print(f"\n{'='*60}")
    print("  Update complete. Restart dashboard to clear cache:")
    print("  streamlit run src/dashboard/app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
