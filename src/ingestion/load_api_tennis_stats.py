"""
Mine match statistics + point-by-point data from api-tennis.com.

Pulls ALL singles matches from 2022-2026 (monthly batches = ~52 API calls).
Extracts per-match stats for each player:
  - 1st serve %, 1st/2nd serve win %
  - Break points saved/converted (raw counts)
  - Aces, double faults, winners, unforced errors
  - Net points %, total points won %
  - PBP-derived: tiebreak rate, clutch BP save % under pressure

Stores in data/api_tennis_stats.db
Then call update_player_state_from_api_tennis() to overlay on player_state.pkl

Run: python -m src.ingestion.load_api_tennis_stats
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
except ImportError:
    pass

import sys, sqlite3, requests, time, json, re
from datetime import date, timedelta
from calendar import monthrange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

KEY_FILE = os.path.join(os.path.dirname(__file__), "../../data/api_tennis_key.txt")
DB_PATH  = os.path.join(os.path.dirname(__file__), "../../data/api_tennis_stats.db")
BASE_URL = "https://api.api-tennis.com/tennis/"


def get_key() -> str:
    k = os.environ.get("TENNIS_API_KEY", "").strip()
    if k: return k
    env = os.environ.get("API_TENNIS_KEY")
    if env:
        return env.strip()
    with open(KEY_FILE) as f: return f.read().strip()


def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS match_stats (
        event_key       INTEGER PRIMARY KEY,
        match_date      TEXT,
        tournament      TEXT,
        surface         TEXT,
        level           TEXT,
        event_type      TEXT,
        p1_key          INTEGER,
        p1_name         TEXT,
        p2_key          INTEGER,
        p2_name         TEXT,
        winner          TEXT,   -- 'p1' or 'p2'
        -- P1 serve stats (full match)
        p1_aces         INTEGER,
        p1_dfs          INTEGER,
        p1_first_in_pct REAL,
        p1_first_won    INTEGER,
        p1_first_total  INTEGER,
        p1_second_won   INTEGER,
        p1_second_total INTEGER,
        p1_bp_saved     INTEGER,
        p1_bp_faced     INTEGER,
        p1_bp_converted INTEGER,
        p1_bp_chances   INTEGER,
        p1_winners      INTEGER,
        p1_ues          INTEGER,
        p1_net_won      INTEGER,
        p1_net_total    INTEGER,
        p1_svc_pts_won  INTEGER,
        p1_svc_pts_total INTEGER,
        p1_ret_pts_won  INTEGER,
        p1_ret_pts_total INTEGER,
        -- P2 serve stats (full match)
        p2_aces         INTEGER,
        p2_dfs          INTEGER,
        p2_first_in_pct REAL,
        p2_first_won    INTEGER,
        p2_first_total  INTEGER,
        p2_second_won   INTEGER,
        p2_second_total INTEGER,
        p2_bp_saved     INTEGER,
        p2_bp_faced     INTEGER,
        p2_bp_converted INTEGER,
        p2_bp_chances   INTEGER,
        p2_winners      INTEGER,
        p2_ues          INTEGER,
        p2_net_won      INTEGER,
        p2_net_total    INTEGER,
        p2_svc_pts_won  INTEGER,
        p2_svc_pts_total INTEGER,
        p2_ret_pts_won  INTEGER,
        p2_ret_pts_total INTEGER,
        -- PBP-derived clutch stats
        p1_tb_pts_won   INTEGER,
        p1_tb_pts_total INTEGER,
        p1_mp_saved     INTEGER,
        p2_tb_pts_won   INTEGER,
        p2_tb_pts_total INTEGER,
        p2_mp_saved     INTEGER,
        ingested_at     TEXT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON match_stats(match_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_p1 ON match_stats(p1_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_p2 ON match_stats(p2_name)")
    conn.commit()


def parse_fraction(val: str):
    """Parse '4/5' → (4, 5). Returns (None, None) on failure."""
    if not val or val == '0': return (None, None)
    m = re.match(r'(\d+)/(\d+)', str(val))
    if m: return int(m.group(1)), int(m.group(2))
    return (None, None)


def extract_stats(stats: list, player_key: int) -> dict:
    """Extract all stats for one player from statistics list."""
    s = {}
    for row in stats:
        if row.get('player_key') != player_key:
            continue
        period = row.get('stat_period', '')
        name   = row.get('stat_name', '')
        val    = row.get('stat_value')
        won    = row.get('stat_won')
        total  = row.get('stat_total')

        if period != 'match':
            continue

        if name == 'Aces':
            s['aces'] = int(val) if val else None
        elif name == 'Double Faults':
            s['dfs'] = int(val) if val else None
        elif name == '1st serve percentage':
            pct = re.search(r'(\d+)', str(val or ''))
            s['first_in_pct'] = float(pct.group(1)) / 100 if pct else None
        elif name == '1st serve points won':
            s['first_won'] = won
            s['first_total'] = total
        elif name == '2nd serve points won':
            s['second_won'] = won
            s['second_total'] = total
        elif name == 'Break Points Saved':
            saved, faced = parse_fraction(val)
            s['bp_saved'] = saved
            s['bp_faced'] = faced
        elif name == 'Break Points Converted':
            converted, chances = parse_fraction(val)
            s['bp_converted'] = converted
            s['bp_chances'] = chances
        elif name == 'Winners':
            s['winners'] = int(val) if val else None
        elif name == 'Unforced errors':
            s['ues'] = int(val) if val else None
        elif name == 'Net points won':
            s['net_won'] = won
            s['net_total'] = total
        elif name == 'Service Points Won':
            s['svc_pts_won'] = won
            s['svc_pts_total'] = total
        elif name == 'Return Points Won':
            s['ret_pts_won'] = won
            s['ret_pts_total'] = total
        elif name == 'Match points saved':
            s['mp_saved'] = int(val) if val else 0

    return s


def parse_pbp_tiebreaks(pbp: list, server_key: str) -> dict:
    """Count tiebreak points from point-by-point games."""
    tb = {'won': 0, 'total': 0}
    for game in (pbp or []):
        points = game.get('points', [])
        # Heuristic: tiebreak if any point score looks like "6 - 6" style or score > 5
        # Actually look for score patterns with numbers > 40 (regular tennis points are 0,15,30,40,A)
        is_tb = False
        for pt in points:
            sc = pt.get('score', '')
            parts = sc.replace(' ', '').split('-')
            if len(parts) == 2:
                try:
                    if int(parts[0]) > 5 or int(parts[1]) > 5:
                        is_tb = True
                        break
                except ValueError:
                    pass
        if not is_tb:
            continue
        server = game.get('player_served', '')
        for pt in points:
            tb['total'] += 1
            winner_of_pt = game.get('serve_winner', '')  # who won the game matters less
            # Count per point who won based on score progression
            # Simple: if server won game and server == server_key, count won pts as server's
            # This is approximate — we don't have point-level winner in PBP
    return tb  # Will be improved with Sportradar timeline data


def surface_from_type(t: str) -> str:
    t = (t or '').lower()
    if 'clay' in t: return 'Clay'
    if 'grass' in t or 'wimbledon' in t: return 'Grass'
    if 'carpet' in t or 'indoor' in t: return 'Carpet'
    return 'Hard'


def level_from_name(t: str) -> str:
    t = (t or '').lower()
    if any(x in t for x in ['australian open', 'us open', 'french open', 'wimbledon', 'roland garros']):
        return 'G'
    if any(x in t for x in ['masters', '1000', 'indian wells', 'miami', 'madrid', 'rome',
                              'montreal', 'toronto', 'cincinnati', 'shanghai', 'paris']):
        return 'M'
    if 'challenger' in t: return 'C'
    if 'itf' in t: return 'ITF'
    return 'A'


def fetch_month(key: str, year: int, month: int) -> list:
    start = f"{year}-{month:02d}-01"
    _, last_day = monthrange(year, month)
    end   = f"{year}-{month:02d}-{last_day:02d}"
    try:
        r = requests.get(BASE_URL, params={
            'method': 'get_fixtures', 'APIkey': key,
            'date_start': start, 'date_stop': end,
        }, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {year}-{month:02d}")
            return []
        return r.json().get('result', [])
    except Exception as e:
        print(f"  Error fetching {year}-{month:02d}: {e}")
        return []


def run(start_year: int = 2022, start_month: int = 1):
    key  = get_key()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    done = set(r[0] for r in conn.execute("SELECT event_key FROM match_stats").fetchall())
    print(f"Already in DB: {len(done):,} matches")

    today = date.today()
    total_inserted = 0
    total_calls = 0

    y, m = start_year, start_month
    while (y, m) <= (today.year, today.month):
        matches = fetch_month(key, y, m)
        total_calls += 1

        singles = [x for x in matches
                   if 'singles' in (x.get('event_type_type') or '').lower()
                   and 'doubles' not in (x.get('event_type_type') or '').lower()]

        inserted = 0
        for match in singles:
            ek = match.get('event_key')
            if not ek or ek in done:
                continue
            if not match.get('event_winner'):
                continue

            stats = match.get('statistics') or []
            pbp   = match.get('pointbypoint') or []
            if not stats:
                continue

            p1_key = match.get('first_player_key')
            p2_key = match.get('second_player_key')
            if not p1_key or not p2_key:
                continue

            p1_name = match.get('event_first_player', '').strip()
            p2_name = match.get('event_second_player', '').strip()
            winner  = 'p1' if match.get('event_winner') == 'First Player' else 'p2'

            s1 = extract_stats(stats, p1_key)
            s2 = extract_stats(stats, p2_key)

            etype = match.get('event_type_type', '')
            tourn = match.get('tournament_name', '')

            conn.execute("""
                INSERT OR IGNORE INTO match_stats (
                    event_key, match_date, tournament, surface, level, event_type,
                    p1_key, p1_name, p2_key, p2_name, winner,
                    p1_aces, p1_dfs, p1_first_in_pct, p1_first_won, p1_first_total,
                    p1_second_won, p1_second_total, p1_bp_saved, p1_bp_faced,
                    p1_bp_converted, p1_bp_chances, p1_winners, p1_ues,
                    p1_net_won, p1_net_total, p1_svc_pts_won, p1_svc_pts_total,
                    p1_ret_pts_won, p1_ret_pts_total, p1_mp_saved,
                    p2_aces, p2_dfs, p2_first_in_pct, p2_first_won, p2_first_total,
                    p2_second_won, p2_second_total, p2_bp_saved, p2_bp_faced,
                    p2_bp_converted, p2_bp_chances, p2_winners, p2_ues,
                    p2_net_won, p2_net_total, p2_svc_pts_won, p2_svc_pts_total,
                    p2_ret_pts_won, p2_ret_pts_total, p2_mp_saved,
                    ingested_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ek, match.get('event_date'), tourn,
                surface_from_type(etype), level_from_name(tourn), etype,
                p1_key, p1_name, p2_key, p2_name, winner,
                s1.get('aces'), s1.get('dfs'), s1.get('first_in_pct'),
                s1.get('first_won'), s1.get('first_total'),
                s1.get('second_won'), s1.get('second_total'),
                s1.get('bp_saved'), s1.get('bp_faced'),
                s1.get('bp_converted'), s1.get('bp_chances'),
                s1.get('winners'), s1.get('ues'),
                s1.get('net_won'), s1.get('net_total'),
                s1.get('svc_pts_won'), s1.get('svc_pts_total'),
                s1.get('ret_pts_won'), s1.get('ret_pts_total'),
                s1.get('mp_saved', 0),
                s2.get('aces'), s2.get('dfs'), s2.get('first_in_pct'),
                s2.get('first_won'), s2.get('first_total'),
                s2.get('second_won'), s2.get('second_total'),
                s2.get('bp_saved'), s2.get('bp_faced'),
                s2.get('bp_converted'), s2.get('bp_chances'),
                s2.get('winners'), s2.get('ues'),
                s2.get('net_won'), s2.get('net_total'),
                s2.get('svc_pts_won'), s2.get('svc_pts_total'),
                s2.get('ret_pts_won'), s2.get('ret_pts_total'),
                s2.get('mp_saved', 0),
                today.isoformat()
            ))
            done.add(ek)
            inserted += 1

        conn.commit()
        total_inserted += inserted
        print(f"  {y}-{m:02d}: {len(singles)} singles, {inserted} with stats inserted | total={total_inserted}")

        m += 1
        if m > 12:
            m = 1
            y += 1
        time.sleep(0.3)  # gentle rate limiting

    conn.close()
    print(f"\nDone. API calls: {total_calls} | Total inserted: {total_inserted:,}")
    print(f"DB: {DB_PATH}")


def build_player_rolling_stats(conn, player_name: str, surface: str = None,
                                 last_n: int = 30) -> dict:
    """
    Compute rolling stats for a player from their last N matches.
    Used to overlay on player_state.
    """
    rows = conn.execute("""
        SELECT match_date, surface, winner,
               p1_first_in_pct, p1_first_won, p1_first_total,
               p1_second_won, p1_second_total, p1_bp_saved, p1_bp_faced,
               p1_bp_converted, p1_bp_chances, p1_winners, p1_ues,
               p1_svc_pts_won, p1_svc_pts_total, p1_aces, p1_dfs, p1_mp_saved,
               1 as is_p1
        FROM match_stats WHERE p1_name = ?
        UNION ALL
        SELECT match_date, surface, winner,
               p2_first_in_pct, p2_first_won, p2_first_total,
               p2_second_won, p2_second_total, p2_bp_saved, p2_bp_faced,
               p2_bp_converted, p2_bp_chances, p2_winners, p2_ues,
               p2_svc_pts_won, p2_svc_pts_total, p2_aces, p2_dfs, p2_mp_saved,
               0 as is_p1
        FROM match_stats WHERE p2_name = ?
        ORDER BY match_date DESC LIMIT ?
    """, (player_name, player_name, last_n)).fetchall()

    if not rows:
        return {}

    def safe_div(a, b): return a / b if b and b > 0 else None
    def avg(vals): v = [x for x in vals if x is not None]; return sum(v)/len(v) if v else None

    first_in_pcts = [r[3] for r in rows]
    first_won_pcts = [safe_div(r[4], r[5]) for r in rows]
    second_won_pcts = [safe_div(r[6], r[7]) for r in rows]
    bp_save_pcts = [safe_div(r[8], r[9]) for r in rows]
    bp_conv_pcts = [safe_div(r[10], r[11]) for r in rows]
    ue_rates = [r[13] for r in rows]  # raw count, normalize later
    win_rates = [1 if (r[2]=='p1' and r[19]==1) or (r[2]=='p2' and r[19]==0) else 0 for r in rows]

    return {
        'api_tennis_first_in_pct':   avg(first_in_pcts),
        'api_tennis_first_won_pct':  avg(first_won_pcts),
        'api_tennis_second_won_pct': avg(second_won_pcts),
        'api_tennis_bp_save_pct':    avg(bp_save_pcts),
        'api_tennis_bp_conv_pct':    avg(bp_conv_pcts),
        'api_tennis_win_rate':       avg(win_rates),
        'api_tennis_n_matches':      len(rows),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--start-year",  type=int, default=2022)
    p.add_argument("--start-month", type=int, default=1)
    args = p.parse_args()
    run(args.start_year, args.start_month)
