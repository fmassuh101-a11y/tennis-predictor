"""
Download ATP match results + betting odds from tennis-data.co.uk
This fills the 2024-2025 gap and gives us real bookmaker odds.
URL pattern: http://www.tennis-data.co.uk/[year]/[year].zip
"""

import os
import io
import time
import zipfile
import requests
import pandas as pd

ODDS_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/odds")
BASE_URL = "http://www.tennis-data.co.uk"

# Years to download - go back to 2010 for odds history
YEARS = range(2010, 2027)


def download_year(year: int) -> pd.DataFrame | None:
    """Try to download a year's worth of ATP matches + odds."""
    # tennis-data.co.uk movió sus archivos detrás de una carpeta con nombre
    # ofuscado. Las rutas directas /AÑO/AÑO.xlsx empezaron a dar 404 en algún
    # momento de 2026 y por eso la descarga dejó de traer datos nuevos sin
    # avisar: el pipeline seguía corriendo y la base se quedó congelada en
    # marzo. La ruta de abajo se sacó del HTML de su propia página de
    # descargas, y va PRIMERO para que sea la que se use.
    CARPETA = "hrjk-85HytOjkhth76j_ygh4jf7"
    urls_to_try = [
        f"{BASE_URL}/{CARPETA}/{year}/{year}.xlsx",
        f"{BASE_URL}/{year}/{year}.zip",
        f"{BASE_URL}/{year}/{year}.xlsx",
        f"{BASE_URL}/{year}/atp_{year}.xlsx",
        f"{BASE_URL}/{year}/ATP_{year}.xlsx",
        f"{BASE_URL}/{year}/{year}atp.xlsx",
        f"{BASE_URL}/{year}/ATP_{year}.csv",
        f"{BASE_URL}/{year}/atp_{year}.csv",
    ]

    for url in urls_to_try:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue

            print(f"  Found: {url}")
            content_type = r.headers.get("Content-Type", "")

            # Handle ZIP
            if url.endswith(".zip") or "zip" in content_type:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                csv_files = [f for f in z.namelist()
                             if f.endswith(".csv") or f.endswith(".xlsx")]
                if not csv_files:
                    continue
                fname = csv_files[0]
                raw_path = os.path.join(ODDS_DIR, f"odds_{year}.csv")
                with open(raw_path, "wb") as f:
                    f.write(z.read(fname))
                return raw_path

            # Handle CSV directly
            elif url.endswith(".csv"):
                raw_path = os.path.join(ODDS_DIR, f"odds_{year}.csv")
                with open(raw_path, "wb") as f:
                    f.write(r.content)
                return raw_path

            # Handle Excel
            elif url.endswith(".xlsx"):
                raw_path = os.path.join(ODDS_DIR, f"odds_{year}.xlsx")
                with open(raw_path, "wb") as f:
                    f.write(r.content)
                return raw_path

        except Exception as e:
            continue

    return None


def load_odds_file(fpath: str) -> pd.DataFrame | None:
    """Load an odds file — auto-detects Excel vs CSV by magic bytes."""
    try:
        with open(fpath, "rb") as f:
            magic = f.read(4)
        # PK = ZIP/XLSX, D0CF = legacy XLS
        is_excel = magic[:2] == b"PK" or magic[:4] == b"\xd0\xcf\x11\xe0"
        if is_excel:
            df = pd.read_excel(fpath, engine="openpyxl")
        else:
            df = pd.read_csv(fpath, encoding="latin1", low_memory=False)
        return df
    except Exception as e:
        print(f"  Error reading {fpath}: {e}")
        return None


ODDS_COLUMN_MAP = {
    # Standard tennis-data.co.uk column names
    "Date": "match_date",
    "Tournament": "tourney_name",
    "Series": "tourney_level",
    # OJO: tennis-data.co.uk trae DOS columnas parecidas. "Court" dice si la
    # cancha es techada o al aire libre (Indoor/Outdoor); "Surface" dice el
    # material (Hard/Clay/Grass), que es lo que importa para el Elo por
    # superficie. Estaba tomando "Court", así que todos los partidos entraban
    # con superficie "Outdoor" y el Elo por superficie nunca se separaba: se
    # quedaba en el valor inicial de 1500 para todo el mundo.
    "Surface": "surface",
    "Court": "court_type",
    "Round": "round",
    "Winner": "winner_name",
    "Loser": "loser_name",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "WPts": "winner_rank_points",
    "LPts": "loser_rank_points",
    "W1": "w_sets_1", "W2": "w_sets_2", "W3": "w_sets_3",
    "L1": "l_sets_1", "L2": "l_sets_2", "L3": "l_sets_3",
    "Wsets": "w_sets_total",
    "Lsets": "l_sets_total",
    # Best bookmaker odds (these are the key columns)
    "B365W": "odds_b365_winner",
    "B365L": "odds_b365_loser",
    "PSW": "odds_ps_winner",
    "PSL": "odds_ps_loser",
    "MaxW": "odds_max_winner",
    "MaxL": "odds_max_loser",
    "AvgW": "odds_avg_winner",
    "AvgL": "odds_avg_loser",
}


def normalize_odds_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Standardize column names and add year."""
    df = df.rename(columns=ODDS_COLUMN_MAP)
    # Keep only mapped columns that exist
    keep = [v for k, v in ODDS_COLUMN_MAP.items() if v in df.columns]
    df = df[keep].copy()
    df["year"] = year
    # Convert odds to float
    for col in df.columns:
        if col.startswith("odds_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def download_all():
    os.makedirs(ODDS_DIR, exist_ok=True)
    all_dfs = []

    for year in YEARS:
        raw_path = os.path.join(ODDS_DIR, f"odds_{year}.csv")
        xlsx_path = os.path.join(ODDS_DIR, f"odds_{year}.xlsx")

        if os.path.exists(raw_path) or os.path.exists(xlsx_path):
            print(f"  {year}: already downloaded, skipping")
            fpath = raw_path if os.path.exists(raw_path) else xlsx_path
        else:
            print(f"Downloading {year}...")
            fpath = download_year(year)
            if fpath is None:
                print(f"  {year}: NOT FOUND on tennis-data.co.uk")
                time.sleep(0.5)
                continue
            time.sleep(0.5)

        df = load_odds_file(fpath)
        if df is not None and len(df) > 0:
            df_norm = normalize_odds_df(df, year)
            all_dfs.append(df_norm)
            print(f"  {year}: {len(df)} matches loaded")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(ODDS_DIR, "odds_combined.csv")
        combined.to_csv(out_path, index=False)
        print(f"\nSaved {len(combined):,} total odds records to {out_path}")
        return combined
    else:
        print("No odds data downloaded.")
        return None


if __name__ == "__main__":
    df = download_all()
    if df is not None:
        print("\nColumns available:", list(df.columns))
        print("Years covered:", sorted(df["year"].unique()))
        print("Sample records:\n", df.tail(10).to_string())
