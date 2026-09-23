"""
Download all ATP and ITF match data from Jeff Sackmann's GitHub repos.
Saves raw CSVs to data/raw/sackmann/atp/ and data/raw/sackmann/itf/
"""

import os
import time
import requests
from tqdm import tqdm

BASE_RAW = os.path.join(os.path.dirname(__file__), "../../data/raw/sackmann")
ATP_DIR = os.path.join(BASE_RAW, "atp")
ITF_DIR = os.path.join(BASE_RAW, "itf")

ATP_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
ITF_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_itf/master"

# ATP: main tour matches + challenger/qualifying matches
ATP_YEARS = range(1968, 2025)  # 1968–2024

# ITF men's main draw files (Sackmann ITF repo naming convention)
ITF_YEARS = range(2000, 2025)  # ITF data reliable from 2000


def download_file(url: str, dest_path: str) -> bool:
    """Download a single file. Returns True if successful."""
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping: {os.path.basename(dest_path)}")
        return True

    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        with open(dest_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"  NOT FOUND (HTTP {response.status_code}): {url}")
        return False


def download_atp():
    """Download all ATP main tour + challenger/qualifying CSVs."""
    os.makedirs(ATP_DIR, exist_ok=True)
    print("\n=== Downloading ATP Main Tour Matches ===")

    success, failed = 0, 0
    for year in tqdm(ATP_YEARS, desc="ATP matches"):
        # Main tour
        url = f"{ATP_BASE_URL}/atp_matches_{year}.csv"
        dest = os.path.join(ATP_DIR, f"atp_matches_{year}.csv")
        if download_file(url, dest):
            success += 1
        else:
            failed += 1
        time.sleep(0.1)  # be polite to GitHub

    print(f"\n=== Downloading ATP Challenger + Qualifying Matches ===")
    for year in tqdm(range(1978, 2025), desc="Challenger/Qual"):
        url = f"{ATP_BASE_URL}/atp_matches_qual_chall_{year}.csv"
        dest = os.path.join(ATP_DIR, f"atp_matches_qual_chall_{year}.csv")
        download_file(url, dest)
        time.sleep(0.1)

    print(f"\nATP download complete: {success} files downloaded, {failed} not found.")


def download_itf():
    """Download ITF men's main draw match CSVs."""
    os.makedirs(ITF_DIR, exist_ok=True)
    print("\n=== Downloading ITF Men's Matches ===")

    success, failed = 0, 0
    for year in tqdm(ITF_YEARS, desc="ITF matches"):
        # Sackmann ITF repo uses this naming
        url = f"{ITF_BASE_URL}/itf_men_main_draw_matches_{year}.csv"
        dest = os.path.join(ITF_DIR, f"itf_men_{year}.csv")
        if download_file(url, dest):
            success += 1
        else:
            failed += 1
        time.sleep(0.1)

    print(f"\nITF download complete: {success} files downloaded, {failed} not found.")


def download_atp_players():
    """Download ATP player info (name, hand, DOB, country)."""
    print("\n=== Downloading ATP Player Index ===")
    url = f"{ATP_BASE_URL}/atp_players.csv"
    dest = os.path.join(ATP_DIR, "atp_players.csv")
    if download_file(url, dest):
        print("Player index downloaded.")
    else:
        print("Player index not found.")


def download_atp_rankings():
    """Download ATP rankings files."""
    os.makedirs(ATP_DIR, exist_ok=True)
    print("\n=== Downloading ATP Rankings ===")
    # Rankings files: atp_rankings_00s.csv, 10s.csv, 20s.csv, current.csv
    for fname in ["atp_rankings_00s.csv", "atp_rankings_10s.csv",
                  "atp_rankings_20s.csv", "atp_rankings_current.csv"]:
        url = f"{ATP_BASE_URL}/{fname}"
        dest = os.path.join(ATP_DIR, fname)
        download_file(url, dest)
        time.sleep(0.1)
    print("Rankings download complete.")


if __name__ == "__main__":
    download_atp_players()
    download_atp_rankings()
    download_atp()
    download_itf()
    print("\nAll downloads complete.")
