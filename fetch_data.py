"""Télécharge les données d'entraînement (source hybride).

- Scores historiques : dataset public openfootball/worldcup.json (aucune clé).
- Altitudes de stade : API Zafronix /stadiums (clé free tier).

Sauvegarde data/{year}.json (openfootball brut) + data/stadiums.json (Zafronix).

Usage : python fetch_data.py
"""
import json
import os
import time

import requests

from config import DATA_DIR, TRAIN_YEARS, WC_API_KEY, WC_BASE_URL

OPENFOOTBALL_BASE = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master"
)


def fetch_openfootball() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for year in TRAIN_YEARS:
        out = os.path.join(DATA_DIR, f"{year}.json")
        if os.path.exists(out):
            print(f"  ✓ {year} déjà présent")
            continue
        url = f"{OPENFOOTBALL_BASE}/{year}/worldcup.json"
        res = requests.get(url, timeout=20)
        if res.status_code == 404:
            print(f"  – {year} indisponible (404)")
            continue
        res.raise_for_status()
        print(f"  ↓ openfootball {year}")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res.json(), f, ensure_ascii=False)
        time.sleep(0.2)


def fetch_stadiums() -> None:
    out = os.path.join(DATA_DIR, "stadiums.json")
    if os.path.exists(out):
        print("  ✓ stadiums déjà présent")
        return
    # Optionnel : ne sert qu'à la feature elevation_m. En cas d'échec (quota 429,
    # réseau…), on N'INTERROMPT PAS le build : elevation_m vaudra simplement 0.
    try:
        res = requests.get(
            f"{WC_BASE_URL}/stadiums",
            headers={"X-API-Key": WC_API_KEY},
            timeout=20,
        )
        res.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ stades indisponibles ({e}). elevation_m=0, on continue.")
        return
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res.json(), f, ensure_ascii=False)
    print("  ↓ Zafronix /stadiums OK")


def main() -> None:
    print("Téléchargement des données d'entraînement")
    fetch_openfootball()
    fetch_stadiums()
    print("Terminé.")


if __name__ == "__main__":
    main()
