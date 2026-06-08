"""Feature engineering pour la prédiction des matchs de Coupe du Monde.

Source : openfootball (scores) + Zafronix (altitudes de stade).
On traite les matchs par ordre chronologique et, pour chacun, on calcule les
features à partir de l'état des statistiques AVANT le match (pas de fuite de
données), puis on met à jour cet état.

Vecteur de features (ordre fixe) :
  - ranking_diff    : proxy de force (taux de victoire historique) team1 - team2
  - goals_avg_diff  : moyenne de buts marqués sur les 5 derniers matchs, diff
  - h2h_win_rate    : taux de réussite team1 dans les confrontations directes
  - is_knockout     : 1 en phase à élimination directe, 0 en poules
  - elevation_m     : altitude du stade (mètres)

Label : 0 = victoire team1, 1 = nul, 2 = victoire team2.
"""
import glob
import json
import os
from collections import defaultdict
from typing import Any

from config import DATA_DIR

FEATURES = [
    "ranking_diff",
    "goals_avg_diff",
    "h2h_win_rate",
    "is_knockout",
    "elevation_m",
]


def outcome(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return 0
    if home_score == away_score:
        return 1
    return 2


def is_knockout(stage: str | None) -> int:
    """Déduit la phase depuis un libellé de stage (côté inférence)."""
    if not stage:
        return 0
    return 0 if stage.lower().startswith("group") else 1


# Fabriques de module (picklables, contrairement aux lambdas) pour defaultdict.
def _new_record() -> list[int]:
    return [0, 0, 0, 0]


def _new_h2h() -> dict[str, int]:
    return {"a": 0, "b": 0, "d": 0}


class Stats:
    """État glissant des équipes, mis à jour match après match."""

    def __init__(self) -> None:
        # team -> [wins, draws, losses, played]
        self.record: dict[str, list[int]] = defaultdict(_new_record)
        # team -> liste chronologique des buts marqués
        self.goals: dict[str, list[int]] = defaultdict(list)
        # (teamA, teamB) triés -> {"a": wins_A, "b": wins_B, "d": draws}
        self.h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(_new_h2h)

    def strength(self, team: str) -> float:
        w, _d, _l, p = self.record[team]
        return w / p if p else 0.5

    def recent_avg(self, team: str) -> float:
        g = self.goals[team][-5:]
        return sum(g) / len(g) if g else 1.0

    def h2h_win_rate(self, t1: str, t2: str) -> float:
        key = tuple(sorted((t1, t2)))
        rec = self.h2h.get(key)
        if not rec:
            return 0.5
        total = rec["a"] + rec["b"] + rec["d"]
        if total == 0:
            return 0.5
        t1_wins = rec["a"] if key[0] == t1 else rec["b"]
        return (t1_wins + 0.5 * rec["d"]) / total

    def features(
        self, t1: str, t2: str, knockout: int, elevation: float
    ) -> list[float]:
        return [
            self.strength(t1) - self.strength(t2),
            self.recent_avg(t1) - self.recent_avg(t2),
            self.h2h_win_rate(t1, t2),
            float(knockout),
            float(elevation),
        ]

    def update(self, home: str, away: str, hs: int, as_: int) -> None:
        res = outcome(hs, as_)
        if res == 0:
            self.record[home][0] += 1
            self.record[away][2] += 1
        elif res == 1:
            self.record[home][1] += 1
            self.record[away][1] += 1
        else:
            self.record[home][2] += 1
            self.record[away][0] += 1
        self.record[home][3] += 1
        self.record[away][3] += 1
        self.goals[home].append(hs)
        self.goals[away].append(as_)
        key = tuple(sorted((home, away)))
        rec = self.h2h[key]
        if res == 1:
            rec["d"] += 1
        else:
            winner = home if res == 0 else away
            rec["a" if key[0] == winner else "b"] += 1


def _norm_city(city: str) -> str:
    return city.strip().lower()


def load_stadium_elevations() -> tuple[dict[str, float], dict[str, float]]:
    """Renvoie (par_id, par_ville) depuis Zafronix /stadiums.

    - par_id   : pour l'inférence (le frontend envoie un stadiumId Zafronix).
    - par_ville: pour l'entraînement (openfootball ne fournit que la ville).
    """
    path = os.path.join(DATA_DIR, "stadiums.json")
    if not os.path.exists(path):
        return {}, {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    by_id: dict[str, float] = {}
    by_city: dict[str, float] = {}
    for s in data:
        if not isinstance(s, dict):
            continue
        elev = float(s.get("elevationM") or 0)
        if s.get("id"):
            by_id[s["id"]] = elev
        if s.get("city"):
            by_city[_norm_city(s["city"])] = elev
    return by_id, by_city


def _city_from_ground(ground: str | None) -> str:
    # openfootball "ground" = "Nom du stade, Ville" -> on garde la ville.
    if not ground:
        return ""
    return ground.split(",")[-1].strip()


def load_matches() -> list[dict[str, Any]]:
    """Matchs openfootball normalisés (score valide), triés par date.

    Chaque entrée : {date, homeTeam, awayTeam, homeScore, awayScore,
                     knockout(bool), city}.
    """
    matches: list[dict[str, Any]] = []
    for path in glob.glob(os.path.join(DATA_DIR, "[0-9]" * 4 + ".json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for m in payload.get("matches", []):
            score = (m.get("score") or {}).get("ft")
            if not score or len(score) < 2:
                continue
            t1, t2 = m.get("team1"), m.get("team2")
            if not t1 or not t2:
                continue
            matches.append(
                {
                    "date": m.get("date") or "",
                    "homeTeam": t1,
                    "awayTeam": t2,
                    "homeScore": int(score[0]),
                    "awayScore": int(score[1]),
                    "knockout": 0 if m.get("group") else 1,
                    "city": _city_from_ground(m.get("ground")),
                }
            )
    matches.sort(key=lambda m: m["date"])
    return matches


def build_dataset() -> tuple[list[list[float]], list[int], Stats, dict[str, float]]:
    """Construit (X, y) et renvoie l'état final + altitudes par id (pour l'inférence)."""
    by_id, by_city = load_stadium_elevations()
    stats = Stats()
    X: list[list[float]] = []
    y: list[int] = []

    for m in load_matches():
        home, away = m["homeTeam"], m["awayTeam"]
        hs, as_ = m["homeScore"], m["awayScore"]
        elev = by_city.get(_norm_city(m["city"]), 0.0)
        X.append(stats.features(home, away, m["knockout"], elev))
        y.append(outcome(hs, as_))
        stats.update(home, away, hs, as_)

    return X, y, stats, by_id
