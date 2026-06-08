# ⚽ predict_ml — API de prédiction Coupe du Monde

Service ML (FastAPI + XGBoost) qui prédit l'issue d'un match (victoire team1 /
nul / victoire team2) à partir de l'historique des Coupes du Monde (1930→2022).

## Sources de données (hybride)

| Donnée | Source | Clé ? |
|---|---|---|
| Scores historiques | [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) — `master/{year}/worldcup.json` | Non |
| Altitudes des stades | API Zafronix `/stadiums` (champ `elevationM`) | Oui (free tier) |

## Pipeline

```
fetch_data.py  →  data/{year}.json (openfootball) + data/stadiums.json (Zafronix)
features.py    →  vecteurs de features (utilisé par train + app)
train.py       →  model.pkl + stats.pkl
app.py         →  API REST /predict, /health
```

## Features

| Feature | Description |
|---|---|
| `ranking_diff` | Proxy de force (taux de victoire historique) team1 − team2 |
| `goals_avg_diff` | Moyenne de buts marqués sur les 5 derniers matchs, différence |
| `h2h_win_rate` | Taux de réussite team1 en confrontations directes |
| `is_knockout` | 1 en phase à élimination directe, 0 en poules |
| `elevation_m` | Altitude du stade (jointure openfootball↔Zafronix par ville) |

Label : `0` = victoire team1 · `1` = nul · `2` = victoire team2.
Modèle : `XGBClassifier(n_estimators=200, max_depth=4)`, 3 classes.

## Démarrage

```bash
cd predict_ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py        # openfootball (1930→2022) + stades Zafronix
python train.py             # entraîne → model.pkl + stats.pkl
uvicorn app:app --reload    # → http://localhost:8000
```

> Le port doit correspondre à `VITE_ML_API_URL` côté frontend (défaut 8000).
> Si le 8000 est pris (ex: `php artisan serve`), lance sur `--port 8001` et
> mets `VITE_ML_API_URL=http://localhost:8001`. Après tout changement de `.env`,
> **redémarre `npm run dev`** (Vite met les variables en cache au démarrage).

Clé Zafronix : variable d'env `WC_API_KEY` (sinon clé free tier par défaut dans `config.py`).

## API

### `POST /predict`

```json
// requête
{ "team1": "France", "team2": "Brazil", "stage": "final", "stadiumId": "metlife-stadium" }

// réponse
{ "team1_win": 41.2, "draw": 28.5, "team2_win": 30.3, "favorite": "France", "confidence": "low" }
```

`confidence` : `low` (<55 %), `medium` (55–65 %), `high` (>65 %).
`stage` accepte les libellés du frontend (`group_a`, `r16`, `final`…) ; seul
le caractère poule/élimination est utilisé. `stadiumId` est un id stade Zafronix.

### `GET /health`

```json
{ "status": "ok", "model_loaded": true }
```

## Intégration frontend

Le dashboard `world-coupe-2026` pointe vers `VITE_ML_API_URL` (défaut
`http://localhost:8000`). Remplacer le stub de `usePrediction.ts` par un appel à
`POST /predict`.

> ⚠️ Conflit de nommage entre sources : openfootball utilise « Soviet Union »,
> « West Germany »… et les matchs 2026 « Korea Republic » vs « South Korea ».
> Prévoir une normalisation des noms côté appelant. Les équipes inconnues du
> modèle reçoivent des valeurs par défaut neutres (prédiction calculable mais
> moins fiable).
