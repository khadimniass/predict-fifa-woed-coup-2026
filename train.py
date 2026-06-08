"""Entraîne le modèle de prédiction (XGBoost, 3 classes) et sauvegarde les artefacts.

Pré-requis : avoir lancé `python fetch_data.py` (remplit data/).
Usage     : python train.py
Produit   : model.pkl (classifieur) + stats.pkl (état + altitudes + features).
"""
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import MODEL_PATH, STATS_PATH
from features import FEATURES, build_dataset


def main() -> None:
    print("Construction du dataset…")
    X, y, stats, elevations = build_dataset()
    if not X:
        raise SystemExit(
            "Aucune donnée. Lance d'abord `python fetch_data.py`."
        )

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)
    print(f"  {len(X_arr)} matchs · {len(FEATURES)} features")
    print(f"  répartition labels (1/N/2) : {np.bincount(y_arr)}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )
    print("Entraînement XGBoost…")
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    print(f"\nAccuracy test : {accuracy_score(y_te, preds):.3f}")
    print(
        classification_report(
            y_te, preds, target_names=["team1_win", "draw", "team2_win"], zero_division=0
        )
    )
    print("Importance des features :")
    for name, imp in sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name:16s} {imp:.3f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(
        {"stats": stats, "elevations": elevations, "features": FEATURES},
        STATS_PATH,
    )
    print(f"\nModèle sauvegardé : {MODEL_PATH}")
    print(f"Stats sauvegardées : {STATS_PATH}")


if __name__ == "__main__":
    main()
