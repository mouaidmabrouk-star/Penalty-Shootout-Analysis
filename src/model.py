"""
Modélisation par régression logistique, fidèle à la méthodologie décrite
dans le rapport (section 5) :

- Restriction aux tirs CADRÉS uniquement (seul périmètre où la question
  "le tireur va-t-il marquer ?" a un sens statistique).
- Exclusion des variables de fuite de données : la direction du plongeon
  du gardien (Keeper) et le fait que le tir soit cadré (OnTarget) ne sont
  connues qu'au moment de la frappe, pas avant.
- class_weight='balanced' pour compenser le déséquilibre des classes
  (~77% de buts parmi les tirs cadrés).
- Évaluation par AUC (aire sous la courbe ROC) plutôt que par précision,
  via une validation croisée à 5 plis (5-fold CV), car la précision est
  trompeuse sur des classes déséquilibrées.
- Comparaison du modèle avec vs. sans les variables handcrafted
  (Mort_Subite, Hauteur) pour mesurer leur apport réel.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from data_preprocessing import load_data, map_hauteur, add_mort_subite

RANDOM_STATE = 42
N_SPLITS = 5


def prepare_features(df):
    """
    Restreint aux tirs cadrés, retire les variables de fuite de données,
    et construit les variables handcrafted (Mort_Subite, Hauteur) plus
    les dummies Poste / Foot.
    """
    df = df.dropna(subset=["Goal", "Poste", "Foot", "Penalty_Number", "Zone", "OnTarget"])

    # Seul périmètre où la question a un sens statistique
    df = df[df["OnTarget"] == 1].copy()

    # Variables de fuite de données à exclure : Keeper (direction du plongeon)
    # et OnTarget (quasi-équivalent à la réponse elle-même)
    df = df.drop(columns=["Keeper", "OnTarget"], errors="ignore")

    df = add_mort_subite(df)
    df["Hauteur"] = df["Zone"].apply(map_hauteur)

    df_encoded = pd.get_dummies(df, columns=["Poste", "Foot", "Hauteur"], drop_first=True)

    hauteur_cols = [c for c in df_encoded.columns if c.startswith("Hauteur_")]
    base_cols = [c for c in df_encoded.columns if c.startswith(("Poste_", "Foot_"))]

    X_full = df_encoded[["Mort_Subite"] + base_cols + hauteur_cols]
    X_baseline = df_encoded[base_cols]  # sans les variables handcrafted
    y = df_encoded["Goal"]

    return X_full, X_baseline, y


def evaluate_auc(X, y, label):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")
    print(f"{label} — AUC moyenne (5-fold CV) : {scores.mean():.3f}")
    print(f"{label} — détail des 5 plis : {np.round(scores, 3)}\n")
    return scores.mean()


def main():
    df = load_data()
    X_full, X_baseline, y = prepare_features(df)

    print("--- ÉVALUATION DU MODÈLE (AUC, class_weight='balanced') ---\n")
    auc_baseline = evaluate_auc(X_baseline, y, "Modèle de base (sans variables handcrafted)")
    auc_full = evaluate_auc(X_full, y, "Modèle complet (avec Mort_Subite + Hauteur)")

    print(f"Gain apporté par les variables handcrafted : +{auc_full - auc_baseline:.3f} d'AUC\n")

    # --- Matrice de confusion (validation croisée) sur le modèle complet ---
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X_full, y, cv=kf)

    cm = confusion_matrix(y, y_pred)
    print("--- MATRICE DE CONFUSION (validation croisée, modèle complet) ---")
    print(f"                 Prédit Arrêt (0)   Prédit But (1)")
    print(f"Réel Arrêt (0)   {cm[0][0]:>14d}   {cm[0][1]:>14d}")
    print(f"Réel But (1)     {cm[1][0]:>14d}   {cm[1][1]:>14d}\n")

    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"Précision globale (volontairement basse, contrepartie du rééquilibrage) : {accuracy:.3f}")

    # --- Interprétabilité : coefficients du modèle final entraîné sur tout le jeu ---
    model.fit(X_full, y)
    importance = pd.DataFrame(
        {"Facteur": X_full.columns, "Impact (Beta)": model.coef_[0]}
    ).sort_values(by="Impact (Beta)", ascending=False)

    print("\n--- INTERPRÉTATION DES FACTEURS ---")
    print("Un coefficient positif augmente les chances de marquer, un négatif les diminue.")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
