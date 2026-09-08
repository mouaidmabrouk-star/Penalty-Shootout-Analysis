"""
Chargement des données et fonctions de feature engineering communes,
utilisées à la fois par les scripts de visualisation et par le modèle.
"""

import pandas as pd

CSV_PATH = "data/penaltyShootouts.csv"


def load_data(path=CSV_PATH):
    """Charge le CSV des tirs au but."""
    return pd.read_csv(path)


def get_area(zone):
    """
    Regroupe les 9 zones du but en 3 colonnes (gauche/centre/droite).
    Numérotation du CSV :
        1 2 3   (haut)
        4 5 6   (milieu)
        7 8 9   (bas)
    """
    if zone in (1, 4, 7):
        return 1  # Gauche
    elif zone in (2, 5, 8):
        return 2  # Centre
    elif zone in (3, 6, 9):
        return 3  # Droite
    return 0  # Hors cadre / zone inconnue


def map_hauteur(zone):
    """Regroupe les 9 zones en 3 hauteurs (Bas / Milieu / Haut)."""
    zone = int(zone)
    if zone in (7, 8, 9):
        return "Bas"
    elif zone in (4, 5, 6):
        return "Milieu"
    elif zone in (1, 2, 3):
        return "Haut"
    return "Inconnu"


def add_mort_subite(df):
    """
    Variable binaire de pression : à partir du 11e tir de la séance
    (= 6e tireur d'une équipe), la séance entre en mort subite.
    """
    df = df.copy()
    df["Mort_Subite"] = (df["Penalty_Number"] >= 11).astype(int)
    return df


def add_score_state(df):
    """
    Calcule, pour chaque tir, si l'équipe qui tire est en train de
    mener, d'être menée, ou à égalité au moment du tir (Score_State),
    en parcourant chaque séance (Game_id) dans l'ordre chronologique.
    """
    df = df.sort_values(["Game_id", "Penalty_Number"]).copy()
    pressure_states = []

    for game_id, game in df.groupby("Game_id"):
        teams = game["Team"].unique()
        score = {team: 0 for team in teams}

        for _, row in game.iterrows():
            current_team = row["Team"]
            other_team = [t for t in teams if t != current_team][0]

            if score[current_team] > score[other_team]:
                pressure_states.append("Leading")
            elif score[current_team] < score[other_team]:
                pressure_states.append("Trailing")
            else:
                pressure_states.append("Tied")

            if row["Goal"] == 1:
                score[current_team] += 1

    df["Score_State"] = pressure_states
    return df


def add_shoot_order(df):
    """Ajoute une colonne indiquant si l'équipe tire en 1er ou en 2nd de la séance."""
    df = df.copy()
    first_team = (
        df.sort_values(["Game_id", "Penalty_Number"])
        .groupby("Game_id")
        .first()["Team"]
    )
    df["First_Shooter"] = df["Game_id"].map(first_team)
    df["Shoot_Order"] = df.apply(
        lambda row: "First Team"
        if row["Team"] == row["First_Shooter"]
        else "Second Team",
        axis=1,
    )
    return df
