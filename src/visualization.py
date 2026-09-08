"""
Génère les visualisations principales du rapport :
- Figure 1 & 2 : taux de réussite par zone, latéralité, tirs cadrés ou non
- Figure 3 : taux de réussite selon le contexte de pression
- Analyses complémentaires : zone préférée par pied, efficacité du gardien,
  répartition des buts par pied, taux de réussite par poste.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import (
    load_data,
    get_area,
    add_score_state,
    add_shoot_order,
)


def zone_matrices(data, zone_layout):
    """Renvoie (taux de réussite %, nombre de tirs) en grilles 3x3 par zone."""
    goal_rate = data.groupby("Zone")["Goal"].mean().mul(100)
    counts = data.groupby("Zone").size()
    rate = np.array(
        [[goal_rate.get(z, np.nan) for z in row] for row in zone_layout]
    )
    n = np.array([[int(counts.get(z, 0)) for z in row] for row in zone_layout])
    return rate, n


def build_annotations(rate, n):
    """Annotation '12.5%\\n(n=8)' par case ; vide si aucun tir."""
    annot = np.empty(rate.shape, dtype=object)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            annot[i, j] = "" if n[i, j] == 0 else f"{rate[i, j]:.1f}%\n(n={n[i, j]})"
    return annot


def plot_zone_heatmaps(data_droitier, data_gaucher, suptitle, zone_layout):
    """Trace côte à côte les heatmaps droitiers / gauchers pour un jeu de données donné."""
    rate_d, n_d = zone_matrices(data_droitier, zone_layout)
    rate_g, n_g = zone_matrices(data_gaucher, zone_layout)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmin = np.nanmin([rate_d, rate_g])
    vmax = np.nanmax([rate_d, rate_g])

    for ax, rate, n, titre in zip(
        axes, [rate_d, rate_g], [n_d, n_g], ["Droitiers", "Gauchers"]
    ):
        sns.heatmap(
            rate,
            annot=build_annotations(rate, n),
            fmt="",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Taux de réussite (%)"},
            xticklabels=["Gauche", "Centre", "Droite"],
            yticklabels=["Haut", "Milieu", "Bas"],
            linewidths=1,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(f"{titre} (total {n.sum()} tirs)", fontweight="bold")

    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    df["Area"] = df["Zone"].apply(get_area)

    df_droitier = df[df["Foot"] == "R"].copy()
    df_gaucher = df[df["Foot"] == "L"].copy()

    df_hors_cadre = df.copy()
    df_hors_cadre.loc[df_hors_cadre["OnTarget"] == 0, "Area"] = 0

    # --- Zone préférée selon le pied (tirs cadrés uniquement) ---
    df_pref = df_hors_cadre[df_hors_cadre["Area"] != 0]
    df_pct = (
        df_pref.groupby("Foot")["Area"]
        .value_counts(normalize=True)
        .mul(100)
        .rename("Percentage")
        .reset_index()
    )
    sns.barplot(x="Foot", y="Percentage", hue="Area", data=df_pct)
    plt.xlabel("Pied")
    plt.ylabel("Pourcentage (%)")
    plt.title("Zone préférée selon le pied")
    plt.legend(title="Zone")
    plt.show()

    # --- Répartition des buts par zone (droitiers vs gauchers) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = ["Gauche", "Centre", "Droite"]

    goals_droitier = (
        df_droitier[df_droitier["Goal"] == 1]
        .groupby("Area")
        .size()
        .reindex([1, 2, 3], fill_value=0)
    )
    axes[0].pie(goals_droitier, labels=labels, autopct="%1.1f%%")
    axes[0].set_title("Droitiers", fontweight="bold")

    goals_gaucher = (
        df_gaucher[df_gaucher["Goal"] == 1]
        .groupby("Area")
        .size()
        .reindex([1, 2, 3], fill_value=0)
    )
    axes[1].pie(goals_gaucher, labels=labels, autopct="%1.1f%%")
    axes[1].set_title("Gauchers", fontweight="bold")
    plt.show()

    # --- Pourcentage d'arrêts du gardien selon la zone (tirs cadrés) ---
    df_cadres = df_hors_cadre[df_hors_cadre["OnTarget"] == 1]
    total_zone = df_cadres.groupby("Zone").size()
    saves_zone = df_cadres[df_cadres["Goal"] == 0].groupby("Zone").size()
    save_pct = (saves_zone / total_zone * 100).fillna(0).reindex(range(1, 10), fill_value=0)

    plt.figure(figsize=(10, 5))
    plt.bar([str(i) for i in range(1, 10)], save_pct)
    plt.xlabel("Zone")
    plt.ylabel("Pourcentage d'arrêts (%)")
    plt.title("Pourcentage d'arrêts du gardien selon la zone")
    for i, value in enumerate(save_pct):
        plt.text(i, value + 0.5, f"{value:.1f}%", ha="center")
    plt.show()

    # --- Équipe qui tire en premier vs en second (Figure 3, partie 1) ---
    df = add_shoot_order(df)
    success_rate = df.groupby("Shoot_Order")["Goal"].mean() * 100
    plt.figure(figsize=(6, 5))
    success_rate.plot(kind="bar")
    plt.ylabel("Taux de réussite (%)")
    plt.xlabel("")
    plt.title("Taux de réussite : 1ère équipe vs 2ème équipe")
    plt.xticks(rotation=0)
    plt.show()

    # --- Taux de réussite selon le contexte de score (Figure 3) ---
    df = add_score_state(df)
    pressure_success = df.groupby("Score_State")["Goal"].mean() * 100
    order = ["Leading", "Tied", "Trailing"]
    pressure_success = pressure_success.reindex(order)

    plt.figure(figsize=(7, 5))
    pressure_success.plot(kind="bar")
    plt.ylabel("Taux de réussite (%)")
    plt.xlabel("")
    plt.title("Taux de réussite selon le contexte au score")
    plt.xticks(rotation=0)
    plt.show()

    # --- Heatmaps 3x3 par zone : tous les tirs vs tirs cadrés (Figures 1 & 2) ---
    zone_layout = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    plot_zone_heatmaps(
        df_droitier, df_gaucher,
        "Taux de réussite par zone — TOUS les tirs", zone_layout,
    )

    df_droitier_cadre = df_droitier[df_droitier["OnTarget"] == 1]
    df_gaucher_cadre = df_gaucher[df_gaucher["OnTarget"] == 1]
    plot_zone_heatmaps(
        df_droitier_cadre, df_gaucher_cadre,
        "Taux de réussite par zone — TIRS CADRÉS uniquement", zone_layout,
    )

    # --- Le gardien part-il du bon côté ? ---
    keeper_to_area = {"L": 1, "C": 2, "R": 3}
    df_keeper = df_cadres.copy()
    df_keeper["Keeper_Area"] = df_keeper["Keeper"].map(keeper_to_area)
    df_keeper["Bon_Cote"] = np.where(
        df_keeper["Keeper_Area"] == df_keeper["Area"], "Bon côté", "Mauvais côté"
    )
    save_rate_keeper = (
        df_keeper.groupby("Bon_Cote")["Goal"]
        .apply(lambda g: (1 - g.mean()) * 100)
        .reindex(["Bon côté", "Mauvais côté"])
    )

    plt.figure(figsize=(6, 5))
    save_rate_keeper.plot(kind="bar", color=["#2a9d8f", "#e76f51"])
    plt.ylabel("Pourcentage d'arrêts (%)")
    plt.xlabel("")
    plt.title("Efficacité du gardien selon la direction du plongeon")
    plt.xticks(rotation=0)
    for i, value in enumerate(save_rate_keeper):
        plt.text(i, value + 0.5, f"{value:.1f}%", ha="center")
    plt.show()

    # --- Pression selon le numéro de tir et tirs décisifs ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    success_by_number = df.groupby("Penalty_Number")["Goal"].mean().mul(100)
    axes[0].plot(success_by_number.index, success_by_number.values, marker="o")
    axes[0].set_xlabel("Numéro du tir dans la séance")
    axes[0].set_ylabel("Taux de réussite (%)")
    axes[0].set_title("La pression monte-t-elle avec le numéro de tir ?")
    axes[0].grid(alpha=0.3)

    elim_success = df.groupby("Elimination")["Goal"].mean().mul(100)
    elim_success.index = ["Tir normal", "Tir décisif"]
    elim_success.plot(kind="bar", ax=axes[1], color=["#457b9d", "#e63946"])
    axes[1].set_ylabel("Taux de réussite (%)")
    axes[1].set_xlabel("")
    axes[1].set_title("Réussite : tir décisif vs tir normal")
    axes[1].tick_params(axis="x", rotation=0)
    for i, value in enumerate(elim_success):
        axes[1].text(i, value + 0.5, f"{value:.1f}%", ha="center")
    plt.tight_layout()
    plt.show()

    # --- Taux de réussite par poste ---
    poste_labels = {"A": "Attaquant", "M": "Milieu", "D": "Défenseur"}
    poste_stats = df.groupby("Poste")["Goal"].agg(["mean", "count"])
    poste_stats["mean"] = poste_stats["mean"] * 100
    poste_stats = poste_stats.rename(index=poste_labels)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(poste_stats.index, poste_stats["mean"], color="#6a4c93")
    plt.ylabel("Taux de réussite (%)")
    plt.xlabel("")
    plt.title("Taux de réussite selon le poste du tireur")
    for bar, (_, row) in zip(bars, poste_stats.iterrows()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['mean']:.1f}%\n(n={int(row['count'])})",
            ha="center",
        )
    plt.show()


if __name__ == "__main__":
    main()
