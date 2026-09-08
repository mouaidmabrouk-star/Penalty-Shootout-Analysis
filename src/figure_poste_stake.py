"""
Figure 4 du rapport : taux de réussite par rang du tir, enjeu (victoire /
survie / normal) et poste du tireur. La taille des points est
proportionnelle au nombre d'observations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_preprocessing import load_data

MIN_N = 5


def classify_stake(group):
    """Pour chaque tir d'une séance, détermine l'enjeu : Victoire / Survie / Normal."""
    group = group.sort_values("Penalty_Number").copy()
    for idx, row in group.iterrows():
        pn = row["Penalty_Number"]
        is_teamA = pn % 2 == 1
        prev = group[group["Penalty_Number"] < pn]
        teamA_goals = prev[prev["Penalty_Number"] % 2 == 1]["Goal"].sum()
        teamB_goals = prev[prev["Penalty_Number"] % 2 == 0]["Goal"].sum()
        teamA_taken = len(prev[prev["Penalty_Number"] % 2 == 1])
        teamB_taken = len(prev[prev["Penalty_Number"] % 2 == 0])

        if is_teamA:
            my_goals, opp_goals = teamA_goals, teamB_goals
            my_taken, opp_taken = teamA_taken, teamB_taken
        else:
            my_goals, opp_goals = teamB_goals, teamA_goals
            my_taken, opp_taken = teamB_taken, teamA_taken

        if my_taken < 5:
            my_remaining_after = 4 - my_taken
            opp_remaining = 5 - opp_taken
        else:
            my_remaining_after = 0
            opp_remaining = 0 if opp_taken > my_taken else 1

        win_if_score = (my_goals + 1) > (opp_goals + opp_remaining)
        lose_if_miss = opp_goals > (my_goals + my_remaining_after)

        group.loc[idx, "stake"] = (
            "Victoire" if win_if_score else ("Survie" if lose_if_miss else "Normal")
        )
        group.loc[idx, "tour"] = (teamA_taken if is_teamA else teamB_taken) + 1
    return group


def main():
    df = load_data()
    df = df.sort_values(["Game_id", "Penalty_Number"])

    result = df.groupby("Game_id", group_keys=False).apply(classify_stake)
    result["tour"] = result["tour"].astype(int)
    result = result[result["Poste"].isin(["A", "D", "M"]) & (result["tour"] <= 5)]

    stats = result.groupby(["tour", "stake", "Poste"]).agg(
        goals=("Goal", "sum"), n=("Goal", "count")
    ).reset_index()
    stats["rate"] = stats["goals"] / stats["n"] * 100

    stake_colors = {"Victoire": "#2ecc71", "Survie": "#e74c3c", "Normal": "#95a5a6"}
    stake_labels = {
        "Victoire": "Victoire (gagner)",
        "Survie": "Survie (élimination)",
        "Normal": "Normal (encore en jeu)",
    }
    poste_markers = {"A": "o", "D": "s", "M": "^"}
    poste_labels = {"A": "Attaquant", "D": "Défenseur", "M": "Milieu"}
    stake_order = ["Normal", "Victoire", "Survie"]
    poste_order = ["A", "M", "D"]

    positions = [(s, p) for s in stake_order for p in poste_order]
    x_offsets = {
        (s, p): (i - len(positions) / 2) * (0.25 / len(positions))
        for i, (s, p) in enumerate(positions)
    }

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="white", zorder=0)
    ax.set_axisbelow(True)

    for _, row in stats.iterrows():
        if row["n"] < MIN_N:
            continue
        x = row["tour"] + x_offsets.get((row["stake"], row["Poste"]), 0)
        ax.scatter(
            x, row["rate"],
            s=80 + row["n"] * 4,
            color=stake_colors[row["stake"]],
            marker=poste_markers[row["Poste"]],
            edgecolors="white", linewidths=0.8, alpha=0.88, zorder=3,
        )
        ax.annotate(
            f"n={int(row['n'])}", xy=(x, row["rate"]),
            xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=7, color=stake_colors[row["stake"]], alpha=0.85,
        )

    for stake in stake_order:
        for poste in poste_order:
            sub = stats[
                (stats["stake"] == stake)
                & (stats["Poste"] == poste)
                & (stats["n"] >= MIN_N)
            ].sort_values("tour")
            if len(sub) < 2:
                continue
            xs = [r["tour"] + x_offsets.get((stake, poste), 0) for _, r in sub.iterrows()]
            ax.plot(
                xs, sub["rate"].values, color=stake_colors[stake],
                alpha=0.25, linewidth=1.2, linestyle="--", zorder=2,
            )

    color_patches = [mpatches.Patch(color=stake_colors[s], label=stake_labels[s]) for s in stake_order]
    marker_handles = [
        plt.scatter([], [], marker=poste_markers[p], color="#555555", s=80, label=poste_labels[p])
        for p in poste_order
    ]
    size_handles = [
        plt.scatter([], [], marker="o", color="#aaaaaa", s=80 + n * 4, label=f"n={n}")
        for n in [5, 15, 30]
    ]

    leg1 = ax.legend(handles=color_patches, title="Enjeu du tir", loc="upper right", framealpha=0.9, fontsize=9, title_fontsize=9)
    leg2 = ax.legend(handles=marker_handles, title="Poste", loc="upper left", framealpha=0.9, fontsize=9, title_fontsize=9)
    ax.legend(handles=size_handles, title="Taille = n", loc="lower right", framealpha=0.9, fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    overall = result["Goal"].mean() * 100
    ax.axhline(overall, color="#333333", linestyle=":", linewidth=1.2, alpha=0.5)
    ax.text(5.55, overall + 1, f"Moy.\n{overall:.1f}%", fontsize=8, color="#333333", va="bottom", ha="right")

    ax.set_xlabel("Rang du penalty (1 = 1er penalty de chaque équipe)", fontsize=11)
    ax.set_ylabel("Taux de réussite (%)", fontsize=11)
    ax.set_title(
        "Taux de réussite par rang, enjeu et poste\n(taille des points proportionnelle à n)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(20, 115)

    plt.tight_layout()
    plt.savefig("../figures/fig4_poste_rang_enjeu.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
