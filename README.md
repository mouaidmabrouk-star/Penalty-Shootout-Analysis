# Analyse Statistique et Modélisation des Tirs au But

Projet de data science (module Traitement de Données, MAM3  Polytech Nice Sophia, Juin 2026) : peut-on prédire l'issue d'un penalty à partir d'informations connues *avant* la frappe (poste du tireur, latéralité, hauteur visée, contexte de pression) ?

> Projet réalisé en groupe de 3 : **Titouan Bembekoff**, **Rémi Debavelaere** et **Mouaid Mabrouk**. Ce repo est ma version réorganisée du travail de groupe, à des fins de portfolio. Le code original (tel qu'écrit collectivement pendant le projet) est disponible sur [RemDebav/Penalty-analysis](https://github.com/RemDebav/Penalty-analysis). Vous trouverez également notre [rapport complet](docs/rapport_projet.pdf) et notre [présentation](docs/presentation.pptx) dans le dossier `docs/`.

## Contexte

La séance de penaltys est souvent qualifiée de « loterie ». Ce projet cherche à déconstruire cette idée : le tir au but est un geste technique exécuté sous forte contrainte psychologique, qui répond à des probabilités modélisables.

L'objectif : transformer un jeu de données historique en outil d'aide à la décision tactique, en répondant à 3 questions :
- Comment optimiser l'ordre de passage des tireurs face à la pression de la « mort subite » ?
- Quelle est l'influence du poste et de la latéralité du tireur ?
- Quelles zones de tir sont les plus rentables ?

## Données

- [Kaggle : World Cup Penalty Shootouts](https://www.kaggle.com/) (jeu de données initial)
- Enrichi avec des données [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- 603 tirs au but, nettoyés et enrichis (poste du tireur, zone visée sur 9 cases, latéralité, enjeu du tir)

## Méthodologie

**1. Data visualisation**  exploration du taux de réussite selon la zone, la latéralité, le contexte de pression (voir `figures/`).

**2. Feature engineering**  deux variables métier construites à la main :
 `Mort_Subite` : variable binaire, activée à partir du 11ème tir de la séance (6ème tireur), point de bascule psychologique non linéaire.
 `Hauteur` : regroupement des 9 zones brutes en 3 catégories (Bas / Milieu / Haut), plus robuste statistiquement que le découpage fin.

**3. Modélisation par régression logistique**, avec 3 choix méthodologiques déterminants :
 **Prévention des fuites de données** : exclusion de la direction du plongeon du gardien et du fait que le tir soit cadré ces informations ne sont connues qu'au moment de la frappe, pas avant. Étude restreinte aux tirs cadrés uniquement.
 **Gestion du déséquilibre des classes** (~77% de buts) via `class_weight='balanced'`.
 **Évaluation par AUC** (aire sous la courbe ROC) plutôt que par précision la précision est trompeuse sur des classes déséquilibrées (un modèle naïf prédisant systématiquement "but" atteindrait déjà 77%).

## Résultats

- Meilleur modèle (avec variables handcrafted) : AUC ≈ 0,635, contre 0,621 pour le modèle de base.
- Le modèle identifie correctement ~70% des arrêts (au prix de fausses alertes sur les buts), là où un modèle naïf n'en détecterait aucun.
- Le gain apporté par les variables handcrafted est présent mais modeste (+0,014 d'AUC) : une fois écartées les variables liées à la frappe elle-même, l'issue d'un penalty reste en grande partie imprévisible  ce qui fait la tension de l'exercice.
- Facteur le plus significatif : viser haut augmente nettement les chances de marquer.

> **Note sur la reproductibilité** : le script `model.py` de ce repo a été reconstruit pour suivre fidèlement la méthodologie décrite ci-dessus (AUC, `class_weight='balanced'`, exclusion des fuites de données). En exécution, il donne des résultats proches mais pas strictement identiques à ceux du rapport (AUC ≈ 0,62 selon le tirage de la validation croisée)  la variation vient du découpage aléatoire des plis et de détails de nettoyage propres à l'exécution originale.

## Structure du repo

```
├── src/
│   ├── data_preprocessing.py    # chargement + feature engineering commun
│   ├── visualization.py         # figures 1 à 3 + analyses complémentaires
│   ├── figure_poste_stake.py    # figure 4 (rang x enjeu x poste)
│   └── model.py                 # régression logistique, AUC, class_weight balanced
├── data/
│   └── penaltyShootouts.csv
├── figures/                     # graphiques générés
├── docs/
│   ├── rapport_projet.pdf
│   └── presentation.pptx
├── requirements.txt
└── .gitignore
```

## Installation et exécution

```bash
pip install -r requirements.txt
cd src
python visualization.py         # génère les figures principales
python figure_poste_stake.py    # génère la figure 4
python model.py                 # entraîne et évalue le modèle
```

## Limites

Le modèle ne capture pas l'intégralité du contexte d'un match : fatigue physique après 120 minutes, historique psychologique entre un tireur et un gardien, scénario du match. Des données de tracking vidéo (vitesse de course d'élan, direction du regard) constitueraient une piste d'amélioration intéressante.
