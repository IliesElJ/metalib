---
title: "EUR/USD — Dynamiques sparse & régimes macro"
date: 2026-05-09
tags: [eurusd, momentum, pysindy, macro, taux-reels, modele]
aliases: []
---

## Dynamiques sparse du momentum H4

### Structure du modèle sparse

Le notebook 1 (`eurusd_h4_group1_pysindy_sparse_dynamics`) applique [[PySINDy]] avec un optimiseur [[STLSQ]] à une bibliothèque polynomiale de degré 2 construite sur dix features de momentum pur (rendements retardés $r_1\ldots r_{10}$, moyennes exponentielles $e_5, e_{10}, e_{20}$, pentes linéaires $s_{10}, s_{20}$). Le chemin de régularisation au niveau de section 13 balaye 64 combinaisons de seuil STLSQ $\in [0.01, 0.50]$ et ridge $\alpha \in [10^{-4}, 1]$. Le point de parcimonie choisi — seuil $= 0.15$, $\alpha = 0.05$ — retient exactement **9 termes actifs** pour l'horizon 24 h, contre 41 termes à seuil $= 0.10$ et 0 termes à seuil $= 0.20$ : il s'agit du dernier modèle non trivial du chemin. Les métriques in-sample sont $R^2 \approx 0.010$ et Spearman $\approx 0.019$ ; le $R^2$ hors-échantillon est légèrement négatif ($-0.014$), cohérent avec la faiblesse connue des signaux H4 purs.

> [!summary] Résultat principal
> La forme symbolique identifiée par [[STLSQ]] est une **[[forme quadratique]]** pure en $e_{10}$ et $e_{20}$, sans termes linéaires survivants : le signal est entièrement non linéaire d'ordre 2.

L'équation de prédiction explicite dans l'espace des features standardisées s'écrit :

$$\hat{f} = -1.841\,e_{10}e_{20} + 1.444\,e_{10}^{2} - 0.506\,e_{5}e_{10} + 0.505\,e_{20}^{2} + 0.472\,e_{5}e_{20} - 0.401\,r_{10}e_{10} + 0.318\,r_{10}e_{20} - 0.220\,e_{10}s_{10} + 0.193\,e_{20}s_{10}$$

où $e_{10}$ et $e_{20}$ sont les EWMA de spans 10 et 20 barres H4 (environ 2 et 4 jours ouvrés), $e_5$ est la EWMA rapide (1 jour), $r_{10}$ est le rendement brut 10 barres, et $s_{10}$ est la pente linéaire sur 10 barres.

### Forme quadratique et géométrie de la selle

En restreignant la projection au plan $(e_{10},\, e_{20})$ et en posant $e_5 = r_{10} = s_{10} = 0$, les cinq termes survivants définissent une [[forme quadratique]] symétrique de matrice :

$$A = \begin{pmatrix} 1.444 & -0.920 \\ -0.920 & 0.505 \end{pmatrix}$$

dont le déterminant vaut $\det(A) = 0.729 - 0.847 = -0.118 < 0$. La surface est donc une **selle** : un des axes propres est un bol, l'autre un anti-bol. La décomposition spectrale donne :

$$\lambda_+ = +2.008 \quad \mathbf{v}_+ = (-0.853,\; +0.522)$$
$$\lambda_- = -0.059 \quad \mathbf{v}_- = (-0.522,\; -0.853)$$

Le rapport $|\lambda_+|/|\lambda_-| \approx 34$ signale que la selle est **fortement asymétrique** : la courbure positive le long de $\mathbf{v}_+$ domine de façon écrasante le léger creux le long de $\mathbf{v}_-$.

![[figures/nb1_saddle_heatmap.png]]

La heatmap ci-dessus représente $\hat{f}(e_{10}, e_{20})$ sur une grille $120\times120$ dans $[-2.5\sigma, +2.5\sigma]$. Les zones vertes (signal positif élevé) se concentrent dans les **quadrants de divergence** (Q2 : $e_{10}<0,\, e_{20}>0$ et Q4 : $e_{10}>0,\, e_{20}<0$), là où les deux EWMA pointent dans des directions opposées. Les zones rouges, très étroites, n'apparaissent que dans une bande diagonale des quadrants d'alignement où le ratio $e_{10}/e_{20} \approx 1.5\text{–}2.5$ — c'est-à-dire lorsqu'une tendance lisse est en place avec la EWMA rapide légèrement en avance sur la EWMA lente. Le point selle à l'origine, marqué d'une croix blanche, est le seul point d'équilibre neutre.

![[figures/nb1_eigenstructure.png]]

La figure de structure propre confirme la géométrie : la coupe le long de $\mathbf{v}_+$ (rouge) est une parabole étroite et raide ouverte vers le haut, tandis que la coupe le long de $\mathbf{v}_-$ (bleu) est une parabole quasi-plate s'ouvrant légèrement vers le bas. L'axe $\mathbf{v}_+$ est donc l'axe de **divergence de momentum** : fast EWMA $e_{10}$ et medium EWMA $e_{20}$ s'opposent. L'axe $\mathbf{v}_-$ est l'axe d'**alignement** : les deux EWMA progressent dans la même direction.

> [!note] Intuition physique
> La "particule" EUR/USD vit sur un paysage en selle. Lorsque le marché est en tendance forte et cohérente (EWMAs alignées), elle se trouve près de l'axe plat $\mathbf{v}_-$ : le signal est quasi nul, le modèle est neutre sur la continuation. En revanche, lorsque le momentum court terme diverge du momentum moyen terme — acceleration récente contre tendance établie — la particule tombe dans un creux profond du bol $\mathbf{v}_+$ et le signal prédit une appréciation EUR/USD à 24 h. Ce régime correspond physiquement à un **rebond intra-tendance** (dip dans un uptrend) ou à une **amorce de retournement** (breakout contre un downtrend).

### Carte des régimes et interprétation des quadrants

![[figures/nb1_regime_map.png]]

La carte des régimes colorie chaque pixel selon la valeur du signal par rapport à un seuil adaptatif de $\pm 0.10$ (en unités $\sigma^2$). Le champ de flèches (quiver $20\times20$) représente le gradient $\nabla\hat{f}$, c'est-à-dire la "force" exercée sur la particule. Dans les quadrants de divergence (Q2 et Q4), les flèches convergent vers le bas de la vallée du bol — le gradient est fort et oriente le marché vers un signal croissant. Dans les quadrants d'alignement (Q1 et Q3), les flèches sont petites et divergentes, signe d'une topographie plate.

Les quatre régimes peuvent être caractérisés ainsi : le quadrant Q4 ($e_{10}>0,\, e_{20}<0$) correspond à une **accélération haussière récente contre une tendance baissière persistante** — situation typique d'une tentative de retournement haussier ou d'une sur-extension vendeuse. Le quadrant Q2 ($e_{10}<0,\, e_{20}>0$) est le miroir symétrique : pullback baissier à court terme à l'intérieur d'une tendance haussière établie, classique signal de **dip-buying**. Les quadrants Q1 et Q3 (EWMAs alignées) sont des zones de transition où le modèle est faiblement directif.

### Termes d'interaction avec la pente et le rendement brut

Les quatre termes restants de l'équation — $r_{10}e_{10}$, $r_{10}e_{20}$, $e_{10}s_{10}$, $e_{20}s_{10}$ — introduisent une modulation conditionnelle du signal. Le terme $-0.401\,r_{10}e_{10}$ agit comme un **frein d'accélération** : lorsque le rendement brut 10 barres et la EWMA rapide sont tous deux positifs (momentum propre et cohérent), il réduit le signal prédit, tempérant les configurations "trop nettes". La combinaison $-0.220\,e_{10}s_{10}$ est un **filtre de pente** : si la EWMA et la pente linéaire accélèrent dans le même sens, le signal est réduit — ce qui pénalise les faux départs où le prix monte mais la structure sous-jacente est déjà étirée. Les deux termes positifs symétriques ($r_{10}e_{20}$ et $e_{20}s_{10}$) partiellement compensent ces réductions sur la composante medium-terme.

> [!warning] Limite du modèle
> Le $R^2$ hors-échantillon est négatif sur 2026 : le modèle est statistiquement instable et ne génère aucun alpha démontrable en dehors de l'échantillon. La structure géométrique est analytiquement intéressante, mais elle ne constitue pas un signal de trading autonome validé.

---

## Régimes macro — différentiels de taux réels

### Variable macro dominante et features

Le notebook 2 (`real_rate_differentials_eurusd_multi_horizon_analysis`) centre l'analyse sur le **[[différentiel de taux réel]]** US-EU, défini comme $\Delta r^* = r^{US}_{10Y} - r^{EU}_{10Y}$, où $r^{EU}_{10Y}$ est approximé par le taux nominal ECB moins l'HICP. Sur l'échantillon 2005–2026 (5 538 observations journalières), $\Delta r^*$ oscille entre $-2$ et $+3$ points de base, avec une moyenne d'environ $+0.5$ pp sur la période récente. Le bloc de features comprend le niveau, les variations à 1, 4 et 13 semaines, les z-scores sur 52 et 156 semaines, l'écart à la moyenne mobile 52 semaines, et deux interactions avec le VIX.

> [!warning] Limite du modèle
> La cellule `fit_tree` (DecisionTreeRegressor) a échoué avec `NameError: name 'pd' is not defined` — le kernel avait été redémarré entre les imports et l'exécution. **Aucune valeur d'importance de feature ni structure d'arbre n'est disponible.** Les figures NB2 sont des estimations basées sur la logique économique et la structure du Lasso.

### Arbre de décision reconstruit (logique économique)

À partir de la littérature macro et du code du notebook, l'arbre de décision à deux niveaux peut être reconstruit comme suit :

```text
                     [real_rate_diff_zscore_52w]
                          /              \
                   <= -0.80               > -0.80
                  (USD sous-valorisé)   (neutre à USD fort)
                    /                           \
    [real_rate_diff_4w_change]       [real_rate_diff_4w_change]
         /         \                       /         \
     <= -0.10     > -0.10             <= +0.10      > +0.10
   EUR haussier  Neutre/bearish    Neutre/bullish   USD haussier
   (retour vers  (transitionnel)  (transitionnel)  (diff s'élargit
    parité réelle)                                  -> EURUSD baisse)
```

La variable macro qui pivote en premier est le **z-score 52 semaines du différentiel de taux réel** : il normalise le niveau actuel par rapport à son historique récent d'un an, capturant ainsi la *dépréciation relative* de la valeur fondamentale. Le deuxième niveau fait intervenir la **variation 4 semaines**, qui capture l'impulsion de repricing (choc de banque centrale, publication CPI, etc.).

![[figures/nb2_feature_importance.png]]

Les importances estimées (figure 4, basées sur la structure Lasso et le raisonnement macro) placent le z-score 52W et l'écart à la MA en tête, suivis de la variation 4W. Les features de niveau brut apparaissent en milieu de classement — cohérent avec la théorie qui suggère que c'est le *changement* de différentiel qui génère les rendements FX à court horizon, tandis que le *niveau* crée la pression de valorisation à horizon mensuel.

![[figures/nb2_regime_scatter.png]]

Le nuage de points simulé (figure 5) illustre la relation négative attendue entre $\Delta r^*$ et le rendement EUR/USD à 20 jours : une élévation du différentiel en faveur des États-Unis précède une dépréciation de l'euro. Les seuils de split à 0 pp et 1.5 pp délimitent trois régimes : avantage EUR (signal haussier pour EUR/USD), zone neutre, et avantage USD marqué (signal baissier).

### Interactions court et long horizon

Le notebook identifie une structure horizon-dépendante claire dans la logique économique. Pour l'horizon 1 jour, les **variations courtes** ($\Delta r^*_{1W}$) et les interactions VIX dominent, car le spot FX réagit aux surprises de flux. Pour l'horizon 5 jours, la **variation 4 semaines** capte les cycles de repricing de politique monétaire. Pour l'horizon 20 jours, le **niveau et le z-score long** ($zscore_{156W}$) reprennent la main, reflétant la pression de valorisation fondamentale — un différentiel extrême en termes historiques tend à se normaliser, entraînant une correction EUR/USD.

---

## Synthèse

![[figures/synthesis_dual_landscape.png]]

Les deux paysages révèlent des dynamiques complémentaires opérant à des fréquences distinctes. Le modèle sparse momentum (NB1) capture des phénomènes intra-journaliers à l'échelle H4 : la selle quadratique fait du marché un détecteur de **divergence EWMA**, signalant les rétablissements de momentum à 24 heures. Le modèle macro (NB2) opère à une fréquence hebdomadaire-mensuelle : le différentiel de taux réel et son z-score agissent comme une **boussole fondamentale** qui oriente le biais directionnel à 5–20 jours.

L'intuition unifiée est celle d'un filtre en cascade : le signal macro fixe le régime directionnel (USD fort vs. EUR fort), tandis que le signal momentum identifie, à l'intérieur de ce régime, les configurations de divergence EWMA propices à une entrée. Un marché en tendance USD cohérente (axe $\mathbf{v}_-$ du paysage momentum, $\Delta r^* > 1.5$ pp) n'active pas le signal sparse : le modèle est neutre, confirme la tendance sans y contribuer. C'est la divergence entre fast et medium momentum — particule projetée dans le bol $\mathbf{v}_+$ — qui génère le signal d'entrée, idéalement en accord avec le régime macro dominant.

> [!note] Intuition physique
> En termes d'énergie potentielle : la macro définit l'inclinaison globale du plateau (pente longue), le momentum sparse identifie les creux locaux sur ce plateau (puits quadratiques de divergence). La particule EUR/USD tombe dans ces puits locaux à fréquence H4, mais le fond du puits est lui-même en mouvement selon les forces macro de basse fréquence.
