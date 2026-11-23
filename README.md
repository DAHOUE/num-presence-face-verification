# Vérification faciale — Num Presence (extrait de code)

## Contexte

Ce dépôt contient un **extrait représentatif de la logique de vérification faciale implémentée dans Num Presence**, une application mobile et plateforme SaaS de digitalisation et de gestion des présences lors d'événements, de formations, de conférences et au sein d'organisations.

Le code source complet de Num Presence est conservé dans un **dépôt privé** et n'est pas accessible publiquement. Ce dépôt public présente donc un extrait isolé, nettoyé et autonome de la même logique de vérification faciale, à des fins de candidature et de démonstration technique. Le code spécifique à l'application, l'infrastructure privée, les identifiants et les données personnelles ou biométriques réelles ne sont pas inclus.

## Ce que cet extrait démontre

- Décodage d'images téléversées avec OpenCV
- Détection de visage et extraction d'embeddings avec le pipeline préentraîné InsightFace
- Normalisation L2 des embeddings avec NumPy
- Calcul de la distance euclidienne entre deux embeddings
- Décision de vérification basée sur un seuil
- Exemple de recherche du plus proche embedding avec SQLAlchemy et pgvector

Cet extrait illustre l'intégration de composants de vision par ordinateur préentraînés dans une application logicielle. Il ne s'agit ni d'un nouveau modèle de reconnaissance faciale entraîné dans ce projet, ni d'un produit biométrique autonome complet.


## Pipeline de traitement

1. **Décodage de l'image :** les octets de l'image sont convertis en matrice d'image BGR.
2. **Détection faciale :** le pipeline préentraîné InsightFace détecte les visages.
3. **Validation du nombre de visages :** l'exemple rejette les images sans visage détecté ou contenant plusieurs visages.
4. **Extraction de l'embedding :** le modèle génère une représentation numérique du visage détecté.
5. **Normalisation :** l'embedding est normalisé selon la norme L2.
6. **Comparaison :** la distance euclidienne est calculée entre l'embedding de la requête et celui de référence.
7. **Décision :** une correspondance est acceptée si la distance est inférieure ou égale au seuil configuré.
8. **Recherche en base de données (optionnelle) :** le code illustre une requête de recherche du candidat le plus proche avec SQLAlchemy/pgvector.

## Technologies

- Python
- InsightFace
- OpenCV
- NumPy
- SQLAlchemy
- PostgreSQL avec pgvector pour l'exemple de comparaison vectorielle en base

## Installation

Installez les dépendances de l'exemple :

```bash
pip install -r requirements.txt
```

InsightFace peut télécharger des modèles préentraînés lors de sa première utilisation. Les contraintes d'installation et de runtime dépendent de l'environnement et des versions des bibliothèques.

## À propos du seuil

La valeur de seuil (`0.7`) est incluse à titre de configuration illustrative, cohérente avec la logique décrite pour cet extrait. **Elle ne constitue pas un seuil universel ni une revendication de précision.** Un seuil doit être calibré et évalué sur des données représentatives des conditions de capture et du modèle utilisés. Il convient de rapporter des métriques de fausses acceptations et de faux rejets, si une évaluation est réalisée.

## Limites

- Cet extrait ne contient pas l'application Num Presence complète ni son code source privé.
- Aucun jeu de données, benchmark, résultat de précision ou validation formelle n'est fourni.
- La détection de vivacité (*liveness detection*) et la protection contre les attaques de présentation ne sont pas implémentées ici.
- Les résultats peuvent être affectés par la pose, l'éclairage, les occultations, la qualité de caméra et d'autres conditions.
- Les embeddings sont des données biométriques sensibles : aucune image réelle ni aucun embedding ne doit être téléversé dans ce dépôt.

## Auteur

Isaac Dahoue — Génie logiciel, IA appliquée et vision par ordinateur
