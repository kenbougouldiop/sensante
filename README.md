# SenSante
Assistant de pré-diagnostic médical pour le Sénégal.

## Description
SenSante utilise le Machine Learning pour aider au pré-diagnostic des maladies courantes : paludisme, grippe et typhoïde.

## Structure du projet
- `data/` : jeu de données patients au format CSV
- `models/` : modèles et encodeurs sérialisés
- `api/` : application FastAPI
- `frontend/` : interface web
- `notebooks/` : scripts d'exploration et d'entraînement

## Installation
1. Activez l'environnement virtuel :
   ```bash
   .venv\Scripts\activate
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Copiez le fichier d'exemple :
   ```bash
   copy .env.example .env
   ```
4. Remplissez `GROQ_API_KEY` si vous utilisez l'outil Groq.

## Entraînement du modèle
Exécutez :
```bash
python notebooks/train_model.py
```
Le modèle sera entraîné, évalué et les artefacts seront enregistrés dans `models/`.

## Lancer l'API
Exécutez :
```bash
uvicorn api.main:app --reload
```
Puis ouvrez `http://127.0.0.1:8000/docs` pour tester l'API.

## Utiliser le frontend
Ouvrez `frontend/index.html` dans un navigateur.
Le formulaire envoie les symptômes au service FastAPI.

## Notes
- `models/` n'est pas versionné dans Git.
- `data/patients_dakar.csv` contient les diagnostics, régions et symptômes.
- Les scripts de `notebooks/` sont organisés en fonctions avec un point d'entrée `main()`.

## Auteur
Ken Bougoul DIOP - L2 GLSI - ESP/UCAD

## Cours
Intégration de Modèles IA - Dr. El Hadji Bassirou TOURE
---
title: SénSanté
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: api/main.py
pinned: false
---

# SénSanté - Prédiction de Maladies

Application de machine learning pour la prédiction de maladies à partir de données médicales.
Projet académique — IMA02, ESP UCAD Dakar.