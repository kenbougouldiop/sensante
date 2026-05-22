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
Ouvrez `http://127.0.0.1:8000` dans un navigateur après avoir démarré l'API.
Le frontend est servi par FastAPI et utilise des appels relatifs vers `/predict` et `/explain`.

## Docker
1. Construisez l'image :
   ```bash
   docker build -t sensante .
   ```
2. Lancez le conteneur :
   ```bash
   docker run -p 8000:8000 -e GROQ_API_KEY=your_key sensante
   ```
3. Ouvrez `http://127.0.0.1:8000`.

## Déploiement Hugging Face Spaces
- Le projet est prêt pour un Space Docker.
- Ajoutez `GROQ_API_KEY` comme secret dans Settings du Space.
- Poussez sur le remote Hugging Face et le build se fera automatiquement.

## Notes
- `models/` n'est pas versionné dans Git.
- `data/patients_dakar.csv` contient les diagnostics, régions et symptômes.
- Les scripts de `notebooks/` sont organisés en fonctions avec un point d'entrée `main()`.

## Auteur
Ken Bougoul DIOP - L2 GLSI - ESP/UCAD

## Cours
Intégration de Modèles IA - Dr. El Hadji Bassirou TOURE
---