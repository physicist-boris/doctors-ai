# Projet Doctors-AI

## Configuration initale

### Création de l'environnement

```
python -m venv doctors-ai-venv
```

### Installation des dépendances

Activez l'environnement virtuel

```
cd doctors-ai-venv/Scripts
```

```
activate.bat
```

Á partir de la racine du projet, installez les dépendances

```
pip install -r env/requirements-dev.txt
```

```
pre-commit install
```

### Setup the server

```
pip install -r env/requirements-server.txt
pip install -e .
python server_app\app.py
```

### Run a query

```
curl -X POST -H "Content-Type: application/json" -d '{"date": "2021-12-30"}' http://localhost:5000/predict
```

```
{
    "predicted_number_admissions": 8,
    "number_in_ed": 17
}
```
