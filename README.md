# Precision Oncology

A local demonstration app for somatic variant analysis and drug recommendations, with a FastAPI backend and browser-based frontend.

> This project is for demonstration and research only. It is not a medical device and must not be used to make clinical decisions.

## Project files

- `index.html` — browser-based frontend
- `backend/main.py` — FastAPI API
- `backend/models/` — trained demonstration model artifacts
- `backend/dbGENVOC.xlsx` — demonstration variant dataset
- `prepare_models.py` — script to regenerate the dataset and model artifacts
- `start_app.bat` — starts the local API and frontend on Windows

## Run on Windows

Install Python 3 and the dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

Then double-click `start_app.bat`, or run the API and frontend separately:

```powershell
py -3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
py -3 -m http.server 5500
```

Open <http://127.0.0.1:5500/index.html>. The API health endpoint is <http://127.0.0.1:8000/health>.

## Local user accounts

Demo accounts are stored in `backend/users.json` on the local machine. That file is intentionally excluded from version control because it can contain personal data and password hashes. The repository includes `backend/users.example.json` as an empty starting point.

## Regenerate demonstration data and models

From the project root, run:

```powershell
py -3 prepare_models.py
```
