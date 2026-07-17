---
title: Consumer Complaint Classification
emoji: 🧭
colorFrom: slate
colorTo: blue
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
---

# Consumer Complaint Classification

An internal Gradio decision-support tool for routing consumer complaints. Analysts paste a narrative, review a category recommendation and top-three alternatives, then inspect local history and category analytics.

## Run locally

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe scripts\train_baseline.py
.venv\Scripts\python.exe app.py
```

The initial app uses a balanced TF-IDF + logistic-regression baseline so it is usable immediately. The repository also includes the recurrent-model preprocessing and training scaffold for the required SimpleRNN, LSTM, and GRU experiments.

## Model experiments

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe scripts\train_recurrent_models.py --epochs 2 --batch-size 256
.venv\Scripts\python.exe scripts\train_distilbert.py --epochs 2 --batch-size 8
.venv\Scripts\python.exe scripts\evaluate_models.py
```

The checked-in local experiment uses a one-epoch, 512-record DistilBERT smoke fine-tune because this workspace is CPU-only. Remove the sample caps for a full fine-tune. Model comparison, reports, confusion matrices, and the selected deployment descriptor are written to `reports/` and `artifacts/deployment_model.json`.

## Deploy to Render

The included `render.yaml` deploys the lightweight inference runtime rather than the full training stack. In Render, create a **Blueprint** from this repository, select the free web-service plan, and deploy. The free service sleeps after inactivity and its local classification history resets when the service restarts.
