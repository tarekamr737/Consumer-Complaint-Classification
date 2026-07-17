# Consumer Complaint Classification

An internal decision-support application for routing consumer complaints. Analysts paste a complaint narrative, review the recommended category with confidence and alternative routes, then retain a local audit trail of recent classifications.

Built with Gradio, scikit-learn, and a TF-IDF plus logistic-regression classifier.

## Why this project

Complaint routing is a high-volume task where inconsistent categorisation creates avoidable manual work. This application provides a fast, explainable recommendation while keeping the routing decision with the analyst.

The interface is designed for operational use:

- Clear two-pane triage workspace for narrative input and model output.
- Ranked category recommendations with confidence expressed as both text and a visual indicator.
- Local classification history and category-distribution reporting.
- Responsive, keyboard-accessible UI with visible focus states and reduced-motion support.

## Live application

Deploy the application with the included Render Blueprint, or run it locally using the instructions below.

## Selected model

The deployed model is a balanced TF-IDF and logistic-regression pipeline. It was selected because it achieved the strongest weighted F1 in the shared comparison while keeping inference lightweight, deterministic, and probability-based.

### Benchmark comparison

All models below were compared on the same 512-record test sample. Weighted F1 is the selection metric because the complaint categories are imbalanced.

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| **TF-IDF + Logistic Regression** | **0.8359** | **0.8445** | **0.8359** | **0.8374** |
| GRU | 0.8027 | 0.8127 | 0.8027 | 0.8030 |
| SimpleRNN | 0.7617 | 0.7746 | 0.7617 | 0.7630 |
| LSTM | 0.7324 | 0.7682 | 0.7324 | 0.7357 |
| DistilBERT smoke run | 0.4824 | 0.4880 | 0.4824 | 0.4355 |

### Evaluation context

- **Dataset:** 124,676 cleaned complaint narratives across five product categories.
- **Data quality:** 37,735 duplicate rows and 10 blank rows were removed before splitting.
- **Split:** deterministic stratified 70/15/15 train, validation, and test split using seed `42`.
- **Class imbalance:** the largest category is 4.179 times the size of the smallest, so class weighting and weighted metrics are used.
- **Baseline full-test result:** on the full 18,702-record test split, the selected baseline achieved 0.8492 accuracy and 0.8505 weighted F1.

The DistilBERT result is a CPU-bounded smoke run, not a full fine-tune. It demonstrates the transformer pipeline but should not be treated as a production comparison without retraining on the full split with appropriate compute.

See [`reports/model_comparison.json`](reports/model_comparison.json) for the shared-sample comparison, [`reports/baseline_metrics.json`](reports/baseline_metrics.json) for the baseline full-test result, and [`artifacts/deployment_model.json`](artifacts/deployment_model.json) for the deployment selection.

## Architecture

```text
Complaint narrative
        |
        v
Text cleaning and TF-IDF features
        |
        v
Balanced logistic-regression classifier
        |
        +--> recommended category
        +--> confidence score
        +--> ranked alternative routes
        |
        v
Gradio triage workspace and local audit history
```

## Repository structure

```text
app.py                     Gradio application entry point
src/complaint_app/         Data preparation, preprocessing, and model code
scripts/                   Repeatable preparation, training, and evaluation commands
artifacts/                 Deployment model and lightweight baseline artifact
reports/                   Evaluation metrics and confusion matrices
data/                      Source data and prepared train, validation, and test splits
render.yaml                Render Blueprint configuration
```

## Run locally

### Prerequisites

- Python 3.11 or later
- A virtual environment

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the runtime dependencies

```powershell
pip install -r requirements.runtime.txt
```

### 3. Start the app

The baseline model artifact is versioned with the project, so no training is required to launch the application.

```powershell
python app.py
```

Open the local URL shown in the terminal, normally `http://127.0.0.1:7860`.

## Reproduce the training workflow

Install the full experiment dependencies first:

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

Then run the workflow in order:

```powershell
python scripts\prepare_data.py
python scripts\train_baseline.py
python scripts\train_recurrent_models.py --epochs 2 --batch-size 256
python scripts\train_distilbert.py --epochs 2 --batch-size 8
python scripts\evaluate_models.py
```

The DistilBERT command is intentionally bounded for a CPU-only smoke run. Train on the complete dataset with suitable compute before considering it for production use.

## Deploy on Render

[`render.yaml`](render.yaml) defines a lightweight web service that installs only the inference dependencies and runs `app.py`.

1. Push this repository to GitHub.
2. In Render, choose **New** then **Blueprint**.
3. Connect this repository and approve the detected `render.yaml` configuration.
4. Deploy the service.

The free Render plan may sleep after inactivity. Classification history is stored locally by the app and resets whenever the service restarts or is redeployed.

## Artifact policy

The lightweight baseline artifact, `artifacts/baseline_classifier.joblib`, is committed so the app runs immediately after cloning or deploying. Generated neural-network models, DistilBERT checkpoints, local classification history, Python caches, and runtime logs are excluded through [`.gitignore`](.gitignore).

## Limitations and responsible use

- This is decision support, not an automated routing authority.
- Confidence indicates the model's relative certainty, not correctness.
- Analysts should review alternatives when a narrative is short, ambiguous, or out of scope.
- Results depend on the quality and coverage of the training data and should be monitored after deployment.

## License

No license has been provided for this repository.
