# TASKS.md

# Consumer Complaint Classification Gradio App

## Phase 1 — Data
- [x] inspect dataset in data folder
- [x] Select `Consumer complaint narrative` & `Product`
- [x] Remove nulls & duplicates
- [x] Handle class imbalance (if needed)
- [x] Stratified train/val/test split

---

## Phase 2 — EDA
- [x] Analyze class distribution
- [x] Analyze complaint lengths & word frequencies

---

## Phase 3 — Preprocessing
### Custom Models
- [x] Clean text
- [x] Tokenize
- [x] Pad sequences

### Transformer
- [x] HuggingFace tokenizer

---

## Phase 4 — Modeling
- [x] Train SimpleRNN
- [x] Train LSTM
- [x] Train GRU
- [x] Fine-tune DistilBERT

---

## Phase 5 — Evaluation
- [x] Accuracy, Precision, Recall, F1
- [x] Confusion Matrix & Classification Report
- [x] Compare models
- [x] Select best model

---

## Phase 6 — Deployment
- [x] Save best model & preprocessing artifacts
- [x] Design UI with **Impeccable**
- [x] Build Gradio app (`gr.Blocks`)
- [x] Apply custom Theme & CSS
- [x] Display predicted category, confidence & top-3 predictions
- [x] Test end-to-end

---

## Deliverables
- [x] Notebooks
- [x] Trained models
- [x] Gradio app
- [x] README
- [x] requirements.txt