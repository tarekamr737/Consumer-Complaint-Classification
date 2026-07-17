"""Consumer Complaint Classification internal decision-support application."""

from __future__ import annotations

import html
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from complaint_app.baseline import BaselineClassifier


ARTIFACT_PATH = ROOT / "artifacts" / "baseline_classifier.joblib"
HISTORY_PATH = ROOT / "artifacts" / "classification_history.json"
EXAMPLE_NARRATIVE = (
    "I found an account on my credit report that does not belong to me. "
    "I disputed it with the credit bureau, but the inaccurate information is still reported."
)


def _load_history() -> list[dict[str, str | float]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_history(history: list[dict[str, str | float]]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history[:100], indent=2), encoding="utf-8")


def _history_frame(history: list[dict[str, str | float]]) -> pd.DataFrame:
    columns = ["Time", "Predicted category", "Confidence"]
    if not history:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(history)
    return frame.rename(
        columns={
            "timestamp": "Time",
            "category": "Predicted category",
            "confidence": "Confidence",
            "status": "Decision status",
        }
    )[columns]


def _analytics_frame(history: list[dict[str, str | float]]) -> pd.DataFrame:
    columns = ["Category", "Classifications", "Share"]
    if not history:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(history)
    counts = frame["category"].value_counts()
    shares = (counts.values / len(frame) * 100).round(1)
    return pd.DataFrame(
        {
            "Category": counts.index.str.replace("_", " ").str.title(),
            "Classifications": counts.values,
            "Share": [f"{share:.1f}%" for share in shares],
        }
    )


def _empty_outputs(message: str):
    history = _load_history()
    return message, pd.DataFrame(columns=["Rank", "Category", "Confidence"]), _history_frame(history), _analytics_frame(history)


def classify(narrative: str):
    narrative = (narrative or "").strip()
    if not narrative:
        return _empty_outputs("<div class='notice error'><strong>Enter a complaint narrative.</strong><br>Paste the customer’s description, then select <em>Classify complaint</em>.</div>")
    if len(narrative) < 20:
        return _empty_outputs("<div class='notice error'><strong>Add a little more detail.</strong><br>Use at least 20 characters so the model has enough context to route the complaint.</div>")
    try:
        classifier = BaselineClassifier(ARTIFACT_PATH)
        ranked = classifier.predict(narrative)[:3]
    except FileNotFoundError:
        return _empty_outputs("<div class='notice error'><strong>Model unavailable.</strong><br>Run <code>python scripts/train_baseline.py</code> before classifying complaints.</div>")

    top = ranked[0]
    confidence_descriptor = "High" if top["confidence"] >= 80 else "Moderate" if top["confidence"] >= 55 else "Low"
    category = str(top["category"]).replace("_", " ").title()
    confidence_width = max(6, min(100, float(top["confidence"])))
    result = (
        "<section class='prediction' aria-live='polite'>"
        "<div class='prediction-top'><p class='eyebrow'>Recommended route</p>"
        f"<span class='confidence-badge confidence-{confidence_descriptor.lower()}'>{confidence_descriptor} confidence</span></div>"
        f"<h2>{html.escape(category)}</h2>"
        f"<div class='confidence-row'><span>Model confidence</span><strong>{top['confidence']:.1f}%</strong></div>"
        f"<div class='confidence-track' role='progressbar' aria-valuenow='{top['confidence']:.1f}' aria-valuemin='0' aria-valuemax='100'><span style='width:{confidence_width:.1f}%'></span></div>"
        "<p class='prediction-note'>Use this recommendation as routing support. Check the alternatives when the complaint is ambiguous.</p>"
        "</section>"
    )
    top_three = pd.DataFrame(
        [
            {"Rank": index + 1, "Category": str(item["category"]).replace("_", " ").title(), "Confidence": f"{item['confidence']:.1f}%"}
            for index, item in enumerate(ranked)
        ]
    )
    history = _load_history()
    history.insert(
        0,
        {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "category": str(top["category"]),
            "confidence": f"{top['confidence']:.1f}%",
            "status": "Ready for review",
        },
    )
    _save_history(history)
    return result, top_three, _history_frame(history), _analytics_frame(history)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root { --ink:oklch(22% .018 255); --ink-muted:oklch(42% .018 255); --paper:oklch(97.8% .006 250); --panel:oklch(100% .004 250); --panel-muted:oklch(95.8% .010 250); --line:oklch(84% .014 255); --blue:oklch(35% .13 255); --blue-hover:oklch(30% .13 255); --blue-soft:oklch(93% .028 255); --green:oklch(39% .09 153); --green-soft:oklch(93% .030 153); --amber:oklch(55% .10 75); --amber-soft:oklch(94% .038 75); --red:oklch(44% .15 28); --red-soft:oklch(94% .030 28); }
body, .gradio-container { background:var(--paper) !important; color:var(--ink) !important; font-family:Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; font-kerning:normal; }
.gradio-container { --body-background-fill:var(--paper); --background-fill-primary:var(--panel); --background-fill-secondary:var(--panel-muted); --body-text-color:var(--ink); --body-text-color-subdued:var(--ink-muted); --font-mono:Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; --block-background-fill:var(--panel); --block-border-color:var(--line); --block-label-background-fill:var(--panel-muted); --block-label-border-color:var(--line); --block-label-text-color:var(--ink); --block-title-text-color:var(--ink); --block-info-text-color:var(--ink-muted); --panel-background-fill:var(--panel); --panel-border-color:var(--line); --border-color-primary:var(--line); --input-background-fill:oklch(96.5% .012 250); --input-background-fill-hover:oklch(96.5% .012 250); --input-background-fill-focus:oklch(96.5% .012 250); --input-border-color:var(--line); --input-border-color-focus:var(--blue); --table-border-color:var(--line); --table-text-color:var(--ink); --table-even-background-fill:var(--panel); --table-odd-background-fill:oklch(97.6% .006 250); --table-row-background-fill:var(--panel); --table-row-focus:var(--blue-soft); --button-secondary-text-color:var(--ink-muted); max-width:1440px !important; padding:0 2rem 3rem !important; }
#app-shell { display:block; }
#utility-bar { background:var(--panel); border-bottom:1px solid var(--line); margin:0 0 2rem; padding:0; overflow:hidden; }
.utility { min-height:64px; display:flex; align-items:center; justify-content:space-between; gap:1rem; font-size:.8125rem; }
.product-lockup { display:flex; align-items:center; gap:.75rem; } .product-mark { width:28px; height:28px; display:grid; place-items:center; border-radius:4px; background:var(--blue); color:var(--panel); font-size:.68rem; font-weight:700; letter-spacing:.02em; } .product-name { font:600 .9375rem/1 "Hanken Grotesk", Inter, sans-serif; letter-spacing:-.015em; } .status { color:var(--green); font-weight:600; } .status::before { content:""; width:7px; height:7px; display:inline-block; margin:0 7px 1px 0; border-radius:50%; background:currentColor; }
.skip-link { position:absolute; top:-48px; left:16px; z-index:10; padding:10px 14px; background:var(--blue); color:var(--panel); border-radius:4px; font-weight:600; } .skip-link:focus { top:12px; }
.workspace-heading { margin:0 0 1.5rem; display:flex; align-items:end; justify-content:space-between; gap:2rem; } .workspace-heading h1 { color:var(--ink) !important; font:600 1.875rem/1.25 "Hanken Grotesk", Inter, sans-serif; letter-spacing:-.02em; margin:0; text-wrap:balance; } .workspace-heading p { max-width:56ch; margin:.35rem 0 0; color:var(--ink-muted) !important; font-size:.875rem; line-height:1.45; } .workspace-heading .meta { color:var(--ink-muted) !important; font-size:.75rem; white-space:nowrap; }
.shell-row { gap:24px !important; align-items:stretch !important; }
.history-rail { align-self:stretch; background:var(--panel-muted); border:1px solid var(--line); border-radius:8px; padding:1rem; min-width:0; }
.rail-head, .section-head { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin-bottom:12px; } .rail-head h2, .section-head h2 { color:var(--ink) !important; margin:0; font:600 1rem/1.3 "Hanken Grotesk", Inter, sans-serif; letter-spacing:-.01em; } .rail-head span, .section-head p { margin:0; color:var(--ink-muted) !important; font-size:.75rem; line-height:1.4; }
.rail-note { margin:16px 0 0; padding-top:12px; border-top:1px solid var(--line); color:var(--ink-muted); font-size:.75rem; line-height:1.5; }
.main-workspace { min-width:0; } .decision-grid { gap:24px !important; align-items:stretch !important; }
.work-panel, .lower-panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1.25rem; min-width:0; }
.work-panel { min-height:420px; } .panel-title { color:var(--ink) !important; margin:0; font:600 1rem/1.3 "Hanken Grotesk", Inter, sans-serif; letter-spacing:-.01em; } .panel-copy { margin:.375rem 0 1rem; color:var(--ink-muted) !important; font-size:.8125rem; line-height:1.5; }
.work-panel .form, .narrative-input, .narrative-input > div, .narrative-input .wrap { background:transparent !important; border-color:var(--line) !important; box-shadow:none !important; } .narrative-input label { color:var(--ink) !important; } .narrative-input textarea { min-height:250px !important; resize:vertical !important; border:1px solid var(--line) !important; border-radius:4px !important; background:oklch(96.5% .012 250) !important; color:var(--ink) !important; font-family:Inter, sans-serif !important; font-size:1rem !important; line-height:1.55 !important; padding:.875rem !important; box-shadow:none !important; } .narrative-input textarea::placeholder { color:oklch(47% .018 255) !important; opacity:1; } .narrative-input textarea:focus { border-color:var(--blue) !important; box-shadow:0 0 0 2px var(--blue-soft) !important; }
.action-row { gap:8px !important; margin-top:1rem !important; } button.primary, button.secondary { min-height:44px !important; border-radius:4px !important; padding:0 .9rem !important; font:600 .8125rem/1 Inter, sans-serif !important; transition:background-color 160ms ease-out, border-color 160ms ease-out, box-shadow 160ms ease-out !important; } button.primary { background:var(--blue) !important; border-color:var(--blue) !important; } button.primary:hover { background:var(--blue-hover) !important; border-color:var(--blue-hover) !important; box-shadow:0 2px 5px oklch(22% .018 255 / .12) !important; } button.primary:active { background:oklch(26% .12 255) !important; } button.secondary { background:var(--panel) !important; border-color:var(--line) !important; color:var(--ink-muted) !important; } button.secondary:hover { background:var(--panel-muted) !important; border-color:oklch(70% .022 255) !important; color:var(--ink) !important; }
.recommendation-area { min-height:205px; } .notice { display:grid; place-items:center; min-height:176px; box-sizing:border-box; padding:1.25rem; background:var(--panel-muted); border:1px dashed var(--line); border-radius:4px; color:var(--ink-muted); font-size:.8125rem; line-height:1.5; } .notice.error { display:block; background:var(--red-soft); border-style:solid; border-color:oklch(80% .06 28); color:var(--red); } .notice strong { color:var(--ink); }
.prediction { padding:1.25rem; background:var(--blue-soft); border:1px solid oklch(79% .05 255); border-radius:4px; } .prediction-top { display:flex; align-items:center; justify-content:space-between; gap:.75rem; } .eyebrow { margin:0; color:var(--blue); font-size:.6875rem; line-height:1.2; font-weight:700; letter-spacing:.085em; text-transform:uppercase; } .prediction h2 { margin:.5rem 0 1.25rem; font:600 1.375rem/1.2 "Hanken Grotesk", Inter, sans-serif; letter-spacing:-.02em; } .confidence-badge { padding:.3rem .45rem; border:1px solid currentColor; border-radius:4px; font-size:.6875rem; font-weight:700; white-space:nowrap; } .confidence-high { color:var(--green); background:var(--green-soft); } .confidence-moderate { color:var(--amber); background:var(--amber-soft); } .confidence-low { color:var(--red); background:var(--red-soft); } .confidence-row { display:flex; justify-content:space-between; margin-bottom:.45rem; color:var(--ink-muted); font-size:.75rem; } .confidence-row strong { color:var(--ink); font-variant-numeric:tabular-nums; } .confidence-track { height:7px; overflow:hidden; background:oklch(83% .030 255); border-radius:2px; } .confidence-track span { display:block; height:100%; background:var(--blue); } .prediction-note { margin:1rem 0 0; color:var(--ink-muted); font-size:.75rem; line-height:1.45; }
.routes-title { margin:1.25rem 0 .625rem; color:var(--ink-muted); font-size:.6875rem; font-weight:700; letter-spacing:.075em; text-transform:uppercase; }
.lower-panel { margin-top:24px; } .lower-panel .section-head { margin-bottom:1rem; }
.dataframe, [data-testid="dataframe"], [data-testid="dataframe"] > div, [data-testid="dataframe"] .table-wrap { background:var(--panel) !important; color:var(--ink) !important; font-family:Inter, sans-serif !important; font-size:.75rem !important; font-variant-numeric:tabular-nums; } .dataframe table, [data-testid="dataframe"] table { width:100% !important; border-collapse:collapse !important; background:var(--panel) !important; color:var(--ink) !important; } .dataframe thead, .dataframe tbody, [data-testid="dataframe"] thead, [data-testid="dataframe"] tbody { background:var(--panel) !important; color:var(--ink) !important; } .dataframe tbody tr, [data-testid="dataframe"] tbody tr { background:var(--panel) !important; color:var(--ink) !important; } .dataframe tbody tr:nth-child(even), [data-testid="dataframe"] tbody tr:nth-child(even) { background:oklch(97.6% .006 250) !important; } .dataframe td, .dataframe th, [data-testid="dataframe"] td, [data-testid="dataframe"] th { background:inherit !important; color:var(--ink) !important; border-color:var(--line) !important; } .dataframe th, [data-testid="dataframe"] th { color:var(--ink-muted) !important; font-size:.6875rem !important; font-weight:700 !important; letter-spacing:.04em; text-transform:uppercase; } .history-rail .dataframe table, .history-rail [data-testid="dataframe"] table { table-layout:fixed !important; } .history-rail .dataframe td, .history-rail .dataframe th, .history-rail [data-testid="dataframe"] td, .history-rail [data-testid="dataframe"] th { overflow-wrap:anywhere; white-space:normal !important; }
.gr-button:focus-visible, textarea:focus-visible, a:focus-visible { outline:2px solid var(--blue) !important; outline-offset:3px !important; }
@media (max-width:900px) { .shell-row { flex-direction:column !important; } .history-rail { max-height:none; } .history-rail .table-wrap { height:220px !important; max-height:220px !important; } .workspace-heading { align-items:start; flex-direction:column; gap:.5rem; } }
@media (max-width:720px) { .gradio-container { padding:0 1rem 2.5rem !important; } #utility-bar { margin:0 -1rem 1.5rem; padding:0 1rem; } .utility { min-height:56px; } .status { font-size:.6875rem; } .workspace-heading h1 { font-size:1.5rem; } .decision-grid { flex-direction:column !important; } .work-panel, .lower-panel, .history-rail { padding:1rem; } .work-panel { min-height:unset; } .narrative-input textarea { min-height:210px !important; } .action-row { flex-wrap:wrap !important; } }
@media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
"""


def build_app() -> gr.Blocks:
    history = _load_history()
    with gr.Blocks(title="Consumer Complaint Classification", css=CSS) as demo:
        gr.HTML("<a class='skip-link' href='#triage-workspace'>Skip to triage workspace</a><header id='utility-bar'><div class='utility'><div class='product-lockup'><span class='product-mark' aria-hidden='true'>CC</span><span class='product-name'>Complaint Classifier</span></div><span class='status'>Model available</span></div></header>")
        gr.HTML("<section class='workspace-heading'><div><h1>Routing workspace</h1><p>Review each complaint as a decision record, then use the model recommendation to route it with informed judgment.</p></div><span class='meta'>Internal decision support</span></section>")
        with gr.Row(equal_height=False, elem_classes="shell-row"):
            with gr.Column(scale=3, min_width=260, elem_classes="history-rail"):
                gr.HTML("<div class='rail-head'><h2>Recent triage</h2><span>Local audit</span></div>")
                history_table = gr.Dataframe(value=_history_frame(history), interactive=False, label="Recent classification history", show_label=False, wrap=True)
                gr.HTML("<p class='rail-note'>The most recent 100 classifications are retained in this local app instance.</p>")
            with gr.Column(scale=9, min_width=400, elem_id="triage-workspace", elem_classes="main-workspace"):
                with gr.Row(equal_height=False, elem_classes="decision-grid"):
                    with gr.Column(scale=7, min_width=300, elem_classes="work-panel"):
                        gr.HTML("<h2 class='panel-title'>Complaint narrative</h2><p class='panel-copy'>Include the issue, product or account type, and relevant disputed activity.</p>")
                        narrative = gr.Textbox(label="Complaint narrative", placeholder="Paste the customer’s complaint narrative here", lines=10, max_lines=18, elem_classes="narrative-input")
                        with gr.Row(elem_classes="action-row"):
                            classify_button = gr.Button("Classify complaint", variant="primary", elem_classes="primary")
                            example_button = gr.Button("Load example", variant="secondary", elem_classes="secondary")
                            clear_button = gr.ClearButton([narrative], value="Clear narrative", elem_classes="secondary")
                    with gr.Column(scale=5, min_width=280, elem_classes="work-panel"):
                        gr.HTML("<h2 class='panel-title'>Model recommendation</h2><p class='panel-copy'>A routing aid, not an automated decision.</p>")
                        recommendation = gr.HTML("<div class='notice' aria-live='polite'>Paste a complaint narrative, then classify it to review the suggested route.</div>", elem_classes="recommendation-area")
                        gr.HTML("<p class='routes-title'>Alternative routes</p>")
                        top_three = gr.Dataframe(value=pd.DataFrame(columns=["Rank", "Category", "Confidence"]), headers=["Rank", "Category", "Confidence"], datatype=["number", "str", "str"], interactive=False, label="Top three routes", show_label=False, wrap=True)
                with gr.Group(elem_classes="lower-panel"):
                    gr.HTML("<div class='section-head'><div><h2>Routing distribution</h2><p>Category share across locally recorded classifications.</p></div></div>")
                    analytics_table = gr.Dataframe(value=_analytics_frame(history), interactive=False, label="Category distribution", show_label=False, wrap=True)

        outputs = [recommendation, top_three, history_table, analytics_table]
        classify_button.click(classify, inputs=narrative, outputs=outputs)
        narrative.submit(classify, inputs=narrative, outputs=outputs)
        example_button.click(lambda: EXAMPLE_NARRATIVE, outputs=narrative)
        clear_button.click(lambda: _empty_outputs("<div class='notice'>No classification yet. Paste a narrative and classify it to see a routing recommendation.</div>"), outputs=outputs)
    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
