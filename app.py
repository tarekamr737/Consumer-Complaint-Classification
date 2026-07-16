"""Consumer Complaint Classification internal decision-support application."""

from __future__ import annotations

import html
import json
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
    columns = ["Time", "Predicted category", "Confidence", "Decision status"]
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
    result = (
        "<section class='prediction' aria-live='polite'>"
        "<p class='eyebrow'>Recommended route</p>"
        f"<h2>{html.escape(category)}</h2>"
        f"<p><strong>{confidence_descriptor} confidence: {top['confidence']:.1f}%.</strong> "
        "Review the alternatives before routing the case.</p>"
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
:root { --ink: oklch(24% 0.025 260); --muted: oklch(46% 0.025 260); --line: oklch(87% 0.012 260); --paper: oklch(98% 0.008 80); --surface: oklch(100% 0.003 80); --blue: oklch(42% 0.14 256); --blue-quiet: oklch(92% 0.035 256); --success: oklch(42% 0.1 150); --danger: oklch(46% 0.16 25); }
body, .gradio-container { background: var(--paper) !important; color: var(--ink) !important; font-family: Georgia, 'Times New Roman', serif !important; }
.gradio-container { max-width: 1440px !important; padding: 0 2rem 3rem !important; }
#utility-bar { border-bottom: 1px solid var(--line); margin: 0 -2rem 2.5rem; padding: 1rem 2rem; background: var(--surface); }
.utility { font-family: ui-sans-serif, system-ui, sans-serif; display:flex; align-items:center; justify-content:space-between; gap:1rem; font-size:.875rem; }
.product-name { font-weight:750; letter-spacing:-.02em; font-size:1rem; color:var(--ink); } .status { color:var(--success); font-weight:700; }
.page-heading { max-width: 68ch; margin: 0 0 2rem; } .page-heading h1 { font-size:2.5rem; line-height:1.08; letter-spacing:-.045em; margin:0 0 .5rem; } .page-heading p { color:var(--muted); font-size:1rem; line-height:1.55; margin:0; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:1.5rem; box-shadow:0 1px 2px oklch(24% .025 260 / .06); }
.panel + .panel { margin-top:1.25rem; } .panel h2 { font-family:ui-sans-serif, system-ui, sans-serif; font-size:1rem; margin:0 0 .35rem; letter-spacing:-.015em; } .panel p { color:var(--muted); margin:.25rem 0 1rem; }
.gr-textbox textarea { min-height:210px !important; font-family:Georgia, 'Times New Roman', serif !important; font-size:1rem !important; line-height:1.55 !important; border-color:var(--line) !important; }
.gr-textbox textarea:focus { box-shadow:0 0 0 3px var(--blue-quiet) !important; border-color:var(--blue) !important; }
button.primary { background:var(--blue) !important; border-color:var(--blue) !important; min-height:44px !important; font-family:ui-sans-serif, system-ui, sans-serif !important; font-weight:700 !important; } button.secondary { min-height:44px !important; }
.prediction { border-top:3px solid var(--blue); padding-top:1rem; } .prediction h2 { color:var(--ink); font-size:1.7rem; margin:.25rem 0 .5rem; } .eyebrow { font-family:ui-sans-serif, system-ui, sans-serif; color:var(--blue) !important; font-size:.75rem !important; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.notice { border-radius:8px; padding:1rem; font-family:ui-sans-serif, system-ui, sans-serif; line-height:1.45; } .notice.error { background:oklch(96% .03 25); color:var(--danger); } code { font-family:ui-monospace, SFMono-Regular, Consolas, monospace; }
.dataframe { font-family:ui-sans-serif, system-ui, sans-serif !important; } .gr-button:focus-visible, textarea:focus-visible { outline:3px solid var(--blue) !important; outline-offset:3px !important; }
@media (max-width: 720px) { .gradio-container { padding:0 1rem 2rem !important; } #utility-bar { margin:0 -1rem 1.75rem; padding:1rem; } .utility { align-items:flex-start; flex-direction:column; } .page-heading h1 { font-size:2rem; } .panel { padding:1rem; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
"""


def build_app() -> gr.Blocks:
    history = _load_history()
    with gr.Blocks(title="Consumer Complaint Classification", css=CSS) as demo:
        gr.HTML("<header id='utility-bar'><div class='utility'><span class='product-name'>Consumer Complaint Classification</span><span class='status'>● Model ready for routing</span></div></header>")
        gr.HTML("<main class='page-heading'><h1>Route a complaint with evidence.</h1><p>Paste a consumer complaint to get a category recommendation, ranked alternatives, and a local audit trail for review.</p></main>")
        with gr.Row(equal_height=False):
            with gr.Column(scale=7):
                with gr.Group(elem_classes="panel"):
                    gr.HTML("<h2>Complaint narrative</h2><p>Include the customer’s issue, account type, and any disputed activity.</p>")
                    narrative = gr.Textbox(label="Complaint narrative", placeholder="Paste the complaint narrative here…", lines=9, max_lines=16, show_label=False)
                    with gr.Row():
                        classify_button = gr.Button("Classify complaint", variant="primary", elem_classes="primary")
                        example_button = gr.Button("Load example", variant="secondary", elem_classes="secondary")
                        clear_button = gr.ClearButton([narrative], value="Clear narrative", elem_classes="secondary")
            with gr.Column(scale=5):
                with gr.Group(elem_classes="panel"):
                    gr.HTML("<h2>Model recommendation</h2><p>Use the prediction to inform, not replace, routing judgment.</p>")
                    recommendation = gr.HTML("<div class='notice'>No classification yet. Paste a narrative and classify it to see a routing recommendation.</div>")
                    top_three = gr.Dataframe(headers=["Rank", "Category", "Confidence"], datatype=["number", "str", "str"], interactive=False, label="Top three categories")
        with gr.Group(elem_classes="panel"):
            gr.HTML("<h2>Recent classifications</h2><p>Stored locally for this app instance. New results appear here after classification.</p>")
            history_table = gr.Dataframe(value=_history_frame(history), interactive=False, label="Classification history")
        with gr.Group(elem_classes="panel"):
            gr.HTML("<h2>Classification analytics</h2><p>Category distribution across the locally recorded classifications.</p>")
            analytics_table = gr.Dataframe(value=_analytics_frame(history), interactive=False, label="Category distribution")

        outputs = [recommendation, top_three, history_table, analytics_table]
        classify_button.click(classify, inputs=narrative, outputs=outputs)
        narrative.submit(classify, inputs=narrative, outputs=outputs)
        example_button.click(lambda: EXAMPLE_NARRATIVE, outputs=narrative)
        clear_button.click(lambda: _empty_outputs("<div class='notice'>No classification yet. Paste a narrative and classify it to see a routing recommendation.</div>"), outputs=outputs)
    return demo


if __name__ == "__main__":
    build_app().launch()
