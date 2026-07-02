"""
app.py

Premium, minimal, product-grade Gradio UI for the Image Authenticity
Detection tool. Frontend only — backend/ML logic is untouched.
"""

import traceback

import gradio as gr

from predict import predict_image


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0a0a0b;
    --glass: rgba(22, 22, 24, 0.72);
    --border: rgba(255, 255, 255, 0.08);
    --text-primary: #ffffff;
    --text-secondary: #a1a1aa;
    --success: #22c55e;
    --success-bg: rgba(34, 197, 94, 0.10);
    --success-border: rgba(34, 197, 94, 0.35);
    --danger: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.10);
    --danger-border: rgba(239, 68, 68, 0.35);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding-top: 56px !important;
    padding-bottom: 40px !important;
}

footer { display: none !important; }

/* ---------- Header ---------- */
#header-title {
    text-align: center;
    margin-bottom: 2px !important;
}
#header-title h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
#header-subtitle {
    text-align: center;
    color: var(--text-secondary) !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    margin-top: 6px !important;
    margin-bottom: 36px !important;
}

/* ---------- Glass panels ---------- */
.panel {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}

.panel-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 14px;
}

/* ---------- Upload box ---------- */
#image-input {
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid var(--border) !important;
    min-height: 340px !important;
}
#image-input .wrap {
    border-radius: 10px !important;
}

/* ---------- Analyze button ---------- */
#analyze-btn {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 11px 18px !important;
    margin-top: 14px !important;
    width: 100%;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    box-shadow: 0 2px 10px rgba(255, 255, 255, 0.06);
}
#analyze-btn:hover {
    transform: scale(1.015);
    box-shadow: 0 4px 16px rgba(255, 255, 255, 0.12);
}
#analyze-btn:active {
    transform: scale(0.99);
}

/* ---------- Result card ---------- */
#result-card {
    min-height: 340px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 18px;
}

.status-box {
    border-radius: 10px;
    padding: 16px 18px;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-align: center;
    border: 1px solid var(--border);
}
.status-success {
    background: var(--success-bg);
    border-color: var(--success-border);
    color: var(--success);
}
.status-danger {
    background: var(--danger-bg);
    border-color: var(--danger-border);
    color: var(--danger);
}
.status-idle {
    background: rgba(255, 255, 255, 0.03);
    border-color: var(--border);
    color: var(--text-secondary);
    font-weight: 400;
}

.confidence-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
.confidence-row span.value {
    color: var(--text-primary);
    font-weight: 600;
}

.confidence-track {
    width: 100%;
    height: 4px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
.confidence-fill-success { background: var(--success); }
.confidence-fill-danger { background: var(--danger); }
.confidence-fill-idle { background: rgba(255, 255, 255, 0.18); }

/* ---------- Footer note ---------- */
#footer-note {
    text-align: center;
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    margin-top: 30px !important;
    opacity: 0.7;
}
"""


def _render_result_html(label: str = None, confidence: float = None, error: str = None) -> str:
    if error:
        return f"""
        <div class="status-box status-danger">{error}</div>
        """

    if label is None:
        return """
        <div class="status-box status-idle">Upload an image and click Analyze</div>
        <div>
            <div class="confidence-row"><span>Confidence</span><span class="value">—</span></div>
            <div class="confidence-track"><div class="confidence-fill confidence-fill-idle" style="width: 0%;"></div></div>
        </div>
        """

    is_real = label == "REAL"
    status_class = "status-success" if is_real else "status-danger"
    fill_class = "confidence-fill-success" if is_real else "confidence-fill-danger"
    status_text = "AUTHENTIC IMAGE DETECTED" if is_real else "SCREEN RECAPTURE DETECTED"
    conf_pct = max(0.0, min(100.0, confidence))

    return f"""
    <div class="status-box {status_class}">{status_text}</div>
    <div>
        <div class="confidence-row"><span>Confidence</span><span class="value">{conf_pct:.1f}%</span></div>
        <div class="confidence-track"><div class="confidence-fill {fill_class}" style="width: {conf_pct:.1f}%;"></div></div>
    </div>
    """


def analyze_image(image_path: str) -> str:
    if image_path is None:
        return _render_result_html(error="Please upload an image first.")

    try:
        label, confidence = predict_image(image_path)
        return _render_result_html(label=label, confidence=confidence)
    except Exception as exc:
        traceback.print_exc()
        return _render_result_html(error=f"Prediction failed: {exc}")


with gr.Blocks(
    title="Image Authenticity Detection",
    theme=gr.themes.Base(),
    css=CUSTOM_CSS,
) as demo:

    with gr.Column(elem_id="header-title"):
        gr.Markdown("# Image Authenticity Detection")
    gr.Markdown(
        "Detect whether an image is camera captured or re-captured through a digital screen.",
        elem_id="header-subtitle",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("Upload Image", elem_classes="panel-label")
            image_input = gr.Image(type="filepath", label=None, show_label=False, elem_id="image-input", height=340)
            analyze_button = gr.Button("Analyze", elem_id="analyze-btn")

        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("Result", elem_classes="panel-label")
            result_html = gr.HTML(_render_result_html(), elem_id="result-card")

    gr.Markdown(
        "Built by Anshuman Singh",
        elem_id="footer-note",
    )

    analyze_button.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[result_html],
    )


if __name__ == "__main__":
    demo.launch()