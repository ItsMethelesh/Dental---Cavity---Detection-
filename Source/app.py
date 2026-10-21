from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import streamlit as st

from cavity_detector.inference import load_model_bundle, predict_scan
from cavity_detector.preprocessing import apply_mask_overlay


MODEL_PATH = Path("models/cavity_cnn.pt")
DEFAULT_THRESHOLD = 0.55
MIN_REGION_AREA = 90.0


st.set_page_config(
    page_title="Dental Cavity Detection Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700;800&family=Manrope:wght@400;600;700&display=swap');

        :root {
            --ink: #e8fbff;
            --ink-soft: rgba(223, 244, 255, 0.82);
            --ink-muted: rgba(190, 220, 236, 0.62);
            --cyan: #31d6f7;
            --blue: #2f7cff;
            --glass: rgba(8, 24, 43, 0.62);
            --glass-strong: rgba(7, 20, 38, 0.82);
            --glass-border: rgba(138, 216, 255, 0.22);
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 8%, rgba(49, 214, 247, 0.13), transparent 34%),
                radial-gradient(circle at 88% 80%, rgba(47, 124, 255, 0.15), transparent 36%),
                linear-gradient(145deg, rgba(3, 12, 27, 0.96), rgba(4, 19, 38, 0.95));
            color: var(--ink);
            font-family: "Manrope", sans-serif;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(47, 124, 255, 0.04) 1px, transparent 1px),
                linear-gradient(90deg, rgba(47, 124, 255, 0.04) 1px, transparent 1px);
            background-size: 34px 34px;
        }

        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 1.6rem;
            max-width: 1260px;
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            color: #ecfbff;
            letter-spacing: 0.01em;
        }

        .app-head {
            margin: 0 0 0.72rem 0;
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            background: linear-gradient(130deg, rgba(8, 30, 56, 0.82), rgba(3, 14, 30, 0.80));
            box-shadow: 0 18px 45px rgba(4, 12, 26, 0.44);
            padding: 0.95rem 1rem;
        }

        .head-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.8rem;
        }

        .head-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }

        .brand-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 999px;
            background: radial-gradient(circle at 35% 35%, #8fffff, var(--cyan));
            box-shadow: 0 0 14px rgba(49, 214, 247, 0.85);
            animation: pulse-dot 1.8s ease-in-out infinite;
        }

        .brand-title {
            margin: 0;
            color: #eefcff;
            font-size: 1.12rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .brand-sub {
            margin: 0.12rem 0 0 0;
            font-size: 0.73rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: rgba(152, 220, 249, 0.70);
        }

        .live-badge {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            color: #8feaff;
            border: 1px solid rgba(49, 214, 247, 0.35);
            background: rgba(49, 214, 247, 0.12);
            padding: 0.36rem 0.6rem;
            border-radius: 999px;
            white-space: nowrap;
        }

        .hero-wrap {
            margin: 0.18rem 0 0.78rem 0;
            border: 1px solid var(--glass-border);
            border-radius: 18px;
            background: linear-gradient(140deg, rgba(7, 29, 55, 0.72), rgba(4, 15, 34, 0.58));
            box-shadow: 0 14px 30px rgba(1, 8, 22, 0.35);
            padding: 1.18rem 1.1rem;
        }

        .hero-chip {
            display: inline-block;
            color: #95ecff;
            border: 1px solid rgba(49, 214, 247, 0.36);
            background: rgba(49, 214, 247, 0.13);
            border-radius: 999px;
            font-size: 0.67rem;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            font-weight: 700;
            padding: 0.28rem 0.62rem;
            margin-bottom: 0.62rem;
        }

        .hero-title {
            color: #f2fcff;
            font-size: 1.86rem;
            line-height: 1.16;
            font-weight: 800;
            margin: 0;
        }

        .hero-highlight {
            background: linear-gradient(90deg, #33d6f7, #6f96ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-sub {
            color: rgba(205, 234, 247, 0.72);
            font-size: 0.93rem;
            margin-top: 0.56rem;
            margin-bottom: 0;
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.66rem 0 0.9rem 0;
        }

        .feature-card {
            border: 1px solid rgba(146, 221, 255, 0.21);
            border-radius: 14px;
            padding: 0.78rem;
            background: linear-gradient(150deg, rgba(9, 30, 57, 0.72), rgba(7, 21, 42, 0.46));
        }

        .feature-title {
            margin: 0;
            color: #e6f9ff;
            font-size: 0.87rem;
            font-weight: 700;
        }

        .feature-desc {
            margin: 0.2rem 0 0 0;
            color: rgba(198, 230, 245, 0.61);
            font-size: 0.74rem;
        }

        .glass-card {
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            background: var(--glass);
            padding: 0.75rem;
            margin-bottom: 0.65rem;
        }

        .workspace-band {
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            background: linear-gradient(148deg, rgba(7, 22, 42, 0.72), rgba(4, 14, 29, 0.66));
            padding: 0.82rem 0.94rem;
            margin: 0.24rem 0 0.66rem 0;
        }

        .workspace-title {
            margin: 0;
            font-size: 0.98rem;
            font-weight: 700;
            color: #e8fbff;
        }

        .workspace-sub {
            margin: 0.18rem 0 0 0;
            font-size: 0.79rem;
            color: var(--ink-muted);
        }

        .upload-frame {
            border: 1px solid rgba(146, 217, 255, 0.25);
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(6, 22, 43, 0.66), rgba(4, 15, 33, 0.54));
            padding: 0.72rem 0.84rem 0.44rem 0.84rem;
            margin-bottom: 0.62rem;
        }

        .card-head {
            margin: 0;
            color: #dff6ff;
            font-size: 0.84rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .card-sub {
            margin: 0.22rem 0 0 0;
            color: rgba(190, 224, 239, 0.64);
            font-size: 0.72rem;
        }

        .region-card {
            border: 1px solid rgba(129, 203, 243, 0.24);
            border-radius: 12px;
            background: rgba(8, 28, 51, 0.56);
            padding: 0.62rem;
            margin-bottom: 0.45rem;
        }

        .region-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.4rem;
            margin-bottom: 0.22rem;
        }

        .region-label {
            color: #e2f7ff;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .region-coord {
            color: rgba(183, 216, 231, 0.62);
            font-size: 0.67rem;
            margin: 0;
        }

        .sev-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.45rem;
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            color: #061a31;
        }

        .sev-high { background: #ef5350; color: #fff; }
        .sev-medium { background: #fb923c; }
        .sev-low { background: #facc15; }

        .risk-pill {
            display: inline-block;
            font-weight: 700;
            padding: 0.22rem 0.54rem;
            border-radius: 999px;
            font-size: 0.79rem;
            margin-left: 0.35rem;
        }

        .risk-high { color: #fff; background: #be3030; }
        .risk-medium { color: #fff; background: #cc7d1f; }
        .risk-low { color: #0e2a40; background: #7cdf7b; }

        [data-testid="stMetric"] {
            border: 1px solid rgba(146, 216, 255, 0.24);
            border-radius: 12px;
            background: rgba(9, 31, 57, 0.56);
            text-align: center;
        }

        [data-testid="stMetric"] > div {
            align-items: center;
        }

        [data-testid="stMetricLabel"] {
            color: var(--ink-muted);
            width: 100%;
            display: flex;
            justify-content: center !important;
            text-align: center !important;
        }

        [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {
            width: 100%;
        }

        [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {
            width: 100%;
            text-align: center !important;
        }

        [data-testid="stMetricValue"] {
            color: #ecfbff;
            font-weight: 700;
            justify-content: center;
            text-align: center;
        }

        [data-testid="stMetricDelta"] {
            justify-content: center;
            text-align: center;
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 2px dashed rgba(49, 214, 247, 0.35) !important;
            border-radius: 15px;
            background: rgba(7, 24, 47, 0.56);
            min-height: 136px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(49, 214, 247, 0.72) !important;
            box-shadow: 0 0 24px rgba(49, 214, 247, 0.2);
        }

        [data-testid="stButton"] button {
            border-radius: 10px;
            border: 1px solid rgba(147, 220, 255, 0.3);
            background: linear-gradient(135deg, rgba(49, 214, 247, 0.22), rgba(47, 124, 255, 0.2));
            color: #dff8ff;
            transition: transform 120ms ease, box-shadow 120ms ease;
        }

        [data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(19, 67, 100, 0.34);
        }

        [data-testid="stButton"] button[kind="primary"] {
            border: none;
            color: #0b243b;
            background: linear-gradient(120deg, #43def8, #5d8fff);
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.34rem;
            background: rgba(6, 21, 40, 0.52);
            border: 1px solid rgba(138, 216, 255, 0.19);
            border-radius: 12px;
            padding: 0.3rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            color: rgba(203, 231, 245, 0.76);
            height: 2.15rem;
            background: transparent;
            border: 1px solid transparent;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(125deg, rgba(49, 214, 247, 0.24), rgba(47, 124, 255, 0.26));
            color: #ecfbff;
            border-color: rgba(152, 227, 255, 0.34);
        }

        [data-testid="stStatusWidget"] {
            border: 1px solid rgba(140, 215, 255, 0.25);
            background: rgba(7, 24, 45, 0.65);
            border-radius: 12px;
        }

        [data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, rgba(4, 22, 43, 0.92), rgba(4, 16, 32, 0.90));
            border-right: 1px solid rgba(145, 219, 255, 0.20);
        }

        [data-testid="stSidebar"] * {
            color: #d9f3ff;
        }

        @media (max-width: 1024px) {
            .feature-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 1.56rem;
            }

            .workspace-band {
                margin-top: 0.16rem;
            }
        }

        @keyframes pulse-dot {
            0% { transform: scale(1.0); opacity: 0.72; }
            50% { transform: scale(1.16); opacity: 1; }
            100% { transform: scale(1.0); opacity: 0.72; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <section class="app-head">
            <div class="head-top">
                <div class="head-brand">
                    <span class="brand-dot"></span>
                    <div>
                        <h1 class="brand-title">DentalAI</h1>
                        <p class="brand-sub">Cavity Detection System</p>
                    </div>
                </div>
                <span class="live-badge">AI Engine Live</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-wrap">
            <span class="hero-chip">CNN-Powered Analysis</span>
            <div class="hero-title">Upload a <span class="hero-highlight">Dental X-Ray</span></div>
            <p class="hero-sub">
                AI scans your image for potential cavity regions and returns confidence-based findings.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards() -> None:
    st.markdown(
        """
        <section class="feature-grid">
            <div class="feature-card">
                <p class="feature-title">CNN Detection</p>
                <p class="feature-desc">Deep model guidance for suspicious cavity localization.</p>
            </div>
            <div class="feature-card">
                <p class="feature-title">Instant Results</p>
                <p class="feature-desc">In-browser scan analysis with overlays and confidence data.</p>
            </div>
            <div class="feature-card">
                <p class="feature-title">Severity Scoring</p>
                <p class="feature-desc">High, medium, and low region ranking for quick triage.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_band(has_result: bool) -> None:
    if has_result:
        title = "Analysis Workspace"
        subtitle = (
            "Adjust threshold, inspect masks, and compare detected regions. "
            "Use New Scan when you want to start over."
        )
    else:
        title = "Upload Workspace"
        subtitle = (
            "Drop one dental X-ray image, then run analysis to generate cavity-risk "
            "overlays and confidence scores."
        )

    st.markdown(
        (
            '<section class="workspace-band">'
            f'<p class="workspace-title">{title}</p>'
            f'<p class="workspace-sub">{subtitle}</p>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def ensure_session_state() -> None:
    st.session_state.setdefault("last_scan_result", None)
    st.session_state.setdefault("last_scan_model_loaded", False)
    st.session_state.setdefault("last_scan_signature", None)
    st.session_state.setdefault("last_scan_cam_threshold", DEFAULT_THRESHOLD)
    st.session_state.setdefault("last_auto_request", None)
    st.session_state.setdefault("last_processing_time", 0.0)
    st.session_state.setdefault("last_region_rows", [])


def clear_scan_state() -> None:
    st.session_state["last_scan_result"] = None
    st.session_state["last_scan_model_loaded"] = False
    st.session_state["last_scan_signature"] = None
    st.session_state["last_auto_request"] = None
    st.session_state["last_processing_time"] = 0.0
    st.session_state["last_region_rows"] = []
    if "single_scan_file" in st.session_state:
        st.session_state["single_scan_file"] = None


def format_risk_label(risk_score: float) -> str:
    if risk_score >= 0.75:
        return "High"
    if risk_score >= 0.45:
        return "Medium"
    return "Low"


def severity_from_confidence(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.50:
        return "medium"
    return "low"


def severity_css(severity: str) -> str:
    return {
        "high": "sev-high",
        "medium": "sev-medium",
        "low": "sev-low",
    }.get(severity, "sev-low")


def risk_pill(label: str) -> str:
    css_class = {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }[label]
    return f'<span class="risk-pill {css_class}">{label} Risk</span>'


@st.cache_resource(show_spinner=False)
def cached_load_model(model_path: str):
    return load_model_bundle(model_path)


def resolve_uploaded_image(key_prefix: str = "scan") -> Path | None:
    uploaded = st.file_uploader(
        "Upload dental X-ray",
        type=["png", "jpg", "jpeg"],
        key=f"{key_prefix}_file",
    )
    if uploaded is None:
        return None

    payload = uploaded.getvalue()
    if not payload:
        return None

    temp_dir = Path(".tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded.name).suffix.lower() or ".png"
    digest = hashlib.md5(payload).hexdigest()
    cached_upload_path = temp_dir / f"{key_prefix}_{digest}{suffix}"
    if not cached_upload_path.exists():
        cached_upload_path.write_bytes(payload)
    return cached_upload_path


def count_regions(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def build_threshold_preview(result: dict, threshold: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    heatmap = np.asarray(result["heatmap"], dtype=np.float32)
    candidate_mask = np.asarray(result["candidate_mask"], dtype=np.uint8)

    cam_mask = np.uint8(heatmap >= threshold) * 255
    combined_mask = cv2.bitwise_or(candidate_mask, cam_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    preview_overlay = apply_mask_overlay(
        result["original_rgb"],
        combined_mask,
        color=(228, 42, 42),
        alpha=0.50,
    )
    lesion_pixels = int(np.count_nonzero(combined_mask))
    preview_regions = count_regions(combined_mask)
    return preview_overlay, combined_mask, lesion_pixels, preview_regions


def encode_rgb_image_png(image_rgb: np.ndarray) -> bytes | None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        return None
    return encoded.tobytes()


def extract_region_rows(result: dict) -> list[dict]:
    mask = np.asarray(result["combined_mask"], dtype=np.uint8)
    heatmap = np.asarray(result["heatmap"], dtype=np.float32)
    fallback_conf = min(1.0, float(result["opencv_score"]) * 9.0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region_rows: list[dict] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_REGION_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi_mask = mask[y : y + h, x : x + w] > 0
        roi_heatmap = heatmap[y : y + h, x : x + w]
        if roi_mask.any() and float(np.max(roi_heatmap)) > 0.0:
            confidence = float(np.mean(roi_heatmap[roi_mask]))
        else:
            confidence = fallback_conf
        confidence = float(np.clip(confidence, 0.0, 1.0))

        region_rows.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "confidence": confidence,
                "severity": severity_from_confidence(confidence),
            }
        )

    region_rows.sort(key=lambda row: row["confidence"], reverse=True)
    for idx, row in enumerate(region_rows, start=1):
        row["region"] = idx
    return region_rows


def draw_detection_boxes(image_rgb: np.ndarray, region_rows: list[dict]) -> np.ndarray:
    if not region_rows:
        return image_rgb

    rendered = image_rgb.copy()
    color_map = {
        "high": (239, 68, 68),
        "medium": (251, 146, 60),
        "low": (250, 204, 21),
    }

    for row in region_rows:
        x = int(row["x"])
        y = int(row["y"])
        w = int(row["width"])
        h = int(row["height"])
        severity = str(row["severity"])
        color = color_map.get(severity, (250, 204, 21))
        label = f"{int(row['confidence'] * 100)}% {severity.upper()}"

        cv2.rectangle(rendered, (x, y), (x + w, y + h), color, 2)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        box_w = text_size[0] + 10
        box_h = text_size[1] + 8
        y_top = max(0, y - box_h - 2)
        cv2.rectangle(rendered, (x, y_top), (x + box_w, y_top + box_h), color, -1)
        cv2.putText(
            rendered,
            label,
            (x + 5, y_top + box_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (5, 18, 34),
            1,
            cv2.LINE_AA,
        )

    return rendered


def summarize_regions(region_rows: list[dict]) -> tuple[int, int, int, float]:
    high = sum(1 for row in region_rows if row["severity"] == "high")
    medium = sum(1 for row in region_rows if row["severity"] == "medium")
    low = sum(1 for row in region_rows if row["severity"] == "low")
    overall_conf = (
        float(np.mean([float(row["confidence"]) for row in region_rows])) if region_rows else 0.0
    )
    return high, medium, low, overall_conf


def render_summary_panel(region_rows: list[dict], processing_time: float) -> None:
    high, medium, low, overall_conf = summarize_regions(region_rows)

    st.markdown(
        """
        <section class="glass-card">
            <p class="card-head">Analysis Summary</p>
            <p class="card-sub">Confidence and severity split across detected regions.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.metric("Overall Confidence", f"{overall_conf * 100:.1f}%")
    st.progress(float(np.clip(overall_conf, 0.0, 1.0)))
    c1, c2, c3 = st.columns(3)
    c1.metric("High", high)
    c2.metric("Medium", medium)
    c3.metric("Low", low)
    st.metric("Processing Time", f"{processing_time:.2f}s")

    st.markdown(
        """
        <section class="glass-card">
            <p class="card-head">Detected Regions</p>
            <p class="card-sub">Bounding-box locations and confidence per region.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if not region_rows:
        st.success("No cavity regions detected in this scan.")
        return

    for row in region_rows[:6]:
        sev = str(row["severity"])
        st.markdown(
            (
                '<section class="region-card">'
                '<div class="region-top">'
                f'<span class="region-label">Region {row["region"]}</span>'
                f'<span class="sev-pill {severity_css(sev)}">{sev}</span>'
                "</div>"
                f'<p class="region-coord">({row["x"]}, {row["y"]}) | '
                f'{row["width"]}x{row["height"]} px</p>'
                "</section>"
            ),
            unsafe_allow_html=True,
        )
        st.progress(float(np.clip(row["confidence"], 0.0, 1.0)))

    rows = [
        {
            "region": row["region"],
            "severity": str(row["severity"]).upper(),
            "confidence_percent": round(float(row["confidence"]) * 100.0, 1),
            "x": row["x"],
            "y": row["y"],
            "width": row["width"],
            "height": row["height"],
            "area_px": row["area"],
        }
        for row in region_rows
    ]
    with st.expander("Region table", expanded=False):
        st.dataframe(rows, use_container_width=True, hide_index=True)


def run_analysis(selected_path: Path, cam_threshold: float) -> tuple[dict, bool, float, list[dict]]:
    start_time = perf_counter()
    model_bundle = cached_load_model(str(MODEL_PATH))
    result = predict_scan(
        selected_path,
        model_bundle=model_bundle,
        cam_threshold=float(cam_threshold),
    )
    elapsed = perf_counter() - start_time
    region_rows = extract_region_rows(result)
    model_loaded = model_bundle is not None
    return result, model_loaded, elapsed, region_rows


def show_prediction(
    result: dict,
    model_loaded: bool,
    default_cam_threshold: float,
    region_rows: list[dict],
    processing_time: float,
    show_boxes: bool,
) -> None:
    risk_score = float(result["risk_score"])
    risk_label = format_risk_label(risk_score)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Score", f"{risk_score * 100:.1f}%")
    c2.metric("Risk Band", risk_label)
    c3.metric("Regions", int(result["region_count"]))
    c4.metric("Cavity Area", f"{float(result['area_ratio']) * 100:.2f}%")
    c5.metric("Time", f"{processing_time:.2f}s")

    st.markdown(
        (
            f'<div class="glass-card">'
            f"<p class='card-head'>Interpretation {risk_pill(risk_label)}</p>"
            f"<p class='card-sub'>OpenCV structural signal: {float(result['opencv_score']) * 100:.2f}%</p>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )

    if risk_label == "High":
        st.error("High-risk signal detected. Review highlighted areas carefully.")
    elif risk_label == "Medium":
        st.warning("Moderate-risk signal detected. Follow-up review is advised.")
    else:
        st.success("Low-risk signal for this scan.")

    model_prob = result.get("probability")
    if model_prob is None:
        st.info(
            "Model checkpoint not loaded. Analysis is running in OpenCV fallback mode."
        )
    elif model_loaded:
        st.caption(f"CNN cavity probability: {float(model_prob) * 100:.2f}%")

    left_col, right_col = st.columns([2.05, 1.0], gap="large")
    widget_prefix = hashlib.md5(str(result["image_path"]).encode("utf-8")).hexdigest()[:10]

    with left_col:
        view_tab, threshold_tab, mask_tab, signal_tab = st.tabs(
            ["Analysis View", "Threshold Lab", "Masks", "Signals"]
        )

        with view_tab:
            layers = {
                "Final Highlighted Output": result["combined_overlay"],
                "CNN Heatmap Overlay": result["heatmap_overlay"],
                "OpenCV Candidate Regions": result["candidate_overlay"],
                "Original X-ray": result["original_rgb"],
            }
            selected_layer = st.radio(
                "View Layer",
                options=list(layers.keys()),
                horizontal=True,
                key=f"{widget_prefix}_layer",
            )

            image_to_show = layers[selected_layer]
            if show_boxes:
                image_to_show = draw_detection_boxes(image_to_show, region_rows)

            st.image(image_to_show, caption=selected_layer, use_container_width=True)

            download_data = encode_rgb_image_png(image_to_show)
            st.download_button(
                "Download current view (PNG)",
                data=download_data,
                file_name="dental_scan_analysis.png",
                mime="image/png",
                disabled=download_data is None,
                key=f"{widget_prefix}_download",
            )

        with threshold_tab:
            if model_prob is None:
                st.info("Threshold Lab is available when a trained CNN checkpoint is present.")
            else:
                preview_threshold = st.slider(
                    "Adjust heatmap threshold",
                    min_value=0.35,
                    max_value=0.85,
                    value=float(default_cam_threshold),
                    step=0.01,
                    key=f"{widget_prefix}_preview_threshold",
                )
                preview_overlay, preview_mask, preview_pixels, preview_regions = build_threshold_preview(
                    result, threshold=preview_threshold
                )

                col_a, col_b = st.columns(2)
                col_a.image(
                    result["combined_overlay"],
                    caption=f"Current Output ({default_cam_threshold:.2f})",
                    use_container_width=True,
                )
                col_b.image(
                    preview_overlay,
                    caption=f"Preview Output ({preview_threshold:.2f})",
                    use_container_width=True,
                )

                baseline_pixels = int(np.count_nonzero(result["combined_mask"]))
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Preview Regions",
                    preview_regions,
                    delta=preview_regions - int(result["region_count"]),
                )
                m2.metric(
                    "Preview Lesion Pixels",
                    preview_pixels,
                    delta=preview_pixels - baseline_pixels,
                )
                m3.metric(
                    "Threshold Shift",
                    f"{preview_threshold - default_cam_threshold:+.2f}",
                )
                st.image(
                    preview_mask,
                    caption="Preview combined mask",
                    clamp=True,
                    use_container_width=True,
                )

        with mask_tab:
            mcol1, mcol2 = st.columns(2)
            mcol1.image(
                result["candidate_mask"],
                caption="OpenCV candidate mask",
                clamp=True,
                use_container_width=True,
            )
            mcol2.image(
                result["combined_mask"],
                caption="Final combined mask",
                clamp=True,
                use_container_width=True,
            )

        with signal_tab:
            st.subheader("Score Breakdown")
            heuristic_proxy = min(1.0, float(result["opencv_score"]) * 9.0)
            st.caption(f"OpenCV heuristic proxy: {heuristic_proxy * 100:.1f}%")
            st.progress(float(np.clip(heuristic_proxy, 0.0, 1.0)))

            if model_prob is not None:
                st.caption(f"CNN probability: {float(model_prob) * 100:.1f}%")
                st.progress(float(np.clip(float(model_prob), 0.0, 1.0)))

            score_rows = [
                {"signal": "Final risk score", "value_percent": round(risk_score * 100, 2)},
                {"signal": "OpenCV raw score", "value_percent": round(float(result["opencv_score"]) * 100, 2)},
                {"signal": "Cavity area ratio", "value_percent": round(float(result["area_ratio"]) * 100, 2)},
            ]
            if model_prob is not None:
                score_rows.append(
                    {
                        "signal": "CNN probability",
                        "value_percent": round(float(model_prob) * 100, 2),
                    }
                )
            st.dataframe(score_rows, use_container_width=True, hide_index=True)

    with right_col:
        render_summary_panel(region_rows=region_rows, processing_time=processing_time)
        st.markdown(
            """
            <section class="glass-card">
                <p class="card-head">Medical Disclaimer</p>
                <p class="card-sub">
                    AI output is for screening support only and does not replace professional dental diagnosis.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_styles()
    ensure_session_state()
    has_result = st.session_state.get("last_scan_result") is not None

    top_left, top_right = st.columns([4.4, 1.2], vertical_alignment="center")
    with top_left:
        render_app_header()
    with top_right:
        if st.button(
            "New Scan",
            use_container_width=True,
            disabled=not has_result,
        ):
            clear_scan_state()
            st.rerun()

    if not has_result:
        render_hero()
        render_feature_cards()
    render_workspace_band(has_result=has_result)

    with st.sidebar:
        st.subheader("Analysis Settings")
        cam_threshold = st.slider(
            "Heatmap Threshold",
            min_value=0.35,
            max_value=0.85,
            value=DEFAULT_THRESHOLD,
            step=0.01,
        )
        auto_analyze = st.toggle(
            "Auto analyze when image or threshold changes",
            value=False,
            key="scan_auto",
        )
        show_boxes = st.toggle(
            "Show detection boxes",
            value=True,
            key="scan_boxes",
        )

        if MODEL_PATH.exists():
            st.caption(f"Model found: {MODEL_PATH}")
        else:
            st.caption(f"Model not found: {MODEL_PATH}")
        st.caption("Training runs in backend only. This UI handles upload and analysis.")

    st.markdown(
        """
        <section class="upload-frame">
            <p class="workspace-title">Image Input</p>
            <p class="workspace-sub">Upload one X-ray and run analysis. Supported formats: JPG, PNG, JPEG.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    upload_col, action_col = st.columns([2.5, 1.0], vertical_alignment="bottom")
    with upload_col:
        selected_path = resolve_uploaded_image(key_prefix="single_scan")
    with action_col:
        run_clicked = st.button("Analyze Image", type="primary", use_container_width=True, key="scan_run")

    request_signature = None
    if selected_path is not None:
        request_signature = f"{selected_path}|{float(cam_threshold):.2f}"

    should_auto_run = (
        auto_analyze
        and request_signature is not None
        and st.session_state.get("last_auto_request") != request_signature
    )
    run_inference = run_clicked or should_auto_run

    if run_inference:
        if selected_path is None:
            st.warning("Upload an image first.")
        else:
            try:
                with st.status("Running dental AI analysis...", expanded=True) as status:
                    st.write("Preprocessing image")
                    st.write("Running model inference")
                    st.write("Generating cavity maps and regions")
                    result, model_loaded, elapsed, region_rows = run_analysis(
                        selected_path,
                        cam_threshold=cam_threshold,
                    )
                    status.update(label="Analysis complete", state="complete")

                st.session_state["last_scan_result"] = result
                st.session_state["last_scan_model_loaded"] = model_loaded
                st.session_state["last_scan_signature"] = request_signature
                st.session_state["last_scan_cam_threshold"] = float(cam_threshold)
                st.session_state["last_auto_request"] = request_signature
                st.session_state["last_processing_time"] = float(elapsed)
                st.session_state["last_region_rows"] = region_rows
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unable to analyze image: {exc}")

    last_result = st.session_state.get("last_scan_result")
    if last_result is None:
        st.info("Upload a dental X-ray, then click Analyze Image to view results.")
        return

    if (
        request_signature is not None
        and st.session_state.get("last_scan_signature") != request_signature
        and not auto_analyze
    ):
        st.caption("Input changed. Click Analyze Image to refresh results.")

    show_prediction(
        last_result,
        model_loaded=bool(st.session_state.get("last_scan_model_loaded")),
        default_cam_threshold=float(st.session_state.get("last_scan_cam_threshold", cam_threshold)),
        region_rows=list(st.session_state.get("last_region_rows", [])),
        processing_time=float(st.session_state.get("last_processing_time", 0.0)),
        show_boxes=bool(show_boxes),
    )


if __name__ == "__main__":
    main()
