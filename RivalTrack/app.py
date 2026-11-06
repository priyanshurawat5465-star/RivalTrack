import re
import time
import pickle
from pathlib import Path
from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="RivalTrack | Competitive Quality & ABSA Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.absa_engine import ABSAEngine, extract_dynamic_aspects_from_clause, score_clause_polarity
from src.live_scraper import scrape_live_product
from src.nlp_pipeline import clean_review_text, is_valid_review

PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================================
# CACHED RESOURCES & DATA LOADERS
# ==========================================
@st.cache_resource(show_spinner=False)
def load_absa_engine(version: str = "v6.0_enterprise_ui"):
    return ABSAEngine()


@st.cache_data(show_spinner=False)
def load_corpus_metrics(version: str = "v6.0_clean"):
    data_path = PROJECT_ROOT / "data" / "cleaned_absa_corpus.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    source_col = "review_text" if "review_text" in df.columns else df.columns[0]
    df = df[df[source_col].apply(is_valid_review)].reset_index(drop=True)
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(1).astype(int)
    df["category"] = df["category"].astype(str).str.strip().str.title()
    return df


@st.cache_data(show_spinner=False)
def compute_historical_category_pareto(category: str, _df_corpus: pd.DataFrame) -> dict:
    df_defects = _df_corpus[(_df_corpus["category"] == category) & (_df_corpus["sentiment"] == 0)]
    corpus_counter = Counter()
    if not df_defects.empty:
        sample_defects = df_defects["review_text"].dropna().astype(str).head(400)
        for txt in sample_defects:
            pol = score_clause_polarity(txt)
            for asp in extract_dynamic_aspects_from_clause(txt, pol["implicit_aspect"]):
                corpus_counter[asp] += 1
    return dict(corpus_counter)


# ==========================================
# PERSISTENT SESSION STATE INITIALIZATION
# ==========================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

if "active_module" not in st.session_state:
    st.session_state.active_module = "Clause-Level ABSA Inspector"

if "sentence_input_text" not in st.session_state:
    st.session_state.sentence_input_text = (
        "The display is gorgeous but the btry drains super fast and the mic is garbage."
    )

if "sentence_analysis_state" not in st.session_state:
    st.session_state.sentence_analysis_state = None

if "live_url_input_text" not in st.session_state:
    st.session_state.live_url_input_text = ""

if "live_scrape_state" not in st.session_state:
    st.session_state.live_scrape_state = None

if "h2h_url_a_text" not in st.session_state:
    st.session_state.h2h_url_a_text = ""

if "h2h_url_b_text" not in st.session_state:
    st.session_state.h2h_url_b_text = ""

if "h2h_comparison_state" not in st.session_state:
    st.session_state.h2h_comparison_state = None

if "selected_pareto_category" not in st.session_state:
    st.session_state.selected_pareto_category = None


# ==========================================
# DYNAMIC HIGH-CONTRAST THEME ENGINE
# ==========================================
def get_theme_palette(mode: str) -> dict:
    if mode == "Dark":
        return {
            "bg_main": "#0B0F19",
            "bg_secondary": "#111827",
            "bg_card": "#161F33",
            "bg_input": "#0F172A",
            "text_primary": "#F8FAFC",
            "text_secondary": "#94A3B8",
            "border": "#26334D",
            "accent_primary": "#3B82F6",
            "accent_secondary": "#6366F1",
            "danger": "#F43F5E",
            "danger_bg": "rgba(244, 63, 94, 0.12)",
            "success": "#10B981",
            "success_bg": "rgba(16, 185, 129, 0.12)",
            "warning": "#F59E0B",
            "warning_bg": "rgba(245, 158, 11, 0.12)",
            "plotly_template": "plotly_dark",
            "grid_color": "#1E293B",
            "bar_primary": "#F43F5E",
            "bar_secondary": "#3B82F6",
            "line_cum": "#38BDF8"
        }
    else:
        return {
            "bg_main": "#F8FAFC",
            "bg_secondary": "#F1F5F9",
            "bg_card": "#FFFFFF",
            "bg_input": "#FFFFFF",
            "text_primary": "#0F172A",
            "text_secondary": "#475569",
            "border": "#CBD5E1",
            "accent_primary": "#1D4ED8",
            "accent_secondary": "#4338CA",
            "danger": "#E11D48",
            "danger_bg": "#FFE4E6",
            "success": "#059669",
            "success_bg": "#D1FAE5",
            "warning": "#D97706",
            "warning_bg": "#FEF3C7",
            "plotly_template": "plotly_white",
            "grid_color": "#E2E8F0",
            "bar_primary": "#E11D48",
            "bar_secondary": "#1D4ED8",
            "line_cum": "#0284C7"
        }


def inject_enterprise_css(p: dict):
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {p['text_primary']};
        }}

        /* Main App & Sidebar Backgrounds */
        .stApp {{
            background-color: {p['bg_main']};
            color: {p['text_primary']};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {p['bg_secondary']};
            border-right: 1px solid {p['border']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {p['text_primary']};
        }}

        /* Clean Header Bar */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        .block-container {{
            padding-top: 1.8rem !important;
            padding-bottom: 3rem !important;
            max-width: 1440px;
        }}

        /* Input Controls (Text Areas, Text Inputs, Selectboxes) */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        div[data-baseweb="select"] > div {{
            background-color: {p['bg_input']} !important;
            color: {p['text_primary']} !important;
            border: 1px solid {p['border']} !important;
            border-radius: 8px !important;
            font-size: 0.94rem !important;
        }}
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {p['accent_primary']} !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
        }}
        label, .stWidgetLabel p {{
            color: {p['text_primary']} !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.01em;
        }}

        /* Primary Action Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, {p['accent_primary']}, {p['accent_secondary']}) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.55rem 1.35rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 0.01em;
            transition: all 0.18s ease-in-out !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35) !important;
            opacity: 0.96 !important;
        }}

        /* Persistent Module Navigation Bar (Radio Horizontal) */
        div[role="radiogroup"] {{
            background-color: {p['bg_secondary']};
            padding: 6px;
            border-radius: 10px;
            border: 1px solid {p['border']};
            gap: 6px;
            display: flex;
            flex-wrap: wrap;
        }}
        div[role="radiogroup"] > label {{
            background-color: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 7px !important;
            padding: 8px 16px !important;
            margin: 0 !important;
            transition: all 0.15s ease;
            cursor: pointer;
        }}
        div[role="radiogroup"] > label:hover {{
            background-color: {p['bg_card']} !important;
            border-color: {p['border']} !important;
        }}
        div[role="radiogroup"] > label[data-checked="true"],
        div[role="radiogroup"] > label:has(input:checked) {{
            background-color: {p['accent_primary']} !important;
            border-color: {p['accent_primary']} !important;
        }}
        div[role="radiogroup"] > label:has(input:checked) p {{
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }}

        /* Progress Bar Styling */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {p['accent_primary']}, {p['success']}) !important;
        }}

        /* Custom Enterprise Cards */
        .rt-header-banner {{
            background-color: {p['bg_card']};
            border: 1px solid {p['border']};
            border-left: 5px solid {p['accent_primary']};
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }}
        .rt-title {{
            font-size: 1.55rem;
            font-weight: 700;
            color: {p['text_primary']};
            margin: 0;
            letter-spacing: -0.02em;
        }}
        .rt-subtitle {{
            font-size: 0.88rem;
            color: {p['text_secondary']};
            margin-top: 0.25rem;
            margin-bottom: 0;
        }}
        .rt-status-pill {{
            background-color: {p['success_bg']};
            color: {p['success']};
            border: 1px solid {p['success']};
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            font-weight: 600;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            white-space: nowrap;
        }}
        .rt-card {{
            background-color: {p['bg_card']};
            border: 1px solid {p['border']};
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        }}
        .rt-kpi-card {{
            background-color: {p['bg_card']};
            border: 1px solid {p['border']};
            border-top: 4px solid {p['accent_primary']};
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
            height: 100%;
        }}
        .rt-kpi-label {{
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {p['text_secondary']};
            margin-bottom: 0.4rem;
        }}
        .rt-kpi-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 700;
            color: {p['text_primary']};
            margin: 0;
        }}
        .rt-kpi-sub {{
            font-size: 0.82rem;
            color: {p['text_secondary']};
            margin-top: 0.35rem;
        }}
        .rt-verdict-banner {{
            background-color: {p['bg_card']};
            border: 1px solid {p['border']};
            border-left: 5px solid {p['success']};
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 1rem 0;
            font-size: 0.95rem;
            color: {p['text_primary']};
        }}
        .rt-section-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {p['text_primary']};
            margin-bottom: 0.25rem;
            letter-spacing: -0.01em;
        }}
        .rt-section-desc {{
            font-size: 0.86rem;
            color: {p['text_secondary']};
            margin-bottom: 1.1rem;
        }}
        .rt-sidebar-stat {{
            background-color: {p['bg_card']};
            border: 1px solid {p['border']};
            border-radius: 8px;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.6rem;
        }}
        .rt-sidebar-stat-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {p['text_secondary']};
            font-weight: 600;
        }}
        .rt-sidebar-stat-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            color: {p['text_primary']};
            margin-top: 2px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def render_kpi_card(label: str, value: str, subtext: str = "", accent_color: str = "#3B82F6"):
    st.markdown(
        f"""
        <div class="rt-kpi-card" style="border-top-color: {accent_color};">
            <div class="rt-kpi-label">{label}</div>
            <div class="rt-kpi-value" style="color: {accent_color};">{value}</div>
            <div class="rt-kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pareto_chart(aspect_counts: Counter, title: str, palette: dict):
    """Renders a high-contrast Vertical Bar + Cumulative Line 80/20 Pareto Chart."""
    cleaned_counts = Counter(aspect_counts)
    if len(cleaned_counts) > 1 and "Overall Build / Quality" in cleaned_counts:
        del cleaned_counts["Overall Build / Quality"]

    if not cleaned_counts:
        st.info("No recurring component defects detected in this sample.")
        return

    top_items = cleaned_counts.most_common(10)
    df_tally = pd.DataFrame(top_items, columns=["Component", "Count"])
    df_tally["Cumulative %"] = (df_tally["Count"].cumsum() / df_tally["Count"].sum()) * 100.0

    x_labels = df_tally["Component"].astype(str).tolist()
    y_counts = [int(v) for v in df_tally["Count"].tolist()]
    y_cum = [round(float(v), 1) for v in df_tally["Cumulative %"].tolist()]
    max_count = max(y_counts) if y_counts else 10

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_counts,
        name="Defect Mentions (Count)",
        marker=dict(
            color=palette["bar_primary"],
            line=dict(color=palette["border"], width=1)
        ),
        orientation="v",
        text=y_counts,
        textposition="outside",
        textfont=dict(color=palette["text_primary"], size=12, family="JetBrains Mono")
    ))

    fig.add_trace(go.Scatter(
        x=x_labels,
        y=y_cum,
        name="Cumulative Share (%)",
        yaxis="y2",
        mode="lines+markers+text",
        text=[f"{v:.0f}%" for v in y_cum],
        textposition="top center",
        textfont=dict(color=palette["line_cum"], size=11, family="JetBrains Mono"),
        line=dict(color=palette["line_cum"], width=3),
        marker=dict(size=8, color=palette["line_cum"], line=dict(width=1.5, color=palette["bg_card"]))
    ))

    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color=palette["warning"],
        annotation_text="80% Pareto Threshold (Vital Few)",
        annotation_position="bottom right",
        annotation_font=dict(color=palette["warning"], size=11),
        yref="y2"
    )

    fig.update_layout(
        template=palette["plotly_template"],
        paper_bgcolor=palette["bg_card"],
        plot_bgcolor=palette["bg_card"],
        font=dict(family="Inter, sans-serif", color=palette["text_primary"]),
        title=dict(text=title, font=dict(size=16, color=palette["text_primary"])),
        xaxis=dict(
            title="Defective Product Component",
            type="category",
            gridcolor=palette["grid_color"],
            tickfont=dict(color=palette["text_primary"])
        ),
        yaxis=dict(
            title="Defect Complaint Count",
            range=[0, max_count * 1.28],
            gridcolor=palette["grid_color"],
            tickfont=dict(color=palette["text_primary"])
        ),
        yaxis2=dict(
            title="Cumulative Share (%)",
            overlaying="y",
            side="right",
            range=[0, 112],
            showgrid=False,
            tickfont=dict(color=palette["line_cum"])
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=palette["text_primary"])
        ),
        margin=dict(l=40, r=40, t=65, b=50),
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# INITIALIZE ENGINE & CORPUS
# ==========================================
try:
    engine = load_absa_engine()
    engine_loaded = True
except Exception as e:
    st.error(f"Failed to initialize ABSA Engine: {e}")
    engine_loaded = False

df_corpus = load_corpus_metrics()


# ==========================================
# SIDEBAR: THEME TOGGLE & SYSTEM TELEMETRY
# ==========================================
with st.sidebar:
    st.markdown("### WORKSPACE SETTINGS")
    selected_theme = st.radio(
        "Interface Theme Mode",
        options=["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True,
        key="sidebar_theme_selector"
    )
    if selected_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = selected_theme
        st.rerun()

    palette = get_theme_palette(st.session_state.theme_mode)
    inject_enterprise_css(palette)

    if st.button("Reset Cache & Session", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.sentence_analysis_state = None
        st.session_state.live_scrape_state = None
        st.session_state.h2h_comparison_state = None
        st.rerun()

    st.markdown("---")
    st.markdown("### CORPUS TELEMETRY")
    if not df_corpus.empty:
        total_reviews = len(df_corpus)
        total_defects = len(df_corpus[df_corpus["sentiment"] == 0])
        defect_ratio = (total_defects / total_reviews) * 100.0

        st.markdown(
            f"""
            <div class="rt-sidebar-stat">
                <div class="rt-sidebar-stat-label">Verified Corpus Volume</div>
                <div class="rt-sidebar-stat-val">{total_reviews:,}</div>
            </div>
            <div class="rt-sidebar-stat">
                <div class="rt-sidebar-stat-label">Indexed Defect Records</div>
                <div class="rt-sidebar-stat-val" style="color: {palette['danger']};">{total_defects:,}</div>
            </div>
            <div class="rt-sidebar-stat">
                <div class="rt-sidebar-stat-label">Baseline Defect Rate</div>
                <div class="rt-sidebar-stat-val" style="color: {palette['warning']};">{defect_ratio:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Domain Distribution")
        cat_counts = df_corpus["category"].value_counts().reset_index()
        cat_counts.columns = ["Domain", "Reviews"]
        st.dataframe(cat_counts, hide_index=True, use_container_width=True)


# Ensure CSS is injected even if sidebar is collapsed
palette = get_theme_palette(st.session_state.theme_mode)
inject_enterprise_css(palette)


# ==========================================
# TOP EXECUTIVE HEADER & PERSISTENT NAVIGATION
# ==========================================
st.markdown(
    """
    <div class="rt-header-banner">
        <div>
            <h1 class="rt-title">RivalTrack Intelligence Platform</h1>
            <p class="rt-subtitle">
                2-Tier Stacking Ensemble &bull; Clause-Level Compositional ABSA &bull; Automated Competitor Defect Benchmarking
            </p>
        </div>
        <div class="rt-status-pill">STACKING ENSEMBLE: ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True
)

MODULES = [
    "Clause-Level ABSA Inspector",
    "Live Product URL Intelligence",
    "Historical Pareto Analytics",
    "Competitor Head-to-Head",
    "Ensemble Architecture & Metrics"
]

active_mod = st.radio(
    "Select Analytics Module",
    options=MODULES,
    index=MODULES.index(st.session_state.active_module),
    horizontal=True,
    label_visibility="collapsed",
    key="active_module"
)

st.write("")


# ==========================================
# MODULE 1: CLAUSE-LEVEL ABSA INSPECTOR
# ==========================================
if active_mod == "Clause-Level ABSA Inspector":
    st.markdown(
        """
        <div class="rt-card">
            <div class="rt-section-title">Clause-Level Aspect-Based Sentiment Inspector</div>
            <div class="rt-section-desc">
                Decomposes compound customer reviews across contrastive discourse pivots, evaluates syntactic negation windows,
                and fuses compositional polarity with the 2-Tier Stacking Ensemble.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    sample_input = st.text_area(
        "Customer Review Input",
        value=st.session_state.sentence_input_text,
        height=105
    )

    col_btn, col_hint = st.columns([1, 4])
    with col_btn:
        run_sentence_btn = st.button("Execute ABSA Analysis", use_container_width=True)

    if run_sentence_btn and engine_loaded:
        st.session_state.sentence_input_text = sample_input
        prog_bar = st.progress(0, text="Initializing NLP normalizer & contraction expansion...")
        time.sleep(0.08)
        prog_bar.progress(35, text="Segmenting contrastive clauses & syntactic negation windows...")
        time.sleep(0.08)
        prog_bar.progress(75, text="Querying Layer-1 Base Learners & Layer-2 Logistic Meta-Learner...")
        analysis_out = engine.analyze(sample_input)
        prog_bar.progress(100, text="Clause-level inference complete.")
        time.sleep(0.12)
        prog_bar.empty()
        st.session_state.sentence_analysis_state = {
            "result": analysis_out,
            "cleaned": clean_review_text(sample_input)
        }

    # Render persisted result (never vanishes when switching modules)
    if st.session_state.sentence_analysis_state is not None:
        saved = st.session_state.sentence_analysis_state
        res = saved["result"]

        c1, c2, c3 = st.columns(3)
        with c1:
            is_def = res["sentiment"] == "Defect"
            render_kpi_card(
                label="Predicted Classification",
                value=res["sentiment"].upper(),
                subtext="Negative component defect flagged" if is_def else "No component failures detected",
                accent_color=palette["danger"] if is_def else palette["success"]
            )
        with c2:
            render_kpi_card(
                label="Ensemble Confidence",
                value=f"{res['confidence']:.1%}",
                subtext="Hybrid Compositional + Meta-Learner probability",
                accent_color=palette["accent_primary"]
            )
        with c3:
            defects_str = ", ".join(res["defective_aspects"]) if res["defective_aspects"] else "None Detected"
            render_kpi_card(
                label="Isolated Failing Components",
                value=defects_str,
                subtext="Extracted strictly from negative clauses",
                accent_color=palette["warning"] if res["defective_aspects"] else palette["success"]
            )

        st.write("")
        st.markdown("#### Clause-by-Clause Decomposition & Polarity Matrix")
        st.code(f"Normalized Lemmatized Tokens: {saved['cleaned']}", language="text")

        clause_rows = []
        for idx, c_info in enumerate(res.get("clause_breakdown", []), 1):
            clause_rows.append({
                "Clause #": idx,
                "Atomic Clause Segment": c_info["clause"],
                "Post-Contrast Pivot": "Yes" if c_info["post_contrast"] else "No",
                "Compositional Score": round(c_info["ling_score"], 2),
                "Stacking ML Prob (Pos)": f"{c_info['ml_pos_prob']:.1%}",
                "Clause Verdict": (
                    "Defect (Negative)" if c_info["is_negative"]
                    else ("Positive" if c_info["is_positive"] else "Neutral")
                )
            })
        if clause_rows:
            st.dataframe(pd.DataFrame(clause_rows), hide_index=True, use_container_width=True)


# ==========================================
# MODULE 2: LIVE PRODUCT URL INTELLIGENCE
# ==========================================
elif active_mod == "Live Product URL Intelligence":
    st.markdown(
        """
        <div class="rt-card">
            <div class="rt-section-title">Live E-Commerce Product URL Intelligence</div>
            <div class="rt-section-desc">
                Extracts multi-page customer reviews from Flipkart or Amazon product URLs, performs batch Clause-Level ABSA,
                and constructs an 80/20 Component Defect Pareto distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    url_input = st.text_input(
        "Target Product URL (Flipkart / Amazon)",
        value=st.session_state.live_url_input_text,
        placeholder="https://www.flipkart.com/..."
    )

    run_scrape_btn = st.button("Extract & Analyze Product Reviews")

    if run_scrape_btn and url_input and engine_loaded:
        st.session_state.live_url_input_text = url_input
        prog = st.progress(0, text="Connecting to target e-commerce endpoint & initializing session cookies...")
        time.sleep(0.1)
        prog.progress(20, text="Paginating across review pages & extracting SSR JSON / DOM payloads...")

        scrape_res = scrape_live_product(url_input, max_pages=10, target_reviews=55)

        if not scrape_res["dataframe"].empty:
            df_live = scrape_res["dataframe"].copy()
            total_rows = len(df_live)

            sentiments = []
            confidences = []
            aspects_list = []
            live_defect_counter = Counter()

            for i, txt in enumerate(df_live["review_text"]):
                pct = 30 + int(((i + 1) / total_rows) * 65)
                prog.progress(min(pct, 96), text=f"Running Clause-Level ABSA on review {i + 1} of {total_rows}...")
                res = engine.analyze(txt)
                sentiments.append(res["sentiment"])
                confidences.append(f"{res['confidence']:.1%}")
                if res["defective_aspects"]:
                    aspects_list.append(", ".join(res["defective_aspects"]))
                    for asp in res["defective_aspects"]:
                        live_defect_counter[asp] += 1
                else:
                    aspects_list.append("None")

            df_live["Predicted Sentiment"] = sentiments
            df_live["Confidence"] = confidences
            df_live["Detected Defects"] = aspects_list

            prog.progress(100, text="Batch inference and Pareto aggregation complete.")
            time.sleep(0.15)
            prog.empty()

            st.session_state.live_scrape_state = {
                "scrape_meta": {
                    "platform": scrape_res["platform"],
                    "category": scrape_res["category"],
                    "product_title": scrape_res["product_title"],
                    "total_scraped": len(df_live)
                },
                "dataframe": df_live,
                "defect_counter": dict(live_defect_counter)
            }
        else:
            prog.empty()
            st.warning("No reviews could be extracted from the provided URL. Verify the link and try again.")

    # Render persisted scrape results (never vanishes on tab switch)
    if st.session_state.live_scrape_state is not None:
        saved_scrape = st.session_state.live_scrape_state
        df_live = saved_scrape["dataframe"]
        meta = saved_scrape["scrape_meta"]
        defect_counter = Counter(saved_scrape["defect_counter"])

        defect_count = int((df_live["Predicted Sentiment"] == "Defect").sum())
        defect_rate = (defect_count / len(df_live)) * 100.0
        top_issue = defect_counter.most_common(1)[0][0] if defect_counter else "None"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                "Analyzed Sample",
                f"{len(df_live)} Reviews",
                f"Source: {meta['platform']}",
                palette["accent_primary"]
            )
        with k2:
            render_kpi_card(
                "Detected Category",
                meta["category"],
                meta["product_title"][:38] + "...",
                palette["accent_secondary"]
            )
        with k3:
            render_kpi_card(
                "Product Defect Rate",
                f"{defect_rate:.1f}%",
                f"{defect_count} of {len(df_live)} reviews flagged",
                palette["danger"] if defect_rate >= 25.0 else palette["success"]
            )
        with k4:
            render_kpi_card(
                "Primary Failure Driver",
                top_issue,
                "Top recurring defective aspect",
                palette["warning"]
            )

        st.write("")
        if defect_counter:
            render_pareto_chart(
                defect_counter,
                f"Component Defect Pareto Distribution — {meta['product_title'][:55]}",
                palette
            )

        st.markdown("#### Audited Review-Level Sentiment & Defect Log")
        st.dataframe(
            df_live[["Predicted Sentiment", "Confidence", "Detected Defects", "review_text", "page_scraped"]],
            use_container_width=True
        )

        csv_bytes = df_live.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Audited Defect Report (CSV)",
            data=csv_bytes,
            file_name="rivaltrack_live_report.csv",
            mime="text/csv"
        )


# ==========================================
# MODULE 3: HISTORICAL PARETO ANALYTICS
# ==========================================
elif active_mod == "Historical Pareto Analytics":
    st.markdown(
        """
        <div class="rt-card">
            <div class="rt-section-title">Historical Category 80/20 Pareto Analytics</div>
            <div class="rt-section-desc">
                Identifies the vital few failing components responsible for 80% of customer complaints across historical category cohorts.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if df_corpus.empty:
        st.warning("Historical corpus file not found.")
    else:
        categories = sorted(df_corpus["category"].unique())
        default_idx = 0
        if st.session_state.selected_pareto_category in categories:
            default_idx = categories.index(st.session_state.selected_pareto_category)

        selected_cat = st.selectbox("Select Product Domain", categories, index=default_idx)
        st.session_state.selected_pareto_category = selected_cat

        cat_df = df_corpus[df_corpus["category"] == selected_cat]
        cat_defects = cat_df[cat_df["sentiment"] == 0]
        cat_rate = (len(cat_defects) / len(cat_df)) * 100.0 if len(cat_df) > 0 else 0.0

        counts_dict = compute_historical_category_pareto(selected_cat, df_corpus)
        corpus_counter = Counter(counts_dict)
        top_cat_issue = corpus_counter.most_common(1)[0][0] if corpus_counter else "N/A"

        m1, m2, m3 = st.columns(3)
        with m1:
            render_kpi_card(
                f"{selected_cat} Cohort Size",
                f"{len(cat_df):,}",
                "Verified historical reviews",
                palette["accent_primary"]
            )
        with m2:
            render_kpi_card(
                "Category Defect Rate",
                f"{cat_rate:.1f}%",
                f"{len(cat_defects):,} defective reviews",
                palette["danger"]
            )
        with m3:
            render_kpi_card(
                "Dominant Category Bottleneck",
                top_cat_issue,
                "Highest frequency failing component",
                palette["warning"]
            )

        st.write("")
        render_pareto_chart(
            corpus_counter,
            f"Historical Component Defect Pareto — {selected_cat}",
            palette
        )


# ==========================================
# MODULE 4: COMPETITOR HEAD-TO-HEAD
# ==========================================
elif active_mod == "Competitor Head-to-Head":
    st.markdown(
        """
        <div class="rt-card">
            <div class="rt-section-title">Side-by-Side Competitor Quality Benchmarking</div>
            <div class="rt-section-desc">
                Benchmarks two competing products simultaneously to compare overall defect rates, reliability deltas,
                and component-level failure distributions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    colA, colB = st.columns(2)
    with colA:
        url_a = st.text_input(
            "Competitor A Product URL",
            value=st.session_state.h2h_url_a_text,
            placeholder="https://www.flipkart.com/..."
        )
    with colB:
        url_b = st.text_input(
            "Competitor B Product URL",
            value=st.session_state.h2h_url_b_text,
            placeholder="https://www.flipkart.com/..."
        )

    def extract_short_brand_name(url: str, title: str, fallback: str) -> str:
        slug_match = re.search(r"flipkart\.com/([^/]+)/p/", url)
        if slug_match:
            words = slug_match.group(1).replace("-", " ").title().split()
            return " ".join(words[:5])
        if title and title != url:
            return " ".join(title.split()[:5])
        return fallback

    run_h2h_btn = st.button("Execute Head-to-Head Benchmark")

    if run_h2h_btn and url_a and url_b and engine_loaded:
        st.session_state.h2h_url_a_text = url_a
        st.session_state.h2h_url_b_text = url_b

        prog = st.progress(0, text="Fetching multi-page reviews for Competitor A...")
        res_a = scrape_live_product(url_a, max_pages=8, target_reviews=50)
        prog.progress(35, text="Fetching multi-page reviews for Competitor B...")
        res_b = scrape_live_product(url_b, max_pages=8, target_reviews=50)

        name_a = extract_short_brand_name(url_a, res_a.get("product_title", ""), "Competitor A")
        name_b = extract_short_brand_name(url_b, res_b.get("product_title", ""), "Competitor B")

        def evaluate_h2h(scrape_data, start_pct: int, end_pct: int, label: str):
            df = scrape_data["dataframe"]
            if df.empty:
                return 0.0, Counter(), 0
            defects = 0
            issues = Counter()
            total = len(df)
            for idx, txt in enumerate(df["review_text"]):
                cur_p = start_pct + int(((idx + 1) / total) * (end_pct - start_pct))
                prog.progress(min(cur_p, 98), text=f"Analyzing {label} reviews ({idx + 1}/{total})...")
                res = engine.analyze(txt)
                if res["sentiment"] == "Defect":
                    defects += 1
                    for asp in res["defective_aspects"]:
                        issues[asp] += 1
            rate = (defects / total) * 100.0
            return rate, issues, total

        rate_a, issues_a, n_a = evaluate_h2h(res_a, 40, 68, name_a)
        rate_b, issues_b, n_b = evaluate_h2h(res_b, 68, 96, name_b)

        prog.progress(100, text="Head-to-Head benchmark complete.")
        time.sleep(0.15)
        prog.empty()

        st.session_state.h2h_comparison_state = {
            "name_a": name_a,
            "name_b": name_b,
            "cat_a": res_a["category"],
            "cat_b": res_b["category"],
            "rate_a": rate_a,
            "rate_b": rate_b,
            "n_a": n_a,
            "n_b": n_b,
            "issues_a": dict(issues_a),
            "issues_b": dict(issues_b)
        }

    # Render persisted Head-to-Head comparison
    if st.session_state.h2h_comparison_state is not None:
        h2h = st.session_state.h2h_comparison_state
        name_a, name_b = h2h["name_a"], h2h["name_b"]
        rate_a, rate_b = h2h["rate_a"], h2h["rate_b"]
        issues_a, issues_b = Counter(h2h["issues_a"]), Counter(h2h["issues_b"])

        if rate_a < rate_b:
            verdict_html = (
                f"<strong>BENCHMARK WINNER: {name_a}</strong> &mdash; Exhibits a "
                f"<strong>{rate_b - rate_a:.1f}% lower defect rate</strong> than {name_b} "
                f"({rate_a:.1f}% vs. {rate_b:.1f}%)."
            )
        elif rate_b < rate_a:
            verdict_html = (
                f"<strong>BENCHMARK WINNER: {name_b}</strong> &mdash; Exhibits a "
                f"<strong>{rate_a - rate_b:.1f}% lower defect rate</strong> than {name_a} "
                f"({rate_b:.1f}% vs. {rate_a:.1f}%)."
            )
        else:
            verdict_html = (
                f"<strong>STATISTICAL TIE:</strong> Both <strong>{name_a}</strong> and "
                f"<strong>{name_b}</strong> recorded an identical defect rate of <strong>{rate_a:.1f}%</strong>."
            )

        st.markdown(f'<div class="rt-verdict-banner">{verdict_html}</div>', unsafe_allow_html=True)

        top_a_str = ", ".join([f"{k} ({v})" for k, v in issues_a.most_common(4)]) or "None detected"
        top_b_str = ", ".join([f"{k} ({v})" for k, v in issues_b.most_common(4)]) or "None detected"

        col1, col2 = st.columns(2)
        with col1:
            render_kpi_card(
                label=f"Competitor A | {h2h['cat_a']} ({h2h['n_a']} Reviews)",
                value=f"{name_a} — {rate_a:.1f}% Defects",
                subtext=f"Primary Issues: {top_a_str}",
                accent_color=palette["danger"] if rate_a > rate_b else palette["success"]
            )
        with col2:
            delta_val = rate_b - rate_a
            render_kpi_card(
                label=f"Competitor B | {h2h['cat_b']} ({h2h['n_b']} Reviews)",
                value=f"{name_b} — {rate_b:.1f}% ({delta_val:+.1f}%)",
                subtext=f"Primary Issues: {top_b_str}",
                accent_color=palette["danger"] if rate_b > rate_a else palette["success"]
            )

        st.write("")
        all_components = [k for k, _ in (issues_a + issues_b).most_common(8)]
        if all_components:
            vals_a = [int(issues_a.get(c, 0)) for c in all_components]
            vals_b = [int(issues_b.get(c, 0)) for c in all_components]
            max_v = max(vals_a + vals_b) if (vals_a or vals_b) else 5

            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(
                x=all_components,
                y=vals_a,
                name=name_a,
                marker_color=palette["bar_primary"],
                text=vals_a,
                textposition="outside",
                textfont=dict(color=palette["text_primary"], family="JetBrains Mono")
            ))
            fig_h2h.add_trace(go.Bar(
                x=all_components,
                y=vals_b,
                name=name_b,
                marker_color=palette["bar_secondary"],
                text=vals_b,
                textposition="outside",
                textfont=dict(color=palette["text_primary"], family="JetBrains Mono")
            ))
            fig_h2h.update_layout(
                template=palette["plotly_template"],
                paper_bgcolor=palette["bg_card"],
                plot_bgcolor=palette["bg_card"],
                font=dict(family="Inter, sans-serif", color=palette["text_primary"]),
                barmode="group",
                title=dict(
                    text=f"Component Failure Comparison: {name_a} vs. {name_b}",
                    font=dict(size=16, color=palette["text_primary"])
                ),
                xaxis=dict(
                    title="Engineering / Service Aspect",
                    type="category",
                    gridcolor=palette["grid_color"],
                    tickfont=dict(color=palette["text_primary"])
                ),
                yaxis=dict(
                    title="Defect Complaint Count",
                    range=[0, max_v * 1.28],
                    gridcolor=palette["grid_color"],
                    tickfont=dict(color=palette["text_primary"])
                ),
                height=440,
                margin=dict(l=40, r=40, t=65, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_h2h, use_container_width=True)


# ==========================================
# MODULE 5: ENSEMBLE ARCHITECTURE & METRICS
# ==========================================
elif active_mod == "Ensemble Architecture & Metrics":
    st.markdown(
        """
        <div class="rt-card">
            <div class="rt-section-title">2-Tier Out-of-Fold Stacking Ensemble Architecture</div>
            <div class="rt-section-desc">
                Evaluates 5-Fold Stratified Out-of-Fold (OOF) base learners (Calibrated LinearSVC, Multinomial Naive Bayes,
                and LightGBM) aggregated via a Layer-2 Logistic Regression Meta-Learner.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    oof_path = PROJECT_ROOT / "models" / "oof_layer1_data.pkl"
    if oof_path.exists():
        with open(oof_path, "rb") as f:
            oof_data = pickle.load(f)

        report = oof_data.get("layer1_report", {})
        if report:
            metrics_df = pd.DataFrame.from_dict(report, orient="index")
            metrics_df.index.name = "Layer-1 Base Learner"
            st.markdown("#### Layer-1 Out-of-Fold Validation Performance")
            st.dataframe(metrics_df.round(4), use_container_width=True)

            # Grouped Bar Chart of Model Metrics
            learners = metrics_df.index.astype(str).tolist()
            f1_scores = [round(float(x) * 100, 2) for x in metrics_df["defect_f1"].tolist()]
            roc_scores = [round(float(x) * 100, 2) for x in metrics_df["roc_auc"].tolist()]

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=learners,
                y=f1_scores,
                name="Defect F1-Score (%)",
                marker_color=palette["bar_primary"],
                text=[f"{v:.1f}%" for v in f1_scores],
                textposition="outside"
            ))
            fig_m.add_trace(go.Bar(
                x=learners,
                y=roc_scores,
                name="ROC-AUC (%)",
                marker_color=palette["bar_secondary"],
                text=[f"{v:.1f}%" for v in roc_scores],
                textposition="outside"
            ))
            fig_m.update_layout(
                template=palette["plotly_template"],
                paper_bgcolor=palette["bg_card"],
                plot_bgcolor=palette["bg_card"],
                font=dict(family="Inter, sans-serif", color=palette["text_primary"]),
                barmode="group",
                title="Layer-1 Base Learner Benchmark (5-Fold Stratified OOF)",
                yaxis=dict(title="Score (%)", range=[0, 112], gridcolor=palette["grid_color"]),
                xaxis=dict(title="Base Classifier", type="category"),
                height=390,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Out-of-fold model telemetry artifact not found in models/oof_layer1_data.pkl.")
