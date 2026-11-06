import re
import pickle
from pathlib import Path
from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="RivalTrack Dashboard", page_icon="📊", layout="wide")

from src.absa_engine import ABSAEngine, extract_dynamic_aspects_from_clause, score_clause_polarity
from src.live_scraper import scrape_live_product
from src.nlp_pipeline import clean_review_text, is_valid_review

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def load_absa_engine(version: str = "v4.0_dynamic_clause_absa"):
    return ABSAEngine()


@st.cache_data(show_spinner=False)
def load_corpus_metrics(version: str = "v4.0_clean"):
    data_path = PROJECT_ROOT / "data" / "cleaned_absa_corpus.csv"
    if not data_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data_path)
    source_col = "review_text" if "review_text" in df.columns else df.columns[0]
    df = df[df[source_col].apply(is_valid_review)].reset_index(drop=True)
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(1).astype(int)
    df["category"] = df["category"].astype(str).str.strip().str.title()
    return df


def render_pareto_chart(aspect_counts: Counter, title: str):
    """Renders a clean Vertical Bar + Cumulative Line 80/20 Pareto Chart (NumPy 2.x safe)."""
    # Remove fallback label if real specific component defects exist
    cleaned_counts = Counter(aspect_counts)
    if len(cleaned_counts) > 1 and "Overall Build / Quality" in cleaned_counts:
        del cleaned_counts["Overall Build / Quality"]

    if not cleaned_counts:
        st.info("No recurring component defects found to plot.")
        return

    top_items = cleaned_counts.most_common(10)
    df_tally = pd.DataFrame(top_items, columns=["Component", "Count"])
    df_tally["Cumulative %"] = (df_tally["Count"].cumsum() / df_tally["Count"].sum()) * 100.0

    # Convert explicitly to Python lists (.tolist()) to prevent Plotly 6 + NumPy 2.x binary serialization bug
    x_labels = df_tally["Component"].astype(str).tolist()
    y_counts = [int(v) for v in df_tally["Count"].tolist()]
    y_cum = [round(float(v), 1) for v in df_tally["Cumulative %"].tolist()]
    max_count = max(y_counts) if y_counts else 10

    fig = go.Figure()

    # 1. Vertical Red Bars (Left Y-Axis: Complaint Frequency)
    fig.add_trace(go.Bar(
        x=x_labels,
        y=y_counts,
        name="Defect Mentions (Count)",
        marker_color="#d9534f",
        orientation="v",
        text=y_counts,
        textposition="outside"
    ))

    # 2. Cumulative Percentage Curve (Right Y-Axis: 0% to 100%)
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=y_cum,
        name="Cumulative Share (%)",
        yaxis="y2",
        mode="lines+markers+text",
        text=[f"{v:.0f}%" for v in y_cum],
        textposition="top center",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=8, color="#1f77b4")
    ))

    # 3. 80% Pareto Cutoff Line
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#ff7f0e",
        annotation_text="80% Pareto Cutoff (Vital Few)",
        annotation_position="bottom right",
        yref="y2"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis=dict(title="Defective Product Component", type="category"),
        yaxis=dict(
            title="Number of Defect Complaints",
            range=[0, max_count * 1.25],
            showgrid=True
        ),
        yaxis2=dict(
            title="Cumulative Defect Percentage (%)",
            overlaying="y",
            side="right",
            range=[0, 110],
            showgrid=False
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=460
    )
    st.plotly_chart(fig, use_container_width=True)


try:
    engine = load_absa_engine()
    engine_loaded = True
except Exception as e:
    st.error(f"Failed to load ABSA Engine: {e}")
    engine_loaded = False

df_corpus = load_corpus_metrics()

# ==========================================
# SIDEBAR OVERVIEW
# ==========================================
st.sidebar.title("📊 RivalTrack Metrics")
if st.sidebar.button("🔄 Reload Models & Clear Cache"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

if not df_corpus.empty:
    total_reviews = len(df_corpus)
    total_defects = len(df_corpus[df_corpus["sentiment"] == 0])
    st.sidebar.metric("Total Historical Reviews", f"{total_reviews:,}")
    st.sidebar.metric("Total Detected Defects", f"{total_defects:,}")
    st.sidebar.metric("System Defect Rate", f"{(total_defects/total_reviews)*100:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.write("**Category Breakdown**")
    cat_counts = df_corpus["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Volume"]
    st.sidebar.dataframe(cat_counts, hide_index=True)

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
st.title("RivalTrack: E-Commerce Competitor Intelligence")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Inference & Scraping",
    "📈 Defect Pareto Analytics",
    "⚔️ Competitor Head-to-Head",
    "🧠 Model Performance"
])

# ------------------------------------------
# TAB 1: LIVE INFERENCE & SCRAPING
# ------------------------------------------
with tab1:
    st.markdown("### Test Any Sentence (Clause-Level ABSA)")
    sample_text = st.text_area(
        "Paste any review snippet here:",
        value="The display is gorgeous but the btry drains super fast and the mic is garbage."
    )

    if st.button("Analyze Sentiment", type="primary") and engine_loaded:
        result = engine.analyze(sample_text)

        col1, col2, col3 = st.columns(3)
        if result["sentiment"] == "Defect":
            col1.error(f"Sentiment: {result['sentiment']}")
        else:
            col1.success(f"Sentiment: {result['sentiment']}")

        col2.info(f"Confidence: {result['confidence']:.1%}")

        if result["defective_aspects"]:
            col3.warning(f"Defects Found: {', '.join(result['defective_aspects'])}")
        else:
            col3.success("Defects Found: None")

        with st.expander("🔬 Inspect Clause-by-Clause Decomposition & NLP Cleaning"):
            st.write("**Normalized & Lemmatized Text:**", clean_review_text(sample_text))
            clause_rows = []
            for idx, c_info in enumerate(result.get("clause_breakdown", []), 1):
                clause_rows.append({
                    "Clause #": idx,
                    "Clause Text": c_info["clause"],
                    "Post-Contrast ('But')": c_info["post_contrast"],
                    "Compositional Polarity": round(c_info["ling_score"], 2),
                    "ML Positive Prob": f"{c_info['ml_pos_prob']:.1%}",
                    "Clause Verdict": "Negative (Defect)" if c_info["is_negative"] else ("Positive" if c_info["is_positive"] else "Neutral")
                })
            if clause_rows:
                st.dataframe(pd.DataFrame(clause_rows), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Scrape & Analyze Live Product URL")
    live_url = st.text_input("Paste a Flipkart or Amazon Product URL:")

    if st.button("Scrape & Analyze URL") and live_url and engine_loaded:
        with st.spinner("🕷️ Extracting authentic reviews from live DOM & SSR state..."):
           scrape_res = scrape_live_product(live_url, max_pages=10, target_reviews=55)

        if not scrape_res["dataframe"].empty:
            df_live = scrape_res["dataframe"].copy()

            sentiments = []
            confidences = []
            aspects_list = []
            live_defect_counter = Counter()

            for txt in df_live["review_text"]:
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

            defect_count = (df_live["Predicted Sentiment"] == "Defect").sum()
            st.success(
                f"Extracted **{len(df_live)} authentic reviews** from **{scrape_res['platform']}** "
                f"| Category: **{scrape_res['category']}** | Defects Flagged: **{defect_count}** ({(defect_count/len(df_live))*100:.1f}%)"
            )

            if live_defect_counter:
                render_pareto_chart(live_defect_counter, f"Live URL Component Defect Pareto — {scrape_res['product_title'][:50]}")

            st.dataframe(
                df_live[["Predicted Sentiment", "Confidence", "Detected Defects", "review_text", "page_scraped"]],
                use_container_width=True
            )

            csv_data = df_live.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Live Defect Report (CSV)",
                data=csv_data,
                file_name="rivaltrack_live_report.csv",
                mime="text/csv"
            )
        else:
            st.warning(
                "No live reviews could be extracted from this URL. The product may have 0 reviews, "
                "or the platform presented a CAPTCHA wall."
            )

# ------------------------------------------
# TAB 2: 80/20 PARETO ANALYTICS (CORPUS)
# ------------------------------------------
with tab2:
    st.markdown("### Historical Category Defect Pareto (Dynamic Aspect Extraction)")
    if df_corpus.empty:
        st.warning("No corpus found.")
    else:
        selected_cat = st.selectbox("Select Category:", sorted(df_corpus["category"].unique()))
        df_defects = df_corpus[(df_corpus["category"] == selected_cat) & (df_corpus["sentiment"] == 0)]

        if not df_defects.empty:
            corpus_counter = Counter()
            sample_defects = df_defects["review_text"].dropna().astype(str).head(400)
            for txt in sample_defects:
                pol = score_clause_polarity(txt)
                for asp in extract_dynamic_aspects_from_clause(txt, pol["implicit_aspect"]):
                    corpus_counter[asp] += 1

            render_pareto_chart(corpus_counter, f"Dynamic Component Defect Pareto — {selected_cat}")

# ------------------------------------------
# TAB 3: COMPETITOR HEAD-TO-HEAD
# ------------------------------------------
with tab3:
    st.markdown("### ⚔️ Side-by-Side Competitor Benchmarking")
    st.caption("Compare any two competitor product URLs to benchmark defect rates, reliability, and component failure distributions.")

    colA, colB = st.columns(2)
    with colA:
        url_a = st.text_input("Brand A Product URL:")
    with colB:
        url_b = st.text_input("Brand B Product URL:")

    def extract_short_brand_name(url: str, title: str, fallback: str) -> str:
        """Extracts a clean, readable product name from the URL slug or page title."""
        slug_match = re.search(r"flipkart\.com/([^/]+)/p/", url)
        if slug_match:
            words = slug_match.group(1).replace("-", " ").title().split()
            return " ".join(words[:5])
        if title and title != url:
            return " ".join(title.split()[:5])
        return fallback

    if st.button("Compare Competitors", type="primary") and url_a and url_b and engine_loaded:
        with st.spinner("Scraping & running Clause-Level ABSA on both competitor products..."):
            res_a = scrape_live_product(url_a, max_pages=8, target_reviews=50)
            res_b = scrape_live_product(url_b, max_pages=8, target_reviews=50)

        name_a = extract_short_brand_name(url_a, res_a.get("product_title", ""), "Brand A")
        name_b = extract_short_brand_name(url_b, res_b.get("product_title", ""), "Brand B")

        def evaluate_h2h(scrape_data):
            df = scrape_data["dataframe"]
            if df.empty:
                return 0.0, Counter(), 0
            defects = 0
            issues = Counter()
            for txt in df["review_text"]:
                res = engine.analyze(txt)
                if res["sentiment"] == "Defect":
                    defects += 1
                    for asp in res["defective_aspects"]:
                        issues[asp] += 1
            rate = (defects / len(df)) * 100.0
            return rate, issues, len(df)

        rate_a, issues_a, n_a = evaluate_h2h(res_a)
        rate_b, issues_b, n_b = evaluate_h2h(res_b)

        # Executive Winner Verdict Banner
        if rate_a < rate_b:
            st.success(f"🏆 **Reliability Winner: {name_a}** — **{rate_b - rate_a:.1f}% lower defect rate** than {name_b} ({rate_a:.1f}% vs {rate_b:.1f}%).")
        elif rate_b < rate_a:
            st.success(f"🏆 **Reliability Winner: {name_b}** — **{rate_a - rate_b:.1f}% lower defect rate** than {name_a} ({rate_b:.1f}% vs {rate_a:.1f}%).")
        else:
            st.info(f"⚖️ **Tie:** Both **{name_a}** and **{name_b}** have an identical defect rate of **{rate_a:.1f}%**.")

        col1, col2 = st.columns(2)
        top_a_str = ", ".join([f"{k} ({v})" for k, v in issues_a.most_common(4)])
        top_b_str = ", ".join([f"{k} ({v})" for k, v in issues_b.most_common(4)])

        with col1:
            st.subheader(f"🅰️ {name_a}")
            st.caption(f"Category: {res_a['category']} | Sample: {n_a} reviews")
            st.metric("Brand A Defect Rate", f"{rate_a:.1f}%")
            st.write(f"**Top Failing Components:** {top_a_str if top_a_str else 'None detected'}")

        with col2:
            st.subheader(f"🅱️ {name_b}")
            st.caption(f"Category: {res_b['category']} | Sample: {n_b} reviews")
            st.metric(
                "Brand B Defect Rate",
                f"{rate_b:.1f}%",
                delta=f"{rate_b - rate_a:+.1f}% vs {name_a.split()[0]}",
                delta_color="inverse"
            )
            st.write(f"**Top Failing Components:** {top_b_str if top_b_str else 'None detected'}")

        # Grouped Side-by-Side Component Defect Chart
        all_components = [k for k, _ in (issues_a + issues_b).most_common(8)]
        if all_components:
            vals_a = [int(issues_a.get(c, 0)) for c in all_components]
            vals_b = [int(issues_b.get(c, 0)) for c in all_components]

            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(
                x=all_components,
                y=vals_a,
                name=name_a,
                marker_color="#d9534f",
                text=vals_a,
                textposition="outside"
            ))
            fig_h2h.add_trace(go.Bar(
                x=all_components,
                y=vals_b,
                name=name_b,
                marker_color="#1f77b4",
                text=vals_b,
                textposition="outside"
            ))
            max_v = max(vals_a + vals_b) if (vals_a or vals_b) else 5
            fig_h2h.update_layout(
                barmode="group",
                title=f"Component Failure Breakdown: {name_a} vs. {name_b}",
                xaxis=dict(title="Product / Service Aspect", type="category"),
                yaxis=dict(title="Number of Defect Complaints", range=[0, max_v * 1.25]),
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_h2h, use_container_width=True)
            
# ------------------------------------------
# TAB 4: MODEL PERFORMANCE & METRICS
# ------------------------------------------
with tab4:
    st.markdown("### 2-Tier Stacking Ensemble Metrics")
    oof_path = PROJECT_ROOT / "models" / "oof_layer1_data.pkl"
    if oof_path.exists():
        with open(oof_path, "rb") as f:
            oof_data = pickle.load(f)

        report = oof_data.get("layer1_report", {})
        if report:
            metrics_df = pd.DataFrame.from_dict(report, orient="index")
            metrics_df.index.name = "Layer 1 Base Learner"
            st.dataframe(metrics_df.style.highlight_max(axis=0, color="darkgreen"), use_container_width=True)