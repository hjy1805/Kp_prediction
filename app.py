# app.py
# KAUST Infectious Diseases Epidemiology Lab
# Digital Health Tool: Genomic Biomarkers → Clinical Risk Prediction

import re
import gzip
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ------------------- CONFIG -------------------
# Mortality (Death)
UNITIGS_CSV_DEATH = "./Unitigs_predictor_DEATH.csv"
MODEL_PATH_DEATH  = "./xgb_fold1_8_Death.joblib"

# ICU admission
UNITIGS_CSV_ICU = "./Unitigs_predictor_ICU.csv"
MODEL_PATH_ICU  = "./xgb_fold5_8_ICU.joblib"

# If unitigs are the *column names* of the CSV, keep True.
UNITIGS_ARE_COLUMNS = True
UNITIGS_SEQ_COLUMN  = "unitig"  # only used if UNITIGS_ARE_COLUMNS=False

# ------------------- PAGE SETUP -------------------
st.set_page_config(
    page_title="KAUST Digital Health | Genomic Risk Prediction",
    layout="wide",
    page_icon="🧬",
)

# Display KAUST logo at the top

st.image("KAUST_Logo.svg", width=250)
st.markdown(
    "<div style='font-size:16px; color:#555; margin-top:4px;'>"
    "KAUST Infectious Diseases Epidemiology Lab"
    "</div>",
    unsafe_allow_html=True
)

# Sidebar branding
with st.sidebar:
    st.markdown("### 🧬 KAUST | Infectious Diseases Epidemiology Lab")
    st.markdown(
        "An **machine learning based digital health tool** for predicting **in-hospital mortality** and "
        "**ICU admission** from bacterial **genomic biomarkers**. This research platform supports "
        "**infection prevention**, and data-driven **precision therapeutics**."
    )
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer**: Research tool. Not for clinical decision-making.")

# Header
st.markdown(
    "<h2 style='margin-bottom:0'>Machine Learning Based Genomic Risk Prediction</h2>"
    "<div style='color:#555;margin-top:4px;'>Mortality & ICU Admission • Developed by the Infectious Diseases Epidemiology Lab at KAUST</div>",
    unsafe_allow_html=True,
)

# ------------------- HELPERS -------------------
def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTacgtnN", "TGCAtgcanN")
    return seq.translate(comp)[::-1]

@st.cache_data(show_spinner=False)
def load_unitigs(unitigs_csv: str, are_columns: bool, seq_col: str):
    df = pd.read_csv(unitigs_csv)
    if are_columns:
        dna_cols = []
        for c in df.columns:
            if isinstance(c, str) and re.fullmatch(r"[ACGTNacgtn]+", c) and len(c) >= 5:
                dna_cols.append(c)
        if not dna_cols:  # fallback
            dna_cols = list(df.columns)
        unitigs = dna_cols
    else:
        if seq_col not in df.columns:
            raise ValueError(f"Column '{seq_col}' not found in {unitigs_csv}")
        unitigs = df[seq_col].astype(str).tolist()

    # De-duplicate preserving order
    seen, ordered = set(), []
    for u in unitigs:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered

def parse_fasta(file_bytes: bytes) -> dict:
    text = file_bytes.decode(errors="ignore")
    seqs = {}
    header, chunks = None, []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seqs[header] = "".join(chunks)
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None:
        seqs[header] = "".join(chunks)
    return seqs

def concat_sequences(fasta_dict: dict) -> str:
    return "NNNNN".join(fasta_dict.values())

def unitig_presence_in_text_single(args):
    """Helper for parallelization."""
    unitigs_chunk, genome_text = args
    genome_text_upper = genome_text.upper()
    calls = []
    for u in unitigs_chunk:
        u_upper = u.upper()
        rc = reverse_complement(u_upper)
        present = (u_upper in genome_text_upper) or (rc in genome_text_upper)
        calls.append(1 if present else 0)
    return calls

def unitig_presence_in_text_parallel(unitigs, genome_text, n_jobs=None):
    """Parallel unitig scanning for large unitig sets (deterministic order)."""
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    chunk_size = int(np.ceil(len(unitigs) / n_jobs))
    chunks = [unitigs[i:i + chunk_size] for i in range(0, len(unitigs), chunk_size)]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit in order, collect results in order
        futures = [executor.submit(unitig_presence_in_text_single, (chunk, genome_text)) for chunk in chunks]
        results = [f.result() for f in futures]  # preserves order of chunks

    # Flatten the results list
    calls = [c for chunk_result in results for c in chunk_result]
    return calls


def wilson_ci_vectorized(p: np.ndarray, n_eff: int = 200, z: float = 1.96):
    """Fast, vectorized 95% CI via Wilson interval (approximate)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    z2 = z ** 2
    denom = 1 + z2 / n_eff
    center = (p + z2 / (2 * n_eff)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_eff + z2 / (4 * n_eff**2)) / denom
    lo = np.clip(center - margin, 0, 1)
    hi = np.clip(center + margin, 0, 1)
    return lo, hi

# ------------------- UI: SELECT OUTCOME -------------------
st.subheader("Select Prediction Target")
outcome = st.radio(
    "Clinical outcome:",
    ["In-hospital Mortality", "ICU Admission"],
    horizontal=True
)

if outcome == "In-hospital Mortality":
    UNITIGS_CSV, MODEL_PATH, label_positive = UNITIGS_CSV_DEATH, MODEL_PATH_DEATH, "mortality"
else:
    UNITIGS_CSV, MODEL_PATH, label_positive = UNITIGS_CSV_ICU, MODEL_PATH_ICU, "ICU_admission"

st.markdown(f"**Outcome selected:** {outcome}")

# ------------------- SESSION STATE INITIALIZATION -------------------
# Initialize session state keys
if "unitigs_cache" not in st.session_state:
    st.session_state["unitigs_cache"] = None
if "pa_df" not in st.session_state:
    st.session_state["pa_df"] = None
if "current_outcome" not in st.session_state:
    st.session_state["current_outcome"] = outcome

# Detect outcome change and invalidate all caches
if st.session_state["current_outcome"] != outcome:
    st.session_state["current_outcome"] = outcome
    # Clear all caches when outcome changes
    st.session_state["shap_cache_meta"] = None
    st.session_state["shap_explainer"] = None
    st.session_state["shap_values"] = None
    st.session_state["unitigs_cache"] = None
    st.session_state["pa_df"] = None
    st.session_state["prediction_results"] = None
    st.session_state["results_label_positive"] = None
    # Clear function caches for this session
    st.cache_data.clear()
    st.info("⚠️ Outcome changed. Previous predictions and SHAP values cleared.")

# ------------------- INPUTS -------------------
st.subheader("Upload bacterial genome assemblies (FASTA)")
files = st.file_uploader(
    "Select files",
    type=["fa", "fasta", "fna", "fa.gz", "fasta.gz"],
    accept_multiple_files=True,
)
col1, col2, col3 = st.columns([1,1,1])
with col1:
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
with col2:
    n_eff = st.select_slider(
        "Uncertainty strength (CI effective n)",
        options=[50, 100, 150, 200, 300, 400, 500],
        value=200,
        help="Larger values → narrower CIs (more confident). Uses fast Wilson interval approximation."
    )
with col3:
    run_shap = st.checkbox("Run Predictive Biomarker Identification (optional)", value=False)

# ------------------- PREDICT -------------------
if st.button("Run Prediction"):
    if not files:
        st.warning("Upload at least one FASTA file.")
        st.stop()

    # Load unitigs and scan
    unitigs = load_unitigs(UNITIGS_CSV, UNITIGS_ARE_COLUMNS, UNITIGS_SEQ_COLUMN)
    st.session_state["unitigs_cache"] = unitigs

    rows = []
    with st.spinner("Scanning genomes for genomic biomarkers..."):
        progress_text = "Scanning uploaded genomes..."
        progress_bar = st.progress(0, text=progress_text)
    
        total_files = len(files)
        rows = []

        for i, up in enumerate(files, start=1):
            raw = up.read()
            if up.name.endswith(".gz"):
                aw = gzip.decompress(raw)
            fasta_dict = parse_fasta(raw)
            if not fasta_dict:
                st.warning(f"{up.name}: no sequences parsed.")
                continue
        
            concat = concat_sequences(fasta_dict)

        # Automatically choose parallel scanning for large unitig sets
            calls = unitig_presence_in_text_parallel(unitigs, concat)

            row = {"sample": Path(up.name).stem}
            row.update({u: c for u, c in zip(unitigs, calls)})
            rows.append(row)

            progress_bar.progress(i / total_files, text=f"Scanning {up.name} ({i}/{total_files})")
        progress_bar.empty()

    st.success("✅ Genome scanning complete!")

    if not rows:
        st.error("No valid FASTA content processed.")
        st.stop()

    pa_df = pd.DataFrame(rows, columns=["sample"] + unitigs)
    pa_df[unitigs] = pa_df[unitigs].astype(np.uint8)
    st.session_state["pa_df"] = pa_df

    # Load model & predict
    st.info("🔄 Loading model...")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model from `{MODEL_PATH}`: {e}")
        st.stop()
    st.success("✅ Model loaded!")

    X = pa_df[unitigs].astype(np.float32)

    # Fast prediction + fast (approx) 95% CI
    st.info("🧬 Running model inference...")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        prob = proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X), dtype=float)
    else:
        # Fallback to labels; cast to pseudo-prob
        pred_raw = model.predict(X)
        uniq = np.unique(pred_raw)
        if set(uniq) - {0, 1}:
            mapping = {uniq.min(): 0, uniq.max(): 1}
            prob = np.vectorize(mapping.get)(pred_raw).astype(float)
        else:
            prob = pred_raw.astype(float)
    st.success("✅ Inference complete!")

    st.info("📊 Computing confidence intervals...")
    ci_lo, ci_hi = wilson_ci_vectorized(prob, n_eff=n_eff, z=1.96)
    pred = (prob >= thr).astype(int)
    st.success("✅ Confidence intervals computed!")

    results = pd.DataFrame({
        "Sample": pa_df["sample"],
        f"Predicted Probability ({label_positive})": prob,
        "95% CI Lower": ci_lo,
        "95% CI Upper": ci_hi,
        f"Prediction (thr={thr:.2f})": pred
    })

    # Cache results to persist across reruns
    st.session_state["prediction_results"] = results
    st.session_state["results_label_positive"] = label_positive

    st.success("✅ Prediction complete.")

    st.markdown("---")

# Display prediction results (persists across reruns)
if st.session_state.get("prediction_results") is not None:
    st.dataframe(st.session_state["prediction_results"], width='stretch')
    st.download_button(
        label="Download Predictions (CSV)",
        data=st.session_state["prediction_results"].to_csv(index=False).encode(),
        file_name=f"predictions_{st.session_state.get('results_label_positive', 'prediction')}.csv",
        mime="text/csv",
        key="download_predictions",
    )

# =============== SHAP SECTION (Outside "Run Prediction" button) ===============
# This section persists independently and doesn't re-trigger when predict button is clicked
if run_shap and st.session_state.get("pa_df") is not None:
    st.markdown("---")
    st.subheader("Predictive Biomarker Identification")
    st.caption(
        "Identifies influential genomic biomarkers driving the predicted risk for the selected sample."
    )

    # Retrieve cached data from session
    pa_df = st.session_state.get("pa_df")
    unitigs = st.session_state.get("unitigs_cache")
    
    if pa_df is None or unitigs is None:
        st.error("❌ No prediction data available. Please run prediction first.")
        st.stop()

    # Load model again for SHAP
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    X = pa_df[unitigs].astype(np.float32)

    # Initialize SHAP session cache keys
    if "shap_cache_meta" not in st.session_state:
        st.session_state["shap_cache_meta"] = None
    if "shap_explainer" not in st.session_state:
        st.session_state["shap_explainer"] = None
    if "shap_values" not in st.session_state:
        st.session_state["shap_values"] = None

    # Simple cache invalidation key: (num_unitigs, num_samples, model_path)
    try:
        shap_meta = (len(unitigs), X.shape[0], MODEL_PATH)
    except Exception:
        shap_meta = None

    need_recompute = st.session_state.get("shap_cache_meta") != shap_meta

    # Default behavior: automatically recompute when inputs change
    # Button allows manual recompute if needed
    compute_shap_btn = st.button("Recompute SHAP values", key="compute_shap_btn")

    if need_recompute or compute_shap_btn:
        with st.spinner("Computing SHAP explainer and values... This may take a while for many samples..."):
            try:
                explainer = shap.TreeExplainer(model, feature_names=unitigs)
                shap_vals = explainer(X)
                st.session_state["shap_explainer"] = explainer
                st.session_state["shap_values"] = shap_vals
                st.session_state["shap_cache_meta"] = shap_meta
                st.success("✅ SHAP values computed and cached. Change plot options freely!")
            except Exception as e:
                st.error(f"❌ SHAP explanation failed: {e}")

    # If shap_values already computed, show plotting UI separately
    if st.session_state.get("shap_values") is not None:
        st.markdown("---")
        st.markdown("**Visualization Controls**")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            sample_id = st.selectbox("Select a sample", list(pa_df["sample"]), key="shap_sample")
        with c2:
            top_n = st.slider("Number of biomarkers to display", 5, 50, 20, key="shap_top_n")

        plot_type = st.radio("Visualization", ["Waterfall", "Bar"], horizontal=True, key="shap_plot_type")

        # Index the selected sample and render plot from cached shap values
        try:
            idx = list(pa_df["sample"]).index(sample_id)
            sv = st.session_state["shap_values"][idx]

            fig, ax = plt.subplots(figsize=(8, 6))
            if plot_type == "Waterfall":
                shap.plots.waterfall(sv, max_display=top_n, show=False)
            else:
                shap.plots.bar(sv, max_display=top_n, show=False)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"❌ Could not render SHAP plot: {e}")
    else:
        st.info("💡 No SHAP values available yet. Click 'Compute SHAP values' above to generate them.")
