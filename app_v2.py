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
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess
import os
from ngboost import NGBRegressor
from ngboost.distns import Normal

# ------------------- CONFIG -------------------
# Mortality (Death)
UNITIGS_CSV_DEATH = "./Unitigs_predictor_DEATH.csv"
MODEL_PATH_DEATH  = "./xgb_fold1_8_Death.joblib"

# ICU admission
UNITIGS_CSV_ICU = "./Unitigs_predictor_ICU.csv"
MODEL_PATH_ICU  = "./xgb_fold5_8_ICU.joblib"

# Length of Stay (continuous/distribution prediction)
UNITIGS_CSV_LOS = "./Unitigs_predictor_los.csv"
MODEL_PATH_LOS  = "./Unitig_model_ngb_log1p_fold1.joblib"

# If unitigs are the *column names* of the CSV, keep True.
UNITIGS_ARE_COLUMNS = True
UNITIGS_SEQ_COLUMN  = "unitig"  # only used if UNITIGS_ARE_COLUMNS=False

# BLAST databases
CARD_FASTA = "./CARD.fasta"
VFDB_FASTA = "./VFDB.fasta"
CARD_DB = "./card_db"
VFDB_DB = "./vfdb_db"
BLAST_MAX_RESULTS = 20

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
        "An **machine learning based digital health tool** for predicting **in-hospital mortality**, "
        "**ICU admission**, and **length of stay (LOS)** from bacterial **genomic biomarkers**. This research platform supports "
        "**infection prevention**, and data-driven **precision therapeutics**."
    )
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer**: Research tool. Not for clinical decision-making.")

# Header
st.markdown(
    "<h2 style='margin-bottom:0'>Machine Learning Based Genomic Risk Prediction</h2>"
    "<div style='color:#555;margin-top:4px;'>Mortality • ICU Admission • Length of Stay • Developed by the Infectious Diseases Epidemiology Lab at KAUST</div>",
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

def unitig_presence_in_text_parallel(unitigs, genome_text, n_jobs=None, progress_callback=None):
    """Parallel unitig scanning for large unitig sets (deterministic order) with progress tracking."""
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    chunk_size = int(np.ceil(len(unitigs) / n_jobs))
    chunks = [unitigs[i:i + chunk_size] for i in range(0, len(unitigs), chunk_size)]

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit in order, collect results in order
        futures = [executor.submit(unitig_presence_in_text_single, (chunk, genome_text)) for chunk in chunks]
        results = []
        for idx, future in enumerate(futures):
            result = future.result()
            results.append(result)
            if progress_callback:
                progress_callback((idx + 1) / len(futures))

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

def get_z_score(ci_level: int) -> float:
    """Return z-score for given confidence level."""
    z_scores = {90: 1.645, 95: 1.96, 99: 2.576}
    return z_scores.get(ci_level, 1.96)

def predict_los_distribution(model, X):
    """
    Extract predicted distribution from NGBRegressor or similar model.
    Returns distribution object with mean and std methods.
    """
    if hasattr(model, 'pred_dist'):
        # NGBRegressor has pred_dist method
        dist = model.pred_dist(X)
        return dist
    else:
        # Fallback for other regression models
        # Attempt to get predictions as mean
        preds = model.predict(X)
        return preds

def plot_los_prediction_intervals(selected_sample_idx, sample_name, mean_los_val, pi_dict):
    """
    Create a plot showing predicted LOS with all three prediction interval levels as vertical bands.
    X-axis: Predicted LOS (days) - back-transformed from log scale
    Y-axis: Prediction interval levels (90%, 95%, 99%)
    pi_dict: Dictionary with keys '90', '95', '99' containing tuples of (pi_lo, pi_hi) in original scale
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract PI values for selected sample (already back-transformed)
    pi_lo_90, pi_hi_90 = pi_dict['90'][0][selected_sample_idx], pi_dict['90'][1][selected_sample_idx]
    pi_lo_95, pi_hi_95 = pi_dict['95'][0][selected_sample_idx], pi_dict['95'][1][selected_sample_idx]
    pi_lo_99, pi_hi_99 = pi_dict['99'][0][selected_sample_idx], pi_dict['99'][1][selected_sample_idx]
    
    # Y positions for each PI level
    y_positions = {
        90: 1,
        95: 2,
        99: 3
    }
    
    # Colors for each PI level
    colors = {
        90: 'lightblue',
        95: 'steelblue',
        99: 'darkblue'
    }
    
    pi_values = {
        90: (pi_lo_90, pi_hi_90),
        95: (pi_lo_95, pi_hi_95),
        99: (pi_lo_99, pi_hi_99)
    }
    
    # Plot horizontal bands for each PI level
    band_height = 0.4
    for pi_level in [90, 95, 99]:
        y_pos = y_positions[pi_level]
        pi_lo, pi_hi = pi_values[pi_level]
        color = colors[pi_level]
        
        # Draw the band
        ax.barh(y_pos, pi_hi - pi_lo, left=pi_lo, height=band_height, 
                color=color, alpha=0.6, edgecolor='black', linewidth=2, label=f'{pi_level}% PI')
    
    # Draw the mean as a vertical line
    ax.axvline(mean_los_val, color='darkred', linestyle='--', linewidth=3, label='Predicted LOS', zorder=10)
    
    # Set x-axis limits to ensure all bars are fully visible with padding
    all_pi_values = [pi_lo_90, pi_hi_90, pi_lo_95, pi_hi_95, pi_lo_99, pi_hi_99]
    x_min = min(all_pi_values)
    x_max = max(all_pi_values)
    x_range = x_max - x_min
    padding = 0.1 * x_range  # Add 10% padding on each side
    ax.set_xlim(max(0, x_min - padding), x_max + padding)
    
    ax.set_xlabel('Length of Stay (days)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Prediction Interval Level', fontsize=13, fontweight='bold')
    ax.set_title(f'Predicted LOS with Prediction Intervals - Sample: {sample_name}', fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['90%', '95%', '99%'], fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    return fig

# ------------------- BLAST FUNCTIONS -------------------
def create_blast_databases():
    """Create BLAST databases from FASTA files if they don't exist."""
    if not os.path.exists(CARD_DB + ".nin"):
        if os.path.exists(CARD_FASTA):
            try:
                subprocess.run(
                    ["makeblastdb", "-in", CARD_FASTA, "-dbtype", "nucl", "-out", CARD_DB],
                    check=True,
                    capture_output=True
                )
            except Exception as e:
                st.warning(f"⚠️ Could not create CARD BLAST database: {e}")
    
    if not os.path.exists(VFDB_DB + ".nin"):
        if os.path.exists(VFDB_FASTA):
            try:
                subprocess.run(
                    ["makeblastdb", "-in", VFDB_FASTA, "-dbtype", "nucl", "-out", VFDB_DB],
                    check=True,
                    capture_output=True
                )
            except Exception as e:
                st.warning(f"⚠️ Could not create VFDB BLAST database: {e}")

def blast_unitig(unitig_seq, unitig_id, db_path, db_name):
    """
    Run BLAST search for a unitig sequence against a database.
    Returns best hit as a single row (or None if no hits).
    """
    try:
        import time
        # Create temporary query file with unique name
        query_file = f"/tmp/query_{db_name}_{int(time.time()*1000)}.fasta"
        with open(query_file, "w") as f:
            f.write(f">{unitig_id}\n{unitig_seq}\n")
        
        # Run BLASTN (no evalue constraint, limit to 1 result for best hit)
        result = subprocess.run(
            [
                "blastn",
                "-query", query_file,
                "-db", db_path,
                "-max_target_seqs", "1",
                "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Clean up query file
        if os.path.exists(query_file):
            os.remove(query_file)
        
        # Parse results - get only best hit
        if result.stdout and result.stdout.strip():
            line = result.stdout.strip().split('\n')[0]  # Get only first line (best hit)
            parts = line.split('\t')
            
            if len(parts) >= 12:
                hit = {
                    'Unitig': unitig_id[:50],  # Truncate long names for display
                    'Subject': parts[1][:100],
                    'Identity %': float(parts[2]),
                    'Length': int(parts[3]),
                    'Mismatches': int(parts[4]),
                    'Gaps': int(parts[5]),
                    'Query Start': int(parts[6]),
                    'Query End': int(parts[7]),
                    'Subject Start': int(parts[8]),
                    'Subject End': int(parts[9]),
                    'E-value': float(parts[10]),
                    'Bitscore': float(parts[11]),
                    'Database': db_name.upper()
                }
                return hit
        
        return None  # No hits
    except subprocess.TimeoutExpired:
        st.warning(f"⚠️ BLAST timeout for unitig {unitig_id[:30]}")
        return None
    except Exception as e:
        st.warning(f"⚠️ BLAST search issue: {str(e)[:100]}")
        return None

def blast_multiple_unitigs(unitig_list, db_path, db_name):
    """
    Run BLAST for multiple unitigs and return their best hits.
    """
    results = []
    for unitig in unitig_list:
        hit = blast_unitig(unitig, db_path, db_name)
        if hit:
            results.append(hit)
    return results

def get_unitig_sequence(unitig_id, unitigs_list):
    """Get the sequence of a unitig from the unitigs list."""
    if unitig_id in unitigs_list:
        return unitig_id
    return None

# ------------------- UI: SELECT OUTCOME -------------------
st.subheader("Select Prediction Target")
outcome = st.radio(
    "Clinical outcome:",
    ["In-hospital Mortality", "ICU Admission", "Length of Stay (LOS)"],
    horizontal=True
)

if outcome == "In-hospital Mortality":
    UNITIGS_CSV, MODEL_PATH, label_positive = UNITIGS_CSV_DEATH, MODEL_PATH_DEATH, "mortality"
    is_los_prediction = False
elif outcome == "ICU Admission":
    UNITIGS_CSV, MODEL_PATH, label_positive = UNITIGS_CSV_ICU, MODEL_PATH_ICU, "ICU_admission"
    is_los_prediction = False
else:  # Length of Stay
    UNITIGS_CSV, MODEL_PATH, label_positive = UNITIGS_CSV_LOS, MODEL_PATH_LOS, "LOS"
    is_los_prediction = True

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

if is_los_prediction:
    # For LOS: no uncertainty strength parameter needed (uses model's std directly)
    col1, col2 = st.columns([1, 1])
    with col1:
        run_shap = st.checkbox("Run Predictive Biomarker Identification (optional)", value=False)
    with col2:
        st.empty()  # Placeholder for alignment
    thr = None
    ci_level = None
    n_eff = None
else:
    # For binary predictions: show threshold and standard uncertainty parameters
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
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
    with col4:
        st.empty()  # Placeholder for alignment
    ci_level = None

# ------------------- PREDICT -------------------
if st.button("Run Prediction"):
    if not files:
        st.warning("Upload at least one FASTA file.")
        st.stop()

    # Load unitigs and scan
    unitigs = load_unitigs(UNITIGS_CSV, UNITIGS_ARE_COLUMNS, UNITIGS_SEQ_COLUMN)
    st.session_state["unitigs_cache"] = unitigs

    rows = []
    total_files = len(files)
    
    with st.status("🔍 Scanning genomes for genomic biomarkers...", expanded=True) as status_container:
        rows = []

        for i, up in enumerate(files, start=1):
            # Update main status
            status_container.update(label=f"🔍 Processing {up.name} ({i}/{total_files})", state="running")
            
            raw = up.read()
            if up.name.endswith(".gz"):
                aw = gzip.decompress(raw)
            fasta_dict = parse_fasta(raw)
            if not fasta_dict:
                st.warning(f"{up.name}: no sequences parsed.")
                continue
        
            concat = concat_sequences(fasta_dict)

            # Create a progress callback for unitig scanning
            unitig_progress_display = {"value": 0}
            status_text_element = st.empty()
            
            def unitig_progress_callback(progress):
                unitig_progress_display["value"] = progress
                status_text_element.write(f"  └─ Scanning unitigs: {int(progress * 100)}% complete")

            # Automatically choose parallel scanning for large unitig sets
            calls = unitig_presence_in_text_parallel(unitigs, concat, progress_callback=unitig_progress_callback)

            row = {"sample": Path(up.name).stem}
            row.update({u: c for u, c in zip(unitigs, calls)})
            rows.append(row)
            
            status_text_element.empty()
            st.write(f"  ✅ {up.name} complete")
        
        status_container.update(label="✅ Genome scanning complete!", state="complete")

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

    st.info("🧬 Running model inference...")
    
    if is_los_prediction:
        # LOS prediction: extract distribution
        try:
            pred_dist = predict_los_distribution(model, X)
            if hasattr(pred_dist, 'mean'):
                mean_los = pred_dist.mean()
                std_los = pred_dist.std()
            else:
                # Fallback: treat as simple predictions
                mean_los = pred_dist
                std_los = np.ones_like(pred_dist) * np.std(pred_dist) if isinstance(pred_dist, np.ndarray) else np.std(model.predict(X))
            st.success("✅ Inference complete!")
            
            st.info("📊 Computing prediction intervals for all confidence levels (90%, 95%, 99%)...")
            # Compute all three PI levels in log scale, then transform back
            pi_levels = [90, 95, 99]
            
            # Back-transform mean and std from log1p scale
            mean_los_original = np.expm1(mean_los)
            
            results_dict = {
                "Sample": pa_df["sample"],
                "Predicted LOS (days)": mean_los_original,
                "Std Dev (log scale)": std_los
            }
            
            for pi_level in pi_levels:
                z_score = get_z_score(pi_level)
                # Compute in log scale
                pi_lo_log = mean_los - z_score * std_los
                pi_hi_log = mean_los + z_score * std_los
                # Transform back to original scale using expm1 (inverse of log1p)
                pi_lo_original = np.expm1(pi_lo_log)
                pi_hi_original = np.expm1(pi_hi_log)
                # Ensure non-negative values
                pi_lo_original = np.maximum(pi_lo_original, 0)
                results_dict[f"{pi_level}% PI Lower"] = pi_lo_original
                results_dict[f"{pi_level}% PI Upper"] = pi_hi_original
            
            results = pd.DataFrame(results_dict)
            st.success(f"✅ Prediction intervals computed for 90%, 95%, and 99%!")
        except Exception as e:
            st.error(f"❌ LOS prediction failed: {e}")
            st.stop()
    else:
        # Binary prediction: probability + classification
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
    st.markdown("---")
    st.subheader("Prediction Results")
    st.dataframe(st.session_state["prediction_results"], width='stretch')
    st.download_button(
        label="Download Predictions (CSV)",
        data=st.session_state["prediction_results"].to_csv(index=False).encode(),
        file_name=f"predictions_{st.session_state.get('results_label_positive', 'prediction')}.csv",
        mime="text/csv",
        key="download_predictions",
    )
    
        # =============== LOS VISUALIZATIONS ===============
    if is_los_prediction and st.session_state.get("prediction_results") is not None:
        st.markdown("---")
        st.subheader("LOS Prediction Intervals")
        
        results_df = st.session_state["prediction_results"]
        samples = results_df["Sample"].values
        mean_los = results_df["Predicted LOS (days)"].values
        
        # Prepare PI data for all three levels (back-transformed to original scale)
        pi_dict = {
            '90': (results_df["90% PI Lower"].values, results_df["90% PI Upper"].values),
            '95': (results_df["95% PI Lower"].values, results_df["95% PI Upper"].values),
            '99': (results_df["99% PI Lower"].values, results_df["99% PI Upper"].values)
        }
        
        # Sample selector
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_sample = st.selectbox(
                "Select a sample to view detailed prediction intervals:",
                samples,
                index=0,
                key="los_sample_selector"
            )
        
        # Get index of selected sample
        selected_idx = np.where(samples == selected_sample)[0][0]
        mean_los_val = mean_los[selected_idx]
        
        st.markdown(f"**Prediction Interval Plot** - X-axis shows predicted LOS range (days), Y-axis shows prediction interval levels (90%, 95%, 99%)")
        fig = plot_los_prediction_intervals(selected_idx, selected_sample, mean_los_val, pi_dict)
        st.pyplot(fig)
        plt.close(fig)

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
            
            # =============== BLAST UNITIG ANNOTATION ===============
            st.markdown("---")
            st.subheader("Genomic Biomarker Annotation Against CARD and VFDB")
            
            # Get top 50 unitigs from SHAP explanation
            shap_feature_names = sv.feature_names if hasattr(sv, 'feature_names') else unitigs
            shap_values_abs = np.abs(sv.values)
            top_indices = np.argsort(shap_values_abs)[-50:][::-1]  # Always get top 50
            top_50_unitigs = [shap_feature_names[i] for i in top_indices]
            
            # Show top N based on user selection (for display)
            display_unitigs = top_50_unitigs[:top_n]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info(f"ℹ️ BLAST will run on top 50 biomarkers. Displaying {len(display_unitigs)} based on your SHAP selection.")
            with col2:
                blast_button = st.button("Run BLAST Analysis", key="blast_button")
            
            if blast_button:
                # Create BLAST databases if needed
                create_blast_databases()
                
                # Run BLAST against both databases for all top 50 unitigs
                st.info("🔄 Running BLAST searches for top 50 biomarkers against CARD and VFDB databases...")
                
                all_results = []
                progress_bar = st.progress(0)
                
                for idx, unitig in enumerate(top_50_unitigs):
                    # BLAST against CARD
                    card_hit = blast_unitig(unitig, f"unitig_{idx}", CARD_DB, "card")
                    if card_hit:
                        all_results.append(card_hit)
                    
                    # BLAST against VFDB
                    vfdb_hit = blast_unitig(unitig, f"unitig_{idx}", VFDB_DB, "vfdb")
                    if vfdb_hit:
                        all_results.append(vfdb_hit)
                    
                    progress_bar.progress((idx + 1) / len(top_50_unitigs))
                
                st.success("✅ BLAST searches complete!")
                
                # Convert to DataFrame
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    
                    # Reorder columns for better visibility
                    column_order = ['Unitig', 'Subject', 'Identity %', 'Length', 'E-value', 'Bitscore', 'Database', 
                                   'Mismatches', 'Gaps', 'Query Start', 'Query End', 'Subject Start', 'Subject End']
                    results_df = results_df[column_order]
                    
                    # Display results
                    st.markdown("**All Best BLAST Hits (Top 50 Biomarkers)**")
                    st.dataframe(
                        results_df.sort_values(['Database', 'E-value']),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    st.download_button(
                        label="Download All BLAST Results (CSV)",
                        data=results_df.to_csv(index=False).encode(),
                        file_name="BLAST_results_all_databases.csv",
                        mime="text/csv",
                        key="download_all_blast"
                    )
                else:
                    st.warning("❌ No BLAST hits found for any unitigs")
                
        except Exception as e:
            st.error(f"❌ Could not render SHAP plot: {e}")
    else:
        st.info("💡 No SHAP values available yet. Click 'Compute SHAP values' above to generate them.")
