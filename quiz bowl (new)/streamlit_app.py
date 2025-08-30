# streamlit_app.py
import io
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# import your existing functions / config
from config import DEFAULT_THRESHOLD, sbert_model
from clustering import process_clues_dataframe
from answer_processing import process_answerline_input

st.set_page_config(page_title="THE CLUELER (web)", layout="wide")

# -----------------------
# Utilities (replace PyQt color helpers)
# -----------------------
def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def interpolate_color(
    value: float,
    lower: float,
    upper: float,
    low_color: Tuple[int, int, int] = (0, 255, 0),
    high_color: Tuple[int, int, int] = (255, 0, 0),
    reverse: bool = False,
) -> str:
    """Return a hex color string interpolated between low_color and high_color."""
    if lower is None or upper is None or np.isnan(lower) or np.isnan(upper) or lower == upper:
        # fallback neutral color
        mid = ((low_color[0] + high_color[0]) // 2,
               (low_color[1] + high_color[1]) // 2,
               (low_color[2] + high_color[2]) // 2)
        return rgb_to_hex(mid)
    v = max(lower, min(value, upper))
    norm = (v - lower) / (upper - lower)
    if reverse:
        norm = 1.0 - norm
    norm = max(0.0, min(norm, 1.0))
    r = int(low_color[0] + norm * (high_color[0] - low_color[0]))
    g = int(low_color[1] + norm * (high_color[1] - low_color[1]))
    b = int(low_color[2] + norm * (high_color[2] - low_color[2]))
    return rgb_to_hex((r, g, b))

def compute_ranges(cluster_stats: List[Tuple]) -> dict:
    """Compute 10th and 90th percentiles for stats used for coloring."""
    if not cluster_stats:
        return {
            "avg_placement": (None, None),
            "cluster_size":  (None, None),
            "avg_distance":  (None, None),
            "quality":       (None, None),
        }
    arr = np.array
    placements = arr([s[0] for s in cluster_stats])
    sizes      = arr([s[1] for s in cluster_stats])
    distances  = arr([s[2] for s in cluster_stats])
    qualities  = arr([s[3] for s in cluster_stats])
    return {
        "avg_placement": tuple(np.percentile(placements, [10, 90])),
        "cluster_size":  tuple(np.percentile(sizes,      [10, 90])),
        "avg_distance":  tuple(np.percentile(distances,  [10, 90])),
        "quality":       tuple(np.percentile(qualities,  [10, 90])),
    }

def cluster_stats_to_df(cluster_stats: List[Tuple], ranges: dict) -> pd.DataFrame:
    """Turn cluster_stats into a DataFrame for display; also compute color hex strings."""
    rows = []
    for stat in cluster_stats:
        avg_place, size, avg_dist, qual, rep_clue, rep_ans, full_clues = stat
        rows.append({
            "Quality": float(qual),
            "Avg Placement": float(avg_place),
            "Cluster Size": int(size),
            "Avg Distance": float(avg_dist),
            "Title Clue": rep_clue,
            "Answer": rep_ans,
            "Full Clues": full_clues,
            # CSS-friendly colors:
            "_c_quality": interpolate_color(qual, *ranges["quality"], reverse=True),
            "_c_avg_place": interpolate_color(avg_place, *ranges["avg_placement"]),
            "_c_size": interpolate_color(size, *ranges["cluster_size"], reverse=True),
            "_c_dist": interpolate_color(avg_dist, *ranges["avg_distance"]),
        })
    df = pd.DataFrame(rows)
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")

# -----------------------
# Streamlit UI
# -----------------------
st.title("THE CLUELER — Web")
st.markdown(
    """
    Web wrapper for your Quiz Bowl cluster sorter.
    Enter an answerline (or many), tune threshold and options, then **Process Answerline**.
    """
)

# --- top controls ---
col1, col2, col3 = st.columns([4, 2, 3])

with col1:
    answerline = st.text_input("Answerline", value="", placeholder="Type an answerline to search (e.g. 'Einstein')")
with col2:
    threshold = st.slider("Threshold", min_value=0.1, max_value=10.0, value=float(DEFAULT_THRESHOLD), step=0.1)
with col3:
    process_btn = st.button("Process Answerline")
    recluster_btn = st.button("Re-cluster")

# advanced options (collapsible)
with st.expander("Advanced options", expanded=False):
    search_type = st.selectbox("Search Type", options=["answer", "question", "all"], index=0)
    exact_phrase = st.checkbox("Exact Phrase", value=False)
    ignore_word_order = st.checkbox("Ignore Word Order", value=True)
    remove_parentheticals = st.checkbox("Remove Parentheticals", value=True)
    set_name = st.text_input("Set Name (optional)", value="")
    difficulties_text = st.text_input("Difficulties (comma-separated)", value="", placeholder="e.g., 1,2,3")
    categories = st.text_input("Categories (comma-separated)", value="", placeholder="e.g., Literature,History")

# status placeholder
status = st.empty()

# persistent (session) state: store clues_df and cluster_stats in session_state for re-cluster
if "clues_df" not in st.session_state:
    st.session_state.clues_df = None
if "cluster_stats" not in st.session_state:
    st.session_state.cluster_stats = []
if "ranges" not in st.session_state:
    st.session_state.ranges = {
        "avg_placement": (None, None),
        "cluster_size":  (None, None),
        "avg_distance":  (None, None),
        "quality":       (None, None),
    }

# helpers to update UI state
def process_answerline_action():
    st.session_state.cluster_stats = []
    st.session_state.ranges = {
        "avg_placement": (None, None),
        "cluster_size":  (None, None),
        "avg_distance":  (None, None),
        "quality":       (None, None),
    }

    if not answerline.strip():
        status.error("Please enter an answerline.")
        return

    status.info("Processing answerline...")
    try:
        # parse difficulties
        difficulties = []
        if difficulties_text.strip():
            try:
                difficulties = [int(x) for x in difficulties_text.split(",") if x.strip().isdigit()]
            except Exception:
                logging.exception("Failed parsing difficulties")
        clues_df = process_answerline_input(
            answerline.strip(),
            searchType=search_type,
            exactPhrase=exact_phrase,
            ignoreWordOrder=ignore_word_order,
            removeParentheticals=remove_parentheticals,
            setName=set_name.strip(),
            difficulties=difficulties,
            categories=categories.strip(),
        )
        if clues_df is None or clues_df.empty:
            st.session_state.clues_df = None
            status.warning("No valid clues found for the given query.")
            return

        st.session_state.clues_df = clues_df
        status.info("Clustering...")
        stats = process_clues_dataframe(clues_df, sbert_model, threshold=threshold)
        st.session_state.cluster_stats = stats or []
        st.session_state.ranges = compute_ranges(st.session_state.cluster_stats)
        status.success("Clustering completed successfully.")
    except Exception as e:
        logging.exception("Error while processing/reclustering")
        status.error(f"Error: {e}")

def recluster_action():
    if st.session_state.clues_df is None:
        status.warning("No data to re-cluster. Process an answerline first.")
        return
    status.info("Re-clustering with new threshold...")
    try:
        stats = process_clues_dataframe(st.session_state.clues_df, sbert_model, threshold=threshold)
        st.session_state.cluster_stats = stats or []
        st.session_state.ranges = compute_ranges(st.session_state.cluster_stats)
        status.success("Re-clustering completed successfully.")
    except Exception as e:
        logging.exception("Error during re-clustering")
        status.error(f"Error: {e}")

if process_btn:
    process_answerline_action()
elif recluster_btn:
    recluster_action()

# Display clusters
st.markdown("## Clusters")
if not st.session_state.cluster_stats:
    st.info("No clusters yet — process an answerline to begin.")
else:
    df = cluster_stats_to_df(st.session_state.cluster_stats, st.session_state.ranges)

    # Allow sorting
    sort_by = st.selectbox("Sort clusters by", options=["Quality", "Avg Placement", "Cluster Size", "Avg Distance"], index=0)
    ascending = st.checkbox("Ascending", value=False)
    df_sorted = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    # Show a compact table with colored badges
    def render_cluster_table(df_display: pd.DataFrame):
        # Build html table
        html = "<table style='width:100%; border-collapse: collapse;'>"
        # header
        html += "<thead><tr>"
        headers = ["Quality", "Avg Placement", "Cluster Size", "Avg Distance", "Title Clue"]
        for h in headers:
            html += f"<th style='text-align:left; padding:6px; border-bottom: 1px solid #ddd'>{h}</th>"
        html += "</tr></thead><tbody>"
        for _, row in df_display.iterrows():
            html += "<tr>"
            html += f"<td style='padding:6px'><span style='background:{row['_c_quality']};padding:4px 8px;border-radius:6px;color:#000'>{row['Quality']:.2f}</span></td>"
            html += f"<td style='padding:6px'><span style='background:{row['_c_avg_place']};padding:4px 8px;border-radius:6px;color:#000'>{row['Avg Placement']:.2f}</span></td>"
            html += f"<td style='padding:6px'><span style='background:{row['_c_size']};padding:4px 8px;border-radius:6px;color:#000'>{row['Cluster Size']}</span></td>"
            html += f"<td style='padding:6px'><span style='background:{row['_c_dist']};padding:4px 8px;border-radius:6px;color:#000'>{row['Avg Distance']:.2f}</span></td>"
            html += f"<td style='padding:6px'>{st.session_state._escape(str(row['Title Clue'])) if hasattr(st.session_state, '_escape') else str(row['Title Clue'])}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        return html

    # streamlit doesn't allow arbitrary HTML escaping helpers; use st.markdown with unsafe_allow_html
    # But to keep things simple and safe, we'll iterate and create expanders for each cluster.
    st.markdown("### Cluster list (expand to see individual clues)")
    # Provide multi-select for download
    titles = df_sorted["Title Clue"].tolist()
    selected_titles = st.multiselect("Select clusters to save (choose by Title Clue)", options=titles, default=[])
    # Display expanders
    for idx, row in df_sorted.iterrows():
        key = f"cluster_{idx}"
        header = f"{row['Title Clue']} — Q={row['Quality']:.2f}, AvgPlace={row['Avg Placement']:.2f}, Size={row['Cluster Size']}, AvgDist={row['Avg Distance']:.2f}"
        with st.expander(header, expanded=False):
            st.write("**Representative answer:**", row["Answer"])
            st.write("**Full clues in cluster:**")
            if isinstance(row["Full Clues"], (list, tuple)):
                for c in row["Full Clues"]:
                    st.write("-", c)
            else:
                st.write(row["Full Clues"])
            # small metadata table
            meta = pd.DataFrame({
                "Quality": [row["Quality"]],
                "Avg Placement":[row["Avg Placement"]],
                "Cluster Size":[row["Cluster Size"]],
                "Avg Distance":[row["Avg Distance"]],
            })
            st.table(meta)

    # Download selected clusters as CSV
    if selected_titles:
        # build CSV data: columns Clue, Answer
        selected_rows = df_sorted[df_sorted["Title Clue"].isin(selected_titles)]
        download_df = pd.DataFrame({
            "Clue": selected_rows["Title Clue"],
            "Answer": selected_rows["Answer"]
        })
        csv_bytes = df_to_csv_bytes(download_df)
        st.download_button(
            "Download selected clusters (CSV)",
            data=csv_bytes,
            file_name="selected_clusters.csv",
            mime="text/csv",
        )
    else:
        st.info("Select clusters in the box above to enable CSV download.")

# footer / tips
st.markdown("---")
st.markdown(
    """
    **Tips:**  
    - If you update clustering code or models, re-run the app (or use Streamlit's hot-reload).  
    - To deploy: use Streamlit Community Cloud (share a public repo) or any host that supports Python.
    """
)
