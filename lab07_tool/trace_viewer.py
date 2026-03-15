"""
Trace Viewer: A simple Streamlit app for visualizing agent traces.

Usage:
    streamlit run trace_viewer.py

The app loads trace JSON files and displays the agent's decision-making
process step by step, including routing decisions, tool calls, retrieval
events, and final answers.
"""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Agent Trace Viewer", layout="wide")
st.title("Agent Trace Viewer")

# ---------------------------------------------------------------------------
# Load traces
# ---------------------------------------------------------------------------
TRACE_DIR = Path("traces")

if not TRACE_DIR.exists():
    st.warning(
        "No `traces/` directory found. Run the agent in the notebook first "
        "to generate trace files, then refresh this page."
    )
    st.stop()

trace_files = sorted(TRACE_DIR.glob("*.json"), reverse=True)

if not trace_files:
    st.warning("No trace files found in `traces/`. Run some queries in the notebook first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: select a trace
# ---------------------------------------------------------------------------
st.sidebar.header("Traces")
selected_file = st.sidebar.selectbox(
    "Select a trace",
    trace_files,
    format_func=lambda f: f.stem,
)

with open(selected_file) as f:
    trace = json.load(f)

# ---------------------------------------------------------------------------
# Display: header
# ---------------------------------------------------------------------------
st.header(f"Query: {trace.get('query', 'N/A')}")

col1, col2 = st.columns(2)
with col1:
    st.metric("Route", trace.get("route", "N/A"))
with col2:
    st.metric("Steps", len(trace.get("steps", [])))

st.subheader("Final Answer")
st.info(trace.get("final_answer", "No answer recorded."))

# ---------------------------------------------------------------------------
# Display: steps
# ---------------------------------------------------------------------------
st.subheader("Execution Steps")

STEP_ICONS = {
    "route": "🧭",
    "routing": "🧭",
    "retrieval": "📚",
    "tool_call": "🔧",
    "llm_call": "🤖",
    "error": "❌",
}

for i, step in enumerate(trace.get("steps", [])):
    step_type = step.get("step_type", "unknown")
    icon = STEP_ICONS.get(step_type, "▪️")

    with st.expander(f"{icon} Step {i + 1}: {step_type}", expanded=(i == 0)):
        if "tool" in step:
            st.write(f"**Tool:** `{step['tool']}`")
        if "query" in step:
            st.write(f"**Query:** {step['query']}")
        if "input" in step:
            st.write("**Input:**")
            st.code(json.dumps(step["input"], indent=2) if isinstance(step["input"], (dict, list)) else str(step["input"]))
        if "output" in step:
            st.write("**Output:**")
            if isinstance(step["output"], (dict, list)):
                st.json(step["output"])
            else:
                st.write(step["output"])
        if "metadata" in step:
            st.write("**Metadata:**")
            st.json(step["metadata"])

# ---------------------------------------------------------------------------
# Display: raw JSON
# ---------------------------------------------------------------------------
with st.expander("Raw Trace JSON"):
    st.json(trace)
