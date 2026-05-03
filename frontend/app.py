"""
Streamlit frontend for the Autonomous Research + Code Agent System.
Minimal, modern, dark UI with real-time step visualization.
"""

import streamlit as st
import requests
import json
import time
from typing import Any, Dict, List

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Research System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000/api/v1"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

/* Hide default chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* Header */
.sys-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1.5rem;
}
.sys-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.sys-badge {
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #22d3ee;
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.25);
    padding: 2px 8px;
    border-radius: 4px;
}

/* Query box */
.stTextArea textarea {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #22d3ee;
}
.stTextArea textarea:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 2px rgba(34,211,238,0.12) !important;
}

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #22d3ee) !important;
    color: #0a0a0f !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e293b;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    color: #22d3ee;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Agent step card */
.step-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-left: 3px solid #22d3ee;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    position: relative;
}
.step-card.success { border-left-color: #10b981; }
.step-card.error   { border-left-color: #ef4444; }
.step-card.pending { border-left-color: #f59e0b; }

.step-agent {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #22d3ee;
    font-weight: 500;
}
.step-output {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.25rem;
    line-height: 1.5;
}
.step-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    margin-top: 0.4rem;
}

/* Pipeline flow */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0;
    overflow-x: auto;
    padding-bottom: 0.5rem;
}
.pipeline-step {
    display: flex;
    align-items: center;
}
.pip-node {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    white-space: nowrap;
}
.pip-node.active  { border-color: #22d3ee; color: #22d3ee; background: rgba(34,211,238,0.06); }
.pip-node.done    { border-color: #10b981; color: #10b981; background: rgba(16,185,129,0.06); }
.pip-arrow { color: #334155; font-size: 0.8rem; padding: 0 4px; }

/* Code block */
.stCode { font-family: 'IBM Plex Mono', monospace !important; }

/* Markdown report */
.report-container {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 2rem 2.5rem;
    line-height: 1.8;
}

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 99px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
}
.score-high  { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.score-mid   { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
.score-low   { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }

.stExpander { background: #111827; border: 1px solid #1e293b !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sys-header">
  <span class="sys-title">⚡ Autonomous Agent System</span>
  <span class="sys-badge">v1.0 · Multi-Agent · RAG · Reflection</span>
</div>
""", unsafe_allow_html=True)

# ── Pipeline diagram ──────────────────────────────────────────────────────────
PIPELINE_STEPS = ["Planner", "Researcher", "Decision", "Coder", "Executor", "Critic", "Reflector", "Writer"]

def render_pipeline(completed_agents: List[str] = []):
    nodes_html = ""
    for i, step in enumerate(PIPELINE_STEPS):
        if step in completed_agents:
            cls = "done"
        else:
            cls = ""
        nodes_html += f'<div class="pipeline-step"><div class="pip-node {cls}">{step}</div>'
        if i < len(PIPELINE_STEPS) - 1:
            nodes_html += '<span class="pip-arrow">→</span>'
        nodes_html += "</div>"
    st.markdown(f'<div class="pipeline">{nodes_html}</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "running" not in st.session_state:
    st.session_state.running = False

# ── Input section ─────────────────────────────────────────────────────────────
col_input, col_opts = st.columns([4, 1])

with col_input:
    query = st.text_area(
        "Research Query",
        placeholder="e.g. Build a machine learning model to classify customer churn using a decision tree, evaluate it with accuracy and ROC-AUC",
        height=110,
        label_visibility="collapsed",
    )

with col_opts:
    max_iter = st.selectbox("Max Iterations", [1, 2, 3], index=1, label_visibility="visible")
    run_btn = st.button("▶ Run Agent", use_container_width=True)

# ── Example queries ───────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.78rem;color:#475569;margin-top:-0.5rem;">Examples: </p>', unsafe_allow_html=True)
example_cols = st.columns(3)
EXAMPLES = [
    "Train a logistic regression model on synthetic binary classification data and report accuracy, precision, recall",
    "Implement quicksort in Python and benchmark it against Python's built-in sort on 10,000 random integers",
    "Build a simple linear regression model to predict house prices using synthetic data with 5 features",
]
for col, ex in zip(example_cols, EXAMPLES):
    with col:
        if st.button(ex[:55] + "…", key=ex, use_container_width=True):
            st.session_state["prefill"] = ex

# Apply prefill
if "prefill" in st.session_state:
    query = st.session_state.pop("prefill")

render_pipeline()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn and query.strip():
    st.session_state.running = True

    with st.spinner("Running agent pipeline…"):
        t0 = time.time()
        try:
            resp = requests.post(
                f"{API_URL}/run",
                json={"query": query, "max_iterations": max_iter},
                timeout=300,
            )
            resp.raise_for_status()
            st.session_state.result = resp.json()
            st.session_state.result["_latency"] = time.time() - t0
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to API. Start the backend: `uvicorn app.main:app --reload`")
            st.session_state.result = None
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.session_state.result = None

    st.session_state.running = False

# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.result

if result:
    metrics = result.get("metrics", {})
    logs: List[Dict] = result.get("logs", [])
    completed = list({log["agent"].replace("Agent", "") for log in logs})

    # Re-render pipeline with completed steps
    render_pipeline(completed)

    # Metric row
    score = metrics.get("evaluation_score", 0)
    score_cls = "score-high" if score >= 0.7 else ("score-mid" if score >= 0.4 else "score-low")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics.get('total_latency_s', 0):.1f}s</div>
            <div class="metric-label">Total Latency</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{metrics.get('iterations', 0)}</div>
            <div class="metric-label">Iterations</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{'✅' if metrics.get('execution_success') else '❌'}</div>
            <div class="metric-label">Execution</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value"><span class="{score_cls} score-badge">{score:.2f}</span></div>
            <div class="metric-label">Quality Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_report, tab_code, tab_steps, tab_metrics = st.tabs(
        ["📄 Report", "💻 Code", "🔍 Agent Steps", "📊 Metrics"]
    )

    with tab_report:
        report = result.get("report", "")
        if report:
            st.markdown(f'<div class="report-container">', unsafe_allow_html=True)
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No report generated.")

    with tab_code:
        code = result.get("code", "")
        if code:
            st.code(code, language="python")
            st.download_button(
                "⬇ Download Code",
                data=code,
                file_name="agent_solution.py",
                mime="text/x-python",
            )
        else:
            st.warning("No code generated.")

    with tab_steps:
        st.markdown("##### Agent Execution Log")
        for log in logs:
            agent = log["agent"]
            out = log["output_summary"]
            is_err = "❌" in out or "Error" in out
            card_cls = "error" if is_err else "success"
            st.markdown(f"""
            <div class="step-card {card_cls}">
                <div class="step-agent">{agent}</div>
                <div class="step-output">{out}</div>
                <div class="step-meta">
                    Iter {log['iteration']} · {log['latency_s']:.2f}s
                    &nbsp;|&nbsp; In: {log['input_summary'][:60]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_metrics:
        st.markdown("##### Performance Breakdown")

        agent_lat = metrics.get("agent_latencies", {})
        if agent_lat:
            import json as _json
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Agent Latencies (s)**")
                for agent, lat in sorted(agent_lat.items(), key=lambda x: -x[1]):
                    pct = int(lat / max(agent_lat.values(), default=1) * 100)
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                        <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:3px;">
                            <span style="font-family:'IBM Plex Mono',monospace;color:#94a3b8">{agent}</span>
                            <span style="color:#22d3ee">{lat:.2f}s</span>
                        </div>
                        <div style="background:#1e293b;border-radius:4px;height:4px;">
                            <div style="background:#22d3ee;width:{pct}%;height:4px;border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with cols[1]:
                st.markdown("**Summary**")
                for k, v in metrics.items():
                    if k != "agent_latencies":
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
                             border-bottom:1px solid #1e293b;font-size:0.82rem;">
                            <span style="color:#64748b">{k.replace('_',' ').title()}</span>
                            <span style="font-family:'IBM Plex Mono',monospace;color:#e2e8f0">{v}</span>
                        </div>
                        """, unsafe_allow_html=True)

elif not result and not st.session_state.running:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#334155;">
        <div style="font-size:3rem;margin-bottom:1rem;">⚡</div>
        <div style="font-size:1rem;font-weight:500;color:#475569">Enter a query above and click Run Agent</div>
        <div style="font-size:0.8rem;color:#334155;margin-top:0.5rem;">
            The system will research, decide, code, execute, and evaluate automatically
        </div>
    </div>
    """, unsafe_allow_html=True)
