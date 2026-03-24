"""AdaGuard Multi-Seed Experiment Viewer — Publication-Quality Visualizations.

Load all JSON results from HPC experiments and produce cross-seed comparison
charts suitable for the paper.

Run with:  streamlit run app_experiments.py
"""

import os
import json
import glob
import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="AdaGuard Experiments", page_icon="🔬", layout="wide")

st.markdown("""<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c; border-radius: 10px; padding: 12px 16px;
    }
    .block-container { padding-top: 1rem; }
</style>""", unsafe_allow_html=True)

COLORS = {
    'none': '#95a5a6', 'fisher': '#667eea', 'maskcrypt': '#e74c3c', 'full': '#2ecc71',
}
STRATEGY_ORDER = ['none', 'fisher', 'maskcrypt', 'full']


# ── Data Loading ─────────────────────────────────────────────────

@st.cache_data
def load_experiment_dir(directory):
    """Load all JSON result files from an experiment directory.

    Expected naming: {strategy}_seed{seed}.json
    Returns dict: {strategy: {seed: data}}
    """
    files = glob.glob(os.path.join(directory, '*.json'))
    results = {}

    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse strategy and seed from filename
        match = re.match(r'(\w+)_seed(\d+)\.json', fname)
        if not match:
            continue

        strategy = match.group(1)
        seed = int(match.group(2))

        with open(fpath) as f:
            data = json.load(f)

        if strategy not in results:
            results[strategy] = {}
        results[strategy][seed] = data

    return results


def build_combined_df(results):
    """Build a single DataFrame from all strategy/seed combinations.

    Columns: round, accuracy, combined_leakscore, ..., strategy, seed
    """
    rows = []
    for strategy, seeds in results.items():
        for seed, data in seeds.items():
            rounds = data.get('rounds', [])
            for rnd in rounds:
                row = {k: v for k, v in rnd.items() if not isinstance(v, (list, dict))}
                row['strategy'] = strategy
                row['seed'] = seed
                rows.append(row)
    return pd.DataFrame(rows)


def compute_stats(df, metric, group_col='round'):
    """Compute mean and std of a metric grouped by round, per strategy."""
    stats = {}
    for strategy in df['strategy'].unique():
        sdf = df[df['strategy'] == strategy]
        grouped = sdf.groupby(group_col)[metric].agg(['mean', 'std', 'count'])
        grouped['std'] = grouped['std'].fillna(0)
        stats[strategy] = grouped
    return stats


# ── Plot Functions ───────────────────────────────────────────────

def plot_with_confidence(df, metric, title, yaxis_title, multiply=1.0, y_range=None):
    """Plot metric over rounds with mean line and shaded std band per strategy."""
    fig = go.Figure()

    for strategy in STRATEGY_ORDER:
        if strategy not in df['strategy'].unique():
            continue
        sdf = df[df['strategy'] == strategy]
        grouped = sdf.groupby('round')[metric].agg(['mean', 'std']).reset_index()
        grouped['std'] = grouped['std'].fillna(0)

        color = COLORS.get(strategy, '#888')
        y_mean = grouped['mean'] * multiply
        y_upper = (grouped['mean'] + grouped['std']) * multiply
        y_lower = (grouped['mean'] - grouped['std']) * multiply

        # Shaded band
        fig.add_trace(go.Scatter(
            x=pd.concat([grouped['round'], grouped['round'][::-1]]),
            y=pd.concat([y_upper, y_lower[::-1]]),
            fill='toself', fillcolor=color.replace(')', ',0.15)').replace('rgb', 'rgba')
                if 'rgb' in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=grouped['round'], y=y_mean,
            mode='lines+markers', name=strategy.upper(),
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title=title, xaxis_title="Round", yaxis_title=yaxis_title,
        template='plotly_dark', height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    )
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig


def plot_final_bar(df, metric, title, yaxis_title, multiply=1.0):
    """Bar chart of final-round metric value per strategy (mean +/- std across seeds)."""
    fig = go.Figure()

    for strategy in STRATEGY_ORDER:
        if strategy not in df['strategy'].unique():
            continue
        sdf = df[df['strategy'] == strategy]
        # Get last round per seed
        last_rounds = sdf.groupby('seed').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
        mean_val = last_rounds[metric].mean() * multiply
        std_val = last_rounds[metric].std() * multiply

        fig.add_trace(go.Bar(
            x=[strategy.upper()], y=[mean_val],
            error_y=dict(type='data', array=[std_val], visible=True),
            marker_color=COLORS.get(strategy, '#888'),
            name=strategy.upper(),
        ))

    fig.update_layout(
        title=title, yaxis_title=yaxis_title,
        template='plotly_dark', height=400, showlegend=False,
    )
    return fig


# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.title("🔬 Experiment Viewer")
st.sidebar.markdown("---")

load_method = st.sidebar.radio("Load from:", ["Upload JSONs", "Local directory"])

results = None
if load_method == "Upload JSONs":
    uploaded = st.sidebar.file_uploader("Upload experiment JSONs", type=['json'],
                                        accept_multiple_files=True)
    if uploaded:
        results = {}
        for f in uploaded:
            fname = f.name
            match = re.match(r'(\w+)_seed(\d+)\.json', fname)
            if match:
                strategy = match.group(1)
                seed = int(match.group(2))
                data = json.loads(f.read())
                if strategy not in results:
                    results[strategy] = {}
                results[strategy][seed] = data
else:
    default_dirs = [
        './hpc_results', './results',
        '/scratch/projects/secure-distributed-ml/results',
    ]
    existing = [d for d in default_dirs if os.path.isdir(d)]
    search_dir = st.sidebar.text_input("Results directory",
                                       value=existing[0] if existing else '.')

    # Look for experiment subdirectories
    if os.path.isdir(search_dir):
        subdirs = sorted([
            d for d in glob.glob(os.path.join(search_dir, 'experiment_*'))
            if os.path.isdir(d)
        ], reverse=True)

        # Also check for flat JSON files
        flat_jsons = glob.glob(os.path.join(search_dir, '*_seed*.json'))

        options = []
        if flat_jsons:
            options.append(search_dir + " (flat)")
        options.extend(subdirs)

        if options:
            selected_dir = st.sidebar.selectbox("Select experiment", options)
            actual_dir = selected_dir.replace(" (flat)", "")
            results = load_experiment_dir(actual_dir)

if not results or all(len(seeds) == 0 for seeds in results.values()):
    st.title("🔬 AdaGuard Multi-Seed Experiment Viewer")
    st.info(
        "Upload or select experiment JSON files to visualize.\n\n"
        "**Expected file format:** `{strategy}_seed{seed}.json`\n\n"
        "**Example:** `fisher_seed42.json`, `none_seed123.json`\n\n"
        "**How to get results:**\n"
        "1. Run on HPC: `sbatch hpc/slurm_experiments.sh`\n"
        "2. Download: `scp -r maguir@hpc-login.oakland.edu:/scratch/projects/secure-distributed-ml/results/experiment_* ./hpc_results/`\n"
        "3. Select the directory here"
    )
    st.stop()


# ── Build DataFrame ──────────────────────────────────────────────

df = build_combined_df(results)
strategies = [s for s in STRATEGY_ORDER if s in df['strategy'].unique()]
seeds = sorted(df['seed'].unique())
num_rounds = df['round'].max()

st.title("🔬 AdaGuard Experiment Results")

# Summary bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategies", len(strategies))
c2.metric("Seeds", len(seeds))
c3.metric("Rounds", int(num_rounds))
c4.metric("Total Runs", len(strategies) * len(seeds))

# Config info
sample_data = next(iter(next(iter(results.values())).values()))
config = sample_data.get('config', {})
meta = sample_data.get('metadata', {})

with st.expander("Experiment Config"):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model", config.get('model', '?'))
    c2.metric("Clients", config.get('num_clients', '?'))
    c3.metric("Per Round", config.get('clients_per_round', '?'))
    c4.metric("Local Steps", config.get('client_local_steps', '?'))
    c5.metric("GPU", meta.get('gpu', 'N/A') or 'CPU')

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Q1: PRIVACY — Does selective encryption stop attacks?
# ══════════════════════════════════════════════════════════════════

st.header("Q1: Does Selective Encryption Stop Privacy Attacks?")

c1, c2 = st.columns(2)
with c1:
    if 'combined_leakscore' in df.columns:
        fig = plot_with_confidence(df, 'combined_leakscore',
                                   "Combined LeakScore Over Rounds", "LeakScore", y_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if 'combined_leakscore' in df.columns:
        fig = plot_final_bar(df, 'combined_leakscore',
                             "Final Round LeakScore (mean +/- std)", "LeakScore")
        st.plotly_chart(fig, use_container_width=True)

# Sub-component breakdown
if 'entropy_avg' in df.columns:
    st.subheader("LeakScore Component Breakdown")
    components = ['entropy_avg', 'label_avg', 'empirical_avg']
    comp_names = ['Entropy', 'Label', 'Empirical']

    fig = make_subplots(rows=1, cols=3, subplot_titles=comp_names, shared_yaxes=True)
    for i, (comp, name) in enumerate(zip(components, comp_names)):
        if comp not in df.columns:
            continue
        for strategy in strategies:
            sdf = df[df['strategy'] == strategy]
            grouped = sdf.groupby('round')[comp].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=grouped['round'], y=grouped[comp],
                mode='lines', name=strategy.upper() if i == 0 else None,
                line=dict(color=COLORS.get(strategy, '#888'), width=2),
                showlegend=(i == 0),
            ), row=1, col=i+1)

    fig.update_layout(template='plotly_dark', height=350,
                      legend=dict(orientation='h', yanchor='bottom', y=1.1))
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Q2: UTILITY — Does it maintain model accuracy?
# ══════════════════════════════════════════════════════════════════

st.header("Q2: Does It Maintain Model Accuracy?")

c1, c2 = st.columns(2)
with c1:
    if 'accuracy' in df.columns:
        fig = plot_with_confidence(df, 'accuracy',
                                   "Model Accuracy Over Rounds", "Accuracy (%)", multiply=100)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if 'accuracy' in df.columns:
        fig = plot_final_bar(df, 'accuracy',
                             "Final Accuracy (mean +/- std)", "Accuracy (%)", multiply=100)
        st.plotly_chart(fig, use_container_width=True)

# Accuracy delta table
if 'accuracy' in df.columns:
    st.subheader("Accuracy Comparison Table")
    table_rows = []
    for strategy in strategies:
        sdf = df[df['strategy'] == strategy]
        last = sdf.groupby('seed').apply(lambda x: x.iloc[-1])
        acc_mean = last['accuracy'].mean() * 100
        acc_std = last['accuracy'].std() * 100
        table_rows.append({
            'Strategy': strategy.upper(),
            'Final Accuracy': f"{acc_mean:.2f} +/- {acc_std:.2f}%",
            'Acc Mean': acc_mean,
        })

    table_df = pd.DataFrame(table_rows)
    # Compute delta from 'none' baseline
    none_acc = table_df[table_df['Strategy'] == 'NONE']['Acc Mean'].values
    if len(none_acc) > 0:
        table_df['Delta vs No Encryption'] = table_df['Acc Mean'].apply(
            lambda x: f"{x - none_acc[0]:+.2f}%")
    st.dataframe(table_df.drop(columns=['Acc Mean']), use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Q3: COST — Is selective encryption cheaper?
# ══════════════════════════════════════════════════════════════════

st.header("Q3: Is Selective Encryption Cheaper Than Full?")

c1, c2 = st.columns(2)
with c1:
    if 'actual_pct_encrypted' in df.columns:
        fig = plot_with_confidence(df, 'actual_pct_encrypted',
                                   "% Parameters Encrypted Over Rounds",
                                   "% Encrypted", multiply=100, y_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if 'round_time_s' in df.columns:
        fig = plot_with_confidence(df, 'round_time_s',
                                   "Time Per Round", "Time (seconds)")
        st.plotly_chart(fig, use_container_width=True)

# Communication overhead estimation
if 'actual_pct_encrypted' in df.columns:
    st.subheader("Communication Overhead Estimate")
    HE_EXPANSION = 50  # typical ciphertext expansion ratio

    overhead_rows = []
    for strategy in strategies:
        sdf = df[df['strategy'] == strategy]
        avg_enc = sdf['actual_pct_encrypted'].mean()
        # overhead = enc_pct * HE_expansion + (1 - enc_pct) * 1.0
        overhead = avg_enc * HE_EXPANSION + (1 - avg_enc) * 1.0
        overhead_rows.append({
            'Strategy': strategy.upper(),
            'Avg % Encrypted': f"{avg_enc * 100:.1f}%",
            'Comm Overhead': f"{overhead:.1f}x",
            'Overhead_val': overhead,
        })

    overhead_df = pd.DataFrame(overhead_rows)
    full_overhead = overhead_df[overhead_df['Strategy'] == 'FULL']['Overhead_val'].values
    if len(full_overhead) > 0:
        overhead_df['Savings vs Full'] = overhead_df['Overhead_val'].apply(
            lambda x: f"{full_overhead[0] / max(x, 0.01):.1f}x cheaper" if x < full_overhead[0] else "baseline")

    st.dataframe(overhead_df.drop(columns=['Overhead_val']), use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# Q4: FISHER vs MASKCRYPT — Which is better?
# ══════════════════════════════════════════════════════════════════

st.header("Q4: Fisher vs MaskCrypt")

# Only show if both strategies exist
if 'fisher' in strategies and 'maskcrypt' in strategies:
    c1, c2 = st.columns(2)

    with c1:
        # Fisher concentration vs MaskCrypt vulnerability
        enc_metrics = ['fisher_concentration', 'maskcrypt_vulnerability']
        available = [m for m in enc_metrics if m in df.columns]
        if available:
            fig = go.Figure()
            for metric in available:
                for strategy in ['fisher', 'maskcrypt']:
                    if strategy not in df['strategy'].unique():
                        continue
                    sdf = df[df['strategy'] == strategy]
                    grouped = sdf.groupby('round')[metric].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=grouped['round'], y=grouped[metric],
                        mode='lines+markers',
                        name=f"{strategy.upper()} - {metric.split('_')[0].title()}",
                        line=dict(color=COLORS.get(strategy, '#888'), width=2),
                    ))
            fig.update_layout(title="Concentration / Vulnerability Scores",
                              xaxis_title="Round", yaxis_title="Score",
                              template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Side-by-side accuracy comparison
        if 'accuracy' in df.columns:
            fig = go.Figure()
            for strategy in ['fisher', 'maskcrypt']:
                sdf = df[df['strategy'] == strategy]
                grouped = sdf.groupby('round')['accuracy'].agg(['mean', 'std']).reset_index()
                fig.add_trace(go.Scatter(
                    x=grouped['round'], y=grouped['mean'] * 100,
                    mode='lines+markers', name=strategy.upper(),
                    line=dict(color=COLORS[strategy], width=2.5),
                ))
            fig.update_layout(title="Accuracy: Fisher vs MaskCrypt",
                              xaxis_title="Round", yaxis_title="Accuracy (%)",
                              template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Head-to-head table
    head2head = []
    for strategy in ['fisher', 'maskcrypt']:
        sdf = df[df['strategy'] == strategy]
        last = sdf.groupby('seed').apply(lambda x: x.iloc[-1])
        head2head.append({
            'Metric': strategy.upper(),
            'Final Accuracy': f"{last['accuracy'].mean()*100:.2f} +/- {last['accuracy'].std()*100:.2f}%",
            'Avg LeakScore': f"{sdf['combined_leakscore'].mean():.4f}" if 'combined_leakscore' in sdf.columns else 'N/A',
            'Avg % Encrypted': f"{sdf['actual_pct_encrypted'].mean()*100:.1f}%" if 'actual_pct_encrypted' in sdf.columns else 'N/A',
            'Avg Round Time': f"{sdf['round_time_s'].mean():.1f}s" if 'round_time_s' in sdf.columns else 'N/A',
        })
    st.dataframe(pd.DataFrame(head2head), use_container_width=True)

else:
    st.info("Need both Fisher and MaskCrypt results for comparison.")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# SUMMARY TABLE — Paper-ready
# ══════════════════════════════════════════════════════════════════

st.header("Summary Table (Paper-Ready)")

summary_rows = []
for strategy in strategies:
    sdf = df[df['strategy'] == strategy]
    last = sdf.groupby('seed').apply(lambda x: x.iloc[-1])

    row = {'Strategy': strategy.upper()}

    if 'accuracy' in last.columns:
        row['Accuracy (%)'] = f"{last['accuracy'].mean()*100:.2f} +/- {last['accuracy'].std()*100:.2f}"
    if 'combined_leakscore' in last.columns:
        row['LeakScore'] = f"{last['combined_leakscore'].mean():.4f} +/- {last['combined_leakscore'].std():.4f}"
    if 'actual_pct_encrypted' in sdf.columns:
        row['% Encrypted'] = f"{sdf['actual_pct_encrypted'].mean()*100:.1f}"
    if 'round_time_s' in sdf.columns:
        row['Avg Round (s)'] = f"{sdf['round_time_s'].mean():.1f}"

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# Download as CSV
csv = summary_df.to_csv(index=False)
st.download_button("Download Summary CSV", csv, "adaguard_summary.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
# RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════════════

with st.expander("Raw Data Explorer"):
    # Filter
    sel_strategies = st.multiselect("Filter strategies", strategies, default=strategies)
    sel_seeds = st.multiselect("Filter seeds", seeds, default=seeds)

    filtered = df[(df['strategy'].isin(sel_strategies)) & (df['seed'].isin(sel_seeds))]
    display_cols = [c for c in filtered.columns
                    if not isinstance(filtered[c].iloc[0], (list, dict)) if len(filtered) > 0]
    if display_cols:
        st.dataframe(filtered[display_cols], use_container_width=True)
