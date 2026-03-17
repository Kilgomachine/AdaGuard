"""AdaGuard Results Viewer — Visualize HPC experiment results locally.

Load JSON output from HPC headless runs and explore all metrics interactively.

Run with:  streamlit run app_viewer.py
"""

import os
import sys
import json
import glob

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="AdaGuard Results Viewer", page_icon="📊", layout="wide")

st.markdown("""<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c; border-radius: 10px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    .block-container { padding-top: 1.5rem; }
</style>""", unsafe_allow_html=True)

COLORS = {
    'primary': '#667eea', 'secondary': '#764ba2', 'success': '#2ecc71',
    'danger': '#e74c3c', 'warning': '#f39c12', 'info': '#3498db',
    'fisher': '#667eea', 'maskcrypt': '#e74c3c', 'none': '#95a5a6', 'full': '#2ecc71',
}


# ── Data Loading ─────────────────────────────────────────────
@st.cache_data
def load_results(file_path):
    with open(file_path) as f:
        return json.load(f)


def find_result_files(directory='.'):
    """Find all JSON result files."""
    patterns = ['*.json', 'results/*.json', 'hpc_results/*.json']
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(directory, p)))
    return sorted(files, key=os.path.getmtime, reverse=True)


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("📊 AdaGuard Results Viewer")
st.sidebar.markdown("---")

# File upload or local selection
load_method = st.sidebar.radio("Load results from:", ["Upload JSON", "Local files"])

data = None
if load_method == "Upload JSON":
    uploaded = st.sidebar.file_uploader("Upload result JSON", type=['json'], accept_multiple_files=True)
    if uploaded:
        data = {}
        for f in uploaded:
            data[f.name] = json.loads(f.read())
else:
    search_dir = st.sidebar.text_input("Search directory", value=".")
    files = find_result_files(search_dir)
    if files:
        selected = st.sidebar.multiselect("Select result files", files, default=files[:1])
        if selected:
            data = {}
            for f in selected:
                data[os.path.basename(f)] = load_results(f)
    else:
        st.sidebar.warning("No JSON files found in directory")

if data is None or len(data) == 0:
    st.title("📊 AdaGuard Results Viewer")
    st.info("Upload or select JSON result files from HPC runs to visualize them.\n\n"
            "**How to get results:**\n"
            "1. Run experiments on HPC: `sbatch hpc/slurm_single_gpu.sh`\n"
            "2. Download results: `scp maguir@hpc-login.oakland.edu:/scratch/projects/secure-distributed-ml/results/*.json ./hpc_results/`\n"
            "3. Upload or select the JSON files here")
    st.stop()


# ── Helper Functions ─────────────────────────────────────────
def rounds_to_df(rounds):
    """Convert list of round dicts to a DataFrame."""
    return pd.DataFrame(rounds)


def plot_metric_over_rounds(df, metrics, title, yaxis_title="Score"):
    """Line chart of metrics over rounds."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, m in enumerate(metrics):
        if m in df.columns:
            fig.add_trace(go.Scatter(
                x=df['round'], y=df[m],
                mode='lines+markers', name=m.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5),
            ))
    fig.update_layout(
        title=title, xaxis_title="Round", yaxis_title=yaxis_title,
        template='plotly_dark', height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def plot_gauge(value, title, max_val=1.0):
    """Create a gauge chart."""
    color = '#2ecc71' if value < 0.3 else '#f39c12' if value < 0.7 else '#e74c3c'
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=value,
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(
            axis=dict(range=[0, max_val]),
            bar=dict(color=color),
            steps=[
                dict(range=[0, 0.3], color='rgba(46,204,113,0.2)'),
                dict(range=[0.3, 0.7], color='rgba(243,156,18,0.2)'),
                dict(range=[0.7, max_val], color='rgba(231,76,60,0.2)'),
            ],
        ),
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10), template='plotly_dark')
    return fig


# ── Main Display ─────────────────────────────────────────────
for filename, result in data.items():
    st.title(f"📊 {filename}")

    meta = result.get('metadata', {})
    config = result.get('config', {})

    # ── Metadata ───────────────────────────────────
    with st.expander("Experiment Metadata", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", config.get('model', 'smallcnn'))
        col2.metric("Device", meta.get('device', 'unknown'))
        col3.metric("GPU", meta.get('gpu', 'N/A') or 'CPU')
        col4.metric("Total Params", f"{meta.get('total_params', 0):,}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy", meta.get('strategy', 'N/A'))
        col2.metric("Clients", config.get('num_clients', 'N/A'))
        col3.metric("Rounds", config.get('num_rounds', 'N/A'))
        col4.metric("Total Time", f"{meta.get('total_time_s', 0):.0f}s")

    # ── Check if this is a comparison or single run ──
    if 'strategies' in result:
        # === COMPARISON VIEW ===
        st.header("Strategy Comparison")
        strategies = result['strategies']

        # Build combined DF
        all_dfs = {}
        for strat_name, rounds in strategies.items():
            df = rounds_to_df(rounds)
            df['strategy'] = strat_name
            all_dfs[strat_name] = df

        combined = pd.concat(all_dfs.values(), ignore_index=True)

        # Q1: Privacy — LeakScore by strategy
        st.subheader("Q1: Does selective encryption stop privacy attacks?")
        fig = go.Figure()
        for strat_name, df in all_dfs.items():
            if 'combined_leakscore' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['round'], y=df['combined_leakscore'],
                    mode='lines+markers', name=strat_name.upper(),
                    line=dict(color=COLORS.get(strat_name, '#888'), width=2),
                ))
        fig.update_layout(
            title="Combined LeakScore by Strategy",
            xaxis_title="Round", yaxis_title="LeakScore",
            template='plotly_dark', height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Q2: Accuracy by strategy
        st.subheader("Q2: Does it maintain model accuracy?")
        fig = go.Figure()
        for strat_name, df in all_dfs.items():
            if 'accuracy' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['round'], y=df['accuracy'] * 100,
                    mode='lines+markers', name=strat_name.upper(),
                    line=dict(color=COLORS.get(strat_name, '#888'), width=2),
                ))
        fig.update_layout(
            title="Global Model Accuracy by Strategy",
            xaxis_title="Round", yaxis_title="Accuracy (%)",
            template='plotly_dark', height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Q3: Cost — encryption percentage
        st.subheader("Q3: Is selective encryption cheaper than full?")
        fig = go.Figure()
        for strat_name, df in all_dfs.items():
            if 'actual_pct_encrypted' in df.columns:
                fig.add_trace(go.Bar(
                    x=[strat_name.upper()],
                    y=[df['actual_pct_encrypted'].mean() * 100],
                    marker_color=COLORS.get(strat_name, '#888'),
                    name=strat_name.upper(),
                ))
        fig.update_layout(
            title="Average % Parameters Encrypted",
            yaxis_title="% Encrypted", template='plotly_dark', height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("Strategy Summary")
        summary_rows = []
        for strat_name, df in all_dfs.items():
            summary_rows.append({
                'Strategy': strat_name.upper(),
                'Final Accuracy (%)': f"{df['accuracy'].iloc[-1] * 100:.1f}" if 'accuracy' in df.columns else 'N/A',
                'Avg LeakScore': f"{df['combined_leakscore'].mean():.4f}" if 'combined_leakscore' in df.columns else 'N/A',
                'Avg % Encrypted': f"{df['actual_pct_encrypted'].mean() * 100:.1f}" if 'actual_pct_encrypted' in df.columns else 'N/A',
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    elif 'rounds' in result:
        # === SINGLE RUN VIEW ===
        rounds = result['rounds']
        df = rounds_to_df(rounds)

        # ── Pretrain History ───────────────────────
        pretrain = result.get('pretrain_history', [])
        if pretrain:
            st.header("Pre-training")
            pt_df = pd.DataFrame(pretrain)
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pt_df['epoch'], y=pt_df['loss'],
                                         mode='lines+markers', name='Loss',
                                         line=dict(color=COLORS['danger'])))
                fig.update_layout(title="Pretrain Loss", template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pt_df['epoch'], y=pt_df['accuracy'],
                                         mode='lines+markers', name='Accuracy',
                                         line=dict(color=COLORS['success'])))
                fig.update_layout(title="Pretrain Accuracy (%)", template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)

        # ── Overview Gauges ────────────────────────
        st.header("Final Round Overview")
        last = df.iloc[-1] if len(df) > 0 else {}

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(plot_gauge(last.get('combined_leakscore', 0), "LeakScore"), use_container_width=True)
        with g2:
            st.plotly_chart(plot_gauge(last.get('accuracy', 0), "Accuracy"), use_container_width=True)
        with g3:
            st.plotly_chart(plot_gauge(last.get('fisher_concentration', 0), "Fisher Conc."), use_container_width=True)
        with g4:
            st.plotly_chart(plot_gauge(last.get('actual_pct_encrypted', 0), "% Encrypted"), use_container_width=True)

        # ── Round Selector ─────────────────────────
        st.header("Per-Round Analysis")
        if len(df) > 1:
            selected_round = st.slider("Select Round", 1, len(df), len(df))
            rnd = df.iloc[selected_round - 1]
        else:
            rnd = last
            selected_round = 1

        # Per-round metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Round", int(rnd.get('round', selected_round)))
        m2.metric("Accuracy", f"{rnd.get('accuracy', 0) * 100:.1f}%")
        m3.metric("LeakScore", f"{rnd.get('combined_leakscore', 0):.4f}")
        m4.metric("Loss", f"{rnd.get('loss', 0):.4f}")
        m5.metric("Encrypted", f"{rnd.get('actual_pct_encrypted', 0) * 100:.1f}%")
        m6.metric("Time", f"{rnd.get('round_time_s', 0):.1f}s")

        # ── Accuracy & Loss Over Rounds ────────────
        st.header("Training Progress")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            if 'accuracy' in df.columns:
                fig.add_trace(go.Scatter(x=df['round'], y=df['accuracy'] * 100,
                                         mode='lines+markers', name='Accuracy',
                                         line=dict(color=COLORS['success'], width=2)))
            fig.update_layout(title="Global Model Accuracy", xaxis_title="Round",
                              yaxis_title="Accuracy (%)", template='plotly_dark', height=350)
            # Vertical line for selected round
            fig.add_vline(x=selected_round, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            if 'loss' in df.columns:
                fig.add_trace(go.Scatter(x=df['round'], y=df['loss'],
                                         mode='lines+markers', name='Loss',
                                         line=dict(color=COLORS['danger'], width=2)))
            fig.update_layout(title="Training Loss", xaxis_title="Round",
                              yaxis_title="Loss", template='plotly_dark', height=350)
            fig.add_vline(x=selected_round, line_dash="dash", line_color="white", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        # ── LeakScore Components ───────────────────
        st.header("LeakScore Components")

        # Combined LeakScore timeline
        fig = go.Figure()
        leak_metrics = ['combined_leakscore', 'entropy_avg', 'label_avg', 'empirical_avg']
        leak_colors = [COLORS['primary'], COLORS['warning'], COLORS['info'], COLORS['danger']]
        for m, c in zip(leak_metrics, leak_colors):
            if m in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['round'], y=df[m], mode='lines+markers',
                    name=m.replace('_', ' ').title(), line=dict(color=c, width=2),
                ))
        fig.add_vline(x=selected_round, line_dash="dash", line_color="white", opacity=0.5)
        fig.update_layout(title="LeakScore Over Rounds", xaxis_title="Round",
                          yaxis_title="Score", template='plotly_dark', height=400,
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Entropy sub-metrics
        entropy_cols = ['shannon_leak_score', 'renyi_leak_score', 'min_entropy_leak_score']
        available_entropy = [c for c in entropy_cols if c in df.columns]
        if available_entropy:
            st.plotly_chart(plot_metric_over_rounds(
                df, available_entropy, "Entropy LeakScore Components"
            ), use_container_width=True)

        # Label sub-metrics
        label_cols = ['glmip_score', 'confidence_gap', 'cosine_leak_score']
        available_label = [c for c in label_cols if c in df.columns]
        if available_label:
            st.plotly_chart(plot_metric_over_rounds(
                df, available_label, "Label LeakScore Components"
            ), use_container_width=True)

        # Empirical sub-metrics
        emp_cols = ['empirical_gradinversion', 'empirical_ginas', 'empirical_ggcdm']
        available_emp = [c for c in emp_cols if c in df.columns]
        if available_emp:
            st.plotly_chart(plot_metric_over_rounds(
                df, available_emp, "Empirical LeakScore (Attack Success)"
            ), use_container_width=True)

        # ── Fisher / MaskCrypt / Magnitude ─────────
        st.header("Encryption Metrics")
        enc_cols = ['fisher_concentration', 'fisher_round_norm', 'maskcrypt_vulnerability', 'magnitude_score']
        available_enc = [c for c in enc_cols if c in df.columns]
        if available_enc:
            st.plotly_chart(plot_metric_over_rounds(
                df, available_enc, "Fisher / MaskCrypt / Magnitude Over Rounds"
            ), use_container_width=True)

        # ── Per-Weight Distribution ────────────────
        if 'fisher_hist' in df.columns:
            st.header("Per-Weight Distributions")
            rnd_data = df.iloc[selected_round - 1]
            c1, c2 = st.columns(2)
            with c1:
                fh = rnd_data.get('fisher_hist')
                if fh and isinstance(fh, list):
                    fig = go.Figure(go.Bar(
                        x=list(range(len(fh))), y=fh,
                        marker_color=COLORS['fisher'],
                    ))
                    fig.update_layout(
                        title=f"Fisher Per-Weight Distribution (Round {selected_round})",
                        xaxis_title="Bin", yaxis_title="Count",
                        template='plotly_dark', height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Stats
                    st.markdown(f"**Mean:** {rnd_data.get('fisher_per_weight_mean', 0):.6f} | "
                                f"**Std:** {rnd_data.get('fisher_per_weight_std', 0):.6f} | "
                                f"**Median:** {rnd_data.get('fisher_per_weight_median', 0):.6f} | "
                                f"**P95:** {rnd_data.get('fisher_per_weight_p95', 0):.6f}")

            with c2:
                mh = rnd_data.get('maskcrypt_hist')
                if mh and isinstance(mh, list):
                    fig = go.Figure(go.Bar(
                        x=list(range(len(mh))), y=mh,
                        marker_color=COLORS['maskcrypt'],
                    ))
                    fig.update_layout(
                        title=f"MaskCrypt Per-Weight Distribution (Round {selected_round})",
                        xaxis_title="Bin", yaxis_title="Count",
                        template='plotly_dark', height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"**Mean:** {rnd_data.get('maskcrypt_per_weight_mean', 0):.6f} | "
                                f"**Std:** {rnd_data.get('maskcrypt_per_weight_std', 0):.6f} | "
                                f"**Median:** {rnd_data.get('maskcrypt_per_weight_median', 0):.6f} | "
                                f"**P95:** {rnd_data.get('maskcrypt_per_weight_p95', 0):.6f}")

        # ── Computational Cost ─────────────────────
        timing_cols = ['fisher_compute_time', 'maskcrypt_compute_time', 'entropy_compute_time',
                       'empirical_compute_time', 'glmip_compute_time']
        available_timing = [c for c in timing_cols if c in df.columns and df[c].sum() > 0]
        if available_timing:
            st.header("Computational Cost")

            c1, c2 = st.columns(2)
            with c1:
                # Timing per round
                st.plotly_chart(plot_metric_over_rounds(
                    df, available_timing, "Metric Computation Time Per Round", "Time (s)"
                ), use_container_width=True)

            with c2:
                # Average timing bar chart
                avg_times = {c.replace('_compute_time', '').title(): df[c].mean()
                             for c in available_timing}
                fig = go.Figure(go.Bar(
                    x=list(avg_times.keys()), y=list(avg_times.values()),
                    marker_color=[COLORS.get(k.lower(), '#888') for k in avg_times.keys()],
                ))
                fig.update_layout(title="Average Computation Time by Metric",
                                  yaxis_title="Time (s)", template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Per-parameter cost
            if 'fisher_per_param_us' in df.columns:
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Fisher per-param", f"{df['fisher_per_param_us'].mean():.3f} μs")
                with c2:
                    if 'maskcrypt_per_param_us' in df.columns:
                        st.metric("MaskCrypt per-param", f"{df['maskcrypt_per_param_us'].mean():.3f} μs")

        # ── Encryption Decisions ───────────────────
        if 'actual_pct_encrypted' in df.columns:
            st.header("Encryption Policy")
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['round'], y=df['actual_pct_encrypted'] * 100,
                    mode='lines+markers', name='% Encrypted',
                    fill='tozeroy', line=dict(color=COLORS['primary']),
                ))
                fig.update_layout(title="Encryption % Over Rounds", xaxis_title="Round",
                                  yaxis_title="% Encrypted", template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'params_encrypted' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['round'], y=df['params_encrypted'],
                        mode='lines+markers', name='Params Encrypted',
                        line=dict(color=COLORS['secondary']),
                    ))
                    fig.update_layout(title="Parameters Encrypted Over Rounds",
                                      xaxis_title="Round", yaxis_title="# Params",
                                      template='plotly_dark', height=350)
                    st.plotly_chart(fig, use_container_width=True)

        # ── Radar Chart ────────────────────────────
        st.header("Threat Radar")
        radar_metrics = {
            'Shannon': last.get('shannon_leak_score', 0),
            'Renyi': last.get('renyi_leak_score', 0),
            'MinEntropy': last.get('min_entropy_leak_score', 0),
            'GLMIP': last.get('glmip_score', 0),
            'ConfGap': last.get('confidence_gap', 0),
            'Cosine': last.get('cosine_leak_score', 0),
            'GradInv': last.get('empirical_gradinversion', 0),
            'GI-NAS': last.get('empirical_ginas', 0),
            'GGCDM': last.get('empirical_ggcdm', 0),
            'Magnitude': last.get('magnitude_score', 0),
        }
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', fillcolor='rgba(102,126,234,0.3)',
            line=dict(color=COLORS['primary'], width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template='plotly_dark', height=450, title="Final Round Threat Profile",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Reconstruction Quality ─────────────────
        recon_cols = ['recon_mse', 'recon_psnr', 'recon_ssim']
        available_recon = [c for c in recon_cols if c in df.columns and df[c].sum() > 0]
        if available_recon:
            st.header("Reconstruction Quality (LeakScore vs Attack Success)")
            try:
                from scipy.stats import pearsonr, spearmanr

                for metric in available_recon:
                    vals = df[metric].dropna()
                    leak_vals = df.loc[vals.index, 'combined_leakscore']
                    if len(vals) > 2:
                        pr, pp = pearsonr(leak_vals, vals)
                        sr, sp = spearmanr(leak_vals, vals)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=leak_vals, y=vals, mode='markers',
                            marker=dict(size=8, color=df.loc[vals.index, 'round'],
                                        colorscale='Viridis', showscale=True,
                                        colorbar=dict(title='Round')),
                            text=[f"Round {r}" for r in df.loc[vals.index, 'round']],
                        ))
                        fig.update_layout(
                            title=f"LeakScore vs {metric} (Pearson r={pr:.3f}, Spearman ρ={sr:.3f})",
                            xaxis_title="Combined LeakScore",
                            yaxis_title=metric.replace('recon_', '').upper(),
                            template='plotly_dark', height=350,
                        )
                        st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Install scipy for correlation analysis: `pip install scipy`")

        # ── Round Timing ───────────────────────────
        if 'round_time_s' in df.columns:
            st.header("Performance")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['round'], y=df['round_time_s'],
                marker_color=COLORS['info'],
            ))
            fig.update_layout(title="Time Per Round", xaxis_title="Round",
                              yaxis_title="Time (s)", template='plotly_dark', height=300)
            st.plotly_chart(fig, use_container_width=True)

            total = df['round_time_s'].sum()
            avg = df['round_time_s'].mean()
            st.markdown(f"**Total:** {total:.1f}s ({total/60:.1f}min) | **Average per round:** {avg:.1f}s")

        # ── Raw Data ───────────────────────────────
        with st.expander("Raw Data Table"):
            display_cols = [c for c in df.columns if not isinstance(df[c].iloc[0], (list, dict))]
            st.dataframe(df[display_cols], use_container_width=True)

    st.markdown("---")
