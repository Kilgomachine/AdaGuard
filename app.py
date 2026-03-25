"""AdaGuard Simulator — Streamlit Web UI with rich visualizations.

Run with:  python run_app.py
"""

import os
import sys

# Fix PyTorch DLL loading on Windows (must happen before importing torch)
if sys.platform == 'win32':
    import importlib.util
    _spec = importlib.util.find_spec('torch')
    if _spec and _spec.origin:
        _torch_lib = os.path.join(os.path.dirname(os.path.dirname(_spec.origin)), 'torch', 'lib')
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
            os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')
            _torch_bin = os.path.join(os.path.dirname(os.path.dirname(_spec.origin)), 'torch', 'bin')
            if os.path.isdir(_torch_bin):
                os.add_dll_directory(_torch_bin)
                os.environ['PATH'] = _torch_bin + os.pathsep + os.environ['PATH']
    del _spec

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaguard.config import DEFAULT_CONFIG, set_seed, get_device
from adaguard.models import create_model
from adaguard.data.cifar10 import load_cifar10, partition_data_non_iid
from adaguard.metrics import (
    EntropyLeakScoreMetric, GLMIPMetric, ConfidenceGapMetric,
    CosineSimilarityMetric, FisherInformationMetric, MaskCryptMetric,
    CombinedLeakScore, EmpiricalLeakScoreMetric,
    LABEL_METRICS, ENTROPY_METRICS, EMPIRICAL_METRICS, ALL_LEAK_METRICS,
)
from adaguard.encryption import AdaptiveEncryptionController
from adaguard.federation.simulator import FederatedSimulator
from adaguard.utils.gradients import extract_gradients, add_noise_to_gradients

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(page_title="AdaGuard Simulator", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c; border-radius: 10px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    .block-container { padding-top: 1.5rem; }
</style>""", unsafe_allow_html=True)

# ── GPU / Device ─────────────────────────────────────────────
device = get_device()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else None

# ── Session State ────────────────────────────────────────────
def init_state():
    defaults = {
        'config': copy.deepcopy(DEFAULT_CONFIG),
        'fl_results': [], 'comparison_results': {},
        'batch_results': {}, 'noise_results': {}, 'weight_results': {},
        'pretrain_history': [],
        'model': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
config = st.session_state.config

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    if torch.cuda.is_available():
        st.success(f"🟢 GPU: {gpu_name} ({gpu_mem})")
    else:
        st.warning("🟡 CPU mode (no CUDA GPU detected)")

    st.markdown("## ⚙️ Configuration")

    with st.expander("🔧 Federated Learning", expanded=True):
        config['num_clients'] = st.slider("Number of Clients", 2, 20, config['num_clients'])
        config['clients_per_round'] = st.slider("Clients per Round", 1, config['num_clients'],
                                                 min(config['clients_per_round'], config['num_clients']))
        config['num_rounds'] = st.slider("FL Rounds", 1, 30, config['num_rounds'])
        config['client_lr'] = st.number_input("Client Learning Rate", 0.001, 0.1,
                                                config.get('client_lr', 0.01), 0.001, format="%.3f")
        config['client_batch_size'] = st.selectbox("Client Batch Size", [1, 2, 4, 8, 16, 32, 64],
                                                    index=[1, 2, 4, 8, 16, 32, 64].index(config['client_batch_size']))

    with st.expander("📊 LeakScore Weights (α, β, γ)"):
        config['alpha'] = st.slider("α (Entropy weight)", 0.0, 3.0, config['alpha'], 0.1)
        config['beta'] = st.slider("β (Label weight)", 0.0, 3.0, config['beta'], 0.1)
        config['gamma'] = st.slider("γ (Empirical weight)", 0.0, 3.0, config['gamma'], 0.1)
        total_w = config['alpha'] + config['beta'] + config['gamma']
        if total_w > 0:
            fig_pie = go.Figure(go.Pie(
                values=[config['alpha'], config['beta'], config['gamma']],
                labels=['α Entropy', 'β Label', 'γ Empirical'],
                marker_colors=['#667eea', '#f093fb', '#4facfe'],
                hole=0.5, textinfo='percent', textfont_size=10,
            ))
            fig_pie.update_layout(height=150, margin=dict(t=5, b=5, l=5, r=5), showlegend=False)
            st.plotly_chart(fig_pie, width="stretch")

    with st.expander("🔐 Encryption Thresholds"):
        config['T1'] = st.slider("T1 (partial threshold)", 0.0, 1.0, config['T1'], 0.05)
        config['T2'] = st.slider("T2 (strong threshold)", config['T1'], 1.0,
                                  max(config['T2'], config['T1']), 0.05)
        config['encryption_top_percent'] = st.slider("Base encrypt %", 0.01, 0.5,
                                                      config['encryption_top_percent'], 0.01)
        fig_g = go.Figure(go.Indicator(
            mode="gauge", value=0.5, title={'text': 'Policy Zones', 'font': {'size': 12}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': 'rgba(0,0,0,0)'},
                'steps': [
                    {'range': [0, config['T1']], 'color': '#2ecc71'},
                    {'range': [config['T1'], config['T2']], 'color': '#f39c12'},
                    {'range': [config['T2'], 1], 'color': '#e74c3c'},
                ],
            }
        ))
        fig_g.update_layout(height=120, margin=dict(t=30, b=5, l=20, r=20))
        st.plotly_chart(fig_g, width="stretch")

    with st.expander("🧪 Attack Settings"):
        config['empirical_iterations'] = st.slider("GI iterations", 5, 100, config['empirical_iterations'])
        config['empirical_lr'] = st.number_input("Attack LR", 0.01, 1.0, config['empirical_lr'], 0.01, format="%.2f")

    with st.expander("🧠 Pre-training"):
        config['pretrain_epochs'] = st.slider("Pretrain Epochs", 1, 10, config['pretrain_epochs'])

    config['seed'] = st.number_input("Random Seed", 0, 9999, config['seed'])


# ── Header ───────────────────────────────────────────────────
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown("# 🛡️ AdaGuard Simulator")
    st.caption("Leakage-Aware Adaptive Encryption for Federated Learning")
with c2:
    st.metric("Device", gpu_name or "CPU")
    st.metric("Params", "2.19M")

# ── Helpers ──────────────────────────────────────────────────
@st.cache_resource
def load_data():
    return load_cifar10()

def results_to_df(results):
    return pd.DataFrame([{k: v for k, v in r.items() if isinstance(v, (int, float, str, bool))} for r in results])

def make_radar(values, categories, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(102, 126, 234, 0.25)',
        line=dict(color='#667eea', width=2), name=title,
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      height=350, margin=dict(t=40, b=20), title=title)
    return fig

def make_gauge(value, title, color_thresholds=None):
    if color_thresholds is None:
        color_thresholds = [config['T1'], config['T2']]
    bar_color = '#2ecc71' if value < color_thresholds[0] else '#f39c12' if value < color_thresholds[1] else '#e74c3c'
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 13}},
        number={'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.6},
            'steps': [
                {'range': [0, color_thresholds[0]], 'color': 'rgba(46,204,113,0.15)'},
                {'range': [color_thresholds[0], color_thresholds[1]], 'color': 'rgba(243,156,18,0.15)'},
                {'range': [color_thresholds[1], 1], 'color': 'rgba(231,76,60,0.15)'},
            ],
            'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.8, 'value': value},
        }
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig

def make_sankey(leak_e, leak_l, leak_emp, combined, policy_level):
    labels = ['Shannon', 'Renyi', 'MinEnt', 'GLMIP', 'ConfGap', 'Cosine',
              'GradInv', 'GI-NAS', 'GGCDM',
              'Entropy\nLeakScore', 'Label\nLeakScore', 'Empirical\nLeakScore',
              'Combined\nLeakScore', 'None', 'Partial', 'Strong']
    source = [0,1,2, 3,4,5, 6,7,8, 9,10,11]
    target = [9,9,9, 10,10,10, 11,11,11, 12,12,12]
    values = [0.33,0.33,0.33, 0.33,0.33,0.33, 0.33,0.33,0.33,
              leak_e, leak_l, leak_emp]
    policy_map = {'none': 13, 'partial': 14, 'strong': 15}
    policy_idx = policy_map.get(policy_level, 13)
    source.append(12); target.append(policy_idx); values.append(combined)
    colors = ['#667eea']*3 + ['#f093fb']*3 + ['#4facfe']*3 + ['#667eea','#f093fb','#4facfe','#8E44AD']
    node_colors = ['#667eea']*3 + ['#f093fb']*3 + ['#4facfe']*3 + ['#667eea','#f093fb','#4facfe',
                   '#8E44AD', '#2ecc71', '#f39c12', '#e74c3c']
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color='#333', width=0.5),
                  label=labels, color=node_colors),
        link=dict(source=source, target=target, value=[max(v, 0.01) for v in values],
                  color=[f'rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.3)' for c in colors]),
    ))
    fig.update_layout(title="LeakScore Data Flow (Sankey)", height=400, margin=dict(t=40, b=10))
    return fig

def make_heatmap(results_list, metric_keys, x_labels, title=""):
    data = [[r.get(m, 0) for m in metric_keys] for r in results_list]
    nice_names = [m.replace('_', ' ').replace('leak score', '').strip().title() for m in metric_keys]
    fig = go.Figure(go.Heatmap(
        z=data, x=nice_names, y=x_labels,
        colorscale='RdYlGn_r', zmin=0, zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in data], texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(title=title, height=max(250, len(x_labels) * 40 + 100),
                      margin=dict(t=40, b=10))
    return fig

def make_waterfall(metrics_dict, title="LeakScore Breakdown"):
    names = ['Shannon', 'Renyi', 'MinEnt', 'GLMIP', 'ConfGap', 'Cosine',
             'GradInv', 'GI-NAS', 'GGCDM', 'Magnitude']
    keys = ['shannon_leak_score', 'renyi_leak_score', 'min_entropy_leak_score',
            'glmip_score', 'confidence_gap', 'cosine_leak_score',
            'empirical_gradinversion', 'empirical_ginas', 'empirical_ggcdm',
            'magnitude_score']
    vals = [metrics_dict.get(k, 0) for k in keys]
    colors = ['#667eea']*3 + ['#f093fb']*3 + ['#4facfe']*3 + ['#E67E22']
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                           text=[f"{v:.3f}" for v in vals], textposition='outside'))
    fig.update_layout(title=title, height=350, yaxis_range=[0, 1.1],
                      margin=dict(t=40, b=10))
    return fig

def make_area_chart(df, cols, names, colors, title=""):
    fig = go.Figure()
    for col, name, color in zip(cols, names, colors):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['round'], y=df[col], name=name, mode='lines',
                fill='tonexty', line=dict(color=color, width=1),
            ))
    fig.update_layout(title=title, height=350, yaxis_range=[0, 1.05], margin=dict(t=40, b=10))
    return fig

def make_polar_bar(values, labels, title=""):
    fig = go.Figure(go.Barpolar(
        r=values, theta=labels,
        marker_color=px.colors.qualitative.Set2[:len(values)],
        marker_line_color='#333', marker_line_width=1, opacity=0.8,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True, tickfont_size=8)),
        title=title, height=350, margin=dict(t=50, b=10),
    )
    return fig


# ── Tabs ─────────────────────────────────────────────────────
tab_run, tab_compare, tab_weights, tab_ablation, tab_data, tab_arch = st.tabs([
    "▶️ Run Simulation", "🔄 Strategy Comparison", "⚖️ Weight Study",
    "📈 Ablations", "📋 Data Export", "🏗️ Architecture",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: RUN SIMULATION
# ═══════════════════════════════════════════════════════════════
with tab_run:
    col1, col2 = st.columns([3, 1])
    with col2:
        strategy = st.selectbox("Encryption Strategy", ['fisher', 'maskcrypt', 'none', 'full'])
        skip_glmip = st.checkbox("Skip GLMIP (saves ~30s/round)", value=False)
        skip_empirical = st.checkbox("Skip Empirical (faster)", value=False)
    with col1:
        st.markdown("### Federated Learning with AdaGuard")
        st.markdown(f"`{config['num_rounds']}` rounds · `{config['clients_per_round']}/{config['num_clients']}` "
                    f"clients · `{strategy}` · α={config['alpha']} β={config['beta']} γ={config['gamma']}")

    if st.button("🚀 Run Simulation", type="primary", use_container_width=True, key="run_sim"):
        set_seed(config['seed'])
        with st.status("Running FL simulation...", expanded=True) as status:
            st.write("📦 Loading CIFAR-10...")
            train_ds, test_ds = load_data()

            st.write("🔀 Partitioning data (non-IID)...")
            client_map = partition_data_non_iid(train_ds, config['num_clients'])

            st.write(f"🧠 Creating SmallCNN on **{device}**...")
            model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)

            sim = FederatedSimulator(model, train_ds, test_ds, client_map, config, device)

            st.write("🏋️ Pre-training model...")
            pretrain_status = st.empty()
            def pretrain_cb(epoch, total, acc, loss):
                pretrain_status.write(f"  Epoch {epoch+1}/{total} — Acc: {acc:.1f}%, Loss: {loss:.4f}")
            pretrain_hist = sim.pretrain(progress_callback=pretrain_cb)
            st.session_state.pretrain_history = pretrain_hist

            results = []
            progress = st.progress(0.0)
            live = st.empty()

            for rnd in range(config['num_rounds']):
                st.write(f"📡 Round {rnd+1}/{config['num_rounds']}...")
                summary = sim.run_round(rnd, encryption_strategy=strategy,
                                        skip_glmip=skip_glmip, skip_empirical=skip_empirical)
                results.append(summary)
                progress.progress((rnd + 1) / config['num_rounds'])
                with live.container():
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("LeakScore", f"{summary.get('combined_leakscore', 0):.3f}")
                    c2.metric("Accuracy", f"{summary.get('accuracy', 0)*100:.1f}%")
                    c3.metric("Encrypted", f"{summary.get('actual_pct_encrypted', 0)*100:.1f}%")
                    c4.metric("Policy", summary.get('encryption_level', 'n/a'))

            st.session_state.fl_results = results
            status.update(label="✅ Simulation complete!", state="complete")

    # ── Results Display ──────────────────────────────────────
    if st.session_state.fl_results:
        results = st.session_state.fl_results
        df = results_to_df(results)

        # ── Pretrain Loss/Accuracy Curves ─────────────────────
        if st.session_state.pretrain_history:
            st.markdown("### 🏋️ Pre-training Progress")
            ph = st.session_state.pretrain_history
            pc1, pc2 = st.columns(2)
            with pc1:
                fig_pl = go.Figure()
                fig_pl.add_trace(go.Scatter(x=[h['epoch'] for h in ph], y=[h['loss'] for h in ph],
                                            mode='lines+markers', line=dict(color='#E74C3C', width=2), name='Loss'))
                fig_pl.update_layout(height=250, title="Training Loss", yaxis_title="Loss", xaxis_title="Epoch",
                                     margin=dict(t=40, b=30))
                st.plotly_chart(fig_pl, width="stretch")
            with pc2:
                fig_pa = go.Figure()
                fig_pa.add_trace(go.Scatter(x=[h['epoch'] for h in ph], y=[h['accuracy'] for h in ph],
                                            mode='lines+markers', line=dict(color='#2ECC71', width=2), name='Accuracy'))
                fig_pa.update_layout(height=250, title="Training Accuracy", yaxis_title="Accuracy (%)", xaxis_title="Epoch",
                                     margin=dict(t=40, b=30))
                st.plotly_chart(fig_pa, width="stretch")

        # ── KPI Row ──────────────────────────────────────────
        st.markdown("---")
        k1, k2, k3, k4, k5 = st.columns(5)
        last = results[-1]
        k1.metric("Final Accuracy", f"{last.get('accuracy',0)*100:.1f}%",
                   delta=f"{(last.get('accuracy',0) - results[0].get('accuracy',0))*100:+.1f}%")
        k2.metric("Avg LeakScore", f"{df.get('combined_leakscore', pd.Series([0])).mean():.3f}")
        k3.metric("Avg Encrypted", f"{df.get('actual_pct_encrypted', pd.Series([0])).mean()*100:.1f}%")
        k4.metric("Fisher Conc", f"{df.get('fisher_concentration', pd.Series([0])).mean():.3f}")
        k5.metric("Magnitude", f"{df.get('magnitude_score', pd.Series([0])).mean():.3f}")

        # ── Per-Round Threat Assessment (with round selector) ─
        st.markdown("### 🎯 Per-Round Threat Assessment")
        round_select = st.slider("Select Round", 1, len(results), len(results), key="round_select")
        selected = results[round_select - 1]

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge(selected.get('combined_leakscore', 0), "Combined LeakScore"), width="stretch")
        with g2:
            st.plotly_chart(make_gauge(selected.get('entropy_avg', 0), "Entropy Score"), width="stretch")
        with g3:
            st.plotly_chart(make_gauge(selected.get('label_avg', 0), "Label Score"), width="stretch")
        with g4:
            st.plotly_chart(make_gauge(selected.get('empirical_avg', 0), "Empirical Score"), width="stretch")

        # Timeline of key scores with selected round marker
        fig_tl = go.Figure()
        for metric, color, name in [
            ('combined_leakscore', '#8E44AD', 'Combined'),
            ('entropy_avg', '#667eea', 'Entropy'),
            ('label_avg', '#f093fb', 'Label'),
            ('empirical_avg', '#4facfe', 'Empirical'),
        ]:
            if metric in df.columns:
                fig_tl.add_trace(go.Scatter(x=df['round'], y=df[metric], name=name,
                                            mode='lines+markers', line=dict(color=color, width=2)))
        fig_tl.add_vline(x=round_select, line_dash="dash", line_color="white", line_width=2,
                          annotation_text=f"R{round_select}", annotation_position="top")
        fig_tl.update_layout(height=300, yaxis_range=[-0.05, 1.05], title="LeakScore Timeline",
                              margin=dict(t=40, b=30))
        st.plotly_chart(fig_tl, width="stretch")

        # ── Radar + Waterfall (for selected round) ────────────
        st.markdown("### 🕸️ Multi-Metric Radar View")
        r1, r2 = st.columns(2)
        with r1:
            radar_cats = ['Shannon', 'Renyi', 'MinEnt', 'GLMIP', 'ConfGap', 'Cosine',
                          'GradInv', 'GI-NAS', 'GGCDM', 'Magnitude']
            radar_keys = ['shannon_leak_score', 'renyi_leak_score', 'min_entropy_leak_score',
                          'glmip_score', 'confidence_gap', 'cosine_leak_score',
                          'empirical_gradinversion', 'empirical_ginas', 'empirical_ggcdm',
                          'magnitude_score']
            radar_vals = [selected.get(k, 0) for k in radar_keys]
            st.plotly_chart(make_radar(radar_vals, radar_cats, f"Round {round_select} Metrics"), width="stretch")
        with r2:
            st.plotly_chart(make_waterfall(selected, f"Metric Breakdown (Round {round_select})"), width="stretch")

        # ── Sankey ───────────────────────────────────────────
        st.markdown("### 🔀 LeakScore Data Flow")
        st.plotly_chart(make_sankey(
            selected.get('entropy_avg', 0), selected.get('label_avg', 0), selected.get('empirical_avg', 0),
            selected.get('combined_leakscore', 0), selected.get('encryption_level', 'none'),
        ), width="stretch")

        # ── Time Series Grid ─────────────────────────────────
        st.markdown("### 📊 Per-Round Metrics")
        fig_ts = make_subplots(rows=2, cols=3,
                               subplot_titles=['Label LeakScore', 'Entropy LeakScore', 'Empirical LeakScore',
                                               'Fisher & MaskCrypt & Magnitude', 'Combined + Encryption', 'Accuracy & Loss'],
                               vertical_spacing=0.12)
        colors_s = px.colors.qualitative.Set2

        for i, (col, name) in enumerate([('glmip_score','GLMIP'),('confidence_gap','ConfGap'),('cosine_leak_score','Cosine')]):
            if col in df.columns:
                fig_ts.add_trace(go.Scatter(x=df['round'], y=df[col], name=name, mode='lines+markers',
                                            line=dict(color=colors_s[i])), row=1, col=1)
        for i, (col, name) in enumerate([('shannon_leak_score','Shannon'),('renyi_leak_score','Renyi'),('min_entropy_leak_score','MinEnt')]):
            if col in df.columns:
                fig_ts.add_trace(go.Scatter(x=df['round'], y=df[col], name=name, mode='lines+markers',
                                            line=dict(color=colors_s[i+3])), row=1, col=2)
        for i, (col, name) in enumerate([('empirical_gradinversion','GradInv'),('empirical_ginas','GI-NAS'),('empirical_ggcdm','GGCDM')]):
            if col in df.columns:
                fig_ts.add_trace(go.Scatter(x=df['round'], y=df[col], name=name, mode='lines+markers',
                                            line=dict(color=colors_s[min(i+6, len(colors_s)-1)])), row=1, col=3)
        # Fisher, MaskCrypt, Magnitude
        if 'fisher_concentration' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['fisher_concentration'], name='Fisher Conc',
                                        mode='lines+markers', line=dict(color='#2ECC71')), row=2, col=1)
        if 'fisher_round_norm' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['fisher_round_norm'], name='Fisher Norm',
                                        mode='lines+markers', line=dict(color='#27AE60', dash='dash')), row=2, col=1)
        if 'maskcrypt_vuln_score' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['maskcrypt_vuln_score'], name='MaskCrypt',
                                        mode='lines+markers', line=dict(color='#E74C3C')), row=2, col=1)
        if 'magnitude_score' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['magnitude_score'], name='Magnitude',
                                        mode='lines+markers', line=dict(color='#E67E22')), row=2, col=1)
        if 'combined_leakscore' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['combined_leakscore'], name='Combined',
                                        mode='lines+markers', line=dict(color='#8E44AD', width=3)), row=2, col=2)
        if 'actual_pct_encrypted' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['actual_pct_encrypted'], name='Encrypt%',
                                        mode='lines+markers', line=dict(color='#E67E22', dash='dash')), row=2, col=2)
        # Accuracy + Loss on same subplot
        if 'accuracy' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['accuracy']*100, name='Accuracy',
                                        mode='lines+markers', line=dict(color='#3498DB', width=3)), row=2, col=3)
        if 'loss' in df.columns:
            fig_ts.add_trace(go.Scatter(x=df['round'], y=df['loss'], name='Loss',
                                        mode='lines+markers', line=dict(color='#E74C3C', dash='dot')), row=2, col=3)
        fig_ts.update_layout(height=650, showlegend=True, legend=dict(font_size=8, orientation='h', y=-0.05))
        for i in range(1, 6):
            fig_ts.update_yaxes(range=[-0.05, 1.05], row=(i-1)//3+1, col=(i-1)%3+1)
        st.plotly_chart(fig_ts, width="stretch")

        # ── Computational Cost ────────────────────────────────
        st.markdown("### ⏱️ Computational Cost")
        time_keys = ['entropy_compute_time', 'glmip_compute_time', 'fisher_compute_time',
                     'maskcrypt_compute_time', 'empirical_compute_time']
        time_names = ['Entropy', 'GLMIP', 'Fisher', 'MaskCrypt', 'Empirical']
        time_colors = ['#667eea', '#f093fb', '#2ECC71', '#E74C3C', '#4facfe']
        avg_times = [df.get(k, pd.Series([0])).mean() * 1000 for k in time_keys]  # ms

        tc1, tc2 = st.columns(2)
        with tc1:
            fig_tc = go.Figure(go.Bar(x=time_names, y=avg_times, marker_color=time_colors,
                                       text=[f"{v:.1f}ms" for v in avg_times], textposition='outside'))
            fig_tc.update_layout(height=300, title="Avg Compute Time per Round (ms)", margin=dict(t=40, b=30))
            st.plotly_chart(fig_tc, width="stretch")
        with tc2:
            # Per-parameter cost
            fisher_pp = df.get('fisher_per_param_us', pd.Series([0])).mean()
            mc_pp = df.get('maskcrypt_per_param_us', pd.Series([0])).mean()
            fig_pp = go.Figure(go.Bar(x=['Fisher', 'MaskCrypt'], y=[fisher_pp, mc_pp],
                                       marker_color=['#2ECC71', '#E74C3C'],
                                       text=[f"{fisher_pp:.3f}μs", f"{mc_pp:.3f}μs"], textposition='outside'))
            fig_pp.update_layout(height=300, title="Per-Parameter Cost (μs)", margin=dict(t=40, b=30))
            st.plotly_chart(fig_pp, width="stretch")

        # ── Per-Weight Distributions ──────────────────────────
        st.markdown("### 📊 Per-Weight Fisher & MaskCrypt Distributions")
        sel_round_data = results[round_select - 1]
        dw1, dw2 = st.columns(2)
        with dw1:
            fisher_hist = sel_round_data.get('fisher_hist')
            if fisher_hist:
                bin_edges = np.linspace(0, 1, 101)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig_fh = go.Figure(go.Bar(x=bin_centers, y=fisher_hist, marker_color='#2ECC71', opacity=0.7))
                fig_fh.update_layout(height=300, title=f"Fisher Per-Weight Distribution (R{round_select})",
                                      xaxis_title="Normalized Fisher Value", yaxis_title="Count",
                                      margin=dict(t=40, b=30))
                st.plotly_chart(fig_fh, width="stretch")
                st.caption(f"Mean: {sel_round_data.get('fisher_per_weight_mean',0):.6f} | "
                          f"Std: {sel_round_data.get('fisher_per_weight_std',0):.6f} | "
                          f"P95: {sel_round_data.get('fisher_per_weight_p95',0):.6f}")
            else:
                st.info("No Fisher histogram data available")
        with dw2:
            mc_hist = sel_round_data.get('maskcrypt_hist')
            if mc_hist:
                mc_max = sel_round_data.get('maskcrypt_hist_max', 1)
                bin_edges = np.linspace(0, mc_max, 101)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig_mh = go.Figure(go.Bar(x=bin_centers, y=mc_hist, marker_color='#E74C3C', opacity=0.7))
                fig_mh.update_layout(height=300, title=f"MaskCrypt Per-Weight Distribution (R{round_select})",
                                      xaxis_title="|v[i]| Value", yaxis_title="Count",
                                      margin=dict(t=40, b=30))
                st.plotly_chart(fig_mh, width="stretch")
                st.caption(f"Mean: {sel_round_data.get('maskcrypt_per_weight_mean',0):.6f} | "
                          f"Std: {sel_round_data.get('maskcrypt_per_weight_std',0):.6f} | "
                          f"P95: {sel_round_data.get('maskcrypt_per_weight_p95',0):.6f}")
            else:
                st.info("No MaskCrypt histogram data available")

        # ── LeakScore ↔ Reconstruction Quality Correlation ────
        if not skip_empirical and 'recon_psnr' in df.columns and df['recon_psnr'].sum() > 0:
            st.markdown("### 📈 LeakScore vs Reconstruction Quality")
            # Compute correlations
            ls_vals = df['combined_leakscore'].values
            psnr_vals = df['recon_psnr'].values
            ssim_vals = df['recon_ssim'].values
            mse_vals = df['recon_mse'].values

            from scipy import stats as sp_stats
            cr1, cr2, cr3 = st.columns(3)
            for container, y_vals, y_name, color in [
                (cr1, psnr_vals, 'PSNR (dB)', '#3498DB'),
                (cr2, ssim_vals, 'SSIM', '#2ECC71'),
                (cr3, mse_vals, 'MSE', '#E74C3C'),
            ]:
                with container:
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Scatter(x=ls_vals, y=y_vals, mode='markers',
                                                   marker=dict(color=color, size=10), name=y_name))
                    # Trend line
                    if len(ls_vals) > 2:
                        z = np.polyfit(ls_vals, y_vals, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(ls_vals.min(), ls_vals.max(), 50)
                        fig_corr.add_trace(go.Scatter(x=x_trend, y=p(x_trend), mode='lines',
                                                       line=dict(color=color, dash='dash'), name='Trend'))
                    try:
                        pearson_r, _ = sp_stats.pearsonr(ls_vals, y_vals)
                        spearman_r, _ = sp_stats.spearmanr(ls_vals, y_vals)
                        title_text = f"LeakScore vs {y_name}<br><sub>Pearson: {pearson_r:.3f} | Spearman: {spearman_r:.3f}</sub>"
                    except Exception:
                        title_text = f"LeakScore vs {y_name}"
                    fig_corr.update_layout(height=300, title=title_text, xaxis_title="Combined LeakScore",
                                            yaxis_title=y_name, margin=dict(t=60, b=30))
                    st.plotly_chart(fig_corr, width="stretch")

        # ── Encryption Zone Chart ────────────────────────────
        st.markdown("### 🔐 Encryption Policy Decisions")
        if 'combined_leakscore' in df.columns:
            fig_z = go.Figure()
            fig_z.add_hrect(y0=0, y1=config['T1'], fillcolor="#2ecc71", opacity=0.12,
                            annotation_text="No Encryption", annotation_position="inside left")
            fig_z.add_hrect(y0=config['T1'], y1=config['T2'], fillcolor="#f39c12", opacity=0.12,
                            annotation_text="Partial Encryption", annotation_position="inside left")
            fig_z.add_hrect(y0=config['T2'], y1=1.0, fillcolor="#e74c3c", opacity=0.12,
                            annotation_text="Strong Encryption", annotation_position="inside left")
            fig_z.add_trace(go.Scatter(x=df['round'], y=df['combined_leakscore'], mode='lines+markers+text',
                                       line=dict(color='#8E44AD', width=3), name='Combined',
                                       text=[f"{v:.2f}" for v in df['combined_leakscore']],
                                       textposition='top center', textfont_size=9))
            fig_z.add_hline(y=config['T1'], line_dash="dot", line_color="#f39c12", line_width=2)
            fig_z.add_hline(y=config['T2'], line_dash="dot", line_color="#e74c3c", line_width=2)
            fig_z.update_layout(height=350, yaxis_range=[-0.05, 1.05], title="LeakScore vs Policy Thresholds")
            st.plotly_chart(fig_z, width="stretch")

        # ── Heatmap ──────────────────────────────────────────
        st.markdown("### 🗺️ Per-Round Heatmap")
        heat_keys = ALL_LEAK_METRICS + ['fisher_concentration', 'maskcrypt_vuln_score', 'magnitude_score']
        heat_data = [{k: r.get(k, 0) for k in heat_keys} for r in results]
        st.plotly_chart(make_heatmap(
            heat_data, [k for k in heat_keys if any(d.get(k, 0) != 0 for d in heat_data)],
            [f"R{r.get('round', i+1)}" for i, r in enumerate(results)],
            "All Metrics Across Rounds"
        ), width="stretch")

        # ── Area Chart ───────────────────────────────────────
        st.markdown("### 📈 Stacked Component Trends")
        if all(c in df.columns for c in ['entropy_avg', 'label_avg', 'empirical_avg']):
            st.plotly_chart(make_area_chart(
                df, ['entropy_avg', 'label_avg', 'empirical_avg'],
                ['Entropy', 'Label', 'Empirical'],
                ['#667eea', '#f093fb', '#4facfe'],
                "LeakScore Component Contributions Over Rounds"
            ), width="stretch")


# ═══════════════════════════════════════════════════════════════
# TAB 2: STRATEGY COMPARISON (Research Questions Dashboard)
# ═══════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🔄 Encryption Strategy Comparison")
    st.markdown("_Answers: Does it stop attacks? Maintain accuracy? Cost less than full encryption? How does Fisher compare to MaskCrypt?_")
    selected_strategies = st.multiselect("Strategies", ['none', 'fisher', 'maskcrypt', 'full'],
                                          default=['none', 'fisher', 'maskcrypt'])
    comp_skip_empirical = st.checkbox("Skip Empirical in comparison (faster)", value=True, key="comp_skip_emp")

    if st.button("🔄 Run Comparison", type="primary", key="run_comp"):
        set_seed(config['seed'])
        train_ds, test_ds = load_data()
        client_map = partition_data_non_iid(train_ds, config['num_clients'])
        base_model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
        all_results = {}

        with st.status("Running comparison...") as status:
            for si, strat in enumerate(selected_strategies):
                st.write(f"Strategy: **{strat}** ({si+1}/{len(selected_strategies)})")
                mc = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
                mc.load_state_dict(base_model.state_dict())
                sim = FederatedSimulator(mc, train_ds, test_ds, client_map, config, device)
                sim.pretrain()
                res = []
                for rnd in range(config['num_rounds']):
                    res.append(sim.run_round(rnd, encryption_strategy=strat,
                                             skip_glmip=True, skip_empirical=comp_skip_empirical))
                all_results[strat] = res
            st.session_state.comparison_results = all_results
            status.update(label="✅ Comparison complete!", state="complete")

    if st.session_state.comparison_results:
        all_r = st.session_state.comparison_results
        cm = {'none': '#95A5A6', 'fisher': '#2ECC71', 'maskcrypt': '#E74C3C', 'full': '#3498DB'}

        # Overview: Accuracy + LeakScore + Encrypted
        fig_comp = make_subplots(rows=1, cols=3, subplot_titles=['Accuracy (%)', 'Combined LeakScore', 'Encrypted (%)'])
        for s, res in all_r.items():
            rnds = [r['round'] for r in res]
            fig_comp.add_trace(go.Scatter(x=rnds, y=[r.get('accuracy',0)*100 for r in res], name=s.capitalize(),
                                          mode='lines+markers', line=dict(color=cm.get(s,'#333'))), row=1, col=1)
            fig_comp.add_trace(go.Scatter(x=rnds, y=[r.get('combined_leakscore',0) for r in res],
                                          name=s.capitalize(), mode='lines+markers', showlegend=False,
                                          line=dict(color=cm.get(s,'#333'))), row=1, col=2)
            fig_comp.add_trace(go.Scatter(x=rnds, y=[r.get('actual_pct_encrypted',0)*100 for r in res],
                                          name=s.capitalize(), mode='lines+markers', showlegend=False,
                                          line=dict(color=cm.get(s,'#333'))), row=1, col=3)
        fig_comp.update_yaxes(range=[-0.05, 1.05], row=1, col=2)
        fig_comp.update_layout(height=400, title_text="Strategy Comparison Dashboard")
        st.plotly_chart(fig_comp, width="stretch")

        # ── Q1: Does it stop privacy attacks? ─────────────────
        st.markdown("#### ❓ Q1: Does it stop privacy attacks?")
        if not comp_skip_empirical and any('recon_psnr' in res[-1] and res[-1].get('recon_psnr', 0) > 0 for res in all_r.values()):
            psnr_data = {s: res[-1].get('recon_psnr', 0) for s, res in all_r.items()}
            ssim_data = {s: res[-1].get('recon_ssim', 0) for s, res in all_r.items()}
            q1a, q1b = st.columns(2)
            with q1a:
                fig_q1 = go.Figure(go.Bar(x=list(psnr_data.keys()), y=list(psnr_data.values()),
                                           marker_color=[cm.get(s, '#333') for s in psnr_data.keys()],
                                           text=[f"{v:.1f}" for v in psnr_data.values()], textposition='outside'))
                fig_q1.update_layout(height=300, title="Reconstruction PSNR by Strategy (lower = safer)")
                st.plotly_chart(fig_q1, width="stretch")
            with q1b:
                fig_q1b = go.Figure(go.Bar(x=list(ssim_data.keys()), y=list(ssim_data.values()),
                                            marker_color=[cm.get(s, '#333') for s in ssim_data.keys()],
                                            text=[f"{v:.3f}" for v in ssim_data.values()], textposition='outside'))
                fig_q1b.update_layout(height=300, title="Reconstruction SSIM by Strategy (lower = safer)")
                st.plotly_chart(fig_q1b, width="stretch")
        else:
            st.info("Enable empirical attacks in comparison to see reconstruction quality metrics. "
                    "Uncheck 'Skip Empirical' above and re-run.")

        # ── Q2: Does it maintain model accuracy? ──────────────
        st.markdown("#### ❓ Q2: Does it maintain model accuracy?")
        baseline_acc = all_r.get('none', [{'accuracy': 0}])[-1].get('accuracy', 0) if 'none' in all_r else 0
        acc_data = {}
        for s, res in all_r.items():
            final_acc = res[-1].get('accuracy', 0)
            acc_data[s] = {'accuracy': final_acc * 100, 'delta': (final_acc - baseline_acc) * 100}

        fig_q2 = go.Figure()
        strats = list(acc_data.keys())
        accs = [acc_data[s]['accuracy'] for s in strats]
        deltas = [acc_data[s]['delta'] for s in strats]
        fig_q2.add_trace(go.Bar(x=strats, y=accs, marker_color=[cm.get(s, '#333') for s in strats],
                                 text=[f"{a:.1f}% ({d:+.1f}%)" for a, d in zip(accs, deltas)], textposition='outside',
                                 name='Accuracy'))
        fig_q2.update_layout(height=300, title="Final Accuracy by Strategy (vs No-Encryption Baseline)")
        st.plotly_chart(fig_q2, width="stretch")

        # ── Q3: Is it cheaper than full encryption? ───────────
        st.markdown("#### ❓ Q3: Is it cheaper than full encryption?")
        HE_EXPANSION = 50  # Homomorphic encryption expansion factor
        q3a, q3b = st.columns(2)
        with q3a:
            enc_pcts = {s: np.mean([r.get('actual_pct_encrypted', 0) for r in res]) for s, res in all_r.items()}
            overhead = {s: pct * HE_EXPANSION for s, pct in enc_pcts.items()}
            fig_q3 = go.Figure(go.Bar(x=list(overhead.keys()), y=list(overhead.values()),
                                       marker_color=[cm.get(s, '#333') for s in overhead.keys()],
                                       text=[f"{v:.1f}×" for v in overhead.values()], textposition='outside'))
            fig_q3.update_layout(height=300, title=f"Communication Overhead ({HE_EXPANSION}× HE expansion)")
            st.plotly_chart(fig_q3, width="stretch")
        with q3b:
            # Timing comparison
            timing_data = {}
            for s, res in all_r.items():
                df_s = results_to_df(res)
                t_fisher = df_s.get('fisher_compute_time', pd.Series([0])).mean() * 1000
                t_mc = df_s.get('maskcrypt_compute_time', pd.Series([0])).mean() * 1000
                timing_data[s] = {'Fisher': t_fisher, 'MaskCrypt': t_mc}
            fig_q3b = go.Figure()
            for method, color in [('Fisher', '#2ECC71'), ('MaskCrypt', '#E74C3C')]:
                fig_q3b.add_trace(go.Bar(x=list(timing_data.keys()),
                                          y=[timing_data[s][method] for s in timing_data],
                                          name=method, marker_color=color))
            fig_q3b.update_layout(height=300, title="Scoring Compute Time (ms)", barmode='group')
            st.plotly_chart(fig_q3b, width="stretch")

        # ── Q4: How does Fisher compare to MaskCrypt? ─────────
        st.markdown("#### ❓ Q4: How does Fisher compare to MaskCrypt?")
        q4a, q4b = st.columns(2)
        with q4a:
            fig_q4 = go.Figure()
            for s in ['fisher', 'maskcrypt']:
                if s in all_r:
                    res = all_r[s]
                    fig_q4.add_trace(go.Scatter(
                        x=[r['round'] for r in res],
                        y=[r.get('fisher_concentration', 0) for r in res],
                        name=f'{s.capitalize()} - Fisher Conc', mode='lines+markers',
                        line=dict(color=cm[s], width=2)))
                    fig_q4.add_trace(go.Scatter(
                        x=[r['round'] for r in res],
                        y=[r.get('maskcrypt_vuln_score', 0) for r in res],
                        name=f'{s.capitalize()} - MC Vuln', mode='lines+markers',
                        line=dict(color=cm[s], dash='dash')))
            fig_q4.update_layout(height=300, title="Fisher vs MaskCrypt Scores per Round", yaxis_range=[-0.05, 1.05])
            st.plotly_chart(fig_q4, width="stretch")
        with q4b:
            # Radar comparison
            fig_radar = go.Figure()
            cats = ['Shannon', 'Renyi', 'Fisher Conc', 'MaskCrypt Vuln', 'Accuracy']
            cat_keys = ['shannon_leak_score', 'renyi_leak_score', 'fisher_concentration',
                         'maskcrypt_vuln_score', 'accuracy']
            for s, res in all_r.items():
                last_r = res[-1]
                vals = [last_r.get(k, 0) for k in cat_keys]
                fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                                     fill='toself', name=s.capitalize(), opacity=0.6))
            fig_radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=350, title="Strategy Profiles")
            st.plotly_chart(fig_radar, width="stretch")

        # Summary table
        rows = []
        for s, res in all_r.items():
            df_s = results_to_df(res)
            rows.append({'Strategy': s.upper(),
                         'Final Acc': f"{res[-1].get('accuracy',0)*100:.1f}%",
                         'Avg LeakScore': f"{df_s.get('combined_leakscore', pd.Series([0])).mean():.3f}",
                         'Avg Encrypted%': f"{df_s.get('actual_pct_encrypted', pd.Series([0])).mean()*100:.1f}%",
                         'Comm Overhead': f"{np.mean([r.get('actual_pct_encrypted',0) for r in res])*HE_EXPANSION:.1f}×",
                         })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3: WEIGHT STUDY
# ═══════════════════════════════════════════════════════════════
with tab_weights:
    st.markdown("### ⚖️ α, β, γ Weight Study")
    presets = st.multiselect("Weight presets",
        ["(1,0,0) Entropy only", "(0,1,0) Label only", "(0,0,1) Empirical only",
         "(1,1,1) Equal", "(2,1,1) Entropy heavy", "(1,2,1) Label heavy", "(1,1,2) Empirical heavy"],
        default=["(1,0,0) Entropy only", "(0,1,0) Label only", "(1,1,1) Equal"], key="ws_presets")

    preset_map = {"(1,0,0) Entropy only":(1,0,0),"(0,1,0) Label only":(0,1,0),"(0,0,1) Empirical only":(0,0,1),
                  "(1,1,1) Equal":(1,1,1),"(2,1,1) Entropy heavy":(2,1,1),"(1,2,1) Label heavy":(1,2,1),
                  "(1,1,2) Empirical heavy":(1,1,2)}

    if st.button("🔬 Run Weight Study", type="primary", key="run_ws"):
        wcs = [preset_map[p] for p in presets]
        set_seed(config['seed'])
        train_ds, test_ds = load_data()
        client_map = partition_data_non_iid(train_ds, config['num_clients'])
        base = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
        all_r = {}
        with st.status("Running weight study...") as status:
            for a, b, g in wcs:
                st.write(f"α={a}, β={b}, γ={g}")
                cfg = copy.deepcopy(config); cfg['alpha'], cfg['beta'], cfg['gamma'] = a, b, g
                mc = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
                mc.load_state_dict(base.state_dict())
                sim = FederatedSimulator(mc, train_ds, test_ds, client_map, cfg, device)
                sim.pretrain()
                res = [sim.run_round(r, encryption_strategy='fisher', skip_glmip=True, skip_empirical=True) for r in range(cfg['num_rounds'])]
                all_r[(a, b, g)] = res
            st.session_state.weight_results = all_r
            status.update(label="✅ Done!", state="complete")

    if st.session_state.weight_results:
        all_r = st.session_state.weight_results
        fig_w = make_subplots(rows=1, cols=3, subplot_titles=['Combined LeakScore', 'Encrypted %', 'Accuracy'])
        cols_list = px.colors.qualitative.Set1
        for i, ((a, b, g), res) in enumerate(all_r.items()):
            lbl = f"α={a} β={b} γ={g}"
            rnds = [r['round'] for r in res]; c = cols_list[i % len(cols_list)]
            fig_w.add_trace(go.Scatter(x=rnds, y=[r.get('combined_leakscore',0) for r in res], name=lbl,
                                       mode='lines+markers', line=dict(color=c)), row=1, col=1)
            fig_w.add_trace(go.Scatter(x=rnds, y=[r.get('actual_pct_encrypted',0)*100 for r in res],
                                       showlegend=False, mode='lines+markers', line=dict(color=c)), row=1, col=2)
            fig_w.add_trace(go.Scatter(x=rnds, y=[r.get('accuracy',0)*100 for r in res],
                                       showlegend=False, mode='lines+markers', line=dict(color=c)), row=1, col=3)
        fig_w.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
        fig_w.update_layout(height=400, title_text="Weight Study Results")
        st.plotly_chart(fig_w, width="stretch")

        vals = [res[-1].get('combined_leakscore', 0) for res in all_r.values()]
        labels = [f"({a},{b},{g})" for a, b, g in all_r.keys()]
        st.plotly_chart(make_polar_bar(vals, labels, "Final LeakScore by Weight Config"), width="stretch")


# ═══════════════════════════════════════════════════════════════
# TAB 4: ABLATION STUDIES
# ═══════════════════════════════════════════════════════════════
with tab_ablation:
    st.markdown("### 📈 Ablation Studies")
    ab_type = st.radio("Study type", ["Batch Size", "Noise Level"], horizontal=True, key="ab_type")

    if ab_type == "Batch Size":
        batch_sizes = st.multiselect("Batch sizes", [1, 2, 4, 8, 16, 32, 64], default=[1, 4, 8, 16, 32], key="bs_sel")
        if st.button("📊 Run Batch Size Study", type="primary", key="run_bs"):
            set_seed(config['seed']); train_ds, _ = load_data()
            model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            e_m = EntropyLeakScoreMetric(num_bins=config['entropy_bins'])
            f_m = FisherInformationMetric(topk=config['fisher_topk'], enc_pct=config['encryption_top_percent'])
            results = {}
            with st.status("Running...") as status:
                from torch.utils.data import DataLoader
                for bs in batch_sizes:
                    st.write(f"Batch size: {bs}")
                    imgs, lbls = next(iter(DataLoader(train_ds, batch_size=bs, shuffle=True)))
                    gd, flat, loss_val, _ = extract_gradients(model, imgs, lbls, criterion, device)
                    m = {}; m.update(e_m.compute(flat, gradient_dict=gd, focus_layers=config['focus_layers']))
                    m.update(f_m.compute(gd)); m['loss'] = loss_val; results[bs] = m
                st.session_state.batch_results = results
                status.update(label="✅ Done!", state="complete")

        if st.session_state.batch_results:
            res = st.session_state.batch_results; bss = sorted(res.keys())
            ms = ['shannon_leak_score', 'renyi_leak_score', 'min_entropy_leak_score', 'fisher_concentration']
            fig_bs = make_subplots(rows=1, cols=4, subplot_titles=[m.replace('_',' ').title() for m in ms])
            for i, m in enumerate(ms):
                vals = [res[bs].get(m, 0) for bs in bss]
                fig_bs.add_trace(go.Bar(x=[str(b) for b in bss], y=vals, marker_color='#667eea',
                                        showlegend=False, text=[f"{v:.3f}" for v in vals], textposition='outside'), row=1, col=i+1)
                fig_bs.update_yaxes(range=[0, 1.1], row=1, col=i+1)
            fig_bs.update_layout(height=350, title_text="Batch Size Effect")
            st.plotly_chart(fig_bs, width="stretch")

            heat_keys = ['shannon_leak_score', 'renyi_leak_score', 'min_entropy_leak_score', 'fisher_concentration', 'pct_encrypted']
            st.plotly_chart(make_heatmap(
                [res[bs] for bs in bss], heat_keys, [f"BS={bs}" for bs in bss], "Batch Size Heatmap"
            ), width="stretch")

    else:
        noise_levels = st.multiselect("Noise σ", [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                                       default=[0, 0.001, 0.01, 0.05, 0.1], key="nl_sel")
        if st.button("📊 Run Noise Study", type="primary", key="run_ns"):
            set_seed(config['seed']); train_ds, _ = load_data()
            model = create_model(config.get('model', 'smallcnn'), num_classes=config['num_classes']).to(device)
            criterion = torch.nn.CrossEntropyLoss()
            e_m = EntropyLeakScoreMetric(num_bins=config['entropy_bins'])
            f_m = FisherInformationMetric(topk=config['fisher_topk'], enc_pct=config['encryption_top_percent'])
            from torch.utils.data import DataLoader
            imgs, lbls = next(iter(DataLoader(train_ds, batch_size=4, shuffle=True)))
            base_gd, base_flat, _, _ = extract_gradients(model, imgs, lbls, criterion, device)
            results = {}
            with st.status("Running...") as status:
                for sigma in noise_levels:
                    st.write(f"σ={sigma}")
                    gd, flat = (base_gd, base_flat) if sigma == 0 else add_noise_to_gradients(base_gd, sigma)
                    m = {}; m.update(e_m.compute(flat, gradient_dict=gd, focus_layers=config['focus_layers']))
                    m.update(f_m.compute(gd)); results[sigma] = m
                st.session_state.noise_results = results
                status.update(label="✅ Done!", state="complete")

        if st.session_state.noise_results:
            res = st.session_state.noise_results; nls = sorted(res.keys())
            fig_n = go.Figure()
            for mk, nm, clr in [('shannon_leak_score','Shannon','#E74C3C'),('renyi_leak_score','Renyi','#3498DB'),
                                 ('fisher_concentration','Fisher','#2ECC71')]:
                fig_n.add_trace(go.Scatter(x=[str(n) for n in nls], y=[res[n].get(mk,0) for n in nls],
                                           name=nm, mode='lines+markers', line=dict(color=clr, width=2)))
            fig_n.update_layout(height=400, title="Noise Effect on Metrics", yaxis_range=[-0.05, 1.05])
            st.plotly_chart(fig_n, width="stretch")


# ═══════════════════════════════════════════════════════════════
# TAB 5: DATA EXPORT
# ═══════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📋 Raw Results & Export")
    if st.session_state.fl_results:
        df = results_to_df(st.session_state.fl_results)
        st.dataframe(df, use_container_width=True, height=400)
        c1, c2 = st.columns(2)
        c1.download_button("📥 Download CSV", df.to_csv(index=False), "adaguard_results.csv", "text/csv")
        c2.download_button("📥 Download JSON", df.to_json(orient='records', indent=2), "adaguard_results.json", "application/json")
    else:
        st.info("Run a simulation first to see results here.")


# ═══════════════════════════════════════════════════════════════
# TAB 6: ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
with tab_arch:
    st.markdown("### 🏗️ System Architecture")

    st.markdown("#### Data Flow Diagram")
    arch_fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20,
                  line=dict(color='#333', width=0.5),
                  label=['CIFAR-10\nData', 'Client 1', 'Client 2', 'Client N',
                         'Local\nGradients', 'LeakScore\nEngine', 'Entropy\nLeakScore', 'Label\nLeakScore',
                         'Empirical\nLeakScore', 'Combined\nLeakScore', 'Encryption\nController',
                         'Fisher\nEncrypt', 'MaskCrypt\nEncrypt', 'Protected\nGradients',
                         'FedAvg\nAggregation', 'Global\nModel'],
                  color=['#3498DB', '#667eea', '#667eea', '#667eea',
                         '#f39c12', '#8E44AD', '#667eea', '#f093fb',
                         '#4facfe', '#8E44AD', '#e74c3c',
                         '#2ecc71', '#e74c3c', '#f39c12',
                         '#3498DB', '#2ecc71']),
        link=dict(
            source=[0,0,0, 1,2,3, 4,4,4, 5,5,5, 6,7,8, 9, 10,10, 11,12, 13, 14],
            target=[1,2,3, 4,4,4, 5,5,5, 6,7,8, 9,9,9, 10, 11,12, 13,13, 14, 15],
            value= [3,3,3, 3,3,3, 3,3,3, 1,1,1, 1,1,1, 3,  1.5,1.5, 3,3, 6, 6],
            color=['rgba(52,152,219,0.2)']*3 + ['rgba(102,126,234,0.2)']*3 +
                  ['rgba(243,156,18,0.2)']*3 + ['rgba(142,68,173,0.2)']*3 +
                  ['rgba(102,126,234,0.15)','rgba(240,147,251,0.15)','rgba(79,172,254,0.15)'] +
                  ['rgba(142,68,173,0.2)'] + ['rgba(231,76,60,0.2)']*2 +
                  ['rgba(46,204,113,0.2)','rgba(231,76,60,0.2)'] +
                  ['rgba(243,156,18,0.2)','rgba(52,152,219,0.2)'],
        ),
    ))
    arch_fig.update_layout(title="AdaGuard System Data Flow", height=500, margin=dict(t=40, b=10))
    st.plotly_chart(arch_fig, width="stretch")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### LeakScore Formula")
        st.latex(r"\text{LeakScore}_{final} = \frac{\alpha \cdot S_{entropy} + \beta \cdot S_{label} + \gamma \cdot S_{empirical}}{\alpha + \beta + \gamma}")
        st.markdown("#### Entropy LeakScore")
        st.latex(r"H_{Shannon} = -\sum_j p_j \log(p_j)")
        st.latex(r"S_{entropy} = 1 - \frac{H}{\log(B)}")
        st.markdown("#### Fisher Information")
        st.latex(r"F_i = g_i^2 \quad \tilde{F}_i = \frac{F_i}{\sum_j F_j}")
    with c2:
        st.markdown("#### Encryption Policy")
        st.markdown(f"""
| Condition | Policy | Action |
|-----------|--------|--------|
| S < **{config['T1']}** | None | No encryption |
| **{config['T1']}** ≤ S < **{config['T2']}** | Partial | Top-k by Fisher/MaskCrypt |
| S ≥ **{config['T2']}** | Strong | Aggressive + validate |
        """)
        st.markdown("#### MaskCrypt Vulnerability (Paper)")
        st.latex(r"v_i = g_i \times (\tilde{w}_i^{exp} - w_i^{trained})")
        st.markdown("#### Gradient Magnitude Score")
        st.latex(r"S_{mag} = \frac{||g||_2}{||g||_2 + 1}")

    st.markdown("---")
    st.markdown("#### GPU Information")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        g1, g2, g3 = st.columns(3)
        g1.metric("GPU", gpu_name)
        g2.metric("VRAM", f"{props.total_memory / 1024**3:.1f} GB")
        g3.metric("CUDA Cores", f"{props.multi_processor_count} SMs")
    else:
        st.info("No GPU detected. Install PyTorch with CUDA for GPU acceleration.")
