#!/usr/bin/env python3
"""Generate AdaGuard Projected Results Report — expected outcomes with graphs."""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image
)

# ── Color palette ──────────────────────────────────────────────────────
C_DARK   = '#1a1a2e'
C_BLUE   = '#0984e3'
C_GREEN  = '#00b894'
C_RED    = '#d63031'
C_ORANGE = '#e17055'
C_PURPLE = '#6c5ce7'
C_TEAL   = '#00cec9'
C_GRAY   = '#636e72'


def fig_to_image(fig, width=6.5*inch, height=3.2*inch):
    """Convert matplotlib figure to ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def style_ax(ax, xlabel, ylabel, title=None):
    """Apply consistent styling to axes."""
    ax.set_xlabel(xlabel, fontsize=9, color='#333')
    ax.set_ylabel(ylabel, fontsize=9, color='#333')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=C_DARK, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ccc')
    ax.spines['bottom'].set_color('#ccc')
    ax.tick_params(colors='#555', labelsize=8)
    ax.grid(axis='y', alpha=0.3, color='#ccc')


# ══════════════════════════════════════════════════════════════════════
# PROJECTION CHARTS
# ══════════════════════════════════════════════════════════════════════

def chart_s1_entropy_bins():
    """S1: Entropy bins — scores should stabilize around 25-100."""
    bins = [10, 25, 50, 100, 200]
    # Projected: scores noisy at low bins, stable in middle, slight noise at high
    shannon  = [0.58, 0.64, 0.65, 0.66, 0.65]
    renyi    = [0.52, 0.60, 0.62, 0.63, 0.62]
    minent   = [0.48, 0.55, 0.57, 0.58, 0.57]
    avg      = [0.53, 0.60, 0.61, 0.62, 0.61]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(bins, shannon, 'o-', color=C_BLUE, label='Shannon', linewidth=2, markersize=6)
    ax1.plot(bins, renyi, 's-', color=C_GREEN, label='Renyi (a=2)', linewidth=2, markersize=6)
    ax1.plot(bins, minent, '^-', color=C_RED, label='MinEntropy', linewidth=2, markersize=6)
    ax1.plot(bins, avg, 'D--', color=C_GRAY, label='EntropyAvg', linewidth=1.5, markersize=5)
    ax1.axvline(x=50, color=C_ORANGE, linestyle=':', alpha=0.7, label='Default (50)')
    ax1.fill_between([25, 100], 0.4, 0.75, alpha=0.08, color=C_GREEN)
    ax1.set_xscale('log')
    ax1.set_xticks(bins)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.legend(fontsize=7, loc='lower right')
    style_ax(ax1, 'Entropy Bins', 'Score (0=safe, 1=risky)', 'Entropy Scores vs Bin Count')
    ax1.set_ylim(0.4, 0.75)

    # Variance chart
    variance = [0.025, 0.010, 0.008, 0.007, 0.009]
    ax2.bar(range(len(bins)), variance, color=[C_RED, C_ORANGE, C_GREEN, C_GREEN, C_ORANGE],
            alpha=0.8, edgecolor='white')
    ax2.set_xticks(range(len(bins)))
    ax2.set_xticklabels([str(b) for b in bins])
    style_ax(ax2, 'Entropy Bins', 'Score Variance Across Clients', 'Score Stability')
    ax2.set_ylim(0, 0.035)

    fig.tight_layout(pad=2)
    return fig


def chart_s2_grad_accum():
    """S2: Gradient accumulation K — privacy vs compute cost."""
    K = [1, 2, 4, 8]
    beff = [4, 8, 16, 32]
    # Attack success rate drops sharply
    asr_gi   = [0.82, 0.45, 0.08, 0.01]
    asr_ginas = [0.75, 0.38, 0.05, 0.005]
    asr_ggcdm = [0.70, 0.35, 0.06, 0.008]
    # SSIM (lower = better defense)
    ssim     = [0.78, 0.42, 0.12, 0.04]
    # Compute cost
    cost     = [1, 2, 4, 8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(K, asr_gi, 'o-', color=C_RED, label='GradInversion ASR', linewidth=2, markersize=7)
    ax1.plot(K, asr_ginas, 's-', color=C_ORANGE, label='GI-NAS ASR', linewidth=2, markersize=7)
    ax1.plot(K, asr_ggcdm, '^-', color=C_PURPLE, label='GGCDM ASR', linewidth=2, markersize=7)
    ax1.plot(K, ssim, 'D--', color=C_GRAY, label='Mean SSIM', linewidth=1.5, markersize=5)
    ax1.axhline(y=0.1, color=C_GREEN, linestyle=':', alpha=0.6, label='ASR < 10% (strong)')
    ax1.fill_between([3.5, 8.5], 0, 0.1, alpha=0.08, color=C_GREEN)
    ax1.set_xticks(K)
    ax1.set_xticklabels([f'K={k}\n(B_eff={b})' for k, b in zip(K, beff)])
    ax1.legend(fontsize=7, loc='upper right')
    style_ax(ax1, 'Accumulation K', 'Attack Success / SSIM', 'Privacy Gain vs K')
    ax1.set_ylim(-0.02, 0.95)

    # Cost-benefit
    privacy_gain = [0, 0.48, 0.90, 0.98]  # normalized
    colors = [C_RED, C_ORANGE, C_GREEN, C_BLUE]
    bars = ax2.bar(range(len(K)), cost, color=colors, alpha=0.35, label='Compute Cost (x)',
                   edgecolor='white', width=0.4)
    ax2b = ax2.twinx()
    ax2b.plot(range(len(K)), privacy_gain, 'ko-', linewidth=2, markersize=8, label='Privacy Gain')
    ax2b.set_ylabel('Privacy Gain (normalized)', fontsize=9, color='#333')
    ax2b.set_ylim(-0.05, 1.1)
    ax2b.spines['top'].set_visible(False)
    ax2b.tick_params(colors='#555', labelsize=8)
    ax2.set_xticks(range(len(K)))
    ax2.set_xticklabels([f'K={k}' for k in K])
    style_ax(ax2, 'Accumulation K', 'Compute Cost (x baseline)', 'Cost-Benefit Tradeoff')
    ax2.legend(fontsize=7, loc='upper left')
    ax2b.legend(fontsize=7, loc='center right')

    fig.tight_layout(pad=2)
    return fig


def chart_s3s4_thresholds():
    """S3-S4: T1 and T2 threshold sweeps."""
    # S3: T1 sweep (T2=0.7)
    t1_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    t1_encrypt_pct = [42, 28, 18, 11, 6]   # % clients getting any encryption
    t1_accuracy    = [88.2, 89.5, 90.3, 90.8, 91.1]  # model accuracy
    t1_asr         = [0.05, 0.10, 0.15, 0.25, 0.38]  # attack success

    # S4: T2 sweep (T1=0.3)
    t2_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
    t2_strong_pct  = [35, 22, 14, 8, 3]    # % clients getting strong encryption
    t2_accuracy    = [88.8, 89.6, 90.3, 90.7, 91.0]
    t2_asr         = [0.04, 0.08, 0.15, 0.22, 0.32]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # S3 chart
    color_acc = C_BLUE
    color_asr = C_RED
    color_enc = C_GREEN

    l1 = ax1.plot(t1_vals, t1_asr, 'o-', color=color_asr, linewidth=2, markersize=7, label='Attack Success Rate')
    ax1b = ax1.twinx()
    l2 = ax1b.plot(t1_vals, t1_accuracy, 's-', color=color_acc, linewidth=2, markersize=7, label='Model Accuracy (%)')
    l3 = ax1b.bar(t1_vals, [e/100 for e in t1_encrypt_pct], width=0.06, alpha=0.25,
                  color=color_enc, label='% Clients Encrypted')
    ax1.axvline(x=0.3, color=C_ORANGE, linestyle=':', alpha=0.7)
    ax1.annotate('Default', xy=(0.3, 0.42), fontsize=7, color=C_ORANGE, ha='center')
    style_ax(ax1, 'T1 Threshold', 'Attack Success Rate', 'S3: Partial Encryption Threshold T1')
    ax1.set_ylim(-0.02, 0.5)
    ax1b.set_ylim(87, 92)
    ax1b.set_ylabel('Model Accuracy (%)', fontsize=9, color=color_acc)
    ax1b.spines['top'].set_visible(False)
    ax1b.tick_params(colors='#555', labelsize=8)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=7, loc='center left')

    # S4 chart
    l1 = ax2.plot(t2_vals, t2_asr, 'o-', color=color_asr, linewidth=2, markersize=7, label='Attack Success Rate')
    ax2b = ax2.twinx()
    l2 = ax2b.plot(t2_vals, t2_accuracy, 's-', color=color_acc, linewidth=2, markersize=7, label='Model Accuracy (%)')
    ax2b.bar(t2_vals, [e/100 for e in t2_strong_pct], width=0.06, alpha=0.25,
             color=color_enc, label='% Strong Encrypted')
    ax2.axvline(x=0.7, color=C_ORANGE, linestyle=':', alpha=0.7)
    ax2.annotate('Default', xy=(0.7, 0.35), fontsize=7, color=C_ORANGE, ha='center')
    style_ax(ax2, 'T2 Threshold', 'Attack Success Rate', 'S4: Strong Encryption Threshold T2')
    ax2.set_ylim(-0.02, 0.42)
    ax2b.set_ylim(87, 92)
    ax2b.set_ylabel('Model Accuracy (%)', fontsize=9, color=color_acc)
    ax2b.spines['top'].set_visible(False)
    ax2b.tick_params(colors='#555', labelsize=8)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=7, loc='center left')

    fig.tight_layout(pad=2)
    return fig


def chart_s5s6s7_weights():
    """S5-S7: LeakScore weight sweeps."""
    weights = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    # S5: alpha (label weight) — detection accuracy
    s5_detect = [0.55, 0.72, 0.82, 0.85, 0.84, 0.82, 0.80, 0.78]
    # S6: beta (entropy weight)
    s6_detect = [0.60, 0.74, 0.82, 0.84, 0.83, 0.81, 0.79, 0.77]
    # S7: gamma (empirical weight)
    s7_detect = [0.82, 0.84, 0.86, 0.87, 0.87, 0.86, 0.85, 0.84]

    # Compute cost for gamma
    s7_cost   = [1.0, 1.8, 2.5, 3.0, 3.2, 3.4, 3.5, 3.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(weights, s5_detect, 'o-', color=C_BLUE, linewidth=2, markersize=6,
             label='S5: alpha (label)')
    ax1.plot(weights, s6_detect, 's-', color=C_GREEN, linewidth=2, markersize=6,
             label='S6: beta (entropy)')
    ax1.plot(weights, s7_detect, '^-', color=C_PURPLE, linewidth=2, markersize=6,
             label='S7: gamma (empirical)')
    ax1.axvline(x=1.0, color=C_ORANGE, linestyle=':', alpha=0.7, label='Default (1.0)')
    ax1.fill_between(weights, [d-0.03 for d in s7_detect], [d+0.03 for d in s7_detect],
                     alpha=0.08, color=C_PURPLE)
    ax1.legend(fontsize=7)
    style_ax(ax1, 'Weight Value', 'LeakScore Detection Accuracy',
             'Detection Accuracy vs Component Weight')
    ax1.set_ylim(0.45, 0.95)

    # Cost-benefit for gamma specifically
    ax2.plot(weights, s7_detect, 'o-', color=C_PURPLE, linewidth=2, markersize=7,
             label='Detection Accuracy')
    ax2b = ax2.twinx()
    ax2b.fill_between(weights, s7_cost, alpha=0.2, color=C_RED)
    ax2b.plot(weights, s7_cost, '--', color=C_RED, linewidth=1.5, label='Compute Cost (x)')
    ax2b.set_ylabel('Relative Compute Cost', fontsize=9, color=C_RED)
    ax2b.set_ylim(0, 5)
    ax2b.spines['top'].set_visible(False)
    ax2b.tick_params(colors='#555', labelsize=8)
    style_ax(ax2, 'gamma (Empirical Weight)', 'Detection Accuracy',
             'S7: Is Empirical Worth the Cost?')
    ax2.set_ylim(0.75, 0.92)
    ax2.legend(fontsize=7, loc='lower left')
    ax2b.legend(fontsize=7, loc='upper left')
    ax2.annotate('Marginal gain\nvs large cost', xy=(2.5, 0.865), fontsize=7,
                 color=C_RED, ha='center', style='italic')

    fig.tight_layout(pad=2)
    return fig


def chart_s8_batch_size():
    """S8: Batch size vs leakability."""
    bs = [1, 4, 8, 16, 32]
    asr_gi    = [0.95, 0.72, 0.35, 0.08, 0.01]
    asr_ginas = [0.92, 0.65, 0.28, 0.06, 0.005]
    asr_ggcdm = [0.88, 0.60, 0.30, 0.07, 0.008]
    leakscore = [0.92, 0.68, 0.45, 0.28, 0.15]
    ssim      = [0.92, 0.65, 0.30, 0.10, 0.03]
    accuracy  = [89.5, 90.3, 90.8, 91.2, 91.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(bs, asr_gi, 'o-', color=C_RED, linewidth=2, markersize=7, label='GradInversion ASR')
    ax1.plot(bs, asr_ginas, 's-', color=C_ORANGE, linewidth=2, markersize=7, label='GI-NAS ASR')
    ax1.plot(bs, asr_ggcdm, '^-', color=C_PURPLE, linewidth=2, markersize=7, label='GGCDM ASR')
    ax1.plot(bs, leakscore, 'D--', color=C_BLUE, linewidth=2, markersize=6, label='LeakScore')
    ax1.fill_between([12, 35], 0, 0.1, alpha=0.08, color=C_GREEN)
    ax1.axhline(y=0.1, color=C_GREEN, linestyle=':', alpha=0.5)
    ax1.annotate('Safe zone (ASR < 10%)', xy=(22, 0.12), fontsize=7, color=C_GREEN)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(bs)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.legend(fontsize=7)
    style_ax(ax1, 'Batch Size', 'Score / Rate', 'Attack Success vs Batch Size')
    ax1.set_ylim(-0.02, 1.0)

    # LeakScore correlation with actual attack success
    ax2.scatter(leakscore, asr_gi, s=100, c=C_RED, marker='o', label='GradInversion', zorder=5)
    ax2.scatter(leakscore, asr_ginas, s=100, c=C_ORANGE, marker='s', label='GI-NAS', zorder=5)
    ax2.scatter(leakscore, asr_ggcdm, s=100, c=C_PURPLE, marker='^', label='GGCDM', zorder=5)
    # Ideal correlation line
    ax2.plot([0, 1], [0, 1], '--', color=C_GRAY, alpha=0.4, label='Perfect correlation')
    for i, b in enumerate(bs):
        ax2.annotate(f'B={b}', (leakscore[i]+0.02, asr_gi[i]-0.03), fontsize=7, color='#555')
    ax2.legend(fontsize=7)
    style_ax(ax2, 'LeakScore (predicted risk)', 'Actual Attack Success Rate',
             'LeakScore Calibration')
    ax2.set_xlim(-0.02, 1.0)
    ax2.set_ylim(-0.02, 1.0)

    fig.tight_layout(pad=2)
    return fig


def chart_s9_samples():
    """S9: GLMIP samples per class."""
    samples = [5, 10, 20, 40]
    glmip_mean = [0.62, 0.67, 0.70, 0.71]
    glmip_std  = [0.12, 0.06, 0.04, 0.03]
    compute_ms = [15, 30, 60, 120]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.errorbar(samples, glmip_mean, yerr=glmip_std, fmt='o-', color=C_BLUE,
                 linewidth=2, markersize=8, capsize=5, capthick=1.5,
                 label='GLMIP Score (mean +/- std)')
    ax1.axvline(x=20, color=C_ORANGE, linestyle=':', alpha=0.7, label='Default (20)')
    ax1.fill_between([10, 40], 0.5, 0.8, alpha=0.06, color=C_GREEN)
    ax1.annotate('Stable region', xy=(22, 0.52), fontsize=7, color=C_GREEN)
    ax1.legend(fontsize=7)
    style_ax(ax1, 'Samples Per Class', 'GLMIP Score', 'GLMIP Accuracy vs Sampling Density')
    ax1.set_ylim(0.45, 0.82)
    ax1.set_xticks(samples)

    # Cost
    colors = [C_ORANGE, C_BLUE, C_GREEN, C_RED]
    ax2.bar(range(len(samples)), compute_ms, color=colors, alpha=0.7, edgecolor='white')
    ax2.set_xticks(range(len(samples)))
    ax2.set_xticklabels([str(s) for s in samples])
    for i, (v, c) in enumerate(zip(compute_ms, compute_ms)):
        ax2.text(i, v + 3, f'{v}ms', ha='center', fontsize=8, color='#333')
    style_ax(ax2, 'Samples Per Class', 'Compute Time (ms/client)', 'GLMIP Compute Cost')
    ax2.set_ylim(0, 150)

    fig.tight_layout(pad=2)
    return fig


def chart_s10_maskcrypt_rho():
    """S10: MaskCrypt encrypt ratio."""
    rho = [0.01, 0.05, 0.10, 0.20, 0.25]
    rho_pct = ['1%', '5%', '10%', '20%', '25%']

    asr_gi    = [0.08, 0.03, 0.01, 0.005, 0.003]
    asr_ginas = [0.12, 0.04, 0.02, 0.008, 0.005]
    ssim      = [0.15, 0.06, 0.03, 0.02, 0.015]
    accuracy  = [91.0, 90.6, 90.2, 89.5, 89.0]
    comm_cost = [1.01, 1.05, 1.10, 1.20, 1.25]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    ax1.plot(rho, asr_gi, 'o-', color=C_RED, linewidth=2, markersize=7, label='GradInversion ASR')
    ax1.plot(rho, asr_ginas, 's-', color=C_ORANGE, linewidth=2, markersize=7, label='GI-NAS ASR')
    ax1.plot(rho, ssim, '^-', color=C_PURPLE, linewidth=2, markersize=7, label='Mean SSIM')
    ax1.axhline(y=0.1, color=C_GREEN, linestyle=':', alpha=0.5)
    ax1.axvline(x=0.10, color=C_ORANGE, linestyle=':', alpha=0.7, label='Default (10%)')
    ax1.annotate('1% already blocks\nmost attacks', xy=(0.015, 0.16), fontsize=7,
                 color=C_BLUE, style='italic')
    ax1.legend(fontsize=7)
    style_ax(ax1, 'Encrypt Ratio (rho)', 'Attack Success / SSIM',
             'MaskCrypt: Privacy vs Encrypt Ratio')
    ax1.set_ylim(-0.01, 0.25)

    # Cost-accuracy tradeoff
    ax2.plot(rho, accuracy, 'o-', color=C_BLUE, linewidth=2, markersize=7, label='Model Accuracy (%)')
    ax2b = ax2.twinx()
    ax2b.bar(rho, [c - 1 for c in comm_cost], width=0.015, alpha=0.35,
             color=C_RED, label='Comm Overhead (%)')
    ax2b.set_ylabel('Communication Overhead', fontsize=9, color=C_RED)
    ax2b.set_ylim(0, 0.35)
    ax2b.spines['top'].set_visible(False)
    ax2b.tick_params(colors='#555', labelsize=8)
    ax2.axvline(x=0.10, color=C_ORANGE, linestyle=':', alpha=0.7)
    style_ax(ax2, 'Encrypt Ratio (rho)', 'Model Accuracy (%)',
             'MaskCrypt: Cost vs Encrypt Ratio')
    ax2.set_ylim(88, 92)
    ax2.legend(fontsize=7, loc='lower left')
    ax2b.legend(fontsize=7, loc='upper right')

    fig.tight_layout(pad=2)
    return fig


def chart_s11_focus_layers():
    """S11: Focus layers for GLMIP."""
    layers = ['All Layers', 'Final FC\n(default)', 'Penultimate FC']
    glmip   = [0.58, 0.70, 0.52]
    confgap = [0.65, 0.65, 0.65]  # not affected by focus_layers
    cossim  = [0.55, 0.55, 0.55]  # not affected
    detect  = [0.72, 0.82, 0.68]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    x = np.arange(len(layers))
    width = 0.22
    ax1.bar(x - width, glmip, width, color=C_BLUE, alpha=0.8, label='GLMIP', edgecolor='white')
    ax1.bar(x, confgap, width, color=C_GREEN, alpha=0.8, label='ConfidenceGap', edgecolor='white')
    ax1.bar(x + width, cossim, width, color=C_PURPLE, alpha=0.8, label='CosineSimilarity', edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend(fontsize=7)
    style_ax(ax1, 'Focus Layers', 'Metric Score', 'Label Metrics by Layer Selection')
    ax1.set_ylim(0, 0.9)

    colors = [C_ORANGE, C_GREEN, C_GRAY]
    ax2.bar(x, detect, color=colors, alpha=0.8, edgecolor='white', width=0.5)
    for i, v in enumerate(detect):
        ax2.text(i, v + 0.015, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='#333')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    style_ax(ax2, 'Focus Layers', 'Overall Detection Accuracy',
             'Detection Accuracy by Layer Selection')
    ax2.set_ylim(0, 0.95)
    ax2.annotate('iDLG theorem:\nfinal FC has\nstrongest signal', xy=(1, 0.60),
                 fontsize=7, color=C_BLUE, ha='center', style='italic')

    fig.tight_layout(pad=2)
    return fig


def chart_viability_privacy():
    """Viability: Privacy comparison across all scenarios."""
    scenarios = ['V1\nNo Def', 'V2\nFull HE', 'V3\nDP', 'V4\nMC-Guide',
                 'V5\nMC-Rand', 'V6\nAdaG', 'V7\nAdaG+K', 'V8\nAggr', 'V9\nConserv']
    asr = [0.82, 0.0, 0.15, 0.03, 0.18, 0.06, 0.02, 0.01, 0.20]
    colors = [C_RED, C_GREEN, C_ORANGE, C_PURPLE, C_GRAY,
              C_BLUE, C_TEAL, C_GREEN, C_ORANGE]

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    bars = ax.bar(range(len(scenarios)), asr, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(y=0.1, color=C_GREEN, linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate('10% ASR threshold', xy=(8.2, 0.11), fontsize=7, color=C_GREEN)
    for i, v in enumerate(asr):
        ax.text(i, v + 0.015, f'{v:.0%}', ha='center', fontsize=7.5, fontweight='bold', color='#333')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=7.5)
    style_ax(ax, '', 'Attack Success Rate (lower = better defense)',
             'Viability Study: Privacy Comparison (Projected)')
    ax.set_ylim(0, 0.95)

    fig.tight_layout(pad=2)
    return fig


def chart_viability_utility_cost():
    """Viability: Utility vs Cost tradeoff."""
    scenarios = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']
    accuracy  = [91.5, 85.0, 87.5, 90.2, 90.5, 90.3, 89.8, 88.2, 91.0]
    encrypt_pct = [0, 100, 0, 10, 10, 18, 22, 45, 6]
    asr = [0.82, 0.0, 0.15, 0.03, 0.18, 0.06, 0.02, 0.01, 0.20]
    labels_full = ['No Defense', 'Full HE', 'DP', 'MC-Guided', 'MC-Random',
                   'AdaGuard', 'AdaG+Accum', 'AdaG-Aggr', 'AdaG-Conserv']
    colors = [C_RED, '#555', C_ORANGE, C_PURPLE, C_GRAY,
              C_BLUE, C_TEAL, C_GREEN, '#daa520']

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.0))

    for i in range(len(scenarios)):
        size = max(200, 1200 * (1 - asr[i]))  # bigger = better privacy
        ax.scatter(encrypt_pct[i], accuracy[i], s=size, c=colors[i],
                   alpha=0.7, edgecolors='white', linewidths=1.5, zorder=5)
        ax.annotate(labels_full[i], (encrypt_pct[i], accuracy[i]),
                    textcoords='offset points', xytext=(8, 5),
                    fontsize=7, color='#333')

    # Ideal region
    ax.fill_between([0, 25], 89, 92.5, alpha=0.06, color=C_GREEN)
    ax.annotate('Ideal zone:\nhigh accuracy,\nlow cost', xy=(5, 91.8),
                fontsize=7, color=C_GREEN, style='italic')

    style_ax(ax, '% Parameters Encrypted (cost)', 'Model Accuracy (%)',
             'Utility vs Cost (bubble size = privacy strength)')
    ax.set_xlim(-3, 110)
    ax.set_ylim(83, 93)

    fig.tight_layout(pad=2)
    return fig


def chart_viability_scalability():
    """V10-V12: Scalability projections."""
    configs = ['Default\n(10 clients)', 'V10\n(100 clients)', 'V11\n(1000 clients)', 'V12\n(100, 50%)']
    leakscore_corr = [0.85, 0.82, 0.78, 0.80]
    accuracy       = [90.3, 89.8, 88.5, 89.2]
    asr            = [0.06, 0.07, 0.10, 0.08]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    x = np.arange(len(configs))
    colors_corr = [C_BLUE, C_GREEN, C_ORANGE, C_TEAL]

    ax1.bar(x, leakscore_corr, color=colors_corr, alpha=0.8, edgecolor='white', width=0.5)
    for i, v in enumerate(leakscore_corr):
        ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='#333')
    ax1.axhline(y=0.75, color=C_GREEN, linestyle=':', alpha=0.5)
    ax1.annotate('Acceptable (> 0.75)', xy=(3.3, 0.755), fontsize=7, color=C_GREEN)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=7.5)
    style_ax(ax1, '', 'LeakScore-ASR Correlation', 'LeakScore Accuracy at Scale')
    ax1.set_ylim(0.6, 0.95)

    width = 0.3
    ax2.bar(x - width/2, accuracy, width, color=C_BLUE, alpha=0.8, label='Accuracy (%)', edgecolor='white')
    ax2b = ax2.twinx()
    ax2b.bar(x + width/2, asr, width, color=C_RED, alpha=0.8, label='ASR', edgecolor='white')
    ax2b.set_ylabel('Attack Success Rate', fontsize=9, color=C_RED)
    ax2b.set_ylim(0, 0.2)
    ax2b.spines['top'].set_visible(False)
    ax2b.tick_params(colors='#555', labelsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=7.5)
    style_ax(ax2, '', 'Model Accuracy (%)', 'Accuracy & Attack Success at Scale')
    ax2.set_ylim(85, 92)
    ax2.legend(fontsize=7, loc='lower left')
    ax2b.legend(fontsize=7, loc='upper right')

    fig.tight_layout(pad=2)
    return fig


# ══════════════════════════════════════════════════════════════════════
# PDF BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('DocTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=HexColor(C_DARK), alignment=TA_CENTER))
    styles.add(ParagraphStyle('DocSubtitle', parent=styles['Normal'],
        fontSize=11, spaceAfter=20, textColor=HexColor('#555555'), alignment=TA_CENTER))
    styles.add(ParagraphStyle('SectionHead', parent=styles['Heading1'],
        fontSize=16, spaceBefore=20, spaceAfter=10, textColor=HexColor(C_DARK), borderWidth=0))
    styles.add(ParagraphStyle('SubHead', parent=styles['Heading2'],
        fontSize=13, spaceBefore=14, spaceAfter=6, textColor=HexColor('#2d3436')))
    styles.add(ParagraphStyle('SubSubHead', parent=styles['Heading3'],
        fontSize=11, spaceBefore=10, spaceAfter=4, textColor=HexColor('#444444')))
    styles.add(ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=9.5, leading=13, spaceAfter=6, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('Equation', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=8, spaceBefore=4,
        fontName='Courier', leftIndent=30, textColor=HexColor('#2c3e50')))
    styles.add(ParagraphStyle('BulletItem', parent=styles['Normal'],
        fontSize=9.5, leading=13, spaceAfter=3, leftIndent=20, bulletIndent=10))
    styles.add(ParagraphStyle('SmallNote', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=HexColor('#777777'), spaceAfter=4))
    return styles


def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor(C_DARK)),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#f5f6fa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(style)
    return t


def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )
    s = build_styles()
    story = []

    # ── TITLE PAGE ─────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("AdaGuard", s['DocTitle']))
    story.append(Paragraph(
        "Projected Experiment Results", s['DocSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor(C_DARK),
                             spaceBefore=10, spaceAfter=10))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "Expected Outcomes for 72 Scenarios<br/>"
        "with Projected Graphs", s['DocSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Firas Aguir", ParagraphStyle('Author', parent=s['Body'],
                            fontSize=12, alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph("Oakland University", ParagraphStyle('Affil', parent=s['Body'],
                            fontSize=10, alignment=TA_CENTER, textColor=HexColor('#666666'))))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("April 2026", ParagraphStyle('Date', parent=s['Body'],
                            fontSize=10, alignment=TA_CENTER, textColor=HexColor('#999999'))))
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph(
        "<i>Note: All graphs show projected/expected results based on prior literature "
        "and theoretical analysis. Actual results will be produced by running the experiments "
        "on the Matilda HPC cluster.</i>",
        ParagraphStyle('Disclaimer', parent=s['Body'], fontSize=9, alignment=TA_CENTER,
                       textColor=HexColor('#999999'))))
    story.append(PageBreak())

    # ── S1: ENTROPY BINS ───────────────────────────────────────────
    story.append(Paragraph("S1: Entropy Bins (5 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variable:</b> entropy_bins = {10, 25, <u>50</u>, 100, 200}", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> Entropy scores should stabilize in the 25-100 bin range. "
        "At 10 bins, the histogram is too coarse to capture gradient structure, yielding "
        "noisy and unreliable scores. Above 100 bins, the histogram overfits to individual "
        "gradient values and variance increases. We expect a plateau between 25-100 bins "
        "confirming the default of 50 is robust.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> Shannon, Renyi, and MinEntropy scores converge by ~25 bins "
        "and remain stable through 200. Score variance is highest at 10 bins. The default "
        "of 50 sits comfortably in the stable region.", s['Body']))
    story.append(fig_to_image(chart_s1_entropy_bins()))
    story.append(PageBreak())

    # ── S2: GRADIENT ACCUMULATION K ────────────────────────────────
    story.append(Paragraph("S2: Gradient Accumulation K (4 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variable:</b> grad_accum_K = {1, 2, <u>4</u>, 8}", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> Attack success rate drops sharply as K increases because the "
        "attacker must recover K*B images from a single averaged gradient. Yin et al. (2021) "
        "showed GradInversion fails around B_eff=8-16 on CIFAR-10. We expect K=4 (B_eff=16) "
        "to be the sweet spot: strong privacy with acceptable 4x compute cost. K=8 provides "
        "marginal additional privacy at double the cost.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> ASR drops from ~82% at K=1 to ~8% at K=4 to ~1% at K=8. "
        "The diminishing returns after K=4 justify the default. SSIM follows the same trend.",
        s['Body']))
    story.append(fig_to_image(chart_s2_grad_accum()))
    story.append(PageBreak())

    # ── S3-S4: THRESHOLDS ──────────────────────────────────────────
    story.append(Paragraph("S3-S4: Encryption Thresholds T1 and T2 (10 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variables:</b> S3: T1 = {0.1-0.5} with T2=0.7 | S4: T2 = {0.5-0.9} with T1=0.3", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> There is a clear tradeoff: lower thresholds encrypt more clients "
        "(better privacy, lower accuracy), higher thresholds encrypt fewer (worse privacy, "
        "higher accuracy). The default T1=0.3, T2=0.7 should sit near the optimal balance point "
        "where the ASR curve starts to flatten (diminishing privacy returns from lower thresholds) "
        "while accuracy is still high.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> S3 shows a smooth tradeoff curve between ASR and accuracy as T1 "
        "moves from 0.1 to 0.5. S4 shows a similar pattern for T2. The default values sit near "
        "the \"knee\" of each curve where marginal privacy gains per unit accuracy loss are highest.",
        s['Body']))
    story.append(fig_to_image(chart_s3s4_thresholds()))
    story.append(PageBreak())

    # ── S5-S7: WEIGHTS ─────────────────────────────────────────────
    story.append(Paragraph("S5-S7: LeakScore Component Weights (24 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variables:</b> alpha, beta, gamma each swept {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}",
        s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> Detection accuracy peaks at moderate weight values (0.5-1.5) for "
        "each component. Setting any weight to 0 should noticeably reduce detection quality, "
        "confirming each metric family adds value. The empirical weight (gamma) should show "
        "the smallest marginal improvement relative to its high compute cost -- justifying "
        "the default of gamma=0.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> Alpha and beta show similar inverted-U curves peaking around "
        "1.0-1.5. Gamma's curve is flatter and barely above the alpha=beta=1, gamma=0 baseline, "
        "while compute cost increases 2-3x. This confirms the default weighting is near-optimal "
        "and empirical probes are not worth the cost for most deployments.", s['Body']))
    story.append(fig_to_image(chart_s5s6s7_weights()))
    story.append(PageBreak())

    # ── S8: BATCH SIZE ─────────────────────────────────────────────
    story.append(Paragraph("S8: Client Batch Size (5 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variable:</b> client_batch_size = {1, <u>4</u>, 8, 16, 32}", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> Batch size is the single most important factor in gradient "
        "leakability. B=1 is trivially invertible (iDLG extracts labels analytically, "
        "single-image reconstruction is well-solved). As B increases, reconstruction quality "
        "degrades exponentially. We expect LeakScore to correctly track this trend, with "
        "high correlation between predicted risk and actual attack success.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> ASR drops from ~95% (B=1) to ~1% (B=32). LeakScore tracks "
        "this closely -- the scatter plot should show strong positive correlation between "
        "LeakScore and actual ASR, validating that our metric is well-calibrated.", s['Body']))
    story.append(fig_to_image(chart_s8_batch_size()))
    story.append(PageBreak())

    # ── S9: GLMIP SAMPLES ─────────────────────────────────────────
    story.append(Paragraph("S9: GLMIP Samples Per Class (4 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variable:</b> mi_samples_per_class = {5, 10, <u>20</u>, 40}", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> GLMIP estimates per-class gradient energy from representative samples. "
        "With few samples (5), the estimate is noisy -- high variance across clients. By 20 samples, "
        "the estimate should stabilize. Doubling to 40 should show minimal improvement, confirming "
        "20 is the cost-accuracy sweet spot.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> GLMIP mean score increases from ~0.62 (5 samples) to ~0.71 "
        "(40 samples), but the error bars shrink dramatically between 5 and 20. The marginal "
        "gain from 20 to 40 is small (0.70 to 0.71) while compute doubles.", s['Body']))
    story.append(fig_to_image(chart_s9_samples()))
    story.append(PageBreak())

    # ── S10: MASKCRYPT RHO ─────────────────────────────────────────
    story.append(Paragraph("S10: MaskCrypt Encrypt Ratio rho (5 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variable:</b> encryption_top_percent = {0.01, 0.05, <u>0.10</u>, 0.20, 0.25}", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> Per Hu &amp; Li (2025), gradient-guided encryption is remarkably "
        "efficient: encrypting just 1% of the most vulnerable parameters blocks most gradient "
        "inversion attacks. This is because v[i] identifies parameters that carry the most "
        "information about the training data. Diminishing returns set in quickly after rho=0.05.",
        s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> ASR drops sharply from rho=0.01 to 0.05, then flattens. "
        "The jump from 1% to 5% encrypted provides most of the privacy benefit. Model accuracy "
        "degrades slightly as more parameters are encrypted. This reproduces Figs. 3, 5, 6 "
        "from the MaskCrypt paper.", s['Body']))
    story.append(fig_to_image(chart_s10_maskcrypt_rho()))
    story.append(PageBreak())

    # ── S11: FOCUS LAYERS ──────────────────────────────────────────
    story.append(Paragraph("S11: Focus Layers for GLMIP (3 runs)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Variable:</b> focus_layers = {None (all), <u>final FC</u>, penultimate FC}", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> Per the iDLG theorem (Zhao et al. 2020), the gradient of the "
        "final fully-connected layer contains the strongest label signal because each output "
        "neuron corresponds directly to a class. The penultimate layer has weaker but still "
        "present label information. Using all layers dilutes the signal with low-information "
        "convolutional layers.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> GLMIP score is highest for final FC (~0.70), lower for "
        "penultimate FC (~0.52), and intermediate for all layers (~0.58). ConfidenceGap and "
        "CosineSimilarity are unaffected by focus_layers (they use the full gradient). "
        "Overall detection accuracy confirms the iDLG-motivated default.", s['Body']))
    story.append(fig_to_image(chart_s11_focus_layers()))
    story.append(PageBreak())

    # ── VIABILITY: PRIVACY ─────────────────────────────────────────
    story.append(Paragraph("Viability Study: Privacy Comparison (V1-V9)", s['SectionHead']))
    story.append(Paragraph(
        "The central question: how does AdaGuard compare to the competition? This chart "
        "projects the attack success rate across all non-scalability viability scenarios.",
        s['Body']))
    story.append(Paragraph(
        "<b>Key expected findings:</b>", s['Body']))
    findings = [
        "V1 (no defense) confirms the threat is real: ~82% attack success rate",
        "V2 (full HE) achieves 0% ASR but at 100% encryption cost",
        "V3 (DP) reduces ASR to ~15% but degrades model accuracy significantly",
        "V4 (MaskCrypt guided) achieves ~3% ASR -- our primary competitor to beat",
        "V5 (MaskCrypt random) at ~18% ASR shows gradient-guided selection is critical",
        "V6 (AdaGuard default) achieves ~6% ASR -- competitive with MaskCrypt at adaptive cost",
        "V7 (AdaGuard + accumulation) achieves ~2% ASR -- <b>best practical defense</b>",
        "V8 (aggressive) matches full HE privacy at ~55% of the cost",
        "V9 (conservative) trades privacy for maximum utility (~20% ASR but 91% accuracy)",
    ]
    for item in findings:
        story.append(Paragraph(item, s['BulletItem'], bulletText='\xe2\x80\xa2'))
    story.append(Spacer(1, 6))
    story.append(fig_to_image(chart_viability_privacy(), height=3.0*inch))
    story.append(PageBreak())

    # ── VIABILITY: UTILITY VS COST ─────────────────────────────────
    story.append(Paragraph("Viability Study: Utility vs Cost Tradeoff", s['SectionHead']))
    story.append(Paragraph(
        "The bubble chart below plots each defense strategy in the utility-cost space. "
        "X-axis: percentage of parameters encrypted (proxy for communication cost). "
        "Y-axis: model accuracy after federated training. Bubble size represents privacy "
        "strength (larger = lower ASR = better defense).", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> AdaGuard variants (V6, V7) should appear in the upper-left "
        "quadrant (high accuracy, low cost) with large bubbles (strong privacy). MaskCrypt (V4) "
        "will be competitive but at a fixed 10% cost regardless of actual risk. The key insight: "
        "AdaGuard's adaptive approach means it encrypts 0% for safe clients and up to 55% for "
        "the riskiest, averaging ~18% overall -- lower than MaskCrypt's fixed 10% because most "
        "clients in a typical round are not high-risk.", s['Body']))
    story.append(fig_to_image(chart_viability_utility_cost(), height=3.5*inch))
    story.append(PageBreak())

    # ── VIABILITY: SCALABILITY ─────────────────────────────────────
    story.append(Paragraph("Viability Study: Scalability (V10-V12)", s['SectionHead']))
    story.append(Paragraph(
        "<b>Key question:</b> Does AdaGuard work beyond 10 clients?", s['Body']))
    story.append(Paragraph(
        "<b>Hypothesis:</b> LeakScore should remain well-calibrated as client count increases, "
        "though we expect a slight degradation because: (1) with more clients, each client has "
        "less data, making individual gradients noisier; (2) gradient accumulation averages "
        "over more heterogeneous data; (3) the metric computation cost scales linearly with "
        "client count. However, the core signal -- whether a gradient leaks label information "
        "-- should persist regardless of federation size.", s['Body']))
    story.append(Paragraph(
        "<b>Expected outcome:</b> LeakScore-ASR correlation drops slightly from 0.85 (10 clients) "
        "to 0.78 (1000 clients) but stays above the 0.75 threshold we consider acceptable. "
        "Model accuracy decreases slightly at 1000 clients due to data splitting. "
        "The high participation scenario (V12, 50%) may show slightly better accuracy due to "
        "more gradient information per round.", s['Body']))
    story.append(fig_to_image(chart_viability_scalability()))

    # ── SUMMARY ────────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(Paragraph("Summary of Projected Results", s['SubHead']))
    summary = make_table(
        ['Scenario Group', 'Key Projection', 'Confidence'],
        [
            ['S1 (entropy bins)', 'Stable in 25-100 range; 50 is robust default', 'High'],
            ['S2 (grad accum K)', 'K=4 is the sweet spot; K=8 marginal gain', 'High (literature-backed)'],
            ['S3-S4 (thresholds)', 'T1=0.3, T2=0.7 near optimal knee of tradeoff', 'Medium'],
            ['S5-S7 (weights)', 'Label+entropy sufficient; empirical not worth cost', 'Medium'],
            ['S8 (batch size)', 'B=1 trivially broken; B>=16 safe', 'High (literature-backed)'],
            ['S9 (GLMIP samples)', '20 samples/class is cost-optimal', 'Medium'],
            ['S10 (MaskCrypt rho)', '1% already effective; diminishing returns after 5%', 'High (paper result)'],
            ['S11 (focus layers)', 'Final FC best per iDLG theorem', 'High (theory-backed)'],
            ['V1-V9 (viability)', 'AdaGuard+accum best practical defense', 'Medium-High'],
            ['V10-V12 (scale)', 'LeakScore stable to 1000 clients', 'Medium'],
        ],
        col_widths=[1.5*inch, 3.5*inch, 2.0*inch]
    )
    story.append(summary)

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor(C_DARK),
                             spaceBefore=10, spaceAfter=10))
    story.append(Paragraph(
        "<i>All projections are based on prior literature (Yin et al. 2021, Zhao et al. 2020, "
        "Hu &amp; Li 2025, Meng et al. 2025, Yu et al. 2025) and theoretical analysis. "
        "Actual results may differ. Experiments to be run on Matilda HPC cluster.</i>",
        ParagraphStyle('Final', parent=s['Body'], fontSize=8, alignment=TA_CENTER,
                       textColor=HexColor('#999999'))))

    # BUILD
    doc.build(story)
    print(f"PDF generated: {output_path}")


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'AdaGuard_Projected_Results.pdf'
    build_pdf(out)
