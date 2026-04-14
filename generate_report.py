#!/usr/bin/env python3
"""Generate AdaGuard Technical Report PDF."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=HexColor('#1a1a2e'),
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontSize=11, spaceAfter=20, textColor=HexColor('#555555'),
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'SectionHead', parent=styles['Heading1'],
        fontSize=16, spaceBefore=20, spaceAfter=10,
        textColor=HexColor('#1a1a2e'), borderWidth=0,
    ))
    styles.add(ParagraphStyle(
        'SubHead', parent=styles['Heading2'],
        fontSize=13, spaceBefore=14, spaceAfter=6,
        textColor=HexColor('#2d3436'),
    ))
    styles.add(ParagraphStyle(
        'SubSubHead', parent=styles['Heading3'],
        fontSize=11, spaceBefore=10, spaceAfter=4,
        textColor=HexColor('#444444'),
    ))
    styles.add(ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=9.5, leading=13, spaceAfter=6,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        'Equation', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=8, spaceBefore=4,
        fontName='Courier', leftIndent=30, textColor=HexColor('#2c3e50'),
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=9.5, leading=13, spaceAfter=3,
        leftIndent=20, bulletIndent=10,
    ))
    styles.add(ParagraphStyle(
        'SmallNote', parent=styles['Normal'],
        fontSize=8, leading=10, textColor=HexColor('#777777'),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'CodeBlock', parent=styles['Normal'],
        fontSize=8.5, leading=11, fontName='Courier',
        leftIndent=20, spaceAfter=6, textColor=HexColor('#2c3e50'),
    ))
    return styles


def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
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


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor('#dddddd'),
                       spaceBefore=6, spaceAfter=6)


def build_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
    )
    s = build_styles()
    story = []
    W = doc.width

    # ===================================================================
    # TITLE PAGE
    # ===================================================================
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("AdaGuard", s['DocTitle']))
    story.append(Paragraph(
        "Adaptive Privacy-Preserving Federated Learning<br/>"
        "with Gradient Leakage Detection", s['DocSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor('#1a1a2e'),
                             spaceBefore=10, spaceAfter=10))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Technical Report: Experiment Design &amp; Architecture", s['DocSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Firas Aguir", ParagraphStyle('Author', parent=s['Body'],
                            fontSize=12, alignment=TA_CENTER, spaceAfter=4)))
    story.append(Paragraph("Oakland University", ParagraphStyle('Affil', parent=s['Body'],
                            fontSize=10, alignment=TA_CENTER, textColor=HexColor('#666666'))))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("April 2026", ParagraphStyle('Date', parent=s['Body'],
                            fontSize=10, alignment=TA_CENTER, textColor=HexColor('#999999'))))
    story.append(PageBreak())

    # ===================================================================
    # TABLE OF CONTENTS
    # ===================================================================
    story.append(Paragraph("Table of Contents", s['SectionHead']))
    toc_items = [
        "<b>Experiment Design</b>",
        "1. Case 1: Sensitivity Study (60 Scenarios)",
        "2. Case 2: Viability Study (12 Scenarios)",
        "3. Artifact Reuse Strategy",
        "",
        "<b>System Architecture</b>",
        "4. System Overview",
        "5. Two-Phase Architecture",
        "6. LeakScore Pipeline",
        "7. Encryption Strategies",
        "8. Gradient Accumulation Defense",
        "9. Attack Implementations",
        "",
        "<b>Infrastructure</b>",
        "10. GPU Detection &amp; Resource Allocation",
        "11. HPC Deployment",
        "12. Verification Checklist",
    ]
    for item in toc_items:
        if item == "":
            story.append(Spacer(1, 4))
        elif item.startswith("<b>"):
            story.append(Paragraph(item, ParagraphStyle('TOCHead', parent=s['Body'],
                                   fontSize=10, spaceAfter=3, spaceBefore=6, leftIndent=10,
                                   textColor=HexColor('#1a1a2e'))))
        else:
            story.append(Paragraph(item, ParagraphStyle('TOC', parent=s['Body'],
                                   fontSize=10, spaceAfter=5, leftIndent=20)))
    story.append(PageBreak())

    # ===================================================================
    # 1. CASE 1: SENSITIVITY STUDY
    # ===================================================================
    story.append(Paragraph("1. Case 1: Sensitivity Study (60 Scenarios)", s['SectionHead']))
    story.append(Paragraph(
        "The sensitivity study is the core of our experimental evaluation. It isolates each "
        "configurable variable in the AdaGuard pipeline and sweeps it across a range of values "
        "while holding all other parameters at their defaults. This systematic approach reveals "
        "how each knob affects LeakScore accuracy, defense quality, and model utility -- answering "
        "the critical question of which parameters matter most and what their optimal values are.",
        s['Body']))
    story.append(Paragraph(
        "Each sensitivity group produces a clear curve: the swept variable on the x-axis, "
        "and privacy/utility/cost metrics on the y-axis. This data directly feeds into the "
        "paper's ablation study figures.", s['Body']))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Overview: 11 Groups, 60 Total Runs", s['SubHead']))

    sens_table = make_table(
        ['Group', 'Variable', 'Sweep Values', 'Runs', 'Goal'],
        [
            ['S1', 'entropy_bins', '10, 25, 50*, 100, 200', '5',
             'Bin resolution effect on entropy scores'],
            ['S2', 'grad_accum_K', '1, 2, 4*, 8', '4',
             'Privacy gain vs compute cost per K'],
            ['S3', 'T1 (T2=0.7)', '0.1, 0.2, 0.3*, 0.4, 0.5', '5',
             'Where partial encryption should kick in'],
            ['S4', 'T2 (T1=0.3)', '0.5, 0.6, 0.7*, 0.8, 0.9', '5',
             'Where strong encryption should kick in'],
            ['S5', 'label_weight', '0, 0.5, 1*, 1.5, 2, 2.5, 3, 3.5', '8',
             'GLMIP + ConfGap + CosSim contribution'],
            ['S6', 'entropy_weight', '0, 0.5, 1*, 1.5, 2, 2.5, 3, 3.5', '8',
             'Shannon + Renyi + MinEnt contribution'],
            ['S7', 'empirical_weight', '0*, 0.5, 1, 1.5, 2, 2.5, 3, 3.5', '8',
             'Lightweight GI probe cost vs benefit'],
            ['S8', 'client_batch_size', '1, 4*, 8, 16, 32', '5',
             'Batch size effect on leakability'],
            ['S9', 'samples_per_class', '5, 10, 20*, 40', '4',
             'GLMIP sampling density'],
            ['S10', 'MaskCrypt rho', '0.01, 0.05, 0.10*, 0.20, 0.25', '5',
             'Encrypt ratio vs defense quality'],
            ['S11', 'focus_layers', 'all, final_fc*, penultimate_fc', '3',
             'Which layers carry most label info'],
        ],
        col_widths=[0.5*inch, 1.2*inch, 1.8*inch, 0.4*inch, 3.1*inch]
    )
    story.append(sens_table)
    story.append(Paragraph("* = default value. All other parameters held at defaults.", s['SmallNote']))

    # Detailed breakdowns per group
    story.append(Spacer(1, 10))
    story.append(Paragraph("Detailed Group Descriptions", s['SubHead']))

    story.append(Paragraph("S1: Entropy Bins (5 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "Controls the histogram resolution for all three entropy metrics (Shannon, Renyi, MinEntropy). "
        "Too few bins may miss gradient structure; too many may overfit to noise. "
        "Sweep: 10, 25, <b>50</b> (default), 100, 200.", s['Body']))

    story.append(Paragraph("S2: Gradient Accumulation K (4 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "The number of independent forward-backward passes averaged before sending the gradient. "
        "K=1 means no accumulation (baseline). Higher K increases B_eff = K*B, making gradient "
        "inversion harder but costing proportionally more compute. "
        "Sweep: 1, 2, <b>4</b> (default), 8. <b>Requires retraining</b> -- each K produces "
        "fundamentally different gradients.", s['Body']))

    story.append(Paragraph("S3: Threshold T1 (5 runs) and S4: Threshold T2 (5 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "T1 is where partial encryption kicks in; T2 is where strong encryption engages. "
        "S3 sweeps T1 with T2 fixed at 0.7; S4 sweeps T2 with T1 fixed at 0.3. "
        "These directly control how aggressively AdaGuard encrypts -- the privacy-utility tradeoff knob.",
        s['Body']))

    story.append(Paragraph("S5-S7: LeakScore Component Weights (8 runs each)", s['SubSubHead']))
    story.append(Paragraph(
        "The LeakScore formula: Combined = (alpha * LabelAvg + beta * EntropyAvg + gamma * EmpiricalAvg) "
        "/ (alpha + beta + gamma). Each group sweeps one weight from 0.0 to 3.5 while holding the others "
        "at default. This reveals which metric family contributes most to accurate leak detection. "
        "S7 (empirical_weight) is particularly important: it tells us whether the computational cost "
        "of running lightweight GI probes is justified by better detection.", s['Body']))

    story.append(Paragraph("S8: Client Batch Size (5 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "Batch size fundamentally changes gradient leakability -- single-sample batches (B=1) are "
        "trivially invertible via iDLG, while larger batches mix information from multiple samples. "
        "Sweep: 1, <b>4</b> (default), 8, 16, 32. <b>Requires retraining</b>.", s['Body']))

    story.append(Paragraph("S9: GLMIP Samples Per Class (4 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "Controls how many representative samples per class are used to estimate the GLMIP "
        "gradient energy distribution. More samples = more accurate estimate but higher cost. "
        "Sweep: 5, 10, <b>20</b> (default), 40.", s['Body']))

    story.append(Paragraph("S10: MaskCrypt Encrypt Ratio rho (5 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "Maps directly to MaskCrypt paper experiments (Hu &amp; Li, IEEE TDSC 2025, Figs. 3, 5, 6). "
        "The encrypt ratio determines what fraction of parameters are encrypted using gradient-guided "
        "vulnerability scoring. Their key finding: rho=0.01 (1%) already blocks gradient inversion. "
        "Sweep: 0.01, 0.05, <b>0.10</b> (default), 0.20, 0.25.", s['Body']))

    story.append(Paragraph("S11: Focus Layers for GLMIP (3 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "Per the iDLG theorem, the final fully-connected layer carries the strongest label signal. "
        "But is restricting GLMIP to only those layers optimal? "
        "Sweep: all layers, <b>final FC</b> (default: fc2.weight, fc2.bias), penultimate FC (fc1).",
        s['Body']))
    story.append(PageBreak())

    # ===================================================================
    # 2. CASE 2: VIABILITY STUDY
    # ===================================================================
    story.append(Paragraph("2. Case 2: Viability Study (12 Scenarios)", s['SectionHead']))
    story.append(Paragraph(
        "The viability study provides direct head-to-head comparisons proving AdaGuard works. "
        "While the sensitivity study answers <i>\"how does each knob affect performance?\"</i>, "
        "the viability study answers the critical question: <i>\"does AdaGuard outperform the "
        "competition?\"</i> Each scenario runs all three full paper-matched attacks (GradInversion "
        "20K iterations, GI-NAS 8 restarts, GGCDM 1K diffusion steps) and measures privacy, "
        "utility, and cost.", s['Body']))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Metrics Per Scenario", s['SubHead']))
    viability_metrics = make_table(
        ['Category', 'Metrics', 'What They Measure'],
        [
            ['Privacy', 'LPIPS, SSIM, PSNR, MSE, FID, ASR', 'How well attacks reconstruct private data'],
            ['Utility', 'Model accuracy (%)', 'Does encryption degrade the trained model?'],
            ['Cost', '% params encrypted, comm. overhead, wall time', 'What is the price of protection?'],
            ['Detection', 'LeakScore vs actual attack success', 'Does LeakScore correctly identify risk?'],
        ],
        col_widths=[1.0*inch, 2.5*inch, 3.5*inch]
    )
    story.append(viability_metrics)

    story.append(Spacer(1, 10))
    story.append(Paragraph("V1-V3: Baselines (No Intelligence)", s['SubHead']))
    story.append(Paragraph(
        "These establish the performance boundaries. V1 (no defense) shows worst-case privacy -- "
        "how badly gradients leak with zero protection. V2 (full encryption) shows the cost ceiling. "
        "V3 (differential privacy) represents the industry-standard alternative.", s['Body']))
    v_base = make_table(
        ['ID', 'Name', 'Strategy', 'Purpose'],
        [
            ['V1', 'no_defense', 'encryption=none', 'Worst-case privacy -- how bad is the leak?'],
            ['V2', 'full_he', 'encryption=full', 'Best privacy / worst cost ceiling'],
            ['V3', 'dp_baseline', 'Gaussian noise (eps=50, delta=1e-5)',
             'Industry-standard DP alternative (matches MaskCrypt paper baseline)'],
        ],
        col_widths=[0.4*inch, 1.1*inch, 2.2*inch, 3.3*inch]
    )
    story.append(v_base)

    story.append(Spacer(1, 8))
    story.append(Paragraph("V4-V5: MaskCrypt -- Our #1 Direct Competitor", s['SubHead']))
    story.append(Paragraph(
        "MaskCrypt (Hu &amp; Li, IEEE TDSC 2025) is the closest competing approach. It uses a "
        "gradient-guided vulnerability score v[i] = gradient[i] * (old_exposed[i] - new_trained[i]) "
        "to select which parameters to encrypt. We test both their gradient-guided mode (Algorithm 2) "
        "and a random mask baseline, at the same encrypt ratio rho=0.10, to demonstrate "
        "the value of intelligent parameter selection.", s['Body']))
    v_mc = make_table(
        ['ID', 'Name', 'Strategy', 'Purpose'],
        [
            ['V4', 'maskcrypt_guided', 'Gradient-guided mask, rho=0.10',
             'Paper default -- #1 direct competitor'],
            ['V5', 'maskcrypt_random', 'Random mask, rho=0.10',
             'Shows value of smart mask selection vs naive'],
        ],
        col_widths=[0.4*inch, 1.4*inch, 2.2*inch, 3.0*inch]
    )
    story.append(v_mc)

    story.append(Spacer(1, 8))
    story.append(Paragraph("V6-V9: AdaGuard Variants (Our Method)", s['SubHead']))
    story.append(Paragraph(
        "Four configurations spanning our method's operating range. V6 is the balanced default, "
        "V7 adds gradient accumulation for the full defense pipeline, V8 tests aggressive encryption "
        "(encrypt early, encrypt often), and V9 tests the conservative extreme.", s['Body']))
    v_ag = make_table(
        ['ID', 'Name', 'Config', 'Purpose'],
        [
            ['V6', 'adaguard_default', 'Fisher, T1=0.3, T2=0.7, no accum',
             'Balanced baseline -- our primary result'],
            ['V7', 'adaguard_with_accum', 'Fisher, T1=0.3, T2=0.7, K=4',
             'Full defense pipeline with gradient accumulation'],
            ['V8', 'adaguard_aggressive', 'Fisher, T1=0.1, T2=0.4',
             'Upper bound -- encrypt early, encrypt often'],
            ['V9', 'adaguard_conservative', 'Fisher, T1=0.5, T2=0.9',
             'Lower bound -- minimal encryption, max utility'],
        ],
        col_widths=[0.4*inch, 1.6*inch, 2.2*inch, 2.8*inch]
    )
    story.append(v_ag)

    story.append(Spacer(1, 8))
    story.append(Paragraph("V10-V12: Scalability Tests", s['SubHead']))
    story.append(Paragraph(
        "Does AdaGuard scale? These scenarios test whether LeakScore remains accurate and the "
        "adaptive encryption controller remains effective as the number of clients and participation "
        "rates change. <b>These require separate Phase 1 training</b> due to different FL topologies.",
        s['Body']))
    v_scale = make_table(
        ['ID', 'Name', 'Config', 'Purpose'],
        [
            ['V10', '100_clients', '100 clients, 10/round (10%)', 'Does LeakScore hold at 100 clients?'],
            ['V11', '1000_clients', '1000 clients, 100/round (10%)', 'Does LeakScore hold at 1000 clients?'],
            ['V12', 'high_participation', '100 clients, 50/round (50%)', 'Does participation rate affect defense?'],
        ],
        col_widths=[0.4*inch, 1.4*inch, 2.4*inch, 2.8*inch]
    )
    story.append(v_scale)

    story.append(Spacer(1, 10))
    story.append(Paragraph("Expected Outcomes", s['SubHead']))
    expected = [
        "V1 (no defense) has highest attack success -- establishes the threat is real",
        "V2 (full HE) has lowest attack success but highest cost -- shows the ceiling",
        "V3 (DP) degrades model accuracy significantly for comparable privacy to MaskCrypt",
        "V4 (MaskCrypt guided) outperforms V5 (random) -- validates gradient-guided selection",
        "V6/V7 (AdaGuard) achieve comparable privacy to V4 at lower encryption cost",
        "V7 (with accumulation) closes any remaining privacy gap vs V2 for high-risk clients",
        "V8 (aggressive) matches V2 privacy at ~55% of the encryption cost",
        "V10-V12 show LeakScore accuracy is stable across client counts and participation rates",
    ]
    for item in expected:
        story.append(Paragraph(item, s['BulletItem'], bulletText='\xe2\x80\xa2'))
    story.append(PageBreak())

    # ===================================================================
    # 3. ARTIFACT REUSE STRATEGY
    # ===================================================================
    story.append(Paragraph("3. Artifact Reuse Strategy", s['SectionHead']))
    story.append(Paragraph(
        "A key efficiency advantage of AdaGuard's two-phase architecture: most scenarios do not "
        "require retraining. The runner auto-detects which metrics need recomputation based on "
        "whether the scenario changes training-time vs post-hoc parameters. "
        "This makes running 72 scenarios practical on limited HPC resources.", s['Body']))

    story.append(Paragraph("Scenarios That Reuse Phase 1 Artifacts (57 of 72)", s['SubHead']))
    story.append(Paragraph(
        "These scenarios only change how saved gradients are interpreted or protected. The raw "
        "gradient data, model states, and training metrics are replayed as-is or with selective "
        "metric recomputation:", s['Body']))

    reuse_table = make_table(
        ['Groups', 'What Changes', 'Recomputation Needed'],
        [
            ['S3, S4 (thresholds)', 'Encryption decision boundaries', 'None -- same metrics, different policy'],
            ['S5, S6, S7 (weights)', 'LeakScore component weights', 'None -- same metrics, different weighting'],
            ['S1 (entropy_bins)', 'Histogram bin count', 'Entropy metrics only (from saved flat_gradient)'],
            ['S9 (samples_per_class)', 'GLMIP sampling density', 'Label metrics only (from saved gradients)'],
            ['S11 (focus_layers)', 'Which layers GLMIP uses', 'Label metrics only (from saved gradients)'],
            ['S10 (MaskCrypt rho)', 'Encrypt ratio', 'None -- re-mask from saved gradients'],
            ['V1-V9 (viability)', 'Encryption strategy', 'None -- apply different encryption to saved grads'],
        ],
        col_widths=[1.5*inch, 2.2*inch, 3.3*inch]
    )
    story.append(reuse_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph("Scenarios Requiring Separate Training (12 of 72)", s['SubHead']))
    retrain_table = make_table(
        ['Groups', 'Variable', 'Why Retraining Needed'],
        [
            ['S2 (4 runs)', 'grad_accum_K', 'K changes how gradients are computed at training time'],
            ['S8 (5 runs)', 'client_batch_size', 'Different batch = entirely different raw gradients'],
            ['V10, V11, V12 (3 runs)', 'num_clients / participation', 'Different FL topology and client selection'],
        ],
        col_widths=[1.3*inch, 1.5*inch, 4.2*inch]
    )
    story.append(retrain_table)
    story.append(Paragraph(
        "Total compute savings: 57 scenarios replay from one training run, avoiding ~57x redundant "
        "Phase 1 compute. Only 12 scenarios need additional training.", s['SmallNote']))
    story.append(PageBreak())

    # ===================================================================
    # 4. SYSTEM OVERVIEW
    # ===================================================================
    story.append(Paragraph("4. System Overview", s['SectionHead']))
    story.append(Paragraph(
        "AdaGuard is a privacy-preserving federated learning framework that detects gradient "
        "leakage risk in real time and adaptively applies encryption to protect client data. "
        "Unlike static approaches (full homomorphic encryption or differential privacy), AdaGuard "
        "measures how much private information each client's gradient update reveals and encrypts "
        "only what is necessary, balancing privacy protection against model utility and "
        "communication cost.", s['Body']))
    story.append(Paragraph(
        "The system operates in two phases: Phase 1 trains the federated model and saves comprehensive "
        "per-client artifacts, while Phase 2 replays those artifacts under 72 different encryption "
        "scenarios to evaluate sensitivity and viability without retraining.", s['Body']))
    story.append(hr())

    story.append(Paragraph("Default Configuration", s['SubHead']))
    cfg_table = make_table(
        ['Parameter', 'Default', 'Description'],
        [
            ['model', 'smallcnn', 'Architecture (smallcnn, resnet18/34/50)'],
            ['num_clients', '10', 'Total FL clients'],
            ['clients_per_round', '3', 'Selected per round'],
            ['num_rounds', '5', 'Training rounds'],
            ['client_batch_size', '4', 'Local batch size B'],
            ['client_lr', '0.01', 'Local SGD learning rate'],
            ['T1', '0.3', 'Partial encryption threshold'],
            ['T2', '0.7', 'Strong encryption threshold'],
            ['label_weight', '1.0', 'GLMIP + ConfGap + CosSim weight'],
            ['entropy_weight', '1.0', 'Shannon + Renyi + MinEnt weight'],
            ['empirical_weight', '0.0', 'Lightweight GI probe weight'],
            ['entropy_bins', '50', 'Histogram bins for entropy'],
            ['grad_accum_K', '4', 'Accumulation passes (B_eff = K*B)'],
            ['encryption_top_percent', '0.1', 'Base encrypt ratio (10%)'],
            ['seed', '42', 'Random seed'],
        ],
        col_widths=[1.6*inch, 0.9*inch, 4.5*inch]
    )
    story.append(cfg_table)
    story.append(PageBreak())

    # ===================================================================
    # 5. TWO-PHASE ARCHITECTURE
    # ===================================================================
    story.append(Paragraph("5. Two-Phase Architecture", s['SectionHead']))

    story.append(Paragraph("Phase 1: Federated Training (run_headless.py)", s['SubHead']))
    story.append(Paragraph(
        "Phase 1 runs standard federated averaging with AdaGuard's leak detection active. "
        "Each round: the server selects clients, clients train locally and compute gradients, "
        "LeakScore metrics are evaluated, adaptive encryption is applied, and protected gradients "
        "are aggregated. All per-client data is saved as artifacts for Phase 2 replay.", s['Body']))

    story.append(Paragraph("Artifacts Saved Per Client Per Round:", s['SubSubHead']))
    art_items = [
        "<b>gradient_dict</b> -- Raw per-layer gradients (for attacks and metric recomputation)",
        "<b>flat_gradient</b> -- Flattened gradient vector (for entropy metrics)",
        "<b>local_state_dict</b> -- Full model state including BatchNorm running stats",
        "<b>local_weights</b> -- Parameter weights (for MaskCrypt vulnerability scoring)",
        "<b>outputs</b> -- Model logits (B, C) (for ConfidenceGap recomputation)",
        "<b>images, labels</b> -- Original training batch (for reconstruction metric ground truth)",
        "<b>fisher_result</b> -- Pre-computed Fisher information (for encryption)",
        "<b>metrics</b> -- All computed scores (entropy_avg, label_avg, combined_leakscore, etc.)",
    ]
    for item in art_items:
        story.append(Paragraph(item, s['BulletItem'], bulletText='\xe2\x80\xa2'))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Phase 2: Scenario Evaluation (run_sensitivity.py)", s['SubHead']))
    story.append(Paragraph(
        "Phase 2 loads saved artifacts and replays them under different configurations without "
        "retraining. Each scenario applies its own encryption strategy, recomputes metrics where "
        "necessary (e.g., different entropy bins), and optionally runs full paper-matched attacks "
        "(GradInversion 20K iterations, GI-NAS 8 restarts, GGCDM 1K diffusion steps).", s['Body']))

    story.append(Paragraph("Phase 2 Workflow Per Round:", s['SubSubHead']))
    phase2_steps = [
        "1. Load global model state + client artifacts from Phase 1",
        "2. Recompute LeakScore with scenario's weights (alpha, beta, gamma)",
        "3. Auto-detect if metrics need recomputation (changed entropy_bins, focus_layers, etc.)",
        "4. Apply scenario's encryption strategy (none, full, dp, maskcrypt, fisher)",
        "5. Re-aggregate protected gradients via FedAvg",
        "6. Evaluate model accuracy on test set",
        "7. Run full attacks on encrypted gradients (optional)",
        "8. Compute reconstruction metrics (LPIPS, SSIM, PSNR, MSE, FID, ASR)",
        "9. Log to Tensorboard + save JSON results + generate CSV summaries",
    ]
    for step in phase2_steps:
        story.append(Paragraph(step, s['BulletItem']))
    story.append(PageBreak())

    # ===================================================================
    # 6. LEAKSCORE PIPELINE
    # ===================================================================
    story.append(Paragraph("6. LeakScore Pipeline", s['SectionHead']))
    story.append(Paragraph(
        "LeakScore is a composite metric that quantifies how much private information a client's "
        "gradient update reveals. It combines three families of sub-metrics, each capturing a "
        "different facet of gradient leakage risk.", s['Body']))

    # Combined formula
    story.append(Paragraph("Combined LeakScore (Eq. 1)", s['SubHead']))
    story.append(Paragraph(
        "LeakScore = (alpha * LabelAvg + beta * EntropyAvg + gamma * EmpiricalAvg)"
        " / (alpha + beta + gamma)", s['Equation']))
    story.append(Paragraph(
        "where LabelAvg = mean(GLMIP, ConfidenceGap, CosineSimilarity), "
        "EntropyAvg = mean(Shannon, Renyi, MinEntropy), "
        "and EmpiricalAvg = mean of lightweight GI attack scores. "
        "Output is clamped to [0, 1].", s['Body']))

    # GLMIP
    story.append(Paragraph("6.1 GLMIP -- Gradient-Label Mutual Information Proxy (Eq. 3)", s['SubHead']))
    story.append(Paragraph(
        "Score<sub>GLMIP</sub> = 1 - H(P(g|c)) / log(C)", s['Equation']))
    story.append(Paragraph(
        "Measures how non-uniformly gradient energy is distributed across classes. "
        "For each class c, we compute the total squared gradient norm (energy) from "
        "samples_per_class representative samples. The probability distribution P(g|c) = "
        "energy_c / total_energy is formed, and its Shannon entropy H is normalized by log(C). "
        "A score near 1.0 means one class dominates the gradient (label leaks); near 0.0 means "
        "uniform distribution (safe).", s['Body']))

    # Confidence Gap
    story.append(Paragraph("6.2 Confidence Gap (Eq. 4)", s['SubHead']))
    story.append(Paragraph(
        "Score<sub>Gap</sub> = max(|g<sub>logits</sub>|) - second_max(|g<sub>logits</sub>|)", s['Equation']))
    story.append(Paragraph(
        "where g<sub>logits</sub> = softmax(z) - one_hot(y), the logit-space gradient of cross-entropy loss. "
        "A large gap means the true class is clearly distinguishable from all others in the gradient, "
        "enabling analytical label extraction via the iDLG theorem. Uses absolute gradient magnitudes "
        "rather than softmax probabilities for paper alignment.", s['Body']))

    # Cosine Similarity
    story.append(Paragraph("6.3 Inter-Class Cosine Similarity", s['SubHead']))
    story.append(Paragraph(
        "Score<sub>Cosine</sub> = 1 - (mean_pairwise_cosine + 1) / 2", s['Equation']))
    story.append(Paragraph(
        "Computes pairwise cosine similarity between per-class mean gradient vectors. "
        "Low inter-class similarity means gradients are class-specific (label information encoded), "
        "yielding a high leak risk score.", s['Body']))

    # Entropy metrics
    story.append(Paragraph("6.4 Entropy Metrics", s['SubHead']))
    story.append(Paragraph(
        "Three entropy variants are computed from the histogram of the flattened gradient vector "
        "(normalized and binned into entropy_bins bins):", s['Body']))

    entropy_table = make_table(
        ['Metric', 'Formula', 'Sensitivity'],
        [
            ['Shannon', 'H = -Sum p*log(p);  Score = 1 - H/log(B)', 'General distribution shape'],
            ['Renyi (a=2)', 'H = -log(Sum p^2);  Score = 1 - H/log(B)', 'Dominant elements (concentration)'],
            ['Min-Entropy', 'H = -log(max(p));  Score = 1 - H/log(B)', 'Worst-case (most conservative)'],
        ],
        col_widths=[1.2*inch, 3.2*inch, 2.6*inch]
    )
    story.append(entropy_table)
    story.append(Paragraph(
        "All scores are in [0, 1] where 0 = safe (high entropy, flat gradient) and "
        "1 = risky (low entropy, peaked gradient easily reconstructed).", s['SmallNote']))
    story.append(PageBreak())

    # ===================================================================
    # 7. ENCRYPTION STRATEGIES
    # ===================================================================
    story.append(Paragraph("7. Encryption Strategies", s['SectionHead']))

    story.append(Paragraph("7.1 Adaptive Encryption Controller (AdaGuard)", s['SubHead']))
    story.append(Paragraph(
        "AdaGuard uses a tiered encryption system based on the combined LeakScore. "
        "Two thresholds (T1, T2) define three zones:", s['Body']))

    enc_table = make_table(
        ['Zone', 'Condition', 'Action', 'Encrypt %'],
        [
            ['None', 'LeakScore < T1', 'No encryption needed', '0%'],
            ['Partial', 'T1 <= LeakScore < T2', 'Linear interpolation from 0% to base%',
             'base * (LS-T1)/(T2-T1)'],
            ['Strong', 'LeakScore >= T2', 'Scale from base% toward 100%',
             'base + (1-base)*(LS-T2)/(1-T2)'],
        ],
        col_widths=[0.8*inch, 1.6*inch, 2.8*inch, 1.8*inch]
    )
    story.append(enc_table)
    story.append(Paragraph(
        "Example with T1=0.3, T2=0.7, base=10%: LeakScore=0.15 -> 0%, "
        "LeakScore=0.5 -> 5%, LeakScore=0.7 -> 10%, LeakScore=0.9 -> 55%.", s['SmallNote']))

    story.append(Paragraph("7.2 MaskCrypt (Hu &amp; Li, IEEE TDSC 2025)", s['SubHead']))
    story.append(Paragraph(
        "Our primary competitor. Selects which parameters to encrypt using a gradient-guided "
        "vulnerability score:", s['Body']))
    story.append(Paragraph(
        "v[i] = gradient[i] * (old_exposed_weight[i] - new_trained_weight[i])", s['Equation']))
    story.append(Paragraph(
        "Parameters with highest |v[i]| are most vulnerable to gradient inversion. The top rho*N "
        "parameters (default rho=10%) are encrypted. We implement both the paper's gradient-guided "
        "mode (Algorithm 2) and a random mask baseline for fair comparison.", s['Body']))

    story.append(Paragraph("7.3 Differential Privacy Baseline", s['SubHead']))
    story.append(Paragraph(
        "Gaussian mechanism matching MaskCrypt paper's baseline (epsilon=50, delta=1e-5):", s['Body']))
    story.append(Paragraph(
        "sigma = sqrt(2 * ln(1.25/delta)) * S / epsilon  (where S = gradient clip norm)", s['Equation']))
    story.append(Paragraph(
        "Process: (1) Clip gradient to max L2 norm S, (2) Add N(0, sigma^2) noise to each parameter. "
        "With default settings, sigma ~ 0.097. DP provides a mathematical privacy guarantee but "
        "degrades model accuracy, especially at stricter epsilon values.", s['Body']))

    story.append(Paragraph("7.4 Full Homomorphic Encryption", s['SubHead']))
    story.append(Paragraph(
        "All parameters encrypted (simulated by zeroing gradients from attacker's view). "
        "Provides maximum privacy but incurs up to 20x communication overhead and prevents "
        "the server from performing any computation on the ciphertext without specialized HE "
        "operations.", s['Body']))
    story.append(PageBreak())

    # ===================================================================
    # 8. GRADIENT ACCUMULATION
    # ===================================================================
    story.append(Paragraph("8. Gradient Accumulation Defense (Eq. 15)", s['SectionHead']))
    story.append(Paragraph(
        "g<sub>acc</sub> = (1/K) * Sum<sub>k=1..K</sub>"
        " [ (1/B) * Sum<sub>i=1..B</sub> grad L(f(x<sub>k,i</sub>), y<sub>k,i</sub>) ]",
        s['Equation']))
    story.append(Paragraph(
        "When a client's LeakScore exceeds the accumulation threshold (default = T2 = 0.7), "
        "the gradient is computed as the average of K independent forward-backward passes, each on "
        "a different batch of size B. The effective batch size becomes B_eff = K * B.", s['Body']))
    story.append(Paragraph(
        "This defense exploits the fundamental limitation of gradient inversion: the attacker must "
        "recover K*B images from a single averaged gradient, making the optimization problem "
        "severely underdetermined. Yin et al. (2021) showed reconstruction quality degrades sharply "
        "around batch size 8-16 for CIFAR-10 with ResNet-18.", s['Body']))

    accum_table = make_table(
        ['K', 'B_eff (B=4)', 'Privacy Effect', 'Compute Cost'],
        [
            ['1', '4', 'None (baseline) -- trivially invertible', '1x'],
            ['2', '8', 'Marginal -- attacks degrade but may succeed', '2x'],
            ['4 (default)', '16', 'Strong -- most GI attacks fail', '4x'],
            ['8', '32', 'Very strong -- uses 32 of ~48 samples per client', '8x'],
        ],
        col_widths=[0.6*inch, 1.1*inch, 3.3*inch, 1.0*inch]
    )
    story.append(accum_table)
    story.append(PageBreak())

    # ===================================================================
    # 9. ATTACK IMPLEMENTATIONS
    # ===================================================================
    story.append(Paragraph("9. Attack Implementations", s['SectionHead']))
    story.append(Paragraph(
        "AdaGuard evaluates defense quality using three state-of-the-art gradient inversion attacks, "
        "each implemented in both a lightweight version (20 iterations, for real-time LeakScore) and "
        "a full paper-matched version (for Phase 2 evaluation).", s['Body']))

    story.append(Paragraph("9.1 GradInversion (Yin et al. 2021)", s['SubHead']))
    story.append(Paragraph(
        "Optimization-based reconstruction with BatchNorm matching regularization. "
        "Full version: 20,000 iterations, Adam lr=0.1, with TV regularization (lambda=1e-4), "
        "L2 regularization (lambda=1e-6), and BN statistics matching (lambda=0.1). "
        "The BN matching is key: it constrains reconstructed images to have natural image statistics, "
        "enabling attacks at larger batch sizes (up to 48 in some cases).", s['Body']))

    story.append(Paragraph("9.2 GI-NAS (Yu et al. 2025)", s['SubHead']))
    story.append(Paragraph(
        "Neural Architecture Search-based reconstruction. Full version: 500 iterations per restart, "
        "8 restarts, Adam lr=1e-3 with cosine annealing. Uses group lasso regularization for "
        "batch-wise consistency and keeps the best reconstruction across all restarts.", s['Body']))

    story.append(Paragraph("9.3 GGCDM (Meng et al. 2025)", s['SubHead']))
    story.append(Paragraph(
        "Diffusion-based gradient-guided reconstruction. Full version: 1,000 diffusion steps, "
        "guidance rate 0.20, cosine noise schedule. Combines gradient matching loss with annealed "
        "stochastic sampling, starting with high noise (exploration) and gradually refining.", s['Body']))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Reconstruction Metrics", s['SubHead']))
    metrics_table = make_table(
        ['Metric', 'Type', 'Good Defense'],
        [
            ['LPIPS', 'Perceptual distance (learned)', 'High (images look different)'],
            ['SSIM', 'Structural similarity', 'Low (no structural match)'],
            ['PSNR', 'Peak signal-to-noise ratio', 'Low (high reconstruction error)'],
            ['MSE', 'Mean squared error', 'High (large pixel differences)'],
            ['FID', 'Frechet Inception Distance', 'High (distribution mismatch)'],
            ['ASR', 'Attack Success Rate (SSIM > threshold)', 'Low (few successful attacks)'],
        ],
        col_widths=[1.0*inch, 2.5*inch, 3.5*inch]
    )
    story.append(metrics_table)
    story.append(PageBreak())

    # ===================================================================
    # 10. GPU DETECTION
    # ===================================================================
    story.append(Paragraph("10. GPU Detection &amp; Resource Allocation", s['SectionHead']))
    story.append(Paragraph(
        "AdaGuard auto-detects available GPUs and adapts worker allocation based on VRAM. "
        "This handles heterogeneous HPC environments where different nodes have different GPU "
        "configurations, and respects CUDA_VISIBLE_DEVICES set by Slurm.", s['Body']))

    gpu_table = make_table(
        ['GPU VRAM', 'Example', 'Clients/GPU', 'Rationale'],
        [
            ['>= 32 GB', 'A100-40GB, A100-80GB', '6', 'Ample VRAM for parallel clients'],
            ['>= 16 GB', 'V100-16GB, T4-16GB', '3', 'Standard research GPU'],
            ['>= 8 GB', 'RTX 3060, RTX 5050', '2', 'Consumer/laptop GPU'],
            ['< 8 GB', 'GTX 1650, MX450', '1', 'Minimal -- sequential processing'],
        ],
        col_widths=[1.0*inch, 2.0*inch, 1.2*inch, 2.8*inch]
    )
    story.append(gpu_table)
    story.append(Paragraph(
        "Workers = num_gpus x clients_per_gpu. With 4x V100 on Matilda HPC: "
        "4 x 3 = 12 concurrent client evaluations per round.", s['SmallNote']))

    # ===================================================================
    # 11. HPC DEPLOYMENT
    # ===================================================================
    story.append(Paragraph("11. HPC Deployment (Matilda Cluster)", s['SectionHead']))
    story.append(Paragraph(
        "The Matilda HPC cluster at Oakland University provides 3 GPU nodes with 4x Tesla V100-PCIE-16GB "
        "each. AdaGuard uses Slurm array jobs to distribute scenarios across nodes.", s['Body']))

    story.append(Paragraph("Phase 1 Training (slurm_1k_clients.sh)", s['SubHead']))
    story.append(Paragraph(
        "12 array jobs: 3 seeds x 4 strategies (none, fisher, maskcrypt, full). "
        "Each job requests 4 GPUs, runs 1000 clients for 250 rounds with full artifact saving.", s['Body']))

    story.append(Paragraph("Phase 2 Scenarios (slurm_sensitivity.sh)", s['SubHead']))
    story.append(Paragraph(
        "72 array jobs (indices 0-71): indices 0-59 map to sensitivity scenarios (S1.1 through S11.3), "
        "indices 60-71 map to viability scenarios (V1 through V12). Each job requests 1 GPU "
        "and 32GB RAM (sufficient for replay).", s['Body']))

    hpc_table = make_table(
        ['Command', 'Effect'],
        [
            ['sbatch hpc/slurm_sensitivity.sh', 'Run all 72 scenarios'],
            ['sbatch --array=0-59 hpc/slurm_sensitivity.sh', 'Sensitivity only (60 jobs)'],
            ['sbatch --array=60-71 hpc/slurm_sensitivity.sh', 'Viability only (12 jobs)'],
            ['NO_ATTACKS=1 sbatch hpc/slurm_sensitivity.sh', 'Skip attacks (metrics only, fast)'],
            ['ROUNDS=0,50,100 sbatch hpc/slurm_sensitivity.sh', 'Evaluate specific rounds only'],
        ],
        col_widths=[3.5*inch, 3.5*inch]
    )
    story.append(hpc_table)
    story.append(PageBreak())

    # ===================================================================
    # 12. VERIFICATION CHECKLIST
    # ===================================================================
    story.append(Paragraph("12. Verification Checklist", s['SectionHead']))
    story.append(Paragraph(
        "The following checks verify correct implementation of all components:", s['Body']))

    checks = [
        ("Registry", "python run_sensitivity.py --list shows 72 scenarios (60 + 12)"),
        ("Scenario IDs", "S1.1 through S11.3 for sensitivity, V1 through V12 for viability"),
        ("Config merge", "_resolve_scenario_config correctly overlays overrides on DEFAULT_CONFIG"),
        ("DP baseline", "DPNoiseEncryptor sigma ~ 0.097 for eps=50, delta=1e-5, clip=1.0"),
        ("MaskCrypt random", "compute_random_mask returns mask_mode='random' with correct encrypt count"),
        ("MaskCrypt guided", "compute returns top-rho*N by descending |v[i]|"),
        ("Encryption dispatch", "Runner handles all 5 types: none, full, dp, maskcrypt, fisher"),
        ("Metric recompute", "S1 triggers entropy recomputation, S9/S11 trigger label recomputation"),
        ("GPU auto-detect", "V100 -> 3 clients/GPU, RTX 5050 -> 1 client/GPU, CPU -> fallback"),
        ("Artifact format", "Phase 1 saves gradient_dict, flat, outputs, images, labels, metrics per client"),
        ("Phase 2 replay", "ScenarioRunner loads artifacts, applies encryption, runs attacks"),
        ("CSV output", "sensitivity_S1.csv through sensitivity_S11.csv + viability_summary.csv"),
        ("TB logging", "Organized by Sensitivity/{group}/ and Viability/{scenario}/"),
        ("Slurm mapping", "Index 0 -> S1.1, index 59 -> last sensitivity, index 60 -> V1"),
    ]

    check_rows = [[name, desc] for name, desc in checks]
    check_table = make_table(
        ['Component', 'Verification'],
        check_rows,
        col_widths=[1.5*inch, 5.5*inch]
    )
    story.append(check_table)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#1a1a2e'),
                             spaceBefore=10, spaceAfter=10))
    story.append(Paragraph(
        "All tests passed as of April 7, 2026. 72 scenarios verified, "
        "all imports clean, GPU detection working on both local (RTX 5050) and HPC (V100) hardware.",
        ParagraphStyle('Final', parent=s['Body'], fontSize=9, alignment=TA_CENTER,
                       textColor=HexColor('#666666'))))

    # BUILD
    doc.build(story)
    print(f"PDF generated: {output_path}")


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'AdaGuard_Technical_Report.pdf'
    build_pdf(out)
