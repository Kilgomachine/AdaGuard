#!/usr/bin/env python3
"""Generate AdaGuard Scenarios Report PDF — focused on what we test and why."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
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

    # ===================================================================
    # TITLE PAGE
    # ===================================================================
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("AdaGuard", s['DocTitle']))
    story.append(Paragraph(
        "Experiment Scenarios:<br/>"
        "What We Test and Why", s['DocSubtitle']))
    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor('#1a1a2e'),
                             spaceBefore=10, spaceAfter=10))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("72 Scenarios: 60 Sensitivity + 12 Viability", s['DocSubtitle']))
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
    # OVERVIEW
    # ===================================================================
    story.append(Paragraph("Experiment Overview", s['SectionHead']))
    story.append(Paragraph(
        "AdaGuard's experimental evaluation is organized into two complementary cases:", s['Body']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Case 1 -- Sensitivity Study (60 scenarios):</b> Isolates each configurable variable "
        "and sweeps it across a range of values while holding everything else at defaults. "
        "This answers: <i>\"Which parameters matter most, and what are their optimal values?\"</i>",
        s['Body']))
    story.append(Paragraph(
        "<b>Case 2 -- Viability Study (12 scenarios):</b> Head-to-head comparisons between "
        "AdaGuard and competing approaches (MaskCrypt, Differential Privacy, Full HE, No Defense). "
        "This answers: <i>\"Does AdaGuard actually outperform the alternatives?\"</i>",
        s['Body']))

    story.append(Spacer(1, 8))
    overview_table = make_table(
        ['', 'Sensitivity (Case 1)', 'Viability (Case 2)'],
        [
            ['Scenarios', '60', '12'],
            ['Purpose', 'Ablation / parameter tuning', 'Competitive comparison'],
            ['What varies', 'One variable at a time', 'Entire defense strategy'],
            ['Attacks', 'Optional (metrics focus)', 'All 3 full attacks per scenario'],
            ['Output', 'Per-group sweep curves', 'Head-to-head comparison tables'],
            ['Retraining', '9 need retraining, 51 reuse artifacts', '3 need retraining, 9 reuse artifacts'],
        ],
        col_widths=[1.4*inch, 2.8*inch, 2.8*inch]
    )
    story.append(overview_table)
    story.append(PageBreak())

    # ===================================================================
    # CASE 1: SENSITIVITY STUDY
    # ===================================================================
    story.append(Paragraph("Case 1: Sensitivity Study (60 Scenarios)", s['SectionHead']))
    story.append(Paragraph(
        "One variable changes at a time. Everything else stays at AdaGuard defaults: "
        "fisher encryption, T1=0.3, T2=0.7, label_weight=1.0, entropy_weight=1.0, "
        "empirical_weight=0.0, K=4, entropy_bins=50, batch_size=4.", s['Body']))

    # ----- S1 -----
    story.append(Paragraph("S1: Entropy Bins (5 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> entropy_bins = {10, 25, <u>50</u>, 100, 200}", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> All three entropy metrics (Shannon, Renyi, MinEntropy) depend on binning "
        "the flattened gradient vector into a histogram. Too few bins lose gradient structure; "
        "too many bins overfit to noise and produce unreliable entropy estimates.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> If entropy scores are sensitive to bin count, then the default "
        "of 50 needs justification. If they are stable across a wide range, we confirm the metric "
        "is robust and the exact bin count is not critical.", s['Body']))
    s1_table = make_table(
        ['ID', 'entropy_bins', 'Expected Effect'],
        [
            ['S1.1', '10', 'Very coarse -- may miss fine gradient structure'],
            ['S1.2', '25', 'Moderate -- some loss of resolution'],
            ['S1.3', '50 (default)', 'Baseline'],
            ['S1.4', '100', 'Fine-grained -- may capture more structure'],
            ['S1.5', '200', 'Very fine -- possible noise sensitivity'],
        ],
        col_widths=[0.6*inch, 1.2*inch, 5.2*inch]
    )
    story.append(s1_table)

    # ----- S2 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("S2: Gradient Accumulation K (4 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> grad_accum_K = {1, 2, <u>4</u>, 8}", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> Gradient accumulation averages K independent mini-batch gradients before "
        "sending to the server, increasing the effective batch size to B_eff = K * B. "
        "The attacker must recover K*B images from a single averaged gradient -- a much harder "
        "optimization problem.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> K=1 (no accumulation) is the baseline where attacks are easiest. "
        "Yin et al. (2021) showed reconstruction degrades sharply around batch 8-16. We need to "
        "quantify the exact privacy-compute tradeoff: each doubling of K doubles compute cost.", s['Body']))
    story.append(Paragraph(
        "<b>Note:</b> Requires retraining -- K changes how gradients are computed at training time.",
        s['SmallNote']))
    s2_table = make_table(
        ['ID', 'K', 'B_eff (B=4)', 'Expected Privacy'],
        [
            ['S2.1', '1 (disabled)', '4', 'None -- trivially invertible'],
            ['S2.2', '2', '8', 'Marginal -- attacks degrade'],
            ['S2.3', '4 (default)', '16', 'Strong -- most attacks fail'],
            ['S2.4', '8', '32', 'Very strong -- 32 images in one gradient'],
        ],
        col_widths=[0.6*inch, 1.2*inch, 1.2*inch, 4.0*inch]
    )
    story.append(s2_table)
    story.append(PageBreak())

    # ----- S3 -----
    story.append(Paragraph("S3: Encryption Threshold T1 (5 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> T1 = {0.1, 0.2, <u>0.3</u>, 0.4, 0.5} with T2 fixed at 0.7", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> T1 is the LeakScore threshold where AdaGuard starts applying partial "
        "encryption. Below T1, gradients pass through unencrypted. Setting T1 too low means "
        "encrypting safe gradients (wasted cost). Setting T1 too high means leaving risky "
        "gradients exposed.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> T1 is the first line of defense. This sweep reveals the optimal "
        "balance point where we start protecting without over-encrypting.", s['Body']))
    s3_table = make_table(
        ['ID', 'T1', 'T2', 'Expected Behavior'],
        [
            ['S3.1', '0.1', '0.7', 'Very cautious -- encrypts almost everything'],
            ['S3.2', '0.2', '0.7', 'Slightly below default'],
            ['S3.3', '0.3 (default)', '0.7', 'Baseline'],
            ['S3.4', '0.4', '0.7', 'More permissive -- fewer clients encrypted'],
            ['S3.5', '0.5', '0.7', 'Only high-risk clients get partial encryption'],
        ],
        col_widths=[0.6*inch, 0.6*inch, 0.6*inch, 5.2*inch]
    )
    story.append(s3_table)

    # ----- S4 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("S4: Encryption Threshold T2 (5 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> T2 = {0.5, 0.6, <u>0.7</u>, 0.8, 0.9} with T1 fixed at 0.3", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> T2 is the LeakScore threshold where AdaGuard escalates to strong encryption "
        "and triggers gradient accumulation. This is the \"red alert\" threshold.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> T2 controls how aggressively we protect the most vulnerable clients. "
        "Too low means unnecessary heavy encryption; too high means some genuinely risky clients "
        "only get partial protection.", s['Body']))
    s4_table = make_table(
        ['ID', 'T1', 'T2', 'Expected Behavior'],
        [
            ['S4.1', '0.3', '0.5', 'Aggressive -- strong encryption triggers early'],
            ['S4.2', '0.3', '0.6', 'Below default'],
            ['S4.3', '0.3', '0.7 (default)', 'Baseline'],
            ['S4.4', '0.3', '0.8', 'Conservative -- only very high-risk clients'],
            ['S4.5', '0.3', '0.9', 'Very conservative -- almost no strong encryption'],
        ],
        col_widths=[0.6*inch, 0.6*inch, 0.6*inch, 5.2*inch]
    )
    story.append(s4_table)
    story.append(PageBreak())

    # ----- S5, S6, S7 -----
    story.append(Paragraph("S5-S7: LeakScore Component Weights (8 runs each, 24 total)", s['SubHead']))
    story.append(Paragraph(
        "<b>Formula:</b> LeakScore = (alpha * LabelAvg + beta * EntropyAvg + gamma * EmpiricalAvg) "
        "/ (alpha + beta + gamma)", s['Equation']))
    story.append(Paragraph(
        "<b>Idea:</b> The LeakScore combines three metric families. Each group sweeps one weight "
        "from 0.0 to 3.5 while holding the others at their defaults (alpha=1.0, beta=1.0, gamma=0.0). "
        "This reveals which metric family contributes most to accurate leak detection.", s['Body']))

    story.append(Spacer(1, 6))
    story.append(Paragraph("S5: Label Weight alpha (8 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> label_weight = {0, 0.5, <u>1.0</u>, 1.5, 2.0, 2.5, 3.0, 3.5}", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> Label metrics (GLMIP, ConfidenceGap, CosineSimilarity) detect "
        "whether the gradient reveals which class the training data belongs to. Setting alpha=0 "
        "removes all label-based detection. If LeakScore accuracy drops sharply, label metrics "
        "are critical. If it barely changes, they may be redundant with entropy metrics.", s['Body']))

    story.append(Spacer(1, 6))
    story.append(Paragraph("S6: Entropy Weight beta (8 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> entropy_weight = {0, 0.5, <u>1.0</u>, 1.5, 2.0, 2.5, 3.0, 3.5}", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> Entropy metrics (Shannon, Renyi, MinEntropy) detect whether the "
        "gradient vector has low entropy (peaked, easy to invert) or high entropy (flat, harder to "
        "invert). This is complementary to label metrics -- entropy captures reconstruction "
        "difficulty regardless of label leakage.", s['Body']))

    story.append(Spacer(1, 6))
    story.append(Paragraph("S7: Empirical Weight gamma (8 runs)", s['SubSubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> empirical_weight = {<u>0</u>, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> Empirical metrics run a lightweight gradient inversion attack "
        "(20 iterations) to directly test if the gradient is invertible. This is the most "
        "computationally expensive metric family. The key question: does the cost of running "
        "GI probes in real-time justify better detection accuracy? If analytical metrics (label + "
        "entropy) are sufficient, we can skip this entirely (gamma=0, the default).", s['Body']))

    weights_table = make_table(
        ['ID Range', 'Weight', 'Sweep Values', 'Metric Family'],
        [
            ['S5.1-S5.8', 'alpha (label)', '0, 0.5, 1*, 1.5, 2, 2.5, 3, 3.5',
             'GLMIP + ConfidenceGap + CosineSimilarity'],
            ['S6.1-S6.8', 'beta (entropy)', '0, 0.5, 1*, 1.5, 2, 2.5, 3, 3.5',
             'Shannon + Renyi + MinEntropy'],
            ['S7.1-S7.8', 'gamma (empirical)', '0*, 0.5, 1, 1.5, 2, 2.5, 3, 3.5',
             'Lightweight GI attack probes'],
        ],
        col_widths=[1.0*inch, 1.2*inch, 2.3*inch, 2.5*inch]
    )
    story.append(weights_table)
    story.append(Paragraph("* = default value.", s['SmallNote']))
    story.append(PageBreak())

    # ----- S8 -----
    story.append(Paragraph("S8: Client Batch Size (5 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> client_batch_size = {1, <u>4</u>, 8, 16, 32}", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> Batch size fundamentally changes gradient leakability. A single-sample "
        "batch (B=1) is trivially invertible via the iDLG theorem -- the label is analytically "
        "extractable from the final layer gradient, and the image can be reconstructed with high "
        "fidelity. As batch size grows, multiple samples are mixed into one gradient, making "
        "inversion exponentially harder.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> This sweep validates that LeakScore correctly assigns higher risk "
        "to small batches and lower risk to large batches. It also reveals the batch size at which "
        "gradient inversion attacks become impractical.", s['Body']))
    story.append(Paragraph(
        "<b>Note:</b> Requires retraining -- different batch sizes produce fundamentally different "
        "raw gradients.", s['SmallNote']))
    s8_table = make_table(
        ['ID', 'Batch Size', 'Expected Leakability'],
        [
            ['S8.1', '1', 'Maximum -- iDLG extracts labels analytically'],
            ['S8.2', '4 (default)', 'High -- attacks succeed with effort'],
            ['S8.3', '8', 'Moderate -- reconstruction degrades'],
            ['S8.4', '16', 'Low -- most attacks fail'],
            ['S8.5', '32', 'Very low -- gradients too mixed to invert'],
        ],
        col_widths=[0.6*inch, 1.2*inch, 5.2*inch]
    )
    story.append(s8_table)

    # ----- S9 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("S9: GLMIP Samples Per Class (4 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> mi_samples_per_class = {5, 10, <u>20</u>, 40}", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> GLMIP estimates per-class gradient energy by computing forward-backward "
        "passes on representative samples from each class. More samples yield a more accurate "
        "estimate of the energy distribution P(g|c), but cost more compute.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> If GLMIP scores stabilize at 10 samples per class, using 20 "
        "wastes compute. If they are still noisy at 20, we may need 40. This calibrates the "
        "cost-accuracy tradeoff of the GLMIP metric.", s['Body']))
    s9_table = make_table(
        ['ID', 'Samples/Class', 'Total Samples (10 classes)', 'Expected'],
        [
            ['S9.1', '5', '50', 'Noisy estimates, possibly unstable'],
            ['S9.2', '10', '100', 'May be sufficient for stable ranking'],
            ['S9.3', '20 (default)', '200', 'Baseline'],
            ['S9.4', '40', '400', 'Most accurate, 2x default cost'],
        ],
        col_widths=[0.6*inch, 1.2*inch, 2.0*inch, 3.2*inch]
    )
    story.append(s9_table)
    story.append(PageBreak())

    # ----- S10 -----
    story.append(Paragraph("S10: MaskCrypt Encrypt Ratio rho (5 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> encryption_top_percent = {0.01, 0.05, <u>0.10</u>, 0.20, 0.25} "
        "with encryption=maskcrypt (gradient-guided)", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> MaskCrypt (Hu &amp; Li, IEEE TDSC 2025) encrypts only the rho fraction "
        "of parameters with highest vulnerability score v[i] = gradient[i] * (old_exposed[i] - "
        "new_trained[i]). The key finding in their paper: rho=0.01 (just 1% of parameters) "
        "already blocks gradient inversion completely.", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> This maps directly to their paper's experiments (Figs. 3, 5, 6). "
        "We reproduce their sweep to validate our MaskCrypt implementation and to provide a fair "
        "comparison point. It also reveals the minimum encryption ratio needed for privacy.", s['Body']))
    s10_table = make_table(
        ['ID', 'rho', '% Encrypted', 'Expected (per MaskCrypt paper)'],
        [
            ['S10.1', '0.01', '1%', 'Already blocks most attacks'],
            ['S10.2', '0.05', '5%', 'Strong defense'],
            ['S10.3', '0.10 (default)', '10%', 'Paper default -- very strong'],
            ['S10.4', '0.20', '20%', 'Diminishing returns on privacy'],
            ['S10.5', '0.25', '25%', 'High cost, marginal privacy gain'],
        ],
        col_widths=[0.6*inch, 0.6*inch, 1.0*inch, 4.8*inch]
    )
    story.append(s10_table)

    # ----- S11 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("S11: Focus Layers for GLMIP (3 runs)", s['SubHead']))
    story.append(Paragraph(
        "<b>Variable:</b> focus_layers = {None (all layers), <u>['fc2.weight', 'fc2.bias']</u>, "
        "['fc1.weight', 'fc1.bias']}", s['Body']))
    story.append(Paragraph(
        "<b>Idea:</b> The iDLG theorem (Zhao et al. 2020) proves that the true label can be "
        "analytically extracted from the gradient of the final fully-connected layer. GLMIP "
        "currently focuses on these final layers by default. But is this restriction optimal, "
        "or does GLMIP benefit from seeing all layers?", s['Body']))
    story.append(Paragraph(
        "<b>Why it matters:</b> If focusing on the final FC layer is clearly best, it confirms "
        "the iDLG-motivated design. If all layers or the penultimate FC performs comparably, "
        "the default could be simplified. If penultimate FC is better in some cases, that suggests "
        "label information propagates deeper than expected.", s['Body']))
    s11_table = make_table(
        ['ID', 'Focus Layers', 'Expected'],
        [
            ['S11.1', 'None (all layers)', 'GLMIP may be diluted by low-signal layers'],
            ['S11.2', 'fc2.weight, fc2.bias (default)', 'Strongest label signal per iDLG theorem'],
            ['S11.3', 'fc1.weight, fc1.bias', 'Penultimate FC -- weaker but still label-correlated'],
        ],
        col_widths=[0.6*inch, 2.4*inch, 4.0*inch]
    )
    story.append(s11_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("Sensitivity Study Summary", s['SubHead']))
    summary_table = make_table(
        ['Group', 'Variable', 'Runs', 'Retraining?'],
        [
            ['S1', 'entropy_bins', '5', 'No -- recompute entropy from saved gradients'],
            ['S2', 'grad_accum_K', '4', 'Yes -- changes gradient computation'],
            ['S3', 'T1 threshold', '5', 'No -- only changes encryption policy'],
            ['S4', 'T2 threshold', '5', 'No -- only changes encryption policy'],
            ['S5', 'label_weight (alpha)', '8', 'No -- only changes LeakScore weighting'],
            ['S6', 'entropy_weight (beta)', '8', 'No -- only changes LeakScore weighting'],
            ['S7', 'empirical_weight (gamma)', '8', 'No -- only changes LeakScore weighting'],
            ['S8', 'client_batch_size', '5', 'Yes -- different batches = different gradients'],
            ['S9', 'mi_samples_per_class', '4', 'No -- recompute GLMIP from saved gradients'],
            ['S10', 'MaskCrypt rho', '5', 'No -- re-mask saved gradients'],
            ['S11', 'focus_layers', '3', 'No -- recompute GLMIP on different layers'],
        ],
        col_widths=[0.5*inch, 1.8*inch, 0.5*inch, 4.2*inch]
    )
    story.append(summary_table)
    story.append(Paragraph("Total: 60 sensitivity runs. Only 9 require retraining (S2 + S8).", s['SmallNote']))
    story.append(PageBreak())

    # ===================================================================
    # CASE 2: VIABILITY STUDY
    # ===================================================================
    story.append(Paragraph("Case 2: Viability Study (12 Scenarios)", s['SectionHead']))
    story.append(Paragraph(
        "The viability study provides direct head-to-head comparisons proving AdaGuard works. "
        "Each scenario runs all three full paper-matched attacks and measures:", s['Body']))
    viability_metrics = [
        "<b>Privacy:</b> LPIPS, SSIM, PSNR, MSE, FID, ASR -- how well can attacks reconstruct private images?",
        "<b>Utility:</b> Model accuracy (%) -- does encryption degrade the trained model?",
        "<b>Cost:</b> % parameters encrypted, communication overhead ratio, wall-clock time",
        "<b>Detection:</b> Does LeakScore correctly predict which gradients are actually vulnerable?",
    ]
    for item in viability_metrics:
        story.append(Paragraph(item, s['BulletItem'], bulletText='\xe2\x80\xa2'))

    # ----- V1-V3 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("V1-V3: Baselines", s['SubHead']))
    story.append(Paragraph(
        "These three scenarios establish the performance boundaries that every defense "
        "method must be measured against.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V1: No Defense</b> (encryption=none)", s['SubSubHead']))
    story.append(Paragraph(
        "Gradients are sent in plaintext. This is the worst-case privacy scenario -- it shows "
        "exactly how much an attacker can reconstruct when there is zero protection. If attacks "
        "fail even here, the threat model is unrealistic. If they succeed, this quantifies the "
        "damage and establishes the baseline that all defenses must improve upon.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V2: Full Homomorphic Encryption</b> (encryption=full)", s['SubHead']))
    story.append(Paragraph(
        "All parameters encrypted. This is the best-case privacy scenario -- no attacker can "
        "reconstruct anything because all gradient information is hidden. But it comes at maximum "
        "cost: up to 20x communication overhead. This sets the ceiling: any practical defense "
        "should approach V2's privacy at a fraction of V2's cost.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V3: Differential Privacy</b> (Gaussian mechanism, eps=50, delta=1e-5)", s['SubSubHead']))
    story.append(Paragraph(
        "The industry-standard alternative: clip gradients to max L2 norm, then add calibrated "
        "Gaussian noise. We use epsilon=50 and delta=1e-5 to match the MaskCrypt paper's DP "
        "baseline. DP provides a mathematical privacy guarantee but typically degrades model "
        "accuracy. This shows how AdaGuard compares to the established approach.", s['Body']))

    # ----- V4-V5 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("V4-V5: MaskCrypt -- Our #1 Direct Competitor", s['SubHead']))
    story.append(Paragraph(
        "MaskCrypt (Hu &amp; Li, IEEE TDSC 2025) is the closest published work to AdaGuard. "
        "It selects which parameters to encrypt using a gradient-guided vulnerability score: "
        "v[i] = gradient[i] * (old_exposed_weight[i] - new_trained_weight[i]). The top rho*N "
        "parameters by |v[i]| are encrypted.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V4: MaskCrypt Gradient-Guided</b> (rho=0.10)", s['SubSubHead']))
    story.append(Paragraph(
        "The paper's recommended configuration using Algorithm 2 for intelligent mask selection. "
        "This is what we must beat. MaskCrypt's key advantage: it encrypts only 10% of parameters "
        "while blocking most attacks. AdaGuard must match this privacy level at comparable or "
        "lower cost, <i>or</i> achieve better privacy at the same cost.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V5: MaskCrypt Random Mask</b> (rho=0.10)", s['SubSubHead']))
    story.append(Paragraph(
        "Same encrypt ratio as V4, but mask indices are chosen uniformly at random instead of "
        "by vulnerability score. This isolates the value of MaskCrypt's gradient-guided selection. "
        "If V4 significantly outperforms V5, their Algorithm 2 is genuinely valuable. If the "
        "gap is small, random encryption at rho=0.10 may be sufficient.", s['Body']))
    story.append(PageBreak())

    # ----- V6-V9 -----
    story.append(Paragraph("V6-V9: AdaGuard Variants", s['SubHead']))
    story.append(Paragraph(
        "Four configurations spanning AdaGuard's operating range, from balanced defaults to "
        "extreme settings.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V6: AdaGuard Default</b> (fisher, T1=0.3, T2=0.7, no accumulation)", s['SubSubHead']))
    story.append(Paragraph(
        "Our primary result. Uses Fisher information-based encryption with the adaptive "
        "controller at default thresholds. No gradient accumulation -- pure LeakScore + "
        "adaptive encryption. This is the simplest form of AdaGuard and the configuration "
        "we expect to present as the main contribution.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V7: AdaGuard with Accumulation</b> (fisher, T1=0.3, T2=0.7, K=4)", s['SubSubHead']))
    story.append(Paragraph(
        "Full defense pipeline: LeakScore detection + adaptive encryption + gradient accumulation "
        "for high-risk clients. When LeakScore >= T2, the client computes K=4 independent "
        "gradient passes, raising B_eff from 4 to 16. This is our strongest defense and should "
        "approach V2 (full HE) privacy for high-risk clients while maintaining V6's efficiency "
        "for low-risk clients.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V8: AdaGuard Aggressive</b> (fisher, T1=0.1, T2=0.4)", s['SubSubHead']))
    story.append(Paragraph(
        "Lower thresholds mean encryption kicks in earlier and stronger. This encrypts more "
        "clients more heavily. It should approach V2's privacy at lower cost, but may over-encrypt "
        "safe clients. Useful as an upper bound on what AdaGuard can achieve if we trade off "
        "some utility for maximum privacy.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V9: AdaGuard Conservative</b> (fisher, T1=0.5, T2=0.9)", s['SubSubHead']))
    story.append(Paragraph(
        "Higher thresholds mean minimal encryption. Only genuinely high-risk clients get "
        "protected. This maximizes model utility and minimizes cost but may leave some "
        "moderately risky clients exposed. Useful as a lower bound on privacy when cost "
        "savings are the priority.", s['Body']))

    v_ag_table = make_table(
        ['ID', 'Name', 'T1', 'T2', 'K', 'Expected Result'],
        [
            ['V6', 'Default', '0.3', '0.7', 'off', 'Primary result -- balanced tradeoff'],
            ['V7', 'With Accum', '0.3', '0.7', '4', 'Strongest -- best privacy for high-risk'],
            ['V8', 'Aggressive', '0.1', '0.4', 'off', 'Upper bound -- max privacy, higher cost'],
            ['V9', 'Conservative', '0.5', '0.9', 'off', 'Lower bound -- max utility, lower privacy'],
        ],
        col_widths=[0.4*inch, 1.0*inch, 0.4*inch, 0.4*inch, 0.4*inch, 4.4*inch]
    )
    story.append(v_ag_table)

    # ----- V10-V12 -----
    story.append(Spacer(1, 10))
    story.append(Paragraph("V10-V12: Scalability Tests", s['SubHead']))
    story.append(Paragraph(
        "Does AdaGuard scale? LeakScore was designed and tested with 10 clients. Real-world "
        "federated learning involves hundreds or thousands of clients with varying participation "
        "rates. These scenarios validate that the system works at scale.", s['Body']))
    story.append(Paragraph(
        "<b>Note:</b> All three require separate Phase 1 training due to different FL topologies.",
        s['SmallNote']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V10: 100 Clients</b> (10 per round = 10% participation)", s['SubSubHead']))
    story.append(Paragraph(
        "A moderate-scale federation. Tests whether LeakScore distributions remain meaningful "
        "when client data is split across 100 partitions (smaller per-client datasets, more "
        "diverse gradients). Also tests whether the adaptive controller's thresholds need "
        "recalibration at this scale.", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V11: 1000 Clients</b> (100 per round = 10% participation)", s['SubSubHead']))
    story.append(Paragraph(
        "Large-scale federation matching our HPC 1K-client experiment. Each client has very "
        "little data (~5 samples per class). Gradients are noisier and less informative. "
        "Key question: does LeakScore still differentiate risky from safe clients?", s['Body']))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>V12: High Participation</b> (100 clients, 50 per round = 50%)", s['SubSubHead']))
    story.append(Paragraph(
        "Same client count as V10 but 5x the participation rate. With more clients per round, "
        "the server aggregates more gradients, potentially diluting individual client signals. "
        "Tests whether high participation changes the attack surface or LeakScore behavior.", s['Body']))

    v_scale_table = make_table(
        ['ID', 'Clients', 'Per Round', 'Participation', 'Key Question'],
        [
            ['V10', '100', '10', '10%', 'Does LeakScore work at moderate scale?'],
            ['V11', '1,000', '100', '10%', 'Does LeakScore work at large scale?'],
            ['V12', '100', '50', '50%', 'Does participation rate affect defense?'],
        ],
        col_widths=[0.4*inch, 0.7*inch, 0.8*inch, 0.9*inch, 4.2*inch]
    )
    story.append(v_scale_table)
    story.append(PageBreak())

    # ===================================================================
    # EXPECTED OUTCOMES
    # ===================================================================
    story.append(Paragraph("Expected Outcomes", s['SectionHead']))
    story.append(Paragraph(
        "If AdaGuard works as designed, the viability study should demonstrate:", s['Body']))

    outcomes = [
        ("<b>V1 (no defense) has high attack success</b> -- confirms the threat is real and "
         "defenses are necessary. Without this, the entire research premise falls apart."),
        ("<b>V2 (full HE) has zero attack success but maximum cost</b> -- sets the privacy "
         "ceiling. Practical defenses should approach V2's privacy at a fraction of the cost."),
        ("<b>V3 (DP) degrades model accuracy more than AdaGuard</b> -- DP adds noise to all "
         "parameters uniformly, while AdaGuard encrypts selectively. AdaGuard should preserve "
         "more model utility for comparable privacy."),
        ("<b>V4 (MaskCrypt guided) outperforms V5 (random)</b> -- validates that gradient-guided "
         "mask selection is genuinely better than random, confirming our implementation matches "
         "the paper."),
        ("<b>V6 (AdaGuard default) achieves comparable privacy to V4 at lower or equal cost</b> "
         "-- the central claim. AdaGuard's adaptive approach should be at least as effective as "
         "MaskCrypt's static approach."),
        ("<b>V7 (with accumulation) closes any privacy gap for high-risk clients</b> -- gradient "
         "accumulation provides an additional defense layer that MaskCrypt lacks."),
        ("<b>V8 vs V9 shows the tunability range</b> -- AdaGuard can be configured from "
         "aggressive to conservative depending on the deployment's privacy requirements."),
        ("<b>V10-V12 show stable LeakScore at scale</b> -- the system doesn't break down when "
         "moving from 10 to 100 to 1000 clients."),
    ]
    for item in outcomes:
        story.append(Paragraph(item, s['BulletItem'], bulletText='\xe2\x80\xa2'))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Complete Scenario Index", s['SubHead']))
    all_scenarios = make_table(
        ['Case', 'IDs', 'Count', 'Retraining'],
        [
            ['Sensitivity', 'S1.1 - S1.5', '5', 'No'],
            ['Sensitivity', 'S2.1 - S2.4', '4', 'Yes'],
            ['Sensitivity', 'S3.1 - S3.5', '5', 'No'],
            ['Sensitivity', 'S4.1 - S4.5', '5', 'No'],
            ['Sensitivity', 'S5.1 - S5.8', '8', 'No'],
            ['Sensitivity', 'S6.1 - S6.8', '8', 'No'],
            ['Sensitivity', 'S7.1 - S7.8', '8', 'No'],
            ['Sensitivity', 'S8.1 - S8.5', '5', 'Yes'],
            ['Sensitivity', 'S9.1 - S9.4', '4', 'No'],
            ['Sensitivity', 'S10.1 - S10.5', '5', 'No'],
            ['Sensitivity', 'S11.1 - S11.3', '3', 'No'],
            ['Viability', 'V1 - V3', '3', 'No'],
            ['Viability', 'V4 - V5', '2', 'No'],
            ['Viability', 'V6 - V9', '4', 'No'],
            ['Viability', 'V10 - V12', '3', 'Yes'],
        ],
        col_widths=[1.2*inch, 1.4*inch, 0.6*inch, 3.8*inch]
    )
    story.append(all_scenarios)
    story.append(Paragraph(
        "Total: 72 scenarios. 57 reuse Phase 1 artifacts (no retraining). "
        "12 require separate training runs.", s['SmallNote']))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#1a1a2e'),
                             spaceBefore=10, spaceAfter=10))
    story.append(Paragraph(
        "AdaGuard -- Oakland University -- April 2026",
        ParagraphStyle('Final', parent=s['Body'], fontSize=9, alignment=TA_CENTER,
                       textColor=HexColor('#666666'))))

    # BUILD
    doc.build(story)
    print(f"PDF generated: {output_path}")


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'AdaGuard_Scenarios_Report.pdf'
    build_pdf(out)
