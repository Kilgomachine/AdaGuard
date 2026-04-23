"""Generate Attack Restructure 2 PDF — documents the pivot from pixel PSNR
to iDLG / LLG+ label-recovery ASR after Restructure 1's full-budget runs
showed every defence collapsing to the noise floor at B=16 Fisher-non-IID.

This document is the companion to Restructure 1: it records the empirical
PSNR results that invalidated the original metric choice, the theoretical
justification for the pivot, the implementation shipped in commit c768d37,
and the acceptance criteria for the upcoming V1/V2/V4/V6 sweep.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)

OUTPUT = "docs/attack_restructure_2.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.8 * inch,
    bottomMargin=0.8 * inch,
    leftMargin=1.0 * inch,
    rightMargin=1.0 * inch,
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "DocTitle", parent=styles["Title"], fontSize=22, spaceAfter=6,
    textColor=HexColor("#1a1a2e"),
))
styles.add(ParagraphStyle(
    "DocSubtitle", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER,
    textColor=HexColor("#555555"), spaceAfter=24,
))
styles.add(ParagraphStyle(
    "SectionHead", parent=styles["Heading1"], fontSize=16, spaceBefore=20,
    spaceAfter=8, textColor=HexColor("#1a1a2e"),
))
styles.add(ParagraphStyle(
    "SubHead", parent=styles["Heading2"], fontSize=13, spaceBefore=14,
    spaceAfter=6, textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10, leading=14,
    spaceAfter=6, textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "Mono", parent=styles["Normal"], fontSize=9, leading=13,
    fontName="Courier", spaceBefore=6, spaceAfter=6,
    backColor=HexColor("#f5f6fa"), borderWidth=1,
    borderColor=HexColor("#dcdde1"), borderPadding=6,
))
styles.add(ParagraphStyle(
    "BulletItem", parent=styles["Normal"], fontSize=10, leading=14,
    leftIndent=20, bulletIndent=10, spaceAfter=3,
    textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "Callout", parent=styles["Normal"], fontSize=10, leading=14,
    spaceBefore=6, spaceAfter=6, leftIndent=10, rightIndent=10,
    backColor=HexColor("#fff6e5"), borderWidth=1,
    borderColor=HexColor("#f0b042"), borderPadding=8,
    textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "Success", parent=styles["Normal"], fontSize=10, leading=14,
    spaceBefore=6, spaceAfter=6, leftIndent=10, rightIndent=10,
    backColor=HexColor("#e8f5e9"), borderWidth=1,
    borderColor=HexColor("#4caf50"), borderPadding=8,
    textColor=HexColor("#2d3436"),
))

story = []
S = styles


def title(t): story.append(Paragraph(t, S["DocTitle"]))
def subtitle(t): story.append(Paragraph(t, S["DocSubtitle"]))
def section(t):
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#dcdde1"),
                            spaceBefore=12, spaceAfter=4))
    story.append(Paragraph(t, S["SectionHead"]))
def subsection(t): story.append(Paragraph(t, S["SubHead"]))
def body(t): story.append(Paragraph(t, S["Body"]))
def mono(t): story.append(Paragraph(t, S["Mono"]))
def bullet(t): story.append(Paragraph(f"\u2022 &nbsp;{t}", S["BulletItem"]))
def callout(t): story.append(Paragraph(t, S["Callout"]))
def success(t): story.append(Paragraph(t, S["Success"]))
def gap(h=6): story.append(Spacer(1, h))


hdr = HexColor("#1a1a2e")


def table(data, widths):
    data_p = [[Paragraph(c, S["Body"]) for c in row] for row in data]
    t = Table(data_p, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), hdr),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dcdde1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    gap()


# ── Title page ──────────────────────────────────────────────────────
title("Attack Restructure 2")
subtitle("AdaGuard Phase 4 &mdash; Pivot from Pixel PSNR to iDLG / LLG+ Label Recovery")
gap(6)
body("<b>Author:</b> Firas Aguir &nbsp;&bull;&nbsp; "
     "<b>Date:</b> 2026-04-22 &nbsp;&bull;&nbsp; "
     "<b>Branch:</b> master &nbsp;&bull;&nbsp; "
     "<b>Commit:</b> c768d37 &nbsp;&bull;&nbsp; "
     "<b>Supersedes:</b> metric choice from <i>attack_restructure_1.pdf</i>")

section("Executive summary")
body("Restructure 1 shipped three paper-faithful gradient-inversion attacks "
     "(GradInversion / GI-NAS / GGCDM) with the understanding that their "
     "reconstruction quality would separate defended from undefended clients on "
     "pixel PSNR. The full-budget HPC sanity run (Slurm jobs 181354, 181355, "
     "181357) shows that this is not the case: every attack saturates at the "
     "~9 dB noise floor regardless of defence. The clean Geiping core on V1 "
     "undefended lands at <b>9.30 dB</b> PSNR; on V6 AdaGuard-defended it "
     "lands at <b>9.05 dB</b>. A 0.25 dB delta is within run-to-run noise.")

callout("<b>Three observations force the pivot.</b> (1) Every defence, "
        "including V1 <i>undefended</i>, sits at the reconstruction noise "
        "floor. (2) Cosine gradient similarity still reaches 0.85 on V1, "
        "so the attack loop is working; the bottleneck is the objective, "
        "not the optimiser. (3) At batch size B=16 with Fisher-non-IID "
        "sampling each client has only ~3 unique labels, which makes the "
        "gradients approximately collinear and pixel-level recovery "
        "ill-posed. The paper\u2019s 94.5% ASR headline is therefore "
        "almost certainly <b>label-recovery accuracy</b> (iDLG / LLG+), "
        "not pixel PSNR.")

success("<b>What shipped in commit c768d37.</b> An analytical label-"
        "recovery metric (<tt>adaguard/attacks/label_recovery.py</tt>), "
        "a per-defence sweep harness "
        "(<tt>tests/label_recovery_sweep.py</tt>), an integration hook in "
        "the existing sanity script, and a CPU-only Slurm wrapper "
        "(<tt>hpc/slurm_label_recovery.sh</tt>). The whole 4-defence "
        "sweep runs in seconds &mdash; no optimisation loop.")

# ── Section 1 — the PSNR collapse ─────────────────────────────────
section("1 &nbsp; Why pixel PSNR is the wrong metric here")

subsection("1.1 &nbsp; Full-budget PSNR at round 249")
body("The following numbers come from the Restructure-1 Slurm runs against "
     "one representative client (<tt>client_106</tt>) pulled from the 300-"
     "client seed-42 Phase-1 artifacts. All attacks at the full paper "
     "iteration budget:")
table([
    ["Attack", "V1 undefended", "V6 AdaGuard", "\u0394 (dB)", "Wallclock"],
    ["GradInversion (Geiping core, 20 k it.)",
     "<b>9.30 dB</b>", "<b>9.05 dB</b>", "+0.25", "637 s"],
    ["GI-NAS (DIP proxy, 2 k it.)",
     "7.22 dB", "~ noise floor", "\u2248 0", "88 s"],
    ["GGCDM (mr = 0.20, DPS scale)",
     "7.86 dB", "~ noise floor", "\u2248 0", "16 s"],
], [2.3 * inch, 1.1 * inch, 1.1 * inch, 0.85 * inch, 0.85 * inch])

subsection("1.2 &nbsp; What is actually happening")
body("The optimiser is not the problem. On V1 undefended we reach cosine "
     "gradient similarity 0.85 between the reconstructed and the real "
     "gradient &mdash; the inversion is locating a gradient that explains "
     "the observation well. The issue is that <i>many different pixel "
     "tensors explain it equally well</i>: under Fisher-non-IID sampling, "
     "each B=16 client tends to draw from 2&ndash;3 classes at a time "
     "(client_106 in this run had labels "
     "[1, 6, 5, 5, 5, 1, 5, 6, 1, 6, 6, 1, 5, 6, 6, 5] &mdash; three "
     "unique values over sixteen samples). When a batch contains many "
     "copies of similar classes, the per-sample gradients are almost "
     "collinear, the inversion is underdetermined, and the final pixels "
     "do not converge to anything recognisable no matter how long the "
     "optimiser runs.")

subsection("1.3 &nbsp; Consistency with the round-50 ablation")
body("Slurm job 181355 reran GradInversion at round 50 (early training, "
     "sharper gradients) and obtained <b>7.29 dB</b> &mdash; worse than "
     "round 249, not better. This confirms that the training-stage "
     "sharpness of the global model is not the bottleneck: the bottleneck "
     "is the B=16 non-IID label structure of Phase-1 clients.")

# ── Section 2 — the literature maps better to label ASR ─────────
section("2 &nbsp; What the literature actually measures")

body("Most federated-learning privacy papers report <b>ASR</b> "
     "(Attack Success Rate) without fully specifying what counts as a "
     "success. Two metrics dominate:")
bullet("<b>Label recovery accuracy</b> &mdash; fraction of a batch\u2019s "
       "true class labels that the attacker can extract from the "
       "gradient. Analytical, no optimisation loop. Canonical for iDLG "
       "(Zhao 2021) and LLG / LLG+ (Wainakh 2022).")
bullet("<b>Pixel-visible ASR</b> &mdash; fraction of samples whose "
       "reconstructed image is \u201crecognisable\u201d (typically PSNR "
       "&gt; 15 dB on CIFAR-10). Expensive to evaluate, sensitive to "
       "batch-level identifiability.")
body("A 94.5% headline against a 300-client B=16 federated deployment is "
     "implausible under the pixel-visible definition (it would imply "
     "essentially perfect reconstruction of arbitrary batches) but is "
     "routine under the label-recovery definition on undefended "
     "gradients: iDLG gives analytical &ldquo;one sample, one argmin&rdquo; "
     "extraction, and LLG+ extends it to batches with only a small "
     "accuracy cost.")

callout("<b>Theoretical backing (iDLG theorem).</b> For cross-entropy "
        "with softmax on a batch of B samples, "
        "<tt>dL/db[c] = (1/B) * \u03a3_i (p_i[c] &minus; 1_{c = y_i})</tt>. "
        "Classes that are <i>present</i> in the batch push their bias-"
        "gradient entry strongly negative, classes that are "
        "<i>absent</i> stay near zero. Ranking classes by "
        "&minus;dL/db and allocating <tt>B</tt> slots by magnitude "
        "(LLG+) recovers the batch label multiset in closed form.")

# ── Section 3 — what shipped ────────────────────────────────────
section("3 &nbsp; Implementation (commit c768d37)")

subsection("3.1 &nbsp; Analytical recovery module")
body("<tt>adaguard/attacks/label_recovery.py</tt> exposes three symbols:")
bullet("<tt>recover_labels_llg(grad_dict, batch_size, num_classes)</tt> "
       "&mdash; returns a length-B predicted-label multiset plus metadata "
       "(which gradient key was consulted, what the signal strength was).")
bullet("<tt>label_recovery_accuracy(predicted, truth)</tt> &mdash; "
       "multiset-overlap accuracy in [0, 1]. Order-independent; counts "
       "match repeats correctly (a batch "
       "with [1, 1, 5, 5, 6, 6, \u2026] demands the prediction also "
       "contains two 1s, not just one).")
bullet("<tt>label_recovery_summary(grad_dict, truth_labels, num_classes)</tt> "
       "&mdash; one-shot helper that returns a JSON-ready dict with ASR, "
       "predicted/truth lists, and signal strength.")

subsection("3.2 &nbsp; Fallback when fc.bias is encrypted")
body("AdaGuard\u2019s Fisher selector ranks parameters by "
     "<tt>F_i = g_i\u00b2</tt> and encrypts the top-k. The final-layer "
     "weights carry the largest gradient magnitude in a reasonably-"
     "trained network, so <tt>fc.bias</tt> is almost always in that "
     "top-k. If the module detects <tt>|fc.bias.grad|\u2081 = 0</tt> "
     "it falls back to the row-sum of <tt>fc.weight.grad</tt>, which "
     "carries the same sign pattern whenever the penultimate "
     "activations are non-negative (true after ReLU + average-pool, as "
     "in ResNet-18). The fallback means the attacker loses ground only "
     "if <i>both</i> fc.weight and fc.bias are encrypted &mdash; which "
     "is precisely what we want the Fisher selector to be doing.")

subsection("3.3 &nbsp; Local self-tests")
body("Synthetic ResNet-style final layer (dim=512, C=10, B=16, label "
     "distribution [1, 6, 5, 5, 5, 1, 5, 6, 1, 6, 6, 1, 5, 6, 6, 5]):")
table([
    ["Scenario", "Simulates", "LLG+ ASR"],
    ["Raw fc.bias + fc.weight", "V1 undefended",       "93.75%"],
    ["fc.bias = 0, fc.weight leaks",
     "V4 MaskCrypt (bias-only mask)",                 "93.75%"],
    ["fc.bias = 0, fc.weight = 0",
     "V2 full HE / V6 Fisher top-k hit",              "0.00%"],
], [2.7 * inch, 2.2 * inch, 1.1 * inch])
body("Interpretation: label recovery is robust to bias-only encryption "
     "but collapses when both weight and bias components of the "
     "classifier are hidden. This is exactly the signal-vs-noise "
     "contrast we need the sweep to reproduce on real Phase-1 "
     "artifacts.")

subsection("3.4 &nbsp; Sweep harness")
body("<tt>tests/label_recovery_sweep.py</tt> iterates over an arbitrary "
     "list of <tt>label=path</tt> defence directories and reports the "
     "mean / min / max ASR over every client at a given round:")
mono("python tests/label_recovery_sweep.py --round 249 \\\n"
     "    --dir V1=/scratch/.../artifacts_none_seed42_300clients \\\n"
     "    --dir V2=/scratch/.../artifacts_full_seed42_300clients \\\n"
     "    --dir V4=/scratch/.../artifacts_maskcrypt_seed42_300clients \\\n"
     "    --dir V6=/scratch/.../artifacts_fisher_seed42_300clients \\\n"
     "    --out results/label_recovery_sweep.json")
body("<tt>hpc/slurm_label_recovery.sh</tt> wraps that command for Matilda "
     "(<tt>partition=general</tt>, no GPU, 30-minute cap; in practice the "
     "whole sweep completes in well under a minute).")

# ── Section 4 — expected outcomes ─────────────────────────────────
section("4 &nbsp; Acceptance criteria")

body("Submit the sweep with")
mono("sbatch hpc/slurm_label_recovery.sh")
body("and record the numbers below. The paper narrative requires the "
     "following contrast:")
table([
    ["Defence", "What we expect", "What would force re-tuning"],
    ["V1 undefended",
     "\u2265 80% mean ASR (iDLG is analytical on a clean fc.bias)",
     "&lt; 60% &mdash; bug in LLG+ or clients have highly degenerate "
     "label sets"],
    ["V2 full HE",
     "\u2264 15% (signal \u2248 0, falls to uniform-guess baseline)",
     "&gt; 25% &mdash; residual plaintext leakage, re-audit HE wrapper"],
    ["V4 MaskCrypt",
     "Depends on mask policy for fc.*; 30&ndash;70% is the "
     "interesting band",
     "Report observed value either way &mdash; no intrinsic target"],
    ["V6 AdaGuard",
     "\u2264 20% (Fisher top-k should cover fc.weight and fc.bias)",
     "&gt; 50% &mdash; Fisher selector is not targeting the label-"
     "critical gradient; investigate top-k ordering"],
], [1.0 * inch, 2.4 * inch, 2.6 * inch])

body("<i>Numbers to fill in after <tt>slurm_label_recovery.sh</tt> "
     "completes:</i>")
table([
    ["Defence", "Mean ASR", "Min / Max ASR", "Signal (mean &#124;&#124;g&#124;&#124;\u2081)", "Source key"],
    ["V1", "__TBD__", "__TBD__", "__TBD__", "fc.bias"],
    ["V2", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
    ["V4", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
    ["V6", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
], [0.7 * inch, 1.0 * inch, 1.1 * inch, 2.0 * inch, 1.2 * inch])

# ── Section 5 — next steps ─────────────────────────────────────────
section("5 &nbsp; Next steps")
bullet("Run the sweep at round 249 and again at round 50 &mdash; label "
       "recovery sometimes trends <i>up</i> with training as the softmax "
       "sharpens, which would be a second independent corroboration.")
bullet("If V1 is in the expected band and V6 collapses as predicted: "
       "add a \u00a7 to the paper titled <i>\u201cBeyond pixel PSNR: "
       "label-recovery ASR and AdaGuard\u2019s Fisher selector\u201d</i> "
       "and fold the numbers into the main experiment table.")
bullet("If V6 does <i>not</i> collapse: inspect "
       "<tt>fisher_result['weights_to_encrypt']</tt> &mdash; the Fisher "
       "top-k may be systematically missing fc.*, in which case the "
       "selector\u2019s normalisation needs revisiting.")
bullet("Fold the pixel-PSNR results (9.30 vs 9.05 dB) into the paper as "
       "a negative result supporting the metric-choice section &mdash; "
       "they are not worthless, they demonstrate why the obvious metric "
       "collapses at realistic batch sizes.")

gap(10)
body("<i>End of Attack Restructure 2.</i>")

doc.build(story)
print(f"PDF generated: {OUTPUT}")
