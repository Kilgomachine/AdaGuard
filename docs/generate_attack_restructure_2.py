"""Generate Attack Restructure 2 PDF — records the protocol-mismatch
diagnosis after Restructure 1's full-budget runs saturated at ~9 dB for
every defence, and documents the corrected plan: re-run at paper-matched
batch sizes (B=1, B=4) instead of pivoting the metric.

Supersedes the first draft of Restructure 2 (the "label-recovery pivot"),
which was based on an incorrect reading of the three target papers.
Cross-verification against Yin 2021 / Yu 2025 / Meng 2025 showed that
none of them use label-recovery or "ASR" as a headline metric — they all
report pixel PSNR / LPIPS / SSIM at specific small batch sizes. The
9 dB collapse was a protocol mismatch (B=16 on 32x32 CIFAR with Fisher
non-IID is harder than any paper's published setting), not a metric
failure.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
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
    "Correction", parent=styles["Normal"], fontSize=10, leading=14,
    spaceBefore=6, spaceAfter=6, leftIndent=10, rightIndent=10,
    backColor=HexColor("#ffebee"), borderWidth=1,
    borderColor=HexColor("#c62828"), borderPadding=8,
    textColor=HexColor("#2d3436"),
))

story = []
S = styles
hdr = HexColor("#1a1a2e")


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
def correction(t): story.append(Paragraph(t, S["Correction"]))
def gap(h=6): story.append(Spacer(1, h))


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
subtitle("AdaGuard Phase 4 &mdash; Protocol-Mismatch Diagnosis and "
         "Paper-Matched Batch-Size Sweep")
gap(6)
body("<b>Author:</b> Firas Aguir &nbsp;&bull;&nbsp; "
     "<b>Date:</b> 2026-04-22 &nbsp;&bull;&nbsp; "
     "<b>Branch:</b> master &nbsp;&bull;&nbsp; "
     "<b>Supersedes:</b> earlier draft of Restructure 2 (&ldquo;label-"
     "recovery pivot&rdquo;), which was based on a misreading of the "
     "three source papers.")

# ── 0. Correction notice ───────────────────────────────────────────
correction("<b>Correction.</b> An earlier draft of this document "
           "proposed pivoting from pixel PSNR to iDLG / LLG+ label-"
           "recovery accuracy as the headline metric, on the belief "
           "that the gradient-inversion literature generally uses "
           "label recovery as its primary ASR. A direct cross-"
           "verification against Yin 2021, Yu 2025 and Meng 2025 "
           "showed this claim was false &mdash; all three papers "
           "report pixel PSNR / LPIPS / SSIM as headline, at small "
           "batch sizes (B=1, B=4, or B=8). Label accuracy is a "
           "sub-component of GradInversion (99.56% at B=8), not the "
           "main claim. This revised draft reverts the pivot and "
           "adopts the correct diagnosis: our 9 dB collapse was a "
           "protocol mismatch, not a metric choice.")

section("Executive summary")
body("Restructure 1 shipped three paper-faithful attacks "
     "(GradInversion / GI-NAS / GGCDM) and a sanity harness. The full-"
     "budget HPC runs gave ~9 dB on every defence, which looked like "
     "saturation and suggested a metric problem. A closer reading of "
     "the three papers shows the issue was the evaluation protocol, "
     "not the metric:")
callout("<b>None</b> of the three attacks were designed for B=16 on "
        "32&times;32 CIFAR-10 with Fisher-non-IID clients. "
        "<b>GradInversion</b> evaluates on ImageNet 224&times;224 at "
        "B=8&ndash;48 (CIFAR is not in the main tables). "
        "<b>GI-NAS</b>\u2019s headline 35.99 dB is on CIFAR-10 at "
        "<b>B=4</b> (arXiv 2405.20725 v4, Table III), not B=16. "
        "<b>GGCDM</b>\u2019s 41.12 dB is on CelebA 256&times;256 at "
        "B=1. Our B=16 run is strictly harder than every published "
        "headline setting.")
body("The corrected plan: re-run the attacks at paper-matched batch "
     "sizes (B=1 and B=4) on the undefended V1 artifacts first, to "
     "verify they land their published numbers. If they do, the "
     "defence-comparison sweep (V1/V2/V4/V6) runs at B=4 where PSNR "
     "has real dynamic range. If they do not, that is a separate "
     "bug to chase in the individual attack implementations. Label "
     "recovery stays in the output as a free side-channel diagnostic "
     "(it is computed off the same gradient dict in milliseconds, no "
     "extra cost) &mdash; but it is not the headline.")

# ── 1. What the three papers actually measure ─────────────────────
section("1 &nbsp; What the three papers actually measure")

body("Cross-verified directly against the source papers on 2026-04-22:")

table([
    ["Paper", "Dataset", "Batch size", "Primary metric", "Headline"],
    ["Yin 2021 (GradInversion)",
     "ImageNet 224&times;224",
     "B = 8&ndash;48",
     "post-registration PSNR, LPIPS, IIP",
     "Not published on CIFAR-10"],
    ["Yu 2025 (GI-NAS, arXiv v4)",
     "CIFAR-10 32&times;32",
     "B = 4",
     "PSNR, SSIM, FSIM, LPIPS",
     "35.99 dB / 0.998 / 0.0009"],
    ["Meng 2025 (GGCDM)",
     "CelebA 256, LSUN, ImageNet",
     "B = 1",
     "RV, MSE, PSNR, LPIPS",
     "41.12 dB on CelebA-256"],
], [1.5 * inch, 1.25 * inch, 0.8 * inch, 1.5 * inch, 1.45 * inch])

bullet("<b>&ldquo;ASR&rdquo; is never defined or reported</b> in any "
       "of the three papers. The label-recovery framing I used in the "
       "earlier draft was wrong.")
bullet("GradInversion <i>does</i> contain a label-extraction sub-"
       "routine (reported as 99.56% at B=8), but that is a "
       "prerequisite for image reconstruction, not the paper\u2019s "
       "claim. The claim is pixel-space reconstruction quality.")
bullet("GI-NAS\u2019s v4 CIFAR number (35.99 dB at B=4) is 5 dB "
       "higher than the v1 figure (30.53 dB) I had cached in the "
       "earlier sanity harness. Updated.")

# ── 2. Why the B=16 PSNR collapse is consistent with the papers ───
section("2 &nbsp; Why the B=16 PSNR collapse is consistent")

body("Gradient inversion scales roughly as O(1 / B &middot; C) where "
     "C is the number of classes present in the batch. At <i>fixed "
     "model capacity and iteration budget</i>, the inversion problem "
     "becomes underdetermined as B grows &mdash; different pixel "
     "tensors produce the same batch-mean gradient. Our Phase-1 "
     "Fisher-non-IID sampling gives each B=16 client only ~3 unique "
     "class labels (client_106 round 249: "
     "[1, 6, 5, 5, 5, 1, 5, 6, 1, 6, 6, 1, 5, 6, 6, 5]), which "
     "further degrades identifiability. The 0.85 cosine similarity "
     "we measured between reconstructed and real gradients confirms "
     "the optimiser is working; the pixels simply do not converge "
     "uniquely.")
body("Seen through the lens of the three papers&rsquo; protocols, "
     "this is a predictable outcome:")
table([
    ["Attack @ our B=16 CIFAR", "Paper headline setting", "Expected gap"],
    ["GradInversion 9.30 dB",
     "ImageNet-224 B=8&ndash;48, post-registration",
     "large &mdash; our setting is harder and smaller-image"],
    ["GI-NAS (DIP) 7.22 dB",
     "CIFAR B=4, 35.99 dB with NAS",
     "~30 dB &mdash; split between B=16 penalty and missing NAS"],
    ["GGCDM 7.86 dB",
     "CelebA-256 B=1 with FFHQ prior",
     "out-of-domain (natural 32&times;32 vs faces-256); no signal"],
], [1.8 * inch, 2.3 * inch, 2.4 * inch])

# ── 3. Corrected implementation plan ───────────────────────────────
section("3 &nbsp; Corrected plan (commit on this branch)")

subsection("3.1 &nbsp; Batch-size override in the sanity harness")
body("<tt>tests/attack_sanity_check.py</tt> now takes <tt>--batch-size "
     "N</tt>. When N is less than the saved client batch (typically 16):")
bullet("It sub-samples the first N images and labels from the artifact.")
bullet("It reconstructs the Phase-1 <i>local</i> model as "
       "<tt>global_state &minus; weight_delta</tt> (both are saved).")
bullet("It does a forward-backward on the sub-batch to produce a "
       "fresh <tt>gradient_dict</tt> at the requested batch size.")
bullet("It feeds that gradient through the attack exactly the same "
       "way as the full-batch path. No Phase-1 rerun required; "
       "existing <tt>artifacts_none_seed42_300clients</tt> can service "
       "every B in [1, 16].")

subsection("3.2 &nbsp; Label recovery is a diagnostic, not the headline")
body("The earlier <tt>tests/label_recovery_sweep.py</tt> and "
     "<tt>hpc/slurm_label_recovery.sh</tt> are removed. "
     "<tt>adaguard/attacks/label_recovery.py</tt> is kept as a "
     "utility; the sanity harness prints and stores the label-ASR "
     "side-channel alongside every PSNR run, because the information "
     "is free (analytical, milliseconds) and useful for understanding "
     "which gradients carry classifier signal under each defence. "
     "But it is explicitly labelled a side-channel, not the paper\u2019s "
     "primary metric.")

subsection("3.3 &nbsp; Paper-matched sweep")
body("<tt>hpc/slurm_attack_sanity.sh</tt> now runs this sequence on "
     "the undefended V1 dir:")
mono("GradInversion  @ B=1   (iDLG-scale, 20 k it., Geiping recipe)\n"
     "GradInversion  @ B=4   (small-batch, 20 k it.)\n"
     "GradInversion  @ B=16  (stress test, historical comparison)\n"
     "GI-NAS (DIP)   @ B=1   (pure DIP baseline)\n"
     "GI-NAS (DIP)   @ B=4   (paper-matched, 35.99 dB target)")
body("GGCDM is omitted from the default sweep &mdash; it is a "
     "diffusion attack targeted at 256&times;256 faces and running it "
     "on 32&times;32 CIFAR is out-of-domain. Re-enable with "
     "<tt>ATTACK=ggcdm BATCH=1 sbatch hpc/slurm_attack_sanity.sh</tt> "
     "if wanted.")

# ── 4. Acceptance criteria ─────────────────────────────────────────
section("4 &nbsp; Acceptance criteria")

body("Two independent thresholds. If both are met, the paper "
     "narrative has a clean story; if either fails, we investigate "
     "the relevant attack implementation before running the V1/V6 "
     "comparison.")
table([
    ["Run", "What we expect", "What would force re-tuning"],
    ["GradInversion B=1",
     "\u2265 15 dB (iDLG-scale target)",
     "&lt; 10 dB &mdash; Geiping recipe mis-wired"],
    ["GradInversion B=4",
     "\u2265 12 dB (between B=1 and B=16)",
     "&lt; 8 dB &mdash; optimisation budget or step-size wrong"],
    ["GradInversion B=16",
     "~9 dB (reproduces Restructure 1)",
     "drift from 9.30 dB &mdash; artifact bug"],
    ["GI-NAS B=1",
     "\u2265 20 dB (DIP should cope at low B)",
     "&lt; 12 dB &mdash; DIP proxy broken"],
    ["GI-NAS B=4",
     "\u2265 25 dB (within 10 dB of paper\u2019s 35.99)",
     "&lt; 15 dB &mdash; NAS omission more costly than expected"],
], [1.4 * inch, 2.5 * inch, 2.6 * inch])

body("<i>Numbers to fill in after <tt>sbatch hpc/slurm_attack_sanity.sh</tt> "
     "completes:</i>")
table([
    ["Attack / B", "PSNR (dB)", "SSIM", "LPIPS", "Label ASR", "Time"],
    ["GradInv B=1", "__TBD__", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
    ["GradInv B=4", "__TBD__", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
    ["GradInv B=16", "__TBD__", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
    ["GI-NAS B=1", "__TBD__", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
    ["GI-NAS B=4", "__TBD__", "__TBD__", "__TBD__", "__TBD__", "__TBD__"],
], [1.0 * inch, 0.85 * inch, 0.75 * inch, 0.75 * inch, 0.95 * inch, 0.75 * inch])

# ── 5. Next steps ──────────────────────────────────────────────────
section("5 &nbsp; Next steps")
bullet("Submit <tt>sbatch hpc/slurm_attack_sanity.sh</tt> on the "
       "undefended V1 dir. Full sweep is ~6&ndash;8 hours on one V100.")
bullet("If the paper-matched runs land within their acceptance "
       "bands, fold those numbers into the paper as the &ldquo;attacks "
       "work as published&rdquo; baseline, then run the V1/V2/V4/V6 "
       "comparison at B=4 with defence simulated at sweep-time (the "
       "simulator bug we identified: artifacts save the raw pre-"
       "defence gradient, so defence must be re-applied post hoc).")
bullet("If they do not: the per-attack implementation needs a "
       "bug-hunt, not more sweeps. Most likely suspects are step-"
       "size schedule (GradInversion) or generator initialisation "
       "(GI-NAS DIP).")
bullet("Keep the 9.30 / 9.05 dB B=16 numbers in the paper as a "
       "negative-result callout, supporting the protocol-choice "
       "discussion: realistic FL clients have B=16 non-IID, which is "
       "harder than any published gradient-inversion setting.")

gap(10)
body("<i>End of Attack Restructure 2 (revised).</i>")

doc.build(story)
print(f"PDF generated: {OUTPUT}")
