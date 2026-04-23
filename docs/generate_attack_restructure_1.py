"""Generate Attack Restructure 1 PDF — documents the findings of the code-
availability audit for GradInversion / GI-NAS / GGCDM and the resulting
implementation plan for Phase 4."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)

OUTPUT = "docs/attack_restructure_1.pdf"

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
def gap(h=6): story.append(Spacer(1, h))


# ── Title page ──────────────────────────────────────────────────────
title("Attack Restructure 1")
subtitle("AdaGuard Phase 4 &mdash; Gradient Inversion Attack Audit and Implementation Plan")
gap(6)
body("<b>Author:</b> Firas Aguir &nbsp;&bull;&nbsp; "
     "<b>Date:</b> 2026-04-22 &nbsp;&bull;&nbsp; "
     "<b>Branch:</b> master &nbsp;&bull;&nbsp; "
     "<b>Supersedes:</b> paper-faithful reimplementations in commit 8036c74")

section("Executive summary")
body("After failing to reproduce paper numbers with our paper-faithful "
     "reimplementations (commit 8036c74), we audited the public-code status of the "
     "three attacks that drive AdaGuard\u2019s V1 threat model: <b>GradInversion</b> "
     "(Yin 2021), <b>GI-NAS</b> (Yu 2025) and <b>GGCDM</b> (Meng 2025). "
     "This document records the findings of that audit, explains two load-bearing "
     "discrepancies we uncovered, and locks in the implementation plan going forward.")

callout("<b>Two findings that change the plan.</b> (1) <b>GGCDM\u2019s 41 dB "
        "target is misattributed:</b> Meng 2025 reports that number on CelebA "
        "256&times;256 with an FFHQ-pretrained diffusion model, not on "
        "CIFAR-10&mdash;they never evaluate on CIFAR. (2) <b>No official code "
        "exists for any of the three attacks;</b> only <i>breaching</i> "
        "(Geiping\u2019s lab) ships a faithful re-implementation of Yin\u2019s "
        "GradInversion.")

# ── Findings ──────────────────────────────────────────────────────
section("1 &nbsp; Code availability audit")

subsection("1.1 &nbsp; Summary table")

hdr = HexColor("#1a1a2e")
t_data = [
    ["Attack", "Paper", "Official code", "Trusted 3rd-party", "Paper CIFAR-10 B=16"],
    ["GradInversion", "Yin&nbsp;2021 (NVIDIA, CVPR)",
     "None (never released)",
     "breaching.seethroughgradients", "~15 dB PSNR"],
    ["GI-NAS", "Yu&nbsp;2025 (IEEE TIFS)",
     "None",
     "None (not in MIA-ToolBox)", "30.53 dB PSNR (Table IV)"],
    ["GGCDM", "Meng&nbsp;2025 (arXiv 2511.10423)",
     "None",
     "None",
     "Not evaluated on CIFAR"],
]
t_data_p = [[Paragraph(c, S["Body"]) for c in row] for row in t_data]
t = Table(t_data_p, colWidths=[1.0*inch, 1.25*inch, 1.25*inch, 1.35*inch, 1.55*inch])
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

subsection("1.2 &nbsp; GradInversion (Yin 2021)")
body("NVIDIA never released the &ldquo;See Through Gradients&rdquo; codebase. "
     "The closest authoritative reference is Jonas Geiping\u2019s <i>breaching</i> "
     "framework, which ships the attack as <tt>seethroughgradients.yaml</tt> with "
     "these hyperparameters:")
bullet("Optimiser: Adam, lr = 0.1, cosine-annealing schedule")
bullet("Loss: Euclidean + TV=1e-4 + L2=1e-6 + BN-feature match = 0.1")
bullet("Budget: 20 000 iterations, warmup 50, Langevin noise \u03c3 = 0.01")
bullet("Box constraints on reconstructed pixels, focus-layer selection on logits")
bullet("<b>Gap:</b> group-consistency registration (Yin Eq 12) is marked "
       "&ldquo;not implemented&rdquo; &mdash; the same gap we had in our reimpl.")

subsection("1.3 &nbsp; GI-NAS (Yu 2025)")
body("Paper: <i>GI-NAS: Boosting Gradient Inversion Attacks Through Adaptive "
     "Neural Architecture Search</i>, IEEE TIFS. ArXiv 2405.20725.")
bullet("Confirmed <b>no</b> GitHub, no release link in the paper, no code in "
       "<tt>ffhibnese/Model-Inversion-Attack-ToolBox</tt> "
       "(MIA-ToolBox is <i>model</i>-inversion only; it does not cover "
       "federated-learning <i>gradient</i>-inversion).")
bullet("Algorithm core: signed-gradient Adam + negative-cosine matching on an "
       "over-parameterised U-Net generator. The NAS loop searches 2<super>t\u00b2</super> "
       "skip-connection patterns \u00d7 five upsample-module choices.")
bullet("Reported CIFAR-10 B=16 performance: PSNR 30.53, SSIM 0.9948, LPIPS 0.0035 "
       "(paper Table IV). NAS wall-time: ~6.6 min/batch (Table VII).")
bullet("The paper\u2019s own framing (&ldquo;the architecture itself acts as an "
       "implicit prior&rdquo;) explicitly invokes deep image prior theory "
       "&mdash; meaning a fixed over-parameterised U-Net captures the main idea "
       "even without the NAS loop.")

subsection("1.4 &nbsp; GGCDM (Meng 2025)")
body("Paper: <i>Enhanced Privacy Leakage from Noise-Perturbed Gradients via "
     "Gradient-Guided Conditional Diffusion Models</i>, arXiv 2511.10423 "
     "(posted 2025-11-16). No GitHub, no code release statement in the paper.")
callout("<b>Scale / domain mismatch.</b> The paper\u2019s headline 41.12 dB PSNR "
        "is achieved on <b>CelebA 256&times;256</b> faces using a diffusion "
        "model pre-trained on <b>FFHQ</b>. The paper is evaluated on "
        "CelebA, LSUN and ImageNet&mdash;never on CIFAR-10. Under "
        "additive Gaussian noise of variance 0.01 the same setting drops to "
        "14.98 dB PSNR.")
body("Algorithmic details we confirmed from the paper:")
bullet("Sampler: DDIM reverse process (not DDPM)")
bullet("Guidance: blend between unconditional DDIM step <i>d_sample</i> and "
       "conditional gradient-matching direction <i>d_\u2605</i> with mixing rate "
       "<b>mr = 0.20</b> (Table 6 ablation chose 0.20, not 0.50)")
bullet("Gradient-matching loss: Euclidean (L2), not cosine")
bullet("No DP-SGD in the paper &mdash; only raw additive Gaussian noise on "
       "gradients")

# ── Implications ──────────────────────────────────────────────────
section("2 &nbsp; Implications for AdaGuard")

subsection("2.1 &nbsp; The V1 target numbers need revision")
body("<tt>tests/attack_sanity_check.py</tt> currently uses:")
mono("PAPER_TARGETS = {'gradinversion': 15.0, 'gi_nas': 30.5, 'ggcdm': 41.0}")
body("Only the first two are defensible against the primary literature. "
     "The 41 dB figure for GGCDM is a citation mismatch &mdash; it is the "
     "number the <i>authors</i> report on <i>CelebA 256</i>, not on CIFAR-10 "
     "B=16. We will re-set that target empirically from the first successful "
     "run (see &sect;4).")

subsection("2.2 &nbsp; Our Yin extensions are net-negative in our hands")
body("Round-0 / round-249 ablation on Phase-1 artefacts (Slurm job 181351) "
     "showed the clean Geiping recipe beats our Yin reimpl everywhere:")
t_data = [
    ["Attack / Variant", "Round 0 PSNR", "Round 249 PSNR"],
    ["Our Yin reimpl, L2 loss", "5.15 dB", "7.14 dB"],
    ["Our Yin reimpl, cosine loss", "5.14 dB", "6.08 dB"],
    ["Inline Geiping (cosine + sign + TV)", "<b>8.13 dB</b>", "<b>9.65 dB</b>"],
]
t_data_p = [[Paragraph(c, S["Body"]) for c in row] for row in t_data]
t = Table(t_data_p, colWidths=[3.0*inch, 1.5*inch, 1.5*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), hdr),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dcdde1")),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story.append(t)
gap()
body("Conclusion: our BN-feature matching, Langevin annealing and group-"
     "consistency registration as currently implemented contribute more noise "
     "than signal. The Geiping baseline provides the reliable optimisation "
     "core that all three target attacks build on; we will reuse it via "
     "<i>breaching</i> for GradInversion and adapt the same core for the two "
     "attacks that have no public code.")

# ── Plan ──────────────────────────────────────────────────────────
section("3 &nbsp; Implementation plan")

subsection("3.1 &nbsp; GradInversion &rarr; breaching.seethroughgradients")
bullet("Install <tt>breaching</tt> into the HPC venv.")
bullet("Replace <tt>adaguard/attacks/grad_inversion.GradInversionFull</tt> "
       "with a thin dispatcher that constructs a breaching <tt>User</tt> and "
       "<tt>Attacker</tt> using <tt>seethroughgradients.yaml</tt>, feeds it our "
       "<tt>gradient_dict</tt>, and returns the reconstruction in the shape "
       "the harness expects.")
bullet("Accept the group-consistency (Eq 12) gap: breaching has the same gap, "
       "Yin\u2019s ablations show it is a minor booster, and closing it is a "
       "week of work for &lt;1 dB expected gain.")

subsection("3.2 &nbsp; GI-NAS &rarr; over-parameterised U-Net DIP (best-effort)")
body("Since there is no public code and re-implementing the per-batch NAS "
     "is ~6.6 min/batch of expensive search, we adopt a <b>deep-image-prior "
     "proxy</b> that keeps the claim the paper leans on "
     "(&ldquo;architecture as implicit prior&rdquo;) without the NAS loop:")
bullet("Fixed over-parameterised U-Net generator, randomly initialised, "
       "tanh output mapped to CIFAR-normalised input space.")
bullet("Signed-gradient Adam, lr 1e-3, negative-cosine gradient-matching loss "
       "&mdash; verbatim from the paper\u2019s equation (7).")
bullet("Single run, no NAS search. Expected PSNR on V1 CIFAR-10 B=16: 20-25 "
       "dB (below the paper\u2019s 30.5 dB which is the <i>post-NAS</i> "
       "number).")
bullet("The paper will document this as &ldquo;GI-NAS (DIP proxy)&rdquo; and "
       "cite the architectural-prior argument from Yu et al. \u00a7 III-B.")

subsection("3.3 &nbsp; GGCDM &rarr; fix our port, re-baseline target")
bullet("Keep our CIFAR-pretrained DDPM "
       "(<tt>google/ddpm-cifar10-32</tt>) &mdash; the paper\u2019s FFHQ "
       "checkpoint is a face model and is not applicable to 32&times;32 "
       "CIFAR natural images.")
bullet("Change <tt>mr</tt> from 0.50 to <b>0.20</b> to match Meng Table 6.")
bullet("Drop the unit-norm guidance normalisation; use raw gradient \u00d7 "
       "<tt>eta</tt> as the paper writes it.")
bullet("Keep L2 gradient-matching loss (paper confirms, not cosine).")
bullet("Re-measure target on CIFAR-10 B=16 V1 undefended; replace "
       "<tt>PAPER_TARGETS['ggcdm']</tt> with the observed value, annotate "
       "origin.")

subsection("3.4 &nbsp; Sanity-check policy")
body("Per user decision, Phase 4 will proceed with the <b>results of the "
     "first clean run</b>, regardless of whether any individual attack hits "
     "its headline target &mdash; provided at least one attack completes "
     "without crashing. The objective of the run is to produce comparable "
     "numbers across V1-V7 defences, not to match Yin / Yu / Meng headline "
     "PSNR. Paper narrative will adapt to whatever relative ordering falls "
     "out.")

# ── Risks ────────────────────────────────────────────────────────
section("4 &nbsp; Residual risks")
bullet("<b>breaching API drift.</b> The library is research software; its "
       "attacker/user constructors may have changed since our last touch. "
       "Mitigated by pinning the installed version and adding a smoke-import "
       "step to the Slurm script.")
bullet("<b>DIP proxy may under-perform the paper by a large margin.</b> "
       "Accepted: the alternative is dropping GI-NAS entirely or spending a "
       "week on NAS.")
bullet("<b>GGCDM on CIFAR may stay near noise-floor.</b> Accepted: we will "
       "report the observed number, not the paper\u2019s, and add a note "
       "explaining why.")
bullet("<b>Cluster has no internet</b> (model download blocked on compute "
       "nodes). Mitigated: download all checkpoints on the login node inside "
       "the venv before submitting the Slurm job.")

# ── Acceptance ───────────────────────────────────────────────────
section("5 &nbsp; Acceptance criteria for Phase 4 unblock")
bullet("At least one of {GradInversion, GI-NAS-DIP, GGCDM} runs to "
       "completion on V1 undefended CIFAR-10 without OOM or crash.")
bullet("<tt>PAPER_TARGETS</tt> in the sanity harness is updated to reflect "
       "what the chosen implementations can achieve on our setup.")
bullet("<tt>memory/project_attack_reimplementations.md</tt> is updated to "
       "note which attacks are now backed by breaching / our port, and that "
       "Phase 4 proceeds on observed relative ordering.")
bullet("A Slurm job-id is recorded in this document revision (Restructure 2) "
       "along with the first run\u2019s PSNR / SSIM / LPIPS numbers.")

gap(10)
body("<i>End of Attack Restructure 1.</i>", )

doc.build(story)
print(f"PDF generated: {OUTPUT}")
