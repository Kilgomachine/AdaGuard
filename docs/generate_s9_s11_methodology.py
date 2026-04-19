"""Generate methodology report for sensitivity scenarios S9 and S11.

Produces a standalone PDF documenting what each scenario tests, why it
matters, how the recompute works with slim Phase-1 artifacts, and how to
interpret the resulting sweeps.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

OUTPUT = "docs/phase2_sensitivity_s9_s11.pdf"

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
    "Formula", parent=styles["Normal"], fontSize=10, leading=16,
    fontName="Courier", alignment=TA_CENTER, spaceBefore=8, spaceAfter=8,
    backColor=HexColor("#f5f6fa"), borderWidth=1,
    borderColor=HexColor("#dcdde1"), borderPadding=8,
))
styles.add(ParagraphStyle(
    "CodeBlock", parent=styles["Normal"], fontSize=9, leading=12,
    fontName="Courier", alignment=TA_LEFT, spaceBefore=4, spaceAfter=4,
    backColor=HexColor("#f5f6fa"), borderWidth=1,
    borderColor=HexColor("#dcdde1"), borderPadding=6,
    leftIndent=6, rightIndent=6,
))
styles.add(ParagraphStyle(
    "BulletItem", parent=styles["Normal"], fontSize=10, leading=14,
    leftIndent=20, bulletIndent=10, spaceAfter=3,
    textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "Note", parent=styles["Normal"], fontSize=9, leading=12,
    textColor=HexColor("#555555"), spaceAfter=6,
    leftIndent=10, rightIndent=10,
    backColor=HexColor("#fff9e6"), borderWidth=1,
    borderColor=HexColor("#f1c40f"), borderPadding=6,
))

story = []
S = styles


def title(text): story.append(Paragraph(text, S["DocTitle"]))
def subtitle(text): story.append(Paragraph(text, S["DocSubtitle"]))
def section(text):
    story.append(HRFlowable(width="100%", thickness=1,
                            color=HexColor("#dcdde1"),
                            spaceBefore=12, spaceAfter=4))
    story.append(Paragraph(text, S["SectionHead"]))
def subsection(text): story.append(Paragraph(text, S["SubHead"]))
def body(text): story.append(Paragraph(text, S["Body"]))
def formula(text): story.append(Paragraph(text, S["Formula"]))
def code(text): story.append(Paragraph(text, S["CodeBlock"]))
def bullet(text): story.append(Paragraph(f"\u2022  {text}", S["BulletItem"]))
def note(text): story.append(Paragraph(f"<b>Note.</b> {text}", S["Note"]))
def gap(h=6): story.append(Spacer(1, h))


def make_table(rows, col_widths, header_bg="#1a1a2e"):
    rows_p = [[Paragraph(c, S["Body"]) for c in row] for row in rows]
    t = Table(rows_p, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dcdde1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ── Document ──────────────────────────────────────────────────────

title("AdaGuard Phase 2 &mdash; Sensitivity Scenarios S9 &amp; S11")
subtitle("Methodology and interpretation for label-metric sweeps "
         "under slim Phase-1 artifacts")
gap(12)

# ── 1. Context ─────────────────────────────────────────────────────
section("1. Context")
body("Phase 1 of the AdaGuard experiment trains 12 federated-learning "
     "runs (4 encryption strategies &times; 3 seeds) on CIFAR-10 with "
     "ResNet-18 across 300 non-IID clients (3 classes per client). "
     "For each of 125 saved rounds, the per-client artifact "
     "<i>gradient_dict</i>, <i>weight_delta</i>, a scalar <i>metrics</i> "
     "dict and the batch <i>labels</i> are persisted to disk. Full "
     "intermediate tensors (per-sample gradients, per-class mean "
     "gradients, per-sample logits) are <b>not</b> saved &mdash; each "
     "~86 MB client artifact already totals ~3.9 TB across the run, "
     "and keeping richer state would blow past the 10 TB project quota.")
gap()
body("Phase 2 replays those artifacts under 72 alternative configurations "
     "(60 sensitivity + 12 viability scenarios). Most scenarios reweight, "
     "rethreshold, or rescore existing metrics and need no recompute. "
     "Two groups &mdash; <b>S9</b> (GLMIP samples per class) and <b>S11</b> "
     "(focus layers) &mdash; change <i>how</i> the label leak score is "
     "computed, so their metrics must be genuinely recomputed against "
     "the model checkpoint and the public dataset. This report documents "
     "what each sweep tests, how the recompute works, and how to read "
     "the resulting sensitivity curves.")

# ── 2. S9 ──────────────────────────────────────────────────────────
section("2. S9 &mdash; GLMIP Samples Per Class")
subsection("2.1 What it tests")
body("GLMIP (Gradient-Label Mutual Information Proxy, LeakScore "
     "framework) estimates how much class-label information can be read "
     "from gradients. At its core it draws a class-balanced sample from "
     "the training distribution, computes a per-sample gradient for each, "
     "aggregates per-class gradient energy, and derives a score from the "
     "Shannon entropy of the resulting class-energy distribution.")
formula("Score<sub>GLMIP</sub> = 1 &minus; H(P(g|c)) / log(C)")
body("S9 sweeps the number of samples per class, <i>n</i>, used to "
     "estimate P(g|c). The question it answers is: <b>how many balanced "
     "samples does GLMIP need before its leak estimate stabilizes?</b> "
     "Too few samples give a noisy estimate; too many waste compute "
     "without improving the score.")

subsection("2.2 Sweep values")
story.append(make_table(
    [
        ["Scenario ID", "samples_per_class (<i>n</i>)", "Backward passes / round"],
        ["S9.1", "5",  "50  (5 &times; 10 classes)"],
        ["S9.2", "10", "100"],
        ["S9.3", "20 <b>(training default)</b>", "200"],
        ["S9.4", "40", "400"],
    ],
    col_widths=[1.2 * inch, 2.4 * inch, 2.2 * inch],
))
gap()

subsection("2.3 Recompute pipeline")
bullet("Load the global-model checkpoint saved at round <i>r</i>.")
bullet("Load a class-balanced sample of <i>n</i> images per class from "
       "the public CIFAR-10 training split.")
bullet("For each sample, compute the per-sample loss gradient on the "
       "current model weights via a single backward pass.")
bullet("Group gradients by label; compute per-class L2 energy and "
       "per-class mean gradient.")
bullet("GLMIP score from class-energy entropy; Cosine-similarity score "
       "from pairwise similarity between class mean gradients; "
       "Confidence-gap score from a separate balanced forward pass "
       "(80 samples, 8 per class) through the same model.")
bullet("Final Label LeakScore = mean(GLMIP, CosineSim, ConfidenceGap).")

subsection("2.4 Expected interpretation")
body("Plot Label LeakScore vs. <i>n</i> on log-x. Three regimes are "
     "theoretically plausible:")
bullet("<b>Under-sampling (n &lt;= 5):</b> estimate dominated by sampling "
       "noise &mdash; score may oscillate or systematically under-/over-shoot "
       "the true leak level.")
bullet("<b>Stable regime (n &asymp; 10&ndash;40):</b> score plateaus. "
       "The default <i>n</i> = 20 should sit inside this plateau; if not, "
       "the default is under-provisioned.")
bullet("<b>Diminishing returns (n &gt; 40):</b> score converges; further "
       "sampling increases wall time linearly without improving fidelity.")

note("In the paper, S9 supports the choice of default <i>n</i>. If the "
     "plateau is reached by <i>n</i> = 20 the default is justified; if "
     "S9.4 (<i>n</i> = 40) still diverges, the default is undersampling "
     "the class distribution and should be raised.")

# ── 3. S11 ─────────────────────────────────────────────────────────
section("3. S11 &mdash; Focus Layers for GLMIP")

subsection("3.1 What it tests")
body("iDLG (Zhao et al., ICLR 2021) showed that when the loss is "
     "cross-entropy, the sign of the final-layer gradient deterministically "
     "reveals each sample's label. Subsequent work &mdash; including the "
     "LeakScore framework &mdash; concentrates label-leakage measurement "
     "on the last classifier layer (&ldquo;focus layers&rdquo;) because "
     "earlier convolutional layers average label signal over many "
     "feature channels.")
gap()
body("S11 tests that assumption empirically: does restricting GLMIP / "
     "CosineSimilarity to the last FC actually give a sharper leak "
     "signal than (a) using the whole gradient, or (b) using an earlier "
     "penultimate layer?")

subsection("3.2 Sweep values (ResNet-18)")
body("The original plan listed <tt>fc1</tt> and <tt>fc2</tt> as focus "
     "targets. Those layer names came from a SmallCNN baseline and do "
     "not exist in ResNet-18; using them would silently match zero "
     "parameters and produce undefined results. For this run we use "
     "the actual ResNet-18 parameter names:")
gap()
story.append(make_table(
    [
        ["Scenario ID", "focus_layers", "Semantic meaning"],
        ["S11.1", "<tt>None</tt> (all layers)",
         "Broad label signal &mdash; baseline"],
        ["S11.2", "<tt>['fc.weight', 'fc.bias']</tt>",
         "Final classifier (iDLG-style)"],
        ["S11.3", "<tt>['layer4.1.conv2.weight']</tt>",
         "Last residual block, penultimate feature extractor"],
    ],
    col_widths=[0.9 * inch, 2.6 * inch, 2.3 * inch],
))
gap()
body("The ResNet-18 layer hierarchy flows:")
code("conv1 &rarr; bn1 &rarr; layer1 &rarr; layer2 &rarr; layer3 &rarr; "
     "layer4 &rarr; avgpool &rarr; fc")
body("<tt>layer4.1.conv2</tt> is the 3&times;3 conv at the end of the "
     "second BasicBlock of <tt>layer4</tt>, immediately before the "
     "global average pool. It therefore sits one residual block away "
     "from the classifier and provides a clean &ldquo;feature-space&rdquo; "
     "comparator for the final-layer leak signal.")

subsection("3.3 Recompute pipeline")
body("Identical to S9's pipeline, but the gradient subset kept per "
     "sample is filtered by <tt>focus_layers</tt>:")
code("flat = cat([p.grad.flatten() for n, p in model.named_parameters()"
     " if p.grad is not None and n in focus_layers])")
body("<tt>focus_layers=None</tt> keeps every parameter's gradient (the "
     "full flattened vector). All other S11 scenarios restrict the "
     "flattening to the listed names.")

subsection("3.4 Expected interpretation")
body("Plot Label LeakScore across the three S11 cells. iDLG-style theory "
     "predicts:")
bullet("<b>S11.2 (final FC) &gt; S11.3 (penultimate conv):</b> label "
       "signal is strongest at the classifier, diluted deeper in the "
       "network.")
bullet("<b>S11.2 (final FC) &gt; S11.1 (all layers):</b> the final FC "
       "concentrates signal; averaging it with noisy intermediate "
       "gradients dilutes leakage.")
bullet("If we observe <b>S11.3 &gt; S11.2</b>, that is a nontrivial "
       "finding: ResNet-18 may encode class identity more in its "
       "penultimate features than in the classifier&rsquo;s linear "
       "projection. Worth explicit discussion in the paper.")

note("If any S11 cell drops to ~0 while others behave normally, verify "
     "the <tt>focus_layers</tt> string actually matches a "
     "<tt>named_parameter</tt> on the model. A typo produces a silent "
     "empty-gradient path.")

# ── 4. Recompute implementation details ────────────────────────────
section("4. Recompute implementation details")

subsection("4.1 Why we need to recompute at all")
body("Phase 1 stores <i>only</i> each client's layer-averaged gradient "
     "plus a small scalar <i>metrics</i> dict. It does not store per-"
     "sample gradients, per-sample logits, or per-class mean gradients, "
     "because doing so would inflate each of the 46,080 client "
     "artifacts by 10&times; and break the 10 TB quota. Changing either "
     "<tt>mi_samples_per_class</tt> (S9) or <tt>focus_layers</tt> (S11) "
     "affects which per-sample gradients and which gradient components "
     "contribute to the Label LeakScore. Without per-sample data, we "
     "cannot faithfully recompute those scores &mdash; which is why "
     "earlier scenario runs hit a <tt>CosineSimilarityMetric</tt> "
     "signature mismatch and produced garbage.")

subsection("4.2 What makes recomputation faithful")
body("Phase 1's GLMIP implementation (adaguard/metrics/label_leakscore.py, "
     "class <tt>GLMIPMetric</tt>) does not use client gradients at all. "
     "It draws class-balanced samples from the public CIFAR-10 training "
     "split, runs forward+backward through the current global model, and "
     "aggregates per-class gradient energy. Nothing in that pipeline "
     "depends on client-side slim artifacts &mdash; only on the global "
     "model checkpoint, which Phase 1 <i>does</i> save every round.")
gap()
body("Phase 2 therefore replays the exact same GLMIP call with the "
     "scenario-specified <tt>samples_per_class</tt> and <tt>focus_layers</tt>, "
     "on the global model at the target round. The resulting GLMIP "
     "score, class means (fed to CosineSimilarity), and a separate "
     "inference batch (for ConfidenceGap) produce a Label LeakScore "
     "that differs from Phase 1's only in the swept parameter.")

subsection("4.3 Cost")
body("GLMIP dominates: <i>n</i> &times; 10 backward passes through ResNet-18 "
     "per round per scenario. On a V100 this runs at roughly 100 ms per "
     "sample, so:")
story.append(make_table(
    [
        ["Scenario", "Samples / round", "Wall time / round"],
        ["S9.1 (n=5)",   "50",  "~5 s"],
        ["S9.2 (n=10)",  "100", "~10 s"],
        ["S9.3 (n=20)",  "200", "~20 s"],
        ["S9.4 (n=40)",  "400", "~40 s"],
        ["S11.* (n=20)", "200", "~20 s"],
    ],
    col_widths=[1.2 * inch, 1.8 * inch, 1.8 * inch],
))
gap()
body("5 rounds evaluated per scenario &times; 7 scenarios &asymp; 6 added "
     "minutes of compute across the entire 72-task Phase 2 array &mdash; "
     "negligible next to the GradInversion / GI-NAS / GGCDM attack budget.")

subsection("4.4 Determinism and reproducibility")
body("The CIFAR-10 split is loaded deterministically via "
     "<tt>torchvision.datasets.CIFAR10(train=True)</tt>. Within GLMIP, "
     "class-balanced subsets are drawn via "
     "<tt>random.sample</tt>, which is <i>not</i> seeded in Phase 2 to "
     "avoid coupling across scenarios. Consequence: running the same "
     "scenario twice can produce slightly different per-round GLMIP "
     "scores at small <i>n</i>. For stable reporting, either (a) fix a "
     "seed at the start of <tt>_recompute_label_metrics</tt>, or (b) "
     "average Phase 2 runs across multiple seeds &mdash; the latter is "
     "cheaper and matches how the rest of the study is reported.")

# ── 5. Citations to include ────────────────────────────────────────
section("5. Writing notes for the paper")
subsection("5.1 Key citations")
bullet("<b>iDLG</b> &mdash; Zhao, B., Mopuri, K. R., Bilen, H. "
       "<i>iDLG: Improved Deep Leakage from Gradients</i>. ICLR 2021. "
       "(Final-layer label leakage theorem.)")
bullet("<b>MaskCrypt</b> &mdash; Hu, C., Li, B. <i>MaskCrypt: Federated "
       "Learning with Selective Homomorphic Encryption</i>. IEEE TDSC "
       "2025. (Gradient-guided mask selection.)")
bullet("<b>Deep Leakage from Gradients</b> &mdash; Zhu, L., Liu, Z., "
       "Han, S. NeurIPS 2019. (Foundational gradient-inversion paper.)")
bullet("<b>LeakScore / AdaGuard</b> &mdash; internal framework, this "
       "work. (GLMIP, entropy, empirical components.)")

subsection("5.2 Phrasing for the methodology section")
body("Suggested paper phrasing (verbatim-usable):")
code("&ldquo;To verify that our default GLMIP sampling density is "
     "sufficient, we sweep <i>n</i> &isin; {5, 10, 20, 40} samples per class "
     "(S9). To justify our use of the final classifier layer as the "
     "focus point for label-leakage measurement, we compare against "
     "a whole-gradient and a penultimate-conv variant on ResNet-18 "
     "(S11). In both sweeps, Phase 2 replays the original GLMIP "
     "pipeline &mdash; class-balanced sampling from the CIFAR-10 train "
     "split, forward-and-backward through the round-<i>r</i> global "
     "model &mdash; because slim per-client artifacts do not store the "
     "intermediate per-sample gradients that the sweeps modify.&rdquo;")

subsection("5.3 Caveats to disclose")
bullet("S9 samples are class-balanced from the <i>public</i> CIFAR-10 "
       "split, not from the non-IID client partitions. The metric it "
       "reports is therefore a <i>server-side leak estimate</i>, not a "
       "client-side observation. This matches Phase 1's original design.")
bullet("S11 results apply specifically to ResNet-18. The ordering "
       "between final-FC and penultimate-conv may differ for deeper "
       "models (e.g. ResNet-50's three-layer Bottleneck blocks).")
bullet("S11's penultimate layer is the last 3&times;3 conv in layer4 "
       "(<tt>layer4.1.conv2.weight</tt>, 2.36 M params). Alternative "
       "penultimate choices &mdash; the batchnorm just before "
       "<tt>fc</tt>, or the full <tt>layer4.1</tt> block &mdash; would "
       "give different numbers; we picked the most-weights-per-"
       "parameter single layer to match the classifier's single-layer "
       "nature.")

# Build
doc.build(story)
print(f"PDF generated: {OUTPUT}")
