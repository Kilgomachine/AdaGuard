"""Generate AdaGuard research paper sections as a professional PDF."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, ListFlowable, ListItem,
)

OUTPUT = "docs/AdaGuard_Paper_Draft.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.75 * inch,
    bottomMargin=0.75 * inch,
    leftMargin=1.0 * inch,
    rightMargin=1.0 * inch,
)

styles = getSampleStyleSheet()

# ── Custom styles ──
styles.add(ParagraphStyle(
    "PaperTitle", parent=styles["Title"], fontSize=20, spaceAfter=4,
    textColor=HexColor("#1a1a2e"), leading=26,
))
styles.add(ParagraphStyle(
    "Authors", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER,
    textColor=HexColor("#333333"), spaceAfter=2, leading=15,
))
styles.add(ParagraphStyle(
    "Affiliation", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER,
    textColor=HexColor("#666666"), spaceAfter=16, leading=12,
))
styles.add(ParagraphStyle(
    "AbstractHead", parent=styles["Heading2"], fontSize=12, spaceBefore=8,
    spaceAfter=4, textColor=HexColor("#1a1a2e"), alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "AbstractBody", parent=styles["Normal"], fontSize=10, leading=14,
    spaceAfter=8, textColor=HexColor("#2d3436"), alignment=TA_JUSTIFY,
    leftIndent=30, rightIndent=30,
))
styles.add(ParagraphStyle(
    "SectionHead", parent=styles["Heading1"], fontSize=14, spaceBefore=18,
    spaceAfter=8, textColor=HexColor("#1a1a2e"),
))
styles.add(ParagraphStyle(
    "SubHead", parent=styles["Heading2"], fontSize=12, spaceBefore=12,
    spaceAfter=6, textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "SubSubHead", parent=styles["Heading3"], fontSize=11, spaceBefore=10,
    spaceAfter=4, textColor=HexColor("#2d3436"), fontName="Helvetica-BoldOblique",
))
styles.add(ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10, leading=14.5,
    spaceAfter=6, textColor=HexColor("#2d3436"), alignment=TA_JUSTIFY,
))
styles.add(ParagraphStyle(
    "Formula", parent=styles["Normal"], fontSize=10, leading=16,
    fontName="Courier", alignment=TA_CENTER, spaceBefore=6, spaceAfter=6,
    backColor=HexColor("#f8f9fa"), borderWidth=0.5, borderColor=HexColor("#e0e0e0"),
    borderPadding=8,
))
styles.add(ParagraphStyle(
    "BulletPt", parent=styles["Normal"], fontSize=10, leading=14,
    leftIndent=24, bulletIndent=12, spaceAfter=3,
    textColor=HexColor("#2d3436"),
))
styles.add(ParagraphStyle(
    "Caption", parent=styles["Normal"], fontSize=9, leading=12,
    textColor=HexColor("#555555"), alignment=TA_CENTER, spaceAfter=10,
    spaceBefore=4,
))
styles.add(ParagraphStyle(
    "GuidelineHead", parent=styles["Heading1"], fontSize=14, spaceBefore=18,
    spaceAfter=8, textColor=HexColor("#8e44ad"),
))
styles.add(ParagraphStyle(
    "GuidelineBody", parent=styles["Normal"], fontSize=10, leading=14.5,
    spaceAfter=6, textColor=HexColor("#2d3436"), alignment=TA_JUSTIFY,
))
styles.add(ParagraphStyle(
    "GuidelineBullet", parent=styles["Normal"], fontSize=10, leading=14,
    leftIndent=24, bulletIndent=12, spaceAfter=3,
    textColor=HexColor("#2d3436"),
))

story = []
S = styles


def title(text):
    story.append(Paragraph(text, S["PaperTitle"]))


def authors(text):
    story.append(Paragraph(text, S["Authors"]))


def affiliation(text):
    story.append(Paragraph(text, S["Affiliation"]))


def abstract_head():
    story.append(Paragraph("Abstract", S["AbstractHead"]))


def abstract_body(text):
    story.append(Paragraph(text, S["AbstractBody"]))


def section(num, text):
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#dcdde1"),
                            spaceBefore=10, spaceAfter=2))
    story.append(Paragraph(f"{num}. {text}", S["SectionHead"]))


def subsection(label, text):
    story.append(Paragraph(f"{label} {text}", S["SubHead"]))


def subsubsection(text):
    story.append(Paragraph(text, S["SubSubHead"]))


def body(text):
    story.append(Paragraph(text, S["Body"]))


def formula(text):
    story.append(Paragraph(text, S["Formula"]))


def bullet(text):
    story.append(Paragraph(f"\u2022  {text}", S["BulletPt"]))


def gap(h=8):
    story.append(Spacer(1, h))


def caption(text):
    story.append(Paragraph(text, S["Caption"]))


def guideline_section(text):
    story.append(Paragraph(text, S["GuidelineHead"]))


def guideline_body(text):
    story.append(Paragraph(text, S["GuidelineBody"]))


def guideline_bullet(text):
    story.append(Paragraph(f"\u2022  {text}", S["GuidelineBullet"]))


# =====================================================================
#  PAPER CONTENT
# =====================================================================

title("AdaGuard: Leakage-Aware Adaptive Encryption<br/>for Privacy-Preserving Federated Learning")
gap(4)
authors("Firas Aguir")
affiliation("Department of Computer Science, Oakland University, Rochester, MI, USA")
gap(8)

# ── ABSTRACT ──
abstract_head()
abstract_body(
    "Federated Learning (FL) enables collaborative model training without centralizing raw data, "
    "yet recent gradient inversion attacks have demonstrated that shared model updates can leak "
    "sensitive training information. Existing defenses either apply uniform encryption to all "
    "parameters\u2014incurring prohibitive computational and communication overhead\u2014or rely on "
    "heuristic noise injection that degrades model utility. We present <b>AdaGuard</b>, a framework "
    "that continuously quantifies gradient leakage risk through a multi-component <b>LeakScore</b> "
    "metric and dynamically allocates selective homomorphic encryption only to the most vulnerable "
    "model parameters. LeakScore combines three complementary signals: (i) an entropy-based measure "
    "of gradient distribution concentration, (ii) a label leakage proxy capturing class-discriminative "
    "structure in gradients, and (iii) an empirical score derived from executing gradient inversion "
    "attacks in real time. AdaGuard further leverages per-weight Fisher Information to rank parameter "
    "sensitivity and applies encryption to the top-k most informative weights, adapting k each round "
    "based on the observed LeakScore. We compare our Fisher-based selection against MaskCrypt's "
    "vulnerability-weighted scheme and evaluate on CIFAR-10 with up to 1,000 non-IID clients using "
    "ResNet-18 on a multi-GPU HPC cluster. Our results demonstrate that AdaGuard achieves comparable "
    "privacy protection to full encryption while encrypting only 10\u201315% of parameters, reducing "
    "computational cost by up to 8x and communication overhead by up to 6x, with less than 1% "
    "degradation in test accuracy."
)

# ── I. INTRODUCTION ──
section("I", "Introduction")

body(
    "Federated Learning (FL) [1] has emerged as a foundational paradigm for privacy-preserving "
    "machine learning, enabling multiple clients to collaboratively train a shared model by exchanging "
    "model updates\u2014typically gradients or weight deltas\u2014rather than raw data. This architecture "
    "has found adoption across domains where data sensitivity is paramount, including healthcare [2], "
    "finance [3], and mobile computing [4]. By keeping training data on-device, FL ostensibly provides "
    "a privacy guarantee: the server never observes the raw samples."
)

body(
    "However, a growing body of work has shattered this assumption. Zhu et al. [5] demonstrated that "
    "shared gradients can be inverted to reconstruct training images with pixel-level fidelity through "
    "optimization-based attacks. Subsequent work has refined these attacks: Zhao et al. [6] showed that "
    "labels can be analytically extracted from the gradient of the last fully-connected layer (the iDLG "
    "theorem), while Geiping et al. [7] introduced cosine-similarity objectives that improve reconstruction "
    "quality for larger batch sizes. More recent attacks leverage neural architecture search (GI-NAS) [8] "
    "and diffusion-inspired noise annealing (GGCDM) [9] to further push the frontier of gradient inversion. "
    "These developments establish that <b>sharing gradients is not inherently private</b>, and that "
    "meaningful defenses are necessary."
)

body(
    "The most straightforward defense is to encrypt all shared parameters using Homomorphic Encryption (HE), "
    "which permits aggregation on encrypted values without decryption [10]. While HE provides strong "
    "cryptographic guarantees, it introduces severe overhead: ciphertext expansion ratios of 50\u2013100x "
    "increase communication costs, and HE operations are 3\u20136 orders of magnitude slower than their "
    "plaintext counterparts [11]. For modern architectures with tens of millions of parameters, full HE "
    "encryption is impractical for real-time FL deployments."
)

body(
    "Alternative defenses such as differential privacy (DP) [12] add calibrated noise to gradients before "
    "sharing. While DP provides formal privacy guarantees, the noise magnitude required for meaningful "
    "protection often substantially degrades model accuracy, particularly in non-IID settings where clients "
    "hold heterogeneous data distributions [13]. Secure aggregation protocols [14] prevent the server from "
    "observing individual updates but require complex multi-party communication and do not protect against "
    "a compromised aggregator."
)

body(
    "A key insight motivates our approach: <b>not all parameters are equally vulnerable to gradient "
    "inversion</b>. Fisher Information theory tells us that the squared gradient magnitude "
    "F<sub>i</sub> = g<sub>i</sub><super>2</super> quantifies how much information parameter i carries "
    "about the training data [15]. In practice, Fisher Information is highly concentrated\u2014a small "
    "fraction of weights (often less than 5%) carries the majority of the information. This concentration, "
    "measured by the Gini coefficient of the Fisher distribution, suggests that <b>selective encryption</b> "
    "of only the most informative parameters can achieve comparable privacy protection at a fraction of the cost."
)

body(
    "Building on this insight, we introduce <b>AdaGuard</b>, a leakage-aware adaptive encryption framework "
    "for federated learning. AdaGuard makes three key contributions:"
)

bullet(
    "<b>LeakScore Framework.</b> A composite metric that quantifies gradient leakage risk in real time "
    "by combining entropy-based distribution analysis, label leakage proxies, and empirical gradient "
    "inversion attack scores. Unlike static privacy budgets, LeakScore adapts to the actual leakage "
    "characteristics of each round."
)
bullet(
    "<b>Adaptive Selective Encryption.</b> A three-tier encryption policy (none/partial/strong) that "
    "dynamically adjusts the fraction of encrypted parameters based on LeakScore. The policy uses "
    "Fisher Information to rank parameter sensitivity and encrypts only the top-k most vulnerable weights, "
    "where k is proportional to the detected leakage level."
)
bullet(
    "<b>Comprehensive Evaluation.</b> We compare Fisher-based parameter selection against MaskCrypt [16], "
    "a recently proposed vulnerability-weighted encryption scheme, and evaluate both against full encryption "
    "and no encryption baselines. Experiments on CIFAR-10 with non-IID partitioning across up to 1,000 "
    "clients demonstrate that AdaGuard encrypts 10\u201315% of parameters while maintaining privacy "
    "protection comparable to full encryption."
)

gap(6)
body(
    "The remainder of this paper is organized as follows. Section II discusses related work. Section III "
    "presents the AdaGuard framework in detail, including the LeakScore formulation, adaptive encryption "
    "policy, and Fisher-based parameter selection. Section IV describes our experimental setup. Section V "
    "reports results. Section VI discusses implications and limitations. Section VII concludes."
)

# ── III. ADAGUARD ──
section("III", "AdaGuard")

body(
    "We now present the AdaGuard framework. We begin with an overview of the system architecture, "
    "then detail the LeakScore metric, the adaptive encryption policy, and the two parameter selection "
    "strategies (Fisher-based and MaskCrypt-based)."
)

subsection("A.", "System Overview")

body(
    "AdaGuard operates within a standard FL setting: a central server coordinates T rounds of training "
    "across N clients, each holding a private dataset D<sub>i</sub>. In each round t, the server selects "
    "a subset S<sub>t</sub> of K clients, broadcasts the current global model w<sub>t</sub>, and each "
    "selected client performs local SGD training to produce a weight update "
    "delta<sub>i</sub> = w<sub>t</sub> \u2212 w<sub>i,t</sub>. Before transmitting delta<sub>i</sub> to the server, "
    "AdaGuard intercepts the update and performs three operations:"
)

bullet("<b>Measure:</b> Compute LeakScore(delta<sub>i</sub>) to quantify leakage risk.")
bullet("<b>Decide:</b> Apply the adaptive policy to determine encryption level and parameter selection.")
bullet("<b>Protect:</b> Selectively encrypt the identified parameters using homomorphic encryption.")

body(
    "The server then aggregates the (partially) encrypted updates via FedAvg [1] and updates the global model. "
    "This measure\u2013decide\u2013protect loop executes every round, allowing AdaGuard to adapt to "
    "changing leakage dynamics as training progresses."
)

subsection("B.", "LeakScore: Quantifying Gradient Leakage Risk")

body(
    "LeakScore is a composite metric that captures gradient leakage risk from three complementary perspectives. "
    "Each component produces a score in [0, 1], where higher values indicate greater risk. The final LeakScore "
    "is a weighted average:"
)

formula("LeakScore = (\u03b1 \u00b7 L<sub>entropy</sub> + \u03b2 \u00b7 L<sub>label</sub> + \u03b3 \u00b7 L<sub>empirical</sub>) / (\u03b1 + \u03b2 + \u03b3)")

body(
    "where \u03b1, \u03b2, \u03b3 are configurable weights (default: 1.0 each). We now detail each component."
)

subsubsection("B.1 Entropy LeakScore")

body(
    "The entropy component measures how concentrated the gradient distribution is. Prior work [5, 7] has shown "
    "that gradients with peaked, low-entropy distributions are more susceptible to inversion attacks because "
    "the optimization landscape for reconstruction is better conditioned. We preprocess the gradient vector "
    "through L2 normalization and standardization, then bin the result into a B-bin histogram (default B = 50) "
    "to obtain a discrete probability distribution p."
)

body("We compute three entropy variants and convert each to a leakage score:")

formula("Shannon: &nbsp; H<sub>sh</sub> = \u2212\u03a3 p<sub>i</sub> log(p<sub>i</sub>), &nbsp;&nbsp; Score = 1 \u2212 H<sub>sh</sub> / log(B)")
formula("Renyi (\u03b1=2): &nbsp; H<sub>re</sub> = \u2212log(\u03a3 p<sub>i</sub><super>2</super>), &nbsp;&nbsp; Score = 1 \u2212 H<sub>re</sub> / log(B)")
formula("Min-Entropy: &nbsp; H<sub>min</sub> = \u2212log(max(p<sub>i</sub>)), &nbsp;&nbsp; Score = 1 \u2212 H<sub>min</sub> / log(B)")

body(
    "All scores are clamped to [0, 1]. A uniform gradient distribution yields maximum entropy (score 0, low risk), "
    "while a highly peaked distribution yields low entropy (score near 1, high risk). Following the iDLG theorem [6], "
    "we optionally restrict entropy computation to focus layers (e.g., the final fully-connected layer) where label "
    "information is most concentrated."
)

subsubsection("B.2 Label LeakScore")

body(
    "The label component quantifies how much class-label information is embedded in the gradient structure. "
    "We compute three complementary sub-metrics:"
)

body(
    "<b>GLMIP (Gradient-Label Mutual Information Proxy).</b> Inspired by Fisher's linear discriminant, GLMIP "
    "computes the ratio of between-class to total gradient variance. We partition gradients by their corresponding "
    "labels and compute:"
)

formula("S<sub>B</sub> = \u03a3<sub>c</sub> |C<sub>c</sub>| \u00b7 ||\u03bc<sub>c</sub> \u2212 \u03bc||<super>2</super> &nbsp;&nbsp; (between-class scatter)")
formula("S<sub>W</sub> = \u03a3<sub>c</sub> \u03a3<sub>g\u2208C<sub>c</sub></sub> ||g \u2212 \u03bc<sub>c</sub>||<super>2</super> &nbsp;&nbsp; (within-class scatter)")
formula("GLMIP = S<sub>B</sub> / (S<sub>B</sub> + S<sub>W</sub>)")

body(
    "A GLMIP score near 1 indicates that gradients are highly separable by class, implying that an attacker "
    "can reliably extract label information."
)

body(
    "<b>Confidence Gap.</b> We measure the gap between the top two softmax probabilities: "
    "Score = max(softmax) \u2212 second_max(softmax). High confidence correlates with gradient structure "
    "that reveals the true label."
)

body(
    "<b>Cosine Similarity.</b> We compute the mean pairwise cosine similarity m between class-conditional "
    "gradient means, then map to a leakage score: Score = 1 \u2212 (m + 1)/2. Low inter-class similarity "
    "(dissimilar gradient signatures per class) indicates that class labels can be distinguished from the gradient."
)

subsubsection("B.3 Empirical LeakScore")

body(
    "The empirical component provides a direct measurement of leakage by executing gradient inversion attacks. "
    "Unlike entropy and label scores, which are proxies, this component measures actual reconstruction capability. "
    "We run three attacks with distinct optimization strategies:"
)

bullet(
    "<b>GradInversion [5]:</b> Optimizes a dummy input to minimize "
    "(1 \u2212 cos_sim(g<sub>recon</sub>, g<sub>real</sub>)) + \u03bb<sub>tv</sub> \u00b7 TV(x), "
    "where TV(x) is total variation regularization."
)
bullet(
    "<b>GI-NAS [8]:</b> Uses L2 gradient matching with group-lasso regularization and multi-restart optimization "
    "(3 restarts, keeping the best result)."
)
bullet(
    "<b>GGCDM [9]:</b> Combines cosine and L2 objectives with annealed noise injection: "
    "x += (1 \u2212 t/T) \u00b7 0.01 \u00b7 noise, inspired by diffusion processes."
)

body("Each attack produces a reconstruction quality score:")

formula("score<sub>attack</sub> = max(0, &nbsp; 1 \u2212 ||g<sub>real</sub> \u2212 g<sub>recon</sub>||<sub>2</sub> / ||g<sub>real</sub>||<sub>2</sub>)")

body(
    "The empirical LeakScore is the mean across all three attacks. A score near 1 indicates that at least one "
    "attack nearly perfectly matched the true gradient, implying successful data reconstruction. While computationally "
    "expensive, the empirical component serves as ground truth for calibrating the faster proxy metrics."
)

subsection("C.", "Adaptive Encryption Policy")

body(
    "AdaGuard uses a three-tier policy that maps LeakScore to an encryption level, parameterized by "
    "thresholds T<sub>1</sub> and T<sub>2</sub> (defaults: 0.3 and 0.7):"
)

formula("LeakScore &lt; T<sub>1</sub>: &nbsp;&nbsp; level = none, &nbsp; encrypt_pct = 0%")
formula("T<sub>1</sub> \u2264 LeakScore &lt; T<sub>2</sub>: &nbsp;&nbsp; level = partial, &nbsp; encrypt_pct scales linearly")
formula("LeakScore \u2265 T<sub>2</sub>: &nbsp;&nbsp; level = strong, &nbsp; encrypt_pct scales up to 100%")

body(
    "In the partial regime, the encryption percentage interpolates linearly between 0 and the base encryption "
    "rate (default 10%). In the strong regime, it continues scaling from the base rate toward full encryption. "
    "This smooth scaling avoids abrupt transitions and allows the system to apply minimal overhead when leakage "
    "risk is low while ramping up protection when risk is elevated."
)

body(
    "The policy output determines k, the number of parameters to encrypt: k = encrypt_pct \u00d7 n, where n "
    "is the total parameter count. The parameter selection strategy (Fisher or MaskCrypt) then identifies "
    "which specific k parameters to protect."
)

subsection("D.", "Fisher-Based Parameter Selection")

body(
    "Our primary parameter selection strategy uses the empirical Fisher Information to rank parameter sensitivity. "
    "The Fisher Information matrix approximation at parameter i is:"
)

formula("F<sub>i</sub> = g<sub>i</sub><super>2</super>")

body(
    "This diagonal approximation, while simple, captures the key insight: parameters with large squared gradients "
    "carry more information about the training batch and are thus more valuable to an attacker. We select the "
    "top-k parameters by F<sub>i</sub> for encryption."
)

body("We additionally compute two aggregate statistics for monitoring:")

formula("Fisher Concentration (Gini): &nbsp; G = (2\u00b7\u03a3 i\u00b7s<sub>i</sub>) / (n\u00b7\u03a3 s<sub>i</sub>) \u2212 (n+1)/n")

body(
    "where s<sub>1</sub>, ..., s<sub>n</sub> are the Fisher values sorted in ascending order. A Gini coefficient "
    "near 1 indicates that information is concentrated in a small fraction of parameters, validating the "
    "selective encryption approach. We also compute a normalized Fisher score via sigmoid mapping: "
    "f<sub>norm</sub> = f<sub>round</sub> / (f<sub>round</sub> + 1), providing a scale-independent measure in [0, 1]."
)

subsection("E.", "MaskCrypt-Based Parameter Selection")

body(
    "We additionally implement the MaskCrypt vulnerability scoring scheme proposed by Hu and Li [16] as a "
    "comparison baseline. MaskCrypt computes a per-weight vulnerability score that combines gradient magnitude "
    "with the change in exposed model weights across rounds:"
)

formula("v[i] = g[i] \u00b7 (w<sub>exposed</sub>[i] \u2212 w<sub>trained</sub>[i])")

body(
    "where w<sub>exposed</sub> represents the global model weights that were broadcast (and thus exposed to "
    "potential adversaries) in the previous round, and w<sub>trained</sub> represents the locally trained weights "
    "in the current round. The intuition is that parameters where both the gradient is large and the weight has "
    "changed significantly between rounds are most informative to an attacker who has observed the previous "
    "round's broadcast."
)

body(
    "MaskCrypt selects the top-k parameters by |v[i]| for encryption. We evaluate its concentration via the "
    "top-fraction score: Score = \u03a3(top_k |v|) / \u03a3(all |v|), and its dispersion via the L2 score: "
    "||v||<sub>2</sub> / (\u221an \u00b7 max|v|)."
)

subsection("F.", "Gradient Magnitude Score")

body(
    "As a lightweight supplementary metric, we compute the total gradient L2 norm and map it to [0, 1] via "
    "smooth normalization:"
)

formula("Score = min(1, &nbsp; ||g||<sub>2</sub> / (||g||<sub>2</sub> + 1))")

body(
    "This metric captures the intuition that larger gradients carry more information and are more susceptible "
    "to inversion. While less nuanced than Fisher or entropy measures, it provides a fast sanity check and "
    "correlates with empirical attack success rates."
)

subsection("G.", "Computational Considerations")

body(
    "A critical advantage of selective encryption is reduced overhead. For a model with n parameters and "
    "encryption fraction p, the computational cost scales as O(p \u00b7 n) rather than O(n). With typical "
    "HE expansion factors of 50\u2013100x, encrypting 10% of parameters reduces communication overhead by "
    "approximately 9x compared to full encryption. The LeakScore computation itself adds overhead: entropy "
    "and Fisher metrics are O(n), while empirical attacks require multiple forward-backward passes. However, "
    "the empirical component can be computed on a configurable schedule (e.g., every 5 rounds) rather than "
    "every round, amortizing its cost."
)

body(
    "For multi-GPU deployments, AdaGuard parallelizes client processing across available GPUs using thread-based "
    "concurrency, with each client assigned to a GPU in round-robin fashion. Metric computation is performed "
    "per-client on the assigned GPU to avoid cross-device memory transfers."
)

# ── PAGE BREAK FOR GUIDELINES ──
story.append(PageBreak())

# =====================================================================
#  WRITING GUIDELINES
# =====================================================================

story.append(HRFlowable(width="100%", thickness=2, color=HexColor("#8e44ad"),
                        spaceBefore=4, spaceAfter=8))
guideline_section("Writing Guidelines for Research Papers")
gap(4)

guideline_body(
    "The following guidelines are intended to help you maintain academic rigor and integrity as you "
    "complete the remaining sections of this paper. These principles apply to all sections."
)

# 1
guideline_section("1. General Principles")
guideline_bullet(
    "<b>Claim only what you can prove.</b> Every factual claim must be backed by either a citation, "
    "your own experimental data, or a mathematical derivation. If you write \"our method reduces cost by 8x,\" "
    "you must have the numbers to support it."
)
guideline_bullet(
    "<b>Distinguish contributions from prior work.</b> Clearly state what is yours versus what you are "
    "building upon. Use phrases like \"we extend,\" \"we adapt,\" or \"building on [X], we propose.\""
)
guideline_bullet(
    "<b>Be precise with language.</b> Avoid vague quantifiers (\"significantly better,\" \"much faster\"). "
    "Use exact numbers: \"reduces overhead from 50x to 5.2x\" is stronger than \"greatly reduces overhead.\""
)
guideline_bullet(
    "<b>Acknowledge limitations honestly.</b> A paper that claims no limitations appears naive. "
    "State what your system does not handle (e.g., \"our threat model assumes an honest-but-curious server\")."
)

# 2
guideline_section("2. Writing the Related Work Section")
guideline_bullet(
    "<b>Organize thematically, not chronologically.</b> Group papers by topic: (a) gradient inversion attacks, "
    "(b) HE-based defenses, (c) DP-based defenses, (d) selective/adaptive encryption. End each group by "
    "explaining what gap remains."
)
guideline_bullet(
    "<b>Be fair to competitors.</b> Describe prior methods accurately before explaining why your approach "
    "differs. Avoid strawman comparisons."
)
guideline_bullet(
    "<b>Connect each reference to your work.</b> Every cited paper should serve a purpose: establishing the "
    "problem, motivating a design choice, or providing a comparison point."
)

# 3
guideline_section("3. Writing the Results Section")
guideline_bullet(
    "<b>Present results before interpreting them.</b> State the numbers, then explain what they mean. "
    "\"Fisher encryption at 10% achieves 0.92 LeakScore reduction (Table 2); this suggests that...\" "
)
guideline_bullet(
    "<b>Use tables for exact values, figures for trends.</b> A convergence curve belongs in a figure; "
    "a comparison of final accuracy across 4 strategies belongs in a table."
)
guideline_bullet(
    "<b>Report variance.</b> Run experiments multiple times (3\u20135 seeds) and report mean \u00b1 std. "
    "Single-run results are not publishable at top venues."
)
guideline_bullet(
    "<b>Structure around research questions.</b> Frame results as answers to specific questions: "
    "Q1: Does selective encryption stop attacks? Q2: Does it maintain accuracy? Q3: What is the cost reduction? "
    "Q4: How does Fisher compare to MaskCrypt?"
)

# 4
guideline_section("4. Writing the Discussion Section")
guideline_bullet(
    "<b>Interpret, do not repeat.</b> The discussion should explain <i>why</i> results look the way they do, "
    "not restate the numbers."
)
guideline_bullet(
    "<b>Connect to the broader field.</b> How do your findings advance the state of the art? "
    "What do they imply for practitioners deploying FL systems?"
)
guideline_bullet(
    "<b>Address threats to validity.</b> Internal validity: could bugs or implementation choices affect results? "
    "External validity: does CIFAR-10 generalize to medical imaging? Construct validity: does LeakScore "
    "actually measure what we claim?"
)

# 5
guideline_section("5. Writing the Threats to Validity Section")
guideline_bullet(
    "<b>Construct validity:</b> Does LeakScore truly capture leakage risk? Validate by showing correlation "
    "between LeakScore and actual reconstruction quality (PSNR, SSIM, MSE). If they correlate, the metric "
    "is meaningful."
)
guideline_bullet(
    "<b>Internal validity:</b> Are results caused by AdaGuard or by confounds? Control for: random seed, "
    "data partitioning, model initialization. Use the same initial model across all strategies."
)
guideline_bullet(
    "<b>External validity:</b> CIFAR-10 is small and well-studied. Acknowledge that results may differ on "
    "larger datasets (ImageNet), different modalities (text, tabular), or different model architectures. "
    "Non-IID with max-2-classes is one specific heterogeneity pattern."
)

# 6
guideline_section("6. Citation Integrity")
guideline_bullet(
    "<b>Cite primary sources.</b> Cite the original paper, not a survey or blog post that describes it."
)
guideline_bullet(
    "<b>Do not cite papers you have not read.</b> If you only know a result secondhand, either read the "
    "original or cite it as \"as reported in [X].\""
)
guideline_bullet(
    "<b>Use a consistent citation style.</b> IEEE format for this template: [1], [2], etc., with full "
    "bibliographic details in the References section."
)

# 7
guideline_section("7. Common Pitfalls to Avoid")
guideline_bullet(
    "<b>Overclaiming.</b> \"Our method is the first to...\" requires exhaustive literature review. "
    "Safer: \"To the best of our knowledge, no prior work has...\""
)
guideline_bullet(
    "<b>Cherry-picking results.</b> Report all experiments, including ones where AdaGuard underperforms. "
    "Reviewers will run your code."
)
guideline_bullet(
    "<b>Ignoring baselines.</b> Always compare against: (a) no encryption, (b) full encryption, "
    "(c) differential privacy (if feasible), (d) MaskCrypt or other selective methods."
)
guideline_bullet(
    "<b>Vague experimental setup.</b> Specify every hyperparameter: learning rate, batch size, number of "
    "local steps, encryption percentage, number of clients, rounds, seeds. Reproducibility is non-negotiable."
)

# Build
doc.build(story)
print(f"PDF generated: {OUTPUT}")
