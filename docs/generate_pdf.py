"""Generate LeakScore Formulas PDF using reportlab."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

OUTPUT = "docs/leakscore_formulas.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    topMargin=0.8 * inch,
    bottomMargin=0.8 * inch,
    leftMargin=1.0 * inch,
    rightMargin=1.0 * inch,
)

styles = getSampleStyleSheet()

# Custom styles
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
    borderWidth=0, borderPadding=0,
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
    backColor=HexColor("#f5f6fa"), borderWidth=1, borderColor=HexColor("#dcdde1"),
    borderPadding=8,
))
styles.add(ParagraphStyle(
    "BulletItem", parent=styles["Normal"], fontSize=10, leading=14,
    leftIndent=20, bulletIndent=10, spaceAfter=3,
    textColor=HexColor("#2d3436"),
))

story = []
S = styles


def title(text):
    story.append(Paragraph(text, S["DocTitle"]))


def subtitle(text):
    story.append(Paragraph(text, S["DocSubtitle"]))


def section(text):
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#dcdde1"),
                            spaceBefore=12, spaceAfter=4))
    story.append(Paragraph(text, S["SectionHead"]))


def subsection(text):
    story.append(Paragraph(text, S["SubHead"]))


def body(text):
    story.append(Paragraph(text, S["Body"]))


def formula(text):
    story.append(Paragraph(text, S["Formula"]))


def bullet(text):
    story.append(Paragraph(f"\u2022  {text}", S["BulletItem"]))


def gap(h=6):
    story.append(Spacer(1, h))


# ── Document content ──────────────────────────────────────────────

title("AdaGuard LeakScore Framework")
subtitle("Mathematical Formulas Reference")
gap(12)

# Overall
section("Overall LeakScore")
body("The final LeakScore is a weighted average of three component scores:")
formula("LeakScore = (\u03b1 \u00b7 Entropy + \u03b2 \u00b7 Label + \u03b3 \u00b7 Empirical) / (\u03b1 + \u03b2 + \u03b3)")
bullet("Default weights: \u03b1 = \u03b2 = \u03b3 = 1.0")
bullet("All components and the final score are normalized to [0, 1]")
bullet("A higher LeakScore indicates greater gradient leakage risk")

# 1. Entropy
section("1. Entropy LeakScore")
body("Measures how concentrated the gradient distribution is. "
     "Concentrated gradients are easier to invert, indicating higher privacy risk.")
subsection("Preprocessing Pipeline")
bullet("Flatten gradient tensor into a 1-D vector")
bullet("L2-normalize: g<sub>norm</sub> = g / ||g||<sub>2</sub>")
bullet("Standardize: g<sub>std</sub> = (g<sub>norm</sub> \u2212 \u03bc) / \u03c3")
bullet("Bin into a histogram with B = 50 bins to obtain probability distribution p")

subsection("Sub-metrics")
# Table for entropy metrics
t_data = [
    ["Metric", "Entropy Formula", "Leakage Score"],
    ["Shannon", "H = \u2212\u03a3 p \u00b7 log(p)", "1 \u2212 H / log(B)"],
    ["R\u00e9nyi (\u03b1 = 2)", "H = \u2212log(\u03a3 p\u00b2)", "1 \u2212 H / log(B)"],
    ["Min-Entropy", "H = \u2212log(max(p))", "1 \u2212 H / log(B)"],
]
t_data_p = [[Paragraph(c, S["Body"]) for c in row] for row in t_data]
t = Table(t_data_p, colWidths=[1.4 * inch, 2.6 * inch, 1.8 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dcdde1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(t)
gap()
body("A uniform distribution yields entropy = log(B), giving score 0 (low risk). "
     "A highly peaked distribution yields low entropy, giving score near 1 (high risk).")

# 2. Label
section("2. Label LeakScore")
body("Measures how much class-label information is embedded in the gradients. "
     "Three sub-metrics capture different aspects of label leakage.")

subsection("2a. GLMIP (Gradient-Label Mutual Information Proxy)")
body("Computes the ratio of between-class to total gradient variance, "
     "analogous to Fisher's linear discriminant:")
formula("S<sub>B</sub> = \u03a3<sub>c</sub> |C<sub>c</sub>| \u00b7 ||\u03bc<sub>c</sub> \u2212 \u03bc||<super>2</super>"
        "  &nbsp;&nbsp;&nbsp;  (between-class scatter)")
formula("S<sub>W</sub> = \u03a3<sub>c</sub> \u03a3<sub>g \u2208 C<sub>c</sub></sub>"
        " ||g \u2212 \u03bc<sub>c</sub>||<super>2</super>"
        "  &nbsp;&nbsp;&nbsp;  (within-class scatter)")
formula("Score = S<sub>B</sub> / (S<sub>B</sub> + S<sub>W</sub>)")
bullet("\u03bc<sub>c</sub> = mean gradient for class c;&nbsp;&nbsp; \u03bc = global mean gradient")
bullet("Score near 1 = gradients are highly class-separable (high label leakage)")

subsection("2b. Confidence Gap")
body("Measures prediction certainty from the model's softmax output:")
formula("Score = max(softmax) \u2212 second_max(softmax)")
body("A large gap indicates high model confidence, which correlates with "
     "label-revealing gradient structure.")

subsection("2c. Cosine Similarity")
body("Measures how distinguishable gradients are across classes:")
formula("m = mean of all pairwise cosine similarities between class gradient means")
formula("Score = 1 \u2212 (m + 1) / 2")
body("Low inter-class similarity maps to a high score, indicating an attacker "
     "can separate class signals from the gradient.")

# 3. Empirical
section("3. Empirical LeakScore")
body("Directly measures leakage by attempting to reconstruct training data from gradients. "
     "Three gradient inversion attacks are run, each optimizing a dummy input to match "
     "the observed gradient.")
subsection("Common Scoring Formula")
formula("score = max(0,&nbsp; 1 \u2212 ||g<sub>real</sub> \u2212 g<sub>recon</sub>||<sub>2</sub>"
        " / ||g<sub>real</sub>||<sub>2</sub>)")
body("A score near 1 means the attack nearly perfectly matched the real gradient, "
     "implying successful data reconstruction.")

subsection("3a. GradInversion")
formula("loss = (1 \u2212 cos_sim(g<sub>recon</sub>, g<sub>real</sub>))"
        " + \u03bb<sub>tv</sub> \u00b7 TV(x)")
body("TV(x) is total variation regularization, encouraging smooth reconstructions.")

subsection("3b. GI-NAS")
formula("loss = ||g<sub>recon</sub> \u2212 g<sub>real</sub>||<sub>2</sub><super>2</super>"
        " + \u03bb<sub>gl</sub> \u00b7 Var(x.mean(dim=(2,3)))")
body("Uses multi-restart optimization (default 3 restarts) and keeps the best result.")

subsection("3c. GGCDM")
formula("loss = (1 \u2212 cos_sim(g<sub>recon</sub>, g<sub>real</sub>))"
        " + 0.01 \u00b7 ||g<sub>recon</sub> \u2212 g<sub>real</sub>||<sub>2</sub><super>2</super>")
body("Injects annealed noise: x += (1 \u2212 step/n_iter) \u00b7 0.01 \u00b7 noise, "
     "which decays to zero over the optimization.")
gap()
formula("Empirical LeakScore = mean(GradInversion, GI-NAS, GGCDM)")

# 4. Fisher
section("4. Fisher Information")
body("Identifies which model weights carry the most information about training data. "
     "Used to decide which weights to encrypt via selective homomorphic encryption.")

subsection("Per-weight Fisher (Empirical Approximation)")
formula("F<sub>i</sub> = g<sub>i</sub><super>2</super>")

subsection("Average Fisher (Per Parameter)")
formula("f<sub>round</sub> = \u03a3 F<sub>i</sub> / n &nbsp;&nbsp;&nbsp; where n = total number of parameters")

subsection("Normalized Fisher (Sigmoid Mapping)")
formula("f<sub>norm</sub> = f<sub>round</sub> / (f<sub>round</sub> + 1)")

subsection("Fisher Concentration (Gini Coefficient)")
formula("\u0046\u0303<sub>i</sub> = F<sub>i</sub> / \u03a3 F<sub>j</sub>"
        " &nbsp;&nbsp;&nbsp; (normalize to distribution)")
formula("Sort values ascending: s<sub>1</sub>, s<sub>2</sub>, ..., s<sub>n</sub>")
formula("Gini = (2 \u00b7 \u03a3(i \u00b7 s<sub>i</sub>))"
        " / (n \u00b7 \u03a3 s<sub>i</sub>) \u2212 (n + 1) / n")
body("Gini near 1 means Fisher information is concentrated in a small fraction "
     "of weights, making selective encryption efficient.")
gap()
body("<b>Encryption mask:</b> The top enc_pct% of weights ranked by F<sub>i</sub> "
     "are selected for homomorphic encryption.")

# 5. MaskCrypt
section("5. MaskCrypt Vulnerability")
body("From the MaskCrypt paper (Hu &amp; Li, 2025). Measures per-weight vulnerability "
     "by combining gradient magnitude with the change in exposed model weights across rounds.")

subsection("Per-weight Vulnerability")
formula("v[i] = g[i] \u00b7 (w<sub>exposed</sub>[i] \u2212 w<sub>trained</sub>[i])")
bullet("w<sub>exposed</sub> = global weights exposed to the network from the previous round")
bullet("w<sub>trained</sub> = locally trained weights in the current round")

subsection("Vulnerability Score (Top-fraction Concentration)")
formula("Score = \u03a3(top_k |v|) / \u03a3(all |v|)")
body("where top_k selects the enc_pct% of weights with the largest |v|. "
     "A high score means vulnerability is concentrated in a few weights.")

subsection("L2 Score (Dispersion Measure)")
formula("Score = ||v||<sub>2</sub> / (\u221an \u00b7 max|v|)")
gap()
body("<b>Encryption mask:</b> Weights where |v| exceeds the top enc_pct% threshold "
     "are selected for encryption.")

# 6. Magnitude
section("6. Gradient Magnitude")
body("A fast metric capturing the overall scale of gradients. "
     "Large gradients carry more information and are more susceptible to inversion.")

subsection("Per-layer L2 Norm")
formula("per_layer[name] = ||g<sub>name</sub>||<sub>2</sub>")

subsection("Total L2 Norm")
formula("L2 = \u221a(\u03a3 per_layer_norm<super>2</super>)")

subsection("Magnitude Score (Smooth Normalization)")
formula("Score = min(1,&nbsp; L2 / (L2 + 1))")
body("Maps [0, \u221e) smoothly onto [0, 1). A score near 1 indicates very large "
     "gradients with high information content.")

# Build
doc.build(story)
print(f"PDF generated: {OUTPUT}")
