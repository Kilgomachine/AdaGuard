# AdaGuard LeakScore Framework — Formulas Reference

## Overall LeakScore

The final LeakScore is a weighted average of three component scores:

```
LeakScore = (alpha * Entropy + beta * Label + gamma * Empirical) / (alpha + beta + gamma)
```

- Default weights: alpha = beta = gamma = 1.0
- All components and the final score are normalized to [0, 1]
- A higher LeakScore indicates greater gradient leakage risk

---

## 1. Entropy LeakScore

Measures how concentrated the gradient distribution is. Concentrated gradients are easier to invert, meaning higher privacy risk.

**Preprocessing pipeline:**
1. Flatten gradient tensor into a 1-D vector
2. L2-normalize: g_norm = g / ||g||_2
3. Standardize: g_std = (g_norm - mean) / std
4. Bin into a histogram with B = 50 bins to obtain a probability distribution p

**Sub-metrics:**

| Metric | Entropy Formula | Leakage Score |
|---|---|---|
| Shannon | H_sh = -sum(p * log(p)) | 1 - H_sh / log(B) |
| Renyi (alpha = 2) | H_re = -log(sum(p^2)) | 1 - H_re / log(B) |
| Min-Entropy | H_min = -log(max(p)) | 1 - H_min / log(B) |

All scores are clamped to [0, 1]. A uniform distribution yields entropy equal to log(B), giving a score of 0 (low risk). A highly peaked distribution yields low entropy, giving a score near 1 (high risk).

---

## 2. Label LeakScore

Measures how much class-label information is embedded in the gradients. Three complementary sub-metrics capture different aspects of label leakage.

### 2a. GLMIP (Gradient-Label Mutual Information Proxy)

Computes the ratio of between-class to total gradient variance, analogous to Fisher's linear discriminant:

```
S_B = sum_c  |C_c| * ||mu_c - mu||^2       (between-class scatter)
S_W = sum_c  sum_{g in C_c} ||g - mu_c||^2  (within-class scatter)

Score = S_B / (S_B + S_W)
```

- mu_c = mean gradient for class c
- mu = global mean gradient
- A score near 1 means gradients are highly class-separable (high label leakage)

### 2b. Confidence Gap

Measures prediction certainty from the model's softmax output:

```
Score = max(softmax) - second_max(softmax)
```

A large gap means the model is very confident, which correlates with label-revealing gradient structure.

### 2c. Cosine Similarity

Measures how distinguishable gradients are across classes:

```
m = mean of all pairwise cosine similarities between class gradient means

Score = 1 - (m + 1) / 2
```

Low inter-class similarity (dissimilar gradients per class) maps to a high score, indicating that an attacker can separate class signals from the gradient.

---

## 3. Empirical LeakScore (Gradient Inversion Attacks)

Directly measures leakage by attempting to reconstruct training data from gradients. Three attack algorithms are run, each optimizing a dummy input to match the observed gradient.

**Common scoring formula (all three attacks):**

```
score = max(0,  1 - ||g_real - g_reconstructed||_2 / ||g_real||_2)
```

A score near 1 means the attack nearly perfectly matched the real gradient, implying successful reconstruction.

### 3a. GradInversion

```
loss = (1 - cosine_similarity(g_recon, g_real)) + lambda_tv * TV(x)
```

TV(x) is total variation regularization, encouraging smooth reconstructions.

### 3b. GI-NAS

```
loss = ||g_recon - g_real||_2^2 + lambda_gl * Var(x.mean(dim=(2,3)))
```

Uses multi-restart optimization (default 3 restarts) and keeps the best result.

### 3c. GGCDM

```
loss = (1 - cosine_similarity(g_recon, g_real)) + 0.01 * ||g_recon - g_real||_2^2
```

Injects annealed noise during optimization: x += (1 - step/n_iter) * 0.01 * noise, which decays to zero over the course of optimization.

**Empirical LeakScore = mean(GradInversion, GI-NAS, GGCDM)**

---

## 4. Fisher Information

Identifies which model weights carry the most information about the training data, used to decide which weights to encrypt.

**Per-weight Fisher information (empirical approximation):**

```
F_i = g_i^2
```

**Average Fisher (per parameter):**

```
f_round = sum(F_i) / n       where n = total number of parameters
```

**Normalized Fisher (sigmoid mapping to [0, 1]):**

```
f_round_norm = f_round / (f_round + 1)
```

**Fisher concentration (Gini coefficient):**

```
F_tilde_i = F_i / sum(F_j)                   (normalize to distribution)
Sort F_tilde values in ascending order: s_1, s_2, ..., s_n

Gini = (2 * sum(i * s_i)) / (n * sum(s_i)) - (n + 1) / n
```

A Gini coefficient near 1 means Fisher information is concentrated in a small fraction of weights, making selective encryption efficient.

**Encryption mask:** The top enc_pct% of weights ranked by F_i are selected for homomorphic encryption.

---

## 5. MaskCrypt Vulnerability

From the MaskCrypt paper (Hu & Li, 2025). Measures per-weight vulnerability by combining gradient magnitude with the change in exposed model weights across rounds.

**Per-weight vulnerability:**

```
v[i] = g[i] * (w_exposed[i] - w_trained[i])
```

- w_exposed = global model weights exposed to the network from the previous round
- w_trained = locally trained model weights in the current round

**Vulnerability score (top-fraction concentration):**

```
Score = sum(top_k |v|) / sum(all |v|)
```

where top_k selects the enc_pct% of weights with the largest |v|. A high score means vulnerability is concentrated in a few weights.

**L2 score (dispersion measure):**

```
Score = ||v||_2 / (sqrt(n) * max(|v|))
```

**Encryption mask:** Weights where |v| exceeds the top enc_pct% threshold are selected for encryption.

---

## 6. Gradient Magnitude

A simple, fast metric that captures the overall scale of gradients. Large gradients carry more information and are more susceptible to inversion.

**Per-layer L2 norm:**

```
per_layer[name] = ||g_name||_2
```

**Total L2 norm:**

```
L2 = sqrt(sum(per_layer_norm^2))
```

**Magnitude score (smooth normalization):**

```
Score = min(1, L2 / (L2 + 1))
```

This maps [0, infinity) smoothly onto [0, 1). A score near 1 indicates very large gradients with high information content.
