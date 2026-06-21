# AdaGuard: Leakage, Aware Adaptive Selective Encryption for Federated Learning

Replication package for **"AdaGuard: Fisher-Information-Guided Selective Homomorphic Encryption against Gradient-Inversion Attacks in Federated Learning"**.

This repository contains the full source code, configuration files, HPC job scripts, and analysis utilities required to reproduce every result reported in the paper, including the headline 4×3 defence×attack matrix, the architecture-transfer study across ResNet-10 / ResNet-18 / ResNet-50, and the extended encryption-fraction sweep that establishes the ResNet-50 PSNR floor.


## Overview

AdaGuard is a Federated-Learning privacy defence that combines:

- **Empirical Fisher-Information ranking** of the per-parameter leakage surface, encrypting the top-$\rho$ fraction with CKKS-style homomorphic encryption while leaving the remainder in cleartext.
- A structural **classifier-head guarantee** that force-includes classifier-head tensors regardless of Fisher score, closing the iDLG / LLG label-recovery side channel.
- A multi-metric **LeakScore** scaffolding (entropy concentration, label identifiability, optional shadow-attack signals) for adaptive per-round response.

The headline evaluation is a $4\times3$ defence ($V_1$ undefended, $V_2$ FHE upper-bound, $V_3$ MaskCrypt, $V_4$ AdaGuard-Fisher) × attack (GradInversion, GGCDM, GI-NAS) matrix at multi-seed mean over seeds $\{42, 123, 456\}$.

### Headline result

| Defence | GradInversion (PSNR ↓) | GI-NAS (PSNR ↓) | GGCDM (PSNR ↓) | Label-recovery ASR ↓ |
|---|---|---|---|---|
| V1 (undefended FedAvg) | 14.46 dB | recognizable | varies | 1.000 |
| V3 (MaskCrypt) | blob band | blob band | varies | 0.000 |
| **V4 (AdaGuard-Fisher)** | **noise band** | **noise band** | **noise / blob** | **0.000** |

See `paper_artifacts/` and the per-cell JSON files released alongside this repository for the full multi-seed numbers.

### Architecture-transfer finding

ASR is architecture-portable (= 0 across ResNet-10, ResNet-18, ResNet-50). Pixel-channel PSNR exhibits an architecture-scaling effect concentrated on GradInversion: ResNet-50 at $\rho = 0.10$ sits in the blob band, but the **extended $\rho$-sweep on ResNet-50** shows the PSNR floor is bounded by non-gradient leakage (BatchNorm running statistics, label-conditional attack priors) rather than by the encryption budget: GradInversion PSNR spans only 0.84 dB across $\rho \in \{0.05, \ldots, 0.90\}$, entirely in the noise band.

## Repository Structure

```
.
├── adaguard/                  # Core library
│   ├── attacks/               # Attack-faithful proxies (GradInversion / GGCDM / GI-NAS)
│   ├── encryption/            # Fisher / MaskCrypt / SelectiveShield encryptors
│   ├── federation/            # FedAvg simulator, client sampling
│   ├── metrics/               # LeakScore components (Entropy / Label / Empirical)
│   ├── models/                # ResNet-10 / ResNet-18 / ResNet-50 backbones
│   ├── scenarios/             # Defence scenarios V1..V5
│   └── visualization/         # Plotters
├── hpc/                       # SLURM scripts and YAML configs for Matilda
│   ├── config_resnet10_t2_low.yaml
│   ├── config_resnet50_t2_low.yaml
│   ├── slurm_resnet50_extended_phase2_only.sh
│   └── slurm_*.sh
├── scripts/
│   ├── analyze_r50_extended.py     # ResNet-50 floor-vs-scaling analysis
│   └── paper/                      # Table / figure builders
├── tests/
│   ├── attack_sanity_check.py      # Phase-2 attack replay harness
│   └── test_fisher_classifier_head_guarantee.py  # unit-test fixture
├── paper_artifacts/           # Released per-cell JSONs and figures
├── run_headless.py            # Phase-1: train and save per-round artefacts
├── run_attacks.py             # Phase-2: replay attacks against saved artefacts
├── run_sensitivity.py         # LeakScore sensitivity sweep
├── recompute_metrics.py       # Re-derive metrics from saved artefacts
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md                  # This file
```

## Installation

The code targets Python ≥ 3.10 with PyTorch and a CUDA-capable GPU (Tesla V100 16 GB or comparable). Install dependencies into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The CIFAR-10 dataset is automatically downloaded on first run into `data/cifar-10-batches-py/`.

### HPC

The `hpc/` directory contains SLURM job scripts targeting an HPC cluster with V100 GPUs. Adjust the partition / account / paths in the `#SBATCH` headers before submission.

## Reproducing the headline results

The paper uses a two-phase evaluation protocol so that V1/V2/V3/V4 defences are evaluated against the **same** saved gradient artefact and any observed difference is attributable to the encryption policy rather than to client-sampling variance.

### Phase 1: train and save artefacts

```bash
python run_headless.py \
    --config hpc/config_large.yaml \
    --output results/training.json \
    --tb-dir results/tb_logs \
    --artifacts-dir results/artifacts \
    --seed 42
```

This trains a ResNet-18 over 250 federated rounds on the hard 3-of-10 non-IID partition (300 clients) and saves the per-round gradient state to `results/artifacts/round_*/`.

### Phase 2: replay attacks against saved artefacts

```bash
python tests/attack_sanity_check.py \
    --artifact-dir results/artifacts \
    --round 249 \
    --attack gradinversion \
    --n-iter 20000 \
    --batch-size 1 \
    --defence fisher \
    --defence-pct 0.10 \
    --variant paper \
    --model resnet18 \
    --out results/attacks/fisher_gradinversion_b1_rho0p10_round249_seed42.json
```

For the full headline matrix, sweep `--attack ∈ {gradinversion, ggcdm, gi_nas}` and `--defence ∈ {none, fhe, maskcrypt, fisher}` over seeds {42, 123, 456}. The corresponding SLURM script is `hpc/slurm_defence_sweep_v2.sh`.

### Architecture-transfer study

The ResNet-10 / ResNet-50 backbones use dedicated configs and SLURM scripts:

```bash
# Phase-1
sbatch hpc/slurm_resnet50_train_and_attack.sh
# Phase-2-only against existing artefacts
sbatch hpc/slurm_resnet50_extended_phase2_only.sh
```

### ResNet-50 floor analysis

The extended-$\rho$ sweep on ResNet-50 ($\rho \in \{0.05, 0.10, \ldots, 0.90\}$) is reproduced by `hpc/slurm_resnet50_extended_phase2_only.sh`. Analyse the resulting 33 JSON files with:

```bash
python scripts/analyze_r50_extended.py <ATTACK_DIR>
```

This prints the per-attack PSNR pivot table, the floor-vs-scaling verdict (FLOOR HOLDS / BORDERLINE / SCALES per attack family), and the band-membership summary used in the paper's Figure on the floor result.

### Unit tests

```bash
pytest tests/
```

The `test_fisher_classifier_head_guarantee.py` fixture locks the structural patch that prevents the small `fc.bias` tensor from falling below the global top-$K$ Fisher threshold (the failure mode diagnosed on seed-456 at ResNet-18 round 249).

## Key findings

### Privacy outcome, label channel collapses universally
Across all 12 defence×attack cells, every defence (FHE, MaskCrypt, AdaGuard-Fisher) reduces label-recovery ASR to 0. The classifier-head guarantee makes this architecture-portable: ASR = 0 on every ResNet-10 / 18 / 50 cell at $\rho = 0.10$ (108 cells total).

### Pixel channel, AdaGuard's Fisher targeting holds the noise band
On the headline ResNet-18 matrix, V4 holds GradInversion and GI-NAS in the noise band (PSNR < 10 dB) at $\rho = 0.10$, matching the V2 (FHE) cryptographic upper bound to within multi-seed dispersion while encrypting an order of magnitude less of the gradient surface.

### Architecture-scaling caveat (concentrated on GradInversion at converged stages)
At fixed $\rho = 0.10$ and late training, GradInversion PSNR climbs with parameter count: ResNet-10 (4.9 M) noise, ResNet-18 (11.2 M) noise, ResNet-50 (23.5 M) blob.

### ResNet-50 floor is non-gradient-bounded
The extended $\rho$-sweep on ResNet-50 reveals that PSNR does not vary with encryption budget across $\rho \in \{0.05, \ldots, 0.90\}$, GradInversion sits in a 0.84 dB band entirely in the noise region. The residual reconstruction is therefore bounded by non-gradient leakage (BatchNorm running stats, attack label-conditional priors), not by the unencrypted gradient mass. LPIPS does rise monotonically with $\rho$, indicating that the defence has a perceptual effect even when PSNR is floor-bounded.

### Fisher mass trajectory
On ResNet-50 across training (rounds 50 / 100 / 150), the `fc.weight` Fisher rank drops from #1 to #3 as convolutional layers learn features. This is direct mechanistic evidence that the classifier-head guarantee is structurally redundant early in training and load-bearing late.

## Dataset and artefacts

We release the per-cell attack JSONs, training trajectories, and reconstruction images for the headline matrix and the architecture-transfer study under `paper_artifacts/`. The full per-round model checkpoints (~700 GB for the ResNet-50 trajectories) are available on request given storage constraints.



## License

Released under the MIT License unless otherwise noted in third-party components (`breaching`, `lpips`, `diffusers`).

## Contact

For questions about reproducing the results or accessing the full per-round artefact archive, please open an issue or contact the corresponding author.
