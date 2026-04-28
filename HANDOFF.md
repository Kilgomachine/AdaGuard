# Handoff to the next Claude session

You are picking up the AdaGuard IEEE TDSC paper finalization on a new
machine. The previous session's conversation transcripts and project
memory are machine-local on the user's old machine and **will not** be
available. This file is the bridge.

Read this top-to-bottom before doing anything else. Then ask the user
what they want to work on.

---

## Identity

- **Project**: AdaGuard — adaptive multi-metric defence against
  gradient-inversion attacks in federated learning.
- **Target venue**: IEEE Transactions on Dependable and Secure
  Computing (TDSC). Currently in pre-submission review.
- **User**: Firas Aguir, OU graduate student. Working with a
  professor (not yet looped in for review at handoff time).

## Where things live

| Asset | Location | In git? |
|---|---|---|
| Code | https://github.com/Kilgomachine/AdaGuard (master) | yes |
| Paper LaTeX source | Overleaf (canonical) | **no** (`paper_overleaf/` is gitignored) |
| Paper deliverable | `paper_overleaf.zip` (local build artefact) | **no** (`*.zip` gitignored) |
| Result JSONs | `data/paper_data/...` | yes (committed in c5d49d3 for migration; see "Known cruft" below) |
| HPC | Matilda @ Oakland — `ssh maguir@hpc-login.oakland.edu` (Duo MFA) | n/a |
| HPC project | `/projects/secure-distributed-ml/AdaGuard` | mirrors GitHub |
| HPC scratch | `/scratch/projects/secure-distributed-ml/` | results dir, 45-day expiry |

The user's old machine had OneDrive sync; the new machine does **not**.
This is why we pushed result data into git.

## Status at handoff

- Paper is in good shape. About to be sent to the professor for review.
- The user has the latest `paper_overleaf.zip` from the previous session
  and will share it in the new conversation.
- Final committed state: `c5d49d3` (data snapshot) on top of `1f7b21f`
  (scripts + tables) on top of `e6ac2f7` (the grad_accum bug fix).

## Defence variant naming (post-renumbering)

V-labels were originally V1, V2, V4, V6 with V3 (DP) and V5 (MaskCrypt
random) skipped. **Renumbered to V1–V4 contiguous** at the user's
request right before handoff. If you read older session transcripts or
git history, you'll see V6 = AdaGuard, V4 = MaskCrypt — translate.

| V-label | Defence | File-name token |
|---|---|---|
| V1 | undefended FedAvg (control) | `none` |
| V2 | FHE (full encryption, cryptographic upper bound control) | `full` / `fhe` |
| V3 | MaskCrypt (gradient-magnitude-guided selective HE) | `maskcrypt` |
| V4 | AdaGuard-Fisher (our contribution; Fisher-guided selective HE) | `fisher` |

The file-name tokens (`none`, `full` / `fhe`, `maskcrypt`, `fisher`)
are baked into the data dirs — those are NOT renumbered. Only V-labels
in the paper prose changed.

## Sensitivity scenario naming (post-renumbering)

S-labels were S1, S3, S4, S5, S6, S7, S9, S11 with gaps. **Renumbered
to S1–S8 contiguous** in the same pass. The data file
`data/paper_data/sensitivity/sensitivity_aggregate.json` still uses
the original S1..S11 keys; the **paper** uses the new S1..S8.

| New label | Variable | Original |
|---|---|---|
| S1 | `entropy_bins` | S1 |
| S2 | $T_1$ | S3 |
| S3 | $T_2$ | S4 |
| S4 | $\alpha$ (label_weight) | S5 |
| S5 | $\beta$ (entropy_weight) | S6 |
| S6 | $\gamma$ (empirical_weight) | S7 |
| S7 | `mi_samples_per_class` | S9 |
| S8 | `focus_layers` | S11 |

Three scenarios EXIST in the data but are intentionally omitted from
the paper's sensitivity table. Do not promote them without thinking
hard:

- **Original S2 (`grad_accum_K`)** and **original S8 (`client_batch_size`)**
  — replay-harness tautologies. The cached round-249 gradient is
  invariant under these knobs because they only affect training-time
  gradient computation. Documented in §VI.D Limitations.
- **Original S10 (`maskcrypt_enc_pct`)** — V3-specific (MaskCrypt's ρ
  curve). Implicitly subsumed by the V4 ρ-sweep we ran for RQ5(iv).

## Recently shipped fixes you must know about

### `grad_accum_threshold` bug (commit e6ac2f7)

`adaguard/config.py` had `'grad_accum_threshold': 0.7` hardcoded in
DEFAULT_CONFIG. The simulator's fallback `config.get('grad_accum_threshold', config.get('T2', 0.7))`
only fired when the key was missing, but the default ensured it was
always present at 0.7. So lowering T2 in YAML didn't change accumulation
behaviour — accum **never fired** in the so-called "T2lo stress test"
(0/250 rounds despite LeakScore > 0.3 in every round).

**Fix**: default changed to `None`, and `simulator.py:376` now does an
explicit `if accum_threshold is None: accum_threshold = T2`.

This bug was caught by the user's intuition ("V6+accum makes GI-NAS
worse — that doesn't fit the logic, smells like a bug"). Trust their
domain instincts.

### `focus_layers` default

DEFAULT_CONFIG sets `focus_layers: ['fc2.weight', 'fc2.bias']` — legacy
from the old `smallcnn` model. ResNet-18's classifier head is `fc.*`.
**Any new YAML must override `focus_layers: ['fc.weight', 'fc.bias']`**
or the LeakScore label component pulls from a non-existent layer.
`hpc/config_1k.yaml` (the baseline) overrides correctly;
`hpc/config_t2_low.yaml` was patched to also override.

### V4+accum (formerly V6+accum) saga summary

Single-seed stress-test where T2=0.3 forces accumulation every round.
After the bug fix, accum fires 250/250 rounds (verified). Reported in
**§VI.D Limitations**, NOT in the headline Table IV — single-seed
finding, framed as a stress-test rather than a multi-seed result.

Net result: V4+accum **compounds the defence on 2 of 3 attacks**:

| Attack | V4 alone | V4+accum | Δ |
|---|---|---|---|
| GradInversion | 9.40 | 7.86 | −1.54 dB ✓ |
| GI-NAS | 7.20 | 7.78 | +0.59 dB neutral |
| GGCDM | 10.37 | 5.92 | −4.45 dB ✓ |

ASR remains 0.000 across all V4 cells. Last-10-round mean accuracy is
0.634 (FIXED) vs 0.640 (baseline) — within 0.6 points. ~2× per-round
client compute cost when accum fires.

The user noticed late-training oscillation amplitude is ~50% wider in
the FIXED trajectory but pointed out the baseline also oscillates
significantly (this is heavy non-IID, not an accum-induced
instability). Don't make the mistake I did and call the FIXED run's
final-round accuracy a "crash" — it's just a downswing of the same
oscillation pattern visible in baseline.

### Random-vs-guided citation rewrite (§IV)

Originally claimed MaskCrypt and SelectiveShield each demonstrate
"guided beats random" ablation at fixed ρ. **They don't** — verified
by web search of both papers (MaskCrypt's ablations are
encryption-vs-no-encryption + membership inference; SelectiveShield's
sweep τ and ρ but no random baseline). Paragraph in
`04.design-study.tex` was rewritten to claim only what the papers
actually show: "informed mask selection is sufficient" (true) and
"informed-vs-random is orthogonal to AdaGuard's contribution"
(defensible).

If a reviewer pushes on this, the honest answer is: AdaGuard does not
include a Fisher-vs-random ablation in this submission; future
extended-version work could add one.

### "Fully preserving utility" softened

Intro previously said "while fully preserving the baseline model's
utility". Backed by no concrete number anywhere in the paper.
Softened to "at no measurable cost to model utility (final-round test
accuracy within sample-std of the undefended baseline across three
seeds; see §V.B)" and added a **§VI.B Global utility** subsubsection
in `08.discussion.tex` that reports actual numbers:

| Defence | last-10 mean test acc (n=3 seeds) |
|---|---|
| V1 (None) | 0.635 ± 0.027 |
| V2 (FHE) | 0.639 ± 0.002 |
| V3 (MaskCrypt) | 0.639 ± 0.016 |
| V4 (AdaGuard-Fisher) | 0.630 ± 0.010 |

V4 vs V1 = −0.005, well inside sample-std. Numbers are from
`simulator._evaluate()` — held-out CIFAR-10 test set, top-1 argmax,
real evaluation each round. NOT estimated.

## Things explicitly deferred (not blockers)

1. **Delete `data/paper_data/defence/v6_accum/` (the BUGGY data dir)**
   — the user said they'd do this themselves. If still present and
   you're unsure, leave it.
2. **Breaching wrapper number verification** — §VI.G cites
   PSNR=6.47 dB, cosine=0.60, ASR=1.0 from a one-off Matilda slurm log.
   Numbers are not in local JSON. User will verify post-submission.
3. **Breaching wrapper debug** — wrapper produces ~6 dB vs ~18 dB
   reference. Cosine match converges (0.60) and label decoder works
   (ASR=1.0), so the bug is wrapper-level (normalisation /
   batch-reduction / parameter ordering). Documented as a
   reproducibility limitation, deferred to extended-version follow-up.
4. **Multi-seed V4+accum sweep over T2 ∈ {0.3, 0.5, 0.7}** — explicitly
   noted as future work in §VI.D.
5. **`l1`/`l2`/`l3` orphan citations** — already removed from the bib
   (commit e6ac2f7's lineage).

## Things explicitly declined — do NOT propose

- **V5 / MaskCrypt-random ablation** — user declined ("we forgo
  MaskCrypt"). Cite-only path was taken.
- **Fisher-vs-random V_R ablation** — user declined; cite-only via §IV
  paragraph.
- **Threshold-based defence variant** (encrypt where Fisher > τ instead
  of top-ρ%) — user declined ("no luxury for new things").
- **Promoting V4+accum to a Table IV row** — single-seed; lives in
  §VI.D as a limitation finding. Don't promote without a multi-seed
  follow-up.

## User's working style — observed patterns

- **Strategic over tactical**: dislikes diving into code without
  framing. Will say "we don't want to go into a rabbit hole here / so
  let's be strategic about it." When asking for action, give the
  options first, recommend one, then act.
- **Catches overreactions**: pointed out the late-training oscillation
  was baseline behaviour, not an accum bug. When you flag something as
  alarming, double-check by looking at the comparable baseline.
- **Honest > polished**: agreed to drop "fully preserving utility"
  immediately when called out. Will accept honest scope reductions.
- **Trusts intuition, often correctly**: sniffed out the
  grad_accum_threshold bug from the V4+accum results pattern. Take
  user-flagged "this doesn't fit the logic" seriously.
- **Time pressure**: working under deadline. Won't have "another day"
  to iterate. Don't propose multi-step plans they need to execute when
  a single-step plan works.
- **Short replies**: "yeah sure", "obviously B", "okay this is good"
  — don't need long confirmation, just execute.
- **Catches our typos and misreadings of own data** — happens with
  human-Claude pairs at the end of long sessions. Cross-check
  numerical claims against the underlying JSON before stating them.

## Critical conventions

- **Never commit unless explicitly asked.** Standard Claude Code
  convention applies.
- **`paper_overleaf/` is gitignored** — Overleaf is the LaTeX
  source-of-truth. Don't push LaTeX changes to GitHub. The local copy
  is a sync zone.
- **`*.zip` is gitignored** — `paper_overleaf.zip` is a build artefact,
  never commit it.
- **Numerical claims need JSON backing.** A previous audit found one
  real mismatch (consistency table claim) and one numeric drift
  (11.17M → 11.18M ResNet-18 param count). Both fixed. Don't
  re-introduce.
- **The `data/paper_data/` data committed to git in c5d49d3 is for
  migration only.** The user plans to retract it after migration via
  `git rm --cached -r` plus `.gitignore` patterns. Don't be surprised
  if a future commit removes those tracked paths.

## What probably happens next

User opens the new conversation, drops the `paper_overleaf.zip`,
and asks for one of:

- "I'm reading the paper, here's what I see" — apply edits, rebuild
  zip, repeat.
- "Professor sent feedback X" — apply, rebuild, ship.
- "Help me with item Y from the deferred list" — usually breaching
  debug or multi-seed V4+accum.

The paper is **near-final**. Don't propose new experiments or
restructuring unless the user asks. Editing/responding to professor
feedback is the expected mode.

## File map cheat sheet

| Need | Look in |
|---|---|
| LaTeX prose | `paper_overleaf/*.tex` (after user supplies the zip) |
| Auto-built tables | `paper_artifacts/tables/*.tex` |
| Hand-built tables | `paper_overleaf/tables/*.tex` |
| Defence sweep results (single-seed) | `data/paper_data/defence/*.json` |
| Defence sweep results (multi-seed) | `data/paper_data/defence/seed{42,123,456}/` |
| V4+accum FIXED results | `data/paper_data/defence/v6_accum_FIXED/` (NB: dir name predates the V6→V4 renumbering) |
| ρ-sweep results | `data/paper_data/sensitivity/rho_sweep/` |
| Sensitivity aggregate | `data/paper_data/sensitivity/sensitivity_aggregate.json` |
| Training trajectories | `data/paper_data/training/{none,full,maskcrypt,fisher}_seed{42,123,456}_300clients.json` |
| Buggy V4+accum trajectory (forensic) | `data/paper_data/training/v6_accum/fisher_seed42_T2lo.json` |
| FIXED V4+accum trajectory | `data/paper_data/training/v6_accum/fisher_seed42_T2lo_RERUN.json` |
| Slurm scripts | `hpc/slurm_*.sh` |
| HPC YAML configs | `hpc/config_*.yaml` |
| Paper-build pipeline | `scripts/paper/_data.py`, `scripts/paper/build_tables.py` |
| Replay/attack harness | `tests/attack_sanity_check.py` |

## A note on commits and data tracking

Two snapshot commits (`1f7b21f`, `c5d49d3`) were pushed JUST for the
machine migration — to bring 70+ result JSONs across to the new
machine without OneDrive. The user plans to untrack these data dirs
after migration. If you find yourself confused about why 30 MB of
JSONs are tracked, that's the answer.

## Last words

The paper is good work. The user has been honest about scope, fixed
real bugs (grad_accum_threshold), removed real overclaims (the
"guided beats random" citation, the "fully preserving utility"
phrasing). Trust the work that's there, don't bloat it, and let them
ship.

— previous Claude, 2026-04-27
