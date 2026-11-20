# Atlas-Interoperability Layer for Connectome Learning

This is a **reviewer-facing software snapshot** accompanying the manuscript
**"A lightweight atlas-interoperability layer for connectome learning across
heterogeneous brain atlases"**, currently under review at *Nature
Communications*. It is not a final public release, and nothing here implies
endorsement or acceptance by the journal.

**Reviewers are invited to run the synthetic examples** in §2. They need no
data access and run on a laptop CPU in a few minutes.

The toolkit provides:

1. **Atlas mapping**: fit a directed atlas-to-atlas mapping from paired regional
   fMRI time series, save and reload it, apply it to existing FC or SC, and
   evaluate reconstruction similarity.
2. **FLECHA**: federated learning (FedAvg) across sites whose connectomes are in
   different atlases, plus the manuscript's simulated-site experiment with
   FLECHA, Single-site and Uniform comparisons.
3. **MECHA**: meta-matching across atlas spaces: a multi-output DNN trained on
   transformed connectomes, ridge adaptation on small support sets, support-only
   ridge and DNN baselines, and paired comparisons.
4. **Evaluation pipelines**: reconstruction similarity, native vs cross-atlas ridge
   prediction, and the resolution and reconstruction-prediction analyses.
5. **A synthetic demo suite** that runs all of the above on CPU with no external data.

![Framework](figures/figure1_framework.png)

> **Status.** This is a manuscript-aligned implementation, tested on
> synthetic data only. It is a reconstruction from the manuscript, not the
> original historical experiment code, and it has not been used to reproduce
> the manuscript's numbers. See
> [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for what was reused,
> corrected and reimplemented, and for every assumption.

---

## 1. Installation

Requirements: Python ≥ 3.10 and the packages in [requirements.txt](requirements.txt)
(numpy, scipy, pandas, scikit-learn, torch, POT, tqdm). A GPU is optional;
CUDA is used if available (`--device auto`), otherwise the CPU.

Run every command from the **repository root** (the directory containing
this README).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt     # runtime deps + pytest
pip install -e . --no-deps              # installs the atlas_interop package
```

Use `python -m atlas_interop <command>` as shown throughout. Installation time
was not measured (packages were installed once, on the tested machine, from
PyPI; torch from the PyTorch CPU wheel index). Installation has not been
checked on any other machine.

**Tested environment** (the only one tested): macOS 26.6 on Apple M5 (arm64),
CPU only, Python 3.14.10, numpy 2.5.5, scipy 1.18.1, pandas 3.1.1,
scikit-learn 1.9.1, torch 2.15.0 (CPU wheel), POT 0.9.7.post1, tqdm 4.70.1,
pytest 9.2.0. Other platforms and the lower bounds in `requirements.txt` have
not been tested.

---

## 2. Synthetic examples for reviewers

All examples generate deterministic synthetic data, run the real public CLI
commands, and check their outputs. They write only inside the `--out-dir` you
give, which must be new or an earlier demo directory (then pass `--force`).

```bash
python -m pytest                                    # 66 tests

# Atlas mapping, FLECHA (explicit sites + five-site simulation), evaluation pipeline
python -m atlas_interop demo --out-dir demo_runs/quick --workflows transform,flecha,flecha-sim,evaluate

# Synthetic MECHA experiment (§6.1)
python -m atlas_interop demo --out-dir demo_runs/mecha --workflows mecha
```

Measured once each on the tested machine (Apple M5, CPU, the environment
above):

| Command | Time | Peak memory |
|---|---|---|
| tests | 30 s | not measured |
| four-workflow command | 108 s | 0.65 GB |
| MECHA experiment | 73 s | 0.89 GB (141 MB written) |

Omitting `--workflows` runs all five workflows in one directory. That combined
command was not separately timed; expect roughly the sum, about 3 minutes.
Runtimes elsewhere are unmeasured.

| Output (under `--out-dir`) | Contents |
|---|---|
| `demo_report.json` | pass/fail and runtime per workflow, every command run, headline numbers |
| `data/` | small shared cohort for `transform`/`flecha`/`evaluate` (atlases `synA` 24, `synB` 18, `synC` 30 regions) |
| `transform/` | fitted mappings, mapped FC/SC, reconstruction summaries |
| `flecha_sites/` | public explicit-site FLECHA: sites native in `synA` and `synC`, held-out test site native in `synB` |
| `flecha_simulated/` | five-site, five-atlas FLECHA experiment (§5c); results in `flecha_simulated/results/` |
| `evaluation/` | reconstruction + prediction runs, `analysis_summary.json` |
| `mecha/` | the MECHA experiment; results in `mecha/results/` (§6.1) |
| `*/participant_level/` | synthetic per-participant outputs (local only, §8) |

All examples use synthetic-specific presets (`demo` for FLECHA, transform and
evaluation; `synthetic` for MECHA). Their numbers show that the code runs as
designed. They are not manuscript results.

---

## 3. Data formats

All inputs carry **explicit string IDs**. Files are joined by ID, never by row
position. IDs are opaque: they are not parsed, and leading zeros are preserved.
Numeric ID arrays are rejected.

### Connectomes: `.npz`

| key | shape / type | required |
|---|---|---|
| `mats` | float, `(n_regions, n_regions, n_samples)`, symmetric, finite | yes |
| `subject_ids` | string, `(n_samples,)` | yes |
| `session_ids` | string, `(n_samples,)` (repeat visits) | no |
| `atlas` | string scalar | recommended |
| `region_ids` | string, `(n_regions,)` region order | recommended |
| `metadata_json` | JSON string | no |

### Regional time series: `.npz`

Same keys, but `mats` has shape `(n_samples, n_regions, n_timepoints)`.

### Phenotype tables: `.csv` / `.tsv`

A `subject_id` column, optionally a `session_id` column, and one named numeric
column per phenotype. Empty cells and `NA`/`NaN` mean missing. Any other
non-numeric value is an error.

### ID lists: `.txt`

One ID per line; blank lines and `#` comments are ignored.

Writing your own files:

```python
import numpy as np
np.savez_compressed("fc_MyAtlas.npz",
    mats=fc,                                     # (n_regions, n_regions, n_samples)
    subject_ids=np.asarray(ids, dtype=str),      # keep as strings
    atlas=np.array("MyAtlas"),
    region_ids=np.asarray(region_labels, dtype=str))
```

**Checks performed.** Duplicate keys, missing keys where exact pairing is
required, wrong axis order, non-square or asymmetric connectomes, non-finite
values, atlas-name mismatches, region-count mismatches, region-order
mismatches (when `region_ids` are present on both sides), and prohibited
participant overlap all raise an error. Without `region_ids`, only the region
*count* can be checked; reordered regions with the same count cannot be
detected.

With `session_ids`, joins use (subject, session). Cohort-separation checks and
simulated-site partitions always use subject IDs, so all visits of one
participant stay on the same side of a split.

---

## 4. Atlas mapping

```bash
# Fit: needs ONLY paired reference time series (same participants in both files)
python -m atlas_interop fit-mapping \
    --source-timeseries ref/timeseries_Schaefer200.npz \
    --target-timeseries ref/timeseries_Glasser.npz \
    --source-atlas Schaefer200 --target-atlas Glasser \
    --out mappings/Schaefer200_to_Glasser.npz \
    --reference-ids-out local/reference_ids.txt      # optional, participant-level

# Apply: needs ONLY the mapping and source-atlas FC or SC
python -m atlas_interop apply-mapping \
    --mapping mappings/Schaefer200_to_Glasser.npz \
    --connectomes cohort/fc_Schaefer200.npz --modality fc \
    --out cohort/fc_Schaefer200_to_Glasser.npz       # rank defaults to 50

# Optional evaluation: needs native target-atlas connectomes of the same participants
python -m atlas_interop eval-reconstruction \
    --mapped cohort/fc_Schaefer200_to_Glasser.npz \
    --native-target cohort/fc_Glasser.npz --out-dir results/recon
```

**Mapping file (format version 2).** It contains the matrix `mapping` with
shape `(n_target, n_source)`, `orientation = target_by_source`,
`normalization = source_columns_sum_to_one`, the atlas names, the dimensions,
optional region IDs, and fit settings. It contains **no participant-level
data**. Files from the original release (key `mapping_matrix`) are rejected
with an explanation; refit them.

**Method** (Methods, "Connectome transformations"):
1. Per-participant cost = 1 − min-max-normalized cross-atlas Pearson correlation.
2. Average the participant costs into one group cost M.
3. Solve entropic Sinkhorn (λ = 0.01) with that same M for every participant
   and time point.
4. Column-normalize each plan, average within temporal windows, then across
   windows and participants.

Apply uses the top-k eigencomponents ranked by |eigenvalue|, positive and
negative spectral embeddings, $U_t = T U_s$, and
$Ĉ_t = U_t⁺U_t⁺ᵀ − U_t⁻U_t⁻ᵀ$.

---

## 5. FLECHA

### 5a. Public workflow: your own, already partitioned sites

Each training site supplies connectomes in **its own native atlas**, labels,
and, if its atlas differs from the target, a mapping to the target atlas. The
test site supplies native target-atlas connectomes and labels. No site needs
connectomes in any other atlas.

```json
{
  "target_atlas": "Glasser",
  "phenotype": "age",
  "rank": 50,
  "train_sites": [
    {"name": "siteA", "connectomes": "siteA_fc_Schaefer200.npz", "labels": "siteA.csv",
     "mapping": "mappings/Schaefer200_to_Glasser.npz"},
    {"name": "siteB", "connectomes": "siteB_fc_Shen268.npz", "labels": "siteB.csv",
     "mapping": "mappings/Shen268_to_Glasser.npz"}
  ],
  "test_site": {"name": "siteT", "connectomes": "siteT_fc_Glasser.npz", "labels": "siteT.csv"}
}
```

```bash
python -m atlas_interop flecha --config sites.json --preset ukb --out-dir results/flecha \
    [--single-site]          # also train each site alone, compute-matched
```

Relative paths resolve against the config file's directory. Each run:
1. transforms each site separately;
2. takes local minibatch Adam steps on each client;
3. averages client parameters with **equal client weights**;
4. redistributes the average;
5. evaluates on the test site.

Sites are never concatenated. Training sites and the test site must not share
participants. Missing labels are excluded and counted, never replaced. A site
with fewer labelled samples than the batch size is an error.

> **Not a deployment.** The backend runs every client in one process and must
> be able to read every site's files. It simulates federated optimization. It
> does not communicate across institutions, does not implement secure
> aggregation, and provides no privacy guarantee.

### 5b. Manuscript experiment: simulated sites

One cohort with connectomes in every atlas is partitioned into
N = number of atlases simulated sites, over 10 seeds by default. For each
target atlas:
- site 0 is the native held-out test site;
- each other site uses one non-target atlas as its native atlas;
- FLECHA and Single-site train on those sites' transformed connectomes;
- Uniform trains on the **same participants'** native target connectomes.

All three methods share partitions and test participants.

```json
{
  "phenotype": "age",
  "labels": "phenotypes.csv",
  "atlases": {"Schaefer100": "fc_Schaefer100.npz", "Schaefer200": "fc_Schaefer200.npz", "...": "..."},
  "mappings": {"Schaefer100->Schaefer200": "mappings/Schaefer100_to_Schaefer200.npz", "...": "..."},
  "target_atlases": ["Schaefer100", "Schaefer200"],
  "exclude_subjects": "local/reference_ids.txt"
}
```

```bash
python -m atlas_interop flecha-simulate --config sim.json --preset ukb --family ukb_fc_age \
    --out-dir results/flecha_sim
# combine several runs into one BH family, if needed:
python -m atlas_interop flecha-summarize --runs a/runs.csv b/runs.csv --out summary.csv
```

`summary.csv` gives, per target atlas:
- mean r for each method;
- Δr = FLECHA − Single-site over matched seeds, where multiple Single-site
  sources within a seed are combined in Fisher-z space;
- a one-sided paired Wilcoxon test (FLECHA > Single-site);
- the BH-adjusted q across the configurations in the file;
- the descriptive Uniform − FLECHA gap.

Presets: `ukb` (256/128/64, BatchNorm, batch 64, 100 rounds × 10 local steps)
and `hcp` (128/64, LayerNorm, batch 32, 200 rounds × 5 steps). Both use
dropout 0.2, Adam with lr 1e-3 and weight decay 1e-4, and Single-site epochs
matched to the FLECHA sample budget and clipped to 5–80.

### 5c. Synthetic five-site reviewer experiment

The `flecha-sim` demo workflow (code: `src/atlas_interop/flecha_experiment.py`)
runs a fixed synthetic design. It is a reviewer example only. The public
explicit-site interface (§5a) and the manuscript simulation (§5b) keep
accepting any number of sites and atlases, and their presets are unchanged.

| Item | Value |
|---|---|
| Atlases | `synA` 24, `synB` 18, `synC` 30, `synD` 21, `synE` 27 regions, all generated from one shared latent connectivity process |
| Mappings | `synA/B/C/D → synE`, fit only on paired time series of a separate 12-person reference cohort (checked disjoint from the 600 downstream participants) |
| Sites (per partition seed) | five disjoint, nonempty sites of 120: training sites 1–4 native in `synA`, `synB`, `synC`, `synD`; held-out test site native in `synE` (the common target) |
| Partition seeds | 101–105 (five partitions), frozen in `experiment_config.json` before training |
| Phenotype | `age` (synthetic), complete in this design |
| Preset | `demo`: 32/16 hidden units, BatchNorm, batch 16, 15 rounds × 3 local steps, rank 10 |

How the three methods differ. All are scored on the **same labelled
participants of the synE test site** (identical ID fingerprint per partition):
- **FLECHA**: each training site maps its own native FC into `synE`. The
  shared FedAvg backend runs local minibatch updates on each of the four
  clients and averages their parameters with equal weights after each round.
  Site data are never pooled.
- **Uniform**: the same four site memberships (checked by ID fingerprint) and
  the same FedAvg procedure, but on those participants' **native** `synE` FC,
  generated directly in `synE` and never mapped. It shows what federated
  training would reach if every site already used the target atlas.
- **Single-site**: each training site is fit **separately and alone**:
  - its FC is mapped to `synE`;
  - one model is trained by ordinary minibatch Adam on that site's labelled
    participants only, with no server, rounds, synchronization or averaging;
  - all four site scores are reported, and the per-partition Single-site value
    is their prespecified Fisher-z mean, the aggregation rule stated in the
    manuscript Methods. No site is chosen by test score.

  The standalone budget is
  `ceil(rounds × local_steps × n_clients × batch / n_site)` epochs, clipped to
  5–80 (`flecha.matched_epochs`). In this run that is 24 epochs of 7
  minibatches, so 168 steps on each site, and each site is trained
  independently. FLECHA and Uniform take 180 local steps in total (45 per
  client × 4 clients). Each standalone model therefore processes about the
  same number of samples (2,688) as the whole federated run (2,880), as the
  manuscript's compute-matching rule intends.

Outputs in `flecha_simulated/results/`:
- `runs.csv`: per partition and method: r, MSE, update steps, test and
  training fingerprints;
- `sites.csv`: roles, native atlases, realized counts, fingerprints;
- `summary.csv`: per partition;
- `overall.json`.

**Observed result (one run).** Pearson r on the held-out synE site:

| Partition seed | FLECHA | Uniform | Single-site (Fisher-z mean of 4; min–max) | n test |
|---|---|---|---|---|
| 101 | 0.767 | 0.824 | 0.716 (0.330–0.826) | 120 |
| 102 | 0.790 | 0.832 | 0.752 (0.552–0.819) | 120 |
| 103 | 0.787 | 0.814 | 0.624 (0.349–0.744) | 120 |
| 104 | 0.766 | 0.805 | 0.717 (0.407–0.795) | 120 |
| 105 | 0.737 | 0.821 | 0.748 (0.622–0.807) | 120 |
| mean | 0.769 | 0.819 | 0.711 | |

FLECHA exceeded the Single-site summary in 4 of 5 partitions (mean +0.058) and
was below Uniform in all 5 (mean −0.050). Individual standalone sites varied
widely: the `synC` site averaged 0.45 and the other three 0.76–0.79. Five
partitions are too few for inference (smallest attainable one-sided p is
0.031). These are synthetic numbers, not manuscript results.

---

## 6. MECHA

```bash
# 1. Meta-training: source-atlas training and validation cohorts, transformed to the target atlas
python -m atlas_interop mecha-train \
    --train-connectomes cheadle_train_fc_Schaefer200.npz \
    --val-connectomes   cheadle_val_fc_Schaefer200.npz \
    --mapping mappings/Schaefer200_to_Glasser.npz \
    --phenotypes phenotypes.csv --train-phenotypes "p1,p2,...,p16" \
    --preset ukb --out-dir results/mecha/train

# 2. Adaptation and evaluation: NATIVE target-atlas support and test cohorts
python -m atlas_interop mecha-adapt \
    --checkpoint results/mecha/train/mecha_dnn.pt \
    --support-connectomes reading_fc_Glasser.npz \
    --test-connectomes newcastle_fc_Glasser.npz \
    --phenotypes phenotypes.csv --test-phenotypes "h1,...,h10" \
    --meta-training-ids results/mecha/train/participant_level/meta_training_ids.txt \
    --modality fc --source-label Schaefer200 --target-label Glasser \
    --preset ukb --out-dir results/mecha/adapt \
    [--splits-dir my_splits/]     # support_k{K}_rep{R}.txt; each must contain exactly K IDs
```

- **Training** fits only on the meta-training participants and keeps the epoch
  with the lowest validation loss. The checkpoint stores the architecture, the
  vectorization convention, the target atlas, and the ordered training
  phenotype names. The DNN has one output per meta-training phenotype.
- **Adaptation** uses the DNN's *phenotype predictions* on native support and
  test connectomes as features. It fits one ridge per held-out phenotype on
  each support draw (alpha chosen by CV within the support set) and evaluates
  on the test cohort.
- **Baselines.** Support-only ridge on native edges and a support-only DNN use
  the same support draws and test participants. Test labels are used only for
  scoring.
- **Rejected inputs.** Overlapping training and held-out phenotypes, support,
  test or meta-training participant overlap, and support/test files that were
  produced by `apply-mapping` (non-native) are all rejected.
- **Outputs.** `per_phenotype.csv` records requested and realized support and
  test counts after missing-label filtering. `summary.csv` gives, per k, the
  benchmark (the stronger support-only model by mean Fisher-z), the MECHA gain,
  a one-sided Wilcoxon test on Fisher-z values, and BH q. Combine runs across
  atlas transfers with `mecha-summarize` to form the manuscript's per-modality
  family.

Splits: `mecha-make-splits` writes nested support draws (one permutation of
the support pool per draw seed; each K is a prefix). `mecha-adapt
--baselines-only` fits only the support-only baselines so one run can be
shared by several checkpoints; `mecha-compare` joins MECHA and baseline rows
on (K, draw, phenotype) and refuses to pair rows whose support or test
fingerprints differ.

Manuscript preset `ukb`: 128/64, BatchNorm, dropout 0.5, Adam with $l_r$ 1e-4 and
weight decay 1e-4, batch 256, K ∈ {10, 20, 50, 100, 200}, 10 support draws.
The UKB phenotype construction (190 fields → PCA within categories →
26 components → 16/10 split; Supplementary Tables 2–3) and the
Cheadle/Reading/Newcastle site files are cohort-specific preparation, not part
of this toolkit. For generic use, supply any named phenotype table and choose
the two disjoint phenotype lists.

### 6.1 Canonical synthetic MECHA example

`python -m atlas_interop demo --out-dir demo_runs_v2/mecha --workflows mecha`
runs this fixed design (code: `src/atlas_interop/mecha_experiment.py` and
`synthetic_mecha.py`; full generative model in their docstrings):

| Item | Value |
|---|---|
| Cohorts (subject-disjoint, checked) | mapping reference 20, meta-training **10,000**, validation **1,000**, support pool 1,000, test 2,000 |
| Atlases | source `synS` 36 regions → target `synT` 28 regions; mapping fit on paired reference time series only; spectral rank 20 |
| Inputs | meta-training/validation: source FC transformed to `synT`; support/test: **native** `synT` FC |
| Phenotypes | 16 meta-training, 10 distinct held-out, complete labels |
| Support | K = 10, 20, 40, 100; 10 predetermined draw seeds; nested prefixes of one permutation per seed |
| Checkpoint | one meta-training DNN, reused for every K and draw; reloaded through the public loader |
| Baselines | support-only Ridge (native edges) and support-only DNN on the identical draws and test cohort |
| Preset | `synthetic`: 128/64, BatchNorm, dropout 0.5, lr 1e-3, batch 256, ≤150 epochs, patience 20. Synthetic-specific; the manuscript uses lr 1e-4 (`ukb` preset) |

**Synthetic design.** Connectomes come from 11 phenotype-relevant latent
factors plus 30 nuisance factors that never enter a phenotype. By
construction, the 16 meta-training phenotypes depend on factors 0–3, and the
10 held-out phenotypes depend on the same factors 0–3 (75 % of their
connectome signal) plus factors 7–10 that no meta-training phenotype uses
(25 %). The shared predictive signal is therefore a property of this synthetic
design, not something the experiment tests. Labels are never written into the
connectomes.

Before any held-out score exists, `experiment_config.json` records the
configuration, seeds and planned comparisons. `results/` holds:
- `summary.md` / `summary.csv`: one row per K;
- `per_draw.csv`, `per_draw_per_phenotype.csv`;
- `experiment_report.json`: 21 passed checks, realized counts, split
  fingerprints, validation metrics, and runtimes.

The checks include:
- cohort disjointness;
- exact realized K;
- identical support and test fingerprints for MECHA, Ridge and DNN in every
  K, draw and phenotype;
- checkpoint reload and output order;
- best-validation-epoch selection.

**Observed result (one run, 71.4 s of steps).** The DNN's validation r was
0.595 (range 0.536–0.642 over the 16 phenotypes). Values below are the mean
over 10 draws of the per-draw mean r across the 10 held-out phenotypes:

| K | MECHA | Ridge | DNN | MECHA − Ridge (wins/10) | MECHA − DNN (wins/10) |
|---|---|---|---|---|---|
| 10 | 0.242 | 0.082 | 0.056 | +0.160 (10) | +0.186 (10) |
| 20 | 0.308 | 0.140 | 0.084 | +0.169 (10) | +0.225 (10) |
| 40 | 0.358 | 0.191 | 0.112 | +0.167 (10) | +0.246 (10) |
| 100 | 0.409 | 0.310 | 0.232 | +0.099 (10) | +0.177 (10) |

These are synthetic results from one generator in which the pretraining and
held-out tasks share signal by design. They show the full MECHA pipeline
running on that design. They are not a reproduction of the manuscript and not
evidence about real data. The 10 draws share one test cohort and are not
independent replications. Pretraining size is fixed, so there is no scaling
claim.

---

## 7. Evaluation pipeline

```json
{
  "modality": "fc",
  "native": {"Schaefer200": "fc_Schaefer200.npz", "Glasser": "fc_Glasser.npz"},
  "mapped": {"Schaefer200->Glasser": "fc_Schaefer200_to_Glasser.npz",
             "Glasser->Schaefer200": "fc_Glasser_to_Schaefer200.npz"},
  "phenotypes": "phenotypes.csv",
  "phenotype_columns": ["age", "fluid_intelligence"]
}
```

```bash
python -m atlas_interop evaluate --config eval.json --preset manuscript --out-dir results/eval
```

For every directed pair (and the native diagonal), the pipeline runs these
steps:
1. Draw a 20 % tuning subset. In it, select features by univariate Pearson
   p < 0.1, standardize, and choose the ridge penalty by 3-fold CV over
   10⁻¹ … 10³.
2. Exclude the tuning participants.
3. Run 10 × repeated 10-fold CV. Selection and scaling are re-fit on each
   training fold; models are fit on transformed connectomes and predict from
   **native** target-atlas connectomes.

The pipeline writes these outputs:
- `reconstruction_summary.csv`
- `prediction_runs.csv`, with one row per pair and repeat
- `prediction_tuning.csv`
- `analysis_summary.json`, with Spearman correlations for resolution
  (log n_target/n_source) vs reconstruction, and for reconstruction vs
  prediction

---

## 8. Participant-level outputs

Anything that identifies participants is written under a `participant_level/`
subdirectory with a `NOTICE.txt`:
- per-participant reconstruction scores and predictions
- split membership and support draws
- tuning and reference IDs

Keep these local and do not publish them. Mapping files, summaries and
run-level CSVs contain no participant IDs. `participant_level/`, `demo_runs/`, `demo_runs_v2/`
and `demo_output/` are in `.gitignore`.

---

## 9. What this repository does not contain

- Any UK Biobank, HCP-YA, HCP-D or Yale data, real participant records, or
  trained checkpoints.
- The cohort-specific preprocessing, UKB phenotype construction, and site
  splits used in the manuscript.
- Figure-styling code.

Reproducing the manuscript's numbers requires authorized access to those
datasets and the preparation steps listed in
[docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md#reproduction-prerequisites).

## Data availability

UK Biobank, HCP-YA, HCP-D and the Yale dataset are available through their
respective data-access procedures and data-use agreements. Subject-level data
cannot be redistributed.
