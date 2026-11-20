# Implementation notes

**What this code is.** `atlas_interop` is a *manuscript-aligned implementation
tested on synthetic data*, prepared as a reviewer-facing snapshot for a
manuscript under review. It is **not** the original historical experiment code
(that code was not recovered; the MECHA training script in particular was
missing), and it has **not** been used to numerically reproduce the
manuscript's results. Only synthetic data were used in development and testing.

**Sources checked.** The manuscript main text (Methods, Results, figure
captions) and the Supplement (Supp. Figs. 1–9, Supp. Tables 1–3), as supplied
in the development session. Copies are present locally next to the repository,
but they could not be rendered again in the latest session (no PDF renderer
installed), so later checks relied on the text extracted earlier. Where the
manuscript was silent, the first repository release (commit `68ff5ba`) was used
as evidence. Any remaining choice is marked **ASSUMPTION** below.

## 1. Manuscript-to-code coverage

| Manuscript capability | Section | Code | Status |
|---|---|---|---|
| Group cost matrix M from participant correlation costs | Methods, Connectome transformations | `transform.participant_cost`, `group_cost` | Implemented; corrected (P2) |
| Time-specific entropic OT, λ = 0.01, Sinkhorn | same | `transform.sinkhorn_plan`, `fit_mapping_arrays` | Implemented (reuses POT) |
| Column normalization, window/time/participant averaging, T ∈ R^{n_t×n_s} | same | `fit_mapping_arrays` | Implemented; corrected (P3) |
| Signed spectral embedding, top-k by \|λ\|, k = 50 | same | `transform.signed_embedding`, `transform_connectome` | Reused logic from legacy `spectral_embedding.py`; rank validated |
| Reconstruction similarity | Methods, Evaluation…; Fig. 2 top | `transform.reconstruction_similarity`, `evaluation.run_evaluation` | Implemented |
| Ridge prediction: 20 % tuning, p < 0.1, scaling, 3-fold alpha over 10⁻¹–10³, 10×10-fold CV, native held-out | same; Fig. 2 bottom; Supp. Fig. 3 | `evaluation.predict_pair` | New |
| Resolution vs reconstruction, reconstruction vs prediction | Supp. Figs. 4–5 | `evaluation.analysis_summaries` | New (Spearman, as in the captions) |
| FLECHA FedAvg, equal weights, local minibatch updates | Methods, Federated learning…; Supp. Fig. 1 | `flecha.fedavg_train` | Reused legacy FedAvg logic; refactored |
| Single-site with matched budget, 5–80 epochs | same | `flecha.matched_epochs`, `train_single_site` | Reused legacy rule |
| Uniform (native target data, same participants) | same | `flecha.run_simulated` | Reused legacy design (same split indices, native target file); participants are now matched by ID across atlas files instead of by row position |
| Simulated sites, N = #atlases, 10 seeds, site 0 held out | same | `flecha.partition_sites`, `run_simulated` | Reused legacy split logic; now subject-level |
| UKB and HCP model/training settings | same | `presets.FLECHA_PRESETS` | UKB reused; HCP new (LayerNorm, 128/64, batch 32, 200×5) |
| Fisher-z Single-site aggregation, one-sided Wilcoxon, BH, Uniform–FLECHA gap | same | `flecha.summarize`, `stats` | New |
| MECHA DNN on transformed meta-training cohort, early stopping on validation | Methods, Meta-matching… | `mecha.train_dnn` | **New** (original training code missing) |
| DNN predictions as features; per-phenotype ridge on support; independent test | same; Supp. Fig. 2a | `mecha.adapt_and_evaluate` | Reused the idea from legacy `ukbb_stacking.py`; corrected inputs |
| Support-only ridge and DNN baselines | Fig. 4; Methods | `mecha.adapt_and_evaluate` | New |
| Benchmark selection by Fisher-z mean, gain, one-sided Wilcoxon on Fisher-z, BH within modality | Methods | `mecha.summarize` | New |
| UKB phenotype construction (190 → PCA → 26 → 16/10) | Methods; Supp. Tables 2–3 | — | Not implemented (cohort-specific preparation) |
| Figure styling | Figs. 2–4 | — | Not implemented (out of scope) |

## 2. Corrections that change scientific behavior

**P2: group cost reuse.** Legacy `Optimal_transport.py` called
`mapping(..., train_index=[i])` once per participant, so each participant's
Sinkhorn plans used that participant's own cost matrix. The Methods state that
participant costs are averaged into one group-level M and that M is used for
the time-specific plans. `fit_mapping_arrays` now computes M once and passes
it to every Sinkhorn call. A spy test (`test_group_cost_is_reused_for_every_participant`)
covers this.

**P3: orientation and normalization.** Legacy plans were `n_s × n_t`
(source, target), normalized with `T / T.sum(axis=0)`, so each *target*
region's weights summed to one. The Methods define T ∈ R^{n_t×n_s} with "each
column describ[ing] how a source-atlas region is distributed across
target-atlas regions" and state that the weights from each *source* region
sum to one.
- The new plan P has marginals P1 = v_t and Pᵀ1 = v_s (target rows, source
  columns).
- Each plan is normalized by its column sums.
- Projection is U_t = T U_s.

These change the mapping values, not only its layout: a column-normalized
n_t×n_s matrix is not the transpose of a column-normalized n_s×n_t matrix.
Tests cover unequal sizes, an independent NumPy Sinkhorn solver, and a
hand-computed fit.

**Legacy artifacts.** Mapping files without `format`/`format_version`
(legacy key `mapping_matrix`) are **rejected** with an explanation. They are
not transposed or renormalized, because no transformation of a P3-legacy
matrix fit with per-participant costs equals a manuscript-specified mapping.
No legacy-compatibility path is offered.

**MECHA support and test inputs.** Legacy `ukbb_stacking.py` loaded
*mapped* connectomes for support and test, and by default scored every
subject, support subjects included.
- Support and test are now separate native target-atlas inputs.
- Files carrying a `transformed_from` tag are rejected.
- Support, test and (optionally) meta-training participants must be disjoint.

**MECHA output dimension.** Legacy code set the DNN output size from the
held-out phenotype file. The DNN now has one output per named meta-training
phenotype, and the held-out phenotypes must be a disjoint set.

**MECHA hyperparameters.** The legacy config (3 hidden layers 256/128/64,
dropout 0.25, lr 1e-3, wd 1e-3, batch 128) is replaced in the `ukb` preset by
the Methods values (128/64, BatchNorm, dropout 0.5, lr 1e-4, wd 1e-4,
batch 256).

**FLECHA.**
- Random targets are no longer substituted when labels are missing.
- Clients with fewer labelled samples than the batch size raise an error;
  before, they produced no updates or crashed.
- Simulated-site partitions keep all sessions of a participant together.
- The public interface accepts explicit, already partitioned sites.

**Subject identity.** Every loader now requires string `subject_ids` and joins
by ID. Legacy code paired everything by array position.

**Mapping files carry no participant data.** Legacy `save_mapping` stored the
per-subject `eval_scores` in the mapping file. Fit now saves only the matrix
and fit settings. Participant-level evaluation goes to `participant_level/`.

**Fit no longer needs connectomes.** Legacy `map_nodetime.py` required
native FC for both atlases and saved the mapping only after evaluating it.

## 3. Supplementary Fig. 2b inconsistency

Supp. Fig. 2b labels the support set "Newcastle" and the test set "Reading".
The main text, Table 1 ("UKB Reading … MECHA support"; "UKB Newcastle …
MECHA meta-test") and the Fig. 4 caption say Reading is the support cohort
(K drawn from 5,614) and Newcastle the test cohort (9,470). The
implementation and README follow the Methods: **Reading = support,
Newcastle = test**. The figure label appears to be swapped, and the panel
also reads "Meta-traing". The cohort roles are supplied by file, so the code
does not depend on site names.

## 4. Assumptions (not stated in the manuscript)

| Item | Choice | Evidence / reason | Setting |
|---|---|---|---|
| OT temporal window | 1 time point per window | Legacy code used `frame_size=1`; Methods mention windows without a size. With equal weighting, the final mean is the same for any window size whose windows are full (verified numerically: max difference 3e-16 between window 1 and 10 on a 30-point series). Results differ only when the last window is partial. | `--window-size` |
| OT positivity shift | 1e-5 | Legacy value; Methods: "a small positive constant" | `FitSettings.shift` |
| Sinkhorn solver | POT `ot.sinkhorn`, `method='sinkhorn'`, max 1e6 iterations, stopThr 1e-9 | Methods cite POT; iterations are a legacy value | `FitSettings` |
| Cost min-max scope | across all source–target pairs of one participant | Methods: "across all source-target region pairs" | — |
| Constant activity vector | uniform distribution | min-max undefined; legacy code returned the raw vector | — |
| Edge vectorization | upper triangle, k = 1, row-major (`np.triu_indices`) | legacy `mats2edges.com2edges` | stored in checkpoints |
| Eigen-decomposition | `eigh` of the symmetrized input; zero eigenvalues in the top k are dropped | Methods only define ± parts | reported per run |
| FedAvg client init | all clients start from the global model | standard FedAvg (McMahan et al., ref. 69). Legacy code initialized clients independently, which makes round-1 averaging mix unrelated networks. | `FlechaPreset.client_init` (`shared` / `independent`) |
| FedAvg optimizer state | Adam state persists per client across rounds | legacy behavior | `client_optimizer` (`persistent` / `reset`) |
| BatchNorm running stats | averaged with the other float tensors; `num_batches_tracked` from client 0 | legacy behavior | — |
| FLECHA target scaling | none (raw labels) | legacy behavior | `target_scaling` (`none` / `zscore`) |
| Single-site budget | ceil(rounds × local_steps × n_clients × batch / n_site), clip 5–80 | legacy formula; Methods state the 5–80 bound | — |
| Single-site reporting | each source site is trained and scored; Fisher-z mean per seed in the summary | Methods | — |
| Partition | random, equal-size split of subjects into N groups; site 0 = test | legacy `make_subject_splits`; Methods | — |
| MECHA target scaling | z-score meta-training phenotypes with meta-training statistics | conventional for multi-output MSE; legacy `mics_z_norm` | — |
| MECHA loss | mean squared error over observed entries (NaNs masked) | Methods: "average mean squared error across the 16 meta-training phenotypes"; legacy `msenanloss` | — |
| MECHA max epochs / patience | 300 / 50, restore best validation epoch | legacy `EPOCHS = 300`; the Methods say only "early stopping based on validation loss" | `MechaPreset.max_epochs`, `patience` |
| MECHA cohort inclusion | samples with all meta-training labels missing are dropped; partial NaNs are masked | legacy dataset rule | — |
| Stacking ridge | alpha ∈ {0.1, …, 100}, CV folds 2/3/5 for k ≤ 20/≤ 50/> 50, features unscaled | legacy `ukbb_stacking.get_cv` and grid | `stack_alphas` |
| Support sampling | without replacement, seeded per (k, repeat); explicit files must contain exactly k IDs | Methods: "randomly sampled" | `--splits-dir` |
| Support-only ridge | standardized native edges; alpha ∈ 10⁻¹…10⁵ by inner CV within the support set | not specified | `baseline_ridge_alphas` |
| Support-only DNN | 128/64, BatchNorm, dropout 0.5, Adam lr 1e-3 / wd 1e-4, batch 32, 100 epochs, multi-output over held-out phenotypes, **no early stopping** (there is no labelled data that is neither support nor test) | not specified; ref. 71 | `baseline_dnn_*` |
| Ridge-prediction alpha grid | 9 log-spaced values in 10⁻¹…10³ | range stated; density not | `PredictionPreset.alphas` |
| Ridge-prediction score | Pearson r of predictions pooled over the 10 folds of each repeat | Methods: "for each random split" | `--score pooled|fold_mean` |
| Ridge-prediction tuning subset | drawn once per pair and phenotype with the run seed; subject-level | Methods | `--seed` |
| Reconstruction resolution analysis | Spearman ρ over directed pairs | Supp. Fig. 4 caption reports Spearman, although the main text writes "r" | — |

## 5. Things that cannot be verified by the code

- **Region order.** Without `region_ids` in both the connectome and the
  mapping, only the region count is checked. A file with the same number of
  regions in a different order is accepted. Supply `region_ids` to make this
  check active.
- **Independence of mapping reference data.** The fit records reference IDs
  only if `--reference-ids-out` is given. `flecha-simulate` (via
  `exclude_subjects`) and `mecha-adapt` (via `--meta-training-ids`) can check
  overlap, but only against the lists you provide.
- **Modality.** Nothing stops a user from applying a mapping fit on fMRI to SC.
  This is the manuscript's design, but SC mappings are not modality-specific
  (Discussion, limitations).

## 6. Legacy compatibility

The modules of the first release (`FLECHA/`, `MECHA/`, `atlas_transform/`,
`utils/`, `scripts/` under `src/`, commit `68ff5ba`) have been removed from the
working tree by the authors. They remain in git history only. `atlas_interop`
does not import them. Mapping files saved by that code are rejected (§2).

## 7. Synthetic MECHA reviewer example

`python -m atlas_interop demo --workflows mecha` runs
`atlas_interop.mecha_experiment`. It is the only synthetic MECHA experiment.

A short history:
1. It replaced an earlier small demo: 200 meta-training participants carved out
   of the shared cohort, 6/3 phenotypes, K = 10/20/40 and 3 draws.
2. A later version added a matched "low-sharing" control condition.
3. That condition has now been removed, together with its label columns,
   checkpoint, cross-condition comparison and tests.

Removing the control did not change the retained experiment:
- The control labels reused the same random draws as the retained labels, so
  no random-number stream shifted.
- Regenerated reference time series, connectomes, IDs and phenotype tables are
  identical in loaded content to the previous run, apart from renaming `train_hi_XX` to
  `train_XX`.
- The support-draw fingerprints are identical.
- The rerun gave the same MECHA, Ridge and DNN means.

| Choice | Status |
|---|---|
| Meta-training DNN on transformed source connectomes; early stopping on a separate transformed validation cohort | manuscript-aligned (Methods) |
| Support and test natively in the target atlas | manuscript-aligned |
| DNN phenotype predictions as features; one ridge per held-out phenotype on the support set | manuscript-aligned |
| 16 meta-training / 10 held-out phenotypes; 10 support draws | manuscript-aligned counts |
| Support-only Ridge and DNN on native features, same draws and test cohort | manuscript-aligned design; settings ASSUMPTION (§4) |
| 10,000 / 1,000 meta-training / validation (manuscript 18,000 / 2,000) | synthetic-specific |
| K = 10, 20, 40, 100 (manuscript 10, 20, 50, 100, 200) | synthetic-specific |
| Nested support draws (one permutation per seed, K = prefix) | synthetic-specific design choice |
| Two atlases (36 → 28 regions), rank 20 (manuscript k = 50, with ≥ 100 regions) | synthetic-specific |
| `synthetic` preset: lr 1e-3, ≤ 150 epochs, patience 20 (manuscript lr 1e-4) | synthetic-specific, chosen for CPU run time; manuscript values stay in the `ukb` preset |
| Meta-training and held-out phenotypes share latent factors 0–3 | property of the synthetic generator, not a finding |
| Latent-factor generator with 30 nuisance connectivity factors | synthetic, not a brain model |

The 30 nuisance factors were added before any held-out MECHA score existed. A
plain ridge probe showed that support-only ridge was otherwise near ceiling
(r ≈ 0.7 at K = 100). The checkpoint's best epoch was 147 of the 150-epoch
cap, so longer training might lower validation loss slightly. That cap is part
of the fixed synthetic preset.

## 8. Synthetic five-site FLECHA reviewer example

`python -m atlas_interop demo --workflows flecha-sim` runs
`atlas_interop.flecha_experiment` through the `flecha-sim5` CLI command.

**What the previous version did** (inspected in code before the change). The
`flecha-sim` demo called the general `flecha-simulate` command with three
atlases (`synA`, `synB`, `synC`), two target atlases (`synB`, `synA`) and three
seeds. For each target and seed it:
- split participants into three sites: site 0 was the held-out test site in
  the target atlas, and each other atlas went to one training site;
- trained FLECHA with `fedavg_train` on the two transformed training sites;
- trained Uniform with `fedavg_train` on the same two sites' native target FC;
- trained Single-site with `train_single_site` once per training site, on that
  site alone, for `matched_epochs` epochs.

`train_single_site` was already a standalone minibatch loop with no
aggregation. The summary combined the two site scores per seed by Fisher-z
averaging. No bug was found. The version simply had only three sites, so two
training clients, and the test atlas changed between targets.

**What it does now.**
- Five atlases from one latent process.
- Four mappings to `synE`, fit on a separate reference cohort that is checked
  disjoint from the downstream participants.
- Five disjoint sites per partition seed: `synA`–`synD` for training, `synE`
  as the held-out test site.
- FLECHA and Uniform run through the same `fedavg_train`, on identical site
  memberships (checked by fingerprint).
- Single-site fits each of the four sites standalone with
  `train_single_site`. The per-partition summary is the Fisher-z mean of the
  four site scores, the manuscript's stated aggregation rule; no site is
  selected by test score.
- Every model is scored on the identical labelled synE test participants
  (checked by fingerprint).

Regression tests (`tests/test_flecha_experiment.py`):
- With spies installed, `fedavg_train` and `average_state_dicts` are never
  called on the Single-site path. A deliberate mutation that routed
  Single-site through `fedavg_train` made the test fail.
- Duplicate native atlases are rejected.
- Reference/downstream overlap is rejected.
- Partitions are disjoint and deterministic.

The public `flecha` and `flecha-simulate` commands and the `ukb`/`hcp`
presets are unchanged.

## Reproduction prerequisites

To reproduce manuscript numbers, a user needs:
1. Authorized access to UKB, HCP-YA, HCP-D and the Yale dataset.
2. The UKB Connectome Project matrices for the six FC and three SC atlases,
   and HCP/Yale time series and connectomes for Craddock-200,
   Brainnetome-246, Shen-268 and Schaefer-400, saved in the formats above
   with real IDs and `region_ids`.
3. The 775-participant Cheadle mapping-reference subset, its exclusion from
   downstream analyses, and the 18,000/2,000 Cheadle training/validation split.
4. The UKB phenotype construction (Supp. Tables 2–3) producing named columns
   for the 16 meta-training and 10 held-out components.
5. The `ukb`/`hcp`/`manuscript` presets with rank 50 and 10 seeds or repeats.

Even then, the assumptions in §4 (above all FedAvg initialization, early
stopping patience, baseline settings and the stacking grid) may make results
differ from the manuscript's.
