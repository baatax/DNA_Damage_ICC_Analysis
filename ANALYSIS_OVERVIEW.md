# CellPaint EC50 Analysis Overview (Recreation Guide)

This document explains **exactly what analyses this repository runs** and how to reproduce them in a different repo with a different dataset.

---

## 1) What this pipeline does

For each configured analysis run, the pipeline executes:

1. Load and concatenate datasets from a JSON manifest.
2. Parse/attach numeric dose columns (`dose_uM`, `dose_label`).
3. Select a feature set (`all_features`, `shortlist_interpretable`, or `shortlist_dose_response`).
4. Single-cell QC filtering (area percentile filtering, etc.).
5. Aggregate single-cell rows to well-level profiles (medians + metadata passthrough).
6. Flag outlier wells and optionally remove flagged wells.
7. Normalize well-level features.
8. Compute crowding-correlated feature exclusions.
9. Run **two variants** of downstream analysis:
   - `uncorrected` (all selected features)
   - `crowding_corrected` (selected features minus crowding-correlated features)
10. For each variant, run:
    - PCA + loadings
    - optional UMAP
    - per-channel PCA
    - correlation-distance heatmaps
    - EC50-only subset PCA + distance heatmap
    - DMSO control-only PCA + distance heatmap
        - dose-response fitting + dose-response plots

The same general pipeline is applied to all configured run names, and run-level results are collected in `run_manifest.json`.

---

## 2) Required inputs

### 2.1 Dataset manifest (`datasets.json` style)

The pipeline expects a JSON list with entries like:

```json
[
  {
    "dataset_id": "2KO_P1",
    "drug": "KO",
    "genotype": "2KO",
    "ec50": "666nM",
    "path": "/path/to/dataset_dir"
  }
]
```

Per item, you may provide either:
- `path` (directory/file that can resolve both parquets), or
- explicit `profiles` + `cells` parquet paths.

The loader looks for:
- `profiles_norm.parquet`
- `single_cell_features_numeric.parquet`

in either the provided directory or `Cellpaint_CPSAM/` under it.

### 2.2 Required metadata columns in your data

Defaults assume these columns exist (or are remapped in `analysis.yaml`):

- plate: `dataset_id`
- well: `well`
- site: `site`
- compound: `drug`
- dose: `dose_uM` (computed from dose parsing)
- replicate: `genotype`

### 2.3 Dose source column

You provide `--dose_col` (default `Dilut`).
The pipeline creates:

- `dose_label` (string)
- `dose_uM` (float)

If `--dose_col` is absent in a dataframe, fallback parsing attempts columns such as `Dilut`, `dose`, `treatment`, `subgroup`, `plate`, `base_name`.

---

## 3) Configuration (`analysis.yaml`)

Key knobs to port when recreating:

- `metadata.*`: remap to your dataset’s column names.
- `control`: identify controls by `compound` and/or `dose`.
- `min_cells`: low cell-count QC threshold.
- `normalization`: `control_based` or `global`.
- `runs`: any of `all_features`, `shortlist_interpretable`, `shortlist_dose_response`.
- `pca_components`, `random_seed`, `umap_enabled`.
- `crowding_corr_threshold`.
- `dose_response.min_doses`, `dose_response.max_iter`, `dose_response.per_feature_subset`.
- `outlier_wells.*`: enable/disable and thresholds for dropping QC-flagged wells.

---

## 4) Feature sets used

### 4.1 `all_features`

- Uses all numeric columns except metadata/non-feature exclusions and a few prefixes.

### 4.2 `shortlist_interpretable`

Curated interpretable features including morphology, density, occupancy, channel intensities, localization ratios, and mitochondrial structure metrics.

### 4.3 `shortlist_dose_response`

Curated dose-response-optimized subset emphasizing stable morphology ratios, occupancy, localization, mitochondrial metrics, and an intensity anchor.

Both shortlist runs include fallback logic (for example `nn_dist_px -> nbr_count_r100 -> nbr_count_r50`).

---

## 5) Detailed per-run analysis flow

For each run (`all_features` / shortlist):

1. **Feature coverage report**
   - Writes `tables/feature_coverage.json` with present/fallback/missing status.

2. **Single-cell QC**
   - Area-based percentile filtering by plate.
   - Writes `tables/cell_qc_summary.parquet`.

3. **Aggregation to well profiles**
   - Aggregates selected features to well-level medians.
   - Retains metadata passthrough (compound, dose, genotype, `dose_label`, etc.).
   - Writes:
     - `tables/profiles_raw.parquet`
     - `tables/cell_qc_by_well.parquet`

4. **Well QC + outlier removal**
   - Flags wells for issues such as low/extreme cell counts, area extremes, nearest-neighbor extremes, and feature outliers.
   - Writes `tables/qc_flags.parquet`.
   - If enabled, removes wells matching `outlier_wells.drop_flags`; writes `tables/qc_removed_wells.parquet` when applicable.

5. **Normalization**
   - Control-based or global normalization.
   - Writes:
     - `tables/profiles_norm.parquet`
     - `tables/normalization_diagnostics.json`

6. **QC/normalization diagnostics plots**
   - Cell count distribution, area vs cell count, feature diagnostics,
     control distributions, replicate correlation heatmap.

7. **Crowding correction**
   - Computes a per-well crowding proxy from single-cell columns when available
     (`crowding_local_mean`, `crowding_density`, `crowding_sigma50`, `crowding`),
     else falls back to `cell_count`.
   - Excludes features with `|corr(feature, crowding_metric)| > crowding_corr_threshold`.
   - Writes crowding outputs in `run_dir/crowding/`:
     - `crowding_feature_correlations.csv`
     - `crowding_excluded_features.txt`
     - crowding correlation plots

8. **Branch to two variants**
   - `uncorrected`
   - `crowding_corrected`

---

## 6) Variant analyses (run for each variant)

Inside `<run_name>/<variant_name>/`:

1. **Derived feature creation**
   - Computes transformed dose-response helpers such as:
     - `log_nn_dist`
     - `log2_ratio_<channel>`
     - `logit_occup_<channel>`
     - `mito_frag_index`
     - `mito_network_index`

2. **PCA embedding + loadings**
   - Saves PCA model info to `models/pca_info.json`.
   - Saves PCA scatter/loadings plots in `plots/embedding/`.

3. **Optional UMAP embedding**
   - If enabled in config, computes UMAP from PCA and plots in `plots/embedding/`.

4. **Per-channel PCA**
   - Groups features by imaging channel and runs PCA per channel.
   - Saves per-channel plots and `pca_info.json` under `plots/per_channel/<channel>/`.

5. **Global distance heatmap**
   - Correlation distance on grouped sample labels.
   - Saves `plots/clustering/distance_heatmap_global.png`.

6. **EC50-focused analysis**
   - Selects rows at each drug’s EC50 label from manifest.
   - Runs EC50 PCA + loadings and EC50 distance heatmap.

7. **DMSO/control-only analysis**
   - Selects controls by configured compound/dose logic.
   - Writes `plots/dmso_controls/selection_summary.json`.
   - Runs control-only PCA/loadings and control distance heatmap when feasible.

8. **Dose-response modeling**
   - Builds plate/global control reference profiles.
   - Computes delta features (`delta_*`) from control.
   - Computes:
     - `phenotype_magnitude` (L2 norm of delta vector)
     - `phenotype_angle_to_control` (cosine-based metric)
   - Fits logistic dose-response curves (4PL with fallback to 3PL).
   - Writes tables such as:
     - `tables/profiles_delta.parquet`
     - `tables/dose_response_metrics.parquet`
     - `tables/per_feature_dose_response_metrics.parquet`
     - `tables/dose_response_metrics_per_genotype.parquet` (if genotype fitting possible)
   - Writes model JSON:
     - `models/dose_response_fits.json`
   - Writes dose-response plots under `plots/dose_response/`.

---

## 7) Output structure to expect

Given `--out_dir out/parallel`, top-level outputs include:

- `out/parallel/orchestrator.log`
- `out/parallel/run_manifest.json`
- `out/parallel/<run_name>/...` for each run

Each run directory includes:

- `tables/` (parquet/json metrics + QC tables)
- `plots/` (qc, normalization, embedding, EC50, control-only, per-drug, dose-response)
- `logs/run.log`
- `crowding/` (crowding analysis artifacts)
- `uncorrected/` and `crowding_corrected/` variant subtrees with their own `tables/plots/models`

---

## 8) How to recreate this in another repo/dataset

1. Copy these components:
   - `parallel_analyses/`
   - `data_io.py`, `dose.py`, `features.py`, `constants.py`, `plotting.py`, `utils.py`
   - `analysis.yaml`
   - `analyze_profiles.py` (optional convenience wrapper)

2. Create your dataset manifest (same schema as `datasets.json`).

3. Ensure your parquet files contain required metadata columns, or remap in config.

4. Update `analysis.yaml`:
   - metadata mappings
   - control definition
   - run list and thresholds

5. Run:

```bash
python -m parallel_analyses.cli \
  --datasets_json <your_datasets.json> \
  --out_dir <your_output_dir> \
  --dose_col <your_dose_column> \
  --config <your_analysis.yaml>
```

6. Validate:
   - Check `<out_dir>/run_manifest.json` for run-level success/errors.
   - Confirm each run has both variants (`uncorrected`, `crowding_corrected`) unless skipped for too few features.
   - Check `tables/feature_coverage.json` for missing shortlist features and fallbacks.

---

## 9) Reproducibility checklist

- Pin package versions (especially numpy/pandas/scipy/sklearn/matplotlib/pyarrow/umap-learn).
- Keep `random_seed` fixed.
- Keep identical `analysis.yaml` and dataset manifest formatting.
- Ensure dose labels are consistent with `ec50` strings in your manifest.
- Confirm control definition (`compound` and/or `dose`) matches your assay design.
- Preserve column names or update metadata mappings.

---

## 10) Common migration pitfalls

- **Dose parsing failures**: mismatched `--dose_col` or non-parseable labels.
- **No controls found**: incorrect `control.compound` / `control.dose`.
- **Too few selected features**: shortlist names don’t match new dataset columns.
- **Over-aggressive crowding exclusion**: threshold too low for your data.
- **EC50 subset empty**: manifest `ec50` labels don’t match `dose_label` values.

---

## 11) Minimal command templates

Primary run:

```bash
python -m parallel_analyses.cli \
  --datasets_json datasets.json \
  --out_dir out/parallel \
  --dose_col Dilut \
  --config analysis.yaml
```

Convenience wrapper:

```bash
python analyze_profiles.py \
  --datasets_json datasets.json \
  --out_dir out/parallel \
  --dose_col Dilut
```



## Output layout (2026 update)
- `crowding_analysis/`: crowding tables and crowding plots.
- `crowding_corrected_analysis/`: mirrored `tables/`, `plots/`, `models/` from `crowding_corrected/`.
- `uncorrected_analysis/`: mirrored `tables/`, `plots/`, `models/` from `uncorrected/`.
- Per-drug PCA export folder is removed.
- Dilution enforcement: all datasets are validated against 9 non-control 1:3 dilutions starting at 100uM plus DMSO control.


## 7) User-facing output folder structure

At the end of each run, the pipeline creates these top-level folders inside the run output directory:

- `crowding_analysis/`
  - `tables/` crowding correlation/exclusion tables
  - `plots/` crowding plots including binned correlation-by-channel bar chart
- `crowding_corrected_analysis/`
  - full corrected `tables/`, `plots/`, and `models/`
  - includes `plots/dmso_controls/` and `plots/ec50_focused/` PCA, loading, and top-feature dose-response plots
- `uncorrected_analysis/`
  - full uncorrected `tables/`, `plots/`, and `models/`
  - includes `plots/dmso_controls/` and `plots/ec50_focused/` PCA, loading, and top-feature dose-response plots

Within each variant, dose-response plotting keeps summary/fold-change outputs and EC50 tables on summary panels, and excludes separate fit-curve image generation.
