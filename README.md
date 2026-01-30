# DNA Damage ICC Analysis Pipeline

A production-ready bioinformatics pipeline for analyzing single-cell immunocytochemistry (ICC) data from DNA damage experiments. It processes parquet files containing single-cell features (DNA damage markers, morphology, intensity measurements) from multiple genotypes treated with DNA-damaging compounds, performs comprehensive statistical analysis, and generates publication-quality results and plots.

## Overview

The pipeline is designed for studying how different genetic backgrounds (genotypes) respond to DNA-damaging compounds through dose-response relationships and cellular phenotyping. It supports multi-genotype, multi-drug experimental designs with automatic normalization, quality control, crowding correction, dose-correlated feature analysis, and statistical comparison.

## Project Structure

```
DNA_Damage_ICC_Analysis/
├── run_dna_damage_pipeline.py          # Main pipeline orchestrator (13 steps)
├── dna_damage_parquet_pipeline.py      # Core analysis components
├── dose_response_analysis.py           # Dose-response curve fitting & statistics
├── pipeline_plots.py                   # Publication-quality plot generation
├── demo_dna_damage_pipeline.py         # Examples with synthetic data
├── example_dna_damage_config.json      # Template configuration
├── 2Cohort_TZ_config.json              # Example real-world config (Talazoparib)
└── submit_DDP_pipeline.slurm           # SLURM cluster job submission script
```

## Requirements

- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- pyarrow
- matplotlib (optional — for plot generation; pipeline degrades gracefully without it)

## Quick Start

### 1. Create a configuration file

Copy and edit the template configuration:

```bash
cp example_dna_damage_config.json my_experiment_config.json
```

The JSON config defines your experiment metadata, genotypes, drugs, data file paths, and expected EC50 values. See [Configuration](#configuration) for details.

### 2. Run the pipeline

```bash
python run_dna_damage_pipeline.py my_experiment_config.json \
    --output ./results \
    --workers 8 \
    --crowding-threshold 0.3 \
    --log-level INFO
```

### 3. Run on a SLURM cluster

Edit `submit_DDP_pipeline.slurm` to point to your config file and environment, then:

```bash
sbatch submit_DDP_pipeline.slurm
```

The pipeline supports checkpoint/resume so interrupted jobs can continue where they left off by adding the `--resume` flag.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `config` | (required) | Path to JSON configuration file |
| `--output`, `-o` | from config | Output directory |
| `--workers`, `-w` | 4 | Number of parallel workers |
| `--resume` / `--no-resume` | `--resume` | Resume from checkpoints |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--crowding-threshold` | 0.3 | \|Spearman r\| threshold for crowding correction |
| `--generate-slurm` | — | Generate a SLURM submission script and exit |
| `--slurm-partition` | `normal` | SLURM partition name |
| `--slurm-time` | `4:00:00` | SLURM time limit |
| `--slurm-mem` | `64G` | SLURM memory allocation |
| `--slurm-email` | — | Email for SLURM notifications |

## Pipeline Steps

The pipeline executes 13 sequential analysis steps:

| Step | Name | Description |
|------|------|-------------|
| 1 | Load data | Load parquet data from all genotypes and drugs |
| 2 | Per-genotype preprocessing | Robust z-score normalization within each genotype |
| 3 | Across-all preprocessing | Robust z-score normalization across all genotypes |
| 4 | Crowding metrics | Compile cell density measurements by drug and dose |
| 5 | Crowding correction | Identify and exclude features correlated with cell crowding |
| 6 | QC metrics | Compile quality control metrics per well |
| 7 | Per-well CSVs | Generate well-level aggregated profiles |
| 8 | PCA | Principal Component Analysis (4 variants: 2 modes × corrected/uncorrected) |
| 9 | Dose-response | Fit four-parameter Hill equation curves |
| 10 | Dose-correlated analysis | Identify features correlated with drug dose, PCA at EC50 |
| 11 | Statistics | Genotype comparisons (EC50 z-tests, Mann-Whitney U) |
| 12 | Plot generation | Publication-quality plots for all analyses |
| 13 | Manifest | Save pipeline metadata and file inventory |

## Configuration

The JSON configuration file has two main sections:

### `metadata`

| Field | Description |
|-------|-------------|
| `experiment_name` | Name for the experiment |
| `control_label` | Label for vehicle control (e.g., `"DMSO"`) |
| `dilut_column` | Column name containing dilution/concentration info |
| `output_dir` | Default output directory |

### `genotypes`

Each genotype contains a `description` and a `drugs` map. Each drug entry specifies:

| Field | Description |
|-------|-------------|
| `path` | Path to the parquet file with single-cell features |
| `ec50_um` | Expected EC50 in micromolar (used for dose-correlated analysis) |
| `max_dose` | Maximum dose as a string (e.g., `"100uM"`, `"10mM"`) |
| `moa` | Mechanism of action |
| `dilution_factor` | Dilution series factor (e.g., `3.0` for 3x dilutions) |

EC50 values specified in the config are used by the dose-correlated feature analysis (Step 10) to select the closest available dose for each genotype-drug combination when comparing groups at their respective EC50 concentrations.

## Output Structure

```
output_dir/
├── per_genotype/              # Per-genotype normalized data
├── across_all/                # Across-all-genotypes normalized data
├── per_well/                  # Well-level aggregated profiles
│   ├── per_genotype/
│   └── across_all/
├── crowding/                  # Cell crowding/density metrics
│   ├── crowding_by_drug_dose.csv
│   └── crowding_feature_correlations.csv
├── qc/                        # Quality control reports
├── pca/                       # PCA coordinates, loadings, variance
│   ├── per_genotype/
│   │   ├── uncorrected/
│   │   └── corrected/
│   └── across_all/
│       ├── uncorrected/
│       └── corrected/
├── dose_response/             # Fitted dose-response curves (Hill equation)
│   ├── dose_response_fits.csv
│   ├── dose_response_summary.csv
│   ├── fold_change_vs_control.csv
│   └── config_ec50_values.csv
├── dose_correlated/           # Dose-correlated feature analysis
│   ├── dose_feature_correlations.csv
│   ├── dose_correlated_pca_profiles.csv
│   ├── dose_correlated_pca_loadings.csv
│   ├── dose_correlated_pca_variance.csv
│   └── dose_correlated_summary_by_dose.csv
├── statistics/                # Genotype comparison results
├── plots/                     # Publication-quality plots
│   ├── pca/
│   │   ├── per_genotype/{uncorrected,corrected}/
│   │   └── across_all/{uncorrected,corrected}/
│   ├── crowding/
│   └── dose_correlated/
├── cache/                     # Checkpoint data for resume
└── manifest.json              # Pipeline metadata and file inventory
```

## Key Features Analyzed

- **DNA Damage Markers**: foci count, gamma-H2AX mean intensity
- **Proliferation**: Ki67 positivity and intensity
- **Morphology**: area, perimeter, eccentricity, solidity, circularity
- **Intensity**: DAPI, SYTO RNA mean intensity
- **Crowding**: local density, nearest-neighbor distance, neighbor counts
- **Quality Control**: cell count per well, segmentation coverage, border cells

## Analysis Components

### Crowding Correction

The pipeline computes Spearman correlations between every feature and the primary crowding metric (e.g., nearest-neighbor distance) across all combined raw data. Features with |r| above the crowding threshold (default 0.3) are excluded from a **corrected** feature set. Both corrected and uncorrected analyses run in parallel through PCA and downstream steps, so you can compare results with and without crowding confounds.

A feature exclusion plot is generated showing the correlation of each feature with the crowding metric, with excluded features highlighted in red.

### PCA (Principal Component Analysis)

Four PCA variants are computed:

| Normalization | Correction | Description |
|---------------|-----------|-------------|
| Per-genotype | Uncorrected | All features, normalized per genotype |
| Per-genotype | Corrected | Crowding-excluded features, normalized per genotype |
| Across-all | Uncorrected | All features, normalized across all genotypes |
| Across-all | Corrected | Crowding-excluded features, normalized across all genotypes |

For each PCA variant the pipeline generates:
- **Scatter plot** of PC1 vs PC2 colored by genotype
- **Feature contribution bar charts** for top 15 features per PC (up to 5 PCs)
- **Scree plot** showing individual and cumulative variance explained

### Dose-Correlated Feature Analysis

Identifies features most strongly correlated (Spearman) with drug dose across the across-all normalized dataset. The top 15 features are used to:

1. **PCA at EC50**: Data is filtered to each group's respective EC50 dose (from config) plus DMSO controls, then PCA is run on the dose-correlated features to compare groups
2. **Line plots**: Combined plot of each dose-correlated feature vs drug dose, with one line per genotype and SEM error bands

### Dose-Response Modeling

Fits a four-parameter Hill equation to dose-response data:

```
y = bottom + (top - bottom) / (1 + (EC50 / x)^hill_slope)
```

Returns EC50, Hill slope, R-squared, and confidence intervals for each drug-genotype combination. Config-specified EC50 values are saved alongside fitted values.

### Statistical Comparisons

- Z-tests for EC50 comparisons between genotypes
- Mann-Whitney U tests for response value comparisons at specific doses
- Multiple testing correction (Bonferroni and FDR)

### Normalization

Two normalization modes are supported:
- **Per-genotype**: Robust z-score normalization within each genotype separately
- **Across-all**: Normalize across all genotypes together for direct comparison

## Plots

All plots are generated with a colorblind-friendly palette and saved as 150 DPI PNG files. The plotting module (`pipeline_plots.py`) uses the non-interactive `Agg` backend for headless/cluster compatibility. If matplotlib is not installed, the pipeline skips plot generation and completes all other steps normally.

Generated plots include:
- PCA scatter plots (PC1 vs PC2) for each normalization mode and correction
- Feature contribution bar charts for each principal component
- Variance explained scree plots (individual + cumulative)
- Crowding feature exclusion plot
- Dose-correlation ranking bar chart
- PCA scatter of dose-correlated features at EC50
- Dose-correlated feature line plots vs drug dose

## Demo / Testing

Generate synthetic data and run example analyses:

```bash
# Run all examples
python demo_dna_damage_pipeline.py all

# Run a specific example
python demo_dna_damage_pipeline.py basic
python demo_dna_damage_pipeline.py dose-response
python demo_dna_damage_pipeline.py genotype
python demo_dna_damage_pipeline.py crowding
python demo_dna_damage_pipeline.py qc
```
