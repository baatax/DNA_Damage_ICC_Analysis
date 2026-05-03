# Testin Environment README

This file documents the minimal environment and commands required to test this repository.

## Python version

- Recommended: Python 3.10+

## Required packages

Install the runtime + testing dependencies:

```bash
python -m pip install pandas numpy scipy matplotlib scikit-learn pyarrow pytest
```

## Run all tests (using repo test data)

```bash
PYTHONPATH=. pytest -q
```

The suite uses bundled fixtures under `test_data/` and synthetic smoke data created in tests.

## Integration run with bundled config fixture

```bash
python run_dna_damage_pipeline.py Pilot_1_config_test_config.json --output ./test_output_pilot1
```

## New EC50-neighborhood validation outputs to check

After an integration run, validate these tables are produced under `tables/`:

- `ec50_neighbor_profiles.csv`
- `genotype_comparison_ec50_neighbors_<feature>.csv`
- `genotype_comparison_ec50_band_groups_<feature>.csv`

## What this validates

- dilution-series relabeling behavior
- test-dataset generation from config
- plotting smoke test (including PCA/loadings/EC50-DMSO plotting paths)
- dose-response outputs and QC report generation
