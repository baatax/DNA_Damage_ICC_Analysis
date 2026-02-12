# 2Cohort Talazoparib test data

This directory stores small representative parquet files generated from the real
2-cohort Talazoparib experiment config.

Generate files with:

```bash
python create_test_dataset_from_config.py \
  --source-config 2Cohort_TZ_config.json \
  --output-config 2Cohort_TZ_test_config.json \
  --output-data-dir test_data/2cohort_tz
```

The generated files are referenced by `2Cohort_TZ_test_config.json`.

SLURM driver (cluster):

```bash
sbatch submit_generate_test_dataset.slurm
```
