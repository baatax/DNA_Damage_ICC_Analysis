#!/usr/bin/env python3
"""Create small representative parquet test datasets from an existing experiment config.

This utility reads an experiment JSON config, samples each genotype/drug parquet while
preserving representation across available experimental grouping columns, and writes:
  1) sampled parquet files into a repository-local test data directory
  2) a cloned config JSON that points to those sampled parquet files
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pandas as pd


GROUPING_CANDIDATES = ["Dilut", "dilut_string", "group", "treatment", "is_control"]
ALT_PARQUET_FILENAMES = ("single_cell_features_numeric.parquet",)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _sampling_groups(df: pd.DataFrame) -> List[str]:
    return [col for col in GROUPING_CANDIDATES if col in df.columns]


def _iter_genotype_drug_entries(
    cfg: Dict,
) -> Iterator[Tuple[str, str, Dict]]:
    genotypes = cfg.get("genotypes", {})
    if not isinstance(genotypes, dict):
        raise ValueError("Config key 'genotypes' must be a JSON object.")

    for genotype, geno_cfg in genotypes.items():
        if not isinstance(geno_cfg, dict):
            raise ValueError(f"Config for genotype '{genotype}' must be an object.")
        drugs = geno_cfg.get("drugs", {})
        if not isinstance(drugs, dict):
            raise ValueError(
                f"Config key 'genotypes.{genotype}.drugs' must be a JSON object."
            )
        for drug, drug_cfg in drugs.items():
            if not isinstance(drug_cfg, dict):
                raise ValueError(
                    f"Config for genotype '{genotype}' drug '{drug}' must be an object."
                )
            yield genotype, drug, drug_cfg


def _resolve_source_parquet(path_value: str, config_dir: Path) -> Path:
    raw_path = Path(path_value)
    src_path = raw_path if raw_path.is_absolute() else (config_dir / raw_path).resolve()

    candidates = [src_path]
    candidates.extend(src_path.with_name(name) for name in ALT_PARQUET_FILENAMES)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Source parquet does not exist. Tried: {attempted}")


def _default_test_output_dir(output_config: Path) -> str:
    stem = output_config.stem.replace("_config", "")
    return f"./test_output_{_safe_name(stem.lower())}"


def _validate_sampling_args(min_rows_per_group: int, max_rows: int) -> None:
    if min_rows_per_group < 1:
        raise ValueError("min_rows_per_group must be >= 1")
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")


def sample_representative_rows(
    df: pd.DataFrame,
    random_state: int,
    min_rows_per_group: int,
    max_rows: int,
) -> pd.DataFrame:
    """Sample rows while preserving all columns and group representation.

    Notes
    -----
    - If grouping columns are present, at least one row is sampled from every group
      (subject to group being non-empty).
    - The returned sample is always bounded by `max_rows`.
    """
    _validate_sampling_args(min_rows_per_group, max_rows)

    if df.empty:
        return df.copy()

    grouping_cols = _sampling_groups(df)

    if not grouping_cols:
        n = min(max_rows, len(df))
        return df.sample(n=n, random_state=random_state).copy().reset_index(drop=True)

    grouped = list(df.groupby(grouping_cols, dropna=False, observed=False))
    n_groups = len(grouped)
    if n_groups > max_rows:
        raise ValueError(
            "max_rows is too small to preserve representation across all experimental "
            f"groups ({n_groups} groups > max_rows={max_rows})."
        )

    sampled_parts: List[pd.DataFrame] = []

    # Pass 1: guarantee representation (one row per group)
    for _, group_df in grouped:
        sampled_parts.append(group_df.sample(n=1, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=False)

    # Pass 2: add up to min_rows_per_group for each group when possible
    if min_rows_per_group > 1 and len(sampled) < max_rows:
        additional_parts: List[pd.DataFrame] = []
        for _, group_df in grouped:
            already = sampled.index.intersection(group_df.index)
            remaining_group = group_df.drop(index=already)
            extra_target = min_rows_per_group - len(already)
            if extra_target > 0 and not remaining_group.empty:
                n_extra = min(extra_target, len(remaining_group))
                additional_parts.append(
                    remaining_group.sample(n=n_extra, random_state=random_state)
                )

        if additional_parts:
            additional = pd.concat(additional_parts, ignore_index=False)
            remaining_budget = max_rows - len(sampled)
            if len(additional) > remaining_budget:
                additional = additional.sample(n=remaining_budget, random_state=random_state)
            sampled = pd.concat([sampled, additional], ignore_index=False)

    # Pass 3: top up with remaining rows to fill budget
    if len(sampled) < min(max_rows, len(df)):
        remaining = df.drop(index=sampled.index)
        top_up_n = min(max_rows - len(sampled), len(remaining))
        if top_up_n > 0:
            top_up = remaining.sample(n=top_up_n, random_state=random_state)
            sampled = pd.concat([sampled, top_up], ignore_index=False)

    # Deduplicate in case of repeated index collisions and enforce cap.
    sampled = sampled[~sampled.index.duplicated(keep="first")]
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)

    return sampled.sort_index().reset_index(drop=True)


def build_test_dataset_and_config(
    source_config: Path,
    output_config: Path,
    output_data_dir: Path,
    min_rows_per_group: int,
    max_rows_per_dataset: int,
    random_state: int,
) -> Dict[str, Dict[str, int]]:
    with source_config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    source_config_dir = source_config.resolve().parent

    output_data_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, int]] = {}
    source_parents: List[Path] = []
    entries: List[Tuple[str, str, Dict, Path]] = []

    for genotype, drug, drug_cfg in _iter_genotype_drug_entries(cfg):
        if "path" not in drug_cfg:
            raise KeyError(f"Missing 'path' for {genotype}/{drug}")
        src_path = _resolve_source_parquet(drug_cfg["path"], source_config_dir)
        source_parents.append(src_path.parent)
        entries.append((genotype, drug, drug_cfg, src_path))

    source_root = Path(os.path.commonpath([str(p) for p in source_parents]))

    for genotype, drug, drug_cfg, src_path in entries:
        df = pd.read_parquet(src_path)
        sampled = sample_representative_rows(
            df,
            random_state=random_state,
            min_rows_per_group=min_rows_per_group,
            max_rows=max_rows_per_dataset,
        )

        rel_parent = src_path.parent.relative_to(source_root)
        requested_name = Path(drug_cfg["path"]).name
        out_path = output_data_dir / rel_parent / requested_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_parquet(out_path, index=False)

        # Update config to reference repository-local test data.
        drug_cfg["path"] = f"./{out_path.as_posix()}"

        summary[f"{genotype}/{drug}"] = {
            "source_rows": len(df),
            "sampled_rows": len(sampled),
            "columns": len(sampled.columns),
            "grouping_columns_used": len(_sampling_groups(df)),
        }

    cfg.setdefault("metadata", {})
    cfg["metadata"]["output_dir"] = _default_test_output_dir(output_config)

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-config",
        default="2Cohort_TZ_config.json",
        type=Path,
        help="Path to source experiment config JSON.",
    )
    parser.add_argument(
        "--output-config",
        default="2Cohort_TZ_test_config.json",
        type=Path,
        help="Path to write test config JSON.",
    )
    parser.add_argument(
        "--output-data-dir",
        default=Path("test_data/2cohort_tz"),
        type=Path,
        help="Directory where sampled parquet files will be saved.",
    )
    parser.add_argument(
        "--min-rows-per-group",
        type=int,
        default=5,
        help="Target minimum rows sampled from each experimental group.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=250,
        help="Maximum rows in each sampled genotype/drug parquet.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_test_dataset_and_config(
        source_config=args.source_config,
        output_config=args.output_config,
        output_data_dir=args.output_data_dir,
        min_rows_per_group=args.min_rows_per_group,
        max_rows_per_dataset=args.max_rows_per_dataset,
        random_state=args.seed,
    )

    print("Created representative test datasets:")
    for key, stats in summary.items():
        print(
            f"  - {key}: {stats['sampled_rows']}/{stats['source_rows']} rows, "
            f"{stats['columns']} columns, grouping_cols={stats['grouping_columns_used']}"
        )
    print(f"Wrote config: {args.output_config}")


if __name__ == "__main__":
    main()
