import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_test_dataset_from_config import (
    build_test_dataset_and_config,
    sample_representative_rows,
)


def test_sample_representative_rows_preserves_group_coverage_and_columns():
    df = pd.DataFrame(
        {
            "Dilut": ["DMSO", "DMSO", "1uM", "1uM", "3uM", "3uM"],
            "group": ["CTRL", "CTRL", "P1", "P1", "CTRL", "P1"],
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [10, 20, 30, 40, 50, 60],
        }
    )

    sampled = sample_representative_rows(
        df,
        random_state=1,
        min_rows_per_group=1,
        max_rows=6,
    )

    assert set(sampled.columns) == set(df.columns)

    source_groups = set(tuple(x) for x in df[["Dilut", "group"]].drop_duplicates().values)
    sampled_groups = set(
        tuple(x) for x in sampled[["Dilut", "group"]].drop_duplicates().values
    )
    assert sampled_groups == source_groups


def test_sample_representative_rows_obeys_max_rows_without_group_cols():
    df = pd.DataFrame({"feature_a": list(range(100)), "feature_b": list(range(100, 200))})

    sampled = sample_representative_rows(
        df,
        random_state=7,
        min_rows_per_group=2,
        max_rows=11,
    )

    assert len(sampled) == 11
    assert set(sampled.columns) == set(df.columns)


def test_sample_representative_rows_raises_when_groups_exceed_max_rows():
    df = pd.DataFrame(
        {
            "Dilut": [f"{i}uM" for i in range(12)],
            "feature": list(range(12)),
        }
    )

    with pytest.raises(ValueError, match="max_rows is too small"):
        sample_representative_rows(df, random_state=1, min_rows_per_group=1, max_rows=8)


def test_build_test_dataset_uses_numeric_parquet_fallback(tmp_path: Path):
    src_dir = tmp_path / "data"
    src_dir.mkdir()

    df = pd.DataFrame(
        {
            "Dilut": ["DMSO", "1uM", "3uM"],
            "feature_a": [1.0, 2.0, 3.0],
        }
    )
    numeric_path = src_dir / "single_cell_features_numeric.parquet"
    df.to_parquet(numeric_path, index=False)

    cfg = {
        "metadata": {"output_dir": "./old"},
        "genotypes": {
            "G1": {
                "drugs": {
                    "Talazoparib": {
                        "path": str(src_dir / "single_cell_features.parquet"),
                        "ec50_um": 1.0,
                    }
                }
            }
        },
    }
    src_cfg = tmp_path / "source_config.json"
    out_cfg = tmp_path / "pilot_test_config.json"
    out_data = tmp_path / "test_data"
    src_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    summary = build_test_dataset_and_config(
        source_config=src_cfg,
        output_config=out_cfg,
        output_data_dir=out_data,
        min_rows_per_group=1,
        max_rows_per_dataset=10,
        random_state=0,
    )

    assert "G1/Talazoparib" in summary
    out_loaded = json.loads(out_cfg.read_text(encoding="utf-8"))
    assert out_loaded["metadata"]["output_dir"] == "./test_output_pilot_test"
    assert (
        out_loaded["genotypes"]["G1"]["drugs"]["Talazoparib"]["path"]
        == "./" + (out_data / "single_cell_features.parquet").as_posix()
    )


def test_build_test_dataset_preserves_nested_input_folder_structure(tmp_path: Path):
    src_root = tmp_path / "source"
    nested_dir_a = src_root / "batch_a" / "WT" / "Etoposide"
    nested_dir_b = src_root / "batch_b" / "WT" / "Talazoparib"
    nested_dir_a.mkdir(parents=True)
    nested_dir_b.mkdir(parents=True)
    src_parquet_a = nested_dir_a / "single_cell_features.parquet"
    src_parquet_b = nested_dir_b / "single_cell_features.parquet"
    pd.DataFrame({"Dilut": ["DMSO", "1uM"], "feature_a": [1.0, 2.0]}).to_parquet(src_parquet_a, index=False)
    pd.DataFrame({"Dilut": ["DMSO", "3uM"], "feature_a": [3.0, 4.0]}).to_parquet(src_parquet_b, index=False)

    cfg = {
        "metadata": {"output_dir": "./old"},
        "genotypes": {
            "WT": {
                "drugs": {
                    "Etoposide": {"path": str(src_parquet_a), "ec50_um": 1.0},
                    "Talazoparib": {"path": str(src_parquet_b), "ec50_um": 2.0},
                }
            }
        },
    }
    src_cfg = tmp_path / "source_config.json"
    out_cfg = tmp_path / "test_config.json"
    out_data = tmp_path / "test_data"
    src_cfg.write_text(json.dumps(cfg), encoding="utf-8")

    build_test_dataset_and_config(
        source_config=src_cfg,
        output_config=out_cfg,
        output_data_dir=out_data,
        min_rows_per_group=1,
        max_rows_per_dataset=10,
        random_state=0,
    )

    expected_output_a = out_data / "batch_a" / "WT" / "Etoposide" / "single_cell_features.parquet"
    expected_output_b = out_data / "batch_b" / "WT" / "Talazoparib" / "single_cell_features.parquet"
    assert expected_output_a.exists()
    assert expected_output_b.exists()
    out_loaded = json.loads(out_cfg.read_text(encoding="utf-8"))
    assert (
        out_loaded["genotypes"]["WT"]["drugs"]["Etoposide"]["path"]
        == "./" + expected_output_a.as_posix()
    )
    assert (
        out_loaded["genotypes"]["WT"]["drugs"]["Talazoparib"]["path"]
        == "./" + expected_output_b.as_posix()
    )
