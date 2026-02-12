import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_test_dataset_from_config import sample_representative_rows


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
