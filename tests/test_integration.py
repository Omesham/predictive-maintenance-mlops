
"""Integration tests for the NASA C-MAPSS data pipeline."""

import pandas as pd
import pytest

from process_data import processDataset
from transform_data import transformDatasets
from validate_data import validateRUL


def create_nasa_file(file_path, engine_cycles):
    """Create a small NASA-format dataset containing all 26 input columns."""
    rows = []

    for engine, cycle in engine_cycles:
        # Engine ID + cycle + 3 operational settings + 21 sensor measurements.
        values = [engine, cycle] + [0.0] * 3 + [1.0] * 21
        rows.append(" ".join(map(str, values)))

    file_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_data_loading_and_transformation(tmp_path):
    """Verify real file loading, RUL calculation, and validation work together."""

    # Create synthetic NASA-format training and test files.
    create_nasa_file(tmp_path / "train_FD001.txt", [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)])
    create_nasa_file(tmp_path / "test_FD001.txt", [(1, 1), (1, 2), (2, 1), (2, 2)])
    (tmp_path / "RUL_FD001.txt").write_text("5\n3\n", encoding="utf-8")

    # Load actual files using our existing processing function.
    datasets, rul_datasets = processDataset(tmp_path)

    assert len(datasets) == 2
    assert len(rul_datasets) == 1
    assert datasets["train_FD001"].shape == (5, 26)
    assert datasets["test_FD001"].shape == (4, 26)

    # Transform both datasets using the actual transformation pipeline.
    train_datasets, test_datasets = transformDatasets(tmp_path)
    train = train_datasets["train_FD001"]
    test = test_datasets["test_FD001"]

    # Verify RUL calculations for both engines.
    assert train["RUL"].tolist() == [2, 1, 0, 1, 0]
    assert test["RUL"].tolist() == [6, 5, 4, 3]

    # Verify the transformed datasets contain the expected 27 columns.
    assert train.shape == (5, 27)
    assert test.shape == (4, 27)

    # Verify the resulting RUL labels pass our real validation function.
    assert validateRUL(train) is True
    assert validateRUL(test, isTest=True, df_rul=rul_datasets["RUL_FD001"]) is True


def test_missing_rul_file(tmp_path):
    """Verify transformation fails when a test dataset has no corresponding RUL file."""

    create_nasa_file(tmp_path / "test_FD001.txt", [(1, 1), (1, 2)])

    with pytest.raises(KeyError):
        transformDatasets(tmp_path)


def test_invalid_rul_file(tmp_path):
    """Verify the pipeline rejects negative remaining useful life."""

    create_nasa_file(tmp_path / "test_FD001.txt", [(1, 1), (1, 2)])
    (tmp_path / "RUL_FD001.txt").write_text("-5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="validation failed"):
        transformDatasets(tmp_path)
