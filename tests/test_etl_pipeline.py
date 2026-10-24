
"""Automated tests for the predictive maintenance ETL pipeline."""

from unittest.mock import Mock

import pandas as pd
import pytest

from etl_pipeline import main, runETL, saveAsParquet


def create_nasa_file(file_path):
    """Create a valid NASA-format dataset with 26 columns."""
    rows = []

    for cycle in [1, 2]:
        # Engine ID, cycle, 3 operational settings, and 21 sensor readings.
        values = [1, cycle] + [0.0] * 3 + [1.0] * 21
        rows.append(" ".join(map(str, values)))

    file_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def etl_environment(tmp_path, monkeypatch):
    """Prepare isolated test files and prevent real NASA downloads."""

    # Create the project configuration and source-data directories.
    config_path = tmp_path / "config"
    dataset_path = tmp_path / "data" / "processed"
    config_path.mkdir(parents=True)
    dataset_path.mkdir(parents=True)

    (config_path / "config.yaml").write_text("data:\n  dataset_path: data/processed\n", encoding="utf-8")

    # Generate all four training, test, and reference RUL groups.
    for group in ["FD001", "FD002", "FD003", "FD004"]:
        create_nasa_file(dataset_path / f"train_{group}.txt")
        create_nasa_file(dataset_path / f"test_{group}.txt")
        (dataset_path / f"RUL_{group}.txt").write_text("5\n", encoding="utf-8")

    # Redirect ETL file paths to temporary directories.
    monkeypatch.setattr("etl_pipeline.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("etl_pipeline.RAW_DATA_PATH", tmp_path / "data" / "raw" / "CMAPSSData.zip")
    monkeypatch.setattr("etl_pipeline.PARQUET_PATH", tmp_path / "data" / "curated")

    # Prevent network requests and unnecessary ZIP extraction.
    download_mock = Mock()
    extract_mock = Mock()
    monkeypatch.setattr("etl_pipeline.downloadDataset", download_mock)
    monkeypatch.setattr("etl_pipeline.unZip", extract_mock)

    return dataset_path, download_mock, extract_mock


# ============================================================
# Parquet Storage Tests
# ============================================================

def test_save_as_parquet(tmp_path):
    """Verify datasets are saved as Parquet without DataFrame indexes."""
    df = pd.DataFrame({"unit_number": [1, 1], "time_in_cycles": [1, 2], "RUL": [1, 0]})
    datasets = {"train_FD001": df}
    output_path = tmp_path / "curated" / "train"

    saveAsParquet(datasets, output_path)

    # Confirm the file exists and preserves the original data.
    file_path = output_path / "train_FD001.parquet"
    assert file_path.exists()

    result = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(result, df)


# ============================================================
# ETL Integration Tests
# ============================================================

def test_run_etl_success(etl_environment, tmp_path):
    """Verify the entire ETL pipeline produces eight valid Parquet files."""
    dataset_path, download_mock, extract_mock = etl_environment

    runETL()

    # Verify the pipeline requests the dataset but skips extraction.
    download_mock.assert_called_once()
    extract_mock.assert_not_called()

    # Verify all training and test groups are persisted separately.
    for group in ["FD001", "FD002", "FD003", "FD004"]:
        train_path = tmp_path / "data" / "curated" / "train" / f"train_{group}.parquet"
        test_path = tmp_path / "data" / "curated" / "test" / f"test_{group}.parquet"

        assert train_path.exists()
        assert test_path.exists()

        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        # Verify transformation produced correct RUL labels and 27 columns.
        assert train.shape == (2, 27)
        assert test.shape == (2, 27)
        assert train["RUL"].tolist() == [1, 0]
        assert test["RUL"].tolist() == [6, 5]


def test_run_etl_extracts_missing_files(etl_environment, monkeypatch):
    """Verify missing source files trigger ZIP extraction."""
    dataset_path, download_mock, extract_mock = etl_environment

    # Remove one file to simulate an incomplete extraction.
    missing_file = dataset_path / "train_FD004.txt"
    missing_file.unlink()

    # Simulate extraction restoring the missing file.
    extract_mock.side_effect = lambda zip_path, output_path: create_nasa_file(missing_file)

    runETL()

    extract_mock.assert_called_once()
    assert missing_file.exists()


def test_run_etl_missing_dataset(etl_environment):
    """Verify the pipeline rejects incomplete NASA dataset groups."""
    dataset_path, download_mock, extract_mock = etl_environment

    # Remove one source file and simulate an unsuccessful extraction.
    (dataset_path / "train_FD004.txt").unlink()

    with pytest.raises(ValueError, match="Missing or unexpected"):
        runETL()


def test_run_etl_invalid_dataset(etl_environment):
    """Verify source validation rejects missing sensor measurements."""
    dataset_path, download_mock, extract_mock = etl_environment

    # Replace valid sensor data with a missing measurement.
    values = [1, 1] + [0.0] * 3 + [1.0] * 20 + ["nan"]
    (dataset_path / "train_FD001.txt").write_text(" ".join(map(str, values)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset validation failed"):
        runETL()


# ============================================================
# Pipeline Entry Point Tests
# ============================================================

def test_main_success(monkeypatch):
    """Verify successful execution returns exit code zero."""
    monkeypatch.setattr("etl_pipeline.runETL", Mock())

    assert main() == 0


def test_main_failure(monkeypatch):
    """Verify pipeline exceptions produce a nonzero exit code."""
    monkeypatch.setattr("etl_pipeline.runETL", Mock(side_effect=RuntimeError("Simulated ETL failure")))

    assert main() == 1
