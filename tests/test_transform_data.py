
import pandas as pd
import pytest

from transform_data import calculateRUL, calculateTestRUL, transformDatasets
from validate_data import validateRUL


# ============================================================
# Training RUL Tests
# ============================================================

def test_calculate_rul():
    """Verify RUL decreases to zero at the engine's final cycle."""
    df = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3]})
    result = calculateRUL(df, "test_sample")

    assert result["RUL"].tolist() == [2, 1, 0]


def test_calculate_rul_multiple_engines():
    """Verify each engine's RUL is calculated independently."""
    df = pd.DataFrame({"unit_number": [1, 1, 2, 2, 2], "time_in_cycles": [1, 2, 1, 2, 3]})
    result = calculateRUL(df, "test_sample")

    assert result["RUL"].tolist() == [1, 0, 2, 1, 0]


def test_calculate_rul_unsorted_cycles():
    """Verify RUL calculation does not depend on row order."""
    df = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [3, 1, 2]})
    result = calculateRUL(df, "test_sample")

    assert result["RUL"].tolist() == [0, 2, 1]


def test_calculate_rul_does_not_modify_input():
    """Verify transformation preserves the original DataFrame."""
    df = pd.DataFrame({"unit_number": [1, 1], "time_in_cycles": [1, 2]})
    original = df.copy(deep=True)

    calculateRUL(df, "test_sample")

    pd.testing.assert_frame_equal(df, original)


# ============================================================
# Test RUL Calculation Tests
# ============================================================

def test_calculate_test_rul():
    """Verify NASA's provided final RUL is applied to earlier cycles."""
    df_test = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3]})
    df_rul = pd.DataFrame({"RUL": [5]})

    result = calculateTestRUL(df_test, df_rul, "test_sample")

    assert result["RUL"].tolist() == [7, 6, 5]


def test_calculate_test_rul_multiple_engines():
    """Verify NASA's RUL labels map to their corresponding engines."""
    df_test = pd.DataFrame({"unit_number": [1, 1, 2, 2], "time_in_cycles": [1, 2, 1, 2]})
    df_rul = pd.DataFrame({"RUL": [5, 3]})

    result = calculateTestRUL(df_test, df_rul, "test_sample")

    assert result["RUL"].tolist() == [6, 5, 4, 3]


def test_calculate_test_rul_invalid_engine_ids():
    """Verify unexpected engine IDs raise a ValueError."""
    df_test = pd.DataFrame({"unit_number": [2, 2, 2], "time_in_cycles": [1, 2, 3]})
    df_rul = pd.DataFrame({"RUL": [5]})

    with pytest.raises(ValueError, match="Engine IDs"):
        calculateTestRUL(df_test, df_rul, "test_sample")


def test_calculate_test_rul_does_not_modify_input():
    """Verify test RUL calculation does not mutate its input."""
    df_test = pd.DataFrame({"unit_number": [1, 1], "time_in_cycles": [1, 2]})
    df_rul = pd.DataFrame({"RUL": [5]})
    original = df_test.copy(deep=True)

    calculateTestRUL(df_test, df_rul, "test_sample")

    pd.testing.assert_frame_equal(df_test, original)


# ============================================================
# RUL Validation Tests
# ============================================================

def test_validate_training_rul():
    """Accept valid training RUL values ending at zero."""
    df = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3], "RUL": [2, 1, 0]})

    assert validateRUL(df) is True


def test_validate_training_rul_invalid_final_value():
    """Reject training data whose final RUL is not zero."""
    df = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3], "RUL": [2, 1, 1]})

    assert validateRUL(df) is False


def test_validate_rul_rejects_negative_values():
    """Reject physically invalid negative RUL values."""
    df = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3], "RUL": [2, -1, 0]})

    assert validateRUL(df) is False


def test_validate_rul_rejects_missing_values():
    """Reject datasets containing missing RUL labels."""
    df = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3], "RUL": [2, None, 0]})

    assert validateRUL(df) is False


def test_validate_test_rul():
    """Accept test RUL values matching NASA's reference labels."""
    df_test = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3], "RUL": [7, 6, 5]})
    df_rul = pd.DataFrame({"RUL": [5]})

    assert validateRUL(df_test, isTest=True, df_rul=df_rul) is True


def test_validate_test_rul_incorrect_final_value():
    """Reject test data whose final RUL disagrees with NASA's labels."""
    df_test = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3], "RUL": [7, 6, 4]})
    df_rul = pd.DataFrame({"RUL": [5]})

    assert validateRUL(df_test, isTest=True, df_rul=df_rul) is False


# ============================================================
# Transformation Workflow Tests
# ============================================================

def test_transform_datasets(monkeypatch, tmp_path):
    """Verify the pipeline transforms and validates both dataset types."""
    df_train = pd.DataFrame({"unit_number": [1, 1, 1], "time_in_cycles": [1, 2, 3]})
    df_test = pd.DataFrame({"unit_number": [1, 1], "time_in_cycles": [1, 2]})
    df_rul = pd.DataFrame({"RUL": [5]})

    # Replace file loading with synthetic datasets to keep this a unit test.
    def mock_process_dataset(dataset_path):
        datasets = {"train_FD001": df_train, "test_FD001": df_test}
        rul_datasets = {"RUL_FD001": df_rul}
        return datasets, rul_datasets

    monkeypatch.setattr("transform_data.processDataset", mock_process_dataset)

    train_datasets, test_datasets = transformDatasets(tmp_path)

    # Check that the expected datasets and RUL values are returned.
    assert len(train_datasets) == 1
    assert len(test_datasets) == 1
    assert train_datasets["train_FD001"]["RUL"].tolist() == [2, 1, 0]
    assert test_datasets["test_FD001"]["RUL"].tolist() == [6, 5]


def test_transform_datasets_missing_rul(monkeypatch, tmp_path):
    """Verify the pipeline fails when a test dataset lacks RUL labels."""
    df_test = pd.DataFrame({"unit_number": [1, 1], "time_in_cycles": [1, 2]})

    # Simulate a dataset loader that cannot find the corresponding RUL file.
    def mock_process_dataset(dataset_path):
        return {"test_FD001": df_test}, {}

    monkeypatch.setattr("transform_data.processDataset", mock_process_dataset)

    with pytest.raises(KeyError):
        transformDatasets(tmp_path)
