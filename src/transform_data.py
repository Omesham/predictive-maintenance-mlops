
import argparse
from pathlib import Path

import pandas as pd
import yaml

from logger import configureLogging, PROJECT_ROOT
from process_data import processDataset
from validate_data import validateRUL


# Initialize the shared application logger.
logger = configureLogging()


def calculateRUL(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for a training dataset.

    Training engines operate until failure, so RUL is calculated
    by subtracting the current cycle from the final operating cycle.

    Args:
        df: Training dataset containing engine IDs and operating cycles.
        dataset_name: Name of the dataset for logging.

    Returns:
        DataFrame containing the calculated RUL column.
    """

    logger.info(f"Calculating training RUL for {dataset_name}")

    # Create a copy to avoid modifying the original DataFrame.
    df = df.copy()

    # Find the maximum operating cycle for each engine.
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")

    # Calculate the remaining cycles before engine failure.
    df["RUL"] = max_cycles - df["time_in_cycles"]

    logger.info(f"Training RUL calculated successfully for {dataset_name}. Shape: {df.shape}")

    return df


def calculateTestRUL(df_test: pd.DataFrame, df_rul: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Calculate RUL labels for every row in a test dataset.

    Test engines do not operate until failure. Their corresponding RUL
    files provide the remaining life at the last observed cycle.

    Args:
        df_test: Test dataset containing engine measurements and cycles.
        df_rul: Corresponding RUL reference dataset.
        dataset_name: Name of the dataset for logging.

    Returns:
        Test DataFrame containing the calculated RUL column.

    Raises:
        ValueError: If engine IDs do not match the provided RUL labels.
    """

    logger.info(f"Calculating test RUL for {dataset_name}")

    # Preserve the original test DataFrame.
    df_test = df_test.copy()

    # Verify that every test engine has a corresponding RUL label.
    expected_ids = set(range(1, len(df_rul) + 1))
    actual_ids = set(df_test["unit_number"].unique())

    if actual_ids != expected_ids:
        raise ValueError(f"Engine IDs do not match RUL labels for {dataset_name}")

    # Find the last observed operating cycle for each engine.
    max_cycles = df_test.groupby("unit_number")["time_in_cycles"].transform("max")

    # Map RUL reference values to engine IDs using their original row order.
    rul_mapping = pd.Series(df_rul["RUL"].to_numpy(), index=range(1, len(df_rul) + 1))
    provided_rul = df_test["unit_number"].map(rul_mapping)

    # Calculate RUL for every observed cycle of each test engine.
    df_test["RUL"] = max_cycles + provided_rul - df_test["time_in_cycles"]

    logger.info(f"Test RUL calculated successfully for {dataset_name}. Shape: {df_test.shape}")

    return df_test


def transformDatasets(dataset_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Transform and validate all NASA C-MAPSS training and test datasets.

    Args:
        dataset_path: Directory containing the extracted datasets.

    Returns:
        A tuple containing the transformed training and test dictionaries.

    Raises:
        ValueError: If RUL validation fails.
    """

    logger.info(f"Started dataset transformation from: {dataset_path}")

    # Load training, test, and RUL reference datasets.
    datasets_dict, rul_datasets_dict = processDataset(dataset_path)

    # Store transformed training and test datasets separately.
    train_datasets = {}
    test_datasets = {}

    for dataset_name, df in datasets_dict.items():

        if dataset_name.startswith("train_"):

            # Generate training RUL labels.
            df = calculateRUL(df, dataset_name)

            # Validate the generated RUL labels.
            if not validateRUL(df):
                raise ValueError(f"Training RUL validation failed for {dataset_name}")

            train_datasets[dataset_name] = df

        elif dataset_name.startswith("test_"):

            # Find the corresponding RUL reference dataset.
            rul_key = dataset_name.replace("test_", "RUL_")
            df_rul = rul_datasets_dict[rul_key]

            # Generate test RUL labels.
            df = calculateTestRUL(df, df_rul, dataset_name)

            # Validate the test labels against the RUL reference data.
            if not validateRUL(df, isTest=True, df_rul=df_rul):
                raise ValueError(f"Test RUL validation failed for {dataset_name}")

            test_datasets[dataset_name] = df

        else:
            # Reject unsupported datasets instead of processing them incorrectly.
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info(f"Transformation completed. Training: {len(train_datasets)}, Testing: {len(test_datasets)}")

    return train_datasets, test_datasets


def main() -> int:
    """
    Run dataset transformation from the command line.

    The command-line dataset path overrides the default path in config.yaml.

    Returns:
        0 on success, or 1 if transformation fails.
    """

    try:
        # Configure optional command-line arguments.
        parser = argparse.ArgumentParser(description="Transform NASA C-MAPSS datasets")
        parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset directory")
        args = parser.parse_args()

        # Load the default dataset path from the application configuration.
        config_path = PROJECT_ROOT / "config" / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Prefer the command-line argument when provided.
        dataset_path = args.dataset_path if args.dataset_path else config["data"]["dataset_path"]
        dataset_path = Path(dataset_path)

        # Resolve relative paths against the project root.
        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path

        # Run transformations and receive both dictionaries.
        train_datasets, test_datasets = transformDatasets(dataset_path)

        # Report successful processing.
        logger.info(f"Training datasets ready: {len(train_datasets)}")
        logger.info(f"Test datasets ready: {len(test_datasets)}")

        return 0

    except Exception:
        # Record the error and traceback, then signal failure.
        logger.exception("Dataset transformation failed")
        return 1


# Run only when executed directly, not when imported by the ETL pipeline.
if __name__ == "__main__":
    raise SystemExit(main())
