
"""Orchestrate the NASA C-MAPSS data engineering ETL pipeline."""

from pathlib import Path

import pandas as pd
import yaml

from ingest_data import downloadDataset, unZip
from logger import configureLogging, PROJECT_ROOT
from process_data import processDataset
from transform_data import transformDatasets
from validate_data import validateDataset
from load_database import loadAllDatasets

logger = configureLogging()

DATASET_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "CMAPSSData.zip"
PARQUET_PATH = PROJECT_ROOT / "data" / "curated"


def saveAsParquet(datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Save transformed datasets as separate Parquet files."""
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_name, df in datasets.items():
        file_path = output_path / f"{dataset_name}.parquet"

        # Preserve the original DataFrame without writing its index.
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {dataset_name} to {file_path}")


def runETL() -> None:
    """Run data ingestion, validation, transformation, and persistence."""
    logger.info("Starting predictive maintenance ETL pipeline")

    # Load the existing project configuration.
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_path = Path(config["data"]["dataset_path"])

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    # Download the original NASA dataset.
    downloadDataset(DATASET_URL, RAW_DATA_PATH)

    # Extract the dataset if any required source file is missing.
    dataset_groups = [f"FD{i:03d}" for i in range(1, 5)]
    required_files = [dataset_path / f"{prefix}_{group}.txt" for group in dataset_groups for prefix in ("train", "test", "RUL")]

    if not all(file_path.exists() for file_path in required_files):
        unZip(RAW_DATA_PATH, dataset_path)

    # Load the raw training, test, and reference RUL datasets.
    datasets, rul_datasets = processDataset(dataset_path)

    # Ensure that all four dataset groups are present.
    expected_datasets = {f"{prefix}_{group}" for group in dataset_groups for prefix in ("train", "test")}
    expected_rul = {f"RUL_{group}" for group in dataset_groups}

    if set(datasets) != expected_datasets or set(rul_datasets) != expected_rul:
        raise ValueError("Missing or unexpected NASA C-MAPSS datasets")

    # Validate source datasets before transformation.
    for dataset_name, df in datasets.items():
        if not validateDataset(df, False, dataset_name):
            raise ValueError(f"Dataset validation failed: {dataset_name}")

    for dataset_name, df in rul_datasets.items():
        if not validateDataset(df, True, dataset_name):
            raise ValueError(f"RUL dataset validation failed: {dataset_name}")

    logger.info("All source datasets passed validation")

    # Calculate RUL labels and validate transformed datasets.
    train_datasets, test_datasets = transformDatasets(dataset_path)

    # Store training and test datasets separately in Parquet format.
    saveAsParquet(train_datasets, PARQUET_PATH / "train")
    saveAsParquet(test_datasets, PARQUET_PATH / "test")

    logger.info(f"ETL completed successfully. Training: {len(train_datasets)}, Testing: {len(test_datasets)}")
    
    # Load curated datasets into PostgreSQL after Parquet persistence succeeds.
    inserted_rows = loadAllDatasets()
    logger.info(f"PostgreSQL loading completed. New records inserted: {inserted_rows}")


def main() -> int:
    """Execute the ETL pipeline and report failures."""
    try:
        runETL()
        return 0
    except Exception:
        logger.exception("ETL pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
