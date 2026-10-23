import sys
from pathlib import Path
import pandas as pd
from logger import configureLogging, PROJECT_ROOT


# Initialize the logger.
logger = configureLogging()


def validateDataset(df: pd.DataFrame, isRul: bool, dataset_name: str = "dataset") -> bool:
    """
    Validate the structure and quality of a NASA C-MAPSS dataset.

    Args:
        df: DataFrame containing the dataset to validate.
        isRul: True if the dataset contains Remaining Useful Life values.
        dataset_name: Dataset name used in log messages.

    Returns:
        True if all validation checks pass; otherwise False.
    """

    # Log the dataset being validated.
    logger.info(f"Validating dataset: {dataset_name}")

    # Assume the dataset is valid until a check fails.
    is_valid = True

    # Check that the dataset contains at least one row.
    if df.shape[0] == 0:
        logger.error(f"{dataset_name}: Dataset contains no rows.")
        is_valid = False

    # Determine the expected number of columns.
    # RUL datasets have 1 column.
    # Training and testing datasets have 26 columns.
    expected_columns = 1 if isRul else 26

    # Validate the number of columns.
    if df.shape[1] != expected_columns:
        logger.error(f"{dataset_name}: Expected {expected_columns} columns, found {df.shape[1]}.")
        is_valid = False

    # Count missing values in each column.
    missing_values = df.isnull().sum()

    # Log the columns containing missing values.
    if missing_values.sum() > 0:
        logger.warning(f"{dataset_name}: Missing values found: {missing_values[missing_values > 0].to_dict()}")
        is_valid = False

    # Check for duplicate rows in training and testing datasets.
    # RUL datasets are excluded because different engines can
    # legitimately have identical remaining-life values.
    if not isRul:
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            logger.warning(f"{dataset_name}: Found {duplicate_count} duplicate rows.")
            is_valid = False

    # Verify that all columns contain numeric data.
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    # Log any non-numeric columns.
    if non_numeric_columns:
        logger.error(f"{dataset_name}: Non-numeric columns found: {non_numeric_columns}")
        is_valid = False

    # Log the overall validation result.
    if is_valid:
        logger.info(f"{dataset_name}: Validation passed. Shape: {df.shape}")
    else:
        logger.warning(f"{dataset_name}: Validation failed.")

    # Return the validation result.
    return is_valid


def main() -> int:
    """
    Load and validate NASA C-MAPSS datasets.

    Returns:
        0 if all datasets pass validation; otherwise 1.
    """

    # Import the dataset loading function.
    from process_data import loadDataset

    # Locate the dataset directory.
    dataset_path = PROJECT_ROOT / "data" / "processed"

    logger.info(f"Started validating datasets in: {dataset_path}")

    # Verify that the dataset directory exists.
    if not dataset_path.is_dir():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return 1

    # Find all NASA training, testing, and RUL datasets.
    # Exclude unrelated files such as readme.txt.
    files = sorted(dataset_path.glob("*_FD*.txt"))

    # Check whether any dataset files were found.
    if not files:
        logger.error("No NASA dataset files found.")
        return 1

    # Initialize validation counters.
    passed = 0
    failed = 0

    # Load and validate each dataset.
    for file in files:

        # Determine whether the file contains RUL values.
        isRulFile = file.stem.startswith("RUL_")

        try:
            # Load the dataset into a DataFrame.
            df = loadDataset(file, isRulFile)

            # Validate the DataFrame and update the counters.
            if validateDataset(df, isRulFile, file.stem):
                passed += 1
            else:
                failed += 1

        except Exception:
            # Log any exception and its traceback.
            # Continue validating the remaining datasets.
            logger.exception(f"Failed to load or validate: {file.name}")
            failed += 1

    # Log the final validation summary.
    logger.info(f"Validation completed. Passed: {passed}, Failed: {failed}")

    # Return a success or failure exit code.
    return 0 if failed == 0 else 1


# Execute only when this script is run directly.
if __name__ == "__main__":
    sys.exit(main())
