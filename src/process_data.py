
import pandas as pd
from pathlib import Path
from logger import configureLogging, PROJECT_ROOT


# Initialize the shared logger using config/logging.yaml.
# Logs will be saved to Log/process_data.log.
logger = configureLogging()


def loadDataset(filepath: Path, isRulFile: bool) -> pd.DataFrame:
    """
    Load a NASA C-MAPSS dataset into a Pandas DataFrame.

    Args:
        filepath: Path to the dataset text file.
        isRulFile: True if the file contains Remaining Useful Life values.

    Returns:
        A DataFrame containing the dataset with appropriate column names.
    """

    try:
        # Verify that the dataset file exists.
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        logger.info(f"Loading dataset: {filepath.name}")

        if not isRulFile:

            # Define column names for training and testing datasets.
            # Each dataset contains:
            # 1 engine identifier, 1 cycle column,
            # 3 operational settings, and 21 sensor measurements.
            columns = []
            columns += ["unit_number", "time_in_cycles"]
            columns += [f"operational_settings{i}" for i in range(1, 4)]
            columns += [f"sensor_measurement{i}" for i in range(1, 22)]

            # Read the whitespace-separated dataset.
            # header=None indicates that the file has no column headers.
            # names=columns assigns the 26 predefined column names.
            df = pd.read_csv(filepath, sep=r"\s+", header=None, names=columns)

        else:

            # RUL files contain one column representing the remaining
            # useful life for each engine in the corresponding test dataset.
            df = pd.read_csv(filepath, sep=r"\s+", header=None, names=["RUL"])

        # Log the number of rows and columns loaded.
        logger.info(f"Dataset loaded successfully: {filepath.name}, Shape: {df.shape}")

        return df

    except Exception:
        # Log the error and its traceback before propagating it.
        logger.exception(f"Failed to load dataset: {filepath}")
        raise


def processDataset(dataset_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load and organize NASA C-MAPSS datasets.

    Training and testing datasets are stored in one dictionary.
    RUL datasets are stored in a separate dictionary.

    Args:
        dataset_path: Directory containing the extracted dataset files.

    Returns:
        A tuple containing:
        - Dictionary of training and testing DataFrames.
        - Dictionary of RUL DataFrames.
    """

    # Log the beginning of dataset processing.
    logger.info(f"Started processing datasets in: {dataset_path}")

    # Verify that the dataset directory exists.
    if not dataset_path.is_dir():
        logger.error(f"Dataset directory not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # Initialize dictionaries for the datasets.
    df_dict = {}
    df_rul_dict = {}

    # Find all text files in the specified directory.
    files = sorted(dataset_path.glob("*.txt"))

    # Process each dataset file.
    for file in files:

        # Skip the README file because it is not a dataset.
        if file.stem.lower() == "readme":
            continue

        # Determine whether the current file contains RUL values.
        isRulFile = file.stem.startswith("RUL_")

        # Load the dataset into a DataFrame.
        df = loadDataset(file, isRulFile)

        # Use the filename without its extension as the dictionary key.
        filename = file.stem

        if isRulFile:

            # Store RUL datasets separately from training/testing datasets.
            df_rul_dict[filename] = df

        else:

            # Store training and testing datasets.
            df_dict[filename] = df

    # Log the number of datasets successfully processed.
    logger.info(f"Train/Test datasets processed: {len(df_dict)}")
    logger.info(f"RUL datasets processed: {len(df_rul_dict)}")
    logger.info("Dataset processing completed successfully.")

    # Return both dictionaries to the calling function.
    return df_dict, df_rul_dict


# Execute dataset processing only when this script is run directly.
if __name__ == "__main__":

    # Locate the directory containing the extracted datasets.
    dataset_path = PROJECT_ROOT / "data" / "processed"

    # Load all datasets and organize them into dictionaries.
    datasets, rul_datasets = processDataset(dataset_path)

    # Log the total number of datasets loaded.
    logger.info(f"Train/Test datasets: {len(datasets)}")
    logger.info(f"RUL datasets: {len(rul_datasets)}")

    # Log sample dataset shapes to confirm successful loading.
    logger.info(f"RUL_FD001 shape: {rul_datasets['RUL_FD001'].shape}")
    logger.info(f"train_FD001 shape: {datasets['train_FD001'].shape}")
