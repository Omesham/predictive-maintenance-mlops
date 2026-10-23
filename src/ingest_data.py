
import requests
import zipfile
from pathlib import Path
from logger import configureLogging, PROJECT_ROOT


# Initialize the shared logger using config/logging.yaml.
# Log messages will be saved to Log/ingest_data.log.
logger = configureLogging()


def downloadDataset(url: str, output_path: Path) -> None:
    """
    Download the NASA C-MAPSS dataset from the specified URL.

    Args:
        url: URL of the dataset to download.
        output_path: Destination path for the downloaded ZIP file.

    Returns:
        None.
    """

    try:
        # Check whether the dataset has already been downloaded.
        if not output_path.exists():

            logger.info(f"Starting dataset download from: {url}")

            # Create the destination directory and its parent folders if needed.
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Send an HTTP GET request with a 120-second timeout.
            response = requests.get(url, timeout=120)

            # Raise an exception if the HTTP request was unsuccessful.
            response.raise_for_status()

            # Retrieve the downloaded file contents.
            data = response.content

            # Write the downloaded bytes to the destination ZIP file.
            output_path.write_bytes(data)

            logger.info(f"Dataset downloaded successfully to: {output_path}")

        else:
            # Skip downloading if the ZIP file already exists.
            logger.info(f"Dataset already exists. Skipping download: {output_path}")

    except Exception:
        # Record the exception and its traceback before re-raising.
        logger.exception(f"Failed to download dataset from: {url}")
        raise


def unZip(zipPath: Path, output_path: Path) -> None:
    """
    Extract the downloaded NASA C-MAPSS ZIP archive.

    Args:
        zipPath: Path to the ZIP archive.
        output_path: Destination directory for extracted files.

    Returns:
        None.
    """

    try:
        logger.info(f"Starting dataset extraction from: {zipPath}")

        # Create the extraction directory if it does not exist.
        output_path.mkdir(parents=True, exist_ok=True)

        # Open the ZIP archive and extract all its files.
        with zipfile.ZipFile(zipPath) as zipFile:
            zipFile.extractall(output_path)

        logger.info(f"Dataset extracted successfully to: {output_path}")

    except zipfile.BadZipFile:
        # Log an error if the downloaded archive is invalid.
        logger.exception(f"Invalid ZIP archive: {zipPath}")
        raise

    except Exception:
        # Log any other error encountered during extraction.
        logger.exception(f"Failed to extract dataset: {zipPath}")
        raise


# Execute ingestion only when this script is run directly.
if __name__ == "__main__":

    # NASA C-MAPSS dataset download URL.
    dataset_url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"

    # Define the destination for the downloaded ZIP archive.
    output_path_zip = PROJECT_ROOT / "data" / "raw" / "CMAPSSData.zip"

    # Define the destination for the extracted dataset files.
    output_path_unzip = PROJECT_ROOT / "data" / "processed"

    # Download the ZIP archive if it does not already exist.
    downloadDataset(dataset_url, output_path_zip)

    # Extract the dataset files.
    unZip(output_path_zip, output_path_unzip)
