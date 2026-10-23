
import logging
import logging.config
import sys
from pathlib import Path

import pandas as pd
import yaml


# Get the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Locate the YAML logging configuration file.
LOG_CONFIG_PATH = PROJECT_ROOT / "config" / "logging.yaml"

# Create the Log directory if it does not exist.
LOG_DIRECTORY = PROJECT_ROOT / "Log"
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)


def configureLogging() -> logging.Logger:
    """
    Configure application logging using logging.yaml.

    The log filename is based on the Python script being executed.

    Returns:
        Configured application logger.
    """

    # Check that the YAML configuration file exists.
    if not LOG_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Logging configuration not found: {LOG_CONFIG_PATH}")

    # Read the YAML logging configuration.
    with open(LOG_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Determine the name of the Python script being executed.
    # Example: process_data.py becomes process_data.
    script_name = Path(sys.argv[0]).stem

    # Override the default log filename from logging.yaml.
    # This ensures logs are saved in the project root Log directory.
    config["handlers"]["file"]["filename"] = str(LOG_DIRECTORY / f"{script_name}.log")

    # Apply the logging configuration.
    logging.config.dictConfig(config)

    # Return the application logger defined in logging.yaml.
    return logging.getLogger("predictive_maintenance")