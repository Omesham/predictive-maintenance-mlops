from pathlib import Path
import pandas as pd

from database import getEngine
from logger import configureLogging, PROJECT_ROOT

from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert

logger = configureLogging()
def prepareDataset(file_path: Path) -> pd.DataFrame:
    """Read a Parquet dataset and prepare its columns for PostgreSQL."""

    # Load the transformed dataset.
    df = pd.read_parquet(file_path)

    # Extract dataset information from the filename, e.g., train_FD001.
    dataset_split, dataset_group = file_path.stem.split("_")

    # Add columns identifying the dataset and its split.
    df["dataset_group"] = dataset_group
    df["dataset_split"] = dataset_split

    # Match the PostgreSQL column name.
    df = df.rename(columns={"RUL": "rul"})

    logger.info(f"Prepared {file_path.name} for PostgreSQL. Rows: {len(df)}")

    return df


def loadToPostgres(df: pd.DataFrame, engine, batch_size: int = 1000) -> int:
    """Insert engine records into PostgreSQL in batches, skipping duplicates."""

    # Load the existing PostgreSQL table definition.
    table = Table("engine_cycles", MetaData(), autoload_with=engine)
    inserted_rows = 0

    # Use one transaction so failures roll back the entire dataset load.
    with engine.begin() as connection:
        for start in range(0, len(df), batch_size):
            # Convert a batch of DataFrame rows into database records.
            records = df.iloc[start:start + batch_size].to_dict("records")

            # Return one engine ID for each successfully inserted record.
            statement = insert(table).values(records).on_conflict_do_nothing().returning(table.c.unit_number)
            inserted_rows += len(connection.execute(statement).fetchall())

    logger.info(f"PostgreSQL load completed. Inserted {inserted_rows} of {len(df)} rows")
    return inserted_rows


def loadAllDatasets() -> int:
    """Load all eight transformed NASA datasets into PostgreSQL."""

    # Locate the curated Parquet datasets.
    parquet_path = PROJECT_ROOT / "data" / "curated"
    total_inserted = 0

    # Build the expected file paths for all training and test datasets.
    files = [parquet_path / split / f"{split}_FD{i:03d}.parquet" for split in ("train", "test") for i in range(1, 5)]

    # Verify all eight files exist before starting the database load.
    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing Parquet dataset: {file_path}")

    logger.info("Starting PostgreSQL data loading")

    # Create one database engine and reuse it across all datasets.
    engine = getEngine()

    try:
        for file_path in files:
            df = prepareDataset(file_path)
            total_inserted += loadToPostgres(df, engine)
    finally:
        # Release database connections even if loading fails.
        engine.dispose()

    logger.info(f"All datasets processed. Total records inserted: {total_inserted}")
    return total_inserted

def main() -> int:
    """Execute PostgreSQL data loading and handle errors."""
    try:
        total_inserted = loadAllDatasets()
        logger.info(f"Database loading completed. Inserted {total_inserted} records")
        return 0
    except Exception:
        logger.exception("Database loading failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())