import pandas as pd
from load_database import loadToPostgres
import pytest
from sqlalchemy.exc import IntegrityError
import etl_pipeline


"""Integration tests for PostgreSQL database connections."""

from sqlalchemy import text
from database import getEngine


def test_database_connection(monkeypatch):
    """Verify Python connects to the test database, not production."""

    # Override the database name for this test only.
    monkeypatch.setenv("POSTGRES_DB", "predictive_maintenance_test")

    engine = getEngine()

    try:
        # Ask PostgreSQL which database we connected to.
        with engine.connect() as connection:
            database_name = connection.execute(text("SELECT current_database()")).scalar_one()

        assert database_name == "predictive_maintenance_test"
    finally:
        engine.dispose()
        
        
def test_insert_and_prevent_duplicates(monkeypatch):
    """Verify records are inserted once, even when loaded repeatedly."""

    # Connect only to our test database.
    monkeypatch.setenv("POSTGRES_DB", "predictive_maintenance_test")
    engine = getEngine()

    # Create two synthetic engine-cycle records.
    df = pd.DataFrame({
        "dataset_group": ["FD001", "FD001"],
        "dataset_split": ["train", "train"],
        "unit_number": [999001, 999001],
        "time_in_cycles": [1, 2],
        "rul": [1, 0]
    })

    try:
        # Verify the database before modifying any records.
        with engine.connect() as connection:
            assert connection.execute(text("SELECT current_database()")).scalar_one() == "predictive_maintenance_test"

        # Remove any records left by an earlier test run.
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM engine_cycles WHERE unit_number = 999001"))

        # First load should insert both records.
        assert loadToPostgres(df, engine) == 2

        # Second load should skip both duplicates.
        assert loadToPostgres(df, engine) == 0

        # Confirm PostgreSQL contains exactly two records.
        with engine.connect() as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM engine_cycles WHERE unit_number = 999001")).scalar_one()

        assert count == 2

    finally:
        # Clean up our synthetic records after testing.
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM engine_cycles WHERE unit_number = 999001"))

        engine.dispose()
        
        
        
def test_transaction_rollback(monkeypatch):
    """Verify a failed insert rolls back earlier inserts in the same transaction."""

    # Connect to our isolated test database.
    monkeypatch.setenv("POSTGRES_DB", "predictive_maintenance_test")
    engine = getEngine()

    # Second record violates the database constraint: RUL cannot be negative.
    df = pd.DataFrame({
        "dataset_group": ["FD001", "FD001"],
        "dataset_split": ["train", "train"],
        "unit_number": [999002, 999002],
        "time_in_cycles": [1, 2],
        "rul": [1, -1]
    })

    try:
        # Ensure no records remain from previous test runs.
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM engine_cycles WHERE unit_number = 999002"))

        # Insert one row per batch; the second must fail.
        with pytest.raises(IntegrityError):
            loadToPostgres(df, engine, batch_size=1)

        # The first valid row should also have been rolled back.
        with engine.connect() as connection:
            count = connection.execute(text("SELECT COUNT(*) FROM engine_cycles WHERE unit_number = 999002")).scalar_one()

        assert count == 0
    finally:
        engine.dispose()