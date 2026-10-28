import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use the existing logger without configuring it again.
logger = logging.getLogger("predictive_maintenance")


def getEngine():
    """Create a PostgreSQL engine using environment variables."""

    # Load local credentials. Production can use environment variables.
    # Use local .env credentials during development and testing.
    # Preserve deployment environment variables in production.
    load_dotenv(PROJECT_ROOT / ".env", override=os.getenv("APP_ENV", "development").lower() != "production")

    username = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB")

    # Allow database location to differ between development and AWS.
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("POSTGRES_PORT", "5432"))

    if not all([username, password, database]):
        raise ValueError("Missing PostgreSQL environment variables")

    # Construct the connection URL without exposing credentials.
    url = URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database
    )

    # Create a connection pool and check stale connections before reuse.
    engine = create_engine(url, pool_pre_ping=True)

    logger.info("PostgreSQL engine initialized")
    return engine