import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from logger import configureLogging, PROJECT_ROOT

logger= configureLogging()

def getEngine():
    """Create a PostgreSQL database engine using environment variables."""
    logger.info("Creating database engine...")
    # Load database credentials from the project's .env file.
    load_dotenv(PROJECT_ROOT / ".env")

    # Retrieve the database connection settings.
    username = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    database = os.getenv("POSTGRES_DB")
    
    # Verify that all required database credentials are available.
    if not all([username, password, database]):
        raise ValueError("Missing PostgreSQL environment variables")

    # Build the database URL without exposing credentials in logs.
    url = URL.create(
        drivername="postgresql+psycopg",
        username=username,
        password=password,
        host="localhost",
        port=5432,
        database=database
    )

    logger.info("PostgreSQL connection URL configured")
    # Create the SQLAlchemy engine.
    engine = create_engine(url, pool_pre_ping=True)

    # Verify that PostgreSQL is reachable.
    try:
        with engine.connect() as connection:
            logger.info("Successfully connected to PostgreSQL")
    except Exception:
        logger.exception("PostgreSQL connection failed")
        engine.dispose()
        raise

    return engine