import logging
import uuid
from sqlalchemy import text

from src.database import getEngine
from src.database_log_handler import DatabaseLogHandler


def test_database_log_handler():
    """Verify that application logs are saved to PostgreSQL."""

    # Create the database handler using our test database.
    handler = DatabaseLogHandler()

    # Create an isolated logger to avoid affecting application logging.
    logger = logging.getLogger("test_database_logging")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        # Write a uniquely identifiable test log.        

        message = f"Database logging test {uuid.uuid4()}"
        logger.info(message)

        # Verify that PostgreSQL contains the log entry.
        with handler.engine.connect() as connection:
            result = connection.execute(
                text("""
                    SELECT level, logger_name, message, created_at
                    FROM application_logs
                    WHERE message = :message
                """),
                {"message": message}
            ).fetchone()

        assert result is not None
        assert result.level == "INFO"
        assert result.logger_name == "test_database_logging"
        assert result.message == message
        assert result.created_at is not None

    finally:
        # Remove the handler so other tests aren't affected.
        logger.removeHandler(handler)
        handler.close()
        handler.engine.dispose()