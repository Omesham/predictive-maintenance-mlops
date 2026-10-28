import logging
from sqlalchemy import text

from src.database import getEngine


class DatabaseLogHandler(logging.Handler):
    """Store application log records in PostgreSQL."""
    
    def __init__(self):
        super().__init__()

        # Create the database engine once, not for every log message.
        self.engine = getEngine()
           
    def emit(self, record):
        try:
            # Format the log message, including exception details.
            message = self.format(record)

            # Open a short database transaction for the log entry.
            with self.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO application_logs
                            (level, logger_name, message)
                        VALUES
                            (:level, :logger_name, :message)
                    """),
                    {
                        "level": record.levelname,
                        "logger_name": record.name,
                        "message": message
                    }
                )

        except Exception:
            # Logging failures must not interrupt application requests.
            self.handleError(record)