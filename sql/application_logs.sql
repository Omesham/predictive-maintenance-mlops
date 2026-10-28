-- Store application logs with automatic timestamps.
CREATE TABLE IF NOT EXISTS application_logs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(20) NOT NULL,
    logger_name VARCHAR(255) NOT NULL,
    message TEXT NOT NULL
);

-- Speed up queries that filter logs by timestamp.
CREATE INDEX IF NOT EXISTS idx_application_logs_created_at
ON application_logs (created_at);