from fastapi import FastAPI
from src.logger import configureLogging

logger = configureLogging()
app = FastAPI(title="Predictive Maintenance API")


@app.get("/health")
def health_check():
    try:
        logger.info("Health check requested")
        return {"status": "healthy"}
    except Exception:
        logger.exception("Health check failed")
        raise