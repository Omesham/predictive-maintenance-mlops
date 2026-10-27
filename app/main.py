from fastapi import FastAPI
from src.logger import configureLogging
from app.core.exception_handlers import register_exception_handlers
from fastapi.staticfiles import StaticFiles

logger = configureLogging()
app = FastAPI(title="Predictive Maintenance API")
app.mount("/styles", StaticFiles(directory="app/styles"), name="styles")

register_exception_handlers(app)


@app.get("/health")
def health_check():
    try:
        logger.info("Health check requested")
        return {"status": "healthy"}
    except Exception:
        logger.exception("Health check failed")
        raise