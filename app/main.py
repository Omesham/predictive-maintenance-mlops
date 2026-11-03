from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.logger import configureLogging
from app.core.exception_handlers import register_exception_handlers
from app.api import predict, upload

# Initialize the application and configure logging.
logger = configureLogging()
app = FastAPI(title="Predictive Maintenance API")
# Collect API request metrics and expose them to Prometheus.
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
# Serve the dashboard's CSS and JavaScript files.
app.mount("/styles", StaticFiles(directory="app/styles"), name="styles")
app.mount("/scripts", StaticFiles(directory="app/scripts"), name="scripts")

# Register centralized HTTP exception handlers.
register_exception_handlers(app)

# Register API routes for model predictions and CSV uploads.
app.include_router(predict.router, prefix="/api", tags=["Predictions"])
app.include_router(upload.router, prefix="/api", tags=["Uploads"])


@app.get("/health")
def health_check():
    # Confirm the API is running.
    logger.info("Health check requested")
    return {"status": "healthy"}


@app.get("/", include_in_schema=False)
def dashboard():
    # Serve the main dashboard.
    return FileResponse("app/templates/dashboard.html")