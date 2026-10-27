import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("predictive_maintenance")
templates = Jinja2Templates(directory="app/templates")


def register_exception_handlers(app: FastAPI):

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        status_code = exc.status_code

        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=status_code, content={"detail": exc.detail})

        page = f"errors/{status_code}.html"

        if status_code not in (400, 403, 404, 413, 429):
            page = "errors/500.html"

        return templates.TemplateResponse(
            request=request,
            name=page,
            context={},
            status_code=status_code
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning("Request validation failed: %s", request.url.path)

        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=422, content={"detail": "Invalid request data."})

        return templates.TemplateResponse(
            request=request,
            name="errors/400.html",
            context={},
            status_code=422
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        logger.error("Unexpected server error", exc_info=(type(exc), exc, exc.__traceback__))

        if request.url.path.startswith("/api/"):
            return JSONResponse(status_code=500, content={"detail": "Internal server error."})

        return templates.TemplateResponse(
            request=request,
            name="errors/500.html",
            context={},
            status_code=500
        )