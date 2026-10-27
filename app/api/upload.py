import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()
logger = logging.getLogger("predictive_maintenance")


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        logger.info("CSV upload received: %s", file.filename)
        return {"filename": file.filename, "status": "received"}

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected CSV upload error")
        raise HTTPException(status_code=500, detail="Unable to process the upload.")