import logging
from io import BytesIO, StringIO
from fastapi.responses import StreamingResponse
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.predict import get_model, predict_rul

router = APIRouter()
logger = logging.getLogger("predictive_maintenance")

# Limit uploads to 5 MB and 1,000 rows to protect the public API.
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_ROWS = 1000


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        # Accept CSV files only.
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400,detail="Only CSV files are supported.")

        # Read a limited number of bytes to prevent oversized uploads.
        contents = await file.read(MAX_FILE_SIZE + 1)

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413,detail="CSV file exceeds the 5 MB limit.")

        # Parse the uploaded CSV into a DataFrame.
        try:
            data = pd.read_csv(BytesIO(contents))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
            raise HTTPException(status_code=400,detail="The uploaded file is not a valid CSV.")

        if data.empty or len(data) > MAX_ROWS:
            raise HTTPException(status_code=422,detail="CSV must contain between 1 and 1,000 data rows.")

        # Retrieve the input schema from the registered MLflow model.
        model = get_model()
        schema = model.metadata.get_input_schema()

        if schema is None:
            raise HTTPException(status_code=503,detail="Model input schema is unavailable.")

        # Check that all required model features are present.
        feature_names = [column.name for column in schema.inputs]
        missing = [name for name in feature_names if name not in data.columns]

        if missing:
            raise HTTPException(status_code=422,detail={"missing_columns": missing})

        # Select features in the exact order expected by the model.
        data = data[feature_names].copy()

        # Validate numeric values and convert columns to model-required types.
        for column in schema.inputs:
            name = column.name

            try:
                values = pd.to_numeric(data[name], errors="raise")
            except (ValueError, TypeError):
                raise HTTPException(status_code=422,detail=f"Column '{name}' must contain numeric values.")

            if not values.notna().all():
                raise HTTPException(status_code=422,detail=f"Column '{name}' contains missing values.")

            if column.type.name == "long":
                if not (values % 1 == 0).all():
                    raise HTTPException(status_code=422,detail=f"Column '{name}' must contain whole numbers.")

                data[name] = values.astype("int64")

            elif column.type.name == "double":
                data[name] = values.astype("float64")

        # Run batch inference using the registered XGBoost model.
        predictions = predict_rul(data)

        logger.info("CSV prediction completed: filename=%s, rows=%s", file.filename,len(data))
        
        # Add the model's predictions as a new column.
        data["predicted_rul"] = predictions

        # Convert the DataFrame into CSV text in memory.
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)

        # Move the buffer position back to the beginning.
        csv_buffer.seek(0)

        # Return the CSV as a downloadable file.
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="rul_predictions.csv"'})

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unexpected CSV upload error")
        raise HTTPException(status_code=500, detail="Unable to process the upload.")

    finally:
        await file.close()