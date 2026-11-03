import logging
from pydantic import BaseModel, Field
import dagshub
import mlflow
import pandas as pd
from fastapi import APIRouter, HTTPException



router = APIRouter()
logger = logging.getLogger("predictive_maintenance")

# Reference the production candidate using its registry alias.
MODEL_URI = "models:/CMapss_RUL_XGBoost@champion"

# Cache the model so it is not downloaded for every prediction request.
_model = None


def get_model():
    global _model
    if _model is None:
        # Configure DagsHub only when the model needs loading.
        dagshub.init(repo_owner="Omesham", repo_name="predictive-maintenance-mlops", mlflow=True)
        _model = mlflow.pyfunc.load_model(MODEL_URI)

    return _model


def predict_rul(data: pd.DataFrame):
    try:
        # Generate remaining useful life predictions using the trained model.
        model = get_model()
        return model.predict(data)

    except Exception:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500,detail="Unable to generate predictions.")
        

# Define the input structure for prediction requests.
class PredictionRequest(BaseModel):
    features: list[float] = Field(min_length=1)


@router.post("/predict")
def predict(request: PredictionRequest):
    # Load the model and obtain its expected feature names.
    model = get_model()
    schema = model.metadata.get_input_schema()

    if schema is None:
        logger.error("Model input schema is unavailable")
        raise HTTPException(status_code=503, detail="Model input schema is unavailable.")

    if len(request.features) != len(schema.inputs):
        raise HTTPException(status_code=422, detail=f"Expected {len(schema.inputs)} features.")

    # Preserve the feature order expected by the trained model.
    feature_names = [column.name for column in schema.inputs]

    # Convert the request into a DataFrame for XGBoost inference.
    # Build the input DataFrame using the model's expected feature names.
    data = pd.DataFrame([request.features], columns=feature_names)

    # Convert each column to the data type required by the MLflow model.
    for column in schema.inputs:
        if column.type.name == "long":
            # Integer features must contain whole numbers.
            values = data[column.name]

            if not (values % 1 == 0).all():
                raise HTTPException(
                    status_code=422,
                    detail=f"{column.name} must be an integer."
                )

            data[column.name] = values.astype("int64")

        elif column.type.name == "double":
            data[column.name] = data[column.name].astype("float64")

    # Predict the equipment's remaining useful life.
    predictions = predict_rul(data)

    return {"predicted_rul": float(predictions[0]),"unit": "cycles"}