from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd

REFERENCE_DIRECTORY = Path("/app/data/curated/train")


def load_reference_data(feature_names: list[str]) -> pd.DataFrame:
    datasets = []

    for dataset_name in ("FD001", "FD002", "FD003", "FD004"):
        file_path = REFERENCE_DIRECTORY / f"train_{dataset_name}.parquet"

        data = pd.read_parquet(file_path)

        # Recreate the dataset indicators used during training.
        for name in ("FD001", "FD002", "FD003", "FD004"):
            data[f"dataset_{name}"] = int(name == dataset_name)

        datasets.append(data)

    reference_data = pd.concat(datasets, ignore_index=True)

    missing = [
        name for name in feature_names
        if name not in reference_data.columns
    ]

    if missing:
        raise ValueError(f"Missing reference features: {missing}")

    return reference_data[feature_names].copy()

def detect_data_drift(current_data: pd.DataFrame) -> dict:
    feature_names = current_data.columns.tolist()
    reference_data = load_reference_data(feature_names)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data, current_data=current_data
    )

    return snapshot.dict()