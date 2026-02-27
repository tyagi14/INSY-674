from __future__ import annotations

import json
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import __version__ as _version
from src.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def _load_json_dict(file_path: Path) -> t.Dict[str, t.Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {file_path}, got {type(payload).__name__}.")
    return t.cast(t.Dict[str, t.Any], payload)


def model_file_name() -> str:
    return f"{config.app_config.pipeline_save_file}{_version}.pkl"


def model_artifact_path() -> Path:
    return TRAINED_MODEL_DIR / model_file_name()


def load_dataset(file_name: str, drop_features: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / file_name)
    if drop_features and config.ml_config.drop_features:
        existing_features = [f for f in config.ml_config.drop_features if f in df.columns]
        if existing_features:
            df = df.drop(columns=existing_features)
    return df


def save_pipeline(pipeline_to_persist: Pipeline) -> None:
    save_file_name = model_file_name()
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {file_path}. "
            "Run training first with `python -m src.train_pipeline`."
        )
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(files_to_keep: t.List[str]) -> None:
    do_not_delete = set(files_to_keep + [".gitkeep", "__init__.py"])
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete and model_file.is_file():
            model_file.unlink()


def metadata_file_name() -> str:
    return f"{config.app_config.pipeline_save_file}{_version}_metadata.json"


def metadata_artifact_path() -> Path:
    return TRAINED_MODEL_DIR / metadata_file_name()


def save_metadata(metadata: t.Dict[str, t.Any]) -> None:
    save_file_name = metadata_file_name()
    save_path = TRAINED_MODEL_DIR / save_file_name
    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata() -> t.Optional[t.Dict[str, t.Any]]:
    file_path = metadata_artifact_path()
    if not file_path.exists():
        return None
    return _load_json_dict(file_path)


def evaluation_report_file_name() -> str:
    return f"{config.app_config.pipeline_save_file}{_version}_evaluation.json"


def evaluation_report_artifact_path() -> Path:
    return TRAINED_MODEL_DIR / evaluation_report_file_name()


def save_evaluation_report(report: t.Dict[str, t.Any]) -> None:
    save_path = evaluation_report_artifact_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def load_evaluation_report() -> t.Optional[t.Dict[str, t.Any]]:
    file_path = evaluation_report_artifact_path()
    if not file_path.exists():
        return None
    return _load_json_dict(file_path)


def drift_baseline_file_name() -> str:
    return f"{config.app_config.pipeline_save_file}{_version}_drift_baseline.json"


def drift_baseline_artifact_path() -> Path:
    return TRAINED_MODEL_DIR / drift_baseline_file_name()


def save_drift_baseline(baseline: t.Dict[str, t.Any]) -> None:
    save_path = drift_baseline_artifact_path()
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)


def load_drift_baseline() -> t.Optional[t.Dict[str, t.Any]]:
    file_path = drift_baseline_artifact_path()
    if not file_path.exists():
        return None
    return _load_json_dict(file_path)


def drift_history_file_name() -> str:
    return f"{config.app_config.pipeline_save_file}{_version}_drift_history.jsonl"


def drift_history_artifact_path() -> Path:
    return TRAINED_MODEL_DIR / drift_history_file_name()


def append_drift_history_record(record: t.Dict[str, t.Any]) -> None:
    file_path = drift_history_artifact_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True))
        f.write("\n")


def load_drift_history_records() -> t.List[t.Dict[str, t.Any]]:
    file_path = drift_history_artifact_path()
    if not file_path.exists():
        return []

    records: t.List[t.Dict[str, t.Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(t.cast(t.Dict[str, t.Any], payload))
    return records
