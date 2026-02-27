from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from src import __version__ as model_version
from src.config.core import config
from src.monitoring.drift import compute_drift_report
from src.predict import make_prediction
from src.processing.data_manager import (
    append_drift_history_record,
    load_drift_baseline,
    load_drift_history_records,
    load_metadata,
)

from app import __version__, schemas
from app.config import get_logger, settings

api_router = APIRouter()
logger = get_logger(__name__)
VALID_DRIFT_STATUSES = {"ok", "warn", "drifted", "insufficient_data"}


def _parse_checked_at(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _build_drift_history_record(drift_report: Dict[str, Any]) -> Dict[str, Any]:
    feature_reports = [
        item for item in drift_report.get("feature_reports", [])
        if isinstance(item, dict)
    ]
    psi_values = [float(item.get("psi", 0.0)) for item in feature_reports]
    n_warn_features = sum(1 for item in feature_reports if item.get("status") == "warn")
    n_drifted_features = sum(
        1 for item in feature_reports if item.get("status") == "drifted"
    )

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "model_version": str(drift_report.get("model_version", model_version)),
        "baseline_version": str(drift_report.get("baseline_version", "unknown")),
        "n_records": int(drift_report.get("n_records", 0)),
        "overall_status": str(drift_report.get("overall_status", "ok")),
        "top_drifting_features": [
            str(feature) for feature in drift_report.get("top_drifting_features", [])
        ],
        "max_psi": max(psi_values) if psi_values else 0.0,
        "mean_psi": (sum(psi_values) / len(psi_values)) if psi_values else 0.0,
        "n_warn_features": int(n_warn_features),
        "n_drifted_features": int(n_drifted_features),
    }


def _load_normalized_history_records(
    model_version_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for record in load_drift_history_records():
        if not isinstance(record, dict):
            continue
        try:
            checked_at = _parse_checked_at(str(record.get("checked_at", "")))
            if checked_at is None:
                continue

            normalized_model_version = str(record.get("model_version", ""))
            if model_version_filter and normalized_model_version != model_version_filter:
                continue

            status = str(record.get("overall_status", "ok"))
            if status not in VALID_DRIFT_STATUSES:
                continue

            normalized = {
                "checked_at": checked_at.isoformat(),
                "model_version": normalized_model_version,
                "baseline_version": str(record.get("baseline_version", "unknown")),
                "n_records": int(record.get("n_records", 0)),
                "overall_status": status,
                "top_drifting_features": [
                    str(feature)
                    for feature in record.get("top_drifting_features", [])
                ],
                "max_psi": float(record.get("max_psi", 0.0)),
                "mean_psi": float(record.get("mean_psi", 0.0)),
                "n_warn_features": int(record.get("n_warn_features", 0)),
                "n_drifted_features": int(record.get("n_drifted_features", 0)),
                "_checked_at_dt": checked_at,
            }
        except (TypeError, ValueError):
            continue
        records.append(normalized)

    records.sort(key=lambda item: item["_checked_at_dt"], reverse=True)
    return records


def _history_record_for_response(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key != "_checked_at_dt"}


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    logger.info("Health endpoint requested")
    
    # Load model metadata if available
    metadata_dict = load_metadata()
    model_metadata = None
    if metadata_dict:
        try:
            model_metadata = schemas.ModelMetadata(**metadata_dict)
        except Exception as e:
            logger.warning(f"Failed to parse model metadata: {e}")
    
    health_response = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version,
        model_metadata=model_metadata,
    )
    return health_response.model_dump()


@api_router.post("/monitor/drift", response_model=schemas.DriftResponse, status_code=200)
async def monitor_drift(input_data: schemas.DriftRequest) -> Any:
    try:
        baseline = load_drift_baseline()
        if baseline is None:
            raise HTTPException(
                status_code=503,
                detail="Drift baseline is missing. Run `python -m src.train_pipeline` first.",
            )

        input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
        logger.info(f"Drift check requested for {len(input_df)} row(s)")
        drift_report = compute_drift_report(
            input_df=input_df.replace({np.nan: None}),
            baseline=baseline,
            config=config.ml_config,
        )
        try:
            append_drift_history_record(_build_drift_history_record(drift_report))
        except Exception as exc:
            logger.exception(f"Failed to append drift history record: {exc}")
        return drift_report
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception(f"Drift input validation failure: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected drift monitoring failure: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Drift monitoring failed due to an internal server error.",
        ) from exc


@api_router.get("/monitor/drift/history", response_model=schemas.DriftHistoryResponse, status_code=200)
def monitor_drift_history(
    limit: int = Query(default=50, ge=1, le=500),
    model_version: str | None = Query(default=None),
) -> dict:
    try:
        records = _load_normalized_history_records(model_version_filter=model_version)
        typed_records = [
            schemas.DriftHistoryRecord(**_history_record_for_response(record))
            for record in records[:limit]
        ]
        response = schemas.DriftHistoryResponse(
            model_version=model_version,
            limit=limit,
            total_records=len(records),
            records=typed_records,
        )
        return response.model_dump()
    except Exception as exc:
        logger.exception(f"Failed to fetch drift history: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Drift history lookup failed due to an internal server error.",
        ) from exc


@api_router.get("/monitor/drift/summary", response_model=schemas.DriftSummaryResponse, status_code=200)
def monitor_drift_summary(
    window_days: int = Query(default=7, ge=1, le=365),
    model_version: str | None = Query(default=None),
) -> dict:
    try:
        records = _load_normalized_history_records(model_version_filter=model_version)
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        records_in_window = [
            record for record in records if record["_checked_at_dt"] >= cutoff
        ]

        status_counts_counter = Counter(
            str(record["overall_status"]) for record in records_in_window
        )
        total_checks = len(records_in_window)
        drifted_count = int(status_counts_counter.get("drifted", 0))
        warn_count = int(status_counts_counter.get("warn", 0))

        feature_counter: Counter[str] = Counter()
        for record in records_in_window:
            feature_counter.update(record.get("top_drifting_features", []))

        top_recurrent_features = [
            schemas.FeatureFrequency(feature_name=feature_name, count=count)
            for feature_name, count in sorted(
                feature_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ]

        latest_check_at_dt = max(
            (record["_checked_at_dt"] for record in records_in_window),
            default=None,
        )

        response = schemas.DriftSummaryResponse(
            model_version=model_version,
            window_days=window_days,
            total_checks=total_checks,
            status_counts=schemas.DriftStatusCounts(
                ok=int(status_counts_counter.get("ok", 0)),
                warn=warn_count,
                drifted=drifted_count,
                insufficient_data=int(status_counts_counter.get("insufficient_data", 0)),
            ),
            drift_rate=(drifted_count / total_checks) if total_checks else 0.0,
            warn_rate=(warn_count / total_checks) if total_checks else 0.0,
            top_recurrent_features=top_recurrent_features,
            latest_check_at=latest_check_at_dt.isoformat() if latest_check_at_dt else None,
        )
        return response.model_dump()
    except Exception as exc:
        logger.exception(f"Failed to build drift summary: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Drift summary lookup failed due to an internal server error.",
        ) from exc


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    try:
        input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
        logger.info(f"Prediction requested for {len(input_df)} row(s)")
        results = make_prediction(input_data=input_df.replace({np.nan: None}))
        return results
    except ValueError as exc:
        logger.exception(f"Prediction input validation failure: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error(f"Model artifact missing: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Model artifact missing. Run training first.",
        ) from exc
    except Exception as exc:
        logger.exception(f"Unexpected prediction failure: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        ) from exc
