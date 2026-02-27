from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from src.monitoring.drift import build_drift_baseline


def _valid_input_row(**overrides):
    row = {
        "city": "city_41",
        "city_development_index": 0.827,
        "gender": "Male",
        "relevent_experience": "Has relevent experience",
        "enrolled_university": "Full time course",
        "education_level": "Graduate",
        "major_discipline": "STEM",
        "experience": "9",
        "company_size": "<10",
        "company_type": "Pvt Ltd",
        "last_new_job": "1",
        "training_hours": 21.0,
    }
    row.update(overrides)
    return row


def _test_drift_config() -> SimpleNamespace:
    return SimpleNamespace(
        drift_warn_threshold=0.1,
        drift_alert_threshold=0.25,
        drift_min_samples=50,
        drift_max_categories=20,
    )


def _baseline_for_api_tests():
    baseline_train = pd.DataFrame(
        {
            "city": ["city_41"] * 70 + ["city_11"] * 30,
            "city_development_index": [0.82] * 100,
            "training_hours": [21.0] * 100,
        }
    )
    return build_drift_baseline(x_train=baseline_train, config=_test_drift_config())


def _history_record(
    *,
    checked_at: datetime,
    model_version: str = "0.1.0",
    overall_status: str = "ok",
    top_drifting_features: list[str] | None = None,
    max_psi: float = 0.0,
    mean_psi: float = 0.0,
    n_warn_features: int = 0,
    n_drifted_features: int = 0,
) -> dict:
    return {
        "checked_at": checked_at.isoformat(),
        "model_version": model_version,
        "baseline_version": model_version,
        "n_records": 60,
        "overall_status": overall_status,
        "top_drifting_features": top_drifting_features or [],
        "max_psi": max_psi,
        "mean_psi": mean_psi,
        "n_warn_features": n_warn_features,
        "n_drifted_features": n_drifted_features,
    }


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"]
    assert payload["api_version"]
    assert payload["model_version"]
    # model_metadata should be present (may be null if metadata file doesn't exist)
    assert "model_metadata" in payload


def test_predict_endpoint(client):
    payload = {
        "inputs": [
            {
                "city": "city_41",
                "city_development_index": 0.827,
                "gender": "Male",
                "relevent_experience": "Has relevent experience",
                "enrolled_university": "Full time course",
                "education_level": "Graduate",
                "major_discipline": "STEM",
                "experience": "9",
                "company_size": "<10",
                "company_type": "Pvt Ltd",
                "last_new_job": "1",
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] in [0, 1]
    assert "model_version" in body
    assert "probabilities" in body
    assert isinstance(body["probabilities"], list)
    assert len(body["probabilities"]) == 1


def test_predict_validation_error(client):
    payload = {
        "inputs": [
            {
                "city": "city_41",
                "city_development_index": 0.827,
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_health_endpoint_with_metadata(client, monkeypatch):
    """Test that health endpoint returns model metadata when available."""
    from app import api

    expected_metadata = {
        "model_version": "0.1.0",
        "trained_at": "2026-02-25T19:00:00+00:00",
        "git_sha": "abc1234",
        "metrics": {"accuracy": 0.81, "roc_auc": 0.88},
        "n_rows": 19158,
        "n_features": 12,
        "feature_names": [
            "city",
            "city_development_index",
            "gender",
            "relevent_experience",
            "enrolled_university",
            "education_level",
            "major_discipline",
            "experience",
            "company_size",
            "company_type",
            "last_new_job",
            "training_hours",
        ],
    }
    monkeypatch.setattr(api, "load_metadata", lambda: expected_metadata)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    metadata = payload["model_metadata"]
    assert metadata is not None
    assert metadata["model_version"] == expected_metadata["model_version"]
    assert metadata["trained_at"] == expected_metadata["trained_at"]
    assert metadata["git_sha"] == expected_metadata["git_sha"]
    assert metadata["metrics"]["accuracy"] == expected_metadata["metrics"]["accuracy"]
    assert metadata["metrics"]["roc_auc"] == expected_metadata["metrics"]["roc_auc"]
    assert metadata["n_rows"] == expected_metadata["n_rows"]
    assert metadata["n_features"] == expected_metadata["n_features"]
    assert metadata["feature_names"] == expected_metadata["feature_names"]


def test_health_endpoint_without_metadata(client, monkeypatch):
    """Test that health endpoint returns 200 even when metadata is missing."""
    from app import api

    monkeypatch.setattr(api, "load_metadata", lambda: None)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert "model_metadata" in payload
    assert payload["model_metadata"] is None


def test_monitor_drift_endpoint_success(client, monkeypatch):
    from app import api

    captured_records = []

    monkeypatch.setattr(api, "load_drift_baseline", lambda: _baseline_for_api_tests())
    monkeypatch.setattr(
        api,
        "append_drift_history_record",
        lambda record: captured_records.append(record),
    )
    payload = {"inputs": [_valid_input_row()]}

    response = client.post("/api/v1/monitor/drift", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["n_records"] == 1
    assert "overall_status" in body
    assert "feature_reports" in body
    assert isinstance(body["feature_reports"], list)
    assert len(captured_records) == 1
    assert captured_records[0]["model_version"] == body["model_version"]
    assert captured_records[0]["overall_status"] == body["overall_status"]
    assert "checked_at" in captured_records[0]


def test_monitor_drift_endpoint_missing_baseline(client, monkeypatch):
    from app import api

    monkeypatch.setattr(api, "load_drift_baseline", lambda: None)
    payload = {"inputs": [_valid_input_row()]}

    response = client.post("/api/v1/monitor/drift", json=payload)
    assert response.status_code == 503
    assert "Drift baseline is missing" in response.json()["detail"]


def test_monitor_drift_history_empty(client, monkeypatch):
    from app import api

    monkeypatch.setattr(api, "load_drift_history_records", lambda: [])
    response = client.get("/api/v1/monitor/drift/history")
    assert response.status_code == 200
    body = response.json()
    assert body["total_records"] == 0
    assert body["records"] == []


def test_monitor_drift_history_filter_and_limit(client, monkeypatch):
    from app import api

    now = datetime.now(timezone.utc)
    records = [
        _history_record(
            checked_at=now - timedelta(hours=2),
            model_version="0.1.0",
            overall_status="warn",
            top_drifting_features=["city"],
            max_psi=0.2,
            mean_psi=0.1,
            n_warn_features=1,
        ),
        _history_record(
            checked_at=now - timedelta(hours=1),
            model_version="0.2.0",
            overall_status="drifted",
            top_drifting_features=["training_hours"],
            max_psi=0.4,
            mean_psi=0.3,
            n_drifted_features=1,
        ),
        _history_record(
            checked_at=now,
            model_version="0.1.0",
            overall_status="ok",
            top_drifting_features=[],
            max_psi=0.02,
            mean_psi=0.01,
        ),
    ]

    monkeypatch.setattr(api, "load_drift_history_records", lambda: records)
    response = client.get("/api/v1/monitor/drift/history?model_version=0.1.0&limit=1")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "0.1.0"
    assert body["limit"] == 1
    assert body["total_records"] == 2
    assert len(body["records"]) == 1
    assert body["records"][0]["model_version"] == "0.1.0"
    assert body["records"][0]["overall_status"] == "ok"


def test_monitor_drift_summary_mixed_statuses(client, monkeypatch):
    from app import api

    now = datetime.now(timezone.utc)
    records = [
        _history_record(
            checked_at=now - timedelta(days=1),
            model_version="0.1.0",
            overall_status="ok",
            top_drifting_features=[],
        ),
        _history_record(
            checked_at=now - timedelta(days=2),
            model_version="0.1.0",
            overall_status="warn",
            top_drifting_features=["city"],
            max_psi=0.15,
            mean_psi=0.08,
            n_warn_features=1,
        ),
        _history_record(
            checked_at=now - timedelta(days=3),
            model_version="0.1.0",
            overall_status="drifted",
            top_drifting_features=["city", "training_hours"],
            max_psi=0.35,
            mean_psi=0.2,
            n_drifted_features=1,
        ),
        _history_record(
            checked_at=now - timedelta(days=4),
            model_version="0.2.0",
            overall_status="insufficient_data",
            top_drifting_features=["city"],
        ),
    ]
    monkeypatch.setattr(api, "load_drift_history_records", lambda: records)

    response = client.get("/api/v1/monitor/drift/summary?window_days=7")
    assert response.status_code == 200
    body = response.json()
    assert body["total_checks"] == 4
    assert body["status_counts"]["ok"] == 1
    assert body["status_counts"]["warn"] == 1
    assert body["status_counts"]["drifted"] == 1
    assert body["status_counts"]["insufficient_data"] == 1
    assert body["drift_rate"] == 0.25
    assert body["warn_rate"] == 0.25
    assert len(body["top_recurrent_features"]) > 0
    assert body["top_recurrent_features"][0]["feature_name"] == "city"
    assert body["top_recurrent_features"][0]["count"] == 3
    assert body["latest_check_at"] is not None


def test_monitor_drift_summary_window_and_model_filter(client, monkeypatch):
    from app import api

    now = datetime.now(timezone.utc)
    records = [
        _history_record(
            checked_at=now - timedelta(days=1),
            model_version="0.1.0",
            overall_status="warn",
            top_drifting_features=["city"],
        ),
        _history_record(
            checked_at=now - timedelta(days=10),
            model_version="0.1.0",
            overall_status="drifted",
            top_drifting_features=["training_hours"],
        ),
        _history_record(
            checked_at=now - timedelta(days=1),
            model_version="0.2.0",
            overall_status="drifted",
            top_drifting_features=["experience"],
        ),
    ]
    monkeypatch.setattr(api, "load_drift_history_records", lambda: records)

    response = client.get(
        "/api/v1/monitor/drift/summary?window_days=7&model_version=0.1.0"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "0.1.0"
    assert body["total_checks"] == 1
    assert body["status_counts"]["warn"] == 1
    assert body["status_counts"]["drifted"] == 0
    assert body["drift_rate"] == 0.0
    assert body["warn_rate"] == 1.0
    assert body["top_recurrent_features"][0]["feature_name"] == "city"


def test_monitor_drift_history_validation_error(client):
    response = client.get("/api/v1/monitor/drift/history?limit=0")
    assert response.status_code == 422


def test_monitor_drift_summary_validation_error(client):
    response = client.get("/api/v1/monitor/drift/summary?window_days=0")
    assert response.status_code == 422
