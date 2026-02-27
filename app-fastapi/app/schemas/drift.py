from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.predict import DataInput

FeatureDriftStatus = Literal["ok", "warn", "drifted"]
OverallDriftStatus = Literal["ok", "warn", "drifted", "insufficient_data"]


class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: List[DataInput] = Field(min_length=1, max_length=1000)


class DriftThresholds(BaseModel):
    warn: float
    drifted: float


class FeatureDriftReport(BaseModel):
    feature_name: str
    feature_type: Literal["numeric", "categorical"]
    psi: float
    status: FeatureDriftStatus
    expected_missing_rate: float
    observed_missing_rate: float
    missing_rate_delta: float
    expected_distribution: Dict[str, float]
    observed_distribution: Dict[str, float]


class DriftResponse(BaseModel):
    model_version: str
    baseline_version: str
    n_records: int
    overall_status: OverallDriftStatus
    thresholds: DriftThresholds
    top_drifting_features: List[str]
    feature_reports: List[FeatureDriftReport]
    message: Optional[str] = None


class DriftHistoryRecord(BaseModel):
    checked_at: str
    model_version: str
    baseline_version: str
    n_records: int
    overall_status: OverallDriftStatus
    top_drifting_features: List[str]
    max_psi: float
    mean_psi: float
    n_warn_features: int
    n_drifted_features: int


class DriftHistoryResponse(BaseModel):
    model_version: Optional[str] = None
    limit: int
    total_records: int
    records: List[DriftHistoryRecord]


class DriftStatusCounts(BaseModel):
    ok: int = 0
    warn: int = 0
    drifted: int = 0
    insufficient_data: int = 0


class FeatureFrequency(BaseModel):
    feature_name: str
    count: int


class DriftSummaryResponse(BaseModel):
    model_version: Optional[str] = None
    window_days: int
    total_checks: int
    status_counts: DriftStatusCounts
    drift_rate: float
    warn_rate: float
    top_recurrent_features: List[FeatureFrequency]
    latest_check_at: Optional[str] = None
