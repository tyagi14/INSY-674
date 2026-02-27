from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src import __version__ as _version
from src.config.core import config
from src.monitoring.drift import build_drift_baseline
from src.pipeline import pipe
from src.processing.data_manager import (
    load_dataset,
    save_drift_baseline,
    save_evaluation_report,
    save_metadata,
    save_pipeline,
)


def _get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return "unknown"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _build_evaluation_report(
    *,
    y_true: Any,
    y_pred: Any,
    metrics: Dict[str, float],
    git_sha: str,
    feature_names: list[str],
) -> Dict[str, Any]:
    class_zero_rate = float((y_true == 0).mean())
    class_one_rate = float((y_true == 1).mean())
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "model_version": _version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "validation_rows": int(len(y_true)),
        "metrics": metrics,
        "target_distribution": {
            "class_0_rate": class_zero_rate,
            "class_1_rate": class_one_rate,
        },
        "classification_report": _to_builtin(report),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": _to_builtin(conf_matrix),
        },
        "feature_names": feature_names,
    }

def run_training() -> Dict[str, float]:
    """Train and persist the model pipeline."""

    data = load_dataset(file_name=config.app_config.training_data_file, drop_features=True)
    x = data.drop(columns=[config.ml_config.target])
    y = data[config.ml_config.target].astype(int)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=config.ml_config.test_size,
        random_state=config.ml_config.random_state,
        stratify=y,
    )

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_valid)
    y_proba = pipe.predict_proba(x_valid)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
    }
    drift_baseline = build_drift_baseline(x_train=x_train, config=config.ml_config)
    git_sha = _get_git_sha()

    # Build model metadata
    feature_names = list(x_train.columns) if hasattr(x_train, "columns") else []
    metadata = {
        "model_version": _version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "metrics": metrics,
        "n_rows": len(data),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    evaluation_report = _build_evaluation_report(
        y_true=y_valid,
        y_pred=y_pred,
        metrics=metrics,
        git_sha=git_sha,
        feature_names=feature_names,
    )

    save_pipeline(pipeline_to_persist=pipe)
    save_metadata(metadata=metadata)
    save_evaluation_report(report=evaluation_report)
    save_drift_baseline(baseline=drift_baseline)
    return metrics

def _generate_shap_summary(x_train, x_valid) -> None:
    """Generate a SHAP summary plot with transformed feature names, if SHAP is installed."""

    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return

    try:
        preprocessor = pipe[:-1]
        model = pipe.named_steps["logistic_regression"]

        x_train_t = preprocessor.transform(x_train)
        x_valid_t = preprocessor.transform(x_valid)

        input_feature_names = list(x_train.columns) if hasattr(x_train, "columns") else []
        feature_names = _get_transformed_feature_names(
            preprocessor=preprocessor,
            x_transformed=x_valid_t,
            input_feature_names=input_feature_names,
        )
        print("Feature names used in SHAP:")
        
        print(feature_names)

        if _are_generic_feature_names(feature_names) and len(pipe.steps) > 1:
            pre_scaler = pipe[:-2]
            x_valid_pre_scaler = pre_scaler.transform(x_valid)
            pre_scaler_feature_names = _get_transformed_feature_names(
                preprocessor=pre_scaler,
                x_transformed=x_valid_pre_scaler,
                input_feature_names=input_feature_names,
            )
            if len(pre_scaler_feature_names) == np.asarray(x_valid_t).shape[1]:
                feature_names = pre_scaler_feature_names

        x_train_array = np.nan_to_num(
            np.asarray(x_train_t, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        x_valid_array = np.nan_to_num(
            np.asarray(x_valid_t, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )

        explainer = shap.LinearExplainer(model, x_train_array)
        shap_values = explainer.shap_values(x_valid_array)

        shap.summary_plot(
            shap_values,
            x_valid_array,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=150)
        plt.close()
    except Exception:
        return


def _get_transformed_feature_names(
    preprocessor, x_transformed, input_feature_names: list[str]
) -> list[str]:
    """Resolve transformed feature names for SHAP plotting."""

    if hasattr(x_transformed, "columns"):
        return [str(col) for col in x_transformed.columns]

    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(col) for col in preprocessor.get_feature_names_out()]
        except Exception:
            pass

    n_transformed_features = np.asarray(x_transformed).shape[1]

    if input_feature_names and len(input_feature_names) == n_transformed_features:
        return [str(col) for col in input_feature_names]

    return [f"feature_{idx}" for idx in range(n_transformed_features)]


def _are_generic_feature_names(feature_names: list[str]) -> bool:
    """Return True when names look like SHAP defaults (feature_0, feature_1, ...)."""

    if not feature_names:
        return True

    for idx, name in enumerate(feature_names):
        if name not in {f"feature_{idx}", f"Feature {idx}"}:
            return False

    return True
if __name__ == "__main__":
    training_metrics = run_training()
    print(
        f"Training complete | accuracy={training_metrics['accuracy']:.4f} "
        f"roc_auc={training_metrics['roc_auc']:.4f}"
    )
