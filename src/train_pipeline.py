from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import set_config

from src import __version__ as _version
from src.config.core import config
from src.monitoring.drift import build_drift_baseline
from src.pipeline import pipe
from src.processing.data_manager import load_dataset, save_metadata, save_pipeline


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
    print("Features used in model:")
    print(list(x.columns))
    # Keep tabular transformers in pandas output mode when possible so downstream
    # explainability utilities can preserve feature names.
    set_config(transform_output="pandas")

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_valid)
    y_proba = pipe.predict_proba(x_valid)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
    }

    _generate_shap_summary(x_train=x_train, x_valid=x_valid)

    # Build model metadata
    feature_names = list(x_train.columns) if hasattr(x_train, "columns") else []
    metadata = {
        "model_version": _version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "metrics": metrics,
        "n_rows": len(data),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }

    save_pipeline(pipeline_to_persist=pipe)
    save_metadata(metadata=metadata)
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
