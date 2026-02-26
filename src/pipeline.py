from __future__ import annotations

from typing import cast

from feature_engine.encoding import CountFrequencyEncoder, OneHotEncoder, OrdinalEncoder
from feature_engine.imputation import CategoricalImputer
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config.core import config
from src.processing.features import Mapper

pipe = Pipeline(
    [
        (
            "categorical_missing_imputer",
            CategoricalImputer(
                imputation_method="missing",
                variables=cast(
                    list[str | int],
                    config.ml_config.categorical_missing_with_constant,
                ),
            ),
        ),
        (
            "categorical_frequent_imputer",
            CategoricalImputer(
                imputation_method="frequent",
                variables=cast(
                    list[str | int],
                    config.ml_config.categorical_missing_with_frequent,
                ),
            ),
        ),
        (
            "yeojohnson_transformer",
            YeoJohnsonTransformer(
                variables=cast(
                    list[str | int],
                    config.ml_config.yeojohnson_features,
                ),
            ),
        ),
        (
            "ordinal_encoder",
            OrdinalEncoder(
                encoding_method="ordered",
                variables=cast(
                    list[str | int],
                    config.ml_config.ordinal_features,
                ),
            ),
        ),
        (
            "arbitrary_ordinal_encoder",
            OrdinalEncoder(
                encoding_method="arbitrary",
                variables=cast(
                    list[str | int],
                    config.ml_config.arbitrary_ordinal_features,
                ),
            ),
        ),
        (
            "count_frequency_encoder",
            CountFrequencyEncoder(
                encoding_method="frequency",
                variables=cast(
                    list[str | int],
                    config.ml_config.count_frequency_features,
                ),
            ),
        ),
        (
           "one_hot_encoder",
            OneHotEncoder(
                variables=cast(
                    list[str | int],
                    config.ml_config.one_hot_features,
        ),
        drop_last=False,
        ignore_format=True,
                ),
            ),
        ),
        (
            "experience_mapper",
            Mapper(
                variables=config.ml_config.experience_features,
                mappings=config.ml_config.experience_map,
                default_value=0,
            ),
        ),
        (
            "last_new_job_mapper",
            Mapper(
                variables=config.ml_config.last_new_job_features,
                mappings=config.ml_config.last_new_job_map,
                default_value=0,
            ),
        ),
        (
            "company_size_mapper",
            Mapper(
                variables=config.ml_config.company_size_features,
                mappings=config.ml_config.company_size_map,
                default_value=0,
            ),
        ),
        (
            "numeric_missing_imputer",
            SimpleImputer(strategy="constant", fill_value=0),
        ),
        ("min_max_scaler", MinMaxScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                random_state=config.ml_config.random_state,
                max_iter=config.ml_config.model_max_iter,
                class_weight=config.ml_config.class_weight,
                solver="lbfgs",
                n_jobs=-1,
            ),
        ),
    ]
)
