"""Transformations of features."""

from typing import cast

import pandas as pd
from pydantic import BaseModel


class CategoricalFeature(BaseModel):
    """A categorical feature."""

    name: str
    values: list[str]


class ContinuousFeature(BaseModel):
    """A continuous feature."""

    name: str
    min: float
    max: float


class RegressorInputSpec(BaseModel):
    """A design space."""

    features: list[CategoricalFeature | ContinuousFeature]

    def check_features(
        self,
        features: pd.DataFrame,
    ):
        """Check that the features are valid.

        Specifically, makes sure that (a) the feature is present in the dataframe,
        and (b) that the values are within the allowed set.

        Args:
            features (pd.DataFrame): The dataframe to check.

        Raises:
            ValueError: If the feature is not present in the dataframe or if the values are not within the allowed set.
            TypeError: If the feature is not a supported feature type.
        """
        for feature in self.features:
            if isinstance(feature, CategoricalFeature):
                if feature.name not in features.columns:
                    msg = f"Feature {feature.name} not found in dataframe"
                    raise ValueError(msg)
                if not cast(
                    pd.Series, features[feature.name].isin(feature.values)
                ).all():
                    msg = f"Feature {feature.name} has values that are not in the allowed set: {feature.values}"
                    raise ValueError(msg)
            elif isinstance(feature, ContinuousFeature):
                if feature.name not in features.columns:
                    msg = f"Feature {feature.name} not found in dataframe"
                    raise ValueError(msg)
                if not (
                    (features[feature.name] >= feature.min)
                    & (features[feature.name] <= feature.max)
                ).all():
                    msg = f"Feature {feature.name} has values that are not in the allowed range: {feature.min} to {feature.max}"
                    raise ValueError(msg)
            else:
                msg = f"Unknown feature type: {type(feature): {feature}}"
                raise TypeError(msg)

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform the features.

        Args:
            features (pd.DataFrame): The dataframe to transform.

        Returns:
           transformed_features (pd.DataFrame): The transformed features.
        """
        transformed_features = features.copy(deep=True)
        for feature in self.features:
            if isinstance(feature, CategoricalFeature):
                transformed_features[feature.name] = pd.Categorical(
                    transformed_features[feature.name], categories=feature.values
                ).codes

            elif isinstance(feature, ContinuousFeature):
                low, high = feature.min, feature.max
                transformed_features[feature.name] = (
                    transformed_features[feature.name] - low
                ) / (high - low)
            else:
                msg = f"Unknown feature type: {type(feature): {feature}}"
                raise TypeError(msg)

        transformed_features = transformed_features[[f.name for f in self.features]]

        return cast(pd.DataFrame, transformed_features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform the features.

        Args:
            features (pd.DataFrame): The dataframe to transform.

        Returns:
            transformed_features (pd.DataFrame): The transformed features.
        """
        self.check_features(features)
        return self.transform_features(features)
