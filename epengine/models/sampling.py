"""Conditional Priors and Samplers."""

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator


class SamplingError(Exception):
    """A sampling error."""

    pass


class Sampler(ABC):
    """A sampler."""

    @abstractmethod
    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Sample features from a prior, which may depend on a context."""
        pass


class UniformSampler(BaseModel, Sampler):
    """A uniform sampler which generates values uniformly between a min and max value."""

    min: float
    max: float

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Sample uniformly between a min and max value."""
        return generator.uniform(self.min, self.max, size=n)


class ClippedNormalSampler(BaseModel, Sampler):
    """A clipped normal sampler which generates values from a normal distribution, clipped to a min and max value."""

    mean: float
    std: float
    clip_min: float | None
    clip_max: float | None

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Sample from a normal distribution, clipped to a min and max value."""
        clip_min = self.clip_min if self.clip_min is not None else -np.inf
        clip_max = self.clip_max if self.clip_max is not None else np.inf
        samples = generator.normal(self.mean, self.std, size=n).clip(clip_min, clip_max)
        return samples


class FixedValueSampler(BaseModel):
    """A fixed value sampler which generates a fixed value for all samples."""

    value: float | str | int | bool

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Sample a fixed value."""
        return np.full(n, self.value)


class CategoricalSampler(BaseModel):
    """A categorical sampler which generates values from a categorical distribution."""

    values: list[str] | list[float] | list[int]
    weights: list[float]

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Sample from a categorical distribution."""
        return generator.choice(self.values, size=n, p=self.weights)

    @model_validator(mode="after")
    def check_values_and_weights(self):
        """Check that the values and weights are the same length and normalized."""
        if len(self.values) != len(self.weights):
            msg = "values and weights must be the same length"
            raise ValueError(msg)
        if not np.isclose(sum(self.weights), 1):
            self.weights = [w / sum(self.weights) for w in self.weights]
        return self


class CopySampler(BaseModel):
    """A deterministic sampler which generates a copy of a feature in the provided context dataframe."""

    feature_to_copy: str

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a copy of a feature."""
        if self.feature_to_copy not in context.columns:
            msg = f"Feature to copy {self.feature_to_copy} not found in context dataframe."
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        return context[self.feature_to_copy].to_numpy()


class AddValueSampler(BaseModel):
    """A deterministic sampler which adds a value to a feature."""

    feature_to_add_to: str
    value_to_add: float

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a sum of a feature and a value."""
        if self.feature_to_add_to not in context.columns:
            msg = f"Feature to add to {self.feature_to_add_to} not found in context dataframe."
            raise SamplingError(msg)
        return context[self.feature_to_add_to].to_numpy() + self.value_to_add


class SumValuesSampler(BaseModel):
    """A deterministic sampler which generates a sum of features."""

    features_to_sum: list[str]

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a sum of features."""
        if not all(f in context.columns for f in self.features_to_sum):
            msg = f"All features to sum {self.features_to_sum} must be found in context dataframe."
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        return np.sum(context[self.features_to_sum].to_numpy(), axis=1)


class MultiplyValueSampler(BaseModel):
    """A deterministic sampler which generates a product of a feature and a value."""

    feature_to_multiply: str
    value_to_multiply: float

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a multiply of a feature."""
        if self.feature_to_multiply not in context.columns:
            msg = f"Feature to multiply {self.feature_to_multiply} not found in context dataframe."
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        return context[self.feature_to_multiply].to_numpy() * self.value_to_multiply


class ProductValuesSampler(BaseModel):
    """A deterministic sampler which generates a product of features."""

    features_to_multiply: list[str]

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a product of features."""
        if not all(f in context.columns for f in self.features_to_multiply):
            msg = f"All features to multiply {self.features_to_multiply} must be found in context dataframe."
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        return np.prod(context[self.features_to_multiply].to_numpy(), axis=1)


class InvertSampler(BaseModel):
    """A deterministic sampler which generates the multiplicative inverse of a feature."""

    feature_to_invert: str

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute an invert of a feature."""
        if self.feature_to_invert not in context.columns:
            msg = f"Feature to invert {self.feature_to_invert} not found in context dataframe."
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        return 1 / context[self.feature_to_invert].to_numpy()


class LogSampler(BaseModel):
    """A deterministic sampler which generates a log of a feature."""

    feature_to_log: str
    base: float = np.e

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a log of a feature."""
        if self.feature_to_log not in context.columns:
            msg = (
                f"Feature to log {self.feature_to_log} not found in context dataframe."
            )
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        return np.log(context[self.feature_to_log].to_numpy()) / np.log(self.base)


class ConcatenateFeaturesSampler(BaseModel):
    """A deterministic sampler which concatenates features."""

    features_to_concatenate: list[str]
    separator: str = ":"

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Compute a concatenation of features."""
        if not all(f in context.columns for f in self.features_to_concatenate):
            msg = f"All features to concatenate {self.features_to_concatenate} must be found in context dataframe."
            raise SamplingError(msg)
        if len(context) != n:
            msg = (
                f"Context dataframe must have {n} rows, but it has {len(context)} rows."
            )
            raise SamplingError(msg)
        cols: pd.DataFrame = cast(pd.DataFrame, context[self.features_to_concatenate])
        return cols.astype(str).agg(self.separator.join, axis=1).to_numpy()


# TODO:
# add some more column operations, e.g.
# breakpoints
# thresholding
# regex
# etc

PriorSampler = (
    UniformSampler
    | ClippedNormalSampler
    | FixedValueSampler
    | CategoricalSampler
    | CopySampler
    | AddValueSampler
    | SumValuesSampler
    | MultiplyValueSampler
    | ProductValuesSampler
    | InvertSampler
    | LogSampler
    | ConcatenateFeaturesSampler
)


class ConditionalPriorCondition(BaseModel):
    """A conditional prior condition."""

    match_val: str | float | int | bool
    sampler: PriorSampler

    def sample(
        self, context: pd.DataFrame, n: int, generator: np.random.Generator
    ) -> np.ndarray:
        """Sample from a conditional prior condition."""
        return self.sampler.sample(context, n, generator)


class ConditionalPrior(BaseModel):
    """A conditional prior."""

    source_feature: str
    conditions: list[ConditionalPriorCondition]
    fallback_prior: PriorSampler | None

    def sample(self, context: pd.DataFrame, n: int, generator: np.random.Generator):
        """Sample from a conditional prior."""
        conditional_samples = {
            c.match_val: c.sampler.sample(context, n, generator)
            for c in self.conditions
        }
        test_feature = context[self.source_feature].to_numpy()

        final = np.full(n, np.nan)

        for match_val, samples_for_match_val in conditional_samples.items():
            mask = test_feature == match_val
            final = np.where(mask, samples_for_match_val, final)

        if self.fallback_prior is not None:
            mask = np.isnan(final)
            final = np.where(
                mask, self.fallback_prior.sample(context, n, generator), final
            )

        if (final == np.nan).any():
            msg = (
                "Final array contains NaN values; possibly due to an unmatched value for "
                f"feature {self.source_feature}."
            )
            raise SamplingError(msg)

        return final


class UnconditionalPrior(BaseModel):
    """An unconditional prior."""

    sampler: PriorSampler

    def sample(self, context: pd.DataFrame, n: int, generator: np.random.Generator):
        """Sample from an unconditional prior."""
        return self.sampler.sample(context, n, generator)


Prior = UnconditionalPrior | ConditionalPrior


class Priors(BaseModel):
    """A collection of priors."""

    sampled_features: dict[str, Prior]

    def sample(self, context: pd.DataFrame, n: int, generator: np.random.Generator):
        """Sample from the priors."""
        working_df = context.copy(deep=True)
        for feature, prior in self.sampled_features.items():
            working_df[feature] = prior.sample(working_df, n, generator)
        return working_df
