"""Train an sbem model for a specific fold."""

import math
import tempfile
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import boto3
import numpy as np
import pandas as pd
import yaml
from pydantic import AnyUrl, BaseModel, Field, model_validator

from epengine.models.shoebox_sbem import BasementAtticOccupationConditioningStatus
from epengine.models.transforms import (
    CategoricalFeature,
    ContinuousFeature,
    RegressorInputSpec,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client as S3ClientType
else:
    S3ClientType = object


from epinterface.sbem.fields.spec import (
    CategoricalFieldSpec,
    NumericFieldSpec,
    SemanticModelFields,
)

from epengine.models.base import BaseSpec, LeafSpec
from epengine.models.mixins import WithBucket
from epengine.models.outputs import URIResponse
from epengine.utils.filesys import fetch_uri


# TODO: add tracking of "WORST" in each.
class ConvergenceThresholds(BaseModel):
    """The thresholds for convergence."""

    mae: float = Field(default=0.5, description="The maximum MAE for convergence.")
    rmse: float = Field(default=0.5, description="The maximum RMSE for convergence.")
    mape: float = Field(default=0.15, description="The maximum MAPE for convergence.")
    r2: float = Field(default=0.95, description="The minimum R2 for convergence.")
    cvrmse: float = Field(
        default=0.05, description="The maximum CV_RMSE for convergence."
    )

    @property
    def thresholds(self) -> pd.Series:
        """The thresholds for convergence."""
        return pd.Series(self.model_dump(), name="metric")

    def check_convergence(self, metrics: pd.Series):
        """Check if the metrics have converged.

        Note that this requires the metrics data frame to have the following shape:

        """
        thresholds = pd.Series(self.model_dump(), name="metric")

        # first, we will select the appropriate threshold for each metric
        comparators = thresholds.loc[metrics.index.get_level_values("metric")]
        # we can then copy over the index safely
        comparators.index = metrics.index

        # next, we will flip the sign of the r2 metric since it is a maximization metric rather thin min
        metrics = metrics * np.where(
            metrics.index.get_level_values("metric") == "r2", -1, 1
        )
        comparators = comparators * np.where(
            comparators.index.get_level_values("metric") == "r2", -1, 1
        )

        # run the comparisons
        comparison = metrics < comparators

        # now we will groupby the stratum (e.g. features.weather.file)
        # and by the target (e.g. Electricity, Gas, etc.)
        # we are converged if any of the metrics have converged for that target
        # in that stratum
        comparison_stratum_and_target = comparison.groupby(
            level=[lev for lev in comparison.index.names if lev != "metric"]
        ).any()

        # then we will check that all targets have converged for each stratum
        comparison_strata = comparison_stratum_and_target.groupby(level="stratum").all()

        # finally, we will check that all strata have converged
        comparison_all = comparison_strata.all()

        return (
            comparison_all,
            comparison_strata,
            comparison_stratum_and_target,
            comparison,
        )


# TODO: pass this to train step.
class XGBHyperparameters(BaseModel):
    """The parameters for the xgboost model."""

    max_depth: int = Field(default=5, description="The maximum depth of the tree.")
    eta: float = Field(default=0.1, description="The learning rate.")
    min_child_weight: int = Field(default=3, description="The minimum child weight.")
    subsample: float = Field(default=0.8, description="The subsample rate.")
    colsample_bytree: float = Field(
        default=0.8, description="The column sample by tree rate."
    )
    alpha: float = Field(default=0.01, description="The alpha parameter.")
    lam: float = Field(default=0.01, description="The lambda parameter.")
    gamma: float = Field(default=0.01, description="The gamma parameter.")


ModelHPType = XGBHyperparameters


class StratificationSpec(BaseModel):
    """The stratification spec."""

    field: str = Field(
        default="feature.weather.file", description="The field to stratify the data by."
    )
    sampling: Literal["equal", "error-weighted", "proportional"] = Field(
        default="equal",
        description="The sampling method to use over the strata.",
    )
    aliases: list[str] = Field(
        default_factory=lambda: ["epwzip_path", "epw_path"],
        description="The alias to use for the stratum as a fallback.",
    )

    # TODO: consider allowing the stratification to be a compound with e.g. component_map_uri and semantic_fields_uri and database_uri


class CrossValidationSpec(BaseModel):
    """The cross validation spec."""

    n_folds: int = Field(
        default=5, description="The number of folds for the entire parent task."
    )


class IterationSpec(BaseModel):
    """The iteration spec."""

    n_init: int = Field(default=10000, description="The number of initial samples.")
    min_per_stratum: int = Field(
        default=100, description="The minimum number of samples per stratum."
    )
    n_per_iter: int = Field(
        default=10000,
        description="The number of samples to add per each iteration of the outer loop.",
    )
    max_iters: int = Field(
        default=100,
        description="The maximum number of outer loop iterations to perform.",
    )
    recursion_factor: int = Field(
        default=3, description="The subdivision factor for recursive scatter gathers."
    )
    recursion_max_depth: int = Field(
        default=1, description="The maximum depth of recursion for scatter gathers."
    )


class ProgressiveTrainingSpec(BaseSpec, WithBucket):
    """A spec for iteratively training an SBEM regression model."""

    convergence_criteria: ConvergenceThresholds = Field(
        default_factory=ConvergenceThresholds,
        description="The convergence criteria.",
    )
    model_hyperparameters: ModelHPType = Field(
        default_factory=ModelHPType,
        description="The hyperparameters for the model.",
    )
    stratification: StratificationSpec = Field(
        default_factory=StratificationSpec,
        description="The stratification spec.",
    )
    cross_val: CrossValidationSpec = Field(
        default_factory=CrossValidationSpec,
        description="The cross validation spec.",
    )
    iteration: IterationSpec = Field(
        default_factory=IterationSpec,
        description="The iteration spec.",
    )
    gis_uri: AnyUrl = Field(
        ...,
        description="The uri of the gis data to train on.",
    )
    component_map_uri: AnyUrl = Field(
        ...,
        description="The uri of the component map to train on.",
    )
    semantic_fields_uri: AnyUrl = Field(
        ...,
        description="The uri of the semantic fields to train on.",
    )
    database_uri: AnyUrl = Field(
        ...,
        description="The uri of the database to train on.",
    )

    @property
    def gis_path(self) -> Path:
        """The path to the gis data."""
        return self.fetch_uri(self.gis_uri)

    @cached_property
    def gis_data(self) -> pd.DataFrame:
        """Load the gis data."""
        return pd.read_parquet(self.gis_path)

    @property
    def semantic_fields_path(self) -> Path:
        """The path to the semantic fields data."""
        return self.fetch_uri(self.semantic_fields_uri)

    @cached_property
    def semantic_fields_data(self) -> SemanticModelFields:
        """Load the semantic fields data."""
        with open(self.semantic_fields_path) as f:
            return SemanticModelFields.model_validate(yaml.safe_load(f))

    def s3_key_for_iteration(self, iteration_ix: int) -> str:
        """The s3 root key for the iteration."""
        return f"{self.experiment_id}/iter-{iteration_ix:03d}"

    def upload_self(self, s3_client: S3ClientType):
        """Upload a dumpout of this spec to the s3 bucket root."""
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            fpath = tempdir / "spec.yml"
            with open(fpath, "w") as f:
                yaml.dump(self.model_dump(mode="json"), f, indent=2)
            s3_client.upload_file(
                fpath.as_posix(),
                self.bucket,
                f"hatchet/{self.experiment_id}/artifacts/experiment-spec.yml",
            )


class StageSpec(BaseModel):
    """A spec that is common to both the sample and train stages (and possibly others)."""

    progressive_training_spec: ProgressiveTrainingSpec = Field(
        ...,
        description="The progressive training spec.",
    )
    progressive_training_iteration_ix: int = Field(
        ...,
        description="The index of the current training iteration within the outer loop.",
    )
    data_uri: AnyUrl | None = Field(
        ...,
        description="The uris of the previous simulation results to sample from.",
    )
    stage_type: Literal["sample", "train"] = Field(
        ...,
        description="The type of stage.",
    )

    @cached_property
    def random_generator(self) -> np.random.Generator:
        """The random generator."""
        return np.random.default_rng(self.progressive_training_iteration_ix)

    @cached_property
    def experiment_key(self) -> str:
        """The root key for the experiment."""
        return f"{self.progressive_training_spec.s3_key_for_iteration(self.progressive_training_iteration_ix)}/{self.stage_type}"

    def load_previous_data(self, s3_client: S3ClientType) -> pd.DataFrame | None:
        """Load the previous data."""
        if self.data_uri is None:
            return None
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fpath = tmpdir / "previous_data.parquet"
            fetch_uri(
                uri=self.data_uri,
                local_path=fpath,
                use_cache=False,
                s3=s3_client,
            )
            df = pd.read_parquet(fpath)
        return df


class SampleSpec(StageSpec):
    """A spec for thhe sampling stage of the progressive training."""

    # TODO: add the ability to receive the last set of error metrics and use them to inform the sampling

    def stratified_selection(self) -> pd.DataFrame:
        """Sample the gis data."""
        df = self.progressive_training_spec.gis_data

        stratification_field = self.progressive_training_spec.stratification.field
        stratification_aliases = self.progressive_training_spec.stratification.aliases

        if stratification_field not in df.columns and not any(
            alias in df.columns for alias in stratification_aliases
        ):
            msg = f"Stratification field {stratification_field} not found in gis data.  Please check the field name and/or the aliases."
            raise ValueError(msg)

        if stratification_field not in df.columns:
            stratification_field = next(
                alias for alias in stratification_aliases if alias in df.columns
            )

        strata = cast(list[str], df[stratification_field].unique().tolist())

        if self.progressive_training_spec.stratification.sampling == "equal":
            return self.sample_equally_by_stratum(df, strata, stratification_field)
        elif self.progressive_training_spec.stratification.sampling == "error-weighted":
            msg = "Error-weighted sampling is not yet implemented."
            raise NotImplementedError(msg)
        elif self.progressive_training_spec.stratification.sampling == "proportional":
            msg = "Proportional sampling is not yet implemented."
            raise NotImplementedError(msg)
        else:
            msg = f"Invalid sampling method: {self.progressive_training_spec.stratification.sampling}"
            raise ValueError(msg)

    def sample_equally_by_stratum(
        self, df: pd.DataFrame, strata: list[str], stratification_field: str
    ) -> pd.DataFrame:
        """Sample equally by stratum.

        This will break the dataframe up into n strata and ensure that each strata ends up with the same number of samples.

        Args:
            df (pd.DataFrame): The dataframe to sample from.
            strata (list[str]): The unique values of the strata.
            stratification_field (str): The field to stratify the data by.

        Returns:
            samples (pd.DataFrame): The sampled dataframe.
        """
        stratum_dfs = {
            stratum: df[df[stratification_field] == stratum] for stratum in strata
        }
        n_per_iter = (
            self.progressive_training_spec.iteration.n_per_iter
            if self.progressive_training_iteration_ix != 0
            else self.progressive_training_spec.iteration.n_init
        )
        n_per_stratum = max(
            n_per_iter // len(strata),
            (
                self.progressive_training_spec.iteration.min_per_stratum
                if self.progressive_training_iteration_ix == 0
                else 0
            ),
        )

        # TODO: consider how we want to handle potentially having the same geometry appear in both
        # the training and testing sets.
        # if any(len(stratum_df) < n_per_stratum for stratum_df in stratum_dfs.values()):
        #     msg = "There are not enough buildings in some strata to sample the desired number of buildings per stratum."
        #     # connsider making this a warning?
        #     raise ValueError(msg)

        sampled_strata = {
            stratum: stratum_df.sample(
                n=n_per_stratum, random_state=self.random_generator, replace=True
            )
            for stratum, stratum_df in stratum_dfs.items()
        }
        return cast(pd.DataFrame, pd.concat(sampled_strata.values()))

    def sample_semantic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the semantic fields."""
        # TODO: consider randomizing the locations?
        semantic_fields = self.progressive_training_spec.semantic_fields_data
        for field in semantic_fields.Fields:
            if isinstance(field, CategoricalFieldSpec):
                options = field.Options
                df[field.Name] = self.random_generator.choice(options, size=len(df))
            elif isinstance(field, NumericFieldSpec):
                df[field.Name] = self.random_generator.uniform(
                    field.Min, field.Max, size=len(df)
                )
            else:
                msg = f"Invalid field type: {type(field)}"
                raise TypeError(msg)
        return df

    def sample_basements_and_attics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basement/attics to models."""
        # get the options for the type literal
        options: list[BasementAtticOccupationConditioningStatus] = [
            "none",
            "occupied_unconditioned",
            "unoccupied_unconditioned",
            "occupied_conditioned",
            "unoccupied_conditioned",
        ]
        weights = [0.5, *([0.5 / 4] * 4)]
        # sample the type literal
        df["basement"] = self.random_generator.choice(options, size=len(df), p=weights)
        df["attic"] = self.random_generator.choice(options, size=len(df), p=weights)
        return df

    def sample_wwrs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the wwrs."""
        wwr_min = 0.05
        wwr_max = 0.35
        df["wwr"] = self.random_generator.uniform(wwr_min, wwr_max, size=len(df))
        return df

    def sample_f2f_heights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the f2f heights."""
        f2f_min = 2.3
        f2f_max = 4.3
        df["f2f_height"] = self.random_generator.uniform(f2f_min, f2f_max, size=len(df))
        return df

    def to_sim_specs(self, df: pd.DataFrame):
        """Convert the sampled dataframe to a list of simulation specs.

        For now, we are assuming that all the other necessary fields are present and we are just
        ensuring that sort_index and experiment_id are set appropriately.
        """
        df["semantic_field_context"] = df.apply(
            lambda row: {
                field.Name: row[field.Name]
                for field in self.progressive_training_spec.semantic_fields_data.Fields
            },
            axis=1,
        )
        df["sort_index"] = np.arange(len(df))
        df["experiment_id"] = self.experiment_key
        # TODO: consider allowing the component map/semantic_fields/database to be inherited from the row
        # e.g. to allow multiple component maps and dbs per run.
        df["component_map_uri"] = str(self.progressive_training_spec.component_map_uri)
        df["semantic_fields_uri"] = str(
            self.progressive_training_spec.semantic_fields_uri
        )
        df["db_uri"] = str(self.progressive_training_spec.database_uri)
        return df

    def make_payload(self, s3_client: S3ClientType):
        """Make the payload for the scatter gather task, including generating the simulation specs and serializing them to s3."""
        df = self.stratified_selection()
        df = self.sample_semantic_fields(df)
        df = self.sample_basements_and_attics(df)
        df = self.sample_wwrs(df)
        df = self.sample_f2f_heights(df)
        df = self.to_sim_specs(df)
        # serialize to a parquet file and upload to s3
        bucket = self.progressive_training_spec.bucket
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fpath = tmpdir / "specs.pq"
            df.to_parquet(fpath)
            key = f"hatchet/{self.experiment_key}/specs.pq"
            specs_uri = f"s3://{bucket}/{key}"
            s3_client.upload_file(fpath.as_posix(), bucket, key)

        payload = {
            "specs": specs_uri,
            "bucket": bucket,
            "workflow_name": "simulate_sbem_shoebox",
            "experiment_id": self.experiment_key,
            "recursion_map": {
                "factor": self.progressive_training_spec.iteration.recursion_factor,
                "max_depth": self.progressive_training_spec.iteration.recursion_max_depth,
            },
        }
        return payload

    def combine_results(self, new_data_uri: URIResponse, s3_client: S3ClientType):
        """Combine the results of the previous and new data."""
        previous_data = self.load_previous_data(s3_client)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fpath = tmpdir / "new_data.parquet"
            fetch_uri(
                uri=new_data_uri.uri, local_path=fpath, use_cache=False, s3=s3_client
            )
            # TODO: data frame subsection selection should be a configuration option within the
            # progressive iteration training spec.
            df = cast(
                pd.DataFrame,
                cast(pd.DataFrame, pd.read_hdf(fpath, key="results")),
            )
        if previous_data is not None:
            df = pd.concat([previous_data, df], axis=0)

        # strip out any constant columns
        is_all_zeros = (df.max(axis=0) - df.min(axis=0)).abs() < 1e-5
        df = df.loc[:, ~is_all_zeros]
        # serialize to a parquet file and upload to s3
        bucket = self.progressive_training_spec.bucket
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fpath = tmpdir / "results.parquet"
            df.to_parquet(fpath)
            key = f"hatchet/{self.experiment_key}/full-dataset.pq"
            specs_uri = f"s3://{bucket}/{key}"
            s3_client.upload_file(fpath.as_posix(), bucket, key)
        return specs_uri

    @model_validator(mode="after")
    def check_stage(self):
        """The sampling spec must have stage set to 'sample'."""
        if self.stage_type != "sample":
            msg = f"Invalid stage: {self.stage_type}"
            raise ValueError(msg)
        return self


class TrainFoldSpec(LeafSpec):
    """Train an sbem model for a specific fold.

    The fold is determined by the sort_index, which does mean we need to know the n_folds.

    We will need to know:
    - where the data is
    - the desired stratification (e.g. feature.weather.file)
    - how to divide the data into training and testing splits given the desired stratification

    The data uri should be assumed to have features in the index and targets in the columns.

    TODO: consider the potential for leakage when a stratum has few buildings!

    First, we will subdivide the data into its strata.

    Then for each stratum, we will create a train/test split according to the fold index.

    We wish to return validation metrics with the following hierarchy for the column index
    - train/test ["split_segment"]
    - loc1/loc2 ... ["stratum"]
    - mae/rmse/r2/... ["metric"]

    Theoretically, we also might want to pass in normalization specifications for features and/or targets.
    However, with xgb, this is less imperative.
    """

    n_folds: int = Field(
        ..., description="The number of folds for the entire parent task."
    )
    data_uri: AnyUrl = Field(..., description="The uri of the data to train on.")
    stratification_field: str = Field(
        ...,
        description="The field to stratify the data by for monitoring convergence in parent task.",
    )
    progressive_training_iter_ix: int = Field(
        ...,
        description="The index of the current training iteration within the outer loop.",
    )
    bucket: str = Field(..., description="The bucket to save the models to.")

    @property
    def data_path(self) -> Path:
        """The path to the data."""
        return self.fetch_uri(self.data_uri)

    @cached_property
    def data(self) -> pd.DataFrame:
        """The data."""
        df = pd.read_parquet(self.data_path)["Raw"]
        df = cast(
            pd.DataFrame,
            (
                df.T.groupby(
                    level=[lev for lev in df.columns.names if lev.lower() != "month"]
                )
                .sum()
                .T
            ),
        )
        # TODO: should we assume they are shuffled already?
        # shuffle the order of the rows
        df = df.sample(frac=1, random_state=42, replace=False)
        return df

    @cached_property
    def dparams(self) -> pd.DataFrame:
        """The index of the data."""
        return self.data.index.to_frame()

    @cached_property
    def stratum_names(self) -> list[str]:
        """The values of the stratification field."""
        return sorted(self.dparams[self.stratification_field].unique().tolist())

    @cached_property
    def data_by_stratum(self) -> dict[str, pd.DataFrame]:
        """Subdivide the data by the stratification field.

        We want 1/n_folds data in the test segment for each stratification option,
        so we will need to compute train/test splits separately for each stratum.

        This would not be necessary if we knew that the strata always had equal representation, but
        since we might use things like adaptive sampling or generating samples proportionally to the number of buildings in that stratum,
        e.g. by population, then what *could* happen if we just did a random train/test split is that some strata might end up
        entirely in the train set.
        """
        return {
            val: cast(
                pd.DataFrame, self.data[self.dparams[self.stratification_field] == val]
            )
            for val in self.stratum_names
        }

    @cached_property
    def train_test_split_by_fold_and_stratum(self) -> pd.DataFrame:
        """Create the folds for the data.

        To do this, we will go to each stratum and use a strided step to
        construct each fold, then assign the fold matching the sort_index
        to the test split.  We also recombine the strata since they are now
        safely stratified.
        """
        all_strata = []
        for val in self.stratum_names:
            folds = []
            for i in range(self.n_folds):
                fold = self.data_by_stratum[val].iloc[i :: self.n_folds]
                folds.append(fold)
            folds_df = pd.concat(
                folds,
                axis=0,
                keys=[
                    "test" if i == self.sort_index else "train"
                    for i in range(self.n_folds)
                ],
                names=["split_segment"],
            )
            all_strata.append(folds_df)
        return pd.concat(all_strata)

    @cached_property
    def train_segment(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get the training segment."""
        train_df = cast(
            pd.DataFrame,
            self.train_test_split_by_fold_and_stratum.xs(
                "train", level="split_segment"
            ),
        )
        params = train_df.index.to_frame(index=False)
        targets = train_df
        return params, targets

    @cached_property
    def test_segment(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get the test segment."""
        test_df = cast(
            pd.DataFrame,
            self.train_test_split_by_fold_and_stratum.xs("test", level="split_segment"),
        )
        params = test_df.index.to_frame(index=False)
        targets = test_df
        return params, targets

    @cached_property
    def non_numeric_options(self) -> dict[str, list[str]]:
        """Get the non-numeric options for categorical features.

        We must perform this across the entire dataset not just splits for consistency
        and to ensure we get all options.

        TODO: In the future, this should be based off of transform instructions.
        """
        fparams = self.dparams[
            [col for col in self.dparams.columns if col.startswith("feature.")]
        ]
        non_numeric_cols = fparams.select_dtypes(include=["object"]).columns
        non_numeric_options = {
            col: sorted(cast(pd.Series, fparams[col]).unique().tolist())
            for col in non_numeric_cols
        }
        return non_numeric_options

    @cached_property
    def numeric_min_maxs(self) -> dict[str, tuple[float, float]]:
        """Get the min and max for numeric features.

        We perform this only on the training set to prevent leakage.

        TODO: In the future, this should be based off of transform instructions.

        Args:
            params (pd.DataFrame): The parameters to get the min and max for.

        Returns:
            norm_bounds (dict[str, tuple[float, float]]): The min and max for each numeric feature.
        """
        params, _ = self.train_segment
        fparams = params[[col for col in params.columns if col.startswith("feature.")]]
        numeric_cols = fparams.select_dtypes(include=["number"]).columns
        numeric_min_maxs = {
            col: (float(fparams[col].min()), float(fparams[col].max()))
            for col in numeric_cols
        }
        for col in numeric_min_maxs:
            low, high = numeric_min_maxs[col]
            # we want to floor the "low" value down to the nearest 0.001
            # and ceil the "high" value up to the nearest 0.001
            # e.g. if low is -0.799, we want to set it to -0.800
            # and if high is 0.799, we want to set it to 0.800
            numeric_min_maxs[col] = (
                math.floor(low * 1000) / 1000,
                math.ceil(high * 1000) / 1000,
            )
        return numeric_min_maxs

    @cached_property
    def feature_spec(self) -> RegressorInputSpec:
        """Get the feature spec which can be serialized and reloaded."""
        params, _ = self.train_segment
        features: list[CategoricalFeature | ContinuousFeature] = []
        for col in params.columns:
            if col in self.numeric_min_maxs:
                low, high = self.numeric_min_maxs[col]
                features.append(
                    ContinuousFeature(name=col, min=float(low), max=float(high))
                )
            elif col in self.non_numeric_options:
                opts = self.non_numeric_options[col]
                features.append(CategoricalFeature(name=col, values=opts))
        return RegressorInputSpec(features=features)

    def normalize_params(self, params: pd.DataFrame) -> pd.DataFrame:
        """Normalize the params."""
        regressor_spec = self.feature_spec
        fparams = regressor_spec.transform(params, do_check=False)
        return fparams

    def run(
        self,
    ):
        """Train the model."""
        train_params, train_targets = self.train_segment
        test_params, test_targets = self.test_segment

        # select/transform the params as necessary
        train_params = self.normalize_params(train_params)
        test_params = self.normalize_params(test_params)

        # Train the model
        # train_preds, test_preds = self.train_xgboost(
        #     train_params, train_targets, test_params, test_targets
        # )
        s3_client = boto3.client("s3")
        train_preds, test_preds = self.train_lightgbm(
            train_params, train_targets, test_params, test_targets, s3_client
        )

        # compute the metrics
        global_train_metrics, stratum_train_metrics = self.compute_metrics(
            train_preds, train_targets
        )
        global_test_metrics, stratum_test_metrics = self.compute_metrics(
            test_preds, test_targets
        )

        global_metrics = pd.concat(
            [global_train_metrics, global_test_metrics],
            axis=1,
            keys=["train", "test"],
            names=["split_segment"],
        )
        stratum_metrics = pd.concat(
            [stratum_train_metrics, stratum_test_metrics],
            axis=1,
            keys=["train", "test"],
            names=["split_segment"],
        )
        return {
            "global_metrics": global_metrics,
            "stratum_metrics": stratum_metrics,
        }

    def compute_frame_metrics(
        self, preds: pd.DataFrame, targets: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the metrics."""
        from sklearn.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            mean_squared_error,
            r2_score,
        )

        mae = mean_absolute_error(targets, preds, multioutput="raw_values")
        mse = mean_squared_error(targets, preds, multioutput="raw_values")
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, preds, multioutput="raw_values")
        cvrmse = rmse / (targets.mean(axis=0) + 1e-5)
        mape = mean_absolute_percentage_error(
            targets + 1e-5,
            preds,
            multioutput="raw_values",
        )

        metrics = pd.DataFrame(
            {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "cvrmse": cvrmse,
                "mape": mape,
            },
        )
        metrics.columns.names = ["metric"]
        metrics.index.names = ["target"]
        return metrics

    def compute_metrics(self, preds: pd.DataFrame, targets: pd.DataFrame):
        """Compute the metrics."""
        global_metrics = self.compute_frame_metrics(preds, targets)
        stratum_metric_dfs = {}
        for stratum_name in self.stratum_names:
            stratum_targets = cast(
                pd.DataFrame, targets.xs(stratum_name, level=self.stratification_field)
            )
            stratum_preds = cast(
                pd.DataFrame, preds.xs(stratum_name, level=self.stratification_field)
            )
            metrics = self.compute_frame_metrics(stratum_preds, stratum_targets)
            stratum_metric_dfs[stratum_name] = metrics

        stratum_metrics = pd.concat(
            stratum_metric_dfs,
            axis=1,
            keys=self.stratum_names,
            names=["stratum"],
        )
        global_metrics = (
            global_metrics.set_index(
                pd.Index(
                    [self.sort_index] * len(global_metrics),
                    name="sort_index",
                ),
                append=True,
            )
            .set_index(
                pd.Index(
                    [self.progressive_training_iter_ix] * len(global_metrics),
                    name="progressive_training_iter_ix",
                ),
                append=True,
            )
            .unstack(level="target")
        )

        stratum_metrics = (
            stratum_metrics.set_index(
                pd.Index(
                    [self.sort_index] * len(stratum_metrics),
                    name="sort_index",
                ),
                append=True,
            )
            .set_index(
                pd.Index(
                    [self.progressive_training_iter_ix] * len(stratum_metrics),
                    name="progressive_training_iter_ix",
                ),
                append=True,
            )
            .unstack(level="target")
        )
        return global_metrics, stratum_metrics

    def train_lightgbm(
        self,
        train_params: pd.DataFrame,
        train_targets: pd.DataFrame,
        test_params: pd.DataFrame,
        test_targets: pd.DataFrame,
        s3_client: S3ClientType | None = None,
    ):
        """Train the lightgbm model."""
        import lightgbm as lgb

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
        }
        test_preds = {}
        train_preds = {}
        for col in train_targets.columns:
            lgb_train_data = lgb.Dataset(train_params, label=train_targets[col])
            lgb_test_data = lgb.Dataset(test_params, label=test_targets[col])
            model = lgb.train(
                lgb_params,
                lgb_train_data,
                num_boost_round=4000,
                valid_sets=[lgb_test_data],
                valid_names=["eval"],
                callbacks=[lgb.early_stopping(20)],
            )
            test_preds[col] = pd.Series(
                cast(np.ndarray, model.predict(test_params)),
                index=test_targets.index,
                name=col,
            )
            train_preds[col] = pd.Series(
                cast(np.ndarray, model.predict(train_params)),
                index=train_targets.index,
                name=col,
            )
            if s3_client is not None:
                model_key = self.format_model_key(f"{col}.lgb")
                model_str = model.model_to_string()
                s3_client.put_object(Bucket=self.bucket, Key=model_key, Body=model_str)

        if s3_client is not None:
            import yaml

            space_key = self.format_model_key("space.yml")
            space_str = yaml.dump(
                self.feature_spec.model_dump(mode="json"), indent=2, sort_keys=False
            )
            s3_client.put_object(Bucket=self.bucket, Key=space_key, Body=space_str)

        test_preds = pd.concat(test_preds, axis=1)
        train_preds = pd.concat(train_preds, axis=1)
        return train_preds, test_preds

    @property
    def model_dir_key(self) -> str:
        """Get the key for the model directory."""
        return f"{self.experiment_id}/{self.sort_index}/models"

    def format_model_key(self, model_name: str) -> str:
        """Format the model key."""
        return f"hatchet/{self.model_dir_key}/{model_name}"

    def train_xgboost(
        self,
        train_params: pd.DataFrame,
        train_targets: pd.DataFrame,
        test_params: pd.DataFrame,
        test_targets: pd.DataFrame,
    ):
        """Train the xgboost model."""
        import xgboost as xgb

        hparams = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 5,  # 7
            "eta": 0.1,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            # "alpha": 0.01,
            # "lambda": 0.01,
            # "gamma": 0.01,
        }

        train_dmatrix = xgb.DMatrix(train_params, label=train_targets)
        test_dmatrix = xgb.DMatrix(test_params, label=test_targets)

        model = xgb.train(
            hparams,
            train_dmatrix,
            num_boost_round=2000,
            early_stopping_rounds=20,
            verbose_eval=True,
            evals=[(test_dmatrix, "test")],
        )

        # compute the metrics
        train_preds = model.predict(train_dmatrix)
        test_preds = model.predict(test_dmatrix)
        train_preds = pd.DataFrame(
            train_preds, index=train_targets.index, columns=train_targets.columns
        )
        test_preds = pd.DataFrame(
            test_preds, index=test_targets.index, columns=test_targets.columns
        )

        return train_preds, test_preds


class TrainWithCVSpec(StageSpec):
    """Train an SBEM model using a scatter gather approach for cross-fold validation."""

    @model_validator(mode="after")
    def check_stage(self):
        """The training spec must have stage set to 'train'."""
        if self.stage_type != "train":
            msg = f"Invalid stage: {self.stage_type}"
            raise ValueError(msg)
        return self

    @property
    def schedule(self) -> list[TrainFoldSpec]:
        """Create the task schedule."""
        schedule = []
        data_uri = self.data_uri
        if data_uri is None:
            msg = "Data URI is required for training."
            raise ValueError(msg)

        for i in range(self.progressive_training_spec.cross_val.n_folds):
            schedule.append(
                TrainFoldSpec(
                    # TODO: this should be set in a better manner
                    experiment_id=self.experiment_key,
                    sort_index=i,
                    n_folds=self.progressive_training_spec.cross_val.n_folds,
                    data_uri=data_uri,
                    stratification_field=self.progressive_training_spec.stratification.field,
                    progressive_training_iter_ix=self.progressive_training_iteration_ix,
                    bucket=self.progressive_training_spec.bucket,
                )
            )
        return schedule

    def allocate(self, s3_client: S3ClientType):
        """Allocate the task."""
        # 1. turn the schedule into a parquet dataframe
        df = pd.DataFrame([m.model_dump() for m in self.schedule])
        bucket = self.progressive_training_spec.bucket
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir) / "train_specs.parquet"
            df.to_parquet(temp_path)
            key = f"hatchet/{self.experiment_key}/train_specs.parquet"
            specs_uri = f"s3://{bucket}/{key}"
            s3_client.upload_file(temp_path.as_posix(), bucket, key)

        payload = {
            "specs": specs_uri,
            "bucket": bucket,
            # TODO: this should be selected in a better manner.
            "workflow_name": "train_regressor_with_cv_fold",
            "experiment_id": self.experiment_key,
        }
        return payload

    def check_convergence(self, uri: URIResponse, s3_client: S3ClientType):
        """Check the convergence of the training."""
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            results_path = tempdir / "results.hdf"
            # download the results from s3
            fetch_uri(uri.uri, local_path=results_path, use_cache=False, s3=s3_client)
            results = cast(
                pd.DataFrame, pd.read_hdf(results_path, key="stratum_metrics")
            )

        fold_averages = cast(
            pd.Series,
            results.xs(
                "test",
                level="split_segment",
                axis=1,
            ).mean(axis=0),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            fold_averages_path = Path(tempdir) / "fold-averaged-errors.pq"
            fold_averages.to_frame(
                name=self.progressive_training_iteration_ix
            ).to_parquet(fold_averages_path)
            key = f"hatchet/{self.experiment_key}/fold-averaged-errors.pq"
            bucket = self.progressive_training_spec.bucket
            s3_client.upload_file(fold_averages_path.as_posix(), bucket, key)

        (
            convergence_all,
            convergence_monitor_segment,
            convergence_monitor_segment_and_target,
            convergence,
        ) = self.progressive_training_spec.convergence_criteria.check_convergence(
            fold_averages
        )

        return convergence_all, convergence
