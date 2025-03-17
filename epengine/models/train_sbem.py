"""Train an sbem model for a specific fold."""

import tempfile
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from hatchet_sdk import Context
from pydantic import AnyUrl, BaseModel, Field
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client as S3ClientType
else:
    S3ClientType = object


from epengine.models.base import BaseSpec
from epengine.models.leafs import LeafSpec
from epengine.models.mixins import WithBucket
from epengine.models.outputs import URIResponse


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


class CrossValidationSpec(BaseModel):
    """The cross validation spec."""

    n_folds: int = Field(
        default=5, description="The number of folds for the entire parent task."
    )


class IterationSpec(BaseModel):
    """The iteration spec."""

    n_init: int = Field(default=10000, description="The number of initial samples.")
    n_per_iter: int = Field(
        default=10000,
        description="The number of samples to add per each iteration of the outer loop.",
    )
    max_iters: int = Field(
        default=100,
        description="The maximum number of outer loop iterations to perform.",
    )
    max_samples: int = Field(
        default=1_000_000, description="The maximum number of samples to collect."
    )
    # max_time: float = Field(
    #     default=60 * 60 * 12, description="The maximum time to run the outer loop."
    # )


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

    @property
    def data_path(self) -> Path:
        """The path to the data."""
        return self.fetch_uri(self.data_uri)

    @cached_property
    def data(self) -> pd.DataFrame:
        """The data."""
        df = pd.read_parquet(self.data_path)
        # shuffle the order of the rows
        df = df.sample(frac=1, random_state=42)
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
            col: (cast(float, fparams[col].min()), cast(float, fparams[col].max()))
            for col in numeric_cols
        }
        return numeric_min_maxs

    def normalize_params(self, params: pd.DataFrame) -> pd.DataFrame:
        """Normalize the params."""
        fparams = cast(
            pd.DataFrame,
            params[[col for col in params.columns if col.startswith("feature.")]],
        )
        for col in fparams.columns:
            if col in self.numeric_min_maxs:
                min_val, max_val = self.numeric_min_maxs[col]
                fparams[col] = (fparams[col] - min_val) / (max_val - min_val)
            elif col in self.non_numeric_options:
                unique_vals = self.non_numeric_options[col]
                fparams[col] = pd.Categorical(
                    fparams[col], categories=unique_vals
                ).codes
            else:
                msg = f"Feature {col} is not numeric or categorical and cannot be normalized."
                raise ValueError(msg)
        return fparams

    def run(self):
        """Train the model."""
        train_params, train_targets = self.train_segment
        test_params, test_targets = self.test_segment

        # select/transform the params as necessary
        train_params = self.normalize_params(train_params)
        test_params = self.normalize_params(test_params)

        # Train the model
        train_preds, test_preds = self.train_xgboost(
            train_params, train_targets, test_params, test_targets
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

    def train_xgboost(
        self,
        train_params: pd.DataFrame,
        train_targets: pd.DataFrame,
        test_params: pd.DataFrame,
        test_targets: pd.DataFrame,
    ):
        """Train the xgboost model."""
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


class TrainWithCVSpec(BaseSpec, WithBucket):
    """Train an SBEM model using a scatter gather approach for cross-fold validation."""

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
    thresholds: ConvergenceThresholds = Field(
        default_factory=ConvergenceThresholds,
        description="The thresholds for convergence.",
    )

    @property
    def schedule(self) -> list[TrainFoldSpec]:
        """Create the task schedule."""
        schedule = []
        for i in range(self.n_folds):
            schedule.append(
                TrainFoldSpec(
                    # TODO: this should be set in a better manner
                    experiment_id=self.experiment_id,
                    sort_index=i,
                    n_folds=self.n_folds,
                    data_uri=self.data_uri,
                    stratification_field=self.stratification_field,
                    progressive_training_iter_ix=self.progressive_training_iter_ix,
                )
            )
        return schedule

    async def allocate(self, context: Context, s3_client: S3ClientType):
        """Allocate the task."""
        # 1. turn the schedule into a parquet dataframe
        df = pd.DataFrame([m.model_dump() for m in self.schedule])
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir) / "train_specs.parquet"
            df.to_parquet(temp_path)
            key = f"hatchet/{self.experiment_id}/train_specs.parquet"
            specs_uri = f"s3://{self.bucket}/{key}"
            s3_client.upload_file(temp_path.as_posix(), self.bucket, key)

        payload = {
            "specs": specs_uri,
            "bucket": self.bucket,
            "workflow_name": "train_regressor_with_cv_fold",
            # TODO: this should be selected in a better manner.
            "experiment_id": f"{self.experiment_id}/iter-{self.progressive_training_iter_ix:03d}/train",
        }

        workflowRef = await context.aio.spawn_workflow(
            workflow_name="scatter_gather",
            input=payload,
        )

        context.log("CV Scheduled, waiting for completion...")
        result = await workflowRef.result()
        context.log("CV Completed, collecting results")

        if "collect_children" not in result:
            msg = f"Workflow {workflowRef.workflow_run_id} failed."
            raise RuntimeError(msg)
        collect_children_result = result["collect_children"]
        uri = URIResponse.model_validate(collect_children_result)
        results_path = self.fetch_uri(uri.uri, use_cache=False)
        results = cast(pd.DataFrame, pd.read_hdf(results_path, key="stratum_metrics"))
        fold_averages = cast(
            pd.Series,
            results.xs(
                "test",
                level="split_segment",
                axis=1,
            ).mean(axis=0),
        )
        (
            convergence_all,
            convergence_monitor_segment,
            convergence_monitor_segment_and_target,
            convergence,
        ) = self.thresholds.check_convergence(fold_averages)

        if convergence_all:
            # go to cleanup
            pass
        else:
            # go to sampler
            pass
        context.log("CV Completed, returning results")
        return {"converged": "yas"}
