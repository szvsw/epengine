"""Inference request models.

This module provides a unified system for handling retrofit quantities (costs and incentives)
through a common interface. The system has been refactored to eliminate duplication and
provide better maintainability.

Classes:
- QuantityFactor: Abstract base for all quantity calculations
- FixedQuantity: Fixed amounts (costs or incentives)
- LinearQuantity: Linear calculations based on features
- PercentQuantity: Percentage-based quantities (typically incentives)
- RetrofitQuantity: Container for retrofit interventions
- RetrofitQuantities: Collection of retrofit quantities

The system expects all JSON files to be in the new unified format with quantities and output_key fields.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import boto3
import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, create_model, field_validator
from shapely import Point

from epengine.gis.data.epw_metadata import closest_epw
from epengine.models.sampling import (
    ClippedNormalSampler,
    ConditionalPrior,
    ConditionalPriorCondition,
    CopySampler,
    FixedValueSampler,
    MultiplyValueSampler,
    PowerSampler,
    Prior,
    Priors,
    ProductValuesSampler,
    SumValuesSampler,
    UnconditionalPrior,
    UniformSampler,
)
from epengine.models.shoebox_sbem import (
    BasementAtticOccupationConditioningStatus,
    SBEMSimulationSpec,
)
from epengine.models.transforms import CategoricalFeature, RegressorInputSpec

END_USES = ("Lighting", "Equipment", "DomesticHotWater", "Heating", "Cooling")
FUELS = ("Oil", "NaturalGas", "Electricity")
DATASETS = (
    "Raw",
    "EndUse",
    "Fuel",
    "EndUseCost",
    "EndUseEmissions",
    "FuelCost",
    "FuelEmissions",
)
NORMALIZATIONS = ("Normalized", "Gross")

DATASET_SEGMENT_MAP = {
    "Raw": END_USES,
    "EndUse": END_USES,
    "Fuel": FUELS,
    "EndUseCost": END_USES,
    "EndUseEmissions": END_USES,
    "FuelCost": FUELS,
    "FuelEmissions": FUELS,
}
UNITS_DENOM = {
    "Normalized": "/m2",
    "Gross": "",
}
UNITS_NUMER = {
    "Raw": "kWh",
    "EndUse": "kWh",
    "Fuel": "kWh",
    "EndUseCost": "USD",
    "EndUseEmissions": "tCO2e",
    "FuelCost": "USD",
    "FuelEmissions": "tCO2e",
}
PERCENTILES = {
    0.05: ("p5", "5%"),
    0.1: ("p10", "10%"),
    0.25: ("p25", "25%"),
    0.5: ("p50", "50%"),
    0.75: ("p75", "75%"),
    0.9: ("p90", "90%"),
    0.95: ("p95", "95%"),
}


@dataclass
class SBEMDistributions:
    """Ensemble results and summaries."""

    features: pd.DataFrame
    disaggregations: pd.DataFrame
    totals: pd.DataFrame
    disaggregations_summary: pd.DataFrame
    totals_summary: pd.DataFrame

    @property
    def serialized(self) -> BaseModel:
        """Serialize the SBEMDistributions dataframes into a SBEMInferenceResponseSpec."""
        percentile_mapper = {v[1]: v[0] for v in PERCENTILES.values()}
        disagg_summary_data_renamed = self.disaggregations_summary.rename(
            index=percentile_mapper,
            columns={
                "Domestic Hot Water": "DomesticHotWater",
            },
        )
        totals_summary_renamed = self.totals_summary.rename(
            index=percentile_mapper,
        )
        disagg_dict_data = {}
        for normalization in NORMALIZATIONS:
            normalization_data = {}
            for dataset in DATASETS:
                segment_data = {}
                for segment in DATASET_SEGMENT_MAP[dataset]:
                    data = disagg_summary_data_renamed.loc[
                        :, (normalization, dataset, segment)
                    ].to_dict()  # pyright: ignore [reportArgumentType, reportCallIssue]
                    data["units"] = UNITS_NUMER[dataset] + UNITS_DENOM[normalization]
                    summary = SummarySpec(**data)
                    segment_data[segment] = summary
                normalization_data[dataset] = segment_data
            disagg_dict_data[normalization] = normalization_data
        disagg_dict = DisaggregationsSpec.model_validate(disagg_dict_data)

        total_dict_data = {}
        for normalization in NORMALIZATIONS:
            normalization_data = {}
            for dataset in DATASETS:
                data = totals_summary_renamed.loc[:, (normalization, dataset)].to_dict()  # pyright: ignore [reportArgumentType, reportCallIssue]
                data["units"] = UNITS_NUMER[dataset] + UNITS_DENOM[normalization]
                summary = SummarySpec(**data)
                normalization_data[dataset] = summary
            total_dict_data[normalization] = normalization_data
        totals_dict = TotalsSpec.model_validate(total_dict_data)

        return SBEMInferenceResponseSpec(
            Disaggregation=disagg_dict,
            Total=totals_dict,
        )


@dataclass
class SBEMRetrofitDistributions:
    """Costs and paybacks."""

    costs: pd.DataFrame
    paybacks: pd.Series
    costs_summary: pd.DataFrame
    paybacks_summary: pd.Series

    @property
    def serialized(self) -> BaseModel:
        """Serialize the SBEMRetrofitDistributions dataframes into a SBEMRetrofitDistributionsSpec."""
        field_specs = {}
        field_datas = {}
        percentile_mapper = {v[1]: v[0] for v in PERCENTILES.values()}
        self.costs_summary.rename(index=percentile_mapper, inplace=True)
        self.paybacks_summary.rename(index=percentile_mapper, inplace=True)

        # Process all columns in the costs dataframe
        for col in self.costs.columns:
            col_name = col.split(".")[-1]
            field_specs[col_name] = (SummarySpec, Field(title=col))

            # Get summary data for this column
            if col in self.costs_summary.columns:
                field_data = self.costs_summary.loc[:, col].to_dict()
            else:
                # If column not in summary, create a summary
                col_summary = (
                    self.costs[col]
                    .describe(percentiles=list(PERCENTILES.keys()))
                    .drop(["count"])
                )
                field_data = col_summary.to_dict()
            if not col.startswith(("cost.", "incentive.", "net_cost.")):
                msg = f"Column {col} is not a cost, incentive, or net cost column"
                raise ValueError(msg)
            field_data["units"] = "USD"

            field_datas[col_name] = SummarySpec(**field_data)

        field_specs["payback"] = (SummarySpec, Field(title="payback"))
        payback_data = self.paybacks_summary.to_dict()
        payback_data["units"] = "years"
        field_datas["payback"] = SummarySpec(**payback_data)

        model = create_model(
            "RetrofitCostsSpec",
            **field_specs,
            __config__=ConfigDict(extra="forbid"),
        )
        return model.model_validate(field_datas)


class SummarySpecBase(BaseModel, extra="forbid"):
    """Statistical summary of a dataset column."""

    min: float
    max: float
    mean: float
    std: float
    units: str


def create_summary_spec():
    """Create a summary spec with the percentiles as fields."""
    fields = {}
    for p_str, p_label in PERCENTILES.values():
        fields[p_str] = (float, Field(description=p_label, title=p_str))
    return create_model("SummarySpec", **fields, __base__=SummarySpecBase)


def create_end_use_disaggregation_spec(SummarySpec: type[SummarySpecBase]):
    """Create a end use disaggregation spec with the end uses as fields."""
    fields = {}
    for end_use in END_USES:
        fields[end_use] = (SummarySpec, Field(title=end_use))
    return create_model(
        "EndUseDisaggregationSpec", **fields, __config__=ConfigDict(extra="forbid")
    )


def create_fuel_disaggregation_spec(SummarySpec: type[SummarySpecBase]):
    """Create a fuel disaggregation spec with the fuels as fields."""
    fields = {}
    for fuel in FUELS:
        fields[fuel] = (SummarySpec, Field(title=fuel))
    return create_model(
        "FuelDisaggregationSpec", **fields, __config__=ConfigDict(extra="forbid")
    )


def create_disaggregation_spec(
    EndUseDisaggregationSpec: type[BaseModel], FuelDisaggregationSpec: type[BaseModel]
):
    """Create a disaggregation spec with the datasets as fields."""
    fields = {}
    for dataset, dataset_segments in DATASET_SEGMENT_MAP.items():
        if dataset_segments == END_USES:
            fields[dataset] = (EndUseDisaggregationSpec, Field(title=dataset))
        elif dataset_segments == FUELS:
            fields[dataset] = (FuelDisaggregationSpec, Field(title=dataset))
        else:
            msg = f"Invalid dataset: {dataset}"
            raise ValueError(msg)
    return create_model(
        "DisaggregationSpec", **fields, __config__=ConfigDict(extra="forbid")
    )


def create_disaggregations_spec(DisaggregationSpec: type[BaseModel]):
    """Create a disaggregations spec with the normalizations as fields."""
    fields = {}
    for field in NORMALIZATIONS:
        fields[field] = (DisaggregationSpec, Field(title=field))
    return create_model(
        "DisaggregationsSpec", **fields, __config__=ConfigDict(extra="forbid")
    )


def create_total_spec(SummarySpec: type[SummarySpecBase]):
    """Create a total spec with the datasets as summarized fields."""
    fields = {}
    for field in DATASETS:
        fields[field] = (SummarySpec, Field(title=field))
    return create_model("TotalSpec", **fields, __config__=ConfigDict(extra="forbid"))


def create_totals_spec(TotalSpec: type[BaseModel]):
    """Create a totals spec with the normalizations as fields."""
    fields = {}
    for field in NORMALIZATIONS:
        fields[field] = (TotalSpec, Field(title=field))
    return create_model("TotalsSpec", **fields, __config__=ConfigDict(extra="forbid"))


def create_sbem_inference_response_spec(
    DisaggregationsSpec: type[BaseModel], TotalsSpec: type[BaseModel]
):
    """Create a sbem inference response spec with the disaggregations and totals as fields."""
    return create_model(
        "SBEMInferenceResponseSpec",
        Disaggregation=(DisaggregationsSpec, Field(title="Disaggregation")),
        Total=(TotalsSpec, Field(title="Total")),
        __config__=ConfigDict(extra="forbid"),
    )


def create_sbem_inference_savings_response_spec(
    SBEMInferenceResponseSpec: type[BaseModel],
):
    """Create a sbem inference savings response spec with the original, upgraded, and delta as fields."""
    return create_model(
        "SBEMInferenceSavingsResponseSpec",
        original=(SBEMInferenceResponseSpec, Field(title="Original")),
        upgraded=(SBEMInferenceResponseSpec, Field(title="Upgraded")),
        delta=(SBEMInferenceResponseSpec, Field(title="Delta")),
        __config__=ConfigDict(extra="forbid"),
    )


SummarySpec = create_summary_spec()
EndUseDisaggregationSpec = create_end_use_disaggregation_spec(SummarySpec)
FuelDisaggregationSpec = create_fuel_disaggregation_spec(SummarySpec)
DisaggregationSpec = create_disaggregation_spec(
    EndUseDisaggregationSpec, FuelDisaggregationSpec
)
DisaggregationsSpec = create_disaggregations_spec(DisaggregationSpec)
TotalSpec = create_total_spec(SummarySpec)
TotalsSpec = create_totals_spec(TotalSpec)
SBEMInferenceResponseSpec = create_sbem_inference_response_spec(
    DisaggregationsSpec, TotalsSpec
)
SBEMInferenceSavingsResponseSpec = create_sbem_inference_savings_response_spec(
    SBEMInferenceResponseSpec
)


GLOBAL_MODEL_CACHE: dict[str, lgb.Booster] = {}
GLOBAL_FEATURE_TRANSFORM_CACHE: dict[str, RegressorInputSpec] = {}


class SBEMInferenceRequestSpec(BaseModel):
    """MA inference request spec."""

    rotated_rectangle: str
    neighbor_polys: list[str]
    neighbor_floors: list[float | int | None]
    lat: float
    lon: float
    actual_conditioned_area_m2: float

    short_edge: float
    long_edge: float
    num_floors: int
    orientation: float
    basement: BasementAtticOccupationConditioningStatus
    attic: BasementAtticOccupationConditioningStatus

    semantic_field_context: dict[str, float | str | int]

    source_experiment: str
    bucket: str = "ml-for-bem"

    @cached_property
    def artifact_keys(self) -> tuple[dict[str, str], dict[str, str]]:
        """Get the artifact keys for each of the model files and the yml files."""
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(
            Bucket=self.bucket, Prefix=f"hatchet/{self.source_experiment}"
        )
        if "Contents" not in response:
            msg = f"No contents found for {self.source_experiment}"
            raise ValueError(msg)

        # get all the lgb file keys
        lgb_file_keys = [
            obj["Key"]
            for obj in response["Contents"]
            if "Key" in obj and obj["Key"].endswith(".lgb")
        ]
        # get all the yml files
        yml_file_keys = [
            obj["Key"]
            for obj in response["Contents"]
            if "Key" in obj and obj["Key"].endswith(".yml")
        ]
        yml_files = {Path(key).stem: key for key in yml_file_keys}
        model_files = {Path(key).stem: key for key in lgb_file_keys}
        return model_files, yml_files

    @cached_property
    def source_feature_transform(self) -> RegressorInputSpec:
        """Load the source feature transforms from the space.yml file."""
        import yaml

        _, yml_files = self.artifact_keys
        space_key = yml_files["space"]
        s3 = boto3.client("s3")
        if space_key in GLOBAL_FEATURE_TRANSFORM_CACHE:
            return GLOBAL_FEATURE_TRANSFORM_CACHE[space_key]
        else:
            response = s3.get_object(Bucket=self.bucket, Key=space_key)
            space = yaml.safe_load(response["Body"].read().decode("utf-8"))
            transform = RegressorInputSpec.model_validate(space)
            GLOBAL_FEATURE_TRANSFORM_CACHE[space_key] = transform
            return transform

    @cached_property
    def lgb_models(self) -> dict[str, lgb.Booster]:
        """Load the lgb models from the s3 location."""
        model_files, _ = self.artifact_keys
        lgb_models: dict[str, lgb.Booster] = {}
        s3 = boto3.client("s3")
        for col, key in model_files.items():
            global GLOBAL_MODEL_CACHE
            if key in GLOBAL_MODEL_CACHE:
                lgb_models[col] = GLOBAL_MODEL_CACHE[key]
            else:
                response = s3.get_object(Bucket=self.bucket, Key=key)
                model = lgb.Booster(model_str=response["Body"].read().decode("utf-8"))
                # with tempfile.TemporaryDirectory() as tmpdir:
                #     tmp_path = Path(tmpdir) / "model.lgb"
                #     s3.download_file(
                #         Bucket=self.bucket, Key=key, Filename=tmp_path.as_posix()
                #     )
                #     with open(tmp_path) as f:
                #         model = lgb.Booster(model_str=f.read())
                lgb_models[col] = model
                GLOBAL_MODEL_CACHE[key] = model
        return lgb_models

    @cached_property
    def base_features(self):
        """Create the base features for the inference request.

        These base features will be used to build up the stochastic dataframe.

        This works by creating an intermediate SBEMSimulationSpec to safely rely on its
        feature dict computer. NB: this means we should probably somehow freeze it to
        avoid bitrot, e.g. by using a more specifically tagged image for the inference
        engine which won't get constantly overriden.

        Returns:
            features (pd.Series): The base features for the inference request.
        """
        pts = (
            gpd.GeoSeries([Point(self.lon, self.lat)])
            .set_crs("EPSG:4326")
            .to_crs("EPSG:3857")
        )
        feat = next(
            filter(
                lambda x: x.name == "feature.weather.file",
                self.source_feature_transform.features,
            )
        )
        if not isinstance(feat, CategoricalFeature):
            msg = f"Expected a categorical feature, got {type(feat)}"
            raise TypeError(msg)
        epw_names = [f"'{v}'" for v in feat.values]
        source_filter = f"name in [{', '.join(epw_names)}]"
        epw = closest_epw(
            pts,
            source_filter=source_filter,
        )
        epw_path = epw.iloc[0]["path"]
        epw_ob_path = f"https://climate.onebuilding.org/{epw_path}"
        epw_uri = AnyUrl(epw_ob_path)

        spec = SBEMSimulationSpec(
            rotated_rectangle=self.rotated_rectangle,
            neighbor_polys=self.neighbor_polys,
            neighbor_floors=self.neighbor_floors,
            neighbor_heights=[
                3.5 * (nf if nf is not None else 0) for nf in self.neighbor_floors
            ],
            short_edge=self.short_edge,
            long_edge=self.long_edge,
            num_floors=self.num_floors,
            aspect_ratio=self.long_edge / self.short_edge,
            long_edge_angle=self.orientation,
            semantic_field_context=self.semantic_field_context,
            basement=self.basement,
            attic=self.attic,
            # these fields will be overridden during sampling
            f2f_height=3.5,
            wwr=0.1,
            # no-op fields which will be completely ignored
            rotated_rectangle_area_ratio=1,
            height=self.num_floors * 3.5,
            sort_index=0,
            experiment_id="ma-inference",
            db_uri=AnyUrl("s3://unused/file.pq"),
            semantic_fields_uri=AnyUrl("s3://unused/file.pq"),
            component_map_uri=AnyUrl("s3://unused/file.pq"),
            epwzip_uri=epw_uri,
        )
        return pd.Series({
            k: v for k, v in spec.feature_dict.items() if k.startswith("feature.")
        })

    def make_priors(self):
        """Make the priors."""
        prior_dict: dict[str, Prior] = {}

        wwr_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.14,
                std=0.0125,
                clip_min=0.1001,
                clip_max=0.2999,
            )
        )
        prior_dict["feature.geometry.wwr"] = wwr_prior

        f2f_height_prior = ConditionalPrior(
            source_feature="feature.semantic.Age_bracket",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="pre_1975",
                    sampler=UniformSampler(min=2.5001, max=2.75),
                ),
                ConditionalPriorCondition(
                    match_val="btw_1975_2003",
                    sampler=UniformSampler(min=2.6, max=3.05),
                ),
                ConditionalPriorCondition(
                    match_val="post_2003",
                    sampler=UniformSampler(min=2.6, max=3.05),
                ),
            ],
        )
        prior_dict["feature.geometry.f2f_height"] = f2f_height_prior

        est_footprint_area = self.actual_conditioned_area_m2 / self.num_floors
        modeled_footprint_area = self.short_edge * self.long_edge
        footprint_area_ratio = est_footprint_area / modeled_footprint_area
        uniform_linear_scaling_factor = np.sqrt(footprint_area_ratio)

        est_actual_footprint_area_prior = UnconditionalPrior(
            sampler=FixedValueSampler(value=est_footprint_area)
        )
        prior_dict["feature.geometry.est_actual_footprint_area"] = (
            est_actual_footprint_area_prior
        )
        est_fp_ratio_prior = UnconditionalPrior(
            sampler=FixedValueSampler(value=footprint_area_ratio)
        )
        prior_dict["feature.geometry.est_fp_ratio"] = est_fp_ratio_prior

        est_uniform_linear_scaling_factor_prior = UnconditionalPrior(
            sampler=FixedValueSampler(value=uniform_linear_scaling_factor)
        )
        prior_dict["feature.geometry.est_uniform_linear_scaling_factor"] = (
            est_uniform_linear_scaling_factor_prior
        )

        half_perimeter_prior = UnconditionalPrior(
            sampler=SumValuesSampler(
                features_to_sum=[
                    "feature.geometry.short_edge",
                    "feature.geometry.long_edge",
                ]
            )
        )
        prior_dict["feature.geometry.computed.half_perimeter"] = half_perimeter_prior
        perimeter_prior = UnconditionalPrior(
            sampler=MultiplyValueSampler(
                feature_to_multiply="feature.geometry.computed.half_perimeter",
                value_to_multiply=2,
            )
        )
        prior_dict["feature.geometry.computed.perimeter"] = perimeter_prior

        single_floor_facade_area_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.geometry.computed.perimeter",
                    "feature.geometry.f2f_height",
                ]
            )
        )
        prior_dict["feature.geometry.computed.single_floor_facade_area"] = (
            single_floor_facade_area_prior
        )
        whole_bldg_facade_area_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.geometry.computed.single_floor_facade_area",
                    "feature.geometry.num_floors",
                ]
            )
        )
        prior_dict["feature.geometry.computed.whole_bldg_facade_area"] = (
            whole_bldg_facade_area_prior
        )

        total_linear_facade_distance = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.geometry.computed.perimeter",
                    "feature.geometry.num_floors",
                ]
            )
        )
        prior_dict["feature.geometry.computed.total_linear_facade_distance"] = (
            total_linear_facade_distance
        )

        window_area_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.geometry.computed.whole_bldg_facade_area",
                    "feature.geometry.wwr",
                ]
            )
        )
        prior_dict["feature.geometry.computed.window_area"] = window_area_prior

        footprint_area_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.geometry.short_edge",
                    "feature.geometry.long_edge",
                ]
            )
        )
        prior_dict["feature.geometry.computed.footprint_area"] = footprint_area_prior

        roof_is_attic = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.exists",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=1),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=0),
                ),
            ],
        )
        prior_dict["feature.geometry.roof_is_attic.num"] = roof_is_attic
        roof_is_flat = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.exists",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=0),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=1),
                ),
            ],
        )
        prior_dict["feature.geometry.roof_is_flat.num"] = roof_is_flat

        attic_occupied_num = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.occupied",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=1),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=0),
                ),
            ],
        )
        prior_dict["feature.extra_spaces.attic.occupied.num"] = attic_occupied_num

        attic_conditioned_num = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.conditioned",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=1),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=0),
                ),
            ],
        )
        prior_dict["feature.extra_spaces.attic.conditioned.num"] = attic_conditioned_num
        attic_type_num_prior = UnconditionalPrior(
            sampler=SumValuesSampler(
                features_to_sum=[
                    "feature.extra_spaces.attic.occupied.num",
                    "feature.extra_spaces.attic.conditioned.num",
                ]
            )
        )
        prior_dict["feature.extra_spaces.attic.pitch.type.num"] = attic_type_num_prior

        attic_pitch_if_exists_prior = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.pitch.type.num",
            fallback_prior=UniformSampler(min=6 / 12, max=9 / 12),
            conditions=[
                ConditionalPriorCondition(
                    match_val=0,
                    sampler=UniformSampler(min=4 / 12, max=6 / 12),
                ),
            ],
        )
        prior_dict["feature.extra_spaces.attic.pitch.if_exists"] = (
            attic_pitch_if_exists_prior
        )

        attic_pitch_prior = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.exists",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=CopySampler(
                        feature_to_copy="feature.extra_spaces.attic.pitch.if_exists"
                    ),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=0),
                ),
            ],
        )
        prior_dict["feature.extra_spaces.attic.pitch"] = attic_pitch_prior

        attic_run_prior = UnconditionalPrior(
            sampler=MultiplyValueSampler(
                feature_to_multiply="feature.geometry.short_edge",
                value_to_multiply=0.5,
            )
        )
        prior_dict["feature.extra_spaces.attic.run"] = attic_run_prior

        attic_height_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.extra_spaces.attic.pitch",
                    "feature.extra_spaces.attic.run",
                ]
            )
        )
        prior_dict["feature.geometry.attic_height"] = attic_height_prior

        attic_hypotenuse_a2_prior = UnconditionalPrior(
            sampler=PowerSampler(
                feature_to_power="feature.geometry.attic_height",
                power=2,
            )
        )
        prior_dict["feature.geometry.computed.attic_hypotenuse_a2"] = (
            attic_hypotenuse_a2_prior
        )
        attic_hypotenuse_b2_prior = UnconditionalPrior(
            sampler=PowerSampler(
                feature_to_power="feature.extra_spaces.attic.run",
                power=2,
            )
        )
        prior_dict["feature.geometry.computed.attic_hypotenuse_b2"] = (
            attic_hypotenuse_b2_prior
        )
        attic_hypotenuse_c2_prior = UnconditionalPrior(
            sampler=SumValuesSampler(
                features_to_sum=[
                    "feature.geometry.computed.attic_hypotenuse_a2",
                    "feature.geometry.computed.attic_hypotenuse_b2",
                ]
            )
        )
        prior_dict["feature.geometry.computed.attic_hypotenuse_c2"] = (
            attic_hypotenuse_c2_prior
        )
        attic_hypotenuse_prior = UnconditionalPrior(
            sampler=PowerSampler(
                feature_to_power="feature.geometry.computed.attic_hypotenuse_c2",
                power=0.5,
            )
        )
        prior_dict["feature.geometry.computed.attic_hypotenuse"] = (
            attic_hypotenuse_prior
        )

        half_roof_surface_area_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.geometry.computed.attic_hypotenuse",
                    "feature.geometry.long_edge",
                ]
            )
        )
        prior_dict["feature.geometry.computed.half_roof_surface_area"] = (
            half_roof_surface_area_prior
        )

        roof_surface_area_prior = UnconditionalPrior(
            sampler=MultiplyValueSampler(
                feature_to_multiply="feature.geometry.computed.half_roof_surface_area",
                value_to_multiply=2,
            )
        )
        prior_dict["feature.geometry.computed.roof_surface_area"] = (
            roof_surface_area_prior
        )

        attic_use_fraction_prior = ConditionalPrior(
            source_feature="feature.extra_spaces.attic.occupied",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes", sampler=UniformSampler(min=0.2, max=0.4)
                ),
                ConditionalPriorCondition(
                    match_val="No", sampler=FixedValueSampler(value=0)
                ),
            ],
        )
        basement_use_fraction_prior = ConditionalPrior(
            source_feature="feature.extra_spaces.basement.occupied",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes", sampler=UniformSampler(min=0.2, max=0.4)
                ),
                ConditionalPriorCondition(
                    match_val="No", sampler=FixedValueSampler(value=0)
                ),
            ],
        )
        basement_is_occupied_num = ConditionalPrior(
            source_feature="feature.extra_spaces.basement.occupied",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=1),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=0),
                ),
            ],
        )
        basement_exists_num = ConditionalPrior(
            source_feature="feature.extra_spaces.basement.exists",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=1),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=0),
                ),
            ],
        )
        prior_dict["feature.extra_spaces.basement.exists.num"] = basement_exists_num

        basement_is_not_occupied_num = ConditionalPrior(
            source_feature="feature.extra_spaces.basement.occupied",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Yes",
                    sampler=FixedValueSampler(value=0),
                ),
                ConditionalPriorCondition(
                    match_val="No",
                    sampler=FixedValueSampler(value=1),
                ),
            ],
        )
        prior_dict["feature.extra_spaces.basement.occupied.num"] = (
            basement_is_occupied_num
        )
        prior_dict["feature.extra_spaces.basement.not_occupied.num"] = (
            basement_is_not_occupied_num
        )
        prior_dict["feature.extra_spaces.attic.use_fraction"] = attic_use_fraction_prior
        prior_dict["feature.extra_spaces.basement.use_fraction"] = (
            basement_use_fraction_prior
        )

        heat_cop_prior = ConditionalPrior(
            source_feature="feature.semantic.Heating",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="ElectricResistance",
                    sampler=UniformSampler(min=0.95, max=0.99),
                ),
                ConditionalPriorCondition(
                    match_val="OilHeating",
                    sampler=UniformSampler(min=0.63, max=0.92),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasHeating",
                    sampler=UniformSampler(min=0.60, max=0.925),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasCondensingHeating",
                    sampler=UniformSampler(min=0.60, max=0.925),
                ),
                ConditionalPriorCondition(
                    match_val="ASHPHeating",
                    sampler=UniformSampler(min=2.1, max=4.2),
                ),
                ConditionalPriorCondition(
                    match_val="GSHPHeating",
                    sampler=UniformSampler(min=2.9, max=4.8),
                ),
            ],
        )
        heat_distribution_prior = ConditionalPrior(
            source_feature="feature.semantic.Distribution",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Steam",
                    sampler=UniformSampler(min=0.5, max=0.8),
                ),
                ConditionalPriorCondition(
                    match_val="HotWaterUninsulated",
                    sampler=UniformSampler(min=0.73, max=0.86),
                ),
                ConditionalPriorCondition(
                    match_val="AirDuctsUninsulated",
                    sampler=UniformSampler(min=0.7, max=0.8),
                ),
                ConditionalPriorCondition(
                    match_val="AirDuctsConditionedUninsulated",
                    sampler=UniformSampler(min=0.85, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="HotWaterInsulated",
                    sampler=UniformSampler(min=0.86, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="AirDuctsInsulated",
                    sampler=UniformSampler(min=0.8, max=0.9),
                ),
            ],
        )

        heat_fuel_prior = ConditionalPrior(
            source_feature="feature.semantic.Heating",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="ElectricResistance",
                    sampler=FixedValueSampler(value="Electricity"),
                ),
                ConditionalPriorCondition(
                    match_val="OilHeating",
                    sampler=FixedValueSampler(value="Oil"),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasHeating",
                    sampler=FixedValueSampler(value="NaturalGas"),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasCondensingHeating",
                    sampler=FixedValueSampler(value="NaturalGas"),
                ),
                ConditionalPriorCondition(
                    match_val="ASHPHeating",
                    sampler=FixedValueSampler(value="Electricity"),
                ),
                ConditionalPriorCondition(
                    match_val="GSHPHeating",
                    sampler=FixedValueSampler(value="Electricity"),
                ),
            ],
        )
        prior_dict["feature.factors.system.heat.cop"] = heat_cop_prior
        prior_dict["feature.factors.system.heat.distribution"] = heat_distribution_prior
        prior_dict["feature.factors.system.heat.fuel"] = heat_fuel_prior

        cool_cop_prior = ConditionalPrior(
            source_feature="feature.semantic.Cooling",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="none", sampler=FixedValueSampler(value=999999999999)
                ),
                ConditionalPriorCondition(
                    match_val="ACWindow",
                    sampler=UniformSampler(min=2, max=3.5),
                ),
                ConditionalPriorCondition(
                    match_val="ACCentral",
                    sampler=UniformSampler(min=2.5, max=3.8),
                ),
                ConditionalPriorCondition(
                    match_val="WindowASHP",
                    sampler=UniformSampler(min=3, max=4),
                ),
                # TODO: Optional: make this depend on the ASHP heating cop
                ConditionalPriorCondition(
                    match_val="ASHPCooling",
                    sampler=UniformSampler(min=3.2, max=4.5),
                ),
                ConditionalPriorCondition(
                    match_val="GSHPCooling",
                    sampler=UniformSampler(min=3.8, max=5),
                ),
            ],
        )

        cool_is_distributed_prior = ConditionalPrior(
            source_feature="feature.semantic.Cooling",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="none", sampler=FixedValueSampler(value=False)
                ),
                ConditionalPriorCondition(
                    match_val="ACWindow",
                    sampler=FixedValueSampler(value=False),
                ),
                ConditionalPriorCondition(
                    match_val="ACCentral",
                    sampler=FixedValueSampler(value=True),
                ),
                ConditionalPriorCondition(
                    match_val="WindowASHP",
                    sampler=FixedValueSampler(value=False),
                ),
                # TODO: Optional: make this depend on the ASHP heating cop
                ConditionalPriorCondition(
                    match_val="ASHPCooling",
                    sampler=FixedValueSampler(value=True),
                ),
                ConditionalPriorCondition(
                    match_val="GSHPCooling",
                    sampler=FixedValueSampler(value=True),
                ),
            ],
        )
        cool_distribution_prior = ConditionalPrior(
            source_feature="feature.factors.system.cool.is_distributed",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val=True,
                    sampler=CopySampler(
                        feature_to_copy="feature.factors.system.heat.distribution"
                    ),
                ),
                ConditionalPriorCondition(
                    match_val=False, sampler=FixedValueSampler(value=1)
                ),
            ],
        )
        cool_fuel_prior = UnconditionalPrior(
            sampler=FixedValueSampler(value="Electricity")
        )
        prior_dict["feature.factors.system.cool.cop"] = cool_cop_prior
        prior_dict["feature.factors.system.cool.is_distributed"] = (
            cool_is_distributed_prior
        )
        prior_dict["feature.factors.system.cool.distribution"] = cool_distribution_prior
        prior_dict["feature.factors.system.cool.fuel"] = cool_fuel_prior

        dhw_cop_prior = ConditionalPrior(
            source_feature="feature.semantic.DHW",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="ElectricResistanceDHW",
                    sampler=UniformSampler(min=0.95, max=0.99),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasDHW",
                    sampler=UniformSampler(min=0.55, max=0.8),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasHeatingDHWCombo",
                    sampler=UniformSampler(min=0.8, max=0.95),
                ),
                ConditionalPriorCondition(
                    match_val="HPWH",
                    sampler=UniformSampler(min=2.2, max=3.5),
                ),
            ],
        )
        dhw_distribution_prior = UnconditionalPrior(
            sampler=CopySampler(
                feature_to_copy="feature.factors.system.heat.distribution"
            )
        )
        dhw_fuel_prior = ConditionalPrior(
            source_feature="feature.semantic.DHW",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="ElectricResistanceDHW",
                    sampler=FixedValueSampler(value="Electricity"),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasDHW",
                    sampler=FixedValueSampler(value="NaturalGas"),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasHeatingDHWCombo",
                    sampler=FixedValueSampler(value="NaturalGas"),
                ),
                ConditionalPriorCondition(
                    match_val="HPWH",
                    sampler=FixedValueSampler(value="Electricity"),
                ),
            ],
        )
        prior_dict["feature.factors.system.dhw.cop"] = dhw_cop_prior
        prior_dict["feature.factors.system.dhw.distribution"] = dhw_distribution_prior
        prior_dict["feature.factors.system.dhw.fuel"] = dhw_fuel_prior

        effective_heat_cop_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.factors.system.heat.cop",
                    "feature.factors.system.heat.distribution",
                ]
            )
        )
        effective_cool_cop_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.factors.system.cool.cop",
                    "feature.factors.system.cool.distribution",
                ]
            )
        )
        effective_dhw_cop_prior = UnconditionalPrior(
            sampler=ProductValuesSampler(
                features_to_multiply=[
                    "feature.factors.system.dhw.cop",
                    "feature.factors.system.dhw.distribution",
                ]
            )
        )
        prior_dict["feature.factors.system.heat.effective_cop"] = (
            effective_heat_cop_prior
        )
        prior_dict["feature.factors.system.cool.effective_cop"] = (
            effective_cool_cop_prior
        )
        prior_dict["feature.factors.system.dhw.effective_cop"] = effective_dhw_cop_prior

        gas_price_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.07, std=0.011, clip_min=0.03, clip_max=0.09
            )
        )
        oil_price_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.102, std=0.024, clip_min=0.07, clip_max=0.13
            )
        )
        electricity_price_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.312, std=0.02, clip_min=0.143, clip_max=0.4013
            )
        )
        prior_dict["feature.fuels.price.NaturalGas"] = gas_price_prior
        prior_dict["feature.fuels.price.Electricity"] = electricity_price_prior
        prior_dict["feature.fuels.price.Oil"] = oil_price_prior

        electricity_emissions_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.000298, std=0.0000155, clip_min=0.0001, clip_max=0.0005
            )
        )
        gas_emissions_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.000244, std=0.0000122, clip_min=0.000176, clip_max=0.0003
            )
        )
        oil_emissions_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.000267, std=0.000003, clip_min=0.0002, clip_max=0.0003
            )
        )
        prior_dict["feature.fuels.emissions.Electricity"] = electricity_emissions_prior
        prior_dict["feature.fuels.emissions.NaturalGas"] = gas_emissions_prior
        prior_dict["feature.fuels.emissions.Oil"] = oil_emissions_prior

        # TODO: optionally create the matrix for moving raw values to
        # various energy end uses, fuels, emissions, costs.

        return Priors(sampled_features=prior_dict)

    def make_inference_features(self, n: int):
        """Make the features to use in prediction.

        This includes the base features which are computed by the inference
        request spec's intermediate SBEMSimulationSpec, and then repeated
        n times.

        Args:
            n (int): The number of samples to create.

        Returns:
            df (pd.DataFrame): The features to use in prediction.
        """
        base_features = self.base_features
        return pd.DataFrame([base_features] * n)

    @cached_property
    def generator(self) -> np.random.Generator:
        """The random number generator for the experiment."""
        return np.random.default_rng(42)

    def make_features(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create the features to use in prediction.

        Args:
            n (int): The number of samples to create.

        Returns:
            df (pd.DataFrame): The features to use in prediction.
            df_t (pd.DataFrame): The transformed features to use in prediction.
        """
        df = self.make_inference_features(n)
        priors = self.make_priors()
        # TODO: we should consider removing all features which are in the priors
        # to ensure that there are no strange overwrite behaviors...
        df = df.drop(
            columns=[p for p in priors.sampled_features if p in df.columns],
            axis=1,
        )
        df = priors.sample(df, n, self.generator)
        original_cooling = None
        mask = None
        if "feature.semantic.Cooling" in df.columns:
            mask = df["feature.semantic.Cooling"] == "none"
            original_cooling = df["feature.semantic.Cooling"].copy()
            df.loc[mask, "feature.semantic.Cooling"] = "ACCentral"
        df_t = self.source_feature_transform.transform(df)
        if original_cooling is not None and mask is not None:
            df.loc[mask, "feature.semantic.Cooling"] = original_cooling.loc[mask]
        return df, df_t

    def make_retrofit_cost_features(
        self, features: pd.DataFrame, peak_results: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute features needed for cost calculations after inference has been run.

        Features needed: heating_capacity_kW, county_indicator one hot encoded, has_gas, has_cooling
        """
        # Start with existing features
        cost_features = features.copy()

        # Compute heating capacity based on peak heating load
        peak_heating_per_m2 = peak_results["Raw"]["Heating"]
        gross_peak_kw = peak_heating_per_m2 * self.actual_conditioned_area_m2
        effective_cop = features["feature.factors.system.heat.effective_cop"]
        electrical_capacity_kW = gross_peak_kw / effective_cop
        safety_factor = 1.2
        cost_features["feature.calculated.heating_capacity_kW"] = (
            electrical_capacity_kW * safety_factor
        )

        counties = [
            "Berkshire",
            "Barnstable",
            "Bristol",
            "Dukes",
            "Essex",
            "Franklin",
            "Hampden",
            "Hampshire",
            "Middlesex",
            "Nantucket",
            "Norfolk",
            "Plymouth",
            "Suffolk",
            "Worcester",
        ]
        for county in counties:
            if "feature.location.county" in cost_features.columns:
                cost_features[f"feature.location.in_county_{county}"] = (
                    cost_features["feature.location.county"] == county
                )
            else:
                # Check to ensure that county can always be returned, otherwise it will exclude this cost factor - could instead consider making this raise an error?
                cost_features[f"feature.location.in_county_{county}"] = False
                # TODO: log a warning if the column is missing
                logging.warning(f"County {county} not found in cost features")
                # pd.to_dummies

        # Add gas availability indicator
        gas_heating_systems = ["NaturalGasHeating", "NaturalGasCondensingHeating"]
        cost_features["feature.system.has_gas"] = features[
            "feature.semantic.Heating"
        ].isin(gas_heating_systems)
        cost_features["feature.system.has_gas_not"] = ~cost_features[
            "feature.system.has_gas"
        ]

        # Add cooling availability indicator
        cooling_systems = [
            "ACWindow",
            "ACCentral",
            "WindowASHP",
            "ASHPCooling",
            "GSHPCooling",
        ]
        cost_features["feature.system.has_cooling"] = features[
            "feature.semantic.Cooling"
        ].isin(cooling_systems)
        cost_features["feature.system.has_cooling_not"] = ~cost_features[
            "feature.system.has_cooling"
        ]

        # Add constant feature for intercept terms
        # cost_features["feature.constant.one"] = 1

        return cost_features

    def make_retrofit_incentive_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute features needed for incentive calculations after inference has been run.

        Features needed: fueltype, region, income eligibility
        """
        # Start with existing features
        incentive_features = features.copy()

        # Add region indicator (default to MA)
        incentive_features["feature.location.region"] = "MA"

        # Add fuel type indicators
        incentive_features["feature.fuel.electricity"] = True  # Always available
        incentive_features["feature.fuel.natural_gas"] = features[
            "feature.semantic.Heating"
        ].isin(["NaturalGasHeating", "NaturalGasCondensingHeating"])
        incentive_features["feature.fuel.oil"] = (
            features["feature.semantic.Heating"] == "OilHeating"
        )

        # Add income eligibility (placeholder - will be set based on context)
        incentive_features["feature.eligibility.income_level"] = "All_customers"

        return incentive_features

    def predict(self, df_transformed: pd.DataFrame) -> pd.DataFrame:
        """Predict the results of the inference.

        Args:
            df_transformed (pd.DataFrame): The transformed features to use in prediction.

        Returns:
            raw (pd.DataFrame): The raw predicted results.
        """
        results: list[pd.Series] = []
        for model_name, model in self.lgb_models.items():
            pred = cast(np.ndarray, model.predict(df_transformed))
            results.append(pd.Series(pred, name=model_name))
        # TODO: separate the results into dicts of dfs by splitting on '.'
        # then reconcatenating using pd.concat to create a multi index.
        data = pd.concat(results, axis=1)
        data.columns = data.columns.str.split(".", expand=True)
        return data

    def apply_cops(
        self, *, df_features: pd.DataFrame, df_raw: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the COPs to the results.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_raw (pd.DataFrame): The raw predicted results.

        Returns:
            df_end_uses (pd.DataFrame): The end uses.
        """
        heat_cop = df_features["feature.factors.system.heat.effective_cop"]
        cool_cop = df_features["feature.factors.system.cool.effective_cop"]
        dhw_cop = df_features["feature.factors.system.dhw.effective_cop"]
        df_end_uses = df_raw.copy(deep=True)
        df_end_uses["Heating"] = df_raw["Heating"].div(heat_cop, axis=0)
        df_end_uses["Cooling"] = df_raw["Cooling"].div(cool_cop, axis=0)
        df_end_uses["Domestic Hot Water"] = df_raw["Domestic Hot Water"].div(
            dhw_cop, axis=0
        )
        return df_end_uses

    def separate_fuel_based_end_uses(
        self, *, df_features: pd.DataFrame, df_end_uses: pd.DataFrame
    ):
        """Split the end uses into their component fuel usages.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_end_uses (pd.DataFrame): The end uses (with effective COPs applied).

        Returns:
            df_disaggregated_fuels (pd.DataFrame): The disaggregated fuels.
        """
        heat_fuel = df_features["feature.factors.system.heat.fuel"]
        cool_fuel = df_features["feature.factors.system.cool.fuel"]
        dhw_fuel = df_features["feature.factors.system.dhw.fuel"]
        heat_is_elec = heat_fuel == "Electricity"
        cool_is_elec = cool_fuel == "Electricity"
        dhw_is_elec = dhw_fuel == "Electricity"
        heat_is_gas = heat_fuel == "NaturalGas"
        cool_is_gas = cool_fuel == "NaturalGas"
        dhw_is_gas = dhw_fuel == "NaturalGas"
        heat_is_oil = heat_fuel == "Oil"
        cool_is_oil = cool_fuel == "Oil"
        dhw_is_oil = dhw_fuel == "Oil"
        heat_fuel_is_not_known = ~heat_is_elec & ~heat_is_gas & ~heat_is_oil
        cool_fuel_is_not_known = ~cool_is_elec & ~cool_is_gas & ~cool_is_oil
        dhw_fuel_is_not_known = ~dhw_is_elec & ~dhw_is_gas & ~dhw_is_oil
        if heat_fuel_is_not_known.any():
            msg = "At least one sample has a heating fuel that is not known."
            raise NotImplementedError(msg)
        if cool_fuel_is_not_known.any():
            msg = "At least one sample has a cooling fuel that is not known."
            raise NotImplementedError(msg)
        if dhw_fuel_is_not_known.any():
            msg = "At least one sample has a domestic hot water fuel that is not known."
            raise NotImplementedError(msg)
        heat_elec = df_end_uses["Heating"].mul(heat_is_elec, axis=0)
        heat_gas = df_end_uses["Heating"].mul(heat_is_gas, axis=0)
        heat_oil = df_end_uses["Heating"].mul(heat_is_oil, axis=0)
        cool_elec = df_end_uses["Cooling"].mul(cool_is_elec, axis=0)
        cool_gas = df_end_uses["Cooling"].mul(cool_is_gas, axis=0)
        cool_oil = df_end_uses["Cooling"].mul(cool_is_oil, axis=0)
        dhw_elec = df_end_uses["Domestic Hot Water"].mul(dhw_is_elec, axis=0)
        dhw_gas = df_end_uses["Domestic Hot Water"].mul(dhw_is_gas, axis=0)
        dhw_oil = df_end_uses["Domestic Hot Water"].mul(dhw_is_oil, axis=0)
        lighting = df_end_uses["Lighting"]
        equipment = df_end_uses["Equipment"]

        elec = pd.concat(
            [heat_elec, cool_elec, dhw_elec, lighting, equipment],
            axis=1,
            keys=["Heating", "Cooling", "Domestic Hot Water", "Lighting", "Equipment"],
        )[df_end_uses.columns]
        gas = pd.concat(
            [heat_gas, cool_gas, dhw_gas, lighting * 0, equipment * 0],
            axis=1,
            keys=["Heating", "Cooling", "Domestic Hot Water", "Lighting", "Equipment"],
        )[df_end_uses.columns]
        oil = pd.concat(
            [heat_oil, cool_oil, dhw_oil, lighting * 0, equipment * 0],
            axis=1,
            keys=["Heating", "Cooling", "Domestic Hot Water", "Lighting", "Equipment"],
        )[df_end_uses.columns]

        df_disaggregated_fuels = pd.concat(
            [elec, gas, oil],
            axis=1,
            keys=["Electricity", "NaturalGas", "Oil"],
            names=["Fuel", "EndUse"],
        )
        return df_disaggregated_fuels

    def compute_costs(
        self,
        *,
        df_features: pd.DataFrame,
        df_disaggregated_fuels: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute the costs.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_disaggregated_fuels (pd.DataFrame): The disaggregated fuels.

        Returns:
            df_fuel_costs (pd.DataFrame): The total costs for each fuel.
            df_end_use_costs (pd.DataFrame): The total costs for each end use.
        """
        gas_rate = df_features["feature.fuels.price.NaturalGas"]
        elec_rate = df_features["feature.fuels.price.Electricity"]
        oil_rate = df_features["feature.fuels.price.Oil"]

        elec_costs = df_disaggregated_fuels["Electricity"].mul(elec_rate, axis=0)
        gas_costs = df_disaggregated_fuels["NaturalGas"].mul(gas_rate, axis=0)
        oil_costs = df_disaggregated_fuels["Oil"].mul(oil_rate, axis=0)

        disaggregated_costs = pd.concat(
            [elec_costs, gas_costs, oil_costs],
            axis=1,
            keys=["Electricity", "NaturalGas", "Oil"],
            names=["Fuel", "EndUse"],
        )
        end_use_costs = disaggregated_costs.T.groupby(level=["EndUse"]).sum().T
        fuel_costs = disaggregated_costs.T.groupby(level=["Fuel"]).sum().T

        return fuel_costs, end_use_costs

    def compute_emissions(
        self, *, df_features: pd.DataFrame, df_disaggregated_fuels: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute the emissions.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_disaggregated_fuels (pd.DataFrame): The disaggregated fuels.

        Returns:
            df_fuel_emissions (pd.DataFrame): The emissions for each fuel.
            df_end_use_emissions (pd.DataFrame): The emissions for each end use.
        """
        gas_emissions_factors = df_features["feature.fuels.emissions.NaturalGas"]
        elec_emissions_factors = df_features["feature.fuels.emissions.Electricity"]
        oil_emissions_factors = df_features["feature.fuels.emissions.Oil"]

        elec_emissions = df_disaggregated_fuels["Electricity"].mul(
            elec_emissions_factors, axis=0
        )
        gas_emissions = df_disaggregated_fuels["NaturalGas"].mul(
            gas_emissions_factors, axis=0
        )
        oil_emissions = df_disaggregated_fuels["Oil"].mul(oil_emissions_factors, axis=0)

        disaggregated_emissions = pd.concat(
            [elec_emissions, gas_emissions, oil_emissions],
            axis=1,
            keys=["Electricity", "NaturalGas", "Oil"],
            names=["Fuel", "EndUse"],
        )
        end_use_emissions = disaggregated_emissions.T.groupby(level=["EndUse"]).sum().T
        fuel_emissions = disaggregated_emissions.T.groupby(level=["Fuel"]).sum().T

        return fuel_emissions, end_use_emissions

    def run(self, n: int = 10000):
        """Run the inference.

        Returns:
            df_summary (pd.DataFrame): The summary.
        """
        features, features_transformed = self.make_features(n=n)
        results_raw = self.predict(features_transformed)
        _results_peak = cast(pd.DataFrame, results_raw["Peak"])
        results_energy = cast(pd.DataFrame, results_raw["Energy"])
        return self.compute_distributions(features, results_energy)

    def compute_distributions(self, features: pd.DataFrame, results_raw: pd.DataFrame):
        """Compute the distributions for each metric."""
        results_end_uses = self.apply_cops(df_features=features, df_raw=results_raw)
        results_disaggregated_fuels = self.separate_fuel_based_end_uses(
            df_features=features, df_end_uses=results_end_uses
        )
        results_fuels = results_disaggregated_fuels.T.groupby(level=["Fuel"]).sum().T
        results_fuel_costs, results_end_use_costs = self.compute_costs(
            df_features=features, df_disaggregated_fuels=results_disaggregated_fuels
        )
        results_fuel_emissions, results_end_use_emissions = self.compute_emissions(
            df_features=features, df_disaggregated_fuels=results_disaggregated_fuels
        )

        disaggregated = pd.concat(
            [
                results_raw,
                results_end_uses,
                results_fuels,
                results_end_use_costs,
                results_end_use_emissions,
                results_fuel_costs,
                results_fuel_emissions,
            ],
            axis=1,
            keys=[
                "Raw",
                "EndUse",
                "Fuel",
                "EndUseCost",
                "EndUseEmissions",
                "FuelCost",
                "FuelEmissions",
            ],
            names=["Dataset", "Segment"],
        )
        disaggregated_gross = disaggregated * self.actual_conditioned_area_m2
        disaggregations = pd.concat(
            [disaggregated, disaggregated_gross],
            axis=1,
            keys=["Normalized", "Gross"],
            names=["Normalization"],
        )
        total = disaggregated.T.groupby(level=["Dataset"]).sum().T
        total_gross = total * self.actual_conditioned_area_m2
        totals = pd.concat(
            [total, total_gross],
            axis=1,
            keys=["Normalized", "Gross"],
            names=["Normalization"],
        )
        disaggregations_summary = disaggregations.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])
        totals_summary = totals.describe(percentiles=list(PERCENTILES.keys())).drop([
            "count"
        ])

        return SBEMDistributions(
            features=features,
            disaggregations=disaggregations,
            totals=totals,
            disaggregations_summary=disaggregations_summary,
            totals_summary=totals_summary,
        )


class SBEMInferenceSavingsRequestSpec(BaseModel):
    """An inference request spec for computing savings using matched samples."""

    original: SBEMInferenceRequestSpec
    upgraded_semantic_field_context: dict[str, float | str | int]

    def run(
        self, n: int = 10000
    ) -> dict[
        str,
        "SBEMDistributions | SBEMRetrofitDistributions",
    ]:
        """Run the inference for a savings problem.

        This function will ensure that the original and upgraded models keep aligned
        features for stochastically sampled values, e.g. attic_height, wwr, etc.

        Args:
            n (int): The number of samples to run.

        Returns:
            original_results (SBEMDistributions): The results of the original model.
            new_results (SBEMDistributions): The results of the upgraded model.
            delta_results (SBEMDistributions): The results of the delta between the original and upgraded models.
        """
        original_results = self.original.run(n)
        original_features = original_results.features
        original_priors = self.original.make_priors()

        # first, we must compute which semantic features have changed.
        changed_feature_fields, changed_context_fields = self.changed_context_fields

        changed_feature_names = set(changed_feature_fields.keys())

        # then we will get the priors that must be re-run as they are downstream
        # of the changed features.
        changed_priors = original_priors.select_prior_tree_for_changed_features(
            changed_feature_names
        )

        # then we will take the original features and update the changed semantic
        # features.
        new_features = original_features.copy(deep=True)
        for feature_name, value in changed_feature_fields.items():
            new_features[feature_name] = value

        # then we compute the new features using the changed priors
        new_features = changed_priors.sample(
            new_features, len(new_features), self.original.generator
        )

        # then we run traditional inference on the new features
        mask = None
        original_cooling = None
        if "feature.semantic.Cooling" in new_features.columns:
            original_cooling = new_features["feature.semantic.Cooling"].copy()
            mask = new_features["feature.semantic.Cooling"] == "none"
            new_features.loc[mask, "feature.semantic.Cooling"] = "ACCentral"
        new_transformed_features = self.original.source_feature_transform.transform(
            new_features
        )
        if mask is not None and original_cooling is not None:
            new_features.loc[mask, "feature.semantic.Cooling"] = original_cooling.loc[
                mask
            ]
        new_results_raw = self.original.predict(new_transformed_features)
        new_results_peak = cast(pd.DataFrame, new_results_raw["Peak"])
        new_results_energy = cast(pd.DataFrame, new_results_raw["Energy"])
        new_results = self.original.compute_distributions(
            new_features, new_results_energy
        )
        # new_results_peak = self.original.compute_distributions(
        #     new_features, _new_results_peak
        # )

        # Build features for retrofit costs with calculated values
        features_for_costs = new_features.copy(deep=True)

        # finally, we compute the deltas and the corresponding summary
        # statistics.
        disaggs_delta = original_results.disaggregations - new_results.disaggregations
        totals_delta = original_results.totals - new_results.totals
        disaggs_delta_summary = disaggs_delta.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])
        totals_delta_summary = totals_delta.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])

        # return the results
        delta_results = SBEMDistributions(
            features=new_results.features,
            disaggregations=disaggs_delta,
            totals=totals_delta,
            disaggregations_summary=disaggs_delta_summary,
            totals_summary=totals_delta_summary,
        )

        costs_path = Path(__file__).parent / "data" / "retrofit-costs.json"

        # Load split incentive files
        all_customers_incentives_path = (
            Path(__file__).parent / "data" / "incentives_all_customers.json"
        )
        income_eligible_incentives_path = (
            Path(__file__).parent / "data" / "incentives_income_eligible.json"
        )

        # Load both incentive configurations
        all_customers_incentive_config = RetrofitQuantities.Open(
            all_customers_incentives_path
        )
        income_eligible_incentive_config = RetrofitQuantities.Open(
            income_eligible_incentives_path
        )

        cost_config = RetrofitQuantities.Open(costs_path)

        # Compute features for cost calculations after inference
        features_for_costs = self.original.make_retrofit_cost_features(
            features_for_costs, new_results_peak
        )
        retrofit_costs = self.compute_retrofit_costs(features_for_costs, cost_config)

        # Compute incentives using split incentive files
        (
            all_customers_incentives,
            income_eligible_incentives,
        ) = self.compute_incentives_split(
            features_for_costs,
            all_customers_incentive_config,
            income_eligible_incentive_config,
            retrofit_costs,
        )

        # Compute net costs after incentives
        all_customers_net_costs, income_eligible_net_costs = self.compute_net_costs(
            retrofit_costs, all_customers_incentives, income_eligible_incentives
        )

        # Compute paybacks
        payback_no_incentives = self.compute_payback(retrofit_costs, delta_results)
        # TODO: decide what paybacks to show in outputs
        payback_with_incentives_all = self.compute_payback_with_incentives(
            all_customers_net_costs, delta_results
        )
        payback_with_incentives_income = self.compute_payback_with_incentives(
            income_eligible_net_costs, delta_results
        )

        # Combine all cost and incentive data
        all_costs_data = pd.concat(
            [
                retrofit_costs,
                all_customers_incentives,
                income_eligible_incentives,
                all_customers_net_costs,
                income_eligible_net_costs,
            ],
            axis=1,
        )

        # Create summary statistics
        retrofit_costs_summary = retrofit_costs.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])

        all_customers_net_costs_summary = all_customers_net_costs.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])
        income_eligible_net_costs_summary = income_eligible_net_costs.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])

        payback_no_incentives_summary = payback_no_incentives.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])
        payback_with_incentives_all_summary = payback_with_incentives_all.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])
        payback_with_incentives_income_summary = (
            payback_with_incentives_income.describe(
                percentiles=list(PERCENTILES.keys())
            ).drop(["count"])
        )

        cost_results = SBEMRetrofitDistributions(
            costs=all_costs_data,
            paybacks=payback_no_incentives,
            costs_summary=retrofit_costs_summary,
            paybacks_summary=payback_no_incentives_summary,
        )
        cost_results_with_incentives = SBEMRetrofitDistributions(
            costs=all_costs_data,
            paybacks=payback_with_incentives_all,
            costs_summary=all_customers_net_costs_summary,
            paybacks_summary=payback_with_incentives_all_summary,
        )
        cost_results_with_incentives_income_eligible = SBEMRetrofitDistributions(
            costs=all_costs_data,
            paybacks=payback_with_incentives_income,
            costs_summary=income_eligible_net_costs_summary,
            paybacks_summary=payback_with_incentives_income_summary,
        )

        return {
            "original": original_results,
            "upgraded": new_results,
            "delta": delta_results,
            "retrofit": cost_results,
            "retrofit_with_incentives_all": cost_results_with_incentives,
            "retrofit_with_incentives_income_eligible": cost_results_with_incentives_income_eligible,
        }

    @property
    def changed_context_fields(
        self,
    ) -> tuple[dict[str, float | str | int], dict[str, float | str | int]]:
        """The changed context fields."""
        return {
            f"feature.semantic.{f}": v
            for f, v in self.upgraded_semantic_field_context.items()
            if v != self.original.semantic_field_context[f]
        }, {
            f: v
            for f, v in self.upgraded_semantic_field_context.items()
            if v != self.original.semantic_field_context[f]
        }

    def select_cost_entities(self, costs: "RetrofitQuantities") -> "RetrofitQuantities":
        """Select the cost entities that are relevant to the changed context fields."""
        _changed_feature_fields, changed_context_fields = self.changed_context_fields
        cost_entities: list[RetrofitQuantity] = []
        for context_field, context_value in changed_context_fields.items():
            original_value = self.original.semantic_field_context[context_field]
            retrofit_cost_candidates: list[RetrofitQuantity] = []
            for cost in costs.quantities:
                if cost.trigger_column != context_field:
                    continue
                if cost.final != context_value:
                    continue
                if cost.initial is None or cost.initial == original_value:
                    retrofit_cost_candidates.append(cost)
            if len(retrofit_cost_candidates) == 0:
                msg = f"No retrofit cost found for {context_field} = {original_value} -> {context_value}"
                print(msg)
            elif len(retrofit_cost_candidates) > 1:
                msg = f"Multiple retrofit costs found for {context_field} = {original_value} -> {context_value}:\n {retrofit_cost_candidates}"
                raise ValueError(msg)
            else:
                cost_entities.append(retrofit_cost_candidates[0])
        return RetrofitQuantities(
            quantities=frozenset(cost_entities), output_key="cost"
        )

    def select_incentive_entities(
        self, incentives: "RetrofitQuantities", income_level: str = "All_customers"
    ) -> "RetrofitQuantities":
        """Select the incentive entities that are relevant to the changed context fields and eligibility criteria."""
        _changed_feature_fields, changed_context_fields = self.changed_context_fields
        incentive_entities: list[RetrofitQuantity] = []

        for context_field, context_value in changed_context_fields.items():
            original_value = self.original.semantic_field_context[context_field]
            retrofit_incentive_candidates: list[RetrofitQuantity] = []

            for incentive in incentives.quantities:
                # Check if incentive matches the semantic field and final value
                if incentive.trigger_column != context_field:
                    continue
                if incentive.final != context_value:
                    continue
                if (
                    incentive.initial is not None
                    and incentive.initial != original_value
                ):
                    continue

                # Check eligibility criteria
                if not self._check_eligibility(incentive, income_level):
                    continue

                retrofit_incentive_candidates.append(incentive)

            if len(retrofit_incentive_candidates) == 0:
                msg = f"No retrofit incentive found for {context_field} = {original_value} -> {context_value} (income: {income_level})"
                print(msg)
            elif len(retrofit_incentive_candidates) > 1:
                msg = (
                    f"Multiple retrofit incentives found for {context_field} = {original_value} -> {context_value} "
                    f"(income: {income_level}); selecting all."
                )
                print(msg)
                # Include all applicable incentives (e.g., federal + state)
                incentive_entities.extend(retrofit_incentive_candidates)
            else:
                incentive_entities.append(retrofit_incentive_candidates[0])

        return RetrofitQuantities(
            quantities=frozenset(incentive_entities), output_key="incentive"
        )

    def _check_eligibility(
        self, incentive: "RetrofitQuantity", income_level: str
    ) -> bool:
        """Check if the incentive is eligible based on semantic field context."""
        # For now, assume all incentives are eligible
        # TODO: Implement proper eligibility checking based on incentive metadata
        return True

    def _check_window_wall_eligibility(
        self, features: pd.DataFrame, changed_context_fields: dict
    ) -> bool:
        """Check if window incentives are eligible based on wall insulation upgrades."""
        # Check if there's a window upgrade to Double or Triple pane
        window_upgrade = False
        if "Windows" in changed_context_fields:
            final_window = changed_context_fields["Windows"]
            if final_window in ["DoublePaneLowE", "TriplePaneLowE"]:
                window_upgrade = True

        # Check if there's a wall insulation upgrade
        wall_upgrade = False
        if "Walls" in changed_context_fields:
            final_walls = changed_context_fields["Walls"]
            if final_walls in [
                "FullInsulationWallsCavity",
                "FullInsulationWallsCavityExterior",
            ]:
                wall_upgrade = True

        return window_upgrade and wall_upgrade

    def compute_retrofit_costs(
        self, features: pd.DataFrame, all_costs: "RetrofitQuantities"
    ) -> pd.DataFrame:
        """Compute the retrofit costs for the changed context fields."""
        cost_entities = self.select_cost_entities(all_costs)
        costs_df = cost_entities.compute(features)
        for feature in all_costs.all_semantic_features:
            col_name = f"cost.{feature}"
            if col_name not in costs_df.columns:
                costs_df[col_name] = 0
        return costs_df

    def compute_incentives_split(
        self,
        features: pd.DataFrame,
        all_customers_incentives: "RetrofitQuantities",
        income_eligible_incentives: "RetrofitQuantities",
        costs_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute the incentives for the changed context fields using split incentive files."""
        # Compute features for incentive calculations after inference
        features_for_incentives = self.original.make_retrofit_incentive_features(
            features
        )

        # Compute incentives for both income levels using the split configurations
        all_customers_df = all_customers_incentives.compute(
            features_for_incentives, costs_df
        )
        income_eligible_df = income_eligible_incentives.compute(
            features_for_incentives, costs_df
        )

        # Add missing columns for all semantic features
        all_semantic_features = set(
            all_customers_incentives.all_semantic_features
            + income_eligible_incentives.all_semantic_features
        )
        for feature in all_semantic_features:
            col_name = f"incentive.{feature}"
            if col_name not in all_customers_df.columns:
                all_customers_df[col_name] = 0
            if col_name not in income_eligible_df.columns:
                income_eligible_df[col_name] = 0

        return (
            all_customers_df,
            income_eligible_df,
        )

    def _compute_net_costs_for_incentives(
        self, costs_df: pd.DataFrame, incentives_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Helper method to compute net costs for a given incentive dataframe."""
        net_costs = costs_df.copy()

        # Compute individual net costs for each semantic field
        for col in incentives_df.columns:
            if col.startswith("incentive."):
                semantic_field = col.split(".")[1]
                cost_col = f"cost.{semantic_field}"
                if cost_col in net_costs.columns:
                    net_col = f"net_cost.{semantic_field}"
                    net_costs[net_col] = net_costs[cost_col] - incentives_df[col]

        # calc method here should be total_cost - total_incentive
        # Calculate net cost total: cost total minus incentive total, or sum of existing net costs
        net_costs["net_cost.Total"] = (
            net_costs["cost.Total"] - incentives_df["incentive.Total"]
        )

        # If either column doesn't exist, fall back to summing existing net_cost columns
        if (
            "cost.Total" not in net_costs.columns
            or "incentive.Total" not in incentives_df.columns
        ):
            net_cost_cols = [
                col for col in net_costs.columns if col.startswith("net_cost.")
            ]
            if net_cost_cols:
                net_costs["net_cost.Total"] = net_costs[net_cost_cols].sum(axis=1)
            elif "cost.Total" in net_costs.columns:
                net_costs["net_cost.Total"] = net_costs["cost.Total"]

        # Ensure net cost total is not negative (clip at 0) and warn when clipping
        negatives = net_costs["net_cost.Total"] < 0
        if negatives.any():
            logging.warning(
                "Net cost became negative after incentives; clipping to 0 for %d rows",
                int(negatives.sum()),
            )
            net_costs.loc[negatives, "net_cost.Total"] = 0

        return net_costs

    def compute_net_costs(
        self,
        costs_df: pd.DataFrame,
        all_customers_incentives_df: pd.DataFrame,
        income_eligible_incentives_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute net costs after incentives for both income levels."""
        all_customers_net = self._compute_net_costs_for_incentives(
            costs_df, all_customers_incentives_df
        )
        income_eligible_net = self._compute_net_costs_for_incentives(
            costs_df, income_eligible_incentives_df
        )
        return all_customers_net, income_eligible_net

    def compute_payback(
        self,
        costs_df: pd.DataFrame,
        delta_results: SBEMDistributions,
        cost_column: str = "cost.Total",
    ) -> pd.Series:
        """Compute the payback for the changed context fields.

        Args:
            costs_df: DataFrame containing cost columns
            delta_results: SBEMDistributions containing savings data
            cost_column: Name of the cost column to use for payback calculation
        """
        total_costs = costs_df[cost_column]
        total_savings = delta_results.totals.Gross.FuelCost
        #  TODO: when savings are negative, we are still going to show a very large payback
        # despite the fact that it should be "never".
        payback: pd.Series = total_costs / np.clip(total_savings, 0.01, np.inf)
        payback[payback < 0] = np.inf

        return payback

    def compute_payback_with_incentives(
        self, net_costs_df: pd.DataFrame, delta_results: SBEMDistributions
    ) -> pd.Series:
        """Compute the payback for the changed context fields with incentives applied."""
        return self.compute_payback(net_costs_df, delta_results, "net_cost.Total")


class QuantityFactor(ABC):
    """An abstract base class for all quantity factors (costs, incentives, etc.)."""

    @abstractmethod
    def compute(
        self, features: pd.DataFrame, context_df: pd.DataFrame | None = None
    ) -> pd.Series:
        """Compute the quantity for a given feature.

        Args:
            features: DataFrame containing building features
            context_df: Optional DataFrame containing context (e.g., costs for percentage incentives)
        """
        pass


class LinearQuantity(BaseModel, QuantityFactor, frozen=True):
    """A quantity that is linear in the product of a set of indicator columns."""

    coefficient: float = Field(
        ..., description="The factor to multiply the indicator columns by.", gt=0
    )
    indicator_cols: tuple[str, ...] = Field(
        ...,
        description="The column(s) in the features that should be multiplied by the coefficient.",
    )
    error_scale: float | None = Field(
        ...,
        description="The expected error of the quantity estimate.",
        ge=0,
        le=1,
    )
    units: str = Field(..., description="Final units after all calculations.")
    per: str = Field(
        ...,
        description="A description of the quantity factor's rate unit (e.g. 'total linear facade distance').",
    )
    description: str = Field(
        ...,
        description="An explanation of the quantity factor (e.g. 'must walk the perimeter of each floor to punch holes for insulation.').",
    )
    source: str = Field(
        ...,
        description="The source of the quantity factor (e.g. 'ASHRAE Fundamentals').",
    )

    def compute(
        self, features: pd.DataFrame, context_df: pd.DataFrame | None = None
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
        # Check if this is a context-based calculation (for percentage incentives)
        if self.indicator_cols == ("cost_context",) and context_df is not None:
            # This is a percentage-based calculation using context
            context_cols = [
                col for col in context_df.columns if col.startswith("cost.")
            ]
            if context_cols:
                context_col = context_cols[0]
                base_quantity = context_df[context_col] * self.coefficient
                result = base_quantity
            else:
                result = pd.Series(0, index=features.index)
        else:
            # Standard linear calculation: coefficient * product of indicator columns
            base = features[list(self.indicator_cols)].product(axis=1)
            result = self.coefficient * base

        # Apply error scaling (multiplicative)
        if self.error_scale is not None:
            error_factor = np.random.normal(1.0, self.error_scale, len(features)).clip(
                min=0
            )
            result = result * pd.Series(error_factor, index=features.index)

        return result


class FixedQuantity(BaseModel, QuantityFactor, frozen=True):
    """A quantity that is a fixed amount (costs, incentives, etc.)."""

    amount: float
    error_scale: float | None = Field(
        ...,
        description="The expected error of the quantity estimate.",
        ge=0,
        le=1,
    )
    description: str = Field(
        ...,
        description="A description of the fixed quantity.",
    )
    source: str = Field(
        ...,
        description="The source of the fixed quantity.",
    )

    def compute(
        self, features: pd.DataFrame, context_df: pd.DataFrame | None = None
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
        if self.error_scale is None:
            return pd.Series(
                np.full(len(features), self.amount),
                index=features.index,
            )
        else:
            # Use absolute value of amount for standard deviation to avoid negative scale
            std_dev = abs(self.amount) * self.error_scale
            return pd.Series(
                np.random.normal(self.amount, std_dev, len(features)).clip(min=0),
                index=features.index,
            )


class PercentQuantity(BaseModel, QuantityFactor, frozen=True):
    """A percentage-based quantity (typically used for incentives)."""

    percent: float = Field(
        ..., description="The percentage to apply to the context value.", gt=0, le=1
    )
    limit: float | None = Field(None, description="The maximum quantity amount.")
    limit_unit: str | None = Field(None, description="The unit of the limit.")
    error_scale: float | None = Field(
        ...,
        description="The expected error of the quantity estimate.",
        ge=0,
        le=1,
    )
    description: str = Field(
        ...,
        description="A description of the percentage quantity.",
    )
    source: str = Field(
        ...,
        description="The source of the percentage quantity.",
    )

    def compute(
        self, features: pd.DataFrame, context_df: pd.DataFrame | None = None
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
        if context_df is None:
            return pd.Series(0, index=features.index)

        context_cols = [col for col in context_df.columns if col.startswith("cost.")]
        if not context_cols:
            return pd.Series(0, index=features.index)

        context_col = context_cols[0]
        base_quantity = context_df[context_col] * self.percent

        if self.error_scale is not None:
            error_factor = np.random.normal(1.0, self.error_scale, len(features)).clip(
                min=0
            )
            base_quantity = base_quantity * pd.Series(
                error_factor, index=features.index
            )

        if self.limit is not None:
            base_quantity = base_quantity.clip(upper=self.limit)

        return base_quantity


class RetrofitQuantity(BaseModel, frozen=True):
    """The quantity of a given retrofit intervention (e.g., costs, incentives)."""

    trigger_column: str = Field(..., description="The semantic field to retrofit.")
    initial: str | None = Field(
        ...,
        description="The initial value of the semantic field (`None` signifies any source).",
    )
    final: str = Field(..., description="The final value of the semantic field.")
    quantity_factors: frozenset[LinearQuantity | FixedQuantity | PercentQuantity] = (
        Field(..., description="The quantity factors for the retrofit.")
    )

    @field_validator("quantity_factors", mode="before")
    @classmethod
    def infer_quantity_factor_types(cls, v):
        """Convert quantity factors to proper types based on type field."""
        if isinstance(v, list):
            inferred_factors = []
            for factor in v:
                if isinstance(factor, dict):
                    factor_type = factor.get("type")
                    if factor_type == "FixedQuantity":
                        inferred_factors.append(FixedQuantity(**factor))
                    elif factor_type == "LinearQuantity":
                        # Convert indicator_cols from list to tuple for hashability
                        factor_copy = factor.copy()
                        if isinstance(factor_copy["indicator_cols"], list):
                            factor_copy["indicator_cols"] = tuple(
                                factor_copy["indicator_cols"]
                            )
                        inferred_factors.append(LinearQuantity(**factor_copy))
                    elif factor_type == "PercentQuantity":
                        inferred_factors.append(PercentQuantity(**factor))
                    else:
                        print(f"Warning: Unknown factor type '{factor_type}', skipping")
                        continue
                else:
                    # Already a proper type
                    inferred_factors.append(factor)
            return frozenset(inferred_factors)
        return v

    def compute(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        output_key: str = "cost",
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
        if len(self.quantity_factors) == 0:
            print(
                f"No quantity factors found for {self.trigger_column} = {self.initial} -> {self.final}"
            )
            return pd.Series(
                0, index=features.index, name=f"{output_key}.{self.trigger_column}"
            )

        # IMMUTABLE ORDER: Fixed -> Percent -> Linear
        # This ensures proper sequential application for incentives:
        # 1. Fixed incentives subtract from gross cost
        # 2. Percent incentives apply to net cost after fixed incentives
        # 3. Linear incentives apply to remaining cost
        factor_types = (FixedQuantity, PercentQuantity, LinearQuantity)
        total_quantity = pd.Series(0, index=features.index)
        current_context = context_df.copy() if context_df is not None else None

        for factor_class in factor_types:
            type_factors = [
                f for f in self.quantity_factors if isinstance(f, factor_class)
            ]

            for factor in type_factors:
                if isinstance(factor, FixedQuantity):
                    # Fixed quantities: add directly to total
                    quantity_amount = factor.compute(features, current_context)
                    total_quantity += quantity_amount

                    # Note: Context updates are now handled at the RetrofitQuantities level

                elif isinstance(factor, PercentQuantity):
                    # Percentage quantities: apply to current net cost (after fixed incentives)
                    if current_context is not None:
                        context_col = f"cost.{self.trigger_column}"
                        if context_col in current_context.columns:
                            temp_context_df = pd.DataFrame(
                                {context_col: current_context[context_col]},
                                index=current_context.index,
                            )
                            quantity_amount = factor.compute(features, temp_context_df)
                            total_quantity += quantity_amount

                            # Note: Context updates are now handled at the RetrofitQuantities level
                    else:
                        # No context provided - can't compute percentage
                        quantity_amount = pd.Series(0, index=features.index)

                elif isinstance(factor, LinearQuantity):
                    # Linear quantities: compute based on features or context
                    quantity_amount = factor.compute(features, current_context)
                    total_quantity += quantity_amount

                    # Note: Context updates are now handled at the RetrofitQuantities level

        return total_quantity.rename(f"{output_key}.{self.trigger_column}")


class RetrofitQuantities(BaseModel, frozen=True):
    """The quantities associated with each of the retrofit interventions."""

    quantities: frozenset[RetrofitQuantity]
    output_key: str = Field(
        ...,
        description="The key prefix to use for the output quantity columns.",
    )

    @property
    def all_semantic_features(self) -> list[str]:
        """The list of all features that are used in the quantities."""
        return list({quantity.trigger_column for quantity in self.quantities})

    def compute(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        final_values: set[str] | None = None,
    ) -> pd.DataFrame:
        """Compute the quantities for a given feature."""
        if not self.quantities:
            return pd.DataFrame({f"{self.output_key}.Total": [0] * len(features)})

        quantities_by_trigger = self._group_quantities_by_trigger(final_values)
        # Accumulate metadata for incentives
        incentive_metadata_rows: list[dict] = []
        all_quantities = self._compute_quantities_by_trigger(
            features, context_df, quantities_by_trigger, incentive_metadata_rows
        )

        data = self._combine_quantities(all_quantities, features)

        # Attach metadata column for incentives
        if self.output_key == "incentive":
            # One list of dicts per row, matching index
            metadata_series = pd.Series(
                [incentive_metadata_rows] * len(features),
                index=features.index,
                name="incentive_metadata",
            )
            data = pd.concat([data, metadata_series], axis=1)

        return data

    def _group_quantities_by_trigger(self, final_values: set[str] | None) -> dict:
        """Group quantities by trigger_column and final value."""
        quantities_by_trigger = {}
        for quantity in self.quantities:
            trigger = quantity.trigger_column
            final = quantity.final

            if final_values is not None and final not in final_values:
                continue

            key = (trigger, final)
            if key not in quantities_by_trigger:
                quantities_by_trigger[key] = []
            quantities_by_trigger[key].append(quantity)

        return quantities_by_trigger

    def _compute_quantities_by_trigger(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None,
        quantities_by_trigger: dict,
        incentive_metadata_rows: list[dict] | None = None,
    ) -> list[pd.Series]:
        """Compute quantities for each trigger and final value combination."""
        all_quantities = []
        for (trigger, final), quantities in quantities_by_trigger.items():
            total_quantity = self._compute_quantity_for_trigger(
                features,
                context_df,
                trigger,
                final,
                quantities,
                incentive_metadata_rows,
            )
            final_quantity = total_quantity.rename(
                f"{self.output_key}.{trigger}.{final}"
            )
            all_quantities.append(final_quantity)
        return all_quantities

    def _compute_quantity_for_trigger(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None,
        trigger: str,
        final: str,
        quantities: list[RetrofitQuantity],
        incentive_metadata_rows: list[dict] | None = None,
    ) -> pd.Series:
        """Compute the total quantity for a specific trigger."""
        current_context = context_df.copy() if context_df is not None else None
        total_quantity = pd.Series(0, index=features.index)

        if self.output_key == "incentive" and current_context is not None:
            total_quantity = self._compute_incentive_quantity(
                features,
                current_context,
                trigger,
                final,
                quantities,
                incentive_metadata_rows,
            )
        else:
            # For costs, compute normally
            for quantity in quantities:
                quantity_result = quantity.compute(
                    features, current_context, self.output_key
                )
                total_quantity += quantity_result

        return total_quantity

    def _compute_incentive_quantity(
        self,
        features: pd.DataFrame,
        current_context: pd.DataFrame,
        trigger: str,
        final: str,
        quantities: list[RetrofitQuantity],
        incentive_metadata_rows: list[dict] | None = None,
    ) -> pd.Series:
        """Compute incentive quantity with proper clipping."""
        context_col = f"cost.{trigger}"
        if context_col not in current_context.columns:
            return self._compute_quantities_simple(
                features, current_context, quantities
            )

        gross_cost = current_context[context_col].iloc[0]
        if len(quantities) == 1:
            return self._compute_single_incentive(
                features,
                current_context,
                trigger,
                final,
                quantities[0],
                gross_cost,
                incentive_metadata_rows,
            )
        else:
            return self._compute_multiple_incentives(
                features,
                current_context,
                trigger,
                final,
                quantities,
                context_col,
                gross_cost,
                incentive_metadata_rows,
            )

    def _compute_quantities_simple(
        self,
        features: pd.DataFrame,
        current_context: pd.DataFrame,
        quantities: list[RetrofitQuantity],
    ) -> pd.Series:
        """Compute quantities without context clipping."""
        total_quantity = pd.Series(0, index=features.index)
        for quantity in quantities:
            quantity_result = quantity.compute(
                features, current_context, self.output_key
            )
            total_quantity += quantity_result
        return total_quantity

    def _compute_single_incentive(
        self,
        features: pd.DataFrame,
        current_context: pd.DataFrame,
        trigger: str,
        final: str,
        quantity: RetrofitQuantity,
        gross_cost: float,
        incentive_metadata_rows: list[dict] | None = None,
    ) -> pd.Series:
        """Compute single incentive with clipping."""
        quantity_result = quantity.compute(features, current_context, self.output_key)
        quantity_amount = min(quantity_result.iloc[0], gross_cost)

        # Collect metadata
        if incentive_metadata_rows is not None:
            # Extract program info from first factor that has description/source
            program_name = None
            source = None
            for f in quantity.quantity_factors:
                if hasattr(f, "description") and hasattr(f, "source"):
                    program_name = f.description
                    source = f.source
                    break
            incentive_metadata_rows.append({
                "trigger": trigger,
                "final": final,
                "program_name": program_name,
                "source": source,
                "amount_applied": float(quantity_amount),
            })
        return pd.Series(quantity_amount, index=features.index)

    def _compute_multiple_incentives(
        self,
        features: pd.DataFrame,
        current_context: pd.DataFrame,
        trigger: str,
        final: str,
        quantities: list[RetrofitQuantity],
        context_col: str,
        gross_cost: float,
        incentive_metadata_rows: list[dict] | None = None,
    ) -> pd.Series:
        """Compute multiple incentives sequentially with clipping."""
        sorted_quantities = sorted(quantities, key=self._get_primary_factor_type)
        total_quantity = pd.Series(0, index=features.index)
        remaining_cost = gross_cost

        for quantity in sorted_quantities:
            if remaining_cost <= 0:
                break

            temp_context_df = pd.DataFrame(
                {context_col: pd.Series(remaining_cost, index=features.index)},
                index=features.index,
            )

            quantity_result = quantity.compute(
                features, temp_context_df, self.output_key
            )
            quantity_amount = min(quantity_result.iloc[0], remaining_cost)
            quantity_result = pd.Series(quantity_amount, index=features.index)

            total_quantity += quantity_result
            remaining_cost -= quantity_amount

            # Collect metadata per applied incentive
            if incentive_metadata_rows is not None:
                program_name = None
                source = None
                for f in quantity.quantity_factors:
                    if hasattr(f, "description") and hasattr(f, "source"):
                        program_name = f.description
                        source = f.source
                        break
                incentive_metadata_rows.append({
                    "trigger": trigger,
                    "final": final,
                    "program_name": program_name,
                    "source": source,
                    "amount_applied": float(quantity_amount),
                })

        return total_quantity.clip(upper=gross_cost)

    def _get_primary_factor_type(self, quantity: RetrofitQuantity) -> int:
        """Get the primary factor type for ordering purposes."""
        if any(isinstance(f, FixedQuantity) for f in quantity.quantity_factors):
            return 0  # Fixed first
        elif any(isinstance(f, PercentQuantity) for f in quantity.quantity_factors):
            return 1  # Percent second
        elif any(isinstance(f, LinearQuantity) for f in quantity.quantity_factors):
            return 2  # Linear third
        else:
            return 3  # Unknown last

    def _combine_quantities(
        self, all_quantities: list[pd.Series], features: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine all quantities into final dataframe."""
        if all_quantities:
            quantities_df = pd.concat(all_quantities, axis=1)
        else:
            quantities_df = pd.DataFrame()

        if not quantities_df.empty:
            total = quantities_df.sum(axis=1).rename(f"{self.output_key}.Total")
            data = pd.concat([quantities_df, total], axis=1)
        else:
            data = pd.DataFrame({f"{self.output_key}.Total": [0] * len(features)})
        return data

    @classmethod
    def Open(cls, path: Path) -> "RetrofitQuantities":
        """Open a retrofit quantities file."""
        if path in RETROFIT_QUANTITY_CACHE:
            return RETROFIT_QUANTITY_CACHE[path]
        else:
            with open(path) as f:
                data = json.load(f)

            retrofit_quantities = RetrofitQuantities.model_validate(data)
            RETROFIT_QUANTITY_CACHE[path] = retrofit_quantities
            return retrofit_quantities


RETROFIT_QUANTITY_CACHE: dict[Path, RetrofitQuantities] = {}


class IncentiveFactor(ABC):
    """An abstract base class for all incentive factors."""

    @abstractmethod
    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.Series:
        """Compute the incentive for a given feature and cost."""
        pass


# provided features
"""
feature.geometry.long_edge
feature.geometry.short_edge
feature.geometry.num_floors
feature.geometry.orientation
feature.semantic.*
feature.extra_spaces.basement.exists
feature.extra_spaces.basement.occupied
feature.extra_spaces.basement.conditioned
feature.extra_spaces.attic.exists
feature.extra_spaces.attic.occupied
feature.extra_spaces.attic.conditioned
"""
# sampled features
"""
feature.geometry.wwr
feature.geometry.f2f_height
(attic_pitch)
feature.extra_spaces.basement.use_fraction
feature.extra_spaces.attic.use_fraction
"""

# comptued features
"""
feature.geometry.zoning
feature.geometry.orientation.cos
feature.geometry.orientation.sin
feature.geometry.aspect_ratio
feature.geometry.shading_mask_{i:02d}
feature.geometry.energy_model_conditioned_area
feature.geometry.energy_model_occupied_area
(attic_height) -> feature.geometry.attic_height
(lat, lon) -> feature.weather.*
"""

# features used for costs
"""
"feature.calculated.heating_capacity_kW"
"feature.location.county"
"feature.system.has_cooling"
"feature.system.has_cooling_not"
"feature.system.has_gas"
"feature.system.has_gas_not"
"""
