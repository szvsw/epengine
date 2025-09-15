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
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)
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
OIL_HEATING_SYSTEMS = ["OilHeating"]
NG_HEATING_SYSTEMS = ["NaturalGasHeating", "NaturalGasCondensingHeating"]
INCOME_BRACKETS = [
    "IncomeEligible",
    "AllCustomers",
]
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
    paybacks: pd.DataFrame
    costs_summary: pd.DataFrame
    paybacks_summary: pd.DataFrame

    @property
    def serialized(self) -> BaseModel:
        """Serialize the SBEMRetrofitDistributions dataframes into a SBEMRetrofitDistributionsSpec."""
        field_specs = {}
        field_datas = {}
        percentile_mapper = {v[1]: v[0] for v in PERCENTILES.values()}

        # Create copies of summary dataframes with percentile mapper applied
        costs_summary_renamed = self.costs_summary.rename(index=percentile_mapper)
        # paybacks_summary_renamed = self.paybacks_summary.rename(index=percentile_mapper)

        # Process all columns in the costs dataframe
        for col in self.costs.columns:
            col_name = col.split(".")[-1]
            field_specs[col_name] = (SummarySpec, Field(title=col))

            # Get summary data for this column
            if col in costs_summary_renamed.columns:
                field_data = costs_summary_renamed.loc[:, col].to_dict()
            else:
                # If column not in summary, create a summary
                # Skip non-numerical columns (like metadata columns)
                if not pd.api.types.is_numeric_dtype(self.costs[col]):
                    continue
                col_summary = (
                    self.costs[col]
                    .describe(percentiles=list(PERCENTILES.keys()))
                    .drop(["count"])
                )
                # Apply the same percentile mapper to rename the index
                col_summary_renamed = col_summary.rename(index=percentile_mapper)
                field_data = col_summary_renamed.to_dict()

            if not col.startswith(("cost.", "incentive.", "net_cost.", "payback")):
                msg = f"Column {col} is not a cost, incentive, net cost, or payback column"
                raise ValueError(msg)

            # Set units based on column type
            if col.startswith("payback"):
                field_data["units"] = "years"
            else:
                field_data["units"] = "USD"

            field_datas[col_name] = SummarySpec(**field_data)

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

    # Additional fields for cost and incentive calculations
    county: str

    # Current solar system size in kW (0 if no solar)
    current_solar_size_kW: float = Field(
        default=0.0, ge=0, description="Current solar system size in kW"
    )

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
        base_features = pd.Series({
            k: v for k, v in spec.feature_dict.items() if k.startswith("feature.")
        })
        # Add the new fields that are not part of SBEMSimulationSpec
        base_features["feature.location.county"] = self.county

        return base_features

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
        # TODO: WOuld it be quicker to create one row and then expand it instead - check the pyinstrument
        return pd.DataFrame([base_features] * n)

    @cached_property
    def generator(self) -> np.random.Generator:
        """The random number generator for the experiment."""
        return np.random.default_rng(42)

    def add_solar_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add solar-related features to the features DataFrame."""
        # Add solar yield as a base feature (Massachusetts average)
        yield_sampler = ClippedNormalSampler(
            mean=1100,
            std=150,
            clip_min=800,
            clip_max=1400,
        )
        features["feature.solar.yield_kWh_per_kW_year"] = yield_sampler.sample(
            features, len(features), self.generator
        )
        panel_power_density_sampler = ClippedNormalSampler(
            mean=180,
            std=50,
            clip_min=120,
            clip_max=300,
        )
        features["feature.solar.panel_power_density_w_per_m2"] = (
            panel_power_density_sampler.sample(features, len(features), self.generator)
        )
        # Set default value for OnsiteSolar if not provided
        if "feature.semantic.OnsiteSolar" not in features.columns:
            features["feature.semantic.OnsiteSolar"] = "NoSolarPV"
        # Calculate upgraded solar coverage based on semantic field
        features["feature.solar.upgraded_coverage"] = np.where(
            features["feature.semantic.OnsiteSolar"] == "LowSolarPV",
            0.25,
            np.where(
                features["feature.semantic.OnsiteSolar"] == "MedSolarPV",
                0.50,
                np.where(
                    features["feature.semantic.OnsiteSolar"] == "MaxSolarPV",
                    1.0,
                    0.0,
                ),
            ),
        )

        features["feature.upgrade.solar_pv_kW"] = 0.0

        return features

    def update_max_solar_coverage(
        self, features: pd.DataFrame, electricity_consumption: pd.Series
    ) -> pd.DataFrame:
        """Update the MaxSolarPV coverage when electricity consumption data is available."""
        features = features.copy()

        # Set coverage values based on solar type
        features["feature.solar.upgraded_coverage"] = np.where(
            features["feature.semantic.OnsiteSolar"] == "MaxSolarPV",
            features["feature.solar.upgraded_coverage"],
            np.where(
                features["feature.semantic.OnsiteSolar"] == "LowSolarPV",
                0.25,
                np.where(
                    features["feature.semantic.OnsiteSolar"] == "MedSolarPV",
                    0.50,
                    0.0,
                ),
            ),
        )

        # Handle MaxSolarPV samples - calculate feasible coverage for each
        max_solar_mask = features["feature.semantic.OnsiteSolar"] == "MaxSolarPV"
        if max_solar_mask.any():
            max_feasible = self.calculate_feasible_solar_coverage(
                features.loc[max_solar_mask],
                electricity_consumption.loc[max_solar_mask],
            )
            # Use the maximum feasible coverage for each sample
            features.loc[max_solar_mask, "feature.solar.upgraded_coverage"] = (
                max_feasible
            )

        return features

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

        # Add solar features
        df = self.add_solar_features(df)

        # Defer solar upgrade capacity calculation to the post-prediction phase

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
        self,
        features: pd.DataFrame,
        peak_results: pd.DataFrame,
        elect_eui: pd.Series,
    ) -> pd.DataFrame:
        """Compute features needed for cost calculations after inference has been run.

        Features needed: heating_capacity_kW, county_indicator one hot encoded, has_gas, has_cooling
        """
        # Start with existing features
        cost_features = features.copy()

        # MOve from W to kW
        peak_heating_per_m2 = peak_results["Heating"] * 1000

        # TODO: check if the regression uses heating capacity or electrical capacity
        # effective_cop = features["feature.factors.system.heat.effective_cop"]
        # electrical_capacity_kW = gross_peak_kw / effective_cop
        # heating_capacity_kW = gross_peak_kw

        safety_factor = 1.2
        raw_capacity_kW = peak_heating_per_m2 * safety_factor

        # Map calculated capacity to nearest available equipment size (unless above max)
        available_sizes_kW = np.array([
            5.3,
            7.0,
            8.8,
            10.5,
            12.3,
            14.1,
            17.6,
            21.1,
            24.6,
            28.1,
            31.6,
        ])

        def map_to_available_size(v: float) -> float:
            max_size = available_sizes_kW.max()
            if v > max_size:
                return float(v)
            # round up to the smallest available size that is >= v
            idx = int(np.searchsorted(available_sizes_kW, v, side="left"))
            return float(available_sizes_kW[idx])

        cost_features["feature.calculated.heating_capacity_kW"] = raw_capacity_kW.apply(
            map_to_available_size
        )
        COUNTIES = [
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

        def oh_col_name_for_county(county: str) -> str:
            return f"feature.location.in_county.{county}"

        for county in COUNTIES:
            cost_features[oh_col_name_for_county(county)] = (
                features["feature.location.county"] == county
            )

        county_oh_col_names = [oh_col_name_for_county(county) for county in COUNTIES]
        # Check that each row has exactly one county set to True
        county_df = cost_features[county_oh_col_names]
        rows_with_no_county = county_df.sum(axis=1) == 0
        # check if there are any rows with no county set to True - pydantic validation of the request shoudl catch if nothing is entered, but if an unknown county is entered, then this warning will appear. We coudl update the pydantic valdation to include only the counties we currently consider
        if rows_with_no_county.any():
            msg = f"Found {rows_with_no_county.sum()} rows with no county indicators set to True"
            logging.warning(msg)

        # Add gas availability indicator
        cost_features["feature.system.has_gas.true"] = features[
            "feature.semantic.Heating"
        ].isin(NG_HEATING_SYSTEMS)
        cost_features["feature.system.has_gas.false"] = ~cost_features[
            "feature.system.has_gas.true"
        ]

        # Add cooling availability indicator
        cooling_systems = [
            "ACWindow",
            "ACCentral",
            "WindowASHP",
            "ASHPCooling",
            "GSHPCooling",
        ]
        cost_features["feature.system.has_cooling.true"] = features[
            "feature.semantic.Cooling"
        ].isin(cooling_systems)
        cost_features["feature.system.has_cooling.false"] = ~cost_features[
            "feature.system.has_cooling.true"
        ]

        # Add solar system size for retrofit cost calculations
        if features["feature.semantic.OnsiteSolar"].iloc[0] in [
            "LowSolarPV",
            "MedSolarPV",
            "MaxSolarPV",
        ]:
            # Calculate the required solar system size for the upgrade
            electricity_consumption = elect_eui * self.actual_conditioned_area_m2

            # Update MaxSolarPV coverage if needed
            features = self.update_max_solar_coverage(features, electricity_consumption)
            target_coverage = features["feature.solar.upgraded_coverage"].iloc[0]

            required_system_size = self.calculate_upgraded_solar_system_size(
                target_coverage,
                features,
                electricity_consumption,
            )
            cost_features["feature.upgrade.solar_pv_kW"] = required_system_size

        else:
            # No solar upgrade, set to 0
            cost_features["feature.upgrade.solar_pv_kW"] = 0.0

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
        ].isin(NG_HEATING_SYSTEMS)
        incentive_features["feature.fuel.oil"] = features[
            "feature.semantic.Heating"
        ].isin(OIL_HEATING_SYSTEMS)

        # Add window/wall eligibility check as one-hot encoding
        # Check if there's a window upgrade to Double or Triple pane
        window_upgrade = features["feature.semantic.Windows"].isin([
            "DoublePaneLowE",
            "TriplePaneLowE",
        ])

        # Check if there's a wall insulation upgrade
        wall_upgrade = features["feature.semantic.Walls"].isin([
            "FullInsulationWallsCavity",
            "FullInsulationWallsCavityExterior",
        ])

        # Window incentives are eligible only if both window and wall upgrades are present
        incentive_features["feature.eligibility.window_wall_combo"] = (
            window_upgrade & wall_upgrade
        )
        # Add income bracket features (default to False, will be overridden per bracket)
        for bracket in INCOME_BRACKETS:
            incentive_features[f"feature.homeowner.in_bracket.{bracket}"] = False

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
            df_disaggregated_fuels (pd.DataFrame): The disaggregated fuels with both actual and net electricity.
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

        # Create ACTUAL electricity consumption DataFrame (before solar)
        actual_electricity_consumption = pd.concat(
            [heat_elec, cool_elec, dhw_elec, lighting, equipment],
            axis=1,
            keys=["Heating", "Cooling", "Domestic Hot Water", "Lighting", "Equipment"],
        )[df_end_uses.columns]

        actual_electricity_consumption = cast(
            pd.DataFrame, actual_electricity_consumption
        )

        # Apply solar generation to get NET electricity consumption
        net_electricity_consumption = self.apply_solar_to_electricity_consumption(
            actual_electricity_consumption, df_features
        )

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

        # Store actual electricity consumption for solar calculations
        self._actual_electricity_consumption = actual_electricity_consumption

        # Use net electricity consumption for the main fuel disaggregation

        df_disaggregated_fuels = pd.concat(
            [actual_electricity_consumption, net_electricity_consumption, gas, oil],
            axis=1,
            keys=["Electricity", "NetElectricity", "NaturalGas", "Oil"],
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
        net_elec_costs = df_disaggregated_fuels["NetElectricity"].mul(elec_rate, axis=0)
        gas_costs = df_disaggregated_fuels["NaturalGas"].mul(gas_rate, axis=0)
        oil_costs = df_disaggregated_fuels["Oil"].mul(oil_rate, axis=0)

        disaggregated_costs = pd.concat(
            [elec_costs, net_elec_costs, gas_costs, oil_costs],
            axis=1,
            keys=["Electricity", "NetElectricity", "NaturalGas", "Oil"],
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
        net_elec_emissions = df_disaggregated_fuels["NetElectricity"].mul(
            elec_emissions_factors, axis=0
        )
        gas_emissions = df_disaggregated_fuels["NaturalGas"].mul(
            gas_emissions_factors, axis=0
        )
        oil_emissions = df_disaggregated_fuels["Oil"].mul(oil_emissions_factors, axis=0)

        disaggregated_emissions = pd.concat(
            [elec_emissions, net_elec_emissions, gas_emissions, oil_emissions],
            axis=1,
            keys=["Electricity", "NetElectricity", "NaturalGas", "Oil"],
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

    def calculate_solar_generation(
        self, system_size_kW: float | pd.Series, features: pd.DataFrame
    ) -> pd.Series:
        """Calculate annual solar generation for a given system size."""
        # Use the solar yield feature that was already generated
        solar_yield_series = features["feature.solar.yield_kWh_per_kW_year"]
        # remove this function to be part of another function
        annual_generation = system_size_kW * solar_yield_series
        return annual_generation

    def calculate_feasible_solar_coverage(
        self,
        features: pd.DataFrame,
        actual_electricity_consumption: pd.Series,
    ) -> pd.Series:
        """Calculate the maximum feasible solar coverage based on roof area."""
        # Get roof surface area
        roof_area_m2 = features["feature.geometry.computed.roof_surface_area"]

        # Solar panel assumptions
        # panel_efficiency = 0.22
        panel_power_density = features["feature.solar.panel_power_density_w_per_m2"]
        max_roof_utilization = 0.50  # Only 50% of roof can be covered, assuming we have a fire safety boundary. This is a very high level estimate
        # TODO: Account for roof angle, orientation, and shading

        # Calculate maximum solar capacity possible
        max_solar_area_m2 = roof_area_m2 * max_roof_utilization
        max_solar_capacity_kW = (max_solar_area_m2 * panel_power_density) / 1000
        max_local_solar_capacity_kW = 25
        mask = max_solar_capacity_kW > max_local_solar_capacity_kW
        # set a cap at 25 kW total system size
        if mask.any():
            max_solar_capacity_kW = max_solar_capacity_kW.clip(
                upper=max_local_solar_capacity_kW
            )
            msg = "Max solar capacity is capped at 25 kW"
            logging.warning(msg)
        # Calculate annual generation at max capacity
        max_annual_generation = (
            max_solar_capacity_kW * features["feature.solar.yield_kWh_per_kW_year"]
        )
        # Use actual electricity consumption if provided, otherwise use placeholder

        total_electricity_kWh = actual_electricity_consumption
        # Calculate maximum feasible coverage
        max_coverage = max_annual_generation / np.maximum(total_electricity_kWh, 1)
        max_coverage = np.clip(max_coverage, 0, 1)

        return pd.Series(max_coverage, index=features.index)

    def calculate_upgraded_solar_system_size(
        self,
        target_coverage: float,
        features: pd.DataFrame,
        total_electricity_consumption: pd.Series,
    ) -> pd.Series:
        """Calculate the solar system size needed to achieve target coverage."""
        # Calculate maximum feasible coverage
        max_feasible_coverage = self.calculate_feasible_solar_coverage(
            features, total_electricity_consumption
        )
        # Check feasibility and cap if necessary
        # Cap each sample to its own maximum feasible coverage
        actual_coverage = np.minimum(target_coverage, max_feasible_coverage)

        # Log warning if any samples are being capped
        if (actual_coverage < target_coverage).any():
            max_feasible = max_feasible_coverage.max()
            logging.warning(
                f"Requested solar coverage {target_coverage:.1%} exceeds maximum feasible coverage "
                f"{max_feasible:.1%} for some samples. Capping at maximum feasible amount."
            )
        # Calculate required annual generation
        required_annual_generation = total_electricity_consumption * actual_coverage
        # Calculate required system size (using average yield of 1300 kWh/kW/year)
        required_system_size_kW = (
            required_annual_generation / features["feature.solar.yield_kWh_per_kW_year"]
        )

        return pd.Series(required_system_size_kW, index=features.index)

    def apply_solar_to_electricity_consumption(
        self,
        electricity_consumption: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply solar generation to electricity consumption to get net consumption."""
        net_consumption = electricity_consumption.copy()

        # Get the OnsiteSolar semantic field value - vectorized approach
        if "feature.semantic.OnsiteSolar" not in features.columns:
            # No solar column, return original consumption
            return net_consumption

        onsite_solar = cast(pd.Series, features["feature.semantic.OnsiteSolar"])

        # Calculate total electricity consumption per sample
        total_electricity = np.where(
            onsite_solar == "ExistingSolarPV",
            electricity_consumption.sum(axis=1),
            electricity_consumption.sum(axis=1) * self.actual_conditioned_area_m2,
        )

        # Initialize solar offset array
        solar_offset = pd.Series(0.0, index=features.index)

        # Handle existing solar systems
        existing_solar_mask = (onsite_solar == "ExistingSolarPV") & (
            self.current_solar_size_kW > 0
        )
        if existing_solar_mask.any():
            current_solar_generation = self.calculate_solar_generation(
                self.current_solar_size_kW,
                features.loc[existing_solar_mask],
            )
            current_solar_EUI = (
                current_solar_generation / self.actual_conditioned_area_m2
            )
            existing_total_electricity = electricity_consumption.loc[
                existing_solar_mask
            ].sum(axis=1)
            existing_solar_offset = current_solar_EUI.clip(
                upper=existing_total_electricity
            )
            solar_offset.loc[existing_solar_mask] = existing_solar_offset

        upgraded_solar_mask = onsite_solar.isin([
            "LowSolarPV",
            "MedSolarPV",
            "MaxSolarPV",
        ])
        if upgraded_solar_mask.any():
            upgraded_features = features.loc[upgraded_solar_mask].copy()
            upgraded_total_electricity = (
                electricity_consumption.loc[upgraded_solar_mask].sum(axis=1)
                * self.actual_conditioned_area_m2
            )

            # Update MaxSolarPV coverage if we need to and can't reach 100%
            upgraded_features = self.update_max_solar_coverage(
                upgraded_features, upgraded_total_electricity
            )

            # Calculate systm size for each covg value
            unique_coverages = upgraded_features[
                "feature.solar.upgraded_coverage"
            ].unique()
            upgraded_system_size = pd.Series(0.0, index=upgraded_features.index)

            for coverage in unique_coverages:
                coverage_mask = (
                    upgraded_features["feature.solar.upgraded_coverage"] == coverage
                )
                if coverage_mask.any():
                    coverage_features = upgraded_features.loc[coverage_mask]
                    coverage_electricity = upgraded_total_electricity.loc[coverage_mask]

                    system_size = self.calculate_upgraded_solar_system_size(
                        coverage, coverage_features, coverage_electricity
                    )
                    upgraded_system_size.loc[coverage_mask] = system_size

            upgraded_solar_generation = self.calculate_solar_generation(
                upgraded_system_size, upgraded_features
            )

            upgraded_solar_offset = upgraded_solar_generation.clip(
                upper=upgraded_total_electricity
            )
            solar_offset.loc[upgraded_solar_mask] = upgraded_solar_offset
        solar_mask = (onsite_solar != "NoSolarPV") & (solar_offset > 0)
        if solar_mask.any():
            # Calculate reduction factor for each sample
            reduction_factor = cast(pd.Series, 1 - (solar_offset / total_electricity))
            reduction_factor = reduction_factor.where(total_electricity > 0, 1.0)
            net_consumption.loc[solar_mask] = electricity_consumption.loc[
                solar_mask
            ].mul(reduction_factor.loc[solar_mask], axis=0)

        return net_consumption


class SBEMInferenceSavingsRequestSpec(BaseModel):
    """An inference request spec for computing savings using matched samples."""

    original: SBEMInferenceRequestSpec
    upgraded_semantic_field_context: dict[str, float | str | int]

    def run(  # noqa: C901
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

        # Create an upgraded spec with the new semantic field context
        upgraded_spec = SBEMInferenceRequestSpec(
            **{
                k: v
                for k, v in self.original.model_dump().items()
                if k != "semantic_field_context"
            },
            semantic_field_context=self.upgraded_semantic_field_context,
        )
        # Run inference with the upgraded spec
        new_results = upgraded_spec.run(n)
        # Get peak results for cost calculations
        new_results_raw = upgraded_spec.predict(
            upgraded_spec.source_feature_transform.transform(
                upgraded_spec.make_features(n)[0]
            )
        )
        new_results_peak = cast(pd.DataFrame, new_results_raw["Peak"])
        # new_results_energy = cast(pd.DataFrame, new_results_raw["Energy"])

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

        # Load consolidated incentive file
        incentives_path = Path(__file__).parent / "data" / "incentives.json"
        incentive_config = RetrofitQuantities.Open(incentives_path)

        cost_config = RetrofitQuantities.Open(costs_path)

        # Compute features for cost calculations after inference
        # For solar upgrades, we need to use the ACTUAL electricity consumption (before solar)
        # to calculate the system size needed, not the net consumption if there is alrearyd some solar
        electricity_eui = upgraded_spec._actual_electricity_consumption.sum(axis=1)

        # Calculate the feature distributions for solar features (yield, coverage)
        new_features_with_solar = upgraded_spec.add_solar_features(new_features)

        features_for_costs = upgraded_spec.make_retrofit_cost_features(
            new_features_with_solar, new_results_peak, electricity_eui
        )

        retrofit_costs = self.compute_retrofit_costs(features_for_costs, cost_config)

        features_for_incentives = self.original.make_retrofit_incentive_features(
            new_features
        )
        # Compute incentives for all income brackets
        incentives_by_bracket = self.compute_incentives_by_bracket(
            new_features,
            features_for_incentives,
            incentive_config,
            retrofit_costs,
        )

        # Compute net costs after incentives for all brackets
        net_costs_by_bracket = self.compute_net_costs_by_bracket(
            retrofit_costs, incentives_by_bracket
        )

        # Compute paybacks
        payback_no_incentives = self.compute_payback(retrofit_costs, delta_results)

        # Compute paybacks for each income bracket
        paybacks_by_bracket = {}
        for bracket, net_costs_df in net_costs_by_bracket.items():
            paybacks_by_bracket[bracket] = self.compute_payback(
                net_costs_df, delta_results, f"net_cost.Total.{bracket}"
            )

        # Create separate cost data for each bracket with only the relevant incentive/net_cost and payback columns
        def create_bracket_costs_data(bracket: str | None = None) -> pd.DataFrame:
            """Create cost data for a specific bracket with only its relevant incentive, net_cost, and payback columns.

            - For retrofit (no incentives): include only cost.* and a single payback column
            - For a bracket: include cost.*, that bracket's incentive.* and net_cost.* columns (suffix stripped), and a single payback column
            """
            if bracket is None:
                # No incentives view: costs + payback only
                payback_df = pd.DataFrame({"payback": payback_no_incentives})
                return pd.concat(
                    [
                        retrofit_costs,
                        payback_df,
                    ],
                    axis=1,
                )

            # Bracket-specific view
            incentives_df = incentives_by_bracket[bracket].copy()
            net_costs_df = net_costs_by_bracket[bracket].copy()

            # Strip the bracket suffix from incentive columns, including metadata
            incentive_rename: dict[str, str] = {}
            suffix = f".{bracket}"
            for col in list(incentives_df.columns):
                if col.startswith("incentive.") and col.endswith(suffix):
                    incentive_rename[col] = col[: -len(suffix)]
            if incentive_rename:
                incentives_df = incentives_df.rename(columns=incentive_rename)

            # Strip the bracket suffix from net_cost columns
            net_cost_rename: dict[str, str] = {}
            for col in list(net_costs_df.columns):
                if col.startswith("net_cost.") and col.endswith(suffix):
                    net_cost_rename[col] = col[: -len(suffix)]
            if net_cost_rename:
                net_costs_df = net_costs_df.rename(columns=net_cost_rename)

            payback_df = pd.DataFrame({"payback": paybacks_by_bracket[bracket]})

            return pd.concat(
                [
                    retrofit_costs,
                    incentives_df,
                    net_costs_df,
                    payback_df,
                ],
                axis=1,
            )

        # Create the main costs data (for retrofit without incentives)
        all_costs_data = create_bracket_costs_data()

        def summarize(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
            # Filter out non-numerical columns (like metadata columns) before summarizing
            if isinstance(data, pd.DataFrame):
                # Only include columns that are numeric or can be converted to numeric
                numeric_data = data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    # If no numeric columns, return empty summary
                    return pd.DataFrame()
                summary = numeric_data.describe(
                    percentiles=list(PERCENTILES.keys())
                ).drop(["count"])
                # Apply percentile mapper to rename the index
                percentile_mapper = {v[1]: v[0] for v in PERCENTILES.values()}
                return summary.rename(index=percentile_mapper)
            else:
                # For Series, check if it's numeric
                if pd.api.types.is_numeric_dtype(data):
                    summary = data.describe(percentiles=list(PERCENTILES.keys())).drop([
                        "count"
                    ])
                    # Apply percentile mapper to rename the index
                    percentile_mapper = {v[1]: v[0] for v in PERCENTILES.values()}
                    return summary.rename(index=percentile_mapper)
                else:
                    # Return empty Series if not numeric
                    return pd.Series(dtype=float)

        # Create summary statistics for the entire costs data (includes costs, incentives, net costs, paybacks)
        all_costs_summary = summarize(all_costs_data)

        # Create summary statistics for each bracket
        net_costs_summaries = {}
        payback_summaries = {}
        for bracket, net_costs_df in net_costs_by_bracket.items():
            net_costs_summaries[bracket] = summarize(net_costs_df)
            payback_summaries[bracket] = summarize(paybacks_by_bracket[bracket])

        payback_no_incentives_summary = summarize(payback_no_incentives)

        # Create cost results for retrofit (no incentives)
        cost_results = SBEMRetrofitDistributions(
            costs=all_costs_data,
            paybacks=pd.DataFrame({"payback": payback_no_incentives}),
            costs_summary=cast(pd.DataFrame, all_costs_summary),
            paybacks_summary=cast(
                pd.DataFrame, pd.DataFrame({"payback": payback_no_incentives_summary})
            ),
        )

        # Create cost results for each bracket
        cost_results_by_bracket = {}
        for bracket in incentives_by_bracket:
            bracket_costs_data = create_bracket_costs_data(bracket)
            bracket_costs_summary = summarize(bracket_costs_data)
            cost_results_by_bracket[bracket] = SBEMRetrofitDistributions(
                costs=bracket_costs_data,
                paybacks=pd.DataFrame({"payback": paybacks_by_bracket[bracket]}),
                costs_summary=cast(pd.DataFrame, bracket_costs_summary),
                paybacks_summary=cast(
                    pd.DataFrame, pd.DataFrame({"payback": payback_summaries[bracket]})
                ),
            )

        # Build return dictionary
        result_dict = {
            "original": original_results,
            "upgraded": new_results,
            "delta": delta_results,
            "retrofit": cost_results,
        }

        # Add results for each income bracket
        for bracket in incentives_by_bracket:
            result_dict[f"retrofit_with_incentives_{bracket.lower()}"] = (
                cost_results_by_bracket[bracket]
            )

        return result_dict

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

    # TODO: Move this to RetrofitQuantities
    def select_relevant_entities(
        self, entities: "RetrofitQuantities"
    ) -> "RetrofitQuantities":
        """Select the entities that are relevant to the changed context fields."""
        _changed_feature_fields, changed_context_fields = self.changed_context_fields
        selected_entities: list[RetrofitQuantity] = []

        for context_field, context_value in changed_context_fields.items():
            original_value = self.original.semantic_field_context[context_field]
            entity_candidates: list[RetrofitQuantity] = []

            for entity in entities.quantities:
                # Check if entity matches the semantic field and final value
                if entity.trigger_column != context_field:
                    continue
                if entity.final != context_value:
                    continue
                if entity.initial is not None and entity.initial != original_value:
                    continue

                entity_candidates.append(entity)

            if len(entity_candidates) == 0:
                msg = f"No {entities.output_key} entity found for {context_field} = {original_value} -> {context_value}"
                logging.warning(msg)
                # Debug output for solar

            elif len(entity_candidates) > 1:
                if entities.raise_on_duplicate_trigger:
                    msg = f"Multiple {entities.output_key} entities found for {context_field} = {original_value} -> {context_value}:\n {entity_candidates}"
                    raise ValueError(msg)
                else:
                    msg = f"Multiple {entities.output_key} entities found for {context_field} = {original_value} -> {context_value}; selecting all."
                    logging.warning(msg)
                    # Include all applicable entities (e.g., federal + state incentives)
                    selected_entities.extend(entity_candidates)
            else:
                selected_entities.append(entity_candidates[0])

        return RetrofitQuantities(
            quantities=frozenset(selected_entities),
            output_key=entities.output_key,
            raise_on_duplicate_trigger=entities.raise_on_duplicate_trigger,
            create_metadata=entities.create_metadata,
            metadata_aggregation=entities.metadata_aggregation,
        )

    def compute_retrofit_costs(
        self, features: pd.DataFrame, all_costs_config: "RetrofitQuantities"
    ) -> pd.DataFrame:
        """Compute the retrofit costs for the changed context fields."""
        cost_entities = self.select_relevant_entities(all_costs_config)
        costs_df = cost_entities.compute(features)
        for feature in all_costs_config.all_trigger_features:
            col_name = f"cost.{feature}"
            if col_name not in costs_df.columns:
                costs_df[col_name] = 0
        return costs_df

    def compute_incentives_by_bracket(
        self,
        features: pd.DataFrame,
        features_for_incentives: pd.DataFrame,
        incentive_config: "RetrofitQuantities",
        costs_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Compute incentives for all income brackets using consolidated incentive file.

        Returns:
            dict: Dictionary mapping income bracket names to incentive DataFrames
        """
        # Compute features for incentive calculations after inference
        incentive_entities = self.select_relevant_entities(incentive_config)

        incentives_by_bracket = {}
        for bracket in INCOME_BRACKETS:
            bracket_features = features_for_incentives.copy()
            for b in INCOME_BRACKETS:
                bracket_features[f"feature.homeowner.in_bracket.{b}"] = False
            bracket_features[f"feature.homeowner.in_bracket.{bracket}"] = True
            bracket_incentives = incentive_entities.compute(bracket_features, costs_df)
            # TODO: can we combine the methods of retrofit costs and incentives?
            all_trigger_features = set(incentive_config.all_trigger_features)
            for feature in all_trigger_features:
                col_name = f"incentive.{feature}"
                if col_name not in bracket_incentives.columns:
                    bracket_incentives[col_name] = 0

            # Rename bracket-specific incentive columns to avoid collisions when concatenated
            rename_map: dict[str, str] = {}
            for col in bracket_incentives.columns:
                if col.startswith("incentive."):
                    rename_map[col] = f"{col}.{bracket}"
            if rename_map:
                bracket_incentives = bracket_incentives.rename(columns=rename_map)

            # Handle bracket-specific metadata if enabled
            if (
                incentive_config.create_metadata
                and incentive_config.metadata_aggregation == "bracket"
            ):
                metadata_col = f"incentive.metadata.{bracket}"
                if "incentive.metadata" in bracket_incentives.columns:
                    bracket_incentives[metadata_col] = bracket_incentives[
                        "incentive.metadata"
                    ]
                    bracket_incentives = bracket_incentives.drop(
                        columns=["incentive.metadata"]
                    )

            incentives_by_bracket[bracket] = bracket_incentives

        return incentives_by_bracket

    def _compute_net_costs_for_incentives(
        self, costs_df: pd.DataFrame, incentives_df: pd.DataFrame, bracket: str
    ) -> pd.DataFrame:
        """Helper method to compute net costs for a given incentive dataframe."""
        # Start with an empty DataFrame, only add net cost columns
        net_costs = pd.DataFrame(index=costs_df.index)

        # Compute individual net costs for each semantic field
        for col in incentives_df.columns:
            if col.startswith("incentive."):
                # Handle renamed incentive columns with bracket suffixes
                # Format: incentive.{semantic_field}.{bracket}
                # Note: semantic_field can contain dots (e.g., "Heating.ASHPHeating")
                parts = col.split(".")
                if len(parts) < 3:
                    # Skip malformed columns
                    continue

                # Last part is the bracket, everything between incentive. and .{bracket} is the semantic field
                bracket_suffix = parts[-1]
                semantic_field = ".".join(parts[1:-1])

                # TODO: Add a check that that value of semantic field is in the list of semantic fields, and raise an error
                cost_col = f"cost.{semantic_field}"
                if cost_col in costs_df.columns:
                    net_col = f"net_cost.{semantic_field}.{bracket_suffix}"
                    net_costs[net_col] = costs_df[cost_col] - incentives_df[col]

        # Calculate net cost total: cost total minus incentive total
        incentive_total_col = f"incentive.Total.{bracket}"
        net_costs[f"net_cost.Total.{bracket}"] = (
            costs_df["cost.Total"] - incentives_df[incentive_total_col]
        )
        # Ensure net cost total is not negative (clip at 0) and warn when clipping
        net_cost_total_col = f"net_cost.Total.{bracket}"
        negatives = net_costs[net_cost_total_col] < 0
        if negatives.any():
            logging.warning(
                "Net cost became negative after incentives; clipping to 0 for %d rows",
                int(negatives.sum()),
            )
            net_costs.loc[negatives, net_cost_total_col] = 0

        return net_costs

    def compute_net_costs_by_bracket(
        self,
        costs_df: pd.DataFrame,
        incentives_by_bracket: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Compute net costs after incentives for all income brackets."""
        net_costs_by_bracket = {}

        for bracket, incentives_df in incentives_by_bracket.items():
            net_costs = self._compute_net_costs_for_incentives(
                costs_df, incentives_df, bracket
            )
            net_costs_by_bracket[bracket] = net_costs

        return net_costs_by_bracket

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


class QuantityFactor(ABC):
    """An abstract base class for all quantity factors (costs, incentives, etc.)."""

    @abstractmethod
    def compute(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        trigger_column: str | None = None,
    ) -> pd.Series:
        """Compute the quantity for a given feature.

        Args:
            features: DataFrame containing building features
            context_df: Optional DataFrame containing context (e.g., costs for percentage incentives)
            trigger_column: Optional trigger column name for selecting correct cost column
        """
        pass


class LinearQuantity(BaseModel, QuantityFactor, frozen=True):
    """A quantity that is linear in the product of a set of indicator columns."""

    coefficient: float = Field(
        ..., description="The factor to multiply the indicator columns by.", ge=0
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
    # limit: float | None = Field(
    #     None,
    #     description="The maximum quantity amount.",
    # )
    # limit_unit: str | None = Field(
    #     None,
    #     description="The unit of the limit.",
    # )
    # TODO: Add a limit/cap to the quantity factor

    def compute(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        trigger_column: str | None = None,
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
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
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        trigger_column: str | None = None,
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
        # TODO: Add a clip value as a boolean, and then only execute if it's true, min/max on the clipping
        if self.error_scale is None:
            return pd.Series(
                np.full(len(features), self.amount),
                index=features.index,
            )
        else:
            # Use absolute value of amount for standard deviation to avoid negative scale
            std_dev = abs(self.amount) * self.error_scale
            is_positive = self.amount > 0

            random_values = np.random.normal(self.amount, std_dev, len(features))

            if is_positive:
                clipped_values = np.maximum(random_values, 0)
            else:
                clipped_values = np.minimum(random_values, 0)

            return pd.Series(clipped_values, index=features.index)


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
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        trigger_column: str | None = None,
    ) -> pd.Series:
        """Compute the quantity for a given feature."""
        if context_df is None:
            return pd.Series(0, index=features.index)

        # Find cost columns
        context_cols = [col for col in context_df.columns if col.startswith("cost.")]
        if not context_cols:
            msg = f"No cost columns found in context_df. Available columns: {list(context_df.columns)}"
            raise ValueError(msg)

        detailed_prefix = f"cost.{trigger_column}."
        detailed_cols = [c for c in context_cols if c.startswith(detailed_prefix)]
        if len(detailed_cols) == 1:
            context_col = detailed_cols[0]
        elif len(detailed_cols) > 1:
            # Multiple finals present; fall back to flat trigger column
            expected_cost_col = f"cost.{trigger_column}"
            if expected_cost_col in context_cols:
                context_col = expected_cost_col
            else:
                # If flat not present, choose the first detailed as best-effort
                context_col = detailed_cols[0]
        else:
            expected_cost_col = f"cost.{trigger_column}"
            if expected_cost_col in context_cols:
                context_col = expected_cost_col
            else:
                msg = f"Required cost column '{expected_cost_col}' not found in context_df. Available cost columns: {context_cols}"
                raise ValueError(msg)

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
    order: tuple[str, ...] = Field(
        default=("FixedQuantity", "PercentQuantity", "LinearQuantity"),
        description="The order in which to apply quantity factors.",
    )
    # TODO: Add a validator to the quantity factors to ensure that the order is valid/is not a subset of the order components

    # TODO Add a model validator that checks that all the components in order are present in quantity_factors
    @model_validator(mode="after")
    def validate_order(self):
        """Validate that the order is valid/is not a subset of the order components."""
        if not set(self.order).issubset([
            c.__class__.__name__ for c in self.quantity_factors
        ]):
            msg = f"Order {self.order} is not a subset of the quantity factors {self.quantity_factors}"
            raise ValueError(msg)
        return self

    # TODO: Check if the serializing/deserializing is working
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
                        # FixedQuantity doesn't have indicator_cols anymore
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
                        msg = f"Warning: Unknown factor type '{factor_type}', skipping"
                        logging.warning(msg)
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
            msg = f"No quantity factors found for {self.trigger_column} = {self.initial} -> {self.final}"
            logging.warning(msg)
            return pd.Series(
                0, index=features.index, name=f"{output_key}.{self.trigger_column}"
            )
        # Map string names to actual classes
        factor_class_map = {
            "FixedQuantity": FixedQuantity,
            "PercentQuantity": PercentQuantity,
            "LinearQuantity": LinearQuantity,
        }

        total_quantity = pd.Series(0, index=features.index)
        current_context = context_df.copy() if context_df is not None else None

        for factor_type_name in self.order:
            if factor_type_name not in factor_class_map:
                msg = f"Warning: Unknown factor type '{factor_type_name}' in order, skipping"
                logging.warning(msg)
                continue

            factor_class = factor_class_map[factor_type_name]
            type_factors = [
                f for f in self.quantity_factors if isinstance(f, factor_class)
            ]

            for factor in type_factors:
                # Pass trigger_column to PercentQuantity for correct cost column selection
                if isinstance(factor, PercentQuantity):
                    quantity_amount = factor.compute(
                        features, current_context, self.trigger_column
                    )
                else:
                    # All other factor types use the same compute interface
                    quantity_amount = factor.compute(features, current_context)
                total_quantity += quantity_amount

        return total_quantity.rename(f"{output_key}.{self.trigger_column}")


class RetrofitQuantities(BaseModel, frozen=True):
    """The quantities associated with each of the retrofit interventions."""

    quantities: frozenset[RetrofitQuantity]
    output_key: str = Field(
        ...,
        description="The key prefix to use for the output quantity columns.",
    )
    raise_on_duplicate_trigger: bool = Field(
        ...,
        description="Whether to raise an error if there are duplicate triggers.",
    )
    create_metadata: bool = Field(
        ...,
        description="Whether to create metadata columns for quantities.",
    )
    metadata_aggregation: str | None = Field(
        None,
        description="How to aggregate metadata. Options: 'bracket' for bracket-specific aggregation, None for simple aggregation.",
    )

    @property
    def all_trigger_features(self) -> list[str]:
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
        # Accumulate metadata if requested
        metadata_rows: list[dict] = []
        all_quantities = self._compute_quantities_by_trigger(
            features, context_df, quantities_by_trigger, metadata_rows
        )

        data = self._combine_quantities(all_quantities, features)

        if self.create_metadata:
            # Always add the metadata column - aggregation logic will handle the rest
            metadata_series = pd.Series(
                [metadata_rows] * len(features),
                index=features.index,
                name=f"{self.output_key}.metadata",
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
        metadata_rows: list[dict] | None = None,
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
                metadata_rows,
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
        metadata_rows: list[dict] | None = None,
    ) -> pd.Series:
        """Compute the total quantity for a specific trigger."""
        current_context = context_df.copy() if context_df is not None else None
        total_quantity = pd.Series(0, index=features.index)

        # Use the existing RetrofitQuantity.compute() method for all cases
        for quantity in quantities:
            quantity_result = quantity.compute(
                features, current_context, self.output_key
            )
            total_quantity += quantity_result

        # For incentives, clip the total amount to the cost amount
        if self.output_key == "incentive" and context_df is not None:
            # Prefer detailed cost for the same trigger+final; fallback to flat trigger cost
            detailed_col = f"cost.{trigger}.{final}"
            flat_col = f"cost.{trigger}"
            if detailed_col in context_df.columns:
                total_quantity = total_quantity.clip(upper=context_df[detailed_col])
            elif flat_col in context_df.columns:
                total_quantity = total_quantity.clip(upper=context_df[flat_col])

        # Collect metadata if requested
        if metadata_rows is not None and self.create_metadata:
            self._collect_metadata(
                trigger,
                final,
                quantities,
                total_quantity,
                metadata_rows,
                features,
                current_context,
            )

        return total_quantity

    def _collect_metadata(
        self,
        trigger: str,
        final: str,
        quantities: list[RetrofitQuantity],
        total_quantity: pd.Series,
        metadata_rows: list[dict],
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
    ) -> None:
        """Collect metadata for quantities."""
        for quantity in quantities:
            # Collect metadata for each individual quantity factor that contributes
            for factor in quantity.quantity_factors:
                if hasattr(factor, "description") and hasattr(factor, "source"):
                    # Calculate the amount this specific factor contributed
                    if isinstance(factor, PercentQuantity):
                        # For PercentQuantity, we need to compute it individually
                        factor_result = factor.compute(
                            features, context_df, quantity.trigger_column
                        )
                    else:
                        # For LinearQuantity and FixedQuantity, compute individually
                        factor_result = factor.compute(features, context_df)

                    amount_applied = (
                        float(factor_result.iloc[0]) if len(factor_result) > 0 else 0.0
                    )

                    # Skip zero-amount quantities
                    if abs(amount_applied) == 0:
                        continue

                    metadata_rows.append({
                        "trigger": trigger,
                        "final": final,
                        "program_name": factor.description,
                        "source": factor.source,
                        "amount_applied": amount_applied,
                        "factor_type": factor.__class__.__name__,
                    })

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
