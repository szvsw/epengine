"""Inference request models."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Union, cast

import boto3
import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, create_model
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
        Union["SBEMDistributions", "SBEMRetrofitDistributions", "IncentiveMetadata"],
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
        _new_results_peak = cast(pd.DataFrame, new_results_raw["Peak"])
        new_results_energy = cast(pd.DataFrame, new_results_raw["Energy"])
        new_results = self.original.compute_distributions(
            new_features, new_results_energy
        )
        # new_results_peak = self.original.compute_distributions(
        #     new_features, _new_results_peak
        # )

        # Build features for retrofit costs by adding calculated heating capacity (kW)
        features_for_costs = new_features.copy(deep=True)
        features_for_costs["feature.calculated.heating_capacity_kw"] = (
            self.compute_heating_capacity_kw(_new_results_peak, features_for_costs)
        )

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

        cost_config = RetrofitCosts.Open(
            Path(__file__).parent / "data" / "retrofit-costs.json"
        )
        incentive_config = RetrofitIncentives.Open(
            Path(__file__).parent / "data" / "incentives_format.json"
        )
        retrofit_costs = self.compute_retrofit_costs(new_results.features, cost_config)
        retrofit_costs = self.compute_retrofit_costs(features_for_costs, cost_config)

        # Compute incentives for both income levels
        (
            all_customers_incentives,
            income_eligible_incentives,
            all_customers_metadata,
            income_eligible_metadata,
        ) = self.compute_incentives(
            features_for_costs, incentive_config, retrofit_costs
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
        # included incentives summary in case interesting to build distributions off it
        # all_customers_incentives_summary = all_customers_incentives.describe(
        #     percentiles=list(PERCENTILES.keys())
        # ).drop(["count"])
        # income_eligible_incentives_summary = income_eligible_incentives.describe(
        #     percentiles=list(PERCENTILES.keys())
        # ).drop(["count"])
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
            "incentive_metadata_all": all_customers_metadata,
            "incentive_metadata_income_eligible": income_eligible_metadata,
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

    def select_cost_entities(self, costs: "RetrofitCosts") -> "RetrofitCosts":
        """Select the cost entities that are relevant to the changed context fields."""
        _changed_feature_fields, changed_context_fields = self.changed_context_fields
        cost_entities: list[RetrofitCost] = []
        for context_field, context_value in changed_context_fields.items():
            original_value = self.original.semantic_field_context[context_field]
            retrofit_cost_candidates: list[RetrofitCost] = []
            for cost in costs.costs:
                if cost.semantic_field != context_field:
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
        return RetrofitCosts(costs=cost_entities)

    def select_incentive_entities(
        self, incentives: "RetrofitIncentives", income_level: str = "All_customers"
    ) -> "RetrofitIncentives":
        """Select the incentive entities that are relevant to the changed context fields and eligibility criteria."""
        _changed_feature_fields, changed_context_fields = self.changed_context_fields
        incentive_entities: list[RetrofitIncentive] = []

        for context_field, context_value in changed_context_fields.items():
            original_value = self.original.semantic_field_context[context_field]
            retrofit_incentive_candidates: list[RetrofitIncentive] = []

            for incentive in incentives.incentives:
                # Check if incentive matches the semantic field and final value
                if incentive.semantic_field != context_field:
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

        return RetrofitIncentives(incentives=incentive_entities)

    def _check_eligibility(
        self, incentive: "RetrofitIncentive", income_level: str
    ) -> bool:
        """Check if the incentive is eligible based on semantic field context."""
        eligibility = incentive.eligibility

        # Check region - right now it's just going to default to MA, since that's what all the costs are from
        if "region" in eligibility:
            region = self.original.semantic_field_context.get("Region", "MA")
            if region not in eligibility["region"]:
                return False

        # Check income level
        if "income" in eligibility and income_level not in eligibility["income"]:
            return False

        # Check semantic field eligibility criteria
        for field_name, allowed_values in eligibility.items():
            if field_name in ["region", "income"]:
                continue

            # Get the current value for this semantic field
            current_value = self.original.semantic_field_context.get(field_name)
            if current_value is None:
                # If field not present, treat as non-matching
                print(
                    f"Warning: Semantic field '{field_name}' not found in context for incentive {incentive.program}"
                )
                return False

            if current_value not in allowed_values:
                return False

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

    def compute_heating_capacity_kw(
        self, _new_results_peak: pd.DataFrame, df_features: pd.DataFrame
    ) -> pd.Series:
        """Compute heating capacity in kW based on peak heating load."""
        # Use peak heating load (per m2), scale to gross by area, divide by effective COP, and we also add a 20% safety factor in line with what most contractors are going to do
        if isinstance(_new_results_peak.columns, pd.MultiIndex):
            # Try to select 'Raw' slice if present, else use as-is
            if "Raw" in _new_results_peak.columns.get_level_values(0):
                peak_slice = _new_results_peak["Raw"]
            else:
                peak_slice = _new_results_peak
        else:
            peak_slice = _new_results_peak

        if "Heating" not in peak_slice.columns:
            # Fallback: the column may be a tuple ("Raw", "Heating") when not sliced above
            try:
                peak_heating_per_m2 = _new_results_peak[("Raw", "Heating")]
            except Exception as e:
                msg = "Expected 'Heating' in Peak results (with or without 'Raw' sublevel)."
                raise KeyError(msg) from e
        else:
            peak_heating_per_m2 = peak_slice["Heating"]
        gross_peak_kw = peak_heating_per_m2 * self.original.actual_conditioned_area_m2
        effective_cop = df_features["feature.factors.system.heat.effective_cop"]
        electrical_capacity_kw = gross_peak_kw / effective_cop
        safety_factor = 1.2
        electrical_capacity_kw = (
            pd.Series(electrical_capacity_kw, index=df_features.index) * safety_factor
        )
        return electrical_capacity_kw

    def compute_retrofit_costs(
        self, features: pd.DataFrame, all_costs: "RetrofitCosts"
    ) -> pd.DataFrame:
        """Compute the retrofit costs for the changed context fields."""
        cost_entities = self.select_cost_entities(all_costs)
        costs_df = cost_entities.compute(features)
        for feature in all_costs.all_semantic_features:
            col_name = f"cost.{feature}"
            if col_name not in costs_df.columns:
                costs_df[col_name] = 0
        return costs_df

    def compute_incentives(
        self,
        features: pd.DataFrame,
        all_incentives: "RetrofitIncentives",
        costs_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, "IncentiveMetadata", "IncentiveMetadata"]:
        """Compute the incentives for the changed context fields."""
        # Compute incentives for both income levels
        all_customers_incentives = self.select_incentive_entities(
            all_incentives, "All_customers"
        )
        income_eligible_incentives = self.select_incentive_entities(
            all_incentives, "Income_eligible"
        )

        all_customers_df = all_customers_incentives.compute(features, costs_df)
        income_eligible_df = income_eligible_incentives.compute(features, costs_df)

        for feature in all_incentives.all_semantic_features:
            col_name = f"incentive.{feature}"
            if col_name not in all_customers_df.columns:
                all_customers_df[col_name] = 0
            if col_name not in income_eligible_df.columns:
                income_eligible_df[col_name] = 0

        # Build metadata for both income levels
        all_customers_metadata = self._build_incentive_metadata(
            all_customers_incentives, features, costs_df, "All_customers"
        )
        income_eligible_metadata = self._build_incentive_metadata(
            income_eligible_incentives, features, costs_df, "Income_eligible"
        )

        return (
            all_customers_df,
            income_eligible_df,
            all_customers_metadata,
            income_eligible_metadata,
        )

    def _build_incentive_metadata(
        self,
        incentives: "RetrofitIncentives",
        features: pd.DataFrame,
        costs_df: pd.DataFrame,
        income_level: str,
    ) -> "IncentiveMetadata":
        """Build metadata about applied incentives."""
        applied_incentives = []
        total_amount = 0.0

        for incentive in incentives.incentives:
            incentive_result = incentive.compute(features, costs_df)
            amount = incentive_result.iloc[0] if len(incentive_result) > 0 else 0.0

            if amount > 0:
                # Get details from the first incentive factor
                first_factor = next(iter(incentive.incentive_factors))

                # Determine incentive type
                from epengine.models.inference import (
                    FixedIncentive,
                    PercentIncentive,
                    VariableIncentive,
                )

                if isinstance(first_factor, FixedIncentive):
                    incentive_type = "Fixed"
                elif isinstance(first_factor, PercentIncentive):
                    incentive_type = "Percent"
                elif isinstance(first_factor, VariableIncentive):
                    incentive_type = "Variable"
                else:
                    incentive_type = "Unknown"

                applied_incentive = AppliedIncentive(
                    semantic_field=incentive.semantic_field,
                    program=incentive.program,
                    amount=amount,
                    description=first_factor.description,
                    source=first_factor.source,
                    incentive_type=incentive_type,
                )
                applied_incentives.append(applied_incentive)
                total_amount += amount

        return IncentiveMetadata(
            applied_incentives=applied_incentives,
            total_incentive_amount=total_amount,
            income_level=income_level,
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
        if (
            "cost.Total" in net_costs.columns
            and "incentive.Total" in incentives_df.columns
        ):
            net_costs["net_cost.Total"] = (
                net_costs["cost.Total"] - incentives_df["incentive.Total"]
            )
        else:
            # catch all - should then have the net costs
            net_cost_cols = [
                col for col in net_costs.columns if col.startswith("net_cost.")
            ]
            if net_cost_cols:
                net_costs["net_cost.Total"] = net_costs[net_cost_cols].sum(axis=1)
            else:
                net_costs["net_cost.Total"] = net_costs["cost.Total"]

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
        self, costs_df: pd.DataFrame, delta_results: SBEMDistributions
    ) -> pd.Series:
        """Compute the payback for the changed context fields."""
        total_costs = costs_df["cost.Total"]
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
        total_net_costs = net_costs_df["net_cost.Total"]
        total_savings = delta_results.totals.Gross.FuelCost
        #  TODO: when savings are negative, we are still going to show a very large payback
        # despite the fact that it should be "never".
        payback: pd.Series = total_net_costs / np.clip(total_savings, 0.01, np.inf)
        payback[payback < 0] = np.inf

        return payback


class Cost(ABC):
    """An abstract base class for all costs."""

    @abstractmethod
    def compute(self, features: pd.DataFrame) -> pd.Series:
        """Compute the cost for a given feature."""
        pass


class VariableCost(BaseModel, Cost):
    """A cost that can be based on building features, calculated values, and conditional factors."""

    # Fixed intercept amount (same for all buildings)
    intercept: float = Field(
        default=0.0,
        description="Fixed amount to add to every building's cost calculation.",
    )

    # Feature-based components
    feature_components: list[dict] = Field(
        default_factory=list,
        description="List of feature-based cost components with coefficients and indicator columns.",
    )

    # Calculated components
    calculated_components: list[dict] = Field(
        default_factory=list,
        description="List of calculated cost components with coefficients and calculation methods.",
    )

    # Conditional adjustments
    conditional_factors: dict[str, dict[str, float]] | None = Field(
        None,
        description="Conditional factors to apply based on building characteristics.",
    )

    # Legacy support for simple coefficient-based costs
    coefficient: float | None = Field(
        None, description="Legacy coefficient for backward compatibility.", gt=0
    )
    indicator_cols: list[str] | None = Field(
        None,
        description="Legacy indicator columns for backward compatibility.",
    )

    error_scale: float | None = Field(
        ...,
        description="The expected error of the cost estimate.",
        ge=0,
        le=1,
    )
    units: str = Field(..., description="Final units after all calculations.")
    per: str = Field(
        ...,
        description="A description of the cost factor's rate unit (e.g. 'total linear facade distance').",
    )
    description: str = Field(
        ...,
        description="An explanation of the cost factor (e.g. 'must walk the perimeter of each floor to punch holes for insulation.').",
    )
    source: str = Field(
        ...,
        description="The source of the cost factor (e.g. 'ASHRAE Fundamentals').",
    )

    def compute(self, features: pd.DataFrame) -> pd.Series:
        """Compute the cost for a given feature."""
        # Start with intercept for new-style costs, or 1 for legacy coefficient-based costs
        if self.coefficient is not None and self.indicator_cols is not None:
            # Legacy coefficient-based costs: start with 1, then multiply by coefficient
            result = pd.Series(1.0, index=features.index)
        else:
            # New-style costs: start with intercept
            result = pd.Series(self.intercept, index=features.index)

        # Handle legacy coefficient-based costs (multiplicative)
        if self.coefficient is not None and self.indicator_cols is not None:
            base = features[self.indicator_cols].product(axis=1)
            result = result * self.coefficient * base

        # Apply feature-based components (additive)
        for component in self.feature_components:
            coeff = component["coefficient"]
            cols = component["indicator_cols"]
            component_result = features[cols].product(axis=1) * coeff
            result = result + component_result

        # Apply calculated components (additive)
        for component in self.calculated_components:
            coeff = component["coefficient"]
            calc_method = component["calculation"]
            calculated_value = self._compute_calculated_feature(calc_method, features)
            component_result = calculated_value * coeff
            result = result + component_result

        # Apply conditional factors (additive)
        if self.conditional_factors:
            result = self._apply_conditional_factors(result, features)

        # Apply error scaling (multiplicative)
        if self.error_scale is not None:
            error_factor = np.random.normal(1.0, self.error_scale, len(features)).clip(
                min=0
            )
            result = result * pd.Series(error_factor, index=features.index)

        return result

    def _compute_calculated_feature(self, calc_method: str, features: pd.DataFrame):
        """Compute calculated features like heating capacity."""
        # TODO: potentially also calculated the cooling peak (once stabilized), to ensure that the heat pump is sized to the correct max load
        if calc_method == "heating_capacity_kw":
            precomputed_col = "feature.calculated.heating_capacity_kw"
            if precomputed_col in features.columns:
                return features[precomputed_col]
            # Fallback: approximate capacity from conditioned area if peaks are not available
            area_col = "feature.geometry.energy_model_conditioned_area"
            if area_col in features.columns:
                return features[area_col] * 25 * 0.0031546
            msg = "Missing required features to compute heating_capacity_kw."
            raise KeyError(msg)
        msg = f"Unknown calculated feature method: {calc_method}"
        raise NotImplementedError(msg)

    def _apply_conditional_factors(
        self, result: pd.Series, features: pd.DataFrame
    ) -> pd.Series:
        """Apply conditional factors based on building characteristics."""
        if not self.conditional_factors:
            return result

        for factor_name, factor_values in self.conditional_factors.items():
            if factor_name == "county":
                result = self._apply_county_factors(result, features, factor_values)
            elif factor_name == "has_gas":
                result = self._apply_gas_factors(result, features, factor_values)
            elif factor_name == "has_cooling":
                result = self._apply_cooling_factors(result, features, factor_values)

        return result

    def _apply_county_factors(
        self, result: pd.Series, features: pd.DataFrame, factor_values: dict
    ) -> pd.Series:
        """Apply county-based conditional factors."""
        county_col = self._find_county_column(features)
        if county_col is not None:
            for county, value in factor_values.items():
                mask = features[county_col] == county
                result[mask] = result[mask] + value
        return result

    def _find_county_column(self, features: pd.DataFrame) -> str | None:
        """Find the county column in features."""
        for col in features.columns:
            if "county" in col.lower():
                return col
        return None

    def _apply_gas_factors(
        self, result: pd.Series, features: pd.DataFrame, factor_values: dict
    ) -> pd.Series:
        """Apply gas-based conditional factors."""
        gas_heating_systems = ["NaturalGasHeating", "NaturalGasCondensingHeating"]

        for gas_status, value in factor_values.items():
            has_gas = gas_status == "true"
            mask = features["feature.semantic.Heating"].isin(gas_heating_systems)
            if not has_gas:
                mask = ~mask
            result[mask] = result[mask] + value

        return result

    def _apply_cooling_factors(
        self, result: pd.Series, features: pd.DataFrame, factor_values: dict
    ) -> pd.Series:
        """Apply cooling-based conditional factors."""
        cooling_systems = [
            "ACWindow",
            "ACCentral",
            "WindowASHP",
            "ASHPCooling",
            "GSHPCooling",
        ]

        for cooling_status, value in factor_values.items():
            has_cooling = cooling_status == "true"
            mask = features["feature.semantic.Cooling"].isin(cooling_systems)
            if not has_cooling:
                mask = ~mask
            result[mask] = result[mask] + value

        return result


class FixedCost(BaseModel, Cost):
    """A cost that is a fixed amount."""

    amount: float
    error_scale: float | None = Field(
        ...,
        description="The expected error of the cost estimate.",
        ge=0,
        le=1,
    )
    description: str = Field(
        ...,
        description="A description of the fixed cost (e.g. 'the cost of a new thermostat install').",
    )
    source: str = Field(
        ...,
        description="The source of the fixed cost (e.g. 'ASHRAE Fundamentals').",
    )

    def compute(self, features: pd.DataFrame) -> pd.Series:
        """Compute the cost for a given feature."""
        if self.error_scale is None:
            return pd.Series(
                np.full(len(features), self.amount),
                index=features.index,
            )
        else:
            return pd.Series(
                np.random.normal(
                    self.amount, self.amount * self.error_scale, len(features)
                ).clip(min=0),
                index=features.index,
            )


class RetrofitCost(BaseModel):
    """The cost of a retrofit intervention."""

    semantic_field: str = Field(..., description="The semantic field to retrofit.")
    initial: str | None = Field(
        ...,
        description="The initial value of the semantic field (`None` signifies any source).",
    )
    final: str = Field(..., description="The final value of the semantic field.")
    cost_factors: list[VariableCost | FixedCost] = Field(
        ..., description="The cost factors for the retrofit."
    )

    def compute(self, features: pd.DataFrame) -> pd.Series:
        """Compute the cost for a given feature."""
        if len(self.cost_factors) == 0:
            print(
                f"No cost factors found for {self.semantic_field} = {self.initial} -> {self.final}"
            )
        columns = [cost.compute(features) for cost in self.cost_factors]
        return (
            pd.concat(columns, axis=1).sum(axis=1).rename(f"cost.{self.semantic_field}")
        )


class RetrofitCosts(BaseModel):
    """The costs associated with each of the retrofit interventions."""

    costs: list[RetrofitCost]

    @property
    def all_semantic_features(self) -> list[str]:
        """The list of all features that are used in the costs."""
        return list({cost.semantic_field for cost in self.costs})

    def compute(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute the cost for a given feature."""
        if self.costs:
            # Compute all costs first
            cost_series_list = [cost.compute(features) for cost in self.costs]

            # Create DataFrame from cost series
            costs = pd.DataFrame(cost_series_list).T
            total = costs.sum(axis=1).rename("cost.Total")
            data = pd.concat([costs, total], axis=1)
            return data
        else:
            return pd.DataFrame({"cost.Total": [0] * len(features)})

    @classmethod
    def Open(cls, path: Path) -> "RetrofitCosts":
        """Open a retrofit costs file."""
        if path in RETROFIT_COST_CACHE:
            return RETROFIT_COST_CACHE[path]
        else:
            with open(path) as f:
                data = json.load(f)
            retrofit_costs = RetrofitCosts.model_validate(data)
            RETROFIT_COST_CACHE[path] = retrofit_costs
            return retrofit_costs


RETROFIT_COST_CACHE: dict[Path, RetrofitCosts] = {}


class IncentiveFactor(ABC):
    """An abstract base class for all incentive factors."""

    @abstractmethod
    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.Series:
        """Compute the incentive for a given feature and cost."""
        pass


class FixedIncentive(BaseModel, IncentiveFactor):
    """A fixed incentive amount."""

    amount: float
    error_scale: float | None = Field(
        ...,
        description="The expected error of the incentive estimate, if there are associated ranges due to regions, contractors, etc.",
        ge=0,
        le=1,
    )
    units: Literal["USD"]
    description: str = Field(
        ...,
        description="A description of the fixed incentive.",
    )
    source: str = Field(
        ...,
        description="The source of the fixed incentive.",
    )

    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.Series:
        """Compute the incentive for a given feature."""
        if self.error_scale is None:
            return pd.Series(
                np.full(len(features), self.amount),
                index=features.index,
            )
        else:
            return pd.Series(
                np.random.normal(
                    self.amount, self.amount * self.error_scale, len(features)
                ).clip(min=0),
                index=features.index,
            )


class VariableIncentive(BaseModel, IncentiveFactor):
    """A variable incentive that depends on features."""

    coefficient: float = Field(
        ..., description="The factor to multiply a target by.", gt=0
    )
    error_scale: float | None = Field(
        ...,
        description="The expected error of the incentive estimate.",
        ge=0,
        le=1,
    )
    units: Literal["USD/ton", "USD/kW", "USD/m2", "USD/per_unit"]
    indicator_cols: list[str] = Field(
        ...,
        description="The column(s) in the source data that should be multiplied by the coefficient.",
    )
    description: str = Field(
        ...,
        description="An explanation of the incentive factor.",
    )
    source: str = Field(
        ...,
        description="The source of the incentive factor.",
    )

    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.Series:
        """Compute the incentive for a given feature."""
        base = features[self.indicator_cols].product(axis=1)
        if self.error_scale is None:
            return self.coefficient * base
        else:
            coefficient = np.random.normal(
                self.coefficient,
                self.coefficient * self.error_scale,
                len(features),
            ).clip(min=0)
            return base * pd.Series(coefficient, index=base.index)


class PercentIncentive(BaseModel, IncentiveFactor):
    """A percentage-based incentive."""

    percent: float = Field(
        ..., description="The percentage to apply to the cost.", gt=0, le=1
    )
    limit: float | None = Field(None, description="The maximum incentive amount.")
    limit_unit: Literal["USD"] | None = Field(
        None, description="The unit of the limit."
    )
    error_scale: float | None = Field(
        ...,
        description="The expected error of the incentive estimate.",
        ge=0,
        le=1,
    )
    description: str = Field(
        ...,
        description="A description of the percentage incentive.",
    )
    source: str = Field(
        ...,
        description="The source of the percentage incentive.",
    )

    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.Series:
        """Compute the incentive for a given feature."""
        # Get the cost column (should be the only one in the filtered costs_df)
        cost_cols = [col for col in costs_df.columns if col.startswith("cost.")]

        if not cost_cols:
            return pd.Series(0, index=features.index)

        cost_col = cost_cols[0]
        base_incentive = costs_df[cost_col] * self.percent

        if self.error_scale is not None:
            error_factor = np.random.normal(1.0, self.error_scale, len(features)).clip(
                min=0
            )
            base_incentive = base_incentive * pd.Series(
                error_factor, index=features.index
            )

        if self.limit is not None:
            base_incentive = base_incentive.clip(upper=self.limit)

        return base_incentive


class RetrofitIncentive(BaseModel):
    """The incentive for a retrofit intervention."""

    semantic_field: str = Field(..., description="The semantic field to retrofit.")
    initial: str | None = Field(
        ...,
        description="The initial value of the semantic field (`None` signifies any source).",
    )
    final: str = Field(..., description="The final value of the semantic field.")
    program: str = Field(..., description="The program offering the incentive.")
    eligibility: dict[str, list[str]] = Field(
        ..., description="Eligibility criteria for the incentive."
    )
    incentive_factors: list[FixedIncentive | VariableIncentive | PercentIncentive] = (
        Field(..., description="The incentive factors for the retrofit.")
    )

    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.Series:
        """Compute the incentive for a given feature with proper stacking order.

        Incentives are applied in the following order:
        1. Fixed incentives (applied first to original cost)
        2. Percentage incentives (applied to cost after fixed incentives)
        3. Variable incentives (applied to cost after fixed and percentage incentives)
        """
        if len(self.incentive_factors) == 0:
            print(
                f"No incentive factors found for {self.semantic_field} = {self.initial} -> {self.final}"
            )
            return pd.Series(0, index=features.index)

        # Initialize total incentive
        total_incentive = pd.Series(0, index=features.index)

        # Get the base cost for this semantic field (only needed for percentage/variable incentives)
        cost_col = f"cost.{self.semantic_field}"
        if cost_col in costs_df.columns:
            base_cost = costs_df[cost_col]
            net_cost = base_cost.copy()
        else:
            # If no cost column, we can only apply fixed incentives
            net_cost = None

        # Apply incentives in order: Fixed -> Percentage -> Variable
        incentive_types = {FixedIncentive, PercentIncentive, VariableIncentive}
        for incentive_class in incentive_types:
            type_incentives = [
                f for f in self.incentive_factors if isinstance(f, incentive_class)
            ]

            for factor in type_incentives:
                if isinstance(factor, FixedIncentive):
                    # Fixed incentives work independently of cost (should make sure there is proper error handling, in case a cost is not found)
                    incentive_amount = factor.compute(features, costs_df)
                    total_incentive += incentive_amount
                elif isinstance(factor, PercentIncentive):
                    # Percentage incentives need a cost to compute the % reduction
                    if net_cost is not None:
                        temp_costs_df = pd.DataFrame(
                            {cost_col: net_cost}, index=costs_df.index
                        )
                        incentive_amount = factor.compute(features, temp_costs_df)
                        # The incentive_amount is already the dollar amount (percentage * cost)
                        # So we subtract it from net_cost and add it to total_incentive
                        net_cost = net_cost - incentive_amount
                        total_incentive += incentive_amount
                elif isinstance(factor, VariableIncentive):
                    if net_cost is not None:
                        incentive_amount = factor.compute(features, costs_df)
                        net_cost = net_cost - incentive_amount
                        total_incentive += incentive_amount

        return total_incentive.rename(f"incentive.{self.semantic_field}")


class RetrofitIncentives(BaseModel):
    """The incentives associated with each of the retrofit interventions."""

    incentives: list[RetrofitIncentive]

    @property
    def all_semantic_features(self) -> list[str]:
        """The list of all features that are used in the incentives."""
        return list({incentive.semantic_field for incentive in self.incentives})

    def compute(self, features: pd.DataFrame, costs_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the incentives for a given feature."""
        if not self.incentives:
            return pd.DataFrame({"incentive.Total": [0] * len(features)})

        # Compute each incentive as a Series (columns may duplicate across programs)
        incentives_df = pd.concat(
            [incentive.compute(features, costs_df) for incentive in self.incentives],
            axis=1,
        )

        # Aggregate duplicate columns by summing per column name
        if incentives_df.columns.duplicated().any():
            aggregated = incentives_df.T.groupby(level=0).sum().T
        else:
            aggregated = incentives_df

        # Compute a single Total across all incentive.* columns
        total = aggregated.sum(axis=1).rename("incentive.Total")
        data = pd.concat([aggregated, total], axis=1)
        return data

    @classmethod
    def Open(cls, path: Path) -> "RetrofitIncentives":
        """Open a retrofit incentives file."""
        if path in RETROFIT_INCENTIVE_CACHE:
            return RETROFIT_INCENTIVE_CACHE[path]
        else:
            with open(path) as f:
                data = json.load(f)
            retrofit_incentives = RetrofitIncentives.model_validate(data)
            RETROFIT_INCENTIVE_CACHE[path] = retrofit_incentives
            return retrofit_incentives


RETROFIT_INCENTIVE_CACHE: dict[Path, RetrofitIncentives] = {}


class AppliedIncentive(BaseModel):
    """Details about a single applied incentive."""

    semantic_field: str = Field(
        ..., description="The semantic field this incentive applies to"
    )
    program: str = Field(..., description="The program offering the incentive")
    amount: float = Field(..., description="The incentive amount applied")
    description: str = Field(..., description="Description from incentives_format.json")
    source: str = Field(..., description="Source from incentives_format.json")
    incentive_type: str = Field(
        ..., description="Type of incentive (Fixed, Percent, Variable)"
    )


class IncentiveMetadata(BaseModel):
    """Metadata about applied incentives for a retrofit."""

    applied_incentives: list[AppliedIncentive] = Field(
        ..., description="List of all incentives that were applied"
    )
    total_incentive_amount: float = Field(
        ..., description="Total incentive amount across all programs"
    )
    income_level: str = Field(..., description="Income level these incentives apply to")

    @property
    def serialized(self) -> BaseModel:
        """Serialize the IncentiveMetadata into a BaseModel."""
        return self


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
