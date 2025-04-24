"""Inference request models."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

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

        for col in self.costs_summary.columns:
            col_name = col.split(".")[-1]
            field_specs[col_name] = (SummarySpec, Field(title=col))
            field_data = self.costs_summary.loc[:, col].to_dict()
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
        df_t = self.source_feature_transform.transform(df)
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
        return pd.concat(results, axis=1)

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
        return self.compute_distributions(features, results_raw)

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
    ) -> dict[str, SBEMDistributions | SBEMRetrofitDistributions]:
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
        new_transformed_features = self.original.source_feature_transform.transform(
            new_features
        )
        new_results_raw = self.original.predict(new_transformed_features)
        new_results = self.original.compute_distributions(new_features, new_results_raw)

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
        retrofit_costs = self.compute_retrofit_costs(new_results.features, cost_config)
        payback = self.compute_payback(retrofit_costs, delta_results)
        retrofit_costs_summary = retrofit_costs.describe(
            percentiles=list(PERCENTILES.keys())
        ).drop(["count"])
        payback_summary = payback.describe(percentiles=list(PERCENTILES.keys())).drop([
            "count"
        ])

        cost_results = SBEMRetrofitDistributions(
            costs=retrofit_costs,
            paybacks=payback,
            costs_summary=retrofit_costs_summary,
            paybacks_summary=payback_summary,
        )

        return {
            "original": original_results,
            "upgraded": new_results,
            "delta": delta_results,
            "retrofit": cost_results,
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


class Cost(ABC):
    """An abstract base class for all costs."""

    @abstractmethod
    def compute(self, features: pd.DataFrame) -> pd.Series:
        """Compute the cost for a given feature."""
        pass


class VariableCost(BaseModel, Cost):
    """A cost that is linear in the product of a set of indicator columns."""

    coefficient: float = Field(
        ..., description="The factor to multiply a target by.", gt=0
    )
    error_scale: float | None = Field(
        ...,
        description="The expected error of the cost estimate.",
        ge=0,
        le=1,
    )
    units: Literal["$/m2", "$/m", "$/m3", "$/kW", "$/unknown"]
    indicator_cols: list[str] = Field(
        ...,
        description="The column(s) in the source data that should be multiplied by the coefficient.",
    )
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
            costs = pd.concat([cost.compute(features) for cost in self.costs], axis=1)
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
