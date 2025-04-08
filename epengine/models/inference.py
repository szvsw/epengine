"""Inference request models."""

from dataclasses import dataclass
from functools import cached_property
from typing import cast

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import AnyUrl, BaseModel
from shapely import Point

from epengine.gis.data.epw_metadata import closest_epw
from epengine.models.sampling import (
    ClippedNormalSampler,
    ConditionalPrior,
    ConditionalPriorCondition,
    CopySampler,
    FixedValueSampler,
    Prior,
    Priors,
    ProductValuesSampler,
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
DATASETS = ("Raw", "EndUse", "Fuel", "Cost", "Emissions")
NORMALIZATIONS = ("Normalized", "Gross")

DATASET_SEGMENT_MAP = {
    "Raw": END_USES,
    "EndUse": END_USES,
    "Fuel": FUELS,
    "Cost": FUELS,
    "Emissions": FUELS,
}
UNITS_DENOM = {
    "Normalized": "/m2",
    "Gross": "",
}
UNITS_NUMER = {
    "Raw": "kWh",
    "EndUse": "kWh",
    "Fuel": "kWh",
    "Cost": "USD",
    "Emissions": "tCO2e",
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
    def serialized(self) -> "SBEMInferenceResponseSpec":
        """Serialize the SBEMDistributions dataframes into a SBEMInferenceResponseSpec."""
        disagg_summary_data_renamed = self.disaggregations_summary.rename(
            index={
                "5%": "p5",
                "25%": "p25",
                "50%": "p50",
                "75%": "p75",
                "95%": "p95",
            },
            columns={
                "Domestic Hot Water": "DomesticHotWater",
            },
        )
        totals_summary_renamed = self.totals_summary.rename(
            index={
                "5%": "p5",
                "25%": "p25",
                "50%": "p50",
                "75%": "p75",
                "95%": "p95",
            }
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


class SummarySpec(BaseModel):
    """Statistical summary of a dataset column."""

    min: float
    max: float
    mean: float
    std: float
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float
    units: str


class EndUseDisaggregationSpec(BaseModel):
    """Statistical summary of end use disaggregations."""

    Lighting: SummarySpec
    Equipment: SummarySpec
    DomesticHotWater: SummarySpec
    Heating: SummarySpec
    Cooling: SummarySpec


class FuelDisaggregationSpec(BaseModel):
    """Statistical summary of fuel disaggregations."""

    Oil: SummarySpec
    NaturalGas: SummarySpec
    Electricity: SummarySpec


class DisaggregationSpec(BaseModel):
    """Statistical summary of disaggregation datasets."""

    Raw: EndUseDisaggregationSpec
    EndUse: EndUseDisaggregationSpec
    Fuel: FuelDisaggregationSpec
    Cost: FuelDisaggregationSpec
    Emissions: FuelDisaggregationSpec


class DisaggregationsSpec(BaseModel):
    """Statistical summary of normalized and gross disaggregations."""

    Normalized: DisaggregationSpec
    Gross: DisaggregationSpec


class TotalSpec(BaseModel):
    """Statistical summary of total summed datasets."""

    Raw: SummarySpec
    EndUse: SummarySpec
    Fuel: SummarySpec
    Cost: SummarySpec
    Emissions: SummarySpec


class TotalsSpec(BaseModel):
    """Statistical summary of total summed datasets."""

    Normalized: TotalSpec
    Gross: TotalSpec


class SBEMInferenceResponseSpec(BaseModel):
    """Statistical summary of disaggregation and total datasets."""

    Disaggregation: DisaggregationsSpec
    Total: TotalsSpec


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

    @cached_property
    def source_feature_transform(self) -> RegressorInputSpec:
        """Load the source feature transforms from the space.yml file."""
        from pathlib import Path

        import yaml

        # TODO: construct a path to an s3 location?
        with open(
            Path(__file__).parent.parent / "workflows" / "artifacts" / "space.yml"
        ) as f:
            space = yaml.safe_load(f)
        return RegressorInputSpec.model_validate(space)

    @cached_property
    def lgb_models(self) -> dict[str, lgb.Booster]:
        """Load the lgb models from the s3 location."""
        from pathlib import Path

        lgb_models: dict[str, lgb.Booster] = {}
        for file in (
            Path(__file__).parent.parent / "workflows" / "artifacts" / "models"
        ).glob("*.lgb"):
            with open(file) as f:
                model = lgb.Booster(model_str=f.read())
            model_name = file.stem.replace("model_", "").replace("_", " ")
            lgb_models[model_name] = model
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

        # attic_slope_if_un_un_prior = UnconditionalPrior(
        #     sampler=UniformSampler(min=4/12, max=6/12)
        # )
        # prior_dict["feature.geometry.attic_slope_if_un_un"] = attic_slope_if_un_un_prior

        # attic_slope_if_oc_or_con_prior = UnconditionalPrior(
        #     sampler=UniformSampler(min=6/12, max=9/12)
        # )
        # prior_dict["feature.geometry.attic_slope_if_oc_or_con"] = attic_slope_if_oc_or_con_prior

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
                    sampler=UniformSampler(min=0.95, max=0.95),
                ),
                ConditionalPriorCondition(
                    match_val="OilHeating",
                    sampler=UniformSampler(min=0.8, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasHeating",
                    sampler=UniformSampler(min=0.85, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasCondensingHeating",
                    sampler=UniformSampler(min=0.9, max=0.95),
                ),
                ConditionalPriorCondition(
                    match_val="ASHPHeating",
                    sampler=UniformSampler(min=2, max=4),
                ),
                ConditionalPriorCondition(
                    match_val="GSHPHeating",
                    sampler=UniformSampler(min=3, max=5),
                ),
            ],
        )
        heat_distribution_prior = ConditionalPrior(
            source_feature="feature.semantic.Distribution",
            fallback_prior=None,
            conditions=[
                ConditionalPriorCondition(
                    match_val="Steam",
                    sampler=UniformSampler(min=0.8, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="HotWaterUninsulated",
                    sampler=UniformSampler(min=0.8, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="AirDuctsUninsulated",
                    sampler=UniformSampler(min=0.8, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="AirDuctsConditionedUninsulated",
                    sampler=UniformSampler(min=0.8, max=0.9),
                ),
                ConditionalPriorCondition(
                    match_val="HotWaterInsulated",
                    sampler=UniformSampler(min=0.8, max=0.9),
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
                    match_val="ACWindow",
                    sampler=UniformSampler(min=2, max=4),
                ),
                ConditionalPriorCondition(
                    match_val="ACCentral",
                    sampler=UniformSampler(min=3, max=5),
                ),
                ConditionalPriorCondition(
                    match_val="WindowASHP",
                    sampler=UniformSampler(min=2, max=4),
                ),
                # TODO: Optional: make this depend on the ASHP heating cop
                ConditionalPriorCondition(
                    match_val="ASHPCooling",
                    sampler=UniformSampler(min=2, max=4),
                ),
                ConditionalPriorCondition(
                    match_val="GSHPCooling",
                    sampler=UniformSampler(min=3, max=5),
                ),
            ],
        )

        cool_is_distributed_prior = ConditionalPrior(
            source_feature="feature.semantic.Cooling",
            fallback_prior=None,
            conditions=[
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
                    sampler=UniformSampler(min=0.9, max=0.95),
                ),
                ConditionalPriorCondition(
                    match_val="NaturalGasHeatingDHWCombo",
                    sampler=UniformSampler(min=0.9, max=0.95),
                ),
                ConditionalPriorCondition(
                    match_val="HPWH",
                    sampler=UniformSampler(min=3, max=5),
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
                mean=0.06, std=0.01, clip_min=0.03, clip_max=0.09
            )
        )
        oil_price_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.10, std=0.01, clip_min=0.07, clip_max=0.13
            )
        )
        electricity_price_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.25, std=0.025, clip_min=0.175, clip_max=0.325
            )
        )
        prior_dict["feature.fuels.price.NaturalGas"] = gas_price_prior
        prior_dict["feature.fuels.price.Electricity"] = electricity_price_prior
        prior_dict["feature.fuels.price.Oil"] = oil_price_prior

        electricity_emissions_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.0004, std=0.0001, clip_min=0.0001, clip_max=0.0007
            )
        )
        gas_emissions_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.005, std=0.0005, clip_min=0.0035, clip_max=0.0065
            )
        )
        oil_emissions_prior = UnconditionalPrior(
            sampler=ClippedNormalSampler(
                mean=0.005, std=0.0005, clip_min=0.0035, clip_max=0.0065
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

    def add_sampled_features(self, df: pd.DataFrame):
        """Add the sampled features to the dataframe."""
        attic_slope_if_unoccupied_unconditioned = np.random.uniform(
            4 / 12, 6 / 12, len(df)
        )
        attic_slope_if_occupied_or_conditioned = np.random.uniform(
            6 / 12, 9 / 12, len(df)
        )
        attic_slope = np.where(
            df["feature.extra_spaces.attic.exists"] == "Yes",
            np.where(
                (df["feature.extra_spaces.attic.occupied"] == "Yes")
                | (df["feature.extra_spaces.attic.conditioned"] == "Yes"),
                attic_slope_if_occupied_or_conditioned,
                attic_slope_if_unoccupied_unconditioned,
            ),
            0,
        )
        short_edge = df["feature.geometry.short_edge"]
        df["feature.geometry.attic_height"] = attic_slope * short_edge / 2

        return df

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
        df = priors.sample(df, n, self.generator)
        if (df["feature.extra_spaces.attic.exists"] == "Yes").any():
            msg = "At least one sample has an attic, which is not allowed for this inference request yet."
            # TODO: enable attics by setting up the priors for attic pitch -> attic height
            raise NotImplementedError(msg)
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
        self, df_features: pd.DataFrame, df_result: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply the COPs to the results.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_result (pd.DataFrame): The raw predicted results.

        Returns:
            df_end_uses (pd.DataFrame): The end uses.
        """
        heat_cop = df_features["feature.factors.system.heat.effective_cop"]
        cool_cop = df_features["feature.factors.system.cool.effective_cop"]
        dhw_cop = df_features["feature.factors.system.dhw.effective_cop"]
        df_end_uses = df_result.copy(deep=True)
        df_end_uses["Heating"] = df_result["Heating"].div(heat_cop, axis=0)
        df_end_uses["Cooling"] = df_result["Cooling"].div(cool_cop, axis=0)
        df_end_uses["Domestic Hot Water"] = df_result["Domestic Hot Water"].div(
            dhw_cop, axis=0
        )
        return df_end_uses

    def move_end_uses_to_fuels(
        self, df_features: pd.DataFrame, df_end_uses: pd.DataFrame
    ) -> pd.DataFrame:
        """Move the end uses to the fuels.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_end_uses (pd.DataFrame): The end uses.

        Returns:
            df_fuels (pd.DataFrame): The fuels.
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

        elec = (
            heat_elec
            + cool_elec
            + dhw_elec
            + df_end_uses["Lighting"]
            + df_end_uses["Equipment"]
        ).rename("Electricity")
        gas = (heat_gas + cool_gas + dhw_gas).rename("NaturalGas")
        oil = (heat_oil + cool_oil + dhw_oil).rename("Oil")

        results = pd.concat([elec, gas, oil], axis=1)
        return results

    def compute_costs(
        self, df_features: pd.DataFrame, df_fuels: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the costs.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_fuels (pd.DataFrame): The fuels.

        Returns:
            df_costs (pd.DataFrame): The costs.
        """
        gas_rate = df_features["feature.fuels.price.NaturalGas"]
        elec_rate = df_features["feature.fuels.price.Electricity"]
        oil_rate = df_features["feature.fuels.price.Oil"]
        gas_cost = cast(pd.Series, df_fuels["NaturalGas"].mul(gas_rate, axis=0)).rename(
            "NaturalGas"
        )
        elec_cost = cast(
            pd.Series, df_fuels["Electricity"].mul(elec_rate, axis=0)
        ).rename("Electricity")
        oil_cost = cast(pd.Series, df_fuels["Oil"].mul(oil_rate, axis=0)).rename("Oil")
        costs = pd.concat([gas_cost, elec_cost, oil_cost], axis=1)
        return costs

    def compute_emissions(
        self, df_features: pd.DataFrame, df_fuels: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the emissions.

        Args:
            df_features (pd.DataFrame): The features to use in prediction.
            df_fuels (pd.DataFrame): The fuels.

        Returns:
            df_emissions (pd.DataFrame): The emissions.
        """
        gas_emissions_factors = df_features["feature.fuels.emissions.NaturalGas"]
        elec_emissions_factors = df_features["feature.fuels.emissions.Electricity"]
        oil_emissions_factors = df_features["feature.fuels.emissions.Oil"]
        gas_emissions = cast(
            pd.Series, df_fuels["NaturalGas"].mul(gas_emissions_factors, axis=0)
        ).rename("NaturalGas")
        elec_emissions = cast(
            pd.Series, df_fuels["Electricity"].mul(elec_emissions_factors, axis=0)
        ).rename("Electricity")
        oil_emissions = cast(
            pd.Series, df_fuels["Oil"].mul(oil_emissions_factors, axis=0)
        ).rename("Oil")
        emissions = pd.concat([gas_emissions, elec_emissions, oil_emissions], axis=1)
        return emissions

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
        results_end_uses = self.apply_cops(features, results_raw)
        results_fuels = self.move_end_uses_to_fuels(features, results_end_uses)
        results_costs = self.compute_costs(features, results_fuels)
        results_emissions = self.compute_emissions(features, results_fuels)
        disaggregated = pd.concat(
            [
                results_raw,
                results_end_uses,
                results_fuels,
                results_costs,
                results_emissions,
            ],
            axis=1,
            keys=["Raw", "EndUse", "Fuel", "Cost", "Emissions"],
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
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).drop(["count"])
        totals_summary = totals.describe(
            percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).drop(["count"])

        return SBEMDistributions(
            features=features,
            disaggregations=disaggregations,
            totals=totals,
            disaggregations_summary=disaggregations_summary,
            totals_summary=totals_summary,
        )


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
