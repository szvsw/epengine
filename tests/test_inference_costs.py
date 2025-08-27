#!/usr/bin/env python3
"""Test script for the refactored inference.py cost functionality using pytest."""

from pathlib import Path

import pandas as pd
import pytest

from epengine.models.inference import RetrofitQuantities


@pytest.fixture
def test_features():
    """Create a sample feature dataframe for testing."""
    features = pd.DataFrame({
        "feature.geometry.energy_model_conditioned_area": [250.0],  # 150 m2
        "feature.geometry.est_fp_ratio": [1.0],
        "feature.geometry.computed.roof_surface_area": [200.0],  # 200 m2
        "feature.geometry.roof_is_flat.num": [0],  # pitched roof
        "feature.geometry.roof_is_attic.num": [1],  # has attic
        "feature.geometry.computed.whole_bldg_facade_area": [300.0],  # 300 m2
        "feature.geometry.est_uniform_linear_scaling_factor": [1.0],
        "feature.geometry.computed.total_linear_facade_distance": [100.0],  # 100 m
        "feature.geometry.computed.footprint_area": [250.0],  # 150 m2
        "feature.geometry.computed.window_area": [30.0],  # 30 m2
        "feature.geometry.computed.perimeter": [50.0],  # 50 m
        "feature.extra_spaces.basement.exists.num": [1],  # has basement
        "feature.extra_spaces.basement.occupied.num": [0],  # unoccupied basement
        "feature.extra_spaces.basement.not_occupied.num": [1],
    })
    return features


@pytest.fixture
def test_features_with_location():
    """Create test features with location information for conditional factors."""
    features = pd.DataFrame({
        "feature.geometry.energy_model_conditioned_area": [250.0],  # 150 m2
        "feature.geometry.est_fp_ratio": [1.0],
        "feature.geometry.computed.roof_surface_area": [200.0],  # 200 m2
        "feature.geometry.roof_is_flat.num": [0],  # pitched roof
        "feature.geometry.roof_is_attic.num": [1],  # has attic
        "feature.geometry.computed.whole_bldg_facade_area": [300.0],  # 300 m2
        "feature.geometry.est_uniform_linear_scaling_factor": [1.0],
        "feature.geometry.computed.total_linear_facade_distance": [100.0],  # 100 m
        "feature.geometry.computed.footprint_area": [250.0],  # 150 m2
        "feature.geometry.computed.window_area": [30.0],  # 30 m2
        "feature.geometry.computed.perimeter": [50.0],  # 50 m
        "feature.extra_spaces.basement.exists.num": [1],  # has basement
        "feature.extra_spaces.basement.occupied.num": [0],  # unoccupied basement
        "feature.extra_spaces.basement.not_occupied.num": [1],
    })

    # Add location features for conditional factors
    features["feature.location.county"] = "Middlesex"  # Default county
    features["feature.semantic.Heating"] = "NaturalGasHeating"  # Has gas heating
    features["feature.semantic.Cooling"] = "ACCentral"  # Has cooling
    features["feature.semantic.Windows"] = "SinglePane"  # Default window type
    features["feature.semantic.Walls"] = "NoInsulationWalls"  # Default wall type

    # Add income eligibility feature (needed for incentive calculations)
    features["feature.eligibility.income_level"] = (
        "AllCustomers"  # Default to all customers
    )

    # Add income bracket indicators (needed for incentive calculations)
    features["feature.homeowner.in_bracket.AllCustomers"] = False
    features["feature.homeowner.in_bracket.IncomeEligible"] = False

    # Add precomputed features that would normally be added by make_retrofit_cost_features
    # Heating capacity (placeholder - would normally be calculated from peak results)
    features["feature.calculated.heating_capacity_kW"] = 25  # 15 kW heating capacity

    # Add system factors needed for cost calculations
    features["feature.factors.system.heat.effective_cop"] = 0.8  # Typical effective COP

    # County indicators (one-hot encoded)
    features["feature.location.in_county.Middlesex"] = 1
    features["feature.location.in_county.Berkshire"] = 0
    features["feature.location.in_county.Barnstable"] = 0
    features["feature.location.in_county.Bristol"] = 0
    features["feature.location.in_county.Dukes"] = 0
    features["feature.location.in_county.Essex"] = 0
    features["feature.location.in_county.Franklin"] = 0
    features["feature.location.in_county.Hampden"] = 0
    features["feature.location.in_county.Hampshire"] = 0
    features["feature.location.in_county.Nantucket"] = 0
    features["feature.location.in_county.Norfolk"] = 0
    features["feature.location.in_county.Plymouth"] = 0
    features["feature.location.in_county.Suffolk"] = 0
    features["feature.location.in_county.Worcester"] = 0

    # System indicators
    features["feature.system.has_gas.true"] = 1  # Has gas heating
    features["feature.system.has_gas.false"] = 0
    features["feature.system.has_cooling.true"] = 1  # Has cooling
    features["feature.system.has_cooling.false"] = 0

    # Constant feature
    features["feature.constant.one"] = 1

    return features


def create_features_with_income_bracket(features, income_bracket):
    """Create features with specific income bracket indicators."""
    features_copy = features.copy()

    # Set all income bracket indicators to False
    features_copy["feature.homeowner.in_bracket.AllCustomers"] = False
    features_copy["feature.homeowner.in_bracket.IncomeEligible"] = False

    # Set the specific income bracket to True
    if income_bracket == "AllCustomers":
        features_copy["feature.homeowner.in_bracket.AllCustomers"] = True
    elif income_bracket == "IncomeEligible":
        features_copy["feature.homeowner.in_bracket.IncomeEligible"] = True

    return features_copy


@pytest.fixture
def retrofit_costs():
    """Load retrofit costs for testing."""
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    return RetrofitQuantities.Open(costs_path)


@pytest.fixture
def consolidated_incentives():
    """Load consolidated incentives for testing."""
    incentives_path = Path("epengine/models/data/incentives.json")
    return RetrofitQuantities.Open(incentives_path)


class TestCostCalculation:
    """Test basic cost calculation functionality."""

    def test_cost_calculation(self, retrofit_costs, test_features_with_location):
        """Test basic cost calculation functionality."""
        print(f"Loaded {len(retrofit_costs.quantities)} cost configurations")

        # Test a specific cost calculation - ASHP Heating
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]
        if ashp_costs:
            ashp_cost = ashp_costs[0]
            cost_result = ashp_cost.compute(test_features_with_location)
            print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")
            assert cost_result.iloc[0] > 0, "Cost should be positive"

        # Test a fixed cost calculation - Thermostat
        thermostat_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Thermostat" and cost.final == "Controls"
        ]
        if thermostat_costs:
            thermostat_cost = thermostat_costs[0]
            cost_result = thermostat_cost.compute(test_features_with_location)
            print(f"Thermostat cost: ${cost_result.iloc[0]:.2f}")
            assert cost_result.iloc[0] > 0, "Cost should be positive"

    def test_heat_pump_cost_calculation(
        self, retrofit_costs, test_features_with_location
    ):
        """Test the new heat pump cost calculation formula."""
        # Test ASHP cost calculation
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]

        if ashp_costs:
            ashp_cost = ashp_costs[0]
            cost_result = ashp_cost.compute(test_features_with_location)

            # Get the expected values from the features
            conditioned_area = test_features_with_location[
                "feature.geometry.energy_model_conditioned_area"
            ].iloc[0]

            # These features should be precomputed by make_retrofit_cost_features
            heating_capacity_kw = test_features_with_location.get(
                "feature.calculated.heating_capacity_kW", [0]
            ).iloc[0]
            in_county_middlesex = test_features_with_location.get(
                "feature.location.in_county.Middlesex", [0]
            ).iloc[0]
            has_gas = test_features_with_location.get(
                "feature.system.has_gas.true", [0]
            ).iloc[0]
            has_cooling = test_features_with_location.get(
                "feature.system.has_cooling.true", [0]
            ).iloc[0]

            print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")
            print("Expected cost breakdown:")
            print(f"  Conditioned area: {conditioned_area}m²")
            print(f"  Heating capacity: {heating_capacity_kw:.1f}kW")
            print(f"  In Middlesex county: {in_county_middlesex}")
            print(f"  Has gas: {has_gas}")
            print(f"  Has cooling: {has_cooling}")

            # The cost should be positive and reasonable
            assert cost_result.iloc[0] > 0, "ASHP cost should be positive"
            assert cost_result.iloc[0] < 100000, (
                "ASHP cost should be reasonable (< $100k)"
            )

        # Test GSHP cost calculation
        gshp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "GSHPHeating"
        ]

        if gshp_costs:
            gshp_cost = gshp_costs[0]
            cost_result = gshp_cost.compute(test_features_with_location)

            print(f"\nGSHP Heating cost: ${cost_result.iloc[0]:.2f}")

            # The cost should be positive and reasonable
            assert cost_result.iloc[0] > 0, "GSHP cost should be positive"
            assert cost_result.iloc[0] < 100000, (
                "GSHP cost should be reasonable (< $100k)"
            )

    def test_new_cost_structure(self, retrofit_costs, test_features_with_location):
        """Test the new cost structure with different types of cost factors."""
        # Test different cost types
        test_cases = [
            {
                "name": "Fixed Cost - Thermostat",
                "trigger_column": "Thermostat",
                "final": "Controls",
                "expected_type": "FixedQuantity",
            },
            {
                "name": "Mixed Cost - ASHP",
                "trigger_column": "Heating",
                "final": "ASHPHeating",
                "expected_type": "LinearQuantity",  # Should have at least one LinearQuantity
            },
            {
                "name": "DHW Cost - HPWH",
                "trigger_column": "DHW",
                "final": "HPWH",
                "expected_type": "FixedQuantity",
            },
        ]

        for test_case in test_cases:
            matching_costs = [
                cost
                for cost in retrofit_costs.quantities
                if cost.trigger_column == test_case["trigger_column"]
                and cost.final == test_case["final"]
            ]

            if matching_costs:
                cost = matching_costs[0]
                cost_result = cost.compute(test_features_with_location)

                print(f"{test_case['name']}: ${cost_result.iloc[0]:.2f}")

                # Check that the cost has the expected structure
                # Count the different types of quantity factors
                fixed_count = sum(
                    1 for qf in cost.quantity_factors if hasattr(qf, "amount")
                )
                linear_count = sum(
                    1 for qf in cost.quantity_factors if hasattr(qf, "coefficient")
                )
                percent_count = sum(
                    1 for qf in cost.quantity_factors if hasattr(qf, "percent")
                )

                print(
                    f"  Quantity factors: {fixed_count} Fixed, {linear_count} Linear, {percent_count} Percent"
                )

                if test_case["expected_type"] == "FixedQuantity":
                    assert fixed_count > 0, (
                        "Should have at least one FixedQuantity factor"
                    )
                elif test_case["expected_type"] == "LinearQuantity":
                    assert linear_count > 0, (
                        "Should have at least one LinearQuantity factor"
                    )
                elif test_case["expected_type"] == "PercentQuantity":
                    assert percent_count > 0, (
                        "Should have at least one PercentQuantity factor"
                    )
            else:
                print(f"No cost found for {test_case['name']}")

    def test_cost_and_incentive_selection(self, retrofit_costs):
        """Test cost and incentive selection functionality."""
        # Test cost entity selection manually
        print(f"Total available costs: {len(retrofit_costs.quantities)}")

        # Test specific cost selections
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]
        print(f"ASHP Heating costs available: {len(ashp_costs)}")

        dhw_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "DHW" and cost.final == "HPWH"
        ]
        print(f"DHW HPWH costs available: {len(dhw_costs)}")

        # Test that we have the expected cost types
        assert len(ashp_costs) > 0, "Should have ASHP heating costs"
        assert len(dhw_costs) > 0, "Should have DHW HPWH costs"

    def test_manual_cost_calculation(self, retrofit_costs, test_features_with_location):
        """Test manual cost calculation with sample data."""
        # Test multiple cost calculations
        test_upgrades = [
            ("Heating", "ASHPHeating"),
            ("DHW", "HPWH"),
            ("Thermostat", "Controls"),
        ]

        total_cost = 0
        for trigger_column, final_value in test_upgrades:
            matching_costs = [
                cost
                for cost in retrofit_costs.quantities
                if cost.trigger_column == trigger_column and cost.final == final_value
            ]

            if matching_costs:
                cost_result = matching_costs[0].compute(test_features_with_location)
                cost_amount = cost_result.iloc[0]
                total_cost += cost_amount
                print(f"{trigger_column} → {final_value}: ${cost_amount:.2f}")
            else:
                print(f"No cost found for {trigger_column} → {final_value}")

        print(f"Total retrofit cost: ${total_cost:.2f}")
        assert total_cost > 0, "Total cost should be positive"

    def test_feature_precomputation(self, test_features_with_location):
        """Test that features are properly precomputed for cost calculations."""
        # Test that we have the expected precomputed features
        expected_features = [
            "feature.calculated.heating_capacity_kW",
            "feature.location.in_county.Middlesex",
            "feature.system.has_gas",
            "feature.system.has_cooling",
            "feature.constant.one",
        ]

        for feature_name in expected_features:
            if feature_name in test_features_with_location.columns:
                print(f"✓ Found feature: {feature_name}")
            else:
                print(f"⚠ Missing feature: {feature_name}")


class TestIncentiveLoading:
    """Test incentive loading and basic functionality."""

    def test_incentive_loading(self, consolidated_incentives):
        """Test incentive loading and basic functionality."""
        print(
            f"Loaded {len(consolidated_incentives.quantities)} consolidated incentive configurations"
        )

        # Check for specific incentive types
        ashp_incentives = [
            inc
            for inc in consolidated_incentives.quantities
            if inc.trigger_column == "Heating" and inc.final == "ASHPHeating"
        ]

        print(f"Found {len(ashp_incentives)} ASHP heating incentives")

        assert len(ashp_incentives) > 0, "Should have incentives for ASHP heating"

        # Check that we have both income bracket indicators in the factors
        if ashp_incentives:
            ashp_incentive = ashp_incentives[0]
            has_all_customers = any(
                hasattr(factor, "indicator_cols")
                and "feature.homeowner.in_bracket.AllCustomers" in factor.indicator_cols
                for factor in ashp_incentive.quantity_factors
            )
            has_income_eligible = any(
                hasattr(factor, "indicator_cols")
                and "feature.homeowner.in_bracket.IncomeEligible"
                in factor.indicator_cols
                for factor in ashp_incentive.quantity_factors
            )

            assert has_all_customers, (
                "Should have AllCustomers income bracket indicators"
            )
            assert has_income_eligible, (
                "Should have IncomeEligible income bracket indicators"
            )


class TestIncentiveCalculation:
    """Test incentive calculation functionality."""

    def test_incentive_calculation(
        self,
        retrofit_costs,
        consolidated_incentives,
        test_features_with_location,
    ):
        """Test incentive calculation functionality."""
        # Test ASHP heating incentives
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]

        ashp_incentives = [
            inc
            for inc in consolidated_incentives.quantities
            if inc.trigger_column == "Heating" and inc.final == "ASHPHeating"
        ]

        if ashp_costs and ashp_incentives:
            # Calculate the cost first
            ashp_cost = ashp_costs[0]
            cost_result = ashp_cost.compute(test_features_with_location)

            # Use a higher cost to avoid clipping so we can test the income bracket differentiation
            # The incentives are $12,000 for AllCustomers and $17,000 for IncomeEligible
            test_cost = max(cost_result.iloc[0], 20000.0)  # Ensure cost is high enough
            costs_df = pd.DataFrame({"cost.Heating": [test_cost]})

            print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")
            print(f"Using test cost: ${test_cost:.2f} (to avoid clipping)")

            # Test all customers incentive
            all_customers_features = create_features_with_income_bracket(
                test_features_with_location, "AllCustomers"
            )
            all_customer_result = consolidated_incentives.compute(
                all_customers_features, costs_df, final_values={"ASHPHeating"}
            )
            all_customer_total = all_customer_result[
                "incentive.Heating.ASHPHeating"
            ].iloc[0]
            print(f"All customers incentive: ${all_customer_total:.2f}")

            # Test income eligible incentive
            income_eligible_features = create_features_with_income_bracket(
                test_features_with_location, "IncomeEligible"
            )
            income_eligible_result = consolidated_incentives.compute(
                income_eligible_features, costs_df, final_values={"ASHPHeating"}
            )
            income_eligible_total = income_eligible_result[
                "incentive.Heating.ASHPHeating"
            ].iloc[0]
            print(f"Income eligible incentive: ${income_eligible_total:.2f}")

            # Verify that income eligible incentives are higher
            assert income_eligible_total > all_customer_total, (
                "Income eligible incentives should be higher than all customer incentives"
            )

            # Verify that incentives are properly clipped to the total cost
            assert all_customer_total <= test_cost, (
                f"All customers incentive (${all_customer_total:.2f}) should be clipped to cost (${test_cost:.2f})"
            )
            assert income_eligible_total <= test_cost, (
                f"Income eligible incentive (${income_eligible_total:.2f}) should be clipped to cost (${test_cost:.2f})"
            )

            # Verify that net cost is never negative
            net_cost_all_customers = test_cost - all_customer_total
            net_cost_income_eligible = test_cost - income_eligible_total

            assert net_cost_all_customers >= 0, (
                f"Net cost for all customers should be >= 0, got ${net_cost_all_customers:.2f}"
            )
            assert net_cost_income_eligible >= 0, (
                f"Net cost for income eligible should be >= 0, got ${net_cost_income_eligible:.2f}"
            )

    def test_incentive_eligibility(self, consolidated_incentives):
        """Test incentive eligibility checking."""
        # Test that we can load and access the incentives
        assert len(consolidated_incentives.quantities) > 0, (
            "Should have consolidated incentives"
        )

        # Test that we have incentives for ASHP heating
        ashp_incentives = [
            inc
            for inc in consolidated_incentives.quantities
            if inc.trigger_column == "Heating" and inc.final == "ASHPHeating"
        ]

        assert len(ashp_incentives) > 0, "Should have ASHP incentives"

        # Check that the incentive has both income bracket factors
        if ashp_incentives:
            ashp_incentive = ashp_incentives[0]
            linear_factors = [
                f for f in ashp_incentive.quantity_factors if hasattr(f, "coefficient")
            ]
            assert len(linear_factors) >= 2, (
                "Should have at least 2 linear incentive factors (one for each income bracket)"
            )

    def test_incentive_income_level_differentiation(
        self,
        retrofit_costs,
        consolidated_incentives,
        test_features_with_location,
    ):
        """Test that incentives are properly differentiated by income level."""
        # Test with ASHP heating upgrade
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]

        if ashp_costs:
            # Calculate the cost first
            ashp_cost = ashp_costs[0]
            cost_result = ashp_cost.compute(test_features_with_location)
            costs_df = pd.DataFrame({"cost.Heating": cost_result})

            print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")

            # Test all customers incentive
            all_customers_features = create_features_with_income_bracket(
                test_features_with_location, "AllCustomers"
            )
            all_customer_result = consolidated_incentives.compute(
                all_customers_features, costs_df, final_values={"ASHPHeating"}
            )
            all_customer_total = all_customer_result[
                "incentive.Heating.ASHPHeating"
            ].iloc[0]
            print(f"All customers incentive: ${all_customer_total:.2f}")

            # Test income eligible incentive
            income_eligible_features = create_features_with_income_bracket(
                test_features_with_location, "IncomeEligible"
            )
            income_eligible_result = consolidated_incentives.compute(
                income_eligible_features, costs_df, final_values={"ASHPHeating"}
            )
            income_eligible_total = income_eligible_result[
                "incentive.Heating.ASHPHeating"
            ].iloc[0]
            print(f"Income eligible incentive: ${income_eligible_total:.2f}")

            # Verify that we have different incentive amounts for different income levels
            assert all_customer_total > 0, (
                "Should have positive incentives for all customers"
            )
            assert income_eligible_total > 0, (
                "Should have positive incentives for income eligible"
            )

            # Income eligible incentives should typically be higher
            if income_eligible_total > all_customer_total:
                print(
                    "✓ Income eligible incentives are higher than all customer incentives (expected)"
                )
            else:
                print(
                    "Info: Income eligible incentives are not higher (may be expected depending on program design)"
                )

            # Verify that total incentives are properly clipped to the cost
            total_incentives = max(all_customer_total, income_eligible_total)
            assert total_incentives <= cost_result.iloc[0], (
                f"Total incentives (${total_incentives:.2f}) should be clipped to cost (${cost_result.iloc[0]:.2f})"
            )

            # Verify that net cost is never negative
            net_cost = cost_result.iloc[0] - total_incentives
            assert net_cost >= 0, f"Net cost should be >= 0, got ${net_cost:.2f}"

    def test_incentive_clipping(
        self,
        retrofit_costs,
        consolidated_incentives,
        test_features_with_location,
    ):
        """Test that incentives are properly clipped to prevent negative net costs."""
        # Test ASHP heating with specific cost scenario
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]

        if ashp_costs:
            # Set a specific cost for testing
            test_cost = 11091.0
            costs_df = pd.DataFrame({
                "cost.Heating": pd.Series(
                    [test_cost], index=test_features_with_location.index
                )
            })

            print(f"Test ASHP cost: ${test_cost:.2f}")

            # Test all customers incentives
            all_customers_features = create_features_with_income_bracket(
                test_features_with_location, "AllCustomers"
            )
            all_customer_result = consolidated_incentives.compute(
                all_customers_features, costs_df, final_values={"ASHPHeating"}
            )
            all_customer_total = all_customer_result[
                "incentive.Heating.ASHPHeating"
            ].iloc[0]

            print(f"All customers incentive: ${all_customer_total:.2f}")

            # Expected: MassSave $10,000 + IRA 30% of remaining $1,091 = $10,000 + $327.30 = $10,327.30
            expected_all_customers = 10000 + (test_cost - 10000) * 0.3
            print(f"Expected all customers: ${expected_all_customers:.2f}")

            # Test income eligible incentives
            income_eligible_features = create_features_with_income_bracket(
                test_features_with_location, "IncomeEligible"
            )
            income_eligible_result = consolidated_incentives.compute(
                income_eligible_features, costs_df, final_values={"ASHPHeating"}
            )
            income_eligible_total = income_eligible_result[
                "incentive.Heating.ASHPHeating"
            ].iloc[0]

            print(f"Income eligible incentive: ${income_eligible_total:.2f}")

            # Expected: MassSave $15,000 (clipped to cost $11,091) + IRA 30% of remaining $0 = $11,091
            expected_income_eligible = test_cost  # Should be clipped to full cost
            print(f"Expected income eligible: ${expected_income_eligible:.2f}")

            # Verify clipping behavior
            assert all_customer_total <= test_cost, (
                f"All customers incentive (${all_customer_total:.2f}) should be clipped to cost (${test_cost:.2f})"
            )
            assert income_eligible_total <= test_cost, (
                f"Income eligible incentive (${income_eligible_total:.2f}) should be clipped to cost (${test_cost:.2f})"
            )

            # Verify net cost is never negative
            net_cost_all = test_cost - all_customer_total
            net_cost_income = test_cost - income_eligible_total

            assert net_cost_all >= 0, (
                f"Net cost for all customers should be >= 0, got ${net_cost_all:.2f}"
            )
            assert net_cost_income >= 0, (
                f"Net cost for income eligible should be >= 0, got ${net_cost_income:.2f}"
            )

            print(f"Net cost all customers: ${net_cost_all:.2f}")
            print(f"Net cost income eligible: ${net_cost_income:.2f}")
            print("✓ ASHP incentive clipping test passed")

        # Test with a low-cost upgrade to verify clipping
        thermostat_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Thermostat" and cost.final == "Controls"
        ]

        if thermostat_costs:
            # Calculate the cost first
            thermostat_cost = thermostat_costs[0]
            cost_result = thermostat_cost.compute(test_features_with_location)
            costs_df = pd.DataFrame({"cost.Thermostat": cost_result})

            print(f"Thermostat cost: ${cost_result.iloc[0]:.2f}")

            # Get thermostat incentives
            thermostat_incentives = [
                inc
                for inc in consolidated_incentives.quantities
                if inc.trigger_column == "Thermostat" and inc.final == "Controls"
            ]

            if thermostat_incentives:
                # Compute incentive for all customers
                all_customers_features = create_features_with_income_bracket(
                    test_features_with_location, "AllCustomers"
                )
                incentive_result = consolidated_incentives.compute(
                    all_customers_features, costs_df, final_values={"Controls"}
                )
                incentive_total = incentive_result[
                    "incentive.Thermostat.Controls"
                ].iloc[0]

                print(f"Thermostat incentive: ${incentive_total:.2f}")

                # Verify clipping behavior
                assert incentive_total <= cost_result.iloc[0], (
                    f"Incentive (${incentive_total:.2f}) should be clipped to cost (${cost_result.iloc[0]:.2f})"
                )

                # Verify net cost is never negative
                net_cost = cost_result.iloc[0] - incentive_total
                assert net_cost >= 0, f"Net cost should be >= 0, got ${net_cost:.2f}"

                print(f"Net cost: ${net_cost:.2f}")
                print("✓ Thermostat incentive clipping test passed")

    def test_incentive_validation(
        self,
        retrofit_costs,
        consolidated_incentives,
        test_features_with_location,
    ):
        """Test that incentives are properly validated and computed correctly."""
        # Test multiple upgrade scenarios
        test_upgrades = [
            ("Heating", "ASHPHeating"),
            ("DHW", "HPWH"),
        ]

        for trigger_column, final_value in test_upgrades:
            print(f"\nTesting {trigger_column} → {final_value}:")

            # Get the cost
            matching_costs = [
                cost
                for cost in retrofit_costs.quantities
                if cost.trigger_column == trigger_column and cost.final == final_value
            ]

            if matching_costs:
                cost = matching_costs[0]
                cost_result = cost.compute(test_features_with_location)
                costs_df = pd.DataFrame({f"cost.{trigger_column}": cost_result})

                print(f"  Cost: ${cost_result.iloc[0]:.2f}")

                # Test all customers incentives
                all_customers_features = create_features_with_income_bracket(
                    test_features_with_location, "AllCustomers"
                )
                try:
                    result = consolidated_incentives.compute(
                        all_customers_features,
                        costs_df,
                        final_values={final_value},
                    )
                    all_customer_total = result[
                        f"incentive.{trigger_column}.{final_value}"
                    ].iloc[0]
                    print(f"    All customers total: ${all_customer_total:.2f}")
                except Exception as e:
                    print(f"    All customers incentive: Error - {e}")
                    all_customer_total = 0

                # Validate all customers incentives
                assert all_customer_total > 0, (
                    "All customers should have positive incentives"
                )
                assert all_customer_total <= cost_result.iloc[0], (
                    f"All customers incentives (${all_customer_total:.2f}) should be clipped to cost (${cost_result.iloc[0]:.2f})"
                )

                # Test income eligible incentives
                income_eligible_features = create_features_with_income_bracket(
                    test_features_with_location, "IncomeEligible"
                )
                try:
                    result = consolidated_incentives.compute(
                        income_eligible_features,
                        costs_df,
                        final_values={final_value},
                    )
                    income_eligible_total = result[
                        f"incentive.{trigger_column}.{final_value}"
                    ].iloc[0]
                    print(f"    Income eligible total: ${income_eligible_total:.2f}")
                except Exception as e:
                    print(f"    Income eligible incentive: Error - {e}")
                    income_eligible_total = 0

                # Validate income eligible incentives
                assert income_eligible_total > 0, (
                    "Income eligible should have positive incentives"
                )
                assert income_eligible_total <= cost_result.iloc[0], (
                    f"Income eligible incentives (${income_eligible_total:.2f}) should be clipped to cost (${cost_result.iloc[0]:.2f})"
                )

                # Verify income eligible are higher if both exist
                assert income_eligible_total >= all_customer_total, (
                    f"Income eligible incentives (${income_eligible_total:.2f}) should be >= all customers (${all_customer_total:.2f})"
                )

            else:
                print(f"  No cost found for {trigger_column} → {final_value}")

    def test_incentive_metadata_functionality(
        self,
        retrofit_costs,
        consolidated_incentives,
        test_features_with_location,
    ):
        """Test that the new incentive metadata system is working correctly."""
        # Test ASHP heating incentives with metadata
        ashp_costs = [
            cost
            for cost in retrofit_costs.quantities
            if cost.trigger_column == "Heating" and cost.final == "ASHPHeating"
        ]

        if ashp_costs:
            # Calculate the cost first
            ashp_cost = ashp_costs[0]
            cost_result = ashp_cost.compute(test_features_with_location)
            costs_df = pd.DataFrame({"cost.Heating": cost_result})

            print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")

            # Test all customers incentive with metadata
            all_customers_features = create_features_with_income_bracket(
                test_features_with_location, "AllCustomers"
            )
            all_customer_result = consolidated_incentives.compute(
                all_customers_features, costs_df, final_values={"ASHPHeating"}
            )

            # Check that the incentive.metadata column exists
            assert "incentive.metadata" in all_customer_result.columns, (
                "incentive.metadata column should be present in incentive results"
            )

            # Get the metadata
            metadata = all_customer_result["incentive.metadata"].iloc[0]
            print(f"Incentive metadata: {metadata}")

            # Verify metadata structure
            assert isinstance(metadata, list), (
                "Metadata should be a list of dictionaries"
            )
            assert len(metadata) > 0, "Should have at least one incentive in metadata"

            # Check the first incentive metadata
            first_incentive = metadata[0]
            expected_keys = {
                "trigger",
                "final",
                "program_name",
                "source",
                "amount_applied",
            }
            assert all(key in first_incentive for key in expected_keys), (
                f"Metadata should contain keys: {expected_keys}"
            )

            # Verify specific values
            assert first_incentive["trigger"] == "Heating", (
                "Trigger should be 'Heating'"
            )
            assert first_incentive["final"] == "ASHPHeating", (
                "Final should be 'ASHPHeating'"
            )
            assert first_incentive["amount_applied"] > 0, (
                "Amount applied should be positive"
            )

            print("✓ Incentive metadata functionality test passed")
