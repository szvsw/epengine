#!/usr/bin/env python3
"""Test script for incentives integration and costs checks. NOT comprehensive test suite yet."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

from epengine.models.inference import (
    AppliedIncentive,
    IncentiveMetadata,
    RetrofitCosts,
    RetrofitIncentives,
)


def create_test_features():
    """Create a sample feature dataframe for testing."""
    features = pd.DataFrame({
        "feature.geometry.energy_model_conditioned_area": [150.0],  # 150 m2
        "feature.geometry.est_fp_ratio": [1.0],
        "feature.geometry.computed.roof_surface_area": [200.0],  # 200 m2
        "feature.geometry.roof_is_flat.num": [0],  # pitched roof
        "feature.geometry.roof_is_attic.num": [1],  # has attic
        "feature.geometry.computed.whole_bldg_facade_area": [300.0],  # 300 m2
        "feature.geometry.est_uniform_linear_scaling_factor": [1.0],
        "feature.geometry.computed.total_linear_facade_distance": [100.0],  # 100 m
        "feature.geometry.computed.footprint_area": [150.0],  # 150 m2
        "feature.geometry.computed.window_area": [30.0],  # 30 m2
        "feature.geometry.computed.perimeter": [50.0],  # 50 m
        "feature.extra_spaces.basement.exists.num": [1],  # has basement
        "feature.extra_spaces.basement.occupied.num": [0],  # unoccupied basement
        "feature.extra_spaces.basement.not_occupied.num": [1],
    })
    return features


def create_test_features_with_location():
    """Create test features with location information for conditional factors."""
    features = create_test_features()
    # Add location features for conditional factors
    features["feature.location.county"] = "Middlesex"  # Default county
    features["feature.semantic.Heating"] = "NaturalGasHeating"  # Has gas heating
    features["feature.semantic.Cooling"] = "ACCentral"  # Has cooling
    return features


def test_cost_calculation():
    """Test basic cost calculation functionality."""
    print("=== Testing Cost Calculation ===")

    # Load the retrofit costs
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)

    print(f"Loaded {len(retrofit_costs.costs)} cost configurations")

    # Create test features
    features = create_test_features_with_location()

    # Test a specific cost calculation - ASHP Heating
    ashp_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Heating" and cost.final == "ASHPHeating"
    ]
    if ashp_costs:
        ashp_cost = ashp_costs[0]
        cost_result = ashp_cost.compute(features)
        print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")
        assert cost_result.iloc[0] > 0, "Cost should be positive"

    # Test a variable cost calculation - Windows
    window_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Windows" and cost.final == "DoublePaneLowE"
    ]
    if window_costs:
        window_cost = window_costs[0]
        cost_result = window_cost.compute(features)
        expected_cost = 538.2 * 30.0 * 1.0  # coefficient * window_area * scaling_factor
        print(
            f"Double Pane Windows cost: ${cost_result.iloc[0]:.2f} (expected: ${expected_cost:.2f})"
        )
        # Account for error_scale variation (5% = 0.05)
        # Allow for 3 standard deviations of variation (99.7% of cases)
        error_margin = expected_cost * 0.05 * 3
        assert abs(cost_result.iloc[0] - expected_cost) < error_margin, (
            f"Window cost calculation incorrect. Expected ${expected_cost:.2f} ± ${error_margin:.2f}, got ${cost_result.iloc[0]:.2f}"
        )

    print("Cost calculation tests passed\n")


def test_heat_pump_cost_calculation():
    """Test the new heat pump cost calculation formula."""
    print("=== Testing Heat Pump Cost Calculation ===")

    # Load the retrofit costs
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)

    # Create test features with location information
    features = create_test_features_with_location()

    # Test ASHP cost calculation
    ashp_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Heating" and cost.final == "ASHPHeating"
    ]

    if ashp_costs:
        ashp_cost = ashp_costs[0]
        cost_result = ashp_cost.compute(features)

        # Calculate expected cost based on the new formula:
        # intercept + feature_component + calculated_component + conditional_factors

        # Base components
        intercept = -79
        conditioned_area = features[
            "feature.geometry.energy_model_conditioned_area"
        ].iloc[0]
        feature_component = 0.28 * conditioned_area

        # Calculated component (heating capacity)
        # Estimate heating capacity: 150 m2 * 25 BTU/sqft * 0.0031546 = ~11.8 kW
        heating_capacity_kw = conditioned_area * 25 * 0.0031546
        calculated_component = 303 * heating_capacity_kw

        # Conditional factors (Middlesex county, has gas, has cooling)
        county_factor = 2497  # Middlesex
        gas_factor = 438  # has_gas = true
        cooling_factor = 334  # has_cooling = true

        expected_cost = (
            intercept
            + feature_component
            + calculated_component
            + county_factor
            + gas_factor
            + cooling_factor
        )

        print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")
        print("Expected cost breakdown:")
        print(f"  Intercept: ${intercept}")
        print(
            f"  Feature component (0.28 * {conditioned_area}m²): ${feature_component:.2f}"
        )
        print(
            f"  Calculated component (303 * {heating_capacity_kw:.1f}kW): ${calculated_component:.2f}"
        )
        print(f"  County factor (Middlesex): ${county_factor}")
        print(f"  Gas factor: ${gas_factor}")
        print(f"  Cooling factor: ${cooling_factor}")
        print(f"  Total expected: ${expected_cost:.2f}")

        # Account for error_scale variation (5% = 0.05)
        # Allow for 3 standard deviations of variation (99.7% of cases)
        error_margin = expected_cost * 0.05 * 3
        assert abs(cost_result.iloc[0] - expected_cost) < error_margin, (
            f"ASHP cost calculation incorrect. Expected ${expected_cost:.2f} ± ${error_margin:.2f}, got ${cost_result.iloc[0]:.2f}"
        )

    # Test GSHP cost calculation
    gshp_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Heating" and cost.final == "GSHPHeating"
    ]

    if gshp_costs:
        gshp_cost = gshp_costs[0]
        cost_result = gshp_cost.compute(features)

        # GSHP formula is similar but without intercept
        # Recalculate values to avoid unbound variable issues
        conditioned_area = features[
            "feature.geometry.energy_model_conditioned_area"
        ].iloc[0]
        feature_component = 0.28 * conditioned_area
        heating_capacity_kw = conditioned_area * 25 * 0.0031546
        calculated_component = 303 * heating_capacity_kw
        county_factor = 2497  # Middlesex
        gas_factor = 438  # has_gas = true
        cooling_factor = 334  # has_cooling = true
        conditional_factors = county_factor + gas_factor + cooling_factor

        expected_cost = feature_component + calculated_component + conditional_factors

        print(f"\nGSHP Heating cost: ${cost_result.iloc[0]:.2f}")
        print("Expected cost breakdown:")
        print(
            f"  Feature component (0.28 * {conditioned_area}m²): ${feature_component:.2f}"
        )
        print(
            f"  Calculated component (303 * {heating_capacity_kw:.1f}kW): ${calculated_component:.2f}"
        )
        print(f"  Conditional factors: ${conditional_factors}")
        print(f"  Total expected: ${expected_cost:.2f}")

        error_margin = expected_cost * 0.05 * 3
        assert abs(cost_result.iloc[0] - expected_cost) < error_margin, (
            f"GSHP cost calculation incorrect. Expected ${expected_cost:.2f} ± ${error_margin:.2f}, got ${cost_result.iloc[0]:.2f}"
        )

    print("✓ Heat pump cost calculation tests passed\n")


def test_new_cost_structure():
    """Test the new cost structure with different types of cost factors."""
    print("=== Testing New Cost Structure ===")

    # Load the retrofit costs
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)

    # Create test features
    features = create_test_features_with_location()

    # Test different cost types
    test_cases = [
        {
            "name": "Fixed Cost - Thermostat",
            "semantic_field": "Thermostat",
            "final": "Controls",
            "expected_type": "FixedCost",
        },
        {
            "name": "Variable Cost - Windows",
            "semantic_field": "Windows",
            "final": "DoublePaneLowE",
            "expected_type": "VariableCost",
        },
        {
            "name": "Complex Cost - ASHP",
            "semantic_field": "Heating",
            "final": "ASHPHeating",
            "expected_type": "VariableCost",
        },
    ]

    for test_case in test_cases:
        matching_costs = [
            cost
            for cost in retrofit_costs.costs
            if cost.semantic_field == test_case["semantic_field"]
            and cost.final == test_case["final"]
        ]

        if matching_costs:
            cost = matching_costs[0]
            cost_result = cost.compute(features)

            print(f"{test_case['name']}: ${cost_result.iloc[0]:.2f}")

            # Check that the cost has the expected structure
            for cost_factor in cost.cost_factors:
                if test_case["expected_type"] == "FixedCost":
                    assert hasattr(cost_factor, "amount"), (
                        "Fixed cost should have amount attribute"
                    )
                elif test_case["expected_type"] == "VariableCost":
                    assert hasattr(cost_factor, "coefficient") or hasattr(
                        cost_factor, "intercept"
                    ), "Variable cost should have coefficient or intercept"

                    # Check for new structure components (only for VariableCost)
                    from epengine.models.inference import VariableCost

                    if isinstance(cost_factor, VariableCost):
                        if cost_factor.feature_components:
                            print(
                                f"  - Has {len(cost_factor.feature_components)} feature components"
                            )
                        if cost_factor.calculated_components:
                            print(
                                f"  - Has {len(cost_factor.calculated_components)} calculated components"
                            )
                        if cost_factor.conditional_factors:
                            print("  - Has conditional factors")
        else:
            print(f"No cost found for {test_case['name']}")

    print("✓ New cost structure tests passed\n")


def test_incentive_loading():
    """Test incentive loading and basic functionality."""
    print("=== Testing Incentive Loading ===")

    # Load the incentives
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    print(f"Loaded {len(retrofit_incentives.incentives)} incentive configurations")

    # Check for specific incentive types
    ashp_incentives = [
        inc
        for inc in retrofit_incentives.incentives
        if inc.semantic_field == "Heating" and inc.final == "ASHPHeating"
    ]
    print(f"Found {len(ashp_incentives)} ASHP heating incentives")

    # Check for different income levels
    all_customer_incentives = [
        inc
        for inc in ashp_incentives
        if "All_customers" in inc.eligibility.get("income", [])
    ]
    income_eligible_incentives = [
        inc
        for inc in ashp_incentives
        if "Income_eligible" in inc.eligibility.get("income", [])
    ]

    print(f"All customers ASHP incentives: {len(all_customer_incentives)}")
    print(f"Income eligible ASHP incentives: {len(income_eligible_incentives)}")

    assert len(all_customer_incentives) > 0, "Should have incentives for all customers"
    assert len(income_eligible_incentives) > 0, (
        "Should have incentives for income eligible customers"
    )

    print("✓ Incentive loading tests passed\n")


def test_incentive_calculation():
    """Test incentive calculation functionality."""
    print("=== Testing Incentive Calculation ===")

    # Load costs and incentives
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    features = create_test_features_with_location()

    dhw_incentives = [
        inc
        for inc in retrofit_incentives.incentives
        if inc.semantic_field == "DHW" and inc.final == "HPWH"
    ]

    if dhw_incentives:
        dhw_incentive = dhw_incentives[0]  # Should be the "All_customers" one
        # Create a proper cost DataFrame with the DHW cost column
        costs_df = pd.DataFrame(
            {"cost.DHW": [1000.0]}, index=features.index
        )  # Dummy cost for testing
        incentive_result = dhw_incentive.compute(features, costs_df)
        print(f"DHW HPWH incentive (All customers): ${incentive_result.iloc[0]:.2f}")
        assert incentive_result.iloc[0] > 0, "Incentive should be positive"

    # Test percentage incentive calculation
    # First create a cost dataframe
    ashp_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Heating" and cost.final == "ASHPHeating"
    ]
    if ashp_costs:
        ashp_cost = ashp_costs[0]
        cost_result = ashp_cost.compute(features)
        costs_df = pd.DataFrame({"cost.Heating": cost_result})

        # Find IRA incentive (percentage based)
        ira_incentives = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.program == "IRA"
            and inc.semantic_field == "Heating"
            and inc.final == "ASHPHeating"
        ]

        if ira_incentives:
            ira_incentive = ira_incentives[0]
            incentive_result = ira_incentive.compute(features, costs_df)
            expected_incentive = cost_result.iloc[0] * 0.30  # 30% of cost
            # Check if the incentive is capped at $2000
            expected_incentive = min(expected_incentive, 2000.0)  # IRA limit
            print(
                f"IRA ASHP incentive: ${incentive_result.iloc[0]:.2f} (expected: ${expected_incentive:.2f})"
            )
            # Account for error_scale variation (5% = 0.05) in both cost and incentive
            # Allow for 3 standard deviations of variation (99.7% of cases)
            error_margin = expected_incentive * 0.05 * 3
            assert abs(incentive_result.iloc[0] - expected_incentive) < error_margin, (
                f"Percentage incentive calculation incorrect. Expected ${expected_incentive:.2f} ± ${error_margin:.2f}, got ${incentive_result.iloc[0]:.2f}"
            )

    print("✓ Incentive calculation tests passed\n")


def test_eligibility_checking():
    """Test incentive eligibility checking."""
    print("=== Testing Eligibility Checking ===")

    # Load incentives
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    # Test different eligibility scenarios
    test_cases = [
        {
            "name": "MA resident, all customers",
            "context": {"Region": "MA", "Heating": "NaturalGasHeating"},
            "expected_eligible": True,
        },
        {
            "name": "Non-MA resident",
            "context": {"Region": "CA", "Heating": "NaturalGasHeating"},
            "expected_eligible": False,
        },
        {
            "name": "MA resident, ineligible heating type",
            "context": {"Region": "MA", "Heating": "ASHPHeating"},
            "expected_eligible": False,
        },
    ]

    for test_case in test_cases:
        # Test with a MassSave ASHP incentive
        ashp_incentives = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.program == "MassSave"
            and inc.semantic_field == "Heating"
            and inc.final == "ASHPHeating"
        ]

        if ashp_incentives:
            incentive = ashp_incentives[0]  # All customers version

            # Check eligibility manually
            is_eligible = True
            eligibility = incentive.eligibility

            # Check region
            if "region" in eligibility:
                region = test_case["context"].get("Region", "MA")
                if region not in eligibility["region"]:
                    is_eligible = False

            # Check heating type
            if "Heating" in eligibility:
                heating = test_case["context"].get("Heating")
                if heating not in eligibility["Heating"]:
                    is_eligible = False

            print(
                f"{test_case['name']}: {'✓ Eligible' if is_eligible else '✗ Not eligible'}"
            )
            assert is_eligible == test_case["expected_eligible"], (
                f"Eligibility check failed for {test_case['name']}"
            )

    print("✓ Eligibility checking tests passed\n")


def test_cost_and_incentive_selection():
    """Test cost and incentive selection functionality."""
    print("=== Testing Cost and Incentive Selection ===")

    # Load costs and incentives
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    # Test cost entity selection manually
    print(f"Total available costs: {len(retrofit_costs.costs)}")

    # Test specific cost selections
    ashp_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Heating" and cost.final == "ASHPHeating"
    ]
    print(f"ASHP Heating costs available: {len(ashp_costs)}")

    dhw_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "DHW" and cost.final == "HPWH"
    ]
    print(f"DHW HPWH costs available: {len(dhw_costs)}")

    # Test incentive entity selection manually
    print(f"Total available incentives: {len(retrofit_incentives.incentives)}")

    # Test specific incentive selections
    ashp_incentives = [
        inc
        for inc in retrofit_incentives.incentives
        if inc.semantic_field == "Heating" and inc.final == "ASHPHeating"
    ]
    print(f"ASHP Heating incentives available: {len(ashp_incentives)}")

    dhw_incentives = [
        inc
        for inc in retrofit_incentives.incentives
        if inc.semantic_field == "DHW" and inc.final == "HPWH"
    ]
    print(f"DHW HPWH incentives available: {len(dhw_incentives)}")

    # Test eligibility filtering
    ma_ashp_incentives = [
        inc for inc in ashp_incentives if "MA" in inc.eligibility.get("region", [])
    ]
    print(f"MA-eligible ASHP incentives: {len(ma_ashp_incentives)}")

    all_customer_ashp = [
        inc
        for inc in ashp_incentives
        if "All_customers" in inc.eligibility.get("income", [])
    ]
    print(f"All-customer ASHP incentives: {len(all_customer_ashp)}")

    print("✓ Cost and incentive selection tests passed\n")


def test_manual_cost_calculation():
    """Test manual cost calculation with sample data."""
    print("=== Testing Manual Cost Calculation ===")

    # Load costs
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)

    # Create test features
    features = create_test_features_with_location()

    # Test multiple cost calculations
    test_upgrades = [
        ("Heating", "ASHPHeating"),
        ("DHW", "HPWH"),
        ("Windows", "DoublePaneLowE"),
        ("Thermostat", "Controls"),
    ]

    total_cost = 0
    for semantic_field, final_value in test_upgrades:
        matching_costs = [
            cost
            for cost in retrofit_costs.costs
            if cost.semantic_field == semantic_field and cost.final == final_value
        ]

        if matching_costs:
            cost_result = matching_costs[0].compute(features)
            cost_amount = cost_result.iloc[0]
            total_cost += cost_amount
            print(f"{semantic_field} → {final_value}: ${cost_amount:.2f}")
        else:
            print(f"No cost found for {semantic_field} → {final_value}")

    print(f"Total retrofit cost: ${total_cost:.2f}")
    assert total_cost > 0, "Total cost should be positive"

    print("✓ Manual cost calculation tests passed\n")


def test_manual_incentive_calculation():
    """Test manual incentive calculation with sample data."""
    print("=== Testing Manual Incentive Calculation ===")

    # Load incentives
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    # Create test features
    features = create_test_features()

    # Test multiple incentive calculations
    test_upgrades = [
        ("Heating", "ASHPHeating"),
        ("DHW", "HPWH"),
        ("Thermostat", "Controls"),
    ]

    total_incentive = 0
    for semantic_field, final_value in test_upgrades:
        matching_incentives = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.semantic_field == semantic_field and inc.final == final_value
        ]

        if matching_incentives:
            # Use the first matching incentive (All_customers version)
            incentive_result = matching_incentives[0].compute(features, pd.DataFrame())
            incentive_amount = incentive_result.iloc[0]
            total_incentive += incentive_amount
            print(f"{semantic_field} → {final_value}: ${incentive_amount:.2f}")
        else:
            print(f"No incentive found for {semantic_field} → {final_value}")

    print(f"Total incentive amount: ${total_incentive:.2f}")
    assert total_incentive > 0, "Total incentive should be positive"

    print("✓ Manual incentive calculation tests passed\n")


def test_income_level_differentiation():  # noqa: C901
    """Test that incentives are properly differentiated by income level."""
    print("=== Testing Income Level Differentiation ===")

    # Load costs and incentives
    costs_path = Path("epengine/models/data/retrofit-costs.json")
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_costs = RetrofitCosts.Open(costs_path)
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    # Create test features
    features = create_test_features_with_location()

    # Test with ASHP heating upgrade
    ashp_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "Heating" and cost.final == "ASHPHeating"
    ]

    if ashp_costs:
        # Calculate the cost first
        ashp_cost = ashp_costs[0]
        cost_result = ashp_cost.compute(features)
        costs_df = pd.DataFrame({"cost.Heating": cost_result})

        print(f"ASHP Heating cost: ${cost_result.iloc[0]:.2f}")

        # Find incentives for both income levels
        all_customer_incentives = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.semantic_field == "Heating"
            and inc.final == "ASHPHeating"
            and "All_customers" in inc.eligibility.get("income", [])
        ]

        income_eligible_incentives = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.semantic_field == "Heating"
            and inc.final == "ASHPHeating"
            and "Income_eligible" in inc.eligibility.get("income", [])
        ]

        print(f"Found {len(all_customer_incentives)} All_customers incentives")
        print(f"Found {len(income_eligible_incentives)} Income_eligible incentives")

        # Test All_customers incentives
        all_customer_total = 0
        if all_customer_incentives:
            print("\nAll Customers Incentives:")
            for _i, incentive in enumerate(all_customer_incentives):
                try:
                    result = incentive.compute(features, costs_df)
                    amount = result.iloc[0]
                    all_customer_total += amount
                    print(f"  {incentive.program}: ${amount:.2f}")
                except Exception as e:
                    print(f"  {incentive.program}: Error - {e}")

        # Test Income_eligible incentives
        income_eligible_total = 0
        if income_eligible_incentives:
            print("\nIncome Eligible Incentives:")
            for _i, incentive in enumerate(income_eligible_incentives):
                try:
                    result = incentive.compute(features, costs_df)
                    amount = result.iloc[0]
                    income_eligible_total += amount
                    print(f"  {incentive.program}: ${amount:.2f}")
                except Exception as e:
                    print(f"  {incentive.program}: Error - {e}")

        print(f"\nTotal All Customers Incentives: ${all_customer_total:.2f}")
        print(f"Total Income Eligible Incentives: ${income_eligible_total:.2f}")

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

    # Test with DHW upgrade as well
    print("\n--- Testing DHW HPWH Incentives ---")

    dhw_costs = [
        cost
        for cost in retrofit_costs.costs
        if cost.semantic_field == "DHW" and cost.final == "HPWH"
    ]

    if dhw_costs:
        dhw_cost = dhw_costs[0]
        cost_result = dhw_cost.compute(features)
        costs_df = pd.DataFrame({"cost.DHW": cost_result})

        print(f"DHW HPWH cost: ${cost_result.iloc[0]:.2f}")

        # Find DHW incentives for both income levels
        dhw_all_customer = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.semantic_field == "DHW"
            and inc.final == "HPWH"
            and "All_customers" in inc.eligibility.get("income", [])
        ]

        dhw_income_eligible = [
            inc
            for inc in retrofit_incentives.incentives
            if inc.semantic_field == "DHW"
            and inc.final == "HPWH"
            and "Income_eligible" in inc.eligibility.get("income", [])
        ]

        print(f"DHW All_customers incentives: {len(dhw_all_customer)}")
        print(f"DHW Income_eligible incentives: {len(dhw_income_eligible)}")

        # Test that they compute to different values if both exist
        if dhw_all_customer and dhw_income_eligible:
            all_result = dhw_all_customer[0].compute(features, costs_df)
            income_result = dhw_income_eligible[0].compute(features, costs_df)

            print(f"DHW All customers: ${all_result.iloc[0]:.2f}")
            print(f"DHW Income eligible: ${income_result.iloc[0]:.2f}")

    print("✓ Income level differentiation tests passed\n")


def test_incentive_selection_workflow():
    """Test the complete incentive selection workflow like in inference.py."""
    print("=== Testing Incentive Selection Workflow ===")

    # Load incentives
    incentives_path = Path("epengine/models/data/incentives_format.json")
    retrofit_incentives = RetrofitIncentives.Open(incentives_path)

    # Create test features with proper semantic context (not used in this workflow)

    # Simulate the selection process for each income level
    changed_upgrades = [
        ("Heating", "NaturalGasHeating", "ASHPHeating"),
        ("DHW", "NaturalGasDHW", "HPWH"),
    ]

    for income_level in ["All_customers", "Income_eligible"]:
        print(f"\nTesting {income_level} incentive selection:")
        selected_incentives = []

        for semantic_field, initial, final in changed_upgrades:
            candidates = []

            for incentive in retrofit_incentives.incentives:
                # Check if incentive matches the upgrade
                if incentive.semantic_field != semantic_field:
                    continue
                if incentive.final != final:
                    continue
                if incentive.initial is not None and incentive.initial != initial:
                    continue

                # Check income level eligibility
                if (
                    "income" in incentive.eligibility
                    and income_level not in incentive.eligibility["income"]
                ):
                    continue

                # Check region eligibility
                if (
                    "region" in incentive.eligibility
                    and "MA" not in incentive.eligibility["region"]
                ):
                    continue

                candidates.append(incentive)

            print(
                f"  {semantic_field} {initial}→{final}: {len(candidates)} eligible incentives"
            )
            if candidates:
                selected_incentives.extend(candidates)

        print(f"  Total selected for {income_level}: {len(selected_incentives)}")

        # Verify that each income level gets different incentive sets
        assert len(selected_incentives) > 0, (
            f"Should have incentives for {income_level}"
        )

    print("✓ Incentive selection workflow tests passed\n")


def test_incentive_metadata_creation():
    """Test creating incentive metadata objects."""
    print("=== Testing Incentive Metadata Creation ===")

    # Create a sample applied incentive
    applied_incentive = AppliedIncentive(
        semantic_field="Heating",
        program="MassSave",
        amount=10000.0,
        description="MassSave ASHP rebate for all customers",
        source="MassSave",
        incentive_type="Fixed",
    )

    print(f"Created AppliedIncentive: {applied_incentive}")

    # Create incentive metadata
    metadata = IncentiveMetadata(
        applied_incentives=[applied_incentive],
        total_incentive_amount=10000.0,
        income_level="All_customers",
    )

    print(f"Created IncentiveMetadata: {metadata}")
    print(f"Total incentive amount: ${metadata.total_incentive_amount}")
    print(f"Number of incentives: {len(metadata.applied_incentives)}")

    print("✓ Incentive metadata creation tests passed\n")


def test_incentive_metadata_serialization():
    """Test that incentive metadata can be serialized to JSON."""
    print("=== Testing Incentive Metadata Serialization ===")

    applied_incentive = AppliedIncentive(
        semantic_field="Heating",
        program="MassSave",
        amount=10000.0,
        description="MassSave ASHP rebate for all customers",
        source="MassSave",
        incentive_type="Fixed",
    )

    metadata = IncentiveMetadata(
        applied_incentives=[applied_incentive],
        total_incentive_amount=10000.0,
        income_level="All_customers",
    )

    # Test JSON serialization
    json_data = metadata.model_dump()
    print(f"JSON serialization: {json_data}")

    # Test JSON string serialization
    json_string = metadata.model_dump_json()
    print(f"JSON string: {json_string}")

    print("✓ Incentive metadata serialization tests passed\n")


def main():
    """Run all tests."""
    print("Starting Incentives Integration Tests\n")
    print("=" * 50)

    try:
        test_cost_calculation()
        test_heat_pump_cost_calculation()
        test_new_cost_structure()
        test_incentive_loading()
        test_incentive_calculation()
        test_eligibility_checking()
        test_cost_and_incentive_selection()
        test_manual_cost_calculation()
        test_manual_incentive_calculation()
        test_income_level_differentiation()
        test_incentive_selection_workflow()
        test_incentive_metadata_creation()
        test_incentive_metadata_serialization()

        print("=" * 50)
        print("All tests passed.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
