import math

import pytest
import pandas as pd

from dispatch.interface import DispatchResult
from dispatch.lp_network import solve_from_frames
from dispatch.lp_single import HOURS_PER_YEAR
from io_loader import Frames
from policy.generation_standard import GenerationStandardPolicy, TechnologyStandard


def _two_unit_frames(load_mwh: float) -> Frames:
    demand = pd.DataFrame(
        [{"year": 2030, "region": "test", "demand_mwh": float(load_mwh)}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "baseload",
                "region": "test",
                "fuel": "baseload_fuel",
                "cap_mw": 50.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 8.0,
                "vom_per_mwh": 5.0,
                "fuel_price_per_mmbtu": 1.5,
                "ef_ton_per_mwh": 0.8,
            },
            {
                "unit_id": "peaker",
                "region": "test",
                "fuel": "peaker_fuel",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 7.0,
                "vom_per_mwh": 3.0,
                "fuel_price_per_mmbtu": 3.0,
                "ef_ton_per_mwh": 0.49,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {
                "fuel": "baseload_fuel",
                "covered": True,
                "co2_ton_per_mmbtu": 0.1,
            },
            {
                "fuel": "peaker_fuel",
                "covered": True,
                "co2_ton_per_mmbtu": 0.07,
            },
        ]
    )
    coverage = pd.DataFrame(
        [{"region": "test", "covered": True}]
    )

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
        }
    )


def _single_unit_frames(load_mwh: float) -> Frames:
    demand = pd.DataFrame(
        [{"year": 2030, "region": "solo", "demand_mwh": float(load_mwh)}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "solo_unit",
                "region": "solo",
                "fuel": "solo_fuel",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 9.0,
                "vom_per_mwh": 2.0,
                "fuel_price_per_mmbtu": 2.5,
                "ef_ton_per_mwh": 0.9,
            }
        ]
    )
    fuels = pd.DataFrame(
        [
            {
                "fuel": "solo_fuel",
                "covered": True,
                "co2_ton_per_mmbtu": 0.1,
            }
        ]
    )
    coverage = pd.DataFrame(
        [{"region": "solo", "covered": True}]
    )

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
        }
    )


def test_congestion_leads_to_price_separation() -> None:
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": "north", "demand_mwh": 40.0 * HOURS_PER_YEAR},
            {"year": 2030, "region": "south", "demand_mwh": 60.0 * HOURS_PER_YEAR},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "north_low_cost",
                "region": "north",
                "fuel": "north_supply",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 20.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "south_high_cost",
                "region": "south",
                "fuel": "south_supply",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 50.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "north_supply", "covered": True},
            {"fuel": "south_supply", "covered": True},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": "north", "covered": True},
            {"region": "south", "covered": True},
        ]
    )
    transmission = pd.DataFrame(
        [{"from_region": "north", "to_region": "south", "limit_mw": 15.0}]
    )

    frames = Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "transmission": transmission,
            "coverage": coverage,
        }
    )

    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    assert result.region_prices["north"] < result.region_prices["south"]
    assert result.region_prices["north"] == pytest.approx(20.0, rel=1e-4)
    assert result.region_prices["south"] == pytest.approx(50.0, rel=1e-4)

    assert result.gen_by_fuel["north_supply"] == pytest.approx(55.0 * HOURS_PER_YEAR)
    assert result.gen_by_fuel["south_supply"] == pytest.approx(45.0 * HOURS_PER_YEAR)
    assert math.isclose(result.emissions_tons, 0.0)
    assert ("north", "south") in result.flows
    assert result.flows[("north", "south")] == pytest.approx(15.0 * HOURS_PER_YEAR)
    assert sum(result.emissions_by_region.values()) == pytest.approx(result.emissions_tons)


def test_imports_increase_with_carbon_price() -> None:
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": "covered", "demand_mwh": 100.0 * HOURS_PER_YEAR},
            {"year": 2030, "region": "external", "demand_mwh": 0.0},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "covered_coal",
                "region": "covered",
                "fuel": "covered_supply",
                "cap_mw": 150.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 25.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.5,
            },
            {
                "unit_id": "external_gas",
                "region": "external",
                "fuel": "external_supply",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 30.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "covered_supply", "covered": True},
            {"fuel": "external_supply", "covered": False},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": "covered", "covered": True},
            {"region": "external", "covered": False},
        ]
    )
    transmission = pd.DataFrame(
        [{"from_region": "covered", "to_region": "external", "limit_mw": 200.0}]
    )

    frames = Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "transmission": transmission,
            "coverage": coverage,
        }
    )

    low_price = solve_from_frames(frames, 2030, allowance_cost=0.0, carbon_price=0.0)
    high_price = solve_from_frames(
        frames, 2030, allowance_cost=0.0, carbon_price=40.0
    )

    assert low_price.imports_to_covered == pytest.approx(0.0, abs=1e-6)
    assert high_price.imports_to_covered > low_price.imports_to_covered
    assert high_price.exports_from_covered == pytest.approx(0.0, abs=1e-6)
    assert high_price.region_coverage["covered"] is True
    assert high_price.region_coverage["external"] is False
    assert high_price.region_prices["covered"] == pytest.approx(30.0, rel=1e-4)
    assert high_price.region_prices["external"] == pytest.approx(30.0, rel=1e-4)
    assert high_price.emissions_tons < low_price.emissions_tons
    assert sum(high_price.emissions_by_region.values()) == pytest.approx(
        high_price.emissions_tons
    )


def test_region_coverage_overrides_fuel_flags() -> None:
    demand = pd.DataFrame(
        [
            {"year": 2035, "region": "covered", "demand_mwh": 80.0 * HOURS_PER_YEAR},
            {"year": 2035, "region": "external", "demand_mwh": 0.0},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "covered_clean",
                "region": "covered",
                "fuel": "clean",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 25.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "external_coal",
                "region": "external",
                "fuel": "coal",
                "cap_mw": 200.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 15.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 1.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "clean", "covered": True},
            {"fuel": "coal", "covered": True},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": "covered", "covered": True},
            {"region": "external", "covered": False},
        ]
    )
    transmission = pd.DataFrame(
        [{"from_region": "external", "to_region": "covered", "limit_mw": 500.0}]
    )

    frames = Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "transmission": transmission,
            "coverage": coverage,
        }
    )

    result = solve_from_frames(frames, 2035, allowance_cost=50.0)

    assert result.region_prices["covered"] == pytest.approx(15.0, rel=1e-4)
    assert result.generation_by_region["external"] > 0.0
    assert result.generation_by_coverage["non_covered"] > 0.0


def test_leakage_percentage_helper() -> None:
    baseline = DispatchResult(
        gen_by_fuel={"coal": 60.0},
        region_prices={"region": 25.0},
        emissions_tons=0.0,
        emissions_by_region={"region": 0.0},
        flows={},
        generation_by_region={"region": 60.0},
        generation_by_coverage={"covered": 40.0, "non_covered": 20.0},
    )

    scenario = DispatchResult(
        gen_by_fuel={"coal": 50.0, "gas": 30.0},
        region_prices={"region": 30.0},
        emissions_tons=0.0,
        emissions_by_region={"region": 0.0},
        flows={},
        generation_by_region={"region": 80.0},
        generation_by_coverage={"covered": 45.0, "non_covered": 35.0},
    )

    expected = 100.0 * (35.0 - 20.0) / (80.0 - 60.0)
    assert scenario.leakage_percent(baseline) == pytest.approx(expected)


def test_generation_standard_enforces_share() -> None:
    demand = pd.DataFrame(
        [{"year": 2030, "region": "alpha", "demand_mwh": 100.0 * HOURS_PER_YEAR}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "alpha_wind",
                "region": "alpha",
                "fuel": "wind",
                "cap_mw": 120.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 40.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "alpha_gas",
                "region": "alpha",
                "fuel": "gas",
                "cap_mw": 160.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 10.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.4,
            },
        ]
    )
    fuels = pd.DataFrame([
        {"fuel": "wind", "covered": True},
        {"fuel": "gas", "covered": True},
    ])

    frames = Frames({"demand": demand, "units": units, "fuels": fuels})

    share_df = pd.DataFrame({"alpha": [0.25]}, index=pd.Index([2030], name="year"))
    standard = TechnologyStandard(
        technology="wind", generation_table=share_df, enabled_regions={"alpha"}
    )
    policy = GenerationStandardPolicy([standard])

    result = solve_from_frames(
        frames, 2030, allowance_cost=0.0, generation_standard=policy
    )

    total_generation = result.generation_by_region["alpha"]
    wind_generation = result.gen_by_fuel["wind"]
    gas_generation = result.gen_by_fuel["gas"]

    assert wind_generation == pytest.approx(0.25 * total_generation, rel=1e-4)
    assert gas_generation == pytest.approx(0.75 * total_generation, rel=1e-4)


def test_generation_standard_capacity_violation() -> None:
    demand = pd.DataFrame(
        [{"year": 2030, "region": "alpha", "demand_mwh": 80.0 * HOURS_PER_YEAR}]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "alpha_wind",
                "region": "alpha",
                "fuel": "wind",
                "cap_mw": 100.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 35.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
            },
            {
                "unit_id": "alpha_gas",
                "region": "alpha",
                "fuel": "gas",
                "cap_mw": 150.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 12.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.5,
            },
        ]
    )
    fuels = pd.DataFrame([
        {"fuel": "wind", "covered": True},
        {"fuel": "gas", "covered": True},
    ])

    frames = Frames({"demand": demand, "units": units, "fuels": fuels})

    capacity_df = pd.DataFrame({"alpha": [200.0]}, index=pd.Index([2030], name="year"))
    standard = TechnologyStandard(
        technology="wind", capacity_table=capacity_df, enabled_regions={"alpha"}
    )
    policy = GenerationStandardPolicy([standard])

    with pytest.raises(ValueError, match="wind"):
        solve_from_frames(
            frames, 2030, allowance_cost=0.0, generation_standard=policy
        )

def test_heat_rate_and_fuel_price_determine_variable_cost() -> None:
    frames = _two_unit_frames(load_mwh=500_000.0)
    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    units = frames.units()
    fuels = frames.fuels()
    co2_map = {
        str(row.fuel): float(getattr(row, "co2_ton_per_mmbtu", 0.0))
        for row in fuels.itertuples(index=False)
    }

    variable_costs = {
        row.fuel: float(row.vom_per_mwh)
        + float(row.hr_mmbtu_per_mwh) * float(row.fuel_price_per_mmbtu)
        for row in units.itertuples(index=False)
    }

    active_costs = [
        variable_costs[fuel]
        for fuel, generation in result.gen_by_fuel.items()
        if generation > 0.0
    ]
    assert active_costs
    assert result.region_prices["test"] == pytest.approx(max(active_costs), rel=1e-6)

    expected_emissions = 0.0
    expected_from_fuel = 0.0
    for row in units.itertuples(index=False):
        generation = float(result.gen_by_fuel.get(row.fuel, 0.0))
        expected_emissions += generation * float(row.ef_ton_per_mwh)
        expected_from_fuel += (
            generation * float(row.hr_mmbtu_per_mwh) * co2_map.get(row.fuel, 0.0)
        )

    assert result.emissions_tons == pytest.approx(expected_emissions, rel=1e-6)
    assert result.emissions_tons == pytest.approx(expected_from_fuel, rel=1e-6)


def test_generation_and_emissions_scale_with_demand() -> None:
    base_frames = _single_unit_frames(load_mwh=400_000.0)
    higher_frames = _single_unit_frames(load_mwh=440_000.0)

    base = solve_from_frames(base_frames, 2030, allowance_cost=0.0)
    higher = solve_from_frames(higher_frames, 2030, allowance_cost=0.0)

    assert base.emissions_tons > 0.0
    assert higher.total_generation > base.total_generation

    generation_ratio = higher.total_generation / base.total_generation
    emissions_ratio = higher.emissions_tons / base.emissions_tons

    assert generation_ratio == pytest.approx(1.1, rel=1e-6)
    assert emissions_ratio == pytest.approx(1.1, rel=1e-6)
