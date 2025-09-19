from __future__ import annotations

"""Tests for the linear programming dispatch with a regional network."""

import math
import pytest

from dispatch.interface import DispatchResult
from dispatch.lp_network import solve_from_frames
from dispatch.lp_single import HOURS_PER_YEAR
from io_loader import Frames

pd = pytest.importorskip("pandas")

from dispatch.interface import DispatchResult
from dispatch.lp_network import solve_from_frames
from dispatch.lp_single import HOURS_PER_YEAR
from io_loader import Frames


def test_congestion_leads_to_price_separation() -> None:
    """A binding interface should separate regional prices."""

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
    """Imports into a covered region should rise as allowance prices increase."""

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

    low_price = solve_from_frames(frames, 2030, allowance_cost=0.0)
    high_price = solve_from_frames(frames, 2030, allowance_cost=40.0)

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
    """Units in uncovered regions should not pay allowance costs even if their fuel is covered."""

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
    """The convenience leakage calculator should follow the documented formula."""

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
