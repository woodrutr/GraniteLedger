"""Tests for the single-region merit-order dispatch implementation."""

from __future__ import annotations

import importlib
from typing import Iterable

import pytest

pytest.importorskip("pandas")

from dispatch.interface import DispatchResult
from dispatch.lp_single import _dispatch_merit_order, solve

_fixtures = importlib.import_module("tests.fixtures.dispatch_single_minimal")
baseline_frames = _fixtures.baseline_frames
baseline_units = _fixtures.baseline_units
infeasible_frames = _fixtures.infeasible_frames
frames_with_expansion = _fixtures.frames_with_expansion
expansion_options = _fixtures.expansion_options


def _collect_emissions(costs: Iterable[float]) -> list[float]:
    """Helper returning emissions for the supplied allowance price path."""

    emissions: list[float] = []
    for cost in costs:
        frames = baseline_frames()
        result = solve(2030, cost, frames=frames)
        emissions.append(result.emissions_tons)
    return emissions


def test_merit_order_shifts_with_allowance_cost() -> None:
    """Low allowance prices favour coal while high prices shift toward gas."""

    units = baseline_units()
    load = 1_000_000.0

    low_cost = _dispatch_merit_order(units, load, allowance_cost=0.0)
    high_cost = _dispatch_merit_order(units, load, allowance_cost=50.0)

    wind_cap = low_cost["units"].loc["wind-1", "cap_mwh"]

    assert low_cost["generation"].loc["wind-1"] == pytest.approx(wind_cap)
    assert high_cost["generation"].loc["wind-1"] == pytest.approx(wind_cap)
    assert low_cost["generation"].loc["coal-1"] > high_cost["generation"].loc["coal-1"]
    assert low_cost["generation"].loc["gas-1"] < high_cost["generation"].loc["gas-1"]

    result = solve(2030, 0.0, frames=baseline_frames())

    assert isinstance(result, DispatchResult)
    assert set(result.gen_by_fuel) == {"wind", "coal", "gas"}
    assert result.emissions_by_region["default"] == pytest.approx(result.emissions_tons)
    assert result.flows == {}


def test_price_matches_marginal_cost_of_marginal_unit() -> None:
    """The reported price equals the marginal cost of the last dispatched unit."""

    units = baseline_units()
    summary = _dispatch_merit_order(units, 1_000_000.0, allowance_cost=0.0)

    used = summary["generation"][summary["generation"] > 0.0]
    last_unit = used.index[-1]
    expected_price = summary["units"].loc[last_unit, "marginal_cost"]

    assert summary["price"] == pytest.approx(expected_price)

    result = solve(2030, 0.0, frames=baseline_frames())

    assert result.region_prices["default"] == pytest.approx(expected_price)


def test_emissions_decline_with_allowance_cost() -> None:
    """Increasing the allowance price should not raise emissions."""

    costs = [0.0, 10.0, 30.0, 60.0]
    emissions = _collect_emissions(costs)

    assert all(a >= b - 1e-9 for a, b in zip(emissions, emissions[1:]))


def test_emissions_decline_with_carbon_price() -> None:
    """An exogenous carbon price should suppress emissions."""

    prices = [0.0, 15.0, 45.0]
    emissions: list[float] = []
    for price in prices:
        result = solve(2030, 0.0, frames=baseline_frames(), carbon_price=price)
        emissions.append(result.emissions_tons)

    assert all(a >= b - 1e-9 for a, b in zip(emissions, emissions[1:]))


def test_generation_respects_capacity_limits() -> None:
    """No unit may exceed its annual energy capability."""

    units = baseline_units()
    summary = _dispatch_merit_order(units, 1_000_000.0, allowance_cost=20.0)

    caps = summary["units"]["cap_mwh"]
    for unit_id, dispatched in summary["generation"].items():
        assert dispatched <= caps.loc[unit_id] + 1e-6

    assert summary["generation"].sum() == pytest.approx(1_000_000.0)


def test_infeasible_load_reports_shortfall_and_price() -> None:
    """Loads above total capability return the correct price and shortfall."""

    frames = infeasible_frames()
    demand = frames.demand()
    year = int(demand.iloc[0]["year"])
    load = float(demand.loc[demand["year"] == year, "demand_mwh"].sum())

    summary = _dispatch_merit_order(frames.units(), load, allowance_cost=10.0)
    caps = summary["units"]["cap_mwh"]
    total_cap = caps.sum()
    shortfall_expected = load - total_cap

    assert shortfall_expected > 0.0
    assert summary["shortfall_mwh"] == pytest.approx(shortfall_expected)
    assert summary["generation"].equals(caps)

    used = summary["generation"][summary["generation"] > 0.0]
    last_unit = used.index[-1]
    expected_price = summary["units"].loc[last_unit, "marginal_cost"]

    assert summary["price"] == pytest.approx(expected_price)

    result = solve(year, 10.0, frames=frames)

    assert pytest.approx(total_cap) == sum(result.gen_by_fuel.values())


def test_capacity_expansion_inactive_without_trigger() -> None:
    """Expansion audit should not build capacity when economics are unfavourable."""

    frames = frames_with_expansion()
    result = solve(2030, 0.0, frames=frames, capacity_expansion=True)

    assert result.capacity_builds == []


def test_capacity_expansion_resolves_shortage() -> None:
    """Shortages trigger builds even when NPV would otherwise be negative."""

    frames = infeasible_frames()
    frames = frames.with_frame("expansion", expansion_options())
    demand = frames.demand()
    year = int(demand.iloc[0]["year"])
    load = float(demand.loc[demand["year"] == year, "demand_mwh"].sum())

    result = solve(year, 0.0, frames=frames, capacity_expansion=True)

    assert result.capacity_builds, "expected at least one build to resolve shortage"
    assert any(entry["reason"] == "supply_shortage" for entry in result.capacity_builds)
    assert sum(result.gen_by_fuel.values()) == pytest.approx(load)

    for entry in result.capacity_builds:
        assert {"capex_cost", "opex_cost", "emissions_tons"}.issubset(entry)
        if entry["reason"] == "supply_shortage":
            assert entry["npv_positive"] in {True, False}


def test_capacity_expansion_responds_to_carbon_price() -> None:
    """Positive NPV triggered by high carbon prices induces clean builds."""

    frames = frames_with_expansion()

    baseline = solve(2030, 0.0, frames=frames, capacity_expansion=True, carbon_price=0.0)
    high_price = solve(2030, 0.0, frames=frames, capacity_expansion=True, carbon_price=80.0)

    assert baseline.capacity_builds == []
    assert high_price.capacity_builds, "expected build under elevated carbon price"

    clean_build = next(
        entry for entry in high_price.capacity_builds if entry["reason"] == "npv_positive"
    )
    assert clean_build["npv_positive"] is True
    assert clean_build["capex_cost"] > 0.0
    assert clean_build["opex_cost"] >= 0.0
    assert clean_build["emissions_tons"] == pytest.approx(0.0, abs=1e-9)
    assert high_price.emissions_tons < baseline.emissions_tons
