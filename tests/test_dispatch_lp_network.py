"""Tests for the linear programming dispatch with a regional network."""

from __future__ import annotations

import math

import pytest

from dispatch.lp_network import GeneratorSpec, solve


def test_congestion_leads_to_price_separation() -> None:
    """A binding interface should separate regional prices."""

    loads = {'north': 40.0, 'south': 60.0}
    generators = [
        GeneratorSpec(
            name='north_low_cost',
            region='north',
            fuel='north_supply',
            variable_cost=20.0,
            capacity=200.0,
            emission_rate=0.0,
        ),
        GeneratorSpec(
            name='south_high_cost',
            region='south',
            fuel='south_supply',
            variable_cost=50.0,
            capacity=200.0,
            emission_rate=0.0,
        ),
    ]
    interfaces = {('north', 'south'): 15.0}

    result = solve(loads, generators, interfaces, allowance_cost=0.0)

    assert result.region_prices['north'] < result.region_prices['south']
    assert result.region_prices['north'] == pytest.approx(20.0, rel=1e-4)
    assert result.region_prices['south'] == pytest.approx(50.0, rel=1e-4)

    assert result.gen_by_fuel['north_supply'] == pytest.approx(55.0)
    assert result.gen_by_fuel['south_supply'] == pytest.approx(45.0)
    assert math.isclose(result.emissions_tons, 0.0)


def test_imports_increase_with_carbon_price() -> None:
    """Imports into a covered region should rise as allowance prices increase."""

    loads = {'covered': 100.0, 'external': 0.0}
    generators = [
        GeneratorSpec(
            name='covered_coal',
            region='covered',
            fuel='covered_supply',
            variable_cost=25.0,
            capacity=150.0,
            emission_rate=0.5,
            covered=True,
        ),
        GeneratorSpec(
            name='external_gas',
            region='external',
            fuel='external_supply',
            variable_cost=30.0,
            capacity=200.0,
            emission_rate=0.0,
            covered=False,
        ),
    ]
    interfaces = {('covered', 'external'): 200.0}

    low_price = solve(loads, generators, interfaces, allowance_cost=0.0)
    high_price = solve(loads, generators, interfaces, allowance_cost=40.0)

    imports_low = loads['covered'] - low_price.gen_by_fuel['covered_supply']
    imports_high = loads['covered'] - high_price.gen_by_fuel['covered_supply']

    assert imports_low == pytest.approx(0.0, abs=1e-6)
    assert imports_high > imports_low
    assert high_price.region_prices['covered'] == pytest.approx(30.0, rel=1e-4)
    assert high_price.region_prices['external'] == pytest.approx(30.0, rel=1e-4)
    assert high_price.emissions_tons < low_price.emissions_tons
