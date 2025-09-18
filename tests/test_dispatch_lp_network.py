"""Tests for the linear programming dispatch with a regional network."""

from __future__ import annotations

import math

import pytest

import pandas as pd

from dispatch.lp_network import solve_from_frames
from dispatch.lp_single import HOURS_PER_YEAR
from io_loader import Frames


def test_congestion_leads_to_price_separation() -> None:
    """A binding interface should separate regional prices."""

    demand = pd.DataFrame(
        [
            {'year': 2030, 'region': 'north', 'demand_mwh': 40.0 * HOURS_PER_YEAR},
            {'year': 2030, 'region': 'south', 'demand_mwh': 60.0 * HOURS_PER_YEAR},
        ]
    )
    units = pd.DataFrame(
        [
            {
                'unit_id': 'north_low_cost',
                'region': 'north',
                'fuel': 'north_supply',
                'cap_mw': 200.0,
                'availability': 1.0,
                'hr_mmbtu_per_mwh': 0.0,
                'vom_per_mwh': 20.0,
                'fuel_price_per_mmbtu': 0.0,
                'ef_ton_per_mwh': 0.0,
            },
            {
                'unit_id': 'south_high_cost',
                'region': 'south',
                'fuel': 'south_supply',
                'cap_mw': 200.0,
                'availability': 1.0,
                'hr_mmbtu_per_mwh': 0.0,
                'vom_per_mwh': 50.0,
                'fuel_price_per_mmbtu': 0.0,
                'ef_ton_per_mwh': 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {'fuel': 'north_supply', 'covered': True},
            {'fuel': 'south_supply', 'covered': True},
        ]
    )
    transmission = pd.DataFrame(
        [{'from_region': 'north', 'to_region': 'south', 'limit_mw': 15.0}]
    )

    frames = Frames(
        {
            'demand': demand,
            'units': units,
            'fuels': fuels,
            'transmission': transmission,
        }
    )

    result = solve_from_frames(frames, 2030, allowance_cost=0.0)

    assert result.region_prices['north'] < result.region_prices['south']
    assert result.region_prices['north'] == pytest.approx(20.0, rel=1e-4)
    assert result.region_prices['south'] == pytest.approx(50.0, rel=1e-4)

    assert result.gen_by_fuel['north_supply'] == pytest.approx(55.0 * HOURS_PER_YEAR)
    assert result.gen_by_fuel['south_supply'] == pytest.approx(45.0 * HOURS_PER_YEAR)
    assert math.isclose(result.emissions_tons, 0.0)


def test_imports_increase_with_carbon_price() -> None:
    """Imports into a covered region should rise as allowance prices increase."""

    demand = pd.DataFrame(
        [
            {'year': 2030, 'region': 'covered', 'demand_mwh': 100.0 * HOURS_PER_YEAR},
            {'year': 2030, 'region': 'external', 'demand_mwh': 0.0},
        ]
    )
    units = pd.DataFrame(
        [
            {
                'unit_id': 'covered_coal',
                'region': 'covered',
                'fuel': 'covered_supply',
                'cap_mw': 150.0,
                'availability': 1.0,
                'hr_mmbtu_per_mwh': 0.0,
                'vom_per_mwh': 25.0,
                'fuel_price_per_mmbtu': 0.0,
                'ef_ton_per_mwh': 0.5,
            },
            {
                'unit_id': 'external_gas',
                'region': 'external',
                'fuel': 'external_supply',
                'cap_mw': 200.0,
                'availability': 1.0,
                'hr_mmbtu_per_mwh': 0.0,
                'vom_per_mwh': 30.0,
                'fuel_price_per_mmbtu': 0.0,
                'ef_ton_per_mwh': 0.0,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {'fuel': 'covered_supply', 'covered': True},
            {'fuel': 'external_supply', 'covered': False},
        ]
    )
    transmission = pd.DataFrame(
        [{'from_region': 'covered', 'to_region': 'external', 'limit_mw': 200.0}]
    )

    frames = Frames(
        {
            'demand': demand,
            'units': units,
            'fuels': fuels,
            'transmission': transmission,
        }
    )

    low_price = solve_from_frames(frames, 2030, allowance_cost=0.0)
    high_price = solve_from_frames(frames, 2030, allowance_cost=40.0)

    covered_load = demand.loc[demand['region'] == 'covered', 'demand_mwh'].iloc[0]
    imports_low = covered_load - low_price.gen_by_fuel['covered_supply']
    imports_high = covered_load - high_price.gen_by_fuel['covered_supply']

    assert imports_low == pytest.approx(0.0, abs=1e-6)
    assert imports_high > imports_low
    assert high_price.region_prices['covered'] == pytest.approx(30.0, rel=1e-4)
    assert high_price.region_prices['external'] == pytest.approx(30.0, rel=1e-4)
    assert high_price.emissions_tons < low_price.emissions_tons
