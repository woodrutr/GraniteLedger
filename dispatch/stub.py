"""Simple deterministic dispatch stub used for tests and development."""

from __future__ import annotations

from typing import Dict

from .interface import DispatchResult

# Intercept and slope used to create a smooth response of emissions to the
# allowance cost supplied to :func:`solve`.
EMISSIONS_INTERCEPT: float = 120.0
EMISSIONS_SLOPE: float = 3.0

# Deterministic generation mix that scales mildly with the model year to make
# the stub results appear dynamic without introducing complicated logic.
_BASE_YEAR: int = 2020
_BASE_GEN_BY_FUEL: Dict[str, float] = {
    'coal': 30.0,
    'gas': 45.0,
    'wind': 25.0,
}

# Region prices are fixed so that unit tests can exercise predictable values.
_REGION_PRICES: Dict[str, float] = {
    'east': 35.0,
    'west': 33.0,
}


def solve(
    year: int,
    allowance_cost: float,
    carbon_price: float = 0.0,
    *,
    capacity_expansion: bool = False,
    **_ignored,
) -> DispatchResult:
    """Return a deterministic dispatch result.

    Parameters
    ----------
    year:
        Calendar year for the dispatch run. Used to slightly scale the base
        generation by fuel.
    allowance_cost:
        Carbon allowance price in dollars per ton. Emissions respond linearly
        to the combined effect of ``allowance_cost`` and ``carbon_price`` using
        ``E = max(0, a - b * (allowance_cost + carbon_price))``.
    carbon_price:
        Exogenous carbon price applied to all emissions in dollars per ton.
    """

    effective_price = float(allowance_cost) + float(carbon_price)
    emissions = max(0.0, EMISSIONS_INTERCEPT - EMISSIONS_SLOPE * effective_price)

    # Keep the generation mix dynamic across years while maintaining the same
    # proportions between fuels.
    demand_factor = 1.0 + 0.01 * (year - _BASE_YEAR)
    gen_by_fuel = {
        fuel: output * demand_factor for fuel, output in _BASE_GEN_BY_FUEL.items()
    }

    return DispatchResult(
        gen_by_fuel=gen_by_fuel,
        region_prices=dict(_REGION_PRICES),
        emissions_tons=emissions,
        emissions_by_region={'system': emissions},
        flows={},
        capacity_builds=[],
    )


__all__ = [
    'DispatchResult',
    'EMISSIONS_INTERCEPT',
    'EMISSIONS_SLOPE',
    'solve',
]

