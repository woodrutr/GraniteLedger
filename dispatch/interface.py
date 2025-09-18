"""Interfaces for the simplified dispatch engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DispatchResult:
    """Container for the outputs of a dispatch run.

    Attributes
    ----------
    gen_by_fuel:
        Mapping of fuel name to the dispatched generation in megawatt-hours.
    region_prices:
        Mapping of model region identifiers to their marginal energy prices in
        dollars per megawatt-hour.
    emissions_tons:
        Total carbon dioxide emissions from the dispatch solution measured in tons.
    """

    gen_by_fuel: Dict[str, float]
    region_prices: Dict[str, float]
    emissions_tons: float

