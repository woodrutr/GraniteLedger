"""Interfaces for the simplified dispatch engine."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


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
    emissions_by_region:
        Mapping of model region identifiers to their contribution to total
        emissions measured in tons.
    flows:
        Net energy transfers between regions measured in megawatt-hours. Keys
        are tuples ``(region_a, region_b)`` where positive values indicate a
        flow from ``region_a`` to ``region_b``.
    generation_by_region:
        Mapping of regions to total generation produced within the region.
    generation_by_coverage:
        Aggregated generation grouped by coverage status with keys ``'covered'``
        and ``'non_covered'``.
    imports_to_covered:
        Total net imports flowing into covered regions in megawatt-hours.
    exports_from_covered:
        Total net exports flowing out of covered regions in megawatt-hours.
    region_coverage:
        Mapping of regions to the boolean coverage flag used in the solution.
    """

    gen_by_fuel: Dict[str, float]
    region_prices: Dict[str, float]
    emissions_tons: float
    emissions_by_region: Dict[str, float] = field(default_factory=dict)
    flows: Dict[Tuple[str, str], float] = field(default_factory=dict)
    generation_by_region: Dict[str, float] = field(default_factory=dict)
    generation_by_coverage: Dict[str, float] = field(default_factory=dict)
    imports_to_covered: float = 0.0
    exports_from_covered: float = 0.0
    region_coverage: Dict[str, bool] = field(default_factory=dict)
    generation_by_unit: Dict[str, float] = field(default_factory=dict)
    constraint_duals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    total_cost: float = 0.0

    @property
    def total_generation(self) -> float:
        """Return the total dispatched generation in megawatt-hours."""
        return float(sum(self.gen_by_fuel.values()))

    @property
    def covered_generation(self) -> float:
        """Return total generation attributed to covered regions."""
        return float(self.generation_by_coverage.get("covered", 0.0))

    @property
    def non_covered_generation(self) -> float:
        """Return total generation attributed to non-covered regions."""
        return float(self.generation_by_coverage.get("non_covered", 0.0))

    def leakage_percent(self, baseline: "DispatchResult") -> float:
        """Return leakage relative to ``baseline`` as a percentage.

        Leakage is defined as the ratio of the change in non-covered
        generation to the change in total generation between this result and
        ``baseline``. A positive value indicates that uncovered generation grew
        faster than total generation, signalling leakage.
        """
        delta_total = self.total_generation - baseline.total_generation
        if abs(delta_total) <= 1e-9:
            return 0.0

        delta_uncovered = (
            self.non_covered_generation - baseline.non_covered_generation
        )
        return 100.0 * delta_uncovered / delta_total
