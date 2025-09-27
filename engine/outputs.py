"""Structured containers for storing engine outputs and serialising to CSV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(Any, None)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before working with outputs."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for engine.outputs; install it with `pip install pandas`."
        )


@dataclass(frozen=True)
class EngineOutputs:
    """Container bundling the primary outputs of the annual engine."""

    annual: pd.DataFrame
    emissions_by_region: pd.DataFrame
    price_by_region: pd.DataFrame
    flows: pd.DataFrame
    limiting_factors: list[str] = field(default_factory=list)
    emissions_total: Mapping[int, float] = field(default_factory=dict)
    emissions_by_region_map: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    generation_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["year", "fuel", "generation_mwh"])
    )
    capacity_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "fuel", "capacity_mwh", "capacity_mw"]
        )
    )
    cost_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "year",
                "fuel",
                "variable_cost",
                "allowance_cost",
                "carbon_price_cost",
                "total_cost",
            ]
        )
    )
    emissions_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["year", "fuel", "emissions_tons"])
    )
    stranded_units: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "unit", "capacity_mwh", "capacity_mw"]
        )
    )
    audits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_pandas()

    def to_csv(
        self,
        outdir: str | Path,
        *,
        annual_filename: str = 'annual.csv',
        emissions_filename: str = 'emissions_by_region.csv',
        price_filename: str = 'price_by_region.csv',
        flows_filename: str = 'flows.csv',
        generation_filename: str = 'generation_by_fuel.csv',
        capacity_filename: str = 'capacity_by_fuel.csv',
        cost_filename: str = 'cost_by_fuel.csv',
        emissions_fuel_filename: str = 'emissions_by_fuel.csv',
        stranded_filename: str = 'stranded_units.csv',
    ) -> None:
        """Persist the stored DataFrames to ``outdir`` as CSV files."""

        _ensure_pandas()

        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.annual.to_csv(output_dir / annual_filename, index=False)
        self.emissions_by_region.to_csv(output_dir / emissions_filename, index=False)
        self.price_by_region.to_csv(output_dir / price_filename, index=False)
        self.flows.to_csv(output_dir / flows_filename, index=False)
        if not self.generation_by_fuel.empty:
            self.generation_by_fuel.to_csv(output_dir / generation_filename, index=False)
        if not self.capacity_by_fuel.empty:
            self.capacity_by_fuel.to_csv(output_dir / capacity_filename, index=False)
        if not self.cost_by_fuel.empty:
            self.cost_by_fuel.to_csv(output_dir / cost_filename, index=False)
        if not self.emissions_by_fuel.empty:
            self.emissions_by_fuel.to_csv(
                output_dir / emissions_fuel_filename, index=False
            )
        if not self.stranded_units.empty:
            self.stranded_units.to_csv(output_dir / stranded_filename, index=False)

    def emissions_summary_table(self) -> "pd.DataFrame":
        """Return a normalised emissions-by-region table for reporting."""

        _ensure_pandas()

        frame = self.emissions_by_region.copy()
        if frame.empty:
            return frame

        working = frame.copy()
        if "year" in working.columns:
            working["year"] = pd.to_numeric(working["year"], errors="coerce")
            working = working.dropna(subset=["year"])
            working["year"] = working["year"].astype(int)
        else:
            working["year"] = 0

        if "region" in working.columns:
            working["region"] = working["region"].astype(str)
        else:
            working["region"] = "system"

        working["emissions_tons"] = pd.to_numeric(
            working.get("emissions_tons", 0.0), errors="coerce"
        ).fillna(0.0)

        summary_columns = ["year", "region", "emissions_tons"]
        return working[summary_columns].sort_values(summary_columns[:2]).reset_index(drop=True)


__all__ = ['EngineOutputs']

