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
    ) -> None:
        """Persist the stored DataFrames to ``outdir`` as CSV files."""

        _ensure_pandas()

        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.annual.to_csv(output_dir / annual_filename, index=False)
        self.emissions_by_region.to_csv(output_dir / emissions_filename, index=False)
        self.price_by_region.to_csv(output_dir / price_filename, index=False)
        self.flows.to_csv(output_dir / flows_filename, index=False)

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

