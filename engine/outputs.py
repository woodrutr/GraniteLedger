"""Structured containers for storing engine outputs and serialising to CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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


__all__ = ['EngineOutputs']

