"""Structured containers for storing engine outputs and serialising to CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class EngineOutputs:
    """Container bundling the primary outputs of the annual engine."""

    annual: pd.DataFrame
    emissions_by_region: pd.DataFrame
    price_by_region: pd.DataFrame
    flows: pd.DataFrame

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

        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.annual.to_csv(output_dir / annual_filename, index=False)
        self.emissions_by_region.to_csv(output_dir / emissions_filename, index=False)
        self.price_by_region.to_csv(output_dir / price_filename, index=False)
        self.flows.to_csv(output_dir / flows_filename, index=False)


__all__ = ['EngineOutputs']

