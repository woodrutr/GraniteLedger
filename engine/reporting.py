"""Utilities for exporting standardized reporting CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

from engine.outputs import EngineOutputs, STANDARD_REPORT_COLUMNS

DEFAULT_EXPORT_DIR = Path(__file__).resolve().parents[1] / "results" / "exports"


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before exporting reports."""

    if pd is None:  # pragma: no cover - exercised indirectly
        raise ImportError(
            "pandas is required for engine.reporting; install it with `pip install pandas`."
        )


def export_standard_reports(
    outputs: EngineOutputs, export_dir: str | Path | None = None
) -> Mapping[str, Path]:
    """Write standardized reporting CSVs for core metrics.

    Parameters
    ----------
    outputs:
        Structured outputs produced by the annual engine run.
    export_dir:
        Destination directory for the exported CSV files. Defaults to
        ``results/exports`` relative to the project root.

    Returns
    -------
    Mapping[str, Path]
        Mapping of file stem to the path of the exported CSV file.
    """

    _ensure_pandas()

    target_dir = Path(export_dir) if export_dir is not None else DEFAULT_EXPORT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "emissions_by_region": outputs.emissions_report,
        "generation_by_technology": outputs.generation_by_technology,
        "allowance_prices": outputs.allowance_prices,
        "demand_by_region": outputs.demand_by_region,
        "imports_by_region": outputs.imports_by_region,
    }

    exported: dict[str, Path] = {}
    for name, df in files.items():
        path = target_dir / f"{name}.csv"
        if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive guard
            raise TypeError(f"EngineOutputs.{name} must be a pandas DataFrame")
        missing = [column for column in STANDARD_REPORT_COLUMNS if column not in df.columns]
        if missing:  # pragma: no cover - defensive guard
            raise ValueError(
                f"EngineOutputs.{name} missing required columns: {', '.join(missing)}"
            )
        df.to_csv(path, index=False)
        exported[name] = path

    return exported


__all__ = ["DEFAULT_EXPORT_DIR", "export_standard_reports"]
