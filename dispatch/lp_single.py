"""Single-region deterministic dispatch implementation using a merit order."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from io_loader import Frames

from .interface import DispatchResult

HOURS_PER_YEAR: float = 8760.0
_DEFAULT_REGION: str = "default"
_DISPATCH_TOLERANCE: float = 1e-9

_REQUIRED_COLUMNS = {
    "unit_id",
    "cap_mw",
    "availability",
    "hr_mmbtu_per_mwh",
    "vom_per_mwh",
    "fuel_price_per_mmbtu",
    "ef_ton_per_mwh",
}


def _validate_units_df(units_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the units data with numeric columns enforced."""

    if not isinstance(units_df, pd.DataFrame):
        raise TypeError("units must be provided as a pandas DataFrame")

    missing = [column for column in _REQUIRED_COLUMNS if column not in units_df.columns]
    if missing:
        raise ValueError(f"units data is missing required columns: {missing}")

    cleaned = units_df.copy(deep=True)

    if cleaned["unit_id"].duplicated().any():
        raise ValueError("unit_id values must be unique for dispatch")

    numeric_cols = [
        "cap_mw",
        "availability",
        "hr_mmbtu_per_mwh",
        "vom_per_mwh",
        "fuel_price_per_mmbtu",
        "ef_ton_per_mwh",
    ]

    for column in numeric_cols:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if column == "availability":
            cleaned[column] = cleaned[column].fillna(1.0)
        if cleaned[column].isna().any():
            raise ValueError(f"column '{column}' must contain numeric values")

    cleaned["availability"] = cleaned["availability"].clip(lower=0.0, upper=1.0)

    return cleaned


def _dispatch_merit_order(
    units_df: pd.DataFrame, load_mwh: float, allowance_cost: float
) -> dict:
    """Run the merit-order dispatch returning detailed information for testing."""

    units = _validate_units_df(units_df)

    load = max(0.0, float(load_mwh))
    allowance = float(allowance_cost)

    units = units.assign(
        cap_mwh=(units["cap_mw"] * units["availability"] * HOURS_PER_YEAR).clip(lower=0.0)
    )
    units = units.assign(
        marginal_cost=(
            units["vom_per_mwh"]
            + units["hr_mmbtu_per_mwh"] * units["fuel_price_per_mmbtu"]
            + units["ef_ton_per_mwh"] * allowance
        )
    )

    ordered = units.sort_values(["marginal_cost", "unit_id"]).set_index("unit_id")

    generation = pd.Series(0.0, index=ordered.index, dtype=float)
    remaining = load
    price = 0.0

    for unit_id, row in ordered.iterrows():
        if remaining <= _DISPATCH_TOLERANCE:
            break

        capacity = float(row["cap_mwh"])
        if capacity <= _DISPATCH_TOLERANCE:
            continue

        dispatch = min(capacity, remaining)
        generation.at[unit_id] = dispatch
        remaining -= dispatch

        if dispatch > _DISPATCH_TOLERANCE:
            price = float(row["marginal_cost"])

    remaining = float(max(0.0, remaining))
    total_generation = float(generation.sum())
    if total_generation <= _DISPATCH_TOLERANCE:
        price = 0.0

    emissions = float((generation * ordered["ef_ton_per_mwh"]).sum())

    return {
        "generation": generation,
        "units": ordered,
        "price": float(price),
        "emissions_tons": emissions,
        "shortfall_mwh": remaining,
    }


def _aggregate_generation_by_fuel(generation: pd.Series, units: pd.DataFrame) -> Mapping[str, float]:
    """Aggregate dispatch by fuel label if available, falling back to unit IDs."""

    tol_filtered = generation[generation > _DISPATCH_TOLERANCE]

    if tol_filtered.empty:
        return {}

    if "fuel" in units.columns:
        fuels = units.loc[tol_filtered.index, "fuel"]
        if fuels.isna().any():
            fallback = pd.Series(tol_filtered.index, index=tol_filtered.index)
            fuels = fuels.fillna(fallback)
        grouped = tol_filtered.groupby(fuels).sum()
    else:
        grouped = tol_filtered

    return {str(label): float(value) for label, value in grouped.items()}


def solve(
    year: int,
    allowance_cost: float,
    frames: Optional[Frames | Mapping[str, pd.DataFrame]] = None,
) -> DispatchResult:
    """Solve the single-region dispatch problem using the provided frame data."""

    if frames is None:
        raise ValueError("frames providing demand and units must be supplied")

    frames_obj = Frames.coerce(frames)
    units = frames_obj.units()
    demand = frames_obj.demand_for_year(year)
    load_value = sum(demand.values())

    dispatch = _dispatch_merit_order(units, float(load_value), allowance_cost)

    generation = dispatch["generation"]
    unit_data = dispatch["units"]
    gen_by_fuel = _aggregate_generation_by_fuel(generation, unit_data)

    region_prices = {_DEFAULT_REGION: float(dispatch["price"])}

    return DispatchResult(
        gen_by_fuel=gen_by_fuel,
        region_prices=region_prices,
        emissions_tons=float(dispatch["emissions_tons"]),
    )


__all__ = [
    "DispatchResult",
    "HOURS_PER_YEAR",
    "_aggregate_generation_by_fuel",
    "_dispatch_merit_order",
    "_validate_units_df",
    "solve",
]

