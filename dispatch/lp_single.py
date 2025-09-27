"""Single-region deterministic dispatch implementation using a merit order."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, cast

try:  # pragma: no cover - exercised when pandas missing
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(Any, None)

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


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before continuing."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for dispatch.lp_single; install it with `pip install pandas`."
        )


def _validate_units_df(units_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the units data with numeric columns enforced."""

    _ensure_pandas()

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
    units_df: pd.DataFrame,
    load_mwh: float,
    allowance_cost: float,
    *,
    allowance_covered: bool = True,
    carbon_price: float = 0.0,
) -> dict:
    """Run the merit-order dispatch returning detailed information for testing."""

    _ensure_pandas()

    units = _validate_units_df(units_df)

    load = max(0.0, float(load_mwh))
    allowance = float(allowance_cost)
    price_component = float(carbon_price)

    units = units.assign(
        cap_available_mw=(units["cap_mw"] * units["availability"]).clip(lower=0.0)
    )
    units = units.assign(
        cap_mwh=(units["cap_available_mw"] * HOURS_PER_YEAR).clip(lower=0.0)
    )
    units = units.assign(
        marginal_cost=(
            units["vom_per_mwh"]
            + units["hr_mmbtu_per_mwh"] * units["fuel_price_per_mmbtu"]
            + (units["ef_ton_per_mwh"] * allowance if allowance_covered else 0.0)
            + units["ef_ton_per_mwh"] * price_component
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

    _ensure_pandas()

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


def _aggregate_generation_by_region(
    generation: pd.Series, units: pd.DataFrame
) -> Mapping[str, float]:
    """Aggregate dispatch by region label if available."""

    _ensure_pandas()

    tol_filtered = generation[generation > _DISPATCH_TOLERANCE]
    if tol_filtered.empty:
        return {}

    if "region" in units.columns:
        regions = units.loc[tol_filtered.index, "region"]
        if regions.isna().any():
            fallback = pd.Series(_DEFAULT_REGION, index=tol_filtered.index)
            regions = regions.fillna(fallback)
        regions = regions.astype(str)
    else:
        regions = pd.Series(_DEFAULT_REGION, index=tol_filtered.index)

    grouped = tol_filtered.groupby(regions).sum()
    return {str(label): float(value) for label, value in grouped.items()}


def solve(
    year: int,
    allowance_cost: float,
    *,
    frames: Optional[Frames | Mapping[str, pd.DataFrame]] = None,
    carbon_price: float = 0.0,
) -> DispatchResult:
    """Solve the single-region dispatch problem using the provided frame data."""

    _ensure_pandas()

    if frames is None:
        raise ValueError("frames providing demand and units must be supplied")

    frames_obj = Frames.coerce(frames)
    units = frames_obj.units()
    demand = frames_obj.demand_for_year(year)
    coverage_map = frames_obj.coverage_for_year(year)

    if "region" in units.columns and not units["region"].isna().all():
        unit_regions = {str(region) for region in units["region"].unique()}
    else:
        unit_regions = {_DEFAULT_REGION}

    coverage_flags = {region: bool(coverage_map.get(region, True)) for region in unit_regions}
    allowance_covered = True
    if coverage_flags:
        unique_flags = set(coverage_flags.values())
        if len(unique_flags) > 1:
            raise ValueError('single-region dispatch requires uniform coverage status')
        allowance_covered = unique_flags.pop()

    load_value = sum(demand.values())

    dispatch = _dispatch_merit_order(
        units,
        float(load_value),
        allowance_cost,
        allowance_covered=allowance_covered,
        carbon_price=carbon_price,
    )

    generation = dispatch["generation"]
    unit_data = dispatch["units"]
    gen_by_fuel = _aggregate_generation_by_fuel(generation, unit_data)
    gen_by_region = _aggregate_generation_by_region(generation, unit_data)

    region_prices = {_DEFAULT_REGION: float(dispatch["price"])}

    # emissions by region (codex branch)
    emissions_series = generation * unit_data["ef_ton_per_mwh"]
    emissions_by_region_series = emissions_series.groupby(unit_data["region"]).sum()
    emissions_by_region = {
        str(region): float(value) for region, value in emissions_by_region_series.items()
    }
    if not emissions_by_region:
        emissions_by_region = {_DEFAULT_REGION: 0.0}

    demand_regions = {str(region) for region in demand.keys()}
    for region in demand_regions:
        emissions_by_region.setdefault(region, 0.0)

    generation_by_unit = {str(unit): float(output) for unit, output in generation.items()}
    capacity_mwh_by_unit = {
        str(unit): float(unit_data.loc[unit, "cap_mwh"]) for unit in unit_data.index
    }
    capacity_mw_by_unit = {
        str(unit): float(unit_data.loc[unit, "cap_available_mw"]) for unit in unit_data.index
    }

    capacity_mwh_by_fuel: Dict[str, float] = {}
    capacity_mw_by_fuel: Dict[str, float] = {}
    emissions_by_fuel: Dict[str, float] = {}
    variable_cost_by_fuel: Dict[str, float] = {}
    allowance_cost_by_fuel: Dict[str, float] = {}
    carbon_price_cost_by_fuel: Dict[str, float] = {}
    total_cost_by_fuel: Dict[str, float] = {}

    allowance_component = float(allowance_cost) if allowance_covered else 0.0
    carbon_component = float(carbon_price)

    for unit in unit_data.index:
        row = unit_data.loc[unit]
        fuel = str(row.get("fuel", unit))
        capacity_mwh = float(row["cap_mwh"])
        capacity_mw = float(row["cap_available_mw"])
        capacity_mwh_by_fuel[fuel] = capacity_mwh_by_fuel.get(fuel, 0.0) + capacity_mwh
        capacity_mw_by_fuel[fuel] = capacity_mw_by_fuel.get(fuel, 0.0) + capacity_mw

        dispatched = float(generation.get(unit, 0.0))
        emission_rate = float(row["ef_ton_per_mwh"])
        emissions_value = emission_rate * dispatched
        emissions_by_fuel[fuel] = emissions_by_fuel.get(fuel, 0.0) + emissions_value

        variable_rate = float(row["vom_per_mwh"]) + float(row["hr_mmbtu_per_mwh"]) * float(
            row["fuel_price_per_mmbtu"]
        )
        allowance_rate = emission_rate * allowance_component
        carbon_price_rate = emission_rate * carbon_component
        total_rate = variable_rate + allowance_rate + carbon_price_rate

        variable_cost_by_fuel[fuel] = (
            variable_cost_by_fuel.get(fuel, 0.0) + variable_rate * dispatched
        )
        allowance_cost_by_fuel[fuel] = (
            allowance_cost_by_fuel.get(fuel, 0.0) + allowance_rate * dispatched
        )
        carbon_price_cost_by_fuel[fuel] = (
            carbon_price_cost_by_fuel.get(fuel, 0.0) + carbon_price_rate * dispatched
        )
        total_cost_by_fuel[fuel] = total_cost_by_fuel.get(fuel, 0.0) + total_rate * dispatched

    # coverage and imports/exports (main branch)
    total_generation = float(generation.sum())
    generation_by_coverage = {"covered": 0.0, "non_covered": 0.0}
    coverage_key = "covered" if allowance_covered else "non_covered"
    generation_by_coverage[coverage_key] = total_generation

    imports_to_covered = 0.0
    exports_from_covered = 0.0
    region_coverage: Dict[str, bool] = {}
    for region, load in demand.items():
        region_str = str(region)
        covered = bool(coverage_map.get(region_str, allowance_covered))
        region_coverage[region_str] = covered
        generation_region = gen_by_region.get(region_str, 0.0)
        net_import = load - generation_region
        if covered:
            if net_import > _DISPATCH_TOLERANCE:
                imports_to_covered += net_import
            elif net_import < -_DISPATCH_TOLERANCE:
                exports_from_covered += -net_import

    return DispatchResult(
        gen_by_fuel=gen_by_fuel,
        region_prices=region_prices,
        emissions_tons=float(dispatch["emissions_tons"]),
        emissions_by_region=emissions_by_region,
        flows={},  # no transmission flows tracked in this solver path
        emissions_by_fuel=emissions_by_fuel,
        capacity_mwh_by_fuel=capacity_mwh_by_fuel,
        capacity_mw_by_fuel=capacity_mw_by_fuel,
        generation_by_unit=generation_by_unit,
        capacity_mwh_by_unit=capacity_mwh_by_unit,
        capacity_mw_by_unit=capacity_mw_by_unit,
        variable_cost_by_fuel=variable_cost_by_fuel,
        allowance_cost_by_fuel=allowance_cost_by_fuel,
        carbon_price_cost_by_fuel=carbon_price_cost_by_fuel,
        total_cost_by_fuel=total_cost_by_fuel,
        generation_by_region=gen_by_region,
        generation_by_coverage=generation_by_coverage,
        imports_to_covered=imports_to_covered,
        exports_from_covered=exports_from_covered,
        region_coverage=region_coverage,
    )


__all__ = [
    "DispatchResult",
    "HOURS_PER_YEAR",
    "_aggregate_generation_by_fuel",
    "_aggregate_generation_by_region",
    "_dispatch_merit_order",
    "_validate_units_df",
    "solve",
]

