"""Utility helpers for auditing and triggering capacity expansion builds."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List

import pandas as pd

_TOL = 1e-6
HOURS_PER_YEAR = 8760.0


@dataclass
class PlannedBuild:
    """Container tracking a single capacity expansion decision."""

    candidate_id: str
    unit_id: str
    capacity_mw: float
    reason: str
    npv_positive: bool


def _capital_recovery_factor(rate: float, lifetime: float) -> float:
    """Return the capital recovery factor for ``rate`` and ``lifetime``."""

    lifetime = max(float(lifetime), 1.0)
    if rate <= 0.0:
        return 1.0 / lifetime
    ratio = (1.0 + rate) ** lifetime
    return rate * ratio / (ratio - 1.0)


def _effective_cost(
    row: pd.Series,
    *,
    discount_rate: float,
    allowance_cost: float,
    carbon_price: float,
) -> float:
    """Return the levelized cost of energy for ``row`` including carbon effects."""

    availability = max(float(row.availability), 0.0)
    cap_mw = max(float(row.cap_mw), 0.0)
    if availability <= 0.0 or cap_mw <= 0.0:
        return float("inf")

    crf = _capital_recovery_factor(discount_rate, float(row.lifetime_years))
    annual_capex = float(row.capex_per_mw) * cap_mw * crf
    annual_fixed = float(row.fixed_om_per_mw) * cap_mw
    expected_mwh = cap_mw * availability * HOURS_PER_YEAR
    fixed_component = (annual_capex + annual_fixed) / max(expected_mwh, _TOL)

    variable_component = float(row.vom_per_mwh) + float(row.hr_mmbtu_per_mwh) * float(
        row.fuel_price_per_mmbtu
    )
    carbon_component = float(row.ef_ton_per_mwh) * (allowance_cost + carbon_price)

    return fixed_component + variable_component + carbon_component


def _build_unit_record(row: pd.Series, unit_id: str, capacity_mw: float) -> Dict[str, float]:
    """Return a dictionary describing a new dispatch unit from ``row``."""

    return {
        "unit_id": unit_id,
        "region": str(row.region),
        "fuel": str(row.fuel),
        "cap_mw": float(capacity_mw),
        "availability": float(row.availability),
        "hr_mmbtu_per_mwh": float(row.hr_mmbtu_per_mwh),
        "vom_per_mwh": float(row.vom_per_mwh),
        "fuel_price_per_mmbtu": float(row.fuel_price_per_mmbtu),
        "ef_ton_per_mwh": float(row.ef_ton_per_mwh),
    }


def _expensive_generation(
    generation: pd.Series, unit_costs: pd.Series, threshold: float
) -> float:
    """Return total generation from units with cost above ``threshold``."""

    mask = unit_costs > threshold + _TOL
    if mask.any():
        return float(generation[mask].sum())
    return 0.0


def _create_log_entry(
    record: PlannedBuild,
    row: pd.Series,
    generation_mwh: float,
) -> Dict[str, object]:
    """Return a structured log entry for ``record`` using ``row`` metadata."""

    capacity_mw = float(record.capacity_mw)
    capex_total = float(row.capex_per_mw) * capacity_mw
    fixed_om = float(row.fixed_om_per_mw) * capacity_mw
    variable_rate = float(row.vom_per_mwh) + float(row.hr_mmbtu_per_mwh) * float(
        row.fuel_price_per_mmbtu
    )
    variable_cost = variable_rate * float(generation_mwh)
    emissions = float(row.ef_ton_per_mwh) * float(generation_mwh)

    return {
        "candidate": str(record.candidate_id),
        "unit_id": str(record.unit_id),
        "capacity_mw": capacity_mw,
        "generation_mwh": float(generation_mwh),
        "reason": record.reason,
        "npv_positive": bool(record.npv_positive),
        "capex_cost": capex_total,
        "opex_cost": fixed_om + variable_cost,
        "emissions_tons": emissions,
    }


def plan_capacity_expansion(
    base_units: pd.DataFrame,
    candidates: pd.DataFrame,
    base_summary: Dict[str, object],
    dispatch_solver: Callable[[pd.DataFrame], Dict[str, object]],
    *,
    allowance_cost: float,
    carbon_price: float,
    discount_rate: float,
) -> tuple[pd.DataFrame, Dict[str, object], List[Dict[str, object]]]:
    """Plan expansion decisions returning updated units, summary, and log entries."""

    if candidates.empty:
        return base_units, base_summary, []

    current_units = base_units.copy(deep=True)
    summary = dict(base_summary)
    records: List[PlannedBuild] = []
    used: Dict[int, float] = {idx: 0.0 for idx in range(len(candidates))}

    candidates_sorted = candidates.reset_index(drop=True)
    order = list(candidates_sorted.index)

    # Shortage-driven builds -------------------------------------------------
    shortfall = float(summary.get("shortfall_mwh", 0.0) or 0.0)
    if shortfall > _TOL:
        for idx in order:
            if shortfall <= _TOL:
                break
            row = candidates_sorted.loc[idx]
            max_builds = max(float(row.get("max_builds", 1.0)), 0.0)
            remaining = max_builds - used[idx]
            if remaining <= _TOL:
                continue

            availability = max(float(row.availability), 0.0)
            cap_mw = max(float(row.cap_mw), 0.0)
            if availability <= 0.0 or cap_mw <= 0.0:
                continue

            block_mwh = cap_mw * availability * HOURS_PER_YEAR
            if block_mwh <= _TOL:
                continue

            builds_needed = math.ceil((shortfall - _TOL) / block_mwh)
            builds_to_use = int(min(builds_needed, math.floor(remaining + _TOL)))
            if builds_to_use <= 0:
                continue

            for build_no in range(builds_to_use):
                unit_id = f"{row.unit_id}_build{int(used[idx] + build_no + 1)}"
                current_units = pd.concat(
                    [current_units, pd.DataFrame([_build_unit_record(row, unit_id, cap_mw)])],
                    ignore_index=True,
                )
                records.append(
                    PlannedBuild(
                        candidate_id=str(row.unit_id),
                        unit_id=unit_id,
                        capacity_mw=cap_mw,
                        reason="supply_shortage",
                        npv_positive=_effective_cost(
                            row,
                            discount_rate=discount_rate,
                            allowance_cost=allowance_cost,
                            carbon_price=carbon_price,
                        )
                        < float(summary.get("price", 0.0)) - _TOL,
                    )
                )
            used[idx] += float(builds_to_use)
            shortfall = max(0.0, shortfall - builds_to_use * block_mwh)

        summary = dispatch_solver(current_units)

    # Positive-NPV builds ----------------------------------------------------
    while True:
        price = float(summary.get("price", 0.0) or 0.0)
        generation = summary.get("generation")
        unit_costs = summary.get("units")
        if not isinstance(generation, pd.Series) or not isinstance(unit_costs, pd.DataFrame):
            break
        unit_cost_series = unit_costs["marginal_cost"]
        built_any = False

        for idx in order:
            row = candidates_sorted.loc[idx]
            max_builds = max(float(row.get("max_builds", 1.0)), 0.0)
            remaining = max_builds - used[idx]
            if remaining <= _TOL:
                continue

            effective_cost = _effective_cost(
                row,
                discount_rate=discount_rate,
                allowance_cost=allowance_cost,
                carbon_price=carbon_price,
            )
            if effective_cost >= price - _TOL:
                continue

            expensive = _expensive_generation(generation, unit_cost_series, effective_cost)
            if expensive <= _TOL:
                continue

            availability = max(float(row.availability), 0.0)
            cap_mw = max(float(row.cap_mw), 0.0)
            if availability <= 0.0 or cap_mw <= 0.0:
                continue

            block_capacity = cap_mw * availability * HOURS_PER_YEAR
            max_replacable = remaining * block_capacity
            replace_mwh = min(expensive, max_replacable)
            if replace_mwh <= _TOL:
                continue

            capacity_needed = replace_mwh / (availability * HOURS_PER_YEAR)
            capacity_mw = min(cap_mw, capacity_needed)
            if capacity_mw <= _TOL:
                continue

            build_index = int(used[idx] + 1)
            unit_id = f"{row.unit_id}_build{build_index}"
            current_units = pd.concat(
                [
                    current_units,
                    pd.DataFrame([_build_unit_record(row, unit_id, capacity_mw)]),
                ],
                ignore_index=True,
            )
            records.append(
                PlannedBuild(
                    candidate_id=str(row.unit_id),
                    unit_id=unit_id,
                    capacity_mw=capacity_mw,
                    reason="npv_positive",
                    npv_positive=True,
                )
            )
            used[idx] += 1.0
            summary = dispatch_solver(current_units)
            built_any = True
            break

        if not built_any:
            break

    final_generation = summary.get("generation")
    build_log: List[Dict[str, object]] = []
    if isinstance(final_generation, pd.Series):
        for record in records:
            try:
                row_idx = candidates_sorted.index[candidates_sorted["unit_id"] == record.candidate_id][0]
            except IndexError:
                continue
            row = candidates_sorted.loc[row_idx]
            generation_mwh = float(final_generation.get(record.unit_id, 0.0))
            build_log.append(_create_log_entry(record, row, generation_mwh))

    return current_units, summary, build_log


__all__ = ["plan_capacity_expansion"]

