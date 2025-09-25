"""Annual fixed-point integration between dispatch and allowance market."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

from dispatch.interface import DispatchResult
from dispatch.lp_network import solve_from_frames as solve_network_from_frames
from dispatch.lp_single import solve as solve_single
from engine.outputs import EngineOutputs
from policy.allowance_annual import (
    RGGIPolicyAnnual,
    allowance_initial_state,
    clear_year as allowance_clear_year,
    finalize_period_if_needed as allowance_finalize_period,
)
from policy.allowance_supply import AllowanceSupply

from io_loader import Frames


LOGGER = logging.getLogger(__name__)


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before running the engine."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for engine.run_loop; install it with `pip install pandas`."
        )


def effective_carbon_price(
    allowance_price: float, exogenous_price: float, deep: bool
) -> float:
    """Return the effective marginal carbon price based on configuration."""

    try:
        allowance_component = float(allowance_price)
    except (TypeError, ValueError):
        allowance_component = 0.0
    try:
        exogenous_component = float(exogenous_price)
    except (TypeError, ValueError):
        exogenous_component = 0.0

    if deep:
        return allowance_component + exogenous_component
    return max(allowance_component, exogenous_component)


def _coerce_years(policy: Any, years: Iterable[int] | None) -> list[Any]:
    if years is None:
        series_index = getattr(policy.cap, 'index', [])
        return list(series_index) if series_index is not None else []

    requested = list(years)
    index_obj = getattr(policy.cap, 'index', None)
    series_index = list(index_obj) if index_obj is not None else []
    index_set = set(series_index)

    mapper = getattr(policy, 'compliance_year_for', None)
    selected: list[Any] = []

    for entry in requested:
        if entry in index_set:
            selected.append(entry)
            continue
        target_year = None
        if callable(mapper):
            try:
                target_year = mapper(entry)
            except Exception:  # pragma: no cover - defensive guard
                target_year = None
        if target_year is None:
            try:
                target_year = int(entry)
            except (TypeError, ValueError):
                target_year = None
        if target_year is None:
            continue
        for label in series_index:
            if callable(mapper):
                try:
                    label_year = mapper(label)
                except Exception:  # pragma: no cover - defensive guard
                    continue
            else:
                try:
                    label_year = int(label)
                except (TypeError, ValueError):
                    continue
            if label_year == target_year:
                selected.append(label)

    if not selected:
        return []

    seen: set[Any] = set()
    ordered: list[Any] = []
    for label in selected:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _normalize_progress_year(value: object) -> object:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _compute_period_weights(policy: Any, periods: Sequence[Any]) -> dict[object, float]:
    """Return fractional weights for ``periods`` grouped by compliance year."""

    mapper = getattr(policy, "compliance_year_for", None)
    counts: defaultdict[int, int] = defaultdict(int)
    period_to_year: dict[Any, int] = {}

    for period in periods:
        calendar_year: int | None = None
        if callable(mapper):
            try:
                mapped = mapper(period)
            except Exception:  # pragma: no cover - defensive guard
                mapped = None
            if mapped is not None:
                try:
                    calendar_year = int(mapped)
                except (TypeError, ValueError):
                    calendar_year = None
        if calendar_year is None:
            try:
                calendar_year = int(period)
            except (TypeError, ValueError):
                calendar_year = None
        if calendar_year is None:
            continue
        period_to_year[period] = calendar_year
        counts[calendar_year] += 1

    weights: dict[object, float] = {}
    for period in periods:
        calendar_year = period_to_year.get(period)
        if calendar_year is None:
            weight = 1.0
        else:
            count = counts.get(calendar_year, 1)
            weight = 1.0 / count if count > 0 else 1.0
        weights[period] = weight
        normalized = _normalize_progress_year(period)
        weights[normalized] = weight
        if calendar_year is not None:
            weights[calendar_year] = weight

    return weights


def _initial_price_for_year(price_initial: float | Mapping[int, float], year: int, fallback: float) -> float:
    if isinstance(price_initial, Mapping):
        if year in price_initial:
            return float(price_initial[year])
        if price_initial:
            return float(next(iter(price_initial.values())))
        return float(fallback)
    return float(price_initial)


def _extract_emissions(dispatch_output: object) -> float:
    """Return the emissions ton value from a dispatch output."""

    attr = getattr(dispatch_output, 'emissions_tons', None)
    if attr is not None:
        return float(attr)
    if isinstance(dispatch_output, Mapping) and 'emissions_tons' in dispatch_output:
        return float(dispatch_output['emissions_tons'])
    return float(dispatch_output)


def _policy_value(series: Any, year: int, default: float = 0.0) -> float:
    """Return the policy value for ``year`` falling back to ``default``."""

    if series is None:
        return float(default)
    getter = getattr(series, 'get', None)
    if callable(getter):
        raw = getter(year, default)
    else:
        try:
            raw = series.loc[year]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive guard
            raw = default
    if raw is None:
        return float(default)
    if pd is not None:
        try:
            if pd.isna(raw):  # type: ignore[arg-type]
                return float(default)
        except AttributeError:  # pragma: no cover - defensive guard
            pass
    try:
        return float(raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return float(default)


def _policy_record_for_year(policy: Any, year: int) -> dict[str, float | bool]:
    """Return a mapping of scalar policy values for ``year``."""

    record = {
        'cap': _policy_value(getattr(policy, 'cap', None), year),
        'floor': _policy_value(getattr(policy, 'floor', None), year),
        'ccr1_trigger': _policy_value(getattr(policy, 'ccr1_trigger', None), year),
        'ccr1_qty': _policy_value(getattr(policy, 'ccr1_qty', None), year),
        'ccr2_trigger': _policy_value(getattr(policy, 'ccr2_trigger', None), year),
        'ccr2_qty': _policy_value(getattr(policy, 'ccr2_qty', None), year),
        'enabled': bool(getattr(policy, 'enabled', True)),
        'ccr1_enabled': bool(getattr(policy, 'ccr1_enabled', True)),
        'ccr2_enabled': bool(getattr(policy, 'ccr2_enabled', True)),
    }

    cp_series = getattr(policy, 'cp_id', None)
    cp_value: str | None = None
    if cp_series is not None:
        getter = getattr(cp_series, 'get', None)
        if callable(getter):
            cp_value = getter(year, None)
        else:
            try:
                cp_value = cp_series.loc[year]  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive
                cp_value = None
    record['cp_id'] = str(cp_value) if cp_value is not None else 'NoPolicy'

    surrender_frac = getattr(policy, 'annual_surrender_frac', 0.5)
    try:
        record['annual_surrender_frac'] = float(surrender_frac)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        record['annual_surrender_frac'] = 0.5

    carry_pct = getattr(policy, 'carry_pct', 1.0)
    try:
        record['carry_pct'] = float(carry_pct)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        record['carry_pct'] = 1.0

    record['bank_enabled'] = bool(getattr(policy, 'banking_enabled', True))

    full_compliance_years = set(getattr(policy, 'full_compliance_years', set()))
    if year in full_compliance_years:
        record['full_compliance'] = True
    else:
        try:
            numeric_year = int(year)
        except (TypeError, ValueError):
            numeric_year = None
        if numeric_year is not None and numeric_year in full_compliance_years:
            record['full_compliance'] = True
        else:
            mapper = getattr(policy, 'compliance_year_for', None)
            if callable(mapper):
                try:
                    compliance_year = mapper(year)
                except Exception:  # pragma: no cover - defensive guard
                    compliance_year = None
                record['full_compliance'] = bool(
                    compliance_year is not None and compliance_year in full_compliance_years
                )
            else:
                record['full_compliance'] = False
    return record


def _build_allowance_supply(
    policy: Any,
    year: int,
    *,
    enable_floor: bool,
    enable_ccr: bool,
) -> tuple[AllowanceSupply, dict[str, float | bool]]:
    """Construct :class:`AllowanceSupply` for ``year`` along with the raw record."""

    record = _policy_record_for_year(policy, year)
    ccr1_enabled = bool(record.get('ccr1_enabled', True))
    ccr2_enabled = bool(record.get('ccr2_enabled', True))
    ccr1_qty = float(record.get('ccr1_qty', 0.0)) if ccr1_enabled else 0.0
    ccr2_qty = float(record.get('ccr2_qty', 0.0)) if ccr2_enabled else 0.0
    supply = AllowanceSupply(
        cap=float(record.get('cap', 0.0)),
        floor=float(record.get('floor', 0.0)),
        ccr1_trigger=float(record.get('ccr1_trigger', 0.0)),
        ccr1_qty=ccr1_qty,
        ccr2_trigger=float(record.get('ccr2_trigger', 0.0)),
        ccr2_qty=ccr2_qty,
        enabled=bool(record.get('enabled', True)),
        enable_floor=enable_floor,
        enable_ccr=enable_ccr and (ccr1_qty > 0.0 or ccr2_qty > 0.0),
    )
    record['ccr1_qty'] = ccr1_qty
    record['ccr2_qty'] = ccr2_qty
    return supply, record


def _solve_allowance_market_year(
    dispatch_solver: Callable[[int, float, float], object],
    year: int,
    supply: AllowanceSupply,
    bank_prev: float,
    outstanding_prev: float,
    *,
    policy_enabled: bool,
    high_price: float,
    tol: float,
    max_iter: int,
    annual_surrender_frac: float,
    carry_pct: float,
    banking_enabled: bool,
    carbon_price: float = 0.0,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, object]:
    """Solve for the allowance clearing price for ``year`` using bisection.

    When provided, ``progress_cb`` receives updates for each iteration using the
    signature ``progress_cb(stage, payload)`` where ``stage`` is the literal
    string ``"iteration"`` and ``payload`` includes the ``year``, current
    iteration number, and clearing price estimates.
    """

    tol = max(float(tol), 0.0)
    max_iter_int = max(int(max_iter), 0)
    high_price = float(high_price)

    banking_enabled = bool(banking_enabled)

    def _report_progress(
        stage: str,
        iteration: int,
        price: float | None,
        *,
        status: str,
        shortage: bool | None = None,
    ) -> None:
        if progress_cb is None:
            return
        payload: dict[str, object] = {
            "year": _normalize_progress_year(year),
            "iteration": int(iteration),
            "status": status,
        }
        if price is not None:
            payload["price"] = float(price)
        if shortage is not None:
            payload["shortage"] = bool(shortage)
        progress_cb(stage, payload)

    def _issued_quantities(price: float, allowances: float) -> tuple[float, float]:
        if not supply.enable_ccr or not supply.enabled:
            return 0.0, 0.0
        remaining = max(allowances - float(supply.cap), 0.0)
        if remaining <= 0.0:
            return 0.0, 0.0
        issued1 = 0.0
        issued2 = 0.0
        if price >= supply.ccr1_trigger and supply.ccr1_qty > 0.0:
            issued1 = min(float(supply.ccr1_qty), remaining)
            remaining -= issued1
        if price >= supply.ccr2_trigger and supply.ccr2_qty > 0.0:
            issued2 = min(float(supply.ccr2_qty), remaining)
        return issued1, issued2

    bank_prev = max(0.0, float(bank_prev))
    if not banking_enabled:
        bank_prev = 0.0
    outstanding_prev = max(0.0, float(outstanding_prev))
    carry_pct = max(0.0, float(carry_pct)) if banking_enabled else 0.0
    surrender_frac = max(0.0, min(1.0, float(annual_surrender_frac)))

    def _finalize(summary: dict[str, object]) -> dict[str, object]:
        if banking_enabled:
            return summary
        adjusted = dict(summary)
        adjusted['bank_prev'] = 0.0
        adjusted['bank_unadjusted'] = 0.0
        adjusted['bank_new'] = 0.0
        allowances_total = adjusted.get('allowances_total')
        if allowances_total is None:
            allowances_total = adjusted.get('available_allowances', 0.0)
        adjusted['allowances_total'] = float(allowances_total)
        finalize_section = dict(adjusted.get('finalize', {}))
        finalize_section['bank_final'] = 0.0
        adjusted['finalize'] = finalize_section
        return adjusted

    if not policy_enabled or not supply.enabled:
        clearing_price = 0.0
        dispatch_result = dispatch_solver(
            year, clearing_price, carbon_price=carbon_price
        )
        emissions = _extract_emissions(dispatch_result)
        allowances = max(supply.available_allowances(clearing_price), emissions)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances)
        minted_allowances = float(max(allowances, emissions))
        total_allowances = minted_allowances  # bank is zero when policy disabled
        _report_progress(
            "iteration",
            0,
            clearing_price,
            status="policy-disabled",
            shortage=False,
        )
        return {
            'year': year,
            'p_co2': float(clearing_price),
            'available_allowances': minted_allowances,
            'allowances_total': total_allowances,
            'bank_prev': 0.0,
            'bank_unadjusted': 0.0,
            'bank_new': 0.0,
            'surrendered': 0.0,
            'obligation_new': 0.0,
            'shortage_flag': False,
            'iterations': 0,
            'emissions': float(emissions),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {
                'finalized': False,
                'bank_final': 0.0,
                'remaining_obligation': 0.0,
                'surrendered_additional': 0.0,
            },
            '_dispatch_result': dispatch_result,
        }

    min_price = supply.floor if supply.enable_floor and supply.enabled else 0.0
    low = max(0.0, float(min_price))
    high = max(low, high_price if high_price > 0.0 else low)

    dispatch_low = dispatch_solver(year, low, carbon_price=carbon_price)
    emissions_low = _extract_emissions(dispatch_low)
    allowances_low = supply.available_allowances(low)

    total_allowances_low = bank_prev + allowances_low
    if total_allowances_low >= emissions_low:
        clearing_price = supply.enforce_floor(low)
        if clearing_price != low:
            dispatch_low = dispatch_solver(
                year, clearing_price, carbon_price=carbon_price
            )
            emissions_low = _extract_emissions(dispatch_low)
            allowances_low = supply.available_allowances(clearing_price)
            total_allowances_low = bank_prev + allowances_low
        surrendered = min(surrender_frac * emissions_low, total_allowances_low)
        bank_unadjusted = max(total_allowances_low - surrendered, 0.0)
        obligation = max(outstanding_prev + emissions_low - surrendered, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_low)
        bank_carry = max(bank_unadjusted * carry_pct, 0.0)
        _report_progress(
            "iteration",
            0,
            clearing_price,
            status="surplus",
            shortage=False,
        )
        result = {
            'year': year,
            'p_co2': float(clearing_price),
            'available_allowances': float(allowances_low),
            'allowances_total': float(total_allowances_low),
            'bank_prev': float(bank_prev),
            'bank_unadjusted': float(bank_unadjusted),
            'bank_new': float(bank_carry),
            'surrendered': float(surrendered),
            'obligation_new': float(obligation),
            'shortage_flag': False,
            'iterations': 0,
            'emissions': float(emissions_low),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {
                'finalized': False,
                'bank_final': float(bank_carry),
                'remaining_obligation': float(obligation),
                'surrendered_additional': 0.0,
            },
            '_dispatch_result': dispatch_low,
        }
        return _finalize(result)

    dispatch_high = dispatch_solver(year, high, carbon_price=carbon_price)
    emissions_high = _extract_emissions(dispatch_high)
    allowances_high = supply.available_allowances(high)

    if bank_prev + allowances_high < emissions_high - tol:
        clearing_price = supply.enforce_floor(high)
        if clearing_price != high:
            dispatch_high = dispatch_solver(
                year, clearing_price, carbon_price=carbon_price
            )
            emissions_high = _extract_emissions(dispatch_high)
            allowances_high = supply.available_allowances(clearing_price)
        total_allowances_high = bank_prev + allowances_high
        surrendered = min(surrender_frac * emissions_high, total_allowances_high)
        bank_unadjusted = max(total_allowances_high - surrendered, 0.0)
        obligation = max(outstanding_prev + emissions_high - surrendered, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_high)
        bank_carry = max(bank_unadjusted * carry_pct, 0.0)
        _report_progress(
            "iteration",
            max_iter_int,
            clearing_price,
            status="shortage",
            shortage=True,
        )
        result = {
            'year': year,
            'p_co2': float(clearing_price),
            'available_allowances': float(allowances_high),
            'allowances_total': float(total_allowances_high),
            'bank_prev': float(bank_prev),
            'bank_unadjusted': float(bank_unadjusted),
            'bank_new': float(bank_carry),
            'surrendered': float(surrendered),
            'obligation_new': float(obligation),
            'shortage_flag': True,
            'iterations': max_iter_int,
            'emissions': float(emissions_high),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {
                'finalized': False,
                'bank_final': float(bank_carry),
                'remaining_obligation': float(obligation),
                'surrendered_additional': 0.0,
            },
            '_dispatch_result': dispatch_high,
        }
        return _finalize(result)

    best_price = high
    best_allowances = allowances_high
    best_emissions = emissions_high
    best_dispatch = dispatch_high
    iteration_count = 0

    low_bound = low
    high_bound = high

    for iteration in range(1, max_iter_int + 1):
        mid = 0.5 * (low_bound + high_bound)
        dispatch_mid = dispatch_solver(year, mid, carbon_price=carbon_price)
        emissions_mid = _extract_emissions(dispatch_mid)
        allowances_mid = supply.available_allowances(mid)
        total_allowances_mid = bank_prev + allowances_mid
        iteration_count = iteration
        shortage_mid = total_allowances_mid < emissions_mid
        _report_progress(
            "iteration",
            iteration,
            mid,
            status="bisection",
            shortage=shortage_mid,
        )
        if total_allowances_mid >= emissions_mid:
            best_price = mid
            best_allowances = allowances_mid
            best_emissions = emissions_mid
            best_dispatch = dispatch_mid
            high_bound = mid
            if abs(allowances_mid - emissions_mid) <= tol:
                break
        else:
            low_bound = mid
        if abs(high_bound - low_bound) <= max(tol, 1e-6):
            break

    clearing_price = supply.enforce_floor(best_price)
    if clearing_price != best_price:
        best_dispatch = dispatch_solver(
            year, clearing_price, carbon_price=carbon_price
        )
        best_emissions = _extract_emissions(best_dispatch)
        best_allowances = supply.available_allowances(clearing_price)
    total_allowances = bank_prev + best_allowances

    surrendered = min(surrender_frac * best_emissions, total_allowances)
    bank_unadjusted = max(total_allowances - surrendered, 0.0)
    obligation = max(outstanding_prev + best_emissions - surrendered, 0.0)
    shortage_flag = best_emissions > total_allowances + tol
    ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, best_allowances)
    bank_carry = max(bank_unadjusted * carry_pct, 0.0)

    _report_progress(
        "iteration",
        iteration_count,
        clearing_price,
        status="final",
        shortage=shortage_flag,
    )

    result = {
        'year': year,
        'p_co2': float(clearing_price),
        'available_allowances': float(best_allowances),
        'allowances_total': float(total_allowances),
        'bank_prev': float(bank_prev),
        'bank_unadjusted': float(bank_unadjusted),
        'bank_new': float(bank_carry),
        'surrendered': float(surrendered),
        'obligation_new': float(obligation),
        'shortage_flag': bool(shortage_flag),
        'iterations': iteration_count,
        'emissions': float(best_emissions),
        'ccr1_issued': float(ccr1_issued),
        'ccr2_issued': float(ccr2_issued),
        'finalize': {
            'finalized': False,
            'bank_final': float(bank_carry),
            'remaining_obligation': float(obligation),
            'surrendered_additional': 0.0,
        },
        '_dispatch_result': best_dispatch,
    }
    return _finalize(result)


def run_annual_fixed_point(
    policy: RGGIPolicyAnnual,
    dispatch_model: Callable[[int, float], float],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
) -> dict[int, dict]:
    """Iterate annually to find a fixed-point between dispatch and allowance cost."""

    _ensure_pandas()

    if not callable(dispatch_model):
        raise TypeError('dispatch_model must be callable with signature (year, price) -> emissions')

    years_sequence = _coerce_years(policy, years)
    if not getattr(policy, 'enabled', True):
        results: dict[int, dict] = {}
        for year in years_sequence:
            emissions = _extract_emissions(dispatch_model(year, 0.0))
            record = {
                'year': year,
                'bank_prev': 0.0,
                'available_allowances': emissions,
                'p_co2': 0.0,
                'ccr1_issued': 0.0,
                'ccr2_issued': 0.0,
                'surrendered': 0.0,
                'bank_new': 0.0,
                'obligation_new': 0.0,
                'shortage_flag': False,
                'iterations': 0,
                'emissions': emissions,
                'finalize': {'finalized': False, 'bank_final': 0.0},
            }
            results[year] = record
        return results

    allowance_state = allowance_initial_state()
    results: dict[int, dict] = {}

    banking_enabled = bool(getattr(policy, 'banking_enabled', True))
    bank = float(policy.bank0) if banking_enabled else 0.0
    previous_price = float(price_initial) if not isinstance(price_initial, Mapping) else 0.0

    for year in years_sequence:
        bank_start = bank if banking_enabled else 0.0
        price_guess = _initial_price_for_year(price_initial, year, previous_price)
        iteration_count = 0
        emissions = 0.0
        market_record: dict[str, float | bool | str] | None = None
        accepted_state = None
        base_state_for_year = allowance_state

        for iteration in range(1, max_iter + 1):
            iteration_count = iteration
            dispatch_output = dispatch_model(year, price_guess)
            emissions = _extract_emissions(dispatch_output)
            trial_record, trial_state = allowance_clear_year(
                policy,
                base_state_for_year,
                year,
                emissions_tons=emissions,
                bank_prev=bank_start,
                expected_price_guess=price_guess,
            )
            market_record = trial_record
            cleared_price = float(trial_record['p_co2'])
            if abs(cleared_price - price_guess) <= tol:
                price_guess = cleared_price
                accepted_state = trial_state
                break
            price_guess = price_guess + relaxation * (cleared_price - price_guess)
        else:  # pragma: no cover - defensive guard
            raise RuntimeError(f'Allowance price failed to converge for year {year}')

        assert market_record is not None  # for type checker
        assert accepted_state is not None  # for type checker
        allowance_state = accepted_state
        market_result = dict(market_record)
        market_result['iterations'] = iteration_count
        market_result['emissions'] = emissions

        bank = float(market_result['bank_new'])
        if not banking_enabled:
            bank = 0.0
        finalize_summary, allowance_state = allowance_finalize_period(policy, allowance_state, year)
        market_result['finalize'] = finalize_summary
        if finalize_summary.get('finalized'):
            bank = float(finalize_summary.get('bank_final', bank))
            if not banking_enabled:
                bank = 0.0

        results[year] = market_result
        previous_price = price_guess

    return results


def _dispatch_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    use_network: bool = False,
    period_weights: Mapping[Any, float] | None = None,
    carbon_price_schedule: Mapping[int, float]
    | Mapping[str, Any]
    | float
    | None = None,
    deep_carbon_pricing: bool = False,
) -> Callable[[Any, float, float], object]:

    """Build a dispatch callback that solves using the frame container."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames, carbon_price_schedule=carbon_price_schedule)

    weights: dict[object, float] = {}
    if period_weights:
        for period, raw_weight in period_weights.items():
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError):
                continue
            if weight <= 0.0:
                continue
            weights[period] = weight
            normalized = _normalize_progress_year(period)
            weights[normalized] = weight

    schedule_lookup: dict[int | None, float] = {}

    def _ingest_schedule(payload: Mapping[Any, Any] | float | None) -> None:
        if payload is None:
            return
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                try:
                    year = int(key) if key is not None else None
                except (TypeError, ValueError):
                    continue
                try:
                    schedule_lookup[year] = float(value)
                except (TypeError, ValueError):
                    continue
            return
        try:
            schedule_lookup[None] = float(payload)
        except (TypeError, ValueError):
            return

    if hasattr(frames_obj, "carbon_price_schedule"):
        _ingest_schedule(getattr(frames_obj, "carbon_price_schedule", None))
    _ingest_schedule(carbon_price_schedule)

    def _price_for(period: Any) -> float:
        if not schedule_lookup:
            return 0.0
        try:
            year_key = int(period)
        except (TypeError, ValueError):
            year_key = None
        if year_key is not None and year_key in schedule_lookup:
            return float(schedule_lookup[year_key])
        normalized = _normalize_progress_year(period)
        if isinstance(normalized, int) and normalized in schedule_lookup:
            return float(schedule_lookup[normalized])
        if None in schedule_lookup:
            return float(schedule_lookup[None])
        return 0.0

    def _weight_for(period: Any) -> float:
        weight = weights.get(period)
        if weight is not None:
            return weight
        normalized = _normalize_progress_year(period)
        return float(weights.get(normalized, 1.0))

    def _scaled_frames(period: Any, weight: float) -> Frames:
        if abs(weight - 1.0) <= 1e-12:
            return frames_obj
        try:
            normalized_year = int(period)
        except (TypeError, ValueError):
            normalized_year = period
        demand_df = frames_obj.demand()
        if 'year' not in demand_df.columns:
            return frames_obj
        mask = demand_df['year'] == normalized_year
        if not mask.any():
            return frames_obj
        adjusted = demand_df.copy()
        adjusted.loc[mask, 'demand_mwh'] = (
            adjusted.loc[mask, 'demand_mwh'].astype(float) / weight
        )
        return frames_obj.with_frame('demand', adjusted)

    def _scale_result(result: object, weight: float) -> object:
        if abs(weight - 1.0) <= 1e-12:
            return result
        if isinstance(result, DispatchResult):
            scale = float(weight)
            gen_by_fuel = {key: float(value) * scale for key, value in result.gen_by_fuel.items()}
            emissions_by_region = {
                key: float(value) * scale for key, value in result.emissions_by_region.items()
            }
            flows = {key: float(value) * scale for key, value in result.flows.items()}
            generation_by_region = {
                key: float(value) * scale for key, value in result.generation_by_region.items()
            }
            generation_by_coverage = {
                key: float(value) * scale for key, value in result.generation_by_coverage.items()
            }
            return DispatchResult(
                gen_by_fuel=gen_by_fuel,
                region_prices=dict(result.region_prices),
                emissions_tons=float(result.emissions_tons) * scale,
                emissions_by_region=emissions_by_region,
                flows=flows,
                generation_by_region=generation_by_region,
                generation_by_coverage=generation_by_coverage,
                imports_to_covered=float(result.imports_to_covered) * scale,
                exports_from_covered=float(result.exports_from_covered) * scale,
                region_coverage=dict(result.region_coverage),
            )
        return result

    def _normalize_extra_price(value: float | None) -> float:
        if value in (None, ""):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def dispatch(year: Any, allowance_cost: float, carbon_price: float | None = None):
        weight = _weight_for(year)
        frames_for_year = _scaled_frames(year, weight)
        schedule_price = _price_for(year)
        extra_price = _normalize_extra_price(carbon_price)
        allowance_component = float(allowance_cost)
        exogenous_component = schedule_price + extra_price

        if deep_carbon_pricing:
            dispatch_allowance_cost = allowance_component
            dispatch_carbon_price = exogenous_component
        else:
            dispatch_allowance_cost = effective_carbon_price(
                allowance_component, exogenous_component, deep=False
            )
            dispatch_carbon_price = 0.0

        if use_network:
            raw_result = solve_network_from_frames(
                frames_for_year,
                year,
                dispatch_allowance_cost,
                carbon_price=dispatch_carbon_price,
            )
        else:
            raw_result = solve_single(
                year,
                dispatch_allowance_cost,
                frames=frames_for_year,
                carbon_price=dispatch_carbon_price,
            )

        return _scale_result(raw_result, weight)

    return dispatch


def run_fixed_point_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
    use_network: bool = False,
    carbon_price_schedule: Mapping[int, float]
    | Mapping[str, Any]
    | float
    | None = None,
    deep_carbon_pricing: bool = False,
) -> dict[int, dict]:
    """Run the annual fixed-point integration using in-memory frames."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames, carbon_price_schedule=carbon_price_schedule)
    policy_spec = frames_obj.policy()
    policy = policy_spec.to_policy()
    years_sequence = _coerce_years(policy, years)
    period_weights = _compute_period_weights(policy, years_sequence)
    dispatch_kwargs = dict(
        use_network=use_network,
        period_weights=period_weights,
        carbon_price_schedule=carbon_price_schedule,
    )
    if deep_carbon_pricing:
        dispatch_kwargs["deep_carbon_pricing"] = bool(deep_carbon_pricing)
    dispatch_solver = _dispatch_from_frames(frames_obj, **dispatch_kwargs)

    def dispatch_model(year: int, allowance_cost: float) -> float:
        return _extract_emissions(
            dispatch_solver(year, allowance_cost, carbon_price=0.0)
        )

    return run_annual_fixed_point(
        policy,
        dispatch_model,
        years=years,
        price_initial=price_initial,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
    )


def _build_engine_outputs(
    years: Sequence[Any],
    raw_results: Mapping[Any, Mapping[str, object]],
    dispatch_solver: Callable[..., object],
    policy: RGGIPolicyAnnual,
) -> EngineOutputs:
    """Convert fixed-point results into structured engine outputs."""

    _ensure_pandas()

    aggregated: dict[int, dict[str, object]] = {}

    for period in years:
        summary_raw = raw_results.get(period)
        if summary_raw is None:
            continue
        summary = dict(summary_raw)
        price = float(summary.get("p_co2", 0.0))
        dispatch_result = summary.pop("_dispatch_result", None)
        if dispatch_result is None:
            dispatch_result = dispatch_solver(period, price, carbon_price=0.0)
        emissions_total = float(summary.get("emissions", _extract_emissions(dispatch_result)))

        compliance_year = getattr(policy, "compliance_year_for", None)
        if callable(compliance_year):
            try:
                calendar_year = int(compliance_year(period))
            except Exception:  # pragma: no cover - defensive guard
                calendar_year = int(period)
        else:
            try:
                calendar_year = int(period)
            except (TypeError, ValueError):
                calendar_year = hash(period)

        entry = aggregated.setdefault(
            calendar_year,
            {
                "periods": [],
                "price_last": 0.0,
                "iterations_max": 0,
                "emissions_sum": 0.0,
                "available_allowances_sum": 0.0,
                "bank_prev_first": None,
                "bank_new_last": 0.0,
                "obligation_last": 0.0,
                "shortage_any": False,
                "finalize_last": {},
                "finalized": False,
                "bank_final": 0.0,
                "surrendered_sum": 0.0,
                "surrendered_extra": 0.0,
                "emissions_by_region": defaultdict(float),
                "price_by_region": {},
                "flows": defaultdict(float),
            },
        )

        entry["periods"].append(period)
        entry["price_last"] = price
        allowance_price_last = float(summary.get("p_co2_allowance", price))
        exogenous_price_last = float(summary.get("p_co2_exogenous", 0.0))
        effective_price_last = float(summary.get("p_co2_effective", price))
        entry["allowance_price_last"] = allowance_price_last
        entry["exogenous_price_last"] = exogenous_price_last
        entry["effective_price_last"] = effective_price_last
        iterations_value = summary.get("iterations", 0)
        try:
            iterations_int = int(iterations_value)
        except (TypeError, ValueError):
            iterations_int = 0
        entry["iterations_max"] = max(entry["iterations_max"], iterations_int)
        entry["emissions_sum"] += emissions_total
        entry["available_allowances_sum"] += float(summary.get("available_allowances", 0.0))
        if entry["bank_prev_first"] is None:
            entry["bank_prev_first"] = float(summary.get("bank_prev", 0.0))
        entry["bank_new_last"] = float(summary.get("bank_new", 0.0))
        entry["obligation_last"] = float(summary.get("obligation_new", 0.0))
        entry["shortage_any"] = entry["shortage_any"] or bool(summary.get("shortage_flag", False))
        entry["surrendered_sum"] += float(summary.get("surrendered", 0.0))

        finalize_raw = dict(summary.get("finalize", {}))
        entry["finalize_last"] = finalize_raw
        if finalize_raw:
            entry["finalized"] = bool(finalize_raw.get("finalized", entry["finalized"]))
            entry["bank_final"] = float(finalize_raw.get("bank_final", entry["bank_new_last"]))
            entry["obligation_last"] = float(
                finalize_raw.get("remaining_obligation", entry["obligation_last"])
            )
            entry["shortage_any"] = entry["shortage_any"] or bool(
                finalize_raw.get("shortage_flag", False)
            )
            entry["surrendered_extra"] = float(
                finalize_raw.get("surrendered_additional", entry["surrendered_extra"])
            )
        else:
            entry["bank_final"] = entry.get("bank_final", entry["bank_new_last"])

        emissions_by_region = getattr(dispatch_result, "emissions_by_region", None)
        if isinstance(emissions_by_region, Mapping):
            for region, value in emissions_by_region.items():
                key = str(region)
                entry["emissions_by_region"][key] += float(value)
        else:
            entry["emissions_by_region"]["system"] += emissions_total

        region_prices = getattr(dispatch_result, "region_prices", {})
        if isinstance(region_prices, Mapping):
            price_map = entry["price_by_region"]
            for region, value in region_prices.items():
                price_map[str(region)] = float(value)

        flows = getattr(dispatch_result, "flows", {})
        if isinstance(flows, Mapping):
            flow_map = entry["flows"]
            for key, value in flows.items():
                if isinstance(key, tuple) and len(key) == 2:
                    key_norm = (str(key[0]), str(key[1]))
                    flow_map[key_norm] += float(value)

    annual_rows: list[dict[str, object]] = []
    emissions_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    for year in sorted(aggregated):
        entry = aggregated[year]
        minted = float(entry["available_allowances_sum"])
        bank_prev = float(entry.get("bank_prev_first") or 0.0)
        allowances_total = bank_prev + minted
        surrendered_total = float(entry["surrendered_sum"]) + float(entry.get("surrendered_extra", 0.0))
        bank_final = float(entry.get("bank_final", entry.get("bank_new_last", 0.0)))
        obligation_final = float(entry.get("obligation_last", 0.0))
        price_value = float(entry.get("price_last", 0.0))
        allowance_value = float(entry.get("allowance_price_last", price_value))
        exogenous_value = float(entry.get("exogenous_price_last", 0.0))
        effective_value = float(entry.get("effective_price_last", price_value))
        iterations_value = int(entry.get("iterations_max", 0))
        shortage_flag = bool(entry.get("shortage_any", False))
        finalized = bool(entry.get("finalized", False))

        annual_rows.append(
            {
                "year": year,
                "p_co2": price_value,
                "p_co2_allowance": allowance_value,
                "p_co2_exogenous": exogenous_value,
                "p_co2_effective": effective_value,
                "iterations": iterations_value,
                "emissions_tons": float(entry.get("emissions_sum", 0.0)),
                "available_allowances": minted,
                "allowances_total": allowances_total,
                "bank": bank_final,
                "surrendered": surrendered_total,
                "obligation": obligation_final,
                "finalized": finalized,
                "shortage_flag": shortage_flag,
            }
        )

        for region, value in entry["emissions_by_region"].items():
            emissions_rows.append({"year": year, "region": region, "emissions_tons": float(value)})

        for region, value in entry["price_by_region"].items():
            price_rows.append({"year": year, "region": region, "price": float(value)})

        for (region_a, region_b), value in entry["flows"].items():
            flow_value = float(value)
            if flow_value >= 0.0:
                flow_rows.append(
                    {
                        "year": year,
                        "from_region": region_a,
                        "to_region": region_b,
                        "flow_mwh": flow_value,
                    }
                )
            else:
                flow_rows.append(
                    {
                        "year": year,
                        "from_region": region_b,
                        "to_region": region_a,
                        "flow_mwh": -flow_value,
                    }
                )

    annual_df = pd.DataFrame(annual_rows)
    if not annual_df.empty:
        annual_df = annual_df.sort_values("year").reset_index(drop=True)

    emissions_df = pd.DataFrame(emissions_rows, columns=["year", "region", "emissions_tons"])
    if not emissions_df.empty:
        emissions_df = emissions_df.sort_values(["year", "region"]).reset_index(drop=True)

    price_df = pd.DataFrame(price_rows, columns=["year", "region", "price"])
    if not price_df.empty:
        price_df = price_df.sort_values(["year", "region"]).reset_index(drop=True)

    flows_columns = ["year", "from_region", "to_region", "flow_mwh"]
    flows_df = pd.DataFrame(flow_rows, columns=flows_columns)
    if not flows_df.empty:
        flows_df = flows_df.sort_values(flows_columns[:-1]).reset_index(drop=True)

    return EngineOutputs(
        annual=annual_df,
        emissions_by_region=emissions_df,
        price_by_region=price_df,
        flows=flows_df,
    )


def _coerce_price_schedule(
    schedule: Mapping[int, float] | Mapping[str, Any] | float | None,
) -> dict[int, float]:
    """Normalise ``schedule`` into a mapping keyed by integer years."""

    if schedule is None:
        return {}

    if isinstance(schedule, Mapping):
        normalised: dict[int, float] = {}
        for key, value in schedule.items():
            try:
                year = int(key)
            except (TypeError, ValueError):
                continue
            try:
                price = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            normalised[year] = price
        return normalised

    try:
        value = float(schedule)
    except (TypeError, ValueError):
        return {}

    return {None: value}  # type: ignore[index]


def run_end_to_end_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
    enable_floor: bool = True,
    enable_ccr: bool = True,
    price_cap: float = 1000.0,
    use_network: bool = False,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | float | None = None,
    deep_carbon_pricing: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> EngineOutputs:
    """Run the integrated dispatch and allowance engine returning structured outputs.

    When ``progress_cb`` is provided the callable receives updates for the
    overall run as well as each simulated year using the ``(stage, payload)``
    convention described in :func:`_solve_allowance_market_year`.
    """

    _ensure_pandas()

    frames_obj = Frames.coerce(frames, carbon_price_schedule=carbon_price_schedule)
    policy_spec = frames_obj.policy()
    policy = policy_spec.to_policy()
    years_sequence = _coerce_years(policy, years)
    period_weights = _compute_period_weights(policy, years_sequence)
    dispatch_kwargs = dict(
        use_network=use_network,
        period_weights=period_weights,
        carbon_price_schedule=carbon_price_schedule,
    )
    if deep_carbon_pricing:
        dispatch_kwargs["deep_carbon_pricing"] = bool(deep_carbon_pricing)
    dispatch_solver = _dispatch_from_frames(frames_obj, **dispatch_kwargs)
    years_sequence = list(years_sequence)
    total_years = len(years_sequence)

    if progress_cb is not None:
        progress_cb(
            "run_start",
            {
                "total_years": total_years,
                "years": list(years_sequence),
            },
        )

    results: dict[Any, dict[str, object]] = {}
    policy_enabled_global = bool(getattr(policy, 'enabled', True))
    banking_enabled_global = bool(getattr(policy, 'banking_enabled', True))
    bank_prev = float(policy.bank0) if (policy_enabled_global and banking_enabled_global) else 0.0
    bank_prev = max(0.0, bank_prev)

    cp_track: dict[str, dict[str, float | list[int] | None]] = {}

    price_schedule = _coerce_price_schedule(carbon_price_schedule)

    def _price_for_year(period: Any) -> float:
        try:
            year_int = int(period)
        except (TypeError, ValueError):
            year_int = None
        if year_int is not None and year_int in price_schedule:
            return float(price_schedule[year_int])
        if None in price_schedule:
            return float(price_schedule[None])
        return 0.0

    for idx, year in enumerate(years_sequence):
        if progress_cb is not None:
            progress_cb(
                "year_start",
                {
                    "year": _normalize_progress_year(year),
                    "index": idx,
                    "total_years": total_years,
                },
            )

        carbon_price_value = _price_for_year(year)

        if not policy_enabled_global:
            dispatch_result = dispatch_solver(
                year, 0.0, carbon_price=0.0
            )
            emissions = _extract_emissions(dispatch_result)
            summary_disabled: dict[str, object] = {
                'year': year,
                'p_co2': float(carbon_price_value),
                'p_co2_allowance': 0.0,
                'p_co2_exogenous': float(carbon_price_value),
                'p_co2_effective': effective_carbon_price(
                    0.0, float(carbon_price_value), deep_carbon_pricing
                ),
                'available_allowances': float(emissions),
                'allowances_total': float(emissions),
                'bank_prev': 0.0,
                'bank_unadjusted': 0.0,
                'bank_new': 0.0,
                'surrendered': 0.0,
                'obligation_new': 0.0,
                'shortage_flag': False,
                'iterations': 0,
                'emissions': float(emissions),
                'ccr1_issued': 0.0,
                'ccr2_issued': 0.0,
                'finalize': {
                    'finalized': False,
                    'bank_final': 0.0,
                    'remaining_obligation': 0.0,
                    'surrendered_additional': 0.0,
                },
                '_dispatch_result': dispatch_result,
            }
            results[year] = summary_disabled
            if progress_cb is not None:
                progress_cb(
                    "year_complete",
                    {
                        "year": int(year),
                        "index": idx,
                        "total_years": total_years,
                        "shortage": False,
                        "price": float(carbon_price_value),
                        "iterations": 0,
                    },
                )
            continue

        supply, record = _build_allowance_supply(
            policy,
            year,
            enable_floor=enable_floor,
            enable_ccr=enable_ccr,
        )

        policy_enabled_year = bool(supply.enabled) and policy_enabled_global
        cp_id = str(record.get('cp_id', 'NoPolicy'))
        surrender_frac = float(
            record.get('annual_surrender_frac', getattr(policy, 'annual_surrender_frac', 0.5))
        )
        carry_pct = float(record.get('carry_pct', getattr(policy, 'carry_pct', 1.0)))
        banking_enabled_year = banking_enabled_global and bool(
            record.get('bank_enabled', banking_enabled_global)
        )
        if not policy_enabled_year:
            banking_enabled_year = False
        if not banking_enabled_year:
            carry_pct = 0.0

        bank_prev_effective = bank_prev if (banking_enabled_year and policy_enabled_year) else 0.0

        state: dict[str, float | list[int] | None] | None = None
        outstanding_prev = 0.0
        if policy_enabled_year:
            state = cp_track.setdefault(
                cp_id,
                {
                    'emissions': 0.0,
                    'surrendered': 0.0,
                    'cap': 0.0,
                    'ccr1': 0.0,
                    'ccr2': 0.0,
                    'bank_start': bank_prev_effective,
                    'outstanding': 0.0,
                    'years': [],
                },
            )

            if state.get('bank_start') is None:
                state['bank_start'] = bank_prev_effective

            years_list = state.setdefault('years', [])
            if isinstance(years_list, list) and year not in years_list:
                years_list.append(year)

            outstanding_prev = float(state.get('outstanding', 0.0))

        summary = _solve_allowance_market_year(
            dispatch_solver,
            year,
            supply,
            bank_prev_effective,
            outstanding_prev,
            policy_enabled=policy_enabled_year,
            high_price=price_cap,
            tol=tol,
            max_iter=max_iter,
            annual_surrender_frac=surrender_frac,
            carry_pct=carry_pct,
            banking_enabled=banking_enabled_year,
            carbon_price=0.0,
            progress_cb=progress_cb,
        )

        try:
            allowance_price = float(summary.get('p_co2', 0.0))
        except (TypeError, ValueError):
            allowance_price = 0.0
        exogenous_price = float(carbon_price_value)
        effective_price = effective_carbon_price(
            allowance_price, exogenous_price, deep_carbon_pricing
        )

        summary['p_co2_allowance'] = allowance_price
        summary['p_co2_exogenous'] = exogenous_price
        summary['p_co2_effective'] = effective_price

        LOGGER.debug(
            "Year %s allowance price %.4f, exogenous price %.4f, effective price %.4f (deep=%s)",
            year,
            allowance_price,
            exogenous_price,
            effective_price,
            deep_carbon_pricing,
        )

        emissions = float(summary.get('emissions', 0.0))
        surrendered = float(summary.get('surrendered', 0.0))
        bank_unadjusted = float(summary.get('bank_unadjusted', summary.get('bank_new', 0.0)))
        obligation = float(summary.get('obligation_new', 0.0))
        ccr1_issued = float(summary.get('ccr1_issued', 0.0))
        ccr2_issued = float(summary.get('ccr2_issued', 0.0))

        if state is not None:
            state['emissions'] = float(state.get('emissions', 0.0)) + emissions
            state['surrendered'] = float(state.get('surrendered', 0.0)) + surrendered
            state['cap'] = float(state.get('cap', 0.0)) + float(record.get('cap', 0.0))
            state['ccr1'] = float(state.get('ccr1', 0.0)) + ccr1_issued
            state['ccr2'] = float(state.get('ccr2', 0.0)) + ccr2_issued
            state['outstanding'] = obligation
            state['bank_last_unadjusted'] = bank_unadjusted
            state['bank_last_carried'] = float(summary.get('bank_new', 0.0))

        finalize_summary = dict(summary.get('finalize', {}))

        if not policy_enabled_year:
            finalize_summary.setdefault('finalized', False)
            finalize_summary.setdefault('bank_final', float(summary.get('bank_new', 0.0)))
            finalize_summary.setdefault('remaining_obligation', 0.0)
            finalize_summary.setdefault('surrendered_additional', 0.0)
            summary['finalize'] = finalize_summary
            results[year] = summary
            bank_prev = float(summary.get('bank_new', 0.0)) if banking_enabled_year else 0.0
            if state is not None:
                state['outstanding'] = 0.0
            if progress_cb is not None:
                payload: dict[str, object] = {
                    "year": _normalize_progress_year(year),
                    "index": idx,
                    "total_years": total_years,
                    "shortage": bool(summary.get('shortage_flag', False)),
                }
                price_value = summary.get('p_co2')
                try:
                    if price_value is not None:
                        payload['price'] = float(price_value)
                except (TypeError, ValueError):
                    pass
                iterations_value = summary.get('iterations')
                try:
                    payload['iterations'] = int(iterations_value)
                except (TypeError, ValueError):
                    payload['iterations'] = 0
                progress_cb('year_complete', payload)
            continue

        is_final_year = bool(record.get('full_compliance', False))
        if not is_final_year:
            if idx + 1 < len(years_sequence):
                next_year = years_sequence[idx + 1]
                next_record = _policy_record_for_year(policy, next_year)
                next_cp_id = str(next_record.get('cp_id', 'NoPolicy'))
                if next_cp_id != cp_id:
                    is_final_year = True
            else:
                is_final_year = True

        if is_final_year:
            outstanding_before = obligation
            surrender_additional = min(outstanding_before, bank_unadjusted)
            remaining_obligation = max(outstanding_before - surrender_additional, 0.0)
            bank_after_trueup = max(bank_unadjusted - surrender_additional, 0.0)
            bank_carry = max(bank_after_trueup * carry_pct, 0.0)

            summary['bank_unadjusted'] = bank_after_trueup
            summary['bank_new'] = bank_carry
            summary['obligation_new'] = remaining_obligation

            if state is not None:
                state['surrendered'] = float(state.get('surrendered', 0.0)) + surrender_additional
                state['bank_last_unadjusted'] = bank_after_trueup
                state['bank_last_carried'] = bank_carry
                state['outstanding'] = 0.0

            total_allowances = 0.0
            if state is not None:
                total_allowances = float(state.get('bank_start', 0.0)) + float(state.get('cap', 0.0))
                total_allowances += float(state.get('ccr1', 0.0)) + float(state.get('ccr2', 0.0))

            finalize_summary = {
                'finalized': True,
                'cp_id': cp_id,
                'bank_final': float(bank_carry),
                'remaining_obligation': float(remaining_obligation),
                'surrendered_additional': float(surrender_additional),
                'shortage_flag': bool(remaining_obligation > 1e-9),
                'cp_emissions': float(state.get('emissions', 0.0)) if state is not None else float(emissions),
                'cp_surrendered': float(state.get('surrendered', 0.0)) if state is not None else float(surrendered),
                'cp_cap': float(state.get('cap', 0.0)) if state is not None else float(record.get('cap', 0.0)),
                'cp_ccr1': float(state.get('ccr1', 0.0)) if state is not None else float(ccr1_issued),
                'cp_ccr2': float(state.get('ccr2', 0.0)) if state is not None else float(ccr2_issued),
                'bank_start': float(state.get('bank_start', 0.0)) if state is not None else 0.0,
                'cp_allowances_total': float(total_allowances),
            }

            bank_prev = bank_carry if banking_enabled_year else 0.0
        else:
            bank_carry = max(float(summary.get('bank_new', 0.0)), 0.0)
            finalize_summary = {
                'finalized': False,
                'bank_final': bank_carry,
                'remaining_obligation': float(obligation),
                'surrendered_additional': 0.0,
            }
            summary['bank_new'] = bank_carry
            summary['obligation_new'] = obligation
            bank_prev = bank_carry if banking_enabled_year else 0.0

        summary['finalize'] = finalize_summary
        results[year] = summary

        if progress_cb is not None:
            payload = {
                'year': _normalize_progress_year(year),
                'index': idx,
                'total_years': total_years,
                'shortage': bool(summary.get('shortage_flag', False)),
            }
            price_value = summary.get('p_co2')
            try:
                if price_value is not None:
                    payload['price'] = float(price_value)
            except (TypeError, ValueError):
                pass
            iterations_value = summary.get('iterations')
            try:
                payload['iterations'] = int(iterations_value)
            except (TypeError, ValueError):
                payload['iterations'] = 0
            progress_cb('year_complete', payload)

    ordered_years = [period for period in years_sequence if period in results]
    return _build_engine_outputs(ordered_years, results, dispatch_solver, policy)
