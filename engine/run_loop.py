"""Annual fixed-point integration between dispatch and allowance market."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence, cast

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

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


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before running the engine."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for engine.run_loop; install it with `pip install pandas`."
        )


def _coerce_years(policy: Any, years: Iterable[int] | None) -> list[int]:
    if years is None:
        series_index = getattr(policy.cap, 'index', [])
        years_list = list(series_index) if series_index is not None else []
    else:
        years_list = list(years)
    return sorted({int(y) for y in years_list})


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

    full_compliance_years = getattr(policy, 'full_compliance_years', set())
    try:
        record['full_compliance'] = bool(int(year) in set(full_compliance_years))
    except TypeError:  # pragma: no cover - defensive
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
    ccr1_qty = float(record['ccr1_qty']) if record['ccr1_enabled'] else 0.0
    ccr2_qty = float(record['ccr2_qty']) if record['ccr2_enabled'] else 0.0
    supply = AllowanceSupply(
        cap=float(record['cap']),
        floor=float(record['floor']),
        ccr1_trigger=float(record['ccr1_trigger']),
        ccr1_qty=ccr1_qty,
        ccr2_trigger=float(record['ccr2_trigger']),
        ccr2_qty=ccr2_qty,
        enabled=bool(record.get('enabled', True)),
        enable_floor=enable_floor,
        enable_ccr=enable_ccr and (ccr1_qty > 0.0 or ccr2_qty > 0.0),
    )
    record['ccr1_qty'] = ccr1_qty
    record['ccr2_qty'] = ccr2_qty
    return supply, record


def _solve_allowance_market_year(
    dispatch_solver: Callable[[int, float], object],
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
) -> dict[str, object]:
    """Solve for the allowance clearing price for ``year`` using bisection."""

    tol = max(float(tol), 0.0)
    max_iter_int = max(int(max_iter), 0)
    high_price = float(high_price)

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
    outstanding_prev = max(0.0, float(outstanding_prev))
    carry_pct = max(0.0, float(carry_pct))
    surrender_frac = max(0.0, min(1.0, float(annual_surrender_frac)))

    if not policy_enabled or not supply.enabled:
        clearing_price = 0.0
        dispatch_result = dispatch_solver(year, clearing_price)
        emissions = _extract_emissions(dispatch_result)
        allowances = max(supply.available_allowances(clearing_price), emissions)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances)
        minted_allowances = float(max(allowances, emissions))
        total_allowances = minted_allowances  # bank is zero when policy disabled
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

    dispatch_low = dispatch_solver(year, low)
    emissions_low = _extract_emissions(dispatch_low)
    allowances_low = supply.available_allowances(low)

    total_allowances_low = bank_prev + allowances_low
    if total_allowances_low >= emissions_low:
        clearing_price = supply.enforce_floor(low)
        if clearing_price != low:
            dispatch_low = dispatch_solver(year, clearing_price)
            emissions_low = _extract_emissions(dispatch_low)
            allowances_low = supply.available_allowances(clearing_price)
            total_allowances_low = bank_prev + allowances_low
        surrendered = min(surrender_frac * emissions_low, total_allowances_low)
        bank_unadjusted = max(total_allowances_low - surrendered, 0.0)
        obligation = max(outstanding_prev + emissions_low - surrendered, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_low)
        bank_carry = max(bank_unadjusted * carry_pct, 0.0)
        return {
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

    dispatch_high = dispatch_solver(year, high)
    emissions_high = _extract_emissions(dispatch_high)
    allowances_high = supply.available_allowances(high)

    if bank_prev + allowances_high < emissions_high - tol:
        clearing_price = supply.enforce_floor(high)
        if clearing_price != high:
            dispatch_high = dispatch_solver(year, clearing_price)
            emissions_high = _extract_emissions(dispatch_high)
            allowances_high = supply.available_allowances(clearing_price)
        total_allowances_high = bank_prev + allowances_high
        surrendered = min(surrender_frac * emissions_high, total_allowances_high)
        bank_unadjusted = max(total_allowances_high - surrendered, 0.0)
        obligation = max(outstanding_prev + emissions_high - surrendered, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_high)
        bank_carry = max(bank_unadjusted * carry_pct, 0.0)
        return {
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

    best_price = high
    best_allowances = allowances_high
    best_emissions = emissions_high
    best_dispatch = dispatch_high
    iteration_count = 0

    low_bound = low
    high_bound = high

    for iteration in range(1, max_iter_int + 1):
        mid = 0.5 * (low_bound + high_bound)
        dispatch_mid = dispatch_solver(year, mid)
        emissions_mid = _extract_emissions(dispatch_mid)
        allowances_mid = supply.available_allowances(mid)
        total_allowances_mid = bank_prev + allowances_mid
        iteration_count = iteration
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
        best_dispatch = dispatch_solver(year, clearing_price)
        best_emissions = _extract_emissions(best_dispatch)
        best_allowances = supply.available_allowances(clearing_price)
    total_allowances = bank_prev + best_allowances

    surrendered = min(surrender_frac * best_emissions, total_allowances)
    bank_unadjusted = max(total_allowances - surrendered, 0.0)
    obligation = max(outstanding_prev + best_emissions - surrendered, 0.0)
    shortage_flag = best_emissions > total_allowances + tol
    ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, best_allowances)
    bank_carry = max(bank_unadjusted * carry_pct, 0.0)

    return {
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

    bank = float(policy.bank0)
    previous_price = float(price_initial) if not isinstance(price_initial, Mapping) else 0.0

    for year in years_sequence:
        bank_start = bank
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
        finalize_summary, allowance_state = allowance_finalize_period(policy, allowance_state, year)
        market_result['finalize'] = finalize_summary
        if finalize_summary.get('finalized'):
            bank = float(finalize_summary.get('bank_final', bank))

        results[year] = market_result
        previous_price = price_guess

    return results


def _dispatch_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    use_network: bool = False,
) -> Callable[[int, float], object]:
    """Build a dispatch callback that solves using the frame container."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames)

    if use_network:
        def dispatch(year: int, allowance_cost: float):
            return solve_network_from_frames(frames_obj, year, allowance_cost)
    else:
        def dispatch(year: int, allowance_cost: float):
            return solve_single(year, allowance_cost, frames=frames_obj)

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
) -> dict[int, dict]:
    """Run the annual fixed-point integration using in-memory frames."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames)
    policy_spec = frames_obj.policy()
    policy = policy_spec.to_policy()
    dispatch_solver = _dispatch_from_frames(frames_obj, use_network=use_network)

    def dispatch_model(year: int, allowance_cost: float) -> float:
        return _extract_emissions(dispatch_solver(year, allowance_cost))

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
    years: Sequence[int],
    raw_results: Mapping[int, Mapping[str, object]],
    dispatch_solver: Callable[[int, float], object],
) -> EngineOutputs:
    """Convert annual fixed-point results into structured engine outputs."""

    _ensure_pandas()

    annual_rows: list[dict[str, object]] = []
    emissions_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []

    for year in years:
        summary_raw = raw_results.get(year)
        if summary_raw is None:
            continue
        summary = dict(summary_raw)
        price = float(summary.get("p_co2", 0.0))
        dispatch_result = summary.pop("_dispatch_result", None)
        if dispatch_result is None:
            dispatch_result = dispatch_solver(year, price)
        emissions_total = _extract_emissions(dispatch_result)

        finalize = summary.get("finalize") or {}
        bank_value = float(summary.get("bank_new", 0.0))
        surrendered = float(summary.get("surrendered", 0.0))
        obligation = float(summary.get("obligation_new", 0.0))
        shortage_flag = bool(summary.get("shortage_flag", False))
        finalized = bool(finalize.get("finalized", False))
        if finalized:
            bank_value = float(finalize.get("bank_final", bank_value))
            surrendered += float(finalize.get("surrendered_additional", 0.0))
            obligation = float(finalize.get("remaining_obligation", obligation))
            shortage_flag = shortage_flag or bool(finalize.get("shortage_flag", False))

        iterations_value = summary.get("iterations", 0)
        try:
            iterations = int(iterations_value)
        except (TypeError, ValueError):
            iterations = 0

        allowances_available = float(summary.get("available_allowances", 0.0))
        allowances_total = float(
            summary.get(
                "allowances_total",
                allowances_available + float(summary.get("bank_prev", 0.0)),
            )
        )

        emissions_by_region = getattr(dispatch_result, "emissions_by_region", None)
        if isinstance(emissions_by_region, Mapping):
            for region, value in emissions_by_region.items():
                emissions_rows.append(
                    {"year": year, "region": str(region), "emissions_tons": float(value)}
                )
        else:
            emissions_rows.append(
                {"year": year, "region": "system", "emissions_tons": float(emissions_total)}
            )

        region_prices = getattr(dispatch_result, "region_prices", {})
        if isinstance(region_prices, Mapping):
            for region, value in region_prices.items():
                price_rows.append({"year": year, "region": str(region), "price": float(value)})

        flows = getattr(dispatch_result, "flows", {})
        if isinstance(flows, Mapping):
            for key, flow_value in flows.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                region_a, region_b = key
                flow_float = float(flow_value)
                if flow_float >= 0.0:
                    flow_rows.append(
                        {
                            "year": year,
                            "from_region": str(region_a),
                            "to_region": str(region_b),
                            "flow_mwh": flow_float,
                        }
                    )
                else:
                    flow_rows.append(
                        {
                            "year": year,
                            "from_region": str(region_b),
                            "to_region": str(region_a),
                            "flow_mwh": -flow_float,
                        }
                    )

        annual_rows.append(
            {
                "year": year,
                "p_co2": price,
                "iterations": iterations,
                "emissions_tons": float(emissions_total),
                "available_allowances": allowances_available,
                "allowances_total": allowances_total,
                "bank": bank_value,
                "surrendered": surrendered,
                "obligation": obligation,
                "finalized": finalized,
                "shortage_flag": shortage_flag,
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
) -> EngineOutputs:
    """Run the integrated dispatch and allowance engine returning structured outputs."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames)
    policy_spec = frames_obj.policy()
    policy = policy_spec.to_policy()
    dispatch_solver = _dispatch_from_frames(frames_obj, use_network=use_network)
    years_sequence = _coerce_years(policy, years)

    results: dict[int, dict[str, object]] = {}
    policy_enabled_global = bool(getattr(policy, 'enabled', True))
    bank_prev = float(policy.bank0) if policy_enabled_global else 0.0
    bank_prev = max(0.0, bank_prev)

    cp_track: dict[str, dict[str, float | list[int] | None]] = {}

    for idx, year in enumerate(years_sequence):
        supply, record = _build_allowance_supply(
            policy,
            year,
            enable_floor=enable_floor,
            enable_ccr=enable_ccr,
        )

        policy_enabled_year = bool(supply.enabled) and policy_enabled_global
        cp_id = str(record.get('cp_id', 'NoPolicy'))
        surrender_frac = float(record.get('annual_surrender_frac', getattr(policy, 'annual_surrender_frac', 0.5)))
        carry_pct = float(record.get('carry_pct', getattr(policy, 'carry_pct', 1.0)))

        state = cp_track.setdefault(
            cp_id,
            {
                'emissions': 0.0,
                'surrendered': 0.0,
                'cap': 0.0,
                'ccr1': 0.0,
                'ccr2': 0.0,
                'bank_start': bank_prev,
                'outstanding': 0.0,
                'years': [],
            },
        )

        if state.get('bank_start') is None:
            state['bank_start'] = bank_prev

        years_list = state.setdefault('years', [])
        if isinstance(years_list, list) and year not in years_list:
            years_list.append(year)

        outstanding_prev = float(state.get('outstanding', 0.0))

        summary = _solve_allowance_market_year(
            dispatch_solver,
            year,
            supply,
            bank_prev,
            outstanding_prev,
            policy_enabled=policy_enabled_year,
            high_price=price_cap,
            tol=tol,
            max_iter=max_iter,
            annual_surrender_frac=surrender_frac,
            carry_pct=carry_pct,
        )

        emissions = float(summary.get('emissions', 0.0))
        surrendered = float(summary.get('surrendered', 0.0))
        bank_unadjusted = float(summary.get('bank_unadjusted', summary.get('bank_new', 0.0)))
        obligation = float(summary.get('obligation_new', 0.0))
        ccr1_issued = float(summary.get('ccr1_issued', 0.0))
        ccr2_issued = float(summary.get('ccr2_issued', 0.0))

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
            bank_prev = float(summary.get('bank_new', 0.0))
            state['outstanding'] = 0.0
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

            state['surrendered'] = float(state.get('surrendered', 0.0)) + surrender_additional
            state['bank_last_unadjusted'] = bank_after_trueup
            state['bank_last_carried'] = bank_carry
            state['outstanding'] = 0.0

            total_allowances = float(state.get('bank_start', 0.0)) + float(state.get('cap', 0.0))
            total_allowances += float(state.get('ccr1', 0.0)) + float(state.get('ccr2', 0.0))

            finalize_summary = {
                'finalized': True,
                'cp_id': cp_id,
                'bank_final': float(bank_carry),
                'remaining_obligation': float(remaining_obligation),
                'surrendered_additional': float(surrender_additional),
                'shortage_flag': bool(remaining_obligation > 1e-9),
                'cp_emissions': float(state.get('emissions', 0.0)),
                'cp_surrendered': float(state.get('surrendered', 0.0)),
                'cp_cap': float(state.get('cap', 0.0)),
                'cp_ccr1': float(state.get('ccr1', 0.0)),
                'cp_ccr2': float(state.get('ccr2', 0.0)),
                'bank_start': float(state.get('bank_start', 0.0)),
                'cp_allowances_total': float(total_allowances),
            }

            bank_prev = bank_carry
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
            bank_prev = bank_carry

        summary['finalize'] = finalize_summary
        results[year] = summary

    ordered_years = sorted(results)
    return _build_engine_outputs(ordered_years, results, dispatch_solver)
