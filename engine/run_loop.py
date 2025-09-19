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
from policy.allowance_annual import AllowanceAnnual, RGGIPolicyAnnual
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
    *,
    policy_enabled: bool,
    high_price: float,
    tol: float,
    max_iter: int,
) -> dict[str, object]:
    """Solve for the allowance clearing price for ``year`` using bisection."""

    tol = max(float(tol), 0.0)
    max_iter_int = max(int(max_iter), 0)
    high_price = float(high_price)

    def _issued_quantities(price: float, allowances: float) -> tuple[float, float]:
        if not supply.enable_ccr:
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

    if not policy_enabled:
        clearing_price = 0.0
        dispatch_result = dispatch_solver(year, clearing_price)
        emissions = _extract_emissions(dispatch_result)
        allowances = max(supply.available_allowances(clearing_price), emissions)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances)
        return {
            'year': year,
            'p_co2': float(clearing_price),
            'available_allowances': float(allowances),
            'bank_prev': 0.0,
            'bank_new': 0.0,
            'surrendered': 0.0,
            'obligation_new': 0.0,
            'shortage_flag': False,
            'iterations': 0,
            'emissions': float(emissions),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {'finalized': False, 'bank_final': 0.0},
            '_dispatch_result': dispatch_result,
        }

    min_price = supply.floor if supply.enable_floor else 0.0
    low = max(0.0, float(min_price))
    high = max(low, high_price if high_price > 0.0 else low)

    dispatch_low = dispatch_solver(year, low)
    emissions_low = _extract_emissions(dispatch_low)
    allowances_low = supply.available_allowances(low)

    if allowances_low >= emissions_low:
        clearing_price = supply.enforce_floor(low)
        if clearing_price != low:
            dispatch_low = dispatch_solver(year, clearing_price)
            emissions_low = _extract_emissions(dispatch_low)
            allowances_low = supply.available_allowances(clearing_price)
        bank = max(allowances_low - emissions_low, 0.0)
        obligation = max(emissions_low - allowances_low, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_low)
        return {
            'year': year,
            'p_co2': float(clearing_price),
            'available_allowances': float(allowances_low),
            'bank_prev': 0.0,
            'bank_new': float(bank),
            'surrendered': float(min(emissions_low, allowances_low)),
            'obligation_new': float(obligation),
            'shortage_flag': False,
            'iterations': 0,
            'emissions': float(emissions_low),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {'finalized': False, 'bank_final': float(bank)},
            '_dispatch_result': dispatch_low,
        }

    dispatch_high = dispatch_solver(year, high)
    emissions_high = _extract_emissions(dispatch_high)
    allowances_high = supply.available_allowances(high)

    if allowances_high < emissions_high - tol:
        clearing_price = supply.enforce_floor(high)
        if clearing_price != high:
            dispatch_high = dispatch_solver(year, clearing_price)
            emissions_high = _extract_emissions(dispatch_high)
            allowances_high = supply.available_allowances(clearing_price)
        bank = max(allowances_high - emissions_high, 0.0)
        obligation = max(emissions_high - allowances_high, 0.0)
        ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, allowances_high)
        return {
            'year': year,
            'p_co2': float(clearing_price),
            'available_allowances': float(allowances_high),
            'bank_prev': 0.0,
            'bank_new': float(bank),
            'surrendered': float(min(emissions_high, allowances_high)),
            'obligation_new': float(obligation),
            'shortage_flag': True,
            'iterations': max_iter_int,
            'emissions': float(emissions_high),
            'ccr1_issued': float(ccr1_issued),
            'ccr2_issued': float(ccr2_issued),
            'finalize': {'finalized': False, 'bank_final': float(bank)},
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
        iteration_count = iteration
        if allowances_mid >= emissions_mid:
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

    bank = max(best_allowances - best_emissions, 0.0)
    obligation = max(best_emissions - best_allowances, 0.0)
    shortage_flag = best_emissions > best_allowances + tol
    ccr1_issued, ccr2_issued = _issued_quantities(clearing_price, best_allowances)

    return {
        'year': year,
        'p_co2': float(clearing_price),
        'available_allowances': float(best_allowances),
        'bank_prev': 0.0,
        'bank_new': float(bank),
        'surrendered': float(min(best_emissions, best_allowances)),
        'obligation_new': float(obligation),
        'shortage_flag': bool(shortage_flag),
        'iterations': iteration_count,
        'emissions': float(best_emissions),
        'ccr1_issued': float(ccr1_issued),
        'ccr2_issued': float(ccr2_issued),
        'finalize': {'finalized': False, 'bank_final': float(bank)},
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

    allowance = AllowanceAnnual(policy)
    results: dict[int, dict] = {}

    bank = float(policy.bank0)
    previous_price = float(price_initial) if not isinstance(price_initial, Mapping) else 0.0

    for year in years_sequence:
        bank_start = bank
        price_guess = _initial_price_for_year(price_initial, year, previous_price)
        iteration_count = 0
        emissions = 0.0
        market_result: dict[str, float | bool | str] | None = None

        for iteration in range(1, max_iter + 1):
            iteration_count = iteration
            dispatch_output = dispatch_model(year, price_guess)
            emissions = _extract_emissions(dispatch_output)
            market_result = allowance.clear_year(
                year,
                emissions_tons=emissions,
                bank_prev=bank_start,
                expected_price_guess=price_guess,
            )
            cleared_price = float(market_result['p_co2'])
            if abs(cleared_price - price_guess) <= tol:
                price_guess = cleared_price
                break
            price_guess = price_guess + relaxation * (cleared_price - price_guess)
        else:  # pragma: no cover - defensive guard
            raise RuntimeError(f'Allowance price failed to converge for year {year}')

        assert market_result is not None  # for type checker
        market_result = dict(market_result)
        market_result['iterations'] = iteration_count
        market_result['emissions'] = emissions

        bank = float(market_result['bank_new'])
        finalize_summary = allowance.finalize_period_if_needed(year)
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
    dispatch_solver = _dispatch_from_frames(frames_obj, use_network=use_network)
    years_sequence = _coerce_years(policy_spec, years)

    results: dict[int, dict[str, object]] = {}
    for year in years_sequence:
        supply, record = _build_allowance_supply(
            policy_spec,
            year,
            enable_floor=enable_floor,
            enable_ccr=enable_ccr,
        )
        summary = _solve_allowance_market_year(
            dispatch_solver,
            year,
            supply,
            policy_enabled=bool(record.get('enabled', True)),
            high_price=price_cap,
            tol=tol,
            max_iter=max_iter,
        )
        results[year] = summary

    ordered_years = sorted(results)
    return _build_engine_outputs(ordered_years, results, dispatch_solver)
