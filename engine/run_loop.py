"""Annual fixed-point integration between dispatch and allowance market."""
from __future__ import annotations

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence, cast

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

from dispatch.lp_network import solve_from_frames as solve_network_from_frames
from dispatch.lp_single import solve as solve_single
from engine.outputs import EngineOutputs
from policy.allowance_annual import AllowanceAnnual, RGGIPolicyAnnual

from io_loader import Frames


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before running the engine."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for engine.run_loop; install it with `pip install pandas`."
        )


def _coerce_years(policy: RGGIPolicyAnnual, years: Iterable[int] | None) -> list[int]:
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

    allowance = AllowanceAnnual(policy)
    years_sequence = _coerce_years(policy, years)
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
    use_network: bool = False,
) -> EngineOutputs:
    """Run the integrated dispatch and allowance engine returning structured outputs."""

    _ensure_pandas()

    frames_obj = Frames.coerce(frames)
    policy = frames_obj.policy().to_policy()
    dispatch_solver = _dispatch_from_frames(frames_obj, use_network=use_network)

    def dispatch_model(year: int, allowance_cost: float) -> float:
        return _extract_emissions(dispatch_solver(year, allowance_cost))

    results = run_annual_fixed_point(
        policy,
        dispatch_model,
        years=years,
        price_initial=price_initial,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
    )

    ordered_years = sorted(results)
    return _build_engine_outputs(ordered_years, results, dispatch_solver)
