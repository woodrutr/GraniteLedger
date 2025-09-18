"""Annual fixed-point integration between dispatch and allowance market."""
from __future__ import annotations

from typing import Callable, Iterable, Mapping

import pandas as pd

from dispatch.lp_network import solve_from_frames
from policy.allowance_annual import AllowanceAnnual, RGGIPolicyAnnual

from io_loader import Frames


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
            emissions = float(dispatch_model(year, price_guess))
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


def _dispatch_from_frames(frames: Frames) -> Callable[[int, float], float]:
    """Build a dispatch callback that solves using the frame container."""

    frames_obj = Frames.coerce(frames)

    def dispatch(year: int, allowance_cost: float) -> float:
        result = solve_from_frames(frames_obj, year, allowance_cost)
        return float(result.emissions_tons)

    return dispatch


def run_fixed_point_from_frames(
    frames: Frames | Mapping[str, pd.DataFrame],
    *,
    years: Iterable[int] | None = None,
    price_initial: float | Mapping[int, float] = 0.0,
    tol: float = 1e-3,
    max_iter: int = 25,
    relaxation: float = 0.5,
) -> dict[int, dict]:
    """Run the annual fixed-point integration using in-memory frames."""

    frames_obj = Frames.coerce(frames)
    policy_spec = frames_obj.policy()
    policy = policy_spec.to_policy()
    dispatch_model = _dispatch_from_frames(frames_obj)

    return run_annual_fixed_point(
        policy,
        dispatch_model,
        years=years,
        price_initial=price_initial,
        tol=tol,
        max_iter=max_iter,
        relaxation=relaxation,
    )
