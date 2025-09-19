"""Utilities to construct allowance policy objects from configuration mappings."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

from policy.allowance_annual import RGGIPolicyAnnual


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before accessing policy loader helpers."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for config.policy_loader; install it with `pip install pandas`."
        )

_FILL_DIRECTIVES = {"forward", "ffill", "pad"}


def _coerce_year_list(years: Any) -> list[int]:
    """Convert an iterable of years into a sorted list of unique integers."""

    if years is None:
        return []
    if isinstance(years, Mapping):  # ambiguous structure
        raise TypeError("years must be provided as an iterable of integers, not a mapping")
    if isinstance(years, (str, bytes)):
        raise TypeError("years must be provided as an iterable of integers")

    if not isinstance(years, Iterable):
        raise TypeError("years must be provided as an iterable of integers")

    normalized: list[int] = []
    seen: set[int] = set()
    for value in years:
        try:
            year = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Invalid year value {value!r}; years must be integers") from exc
        if year not in seen:
            normalized.append(year)
            seen.add(year)
    normalized.sort()
    return normalized


def _extract_year_map(entry: Any) -> tuple[Any, bool]:
    """Return the raw year-value mapping and whether forward-fill is requested."""

    fill_forward = False
    year_map = entry
    if isinstance(entry, Mapping):
        fill_token = entry.get("fill")
        if isinstance(fill_token, str) and fill_token.lower() in _FILL_DIRECTIVES:
            fill_forward = True
        elif fill_token and str(fill_token).lower() in _FILL_DIRECTIVES:
            fill_forward = True
        if entry.get("fill_forward"):
            fill_forward = True

        if "values" in entry:
            year_map = entry["values"]
        elif "data" in entry:
            year_map = entry["data"]
        elif "year_map" in entry:
            year_map = entry["year_map"]
        else:  # strip directive keys
            year_map = {k: v for k, v in entry.items() if k not in {"fill", "fill_forward"}}
    return year_map, fill_forward


def _normalize_year_value_pairs(year_map: Any, key: str) -> dict[int, Any]:
    """Convert supported year-value structures into an integer keyed dictionary."""

    _ensure_pandas()

    if isinstance(year_map, Mapping):
        iterator = year_map.items()
    elif isinstance(year_map, Iterable) and not isinstance(year_map, (str, bytes)):
        iterator = []
        for item in year_map:
            if isinstance(item, Mapping):
                year = item.get("year")
                value = item.get("value", item.get("amount"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                year, value = item
            else:
                continue
            iterator.append((year, value))
    else:
        raise TypeError(f"{key} must map years to values")

    normalized: dict[int, Any] = {}
    for year, value in iterator:
        try:
            year_int = int(year)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{key} has non-integer year {year!r}") from exc
        if year_int in normalized:
            raise ValueError(f"{key} defines duplicate year {year_int}")
        if pd.isna(value):
            raise ValueError(f"{key} has missing value for year {year_int}")
        normalized[year_int] = value

    if not normalized:
        raise ValueError(f"{key} must define at least one year")

    return normalized


def _ensure_contiguous(index: Iterable[int], key: str) -> None:
    """Ensure that the provided index covers every year in its span."""

    years = sorted(int(year) for year in index)
    if not years:
        return
    full_span = set(range(years[0], years[-1] + 1))
    missing = sorted(full_span.difference(years))
    if missing:
        raise ValueError(f"{key} has gaps for years: {missing}")


def _validate_year_alignment(series_map: Mapping[str, pd.Series]) -> list[int]:
    """Ensure each series shares the same ordered year index."""

    iterator = iter(series_map.items())
    first_key, first_series = next(iterator)
    expected_years = [int(year) for year in first_series.index]

    for key, series in iterator:
        years = [int(year) for year in series.index]
        if years != expected_years:
            expected_set = set(expected_years)
            year_set = set(years)
            missing = sorted(expected_set - year_set)
            extra = sorted(year_set - expected_set)
            detail_parts: list[str] = []
            if missing:
                detail_parts.append(f"missing years {missing}")
            if extra:
                detail_parts.append(f"extra years {extra}")
            detail = "; ".join(detail_parts) if detail_parts else "misaligned year order"
            raise ValueError(f"{key} {detail}; expected years {expected_years}")
    return expected_years


def _coerce_numeric_series(series: pd.Series, key: str) -> pd.Series:
    """Cast a series to float values, raising a descriptive error on failure."""

    _ensure_pandas()

    try:
        numeric = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} values must be numeric") from exc
    return numeric


def series_from_year_map(cfg: Mapping[str, Any], key: str) -> pd.Series:
    """Extract a pandas Series keyed by year from ``cfg`` for ``key``."""

    _ensure_pandas()

    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    if key not in cfg:
        raise KeyError(f"Configuration missing required key: {key}")

    entry = cfg[key]
    year_map, fill_forward = _extract_year_map(entry)
    if fill_forward and key != "floor":
        raise ValueError(f"Fill-forward is only supported for 'floor', not {key!r}")

    normalized = _normalize_year_value_pairs(year_map, key)
    series = pd.Series(normalized).sort_index()
    series.name = key

    target_years = []
    cfg_years = _coerce_year_list(cfg.get("years")) if "years" in cfg else []
    if cfg_years:
        target_years = cfg_years
    elif fill_forward and not series.empty:
        start, end = int(series.index.min()), int(series.index.max())
        target_years = list(range(start, end + 1))

    if target_years:
        reindexed = series.reindex(target_years)
        if fill_forward:
            reindexed = reindexed.ffill()
        series = reindexed

    if series.isna().any():
        missing_years = [int(idx) for idx, value in series.items() if pd.isna(value)]
        raise ValueError(f"{key} missing values for years: {missing_years}")

    if not target_years:
        _ensure_contiguous(series.index, key)

    series.attrs["fill_forward"] = fill_forward
    return series


def load_annual_policy(cfg: Mapping[str, Any]) -> RGGIPolicyAnnual:
    """Construct an :class:`RGGIPolicyAnnual` from a configuration mapping."""

    _ensure_pandas()

    cap_series = _coerce_numeric_series(series_from_year_map(cfg, "cap"), "cap")
    years = [int(year) for year in cap_series.index]

    cfg_with_years = dict(cfg)
    cfg_with_years["years"] = years

    floor_series = _coerce_numeric_series(series_from_year_map(cfg_with_years, "floor"), "floor")
    ccr1_trigger = _coerce_numeric_series(series_from_year_map(cfg_with_years, "ccr1_trigger"), "ccr1_trigger")
    ccr1_qty = _coerce_numeric_series(series_from_year_map(cfg_with_years, "ccr1_qty"), "ccr1_qty")
    ccr2_trigger = _coerce_numeric_series(series_from_year_map(cfg_with_years, "ccr2_trigger"), "ccr2_trigger")
    ccr2_qty = _coerce_numeric_series(series_from_year_map(cfg_with_years, "ccr2_qty"), "ccr2_qty")

    cp_key = "cp_id" if "cp_id" in cfg else "control_period"
    if cp_key not in cfg:
        raise KeyError("Configuration must include 'cp_id' or 'control_period'")
    cp_series = series_from_year_map(cfg_with_years, cp_key).astype(str)
    cp_series.name = "cp_id"

    _validate_year_alignment(
        {
            "cap": cap_series,
            "floor": floor_series,
            "ccr1_trigger": ccr1_trigger,
            "ccr1_qty": ccr1_qty,
            "ccr2_trigger": ccr2_trigger,
            "ccr2_qty": ccr2_qty,
            "cp_id": cp_series,
        }
    )

    bank0 = float(cfg.get("bank0", 0.0))
    full_compliance_raw = cfg.get("full_compliance_years", set())
    if full_compliance_raw is None:
        full_compliance_years: set[int] = set()
    else:
        if isinstance(full_compliance_raw, (str, bytes)):
            raise TypeError("full_compliance_years must be an iterable of years")
        try:
            full_compliance_years = {int(year) for year in full_compliance_raw}
        except (TypeError, ValueError) as exc:
            raise TypeError("full_compliance_years must be an iterable of years") from exc

    annual_surrender_frac = float(cfg.get("annual_surrender_frac", 0.5))
    carry_pct = float(cfg.get("carry_pct", 1.0))

    return RGGIPolicyAnnual(
        cap=cap_series,
        floor=floor_series,
        ccr1_trigger=ccr1_trigger,
        ccr1_qty=ccr1_qty,
        ccr2_trigger=ccr2_trigger,
        ccr2_qty=ccr2_qty,
        cp_id=cp_series,
        bank0=float(bank0),
        full_compliance_years=full_compliance_years,
        annual_surrender_frac=float(annual_surrender_frac),
        carry_pct=float(carry_pct),
    )


__all__ = ["series_from_year_map", "load_annual_policy"]
