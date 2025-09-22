"""Annual allowance market with banking and compliance logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Set, TYPE_CHECKING, cast

try:  # pragma: no cover - exercised via ImportError handling
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


def _ensure_pandas() -> None:
    """Raise a descriptive error if :mod:`pandas` is unavailable."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for policy.allowance_annual; install it with `pip install pandas`."
        )


class ConfigError(ValueError):
    """Configuration error raised when allowance policy inputs are invalid."""


def _infer_compliance_year(label: Any, resolution: str) -> int:
    """Return the calendar year associated with ``label`` for ``resolution``."""

    if isinstance(label, Integral):
        value = int(label)
        if resolution != "annual" and abs(value) >= 1000:
            return int(value // 1000)
        return value
    if isinstance(label, Real) and not isinstance(label, bool):
        value = int(label)
        if resolution != "annual" and abs(value) >= 1000:
            return int(value // 1000)
        return value
    if isinstance(label, str):
        stripped = label.strip()
        if stripped:
            digits = "".join(ch for ch in stripped if ch.isdigit())
            if digits:
                if resolution != "annual" and len(digits) > 4:
                    return int(digits[:4])
                return int(digits)
    if pd is not None:
        try:
            timestamp = pd.to_datetime(label, errors="coerce")
        except Exception:  # pragma: no cover - defensive
            timestamp = None
        if timestamp is not None and not pd.isna(timestamp):
            return int(timestamp.year)
    raise ValueError(f"Unable to infer compliance year for period label {label!r}")


@dataclass
class RGGIPolicyAnnual:
    """Inputs describing an annual-resolution allowance market."""

    cap: pd.Series
    floor: pd.Series
    ccr1_trigger: pd.Series
    ccr1_qty: pd.Series
    ccr2_trigger: pd.Series
    ccr2_qty: pd.Series
    cp_id: pd.Series
    bank0: float
    full_compliance_years: Set[int] | None
    annual_surrender_frac: float = 0.5
    carry_pct: float = 1.0
    banking_enabled: bool = True
    enabled: bool = True
    ccr1_enabled: bool = True
    ccr2_enabled: bool = True
    control_period_length: int | None = None
    resolution: str = "annual"
    _period_to_year_map: dict[Any, int] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_pandas()

        full_compliance: set[int]
        if self.full_compliance_years is None:
            full_compliance = set()
        else:
            full_compliance = {int(year) for year in self.full_compliance_years}
        self.resolution = str(self.resolution or "annual").strip().lower()

        index_obj = getattr(self.cap, "index", None)
        index_values = list(index_obj) if index_obj is not None else []
        period_to_year: dict[Any, int] = {}
        for label in index_values:
            try:
                period_to_year[label] = _infer_compliance_year(label, self.resolution)
            except ValueError:  # pragma: no cover - defensive guard
                try:
                    period_to_year[label] = int(label)
                except (TypeError, ValueError):
                    period_to_year[label] = label  # type: ignore[assignment]
        self._period_to_year_map = period_to_year

        if self.full_compliance_years is None:
            full_compliance = set()
        else:
            full_compliance = set()
            for entry in self.full_compliance_years:
                if entry in period_to_year:
                    full_compliance.add(int(entry))  # maintain backward compatibility
                else:
                    try:
                        target_year = _infer_compliance_year(entry, self.resolution)
                    except ValueError:
                        continue
                    for label, calendar_year in period_to_year.items():
                        if calendar_year == target_year:
                            full_compliance.add(int(label))
            if not full_compliance and self.full_compliance_years:
                for entry in self.full_compliance_years:
                    try:
                        full_compliance.add(int(entry))
                    except (TypeError, ValueError):
                        continue
        self.full_compliance_years = full_compliance

        self.enabled = bool(self.enabled)
        self.ccr1_enabled = bool(self.ccr1_enabled)
        self.ccr2_enabled = bool(self.ccr2_enabled)
        self.banking_enabled = bool(self.banking_enabled)

        if not self.enabled:
            self.banking_enabled = False

        if not self.banking_enabled:
            self.bank0 = 0.0
            self.carry_pct = 0.0

        if self.control_period_length is not None:
            try:
                control_length = int(self.control_period_length)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError('control_period_length must be an integer or None') from exc
            if control_length <= 0:
                raise ValueError('control_period_length must be a positive integer')
            self.control_period_length = control_length
        else:
            control_length = None

        if not self.full_compliance_years and control_length:
            series_index = getattr(self.cap, 'index', None)
            if series_index is not None:
                ordered_labels = list(series_index)
                compliance_periods = {
                    int(ordered_labels[idx - 1])
                    for idx in range(control_length, len(ordered_labels) + 1, control_length)
                }
                self.full_compliance_years = compliance_periods

        if self.enabled:
            missing: list[str] = []
            series_requirements = {
                'cap': self.cap,
                'floor': self.floor,
                'ccr1_trigger': self.ccr1_trigger,
                'ccr1_qty': self.ccr1_qty,
                'ccr2_trigger': self.ccr2_trigger,
                'ccr2_qty': self.ccr2_qty,
                'cp_id': self.cp_id,
            }
            for name, series in series_requirements.items():
                if series is None:
                    missing.append(name)
                    continue
                length = getattr(series, '__len__', None)
                if callable(length):
                    try:
                        if len(series) == 0:  # type: ignore[arg-type]
                            missing.append(name)
                    except TypeError:  # pragma: no cover - defensive guard
                        continue
            if missing:
                missing_list = ', '.join(sorted(missing))
                raise ConfigError(
                    f'enabled carbon policy requires data for: {missing_list}'
                )

    def compliance_year_for(self, period: Any) -> int:
        """Return the calendar year associated with ``period``."""

        if period in self._period_to_year_map:
            year_value = self._period_to_year_map[period]
            if isinstance(year_value, Integral):
                return int(year_value)
            try:
                return int(year_value)
            except (TypeError, ValueError):
                pass
        try:
            return _infer_compliance_year(period, self.resolution)
        except ValueError:
            try:
                return int(period)  # pragma: no cover - fallback
            except (TypeError, ValueError):
                return hash(period)


@dataclass(frozen=True)
class ControlPeriodLedger:
    """Aggregate compliance totals tracked for a control period."""

    emissions: float = 0.0
    surrendered: float = 0.0
    years: tuple[int, ...] = ()


@dataclass(frozen=True)
class AllowanceMarketState:
    """Immutable snapshot of the allowance market ledger."""

    cp_totals: dict[str, ControlPeriodLedger] = field(default_factory=dict)
    year_records: dict[int, dict[str, float | str | bool]] = field(default_factory=dict)
    bank_history: dict[int, float] = field(default_factory=dict)
    obligation_history: dict[int, float] = field(default_factory=dict)
    finalized_results: dict[int, dict[str, float | bool | str]] = field(default_factory=dict)


def allowance_initial_state() -> AllowanceMarketState:
    """Return a freshly initialized :class:`AllowanceMarketState`."""

    return AllowanceMarketState()


def _coerce_state(state: AllowanceMarketState | None) -> AllowanceMarketState:
    if state is None:
        return AllowanceMarketState()
    return state


def _value(series: pd.Series, year: int, default: float = 0.0) -> float:
    """Return a scalar policy value for ``year`` with ``default`` fallback."""

    if series is None:
        return float(default)
    if isinstance(series, pd.Series):
        raw = series.get(year, default)
    else:
        getter = getattr(series, 'get', None)
        raw = getter(year, default) if callable(getter) else default
    if raw is None or pd.isna(raw):
        return float(default)
    return float(raw)


def _cp_for_year(policy: RGGIPolicyAnnual, year: int) -> str:
    cp_series = policy.cp_id
    if isinstance(cp_series, pd.Series):
        cp = cp_series.get(year)
    else:
        getter = getattr(cp_series, 'get', None)
        cp = getter(year) if callable(getter) else None
    if cp is None:
        raise KeyError(f'Control period id missing for year {year}')
    return str(cp)


def _add_year(years: tuple[int, ...], year: int) -> tuple[int, ...]:
    if year in years:
        return years
    return years + (year,)


def _subtract_record_from_totals(
    totals: ControlPeriodLedger,
    record: dict[str, float | str | bool],
    year: int,
) -> ControlPeriodLedger | None:
    emissions_removed = float(record.get('emissions', 0.0))
    surrendered_removed = float(record.get('surrendered', 0.0))
    new_emissions = max(0.0, totals.emissions - emissions_removed)
    new_surrendered = max(0.0, totals.surrendered - surrendered_removed)
    new_years = tuple(y for y in totals.years if y != year)
    if new_emissions <= 0.0 and new_surrendered <= 0.0 and not new_years:
        return None
    return ControlPeriodLedger(new_emissions, new_surrendered, new_years)


_PRICE_SOLVER_TOL = 1e-6
_PRICE_SOLVER_MAX_ITER = 50
_PRICE_SOLVER_HIGH = 1000.0
_PRICE_SPAN_EPS = 1e-6


def clear_year(
    policy: RGGIPolicyAnnual,
    state: AllowanceMarketState | None,
    year: int,
    *,
    emissions_tons: float,
    bank_prev: float,
    expected_price_guess: float | None = None,
) -> tuple[dict[str, float | str | bool], AllowanceMarketState]:
    """Clear the allowance market for ``year`` and return the new ledger state."""

    _ensure_pandas()
    base_state = _coerce_state(state)

    cp_totals = dict(base_state.cp_totals)
    year_records = dict(base_state.year_records)
    bank_history = dict(base_state.bank_history)
    obligation_history = dict(base_state.obligation_history)
    finalized_results = dict(base_state.finalized_results)

    previous_record = year_records.pop(year, None)
    if previous_record is not None:
        cp_prev = str(previous_record.get('cp_id', ''))
        totals_prev = cp_totals.get(cp_prev)
        if totals_prev is not None:
            updated_totals = _subtract_record_from_totals(totals_prev, previous_record, year)
            if updated_totals is None:
                cp_totals.pop(cp_prev, None)
            else:
                cp_totals[cp_prev] = updated_totals
    bank_history.pop(year, None)
    obligation_history.pop(year, None)
    finalized_results.pop(year, None)

    banking_enabled = bool(getattr(policy, 'banking_enabled', True))

    emissions = max(0.0, float(emissions_tons))
    bank_prev = max(0.0, float(bank_prev))
    if not banking_enabled:
        bank_prev = 0.0

    if not getattr(policy, 'enabled', True):
        record = {
            'year': year,
            'cp_id': 'disabled',
            'emissions': emissions,
            'bank_prev': bank_prev,
            'available_allowances': emissions,
            'p_co2': 0.0,
            'ccr1_issued': 0.0,
            'ccr2_issued': 0.0,
            'surrendered': 0.0,
            'shortage_flag': False,
        }
        year_records[year] = dict(record)
        new_state = AllowanceMarketState(
            cp_totals=cp_totals,
            year_records=year_records,
            bank_history=bank_history,
            obligation_history=obligation_history,
            finalized_results=finalized_results,
        )
        return dict(record), new_state

    cap = max(0.0, _value(policy.cap, year))
    floor = float(_value(policy.floor, year))
    ccr1_trigger = float(_value(policy.ccr1_trigger, year))
    ccr1_qty_raw = _value(policy.ccr1_qty, year)
    ccr2_trigger = float(_value(policy.ccr2_trigger, year))
    ccr2_qty_raw = _value(policy.ccr2_qty, year)
    cp = _cp_for_year(policy, year)

    totals = cp_totals.get(cp, ControlPeriodLedger())

    outstanding_prev = max(0.0, totals.emissions - totals.surrendered)
    ccr1_enabled = bool(getattr(policy, 'ccr1_enabled', True))
    ccr2_enabled = bool(getattr(policy, 'ccr2_enabled', True))
    ccr1_qty = max(0.0, ccr1_qty_raw if ccr1_enabled else 0.0)
    ccr2_qty = max(0.0, ccr2_qty_raw if ccr2_enabled else 0.0)

    tol = _PRICE_SOLVER_TOL
    span_tol = max(tol, _PRICE_SPAN_EPS)
    max_iter = _PRICE_SOLVER_MAX_ITER

    min_price = max(0.0, floor)
    low_bound = min_price

    candidates = [min_price, _PRICE_SOLVER_HIGH]
    if expected_price_guess is not None:
        try:
            candidates.append(float(expected_price_guess))
        except (TypeError, ValueError):
            pass
    if ccr1_qty > 0.0:
        candidates.append(ccr1_trigger)
    if ccr2_qty > 0.0:
        candidates.append(ccr2_trigger)
    high_bound = max(candidates)

    def minted_allowances(price: float) -> tuple[float, float, float]:
        minted = cap
        issued1 = 0.0
        issued2 = 0.0
        if ccr1_qty > 0.0 and price >= ccr1_trigger:
            issued1 = ccr1_qty
            minted += issued1
        if ccr2_qty > 0.0 and price >= ccr2_trigger:
            issued2 = ccr2_qty
            minted += issued2
        return minted, issued1, issued2

    minted_low, ccr1_low, ccr2_low = minted_allowances(low_bound)
    total_allowances_low = bank_prev + minted_low
    if total_allowances_low >= emissions - tol:
        clearing_price = max(low_bound, min_price)
        if clearing_price != low_bound:
            minted_low, ccr1_low, ccr2_low = minted_allowances(clearing_price)
        minted_final = minted_low
        ccr1_final = ccr1_low
        ccr2_final = ccr2_low
        total_allowances = bank_prev + minted_final
    else:
        minted_high, ccr1_high, ccr2_high = minted_allowances(high_bound)
        total_allowances_high = bank_prev + minted_high
        if total_allowances_high < emissions - tol:
            clearing_price = max(high_bound, min_price)
            if clearing_price != high_bound:
                minted_high, ccr1_high, ccr2_high = minted_allowances(clearing_price)
            minted_final = minted_high
            ccr1_final = ccr1_high
            ccr2_final = ccr2_high
            total_allowances = bank_prev + minted_final
        else:
            best_price = high_bound
            best_minted = minted_high
            best_ccr1 = ccr1_high
            best_ccr2 = ccr2_high
            low_current = low_bound
            high_current = high_bound
            for iteration in range(1, max_iter + 1):
                mid = 0.5 * (low_current + high_current)
                minted_mid, ccr1_mid, ccr2_mid = minted_allowances(mid)
                total_allowances_mid = bank_prev + minted_mid
                if total_allowances_mid >= emissions - tol:
                    best_price = mid
                    best_minted = minted_mid
                    best_ccr1 = ccr1_mid
                    best_ccr2 = ccr2_mid
                    high_current = mid
                    if abs(minted_mid - emissions) <= tol:
                        break
                else:
                    low_current = mid
                if abs(high_current - low_current) <= span_tol:
                    break

            clearing_price = max(best_price, min_price)
            if clearing_price != best_price:
                best_minted, best_ccr1, best_ccr2 = minted_allowances(clearing_price)
            minted_final = best_minted
            ccr1_final = best_ccr1
            ccr2_final = best_ccr2
            total_allowances = bank_prev + minted_final

    shortage_flag = emissions > total_allowances + tol

    frac = max(0.0, min(1.0, float(policy.annual_surrender_frac)))
    required_current = frac * emissions
    outstanding_before = outstanding_prev + emissions
    surrender_target = min(required_current, outstanding_before)
    surrendered = min(surrender_target, total_allowances)

    new_emissions_total = totals.emissions + emissions
    new_surrender_total = totals.surrendered + surrendered

    updated_totals = ControlPeriodLedger(
        emissions=new_emissions_total,
        surrendered=new_surrender_total,
        years=_add_year(totals.years, year),
    )
    cp_totals[cp] = updated_totals

    obligation = max(0.0, new_emissions_total - new_surrender_total)
    bank_unadjusted = max(0.0, total_allowances - surrendered)
    carry_pct = float(getattr(policy, 'carry_pct', 1.0))
    if not banking_enabled:
        bank_unadjusted = 0.0
        carry_pct = 0.0
    bank_new = bank_unadjusted * carry_pct

    record = {
        'year': year,
        'cp_id': cp,
        'emissions': emissions,
        'bank_prev': bank_prev,
        'available_allowances': minted_final,
        'allowances_total': total_allowances,
        'p_co2': clearing_price,
        'ccr1_issued': ccr1_final,
        'ccr2_issued': ccr2_final,
        'surrendered': surrendered,
        'bank_new': bank_new,
        'obligation_new': obligation,
        'shortage_flag': shortage_flag,
    }

    year_records[year] = dict(record)
    bank_history[year] = bank_new if banking_enabled else 0.0
    obligation_history[year] = obligation

    new_state = AllowanceMarketState(
        cp_totals=cp_totals,
        year_records=year_records,
        bank_history=bank_history,
        obligation_history=obligation_history,
        finalized_results=finalized_results,
    )

    return dict(record), new_state


def finalize_period_if_needed(
    policy: RGGIPolicyAnnual,
    state: AllowanceMarketState | None,
    year: int,
) -> tuple[dict[str, float | bool | str], AllowanceMarketState]:
    """Apply control-period true-up if ``year`` requires full compliance."""

    _ensure_pandas()
    base_state = _coerce_state(state)

    banking_enabled = bool(getattr(policy, 'banking_enabled', True))

    existing = base_state.finalized_results.get(year)
    if existing is not None:
        return dict(existing), base_state

    cp_totals = dict(base_state.cp_totals)
    bank_history = dict(base_state.bank_history)
    obligation_history = dict(base_state.obligation_history)
    finalized_results = dict(base_state.finalized_results)

    if not getattr(policy, 'enabled', True):
        summary = {'finalized': False}
        finalized_results[year] = dict(summary)
        new_state = AllowanceMarketState(
            cp_totals=cp_totals,
            year_records=base_state.year_records,
            bank_history=bank_history,
            obligation_history=obligation_history,
            finalized_results=finalized_results,
        )
        return dict(summary), new_state

    if year not in policy.full_compliance_years:
        summary = {'finalized': False, 'bank_final': bank_history.get(year)}
        finalized_results[year] = dict(summary)
        new_state = AllowanceMarketState(
            cp_totals=cp_totals,
            year_records=base_state.year_records,
            bank_history=bank_history,
            obligation_history=obligation_history,
            finalized_results=finalized_results,
        )
        return dict(summary), new_state

    record = base_state.year_records.get(year)
    if record is None:
        raise ValueError(f'Year {year} has not been cleared before finalization')

    cp = str(record['cp_id'])
    totals = cp_totals.get(cp, ControlPeriodLedger())
    bank_available = float(bank_history.get(year, 0.0)) if banking_enabled else 0.0

    outstanding = max(0.0, totals.emissions - totals.surrendered)
    surrender_additional = min(outstanding, bank_available)
    new_bank = bank_available - surrender_additional
    new_surrender_total = totals.surrendered + surrender_additional
    remaining_obligation = max(0.0, totals.emissions - new_surrender_total)
    shortage_flag = remaining_obligation > 1e-9

    if shortage_flag:
        cp_totals[cp] = ControlPeriodLedger(
            emissions=totals.emissions,
            surrendered=new_surrender_total,
            years=totals.years,
        )
    else:
        cp_totals.pop(cp, None)

    if banking_enabled and not shortage_flag:
        bank_history[year] = new_bank
    else:
        bank_history[year] = 0.0
    remaining_obligation = remaining_obligation if shortage_flag else 0.0
    obligation_history[year] = remaining_obligation

    summary = {
        'finalized': True,
        'cp_id': cp,
        'surrendered_additional': surrender_additional,
        'bank_final': bank_history[year] if banking_enabled else 0.0,
        'remaining_obligation': remaining_obligation,
        'shortage_flag': shortage_flag,
    }
    if not banking_enabled:
        summary['bank_final'] = 0.0
    finalized_results[year] = dict(summary)

    new_state = AllowanceMarketState(
        cp_totals=cp_totals,
        year_records=base_state.year_records,
        bank_history=bank_history,
        obligation_history=obligation_history,
        finalized_results=finalized_results,
    )

    return dict(summary), new_state


__all__ = [
    "ConfigError",
    "RGGIPolicyAnnual",
    "ControlPeriodLedger",
    "AllowanceMarketState",
    "allowance_initial_state",
    "clear_year",
    "finalize_period_if_needed",
]
