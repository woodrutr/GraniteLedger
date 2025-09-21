"""Annual allowance market with banking and compliance logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, TYPE_CHECKING, cast

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

    def __post_init__(self) -> None:
        _ensure_pandas()

        full_compliance: set[int]
        if self.full_compliance_years is None:
            full_compliance = set()
        else:
            full_compliance = {int(year) for year in self.full_compliance_years}
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
                ordered_years = [int(year) for year in series_index]
                compliance_years = {
                    year
                    for idx, year in enumerate(ordered_years, start=1)
                    if idx % control_length == 0
                }
                self.full_compliance_years = compliance_years

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
    _ = expected_price_guess  # maintained for API compatibility
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

    cap = _value(policy.cap, year)
    floor = _value(policy.floor, year)
    ccr1_trigger = _value(policy.ccr1_trigger, year)
    ccr1_qty = _value(policy.ccr1_qty, year)
    ccr2_trigger = _value(policy.ccr2_trigger, year)
    ccr2_qty = _value(policy.ccr2_qty, year)
    cp = _cp_for_year(policy, year)

    totals = cp_totals.get(cp, ControlPeriodLedger())

    outstanding_prev = max(0.0, totals.emissions - totals.surrendered)

    price = max(0.0, floor)
    available = bank_prev + cap

    def issue(trigger: float, qty: float, enabled: bool) -> float:
        nonlocal available, price
        if not enabled or qty <= 0.0:
            return 0.0
        shortfall = emissions - available
        if shortfall <= 0.0:
            return 0.0
        issued = min(qty, shortfall)
        available += issued
        price = max(price, trigger)
        return issued

    ccr1_enabled = getattr(policy, 'ccr1_enabled', True)
    ccr2_enabled = getattr(policy, 'ccr2_enabled', True)
    ccr1_issued = issue(ccr1_trigger, ccr1_qty, ccr1_enabled)
    ccr2_issued = issue(ccr2_trigger, ccr2_qty, ccr2_enabled)

    available_allowances = bank_prev + cap + ccr1_issued + ccr2_issued
    shortage_flag = emissions > available_allowances + 1e-9

    frac = max(0.0, min(1.0, float(policy.annual_surrender_frac)))
    required_current = frac * emissions
    outstanding_before = outstanding_prev + emissions
    surrender_target = min(required_current, outstanding_before)
    surrendered = min(surrender_target, available_allowances)

    new_emissions_total = totals.emissions + emissions
    new_surrender_total = totals.surrendered + surrendered

    updated_totals = ControlPeriodLedger(
        emissions=new_emissions_total,
        surrendered=new_surrender_total,
        years=_add_year(totals.years, year),
    )
    cp_totals[cp] = updated_totals

    obligation = max(0.0, new_emissions_total - new_surrender_total)
    bank_unadjusted = max(0.0, available_allowances - surrendered)
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
        'available_allowances': available_allowances,
        'p_co2': price,
        'ccr1_issued': ccr1_issued,
        'ccr2_issued': ccr2_issued,
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
