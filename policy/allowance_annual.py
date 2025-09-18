"""Annual allowance market with banking and compliance logic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import pandas as pd


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
    full_compliance_years: Set[int]
    annual_surrender_frac: float = 0.5
    carry_pct: float = 1.0


class AllowanceAnnual:
    """Clears annual allowance markets and tracks bank/compliance state."""

    def __init__(self, policy: RGGIPolicyAnnual):
        self.policy = policy
        self.carry_pct = float(policy.carry_pct)
        self.cp_state: Dict[str, Dict[str, float | list[int]]] = {}
        self.year_records: Dict[int, Dict[str, float | str | bool]] = {}
        self.bank_history: Dict[int, float] = {}
        self.obligation_history: Dict[int, float] = {}
        self.finalized_results: Dict[int, Dict[str, float | bool | str]] = {}

    # ----------------------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------------------
    def _value(self, series: pd.Series, year: int, default: float = 0.0) -> float:
        """Return a scalar value from a Series/dict, defaulting if missing."""

        if series is None:
            return float(default)
        if isinstance(series, pd.Series):
            raw = series.get(year, default)
        else:  # allow dict-like fall-back
            getter = getattr(series, 'get', None)
            raw = getter(year, default) if callable(getter) else default
        if raw is None or pd.isna(raw):
            return float(default)
        return float(raw)

    def _cp_for_year(self, year: int) -> str:
        cp_series = self.policy.cp_id
        if isinstance(cp_series, pd.Series):
            cp = cp_series.get(year)
        else:
            getter = getattr(cp_series, 'get', None)
            cp = getter(year) if callable(getter) else None
        if cp is None:
            raise KeyError(f'Control period id missing for year {year}')
        return str(cp)

    def _get_cp_state(self, cp: str) -> Dict[str, float | list[int]]:
        state = self.cp_state.setdefault(cp, {'emissions': 0.0, 'surrendered': 0.0, 'years': []})
        if 'years' not in state:
            state['years'] = []
        return state

    def _remove_existing_year(self, year: int) -> None:
        record = self.year_records.pop(year, None)
        if record is None:
            return
        cp = str(record['cp_id'])
        state = self._get_cp_state(cp)
        state['emissions'] = float(state.get('emissions', 0.0)) - float(record.get('emissions', 0.0))
        state['surrendered'] = float(state.get('surrendered', 0.0)) - float(record.get('surrendered', 0.0))
        state['emissions'] = max(0.0, state['emissions'])
        state['surrendered'] = max(0.0, state['surrendered'])
        years_list = state.get('years', [])
        if isinstance(years_list, list) and year in years_list:
            years_list.remove(year)
        self.bank_history.pop(year, None)
        self.obligation_history.pop(year, None)
        self.finalized_results.pop(year, None)

    # ----------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------
    def clear_year(
        self,
        year: int,
        emissions_tons: float,
        bank_prev: float,
        expected_price_guess: float | None = None,
    ) -> dict:
        """Clear the allowance market for ``year`` and update ledgers."""

        self._remove_existing_year(year)

        emissions = max(0.0, float(emissions_tons))
        bank_prev = max(0.0, float(bank_prev))

        cap = self._value(self.policy.cap, year)
        floor = self._value(self.policy.floor, year)
        ccr1_trigger = self._value(self.policy.ccr1_trigger, year)
        ccr1_qty = self._value(self.policy.ccr1_qty, year)
        ccr2_trigger = self._value(self.policy.ccr2_trigger, year)
        ccr2_qty = self._value(self.policy.ccr2_qty, year)
        cp = self._cp_for_year(year)

        state = self._get_cp_state(cp)
        years_list = state.setdefault('years', [])
        if year not in years_list:
            years_list.append(year)

        outstanding_prev = float(state.get('emissions', 0.0)) - float(state.get('surrendered', 0.0))
        outstanding_prev = max(0.0, outstanding_prev)

        price = max(0.0, floor)
        available = bank_prev + cap
        ccr1_issued = 0.0
        ccr2_issued = 0.0

        def issue(trigger: float, qty: float) -> float:
            nonlocal available, price
            if qty <= 0.0:
                return 0.0
            shortfall = emissions - available
            if shortfall <= 0.0:
                return 0.0
            issued = min(qty, shortfall)
            available += issued
            price = max(price, trigger)
            return issued

        ccr1_issued = issue(ccr1_trigger, ccr1_qty)
        ccr2_issued = issue(ccr2_trigger, ccr2_qty)

        available_allowances = bank_prev + cap + ccr1_issued + ccr2_issued
        shortage_flag = emissions > available_allowances + 1e-9

        frac = max(0.0, min(1.0, float(self.policy.annual_surrender_frac)))
        required_current = frac * emissions
        outstanding_before = outstanding_prev + emissions
        surrender_target = min(required_current, outstanding_before)
        surrendered = min(surrender_target, available_allowances)

        new_emissions_total = float(state.get('emissions', 0.0)) + emissions
        new_surrender_total = float(state.get('surrendered', 0.0)) + surrendered
        state['emissions'] = new_emissions_total
        state['surrendered'] = new_surrender_total

        obligation = max(0.0, new_emissions_total - new_surrender_total)
        bank_unadjusted = max(0.0, available_allowances - surrendered)
        bank_new = bank_unadjusted * self.carry_pct

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

        self.year_records[year] = record
        self.bank_history[year] = bank_new
        self.obligation_history[year] = obligation

        return record.copy()

    def finalize_period_if_needed(self, year: int) -> dict:
        """Apply control-period true-up if ``year`` requires full compliance."""

        if year in self.finalized_results:
            return self.finalized_results[year].copy()

        if year not in self.policy.full_compliance_years:
            summary = {'finalized': False, 'bank_final': self.bank_history.get(year)}
            self.finalized_results[year] = summary
            return summary.copy()

        record = self.year_records.get(year)
        if record is None:
            raise ValueError(f'Year {year} has not been cleared before finalization')

        cp = str(record['cp_id'])
        state = self._get_cp_state(cp)
        bank_available = self.bank_history.get(year, 0.0)

        outstanding = max(0.0, float(state.get('emissions', 0.0)) - float(state.get('surrendered', 0.0)))
        surrender_additional = min(outstanding, bank_available)
        new_bank = bank_available - surrender_additional
        state['surrendered'] = float(state.get('surrendered', 0.0)) + surrender_additional
        remaining_obligation = max(0.0, float(state.get('emissions', 0.0)) - float(state.get('surrendered', 0.0)))
        shortage_flag = remaining_obligation > 1e-9

        if shortage_flag:
            new_bank = 0.0
        self.bank_history[year] = new_bank
        self.obligation_history[year] = remaining_obligation

        summary = {
            'finalized': True,
            'cp_id': cp,
            'surrendered_additional': surrender_additional,
            'bank_final': new_bank,
            'remaining_obligation': remaining_obligation,
            'shortage_flag': shortage_flag,
        }
        self.finalized_results[year] = summary

        if not shortage_flag:
            state['emissions'] = 0.0
            state['surrendered'] = 0.0
            state['years'] = []

        return summary.copy()
