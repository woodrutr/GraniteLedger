"""Centralised access to validated model input frames."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import pandas as pd

from policy.allowance_annual import RGGIPolicyAnnual

_DEMAND_KEY = "demand"
_UNITS_KEY = "units"
_FUELS_KEY = "fuels"
_TRANSMISSION_KEY = "transmission"
_COVERAGE_KEY = "coverage"
_POLICY_KEY = "policy"


def _normalize_name(name: str) -> str:
    """Normalise frame identifiers to a consistent string key."""

    if not isinstance(name, str):  # pragma: no cover - defensive programming
        name = str(name)
    return name.lower()


def _ensure_dataframe(name: str, value: object) -> pd.DataFrame:
    """Return a defensive copy of ``value`` ensuring it is a DataFrame."""

    if not isinstance(value, pd.DataFrame):
        raise TypeError(f'frame "{name}" must be provided as a pandas DataFrame')
    return value.copy(deep=True)


def _validate_columns(frame: str, df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Ensure ``df`` contains the ``required`` columns, returning a copy."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        columns = ", ".join(missing)
        raise ValueError(f'{frame} frame is missing required columns: {columns}')
    return df.copy(deep=True)


def _require_numeric(frame: str, column: str, series: pd.Series) -> pd.Series:
    """Return ``series`` coerced to numeric values ensuring no missing data."""

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{frame} frame column '{column}' must contain numeric values")
    return numeric


@dataclass(frozen=True)
class PolicySpec:
    """Data contract for allowance policy information."""

    cap: pd.Series
    floor: pd.Series
    ccr1_trigger: pd.Series
    ccr1_qty: pd.Series
    ccr2_trigger: pd.Series
    ccr2_qty: pd.Series
    cp_id: pd.Series
    bank0: float
    full_compliance_years: set[int]
    annual_surrender_frac: float
    carry_pct: float

    def to_policy(self) -> RGGIPolicyAnnual:
        """Instantiate :class:`RGGIPolicyAnnual` from the stored specification."""

        return RGGIPolicyAnnual(
            cap=self.cap,
            floor=self.floor,
            ccr1_trigger=self.ccr1_trigger,
            ccr1_qty=self.ccr1_qty,
            ccr2_trigger=self.ccr2_trigger,
            ccr2_qty=self.ccr2_qty,
            cp_id=self.cp_id,
            bank0=self.bank0,
            full_compliance_years=self.full_compliance_years,
            annual_surrender_frac=self.annual_surrender_frac,
            carry_pct=self.carry_pct,
        )


class Frames(Mapping[str, pd.DataFrame]):
    """Light-weight container offering validated access to model data frames."""

    def __init__(self, frames: Mapping[str, pd.DataFrame] | None = None):
        self._frames: Dict[str, pd.DataFrame] = {}
        if frames:
            for name, df in frames.items():
                key = _normalize_name(name)
                self._frames[key] = _ensure_dataframe(name, df)

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> pd.DataFrame:
        normalized = _normalize_name(key)
        if normalized not in self._frames:
            raise KeyError(f'frame {key!r} is not present')
        return self._frames[normalized].copy(deep=True)

    def __iter__(self) -> Iterator[str]:
        return iter(self._frames)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def coerce(cls, frames: Frames | Mapping[str, pd.DataFrame] | None) -> Frames:
        """Return ``frames`` as a :class:`Frames` instance."""

        if frames is None:
            raise ValueError('frames must be supplied as a Frames instance or mapping')
        if isinstance(frames, cls):
            return frames
        if isinstance(frames, Mapping):
            return cls(frames)
        raise TypeError('frames must be provided as Frames or a mapping of names to DataFrames')

    def with_frame(self, name: str, df: pd.DataFrame) -> "Frames":
        """Return a new container with ``name`` replaced by ``df``."""

        updated = dict(self._frames)
        updated[_normalize_name(name)] = _ensure_dataframe(name, df)
        return Frames(updated)

    # ------------------------------------------------------------------
    # Accessors with schema validation
    # ------------------------------------------------------------------
    def demand(self) -> pd.DataFrame:
        """Return validated demand data with columns (year, region, demand_mwh)."""

        df = self[_DEMAND_KEY]
        df = _validate_columns('demand', df, ['year', 'region', 'demand_mwh'])
        df['year'] = _require_numeric('demand', 'year', df['year']).astype(int)
        df['region'] = df['region'].astype(str)
        df['demand_mwh'] = _require_numeric('demand', 'demand_mwh', df['demand_mwh']).astype(float)

        duplicates = df.duplicated(subset=['year', 'region'])
        if duplicates.any():
            dupes = df.loc[duplicates, ['year', 'region']].to_records(index=False)
            raise ValueError(
                'demand frame contains duplicate year/region pairs: '
                + ', '.join(f'({year}, {region})' for year, region in dupes)
            )

        return df.sort_values(['year', 'region']).reset_index(drop=True)

    def demand_for_year(self, year: int) -> Dict[str, float]:
        """Return demand by region for ``year`` as a mapping."""

        demand = self.demand()
        filtered = demand[demand['year'] == int(year)]
        if filtered.empty:
            raise KeyError(f'demand for year {year} is unavailable')
        grouped = filtered.groupby('region')['demand_mwh'].sum()
        return {str(region): float(value) for region, value in grouped.items()}

    def units(self) -> pd.DataFrame:
        """Return validated generating unit characteristics."""

        required = [
            'unit_id',
            'region',
            'fuel',
            'cap_mw',
            'availability',
            'hr_mmbtu_per_mwh',
            'vom_per_mwh',
            'fuel_price_per_mmbtu',
            'ef_ton_per_mwh',
        ]
        df = self[_UNITS_KEY]
        df = _validate_columns('units', df, required)

        df['unit_id'] = df['unit_id'].astype(str)
        if df['unit_id'].duplicated().any():
            duplicates = sorted(df.loc[df['unit_id'].duplicated(), 'unit_id'].unique())
            raise ValueError('units frame contains duplicate unit_id values: ' + ', '.join(duplicates))

        df['region'] = df['region'].astype(str)
        df['fuel'] = df['fuel'].astype(str)

        numeric_columns = [
            'cap_mw',
            'availability',
            'hr_mmbtu_per_mwh',
            'vom_per_mwh',
            'fuel_price_per_mmbtu',
            'ef_ton_per_mwh',
        ]
        for column in numeric_columns:
            df[column] = _require_numeric('units', column, df[column]).astype(float)

        df['availability'] = df['availability'].clip(lower=0.0, upper=1.0)

        return df.reset_index(drop=True)

    def fuels(self) -> pd.DataFrame:
        """Return validated fuel metadata (fuel label and coverage flag)."""

        df = self[_FUELS_KEY]
        df = _validate_columns('fuels', df, ['fuel', 'covered'])
        df['fuel'] = df['fuel'].astype(str)
        if df['fuel'].duplicated().any():
            duplicates = sorted(df.loc[df['fuel'].duplicated(), 'fuel'].unique())
            raise ValueError('fuels frame contains duplicate fuel labels: ' + ', '.join(duplicates))

        df['covered'] = df['covered'].astype(bool)
        if 'co2_ton_per_mmbtu' in df.columns:
            df['co2_ton_per_mmbtu'] = _require_numeric('fuels', 'co2_ton_per_mmbtu', df['co2_ton_per_mmbtu']).astype(float)

        return df.reset_index(drop=True)

    def transmission(self) -> pd.DataFrame:
        """Return validated transmission limits or an empty frame if absent."""

        if _TRANSMISSION_KEY not in self._frames:
            return pd.DataFrame(columns=['from_region', 'to_region', 'limit_mw'])

        df = self[_TRANSMISSION_KEY]
        if df.empty:
            return pd.DataFrame(columns=['from_region', 'to_region', 'limit_mw'])

        df = _validate_columns('transmission', df, ['from_region', 'to_region', 'limit_mw'])
        df['from_region'] = df['from_region'].astype(str)
        df['to_region'] = df['to_region'].astype(str)
        df['limit_mw'] = _require_numeric('transmission', 'limit_mw', df['limit_mw']).astype(float)

        if (df['limit_mw'] < 0.0).any():
            raise ValueError('transmission frame must specify non-negative limits')

        return df.reset_index(drop=True)

    def transmission_limits(self) -> Dict[Tuple[str, str], float]:
        """Return transmission limits keyed by region pairs."""

        frame = self.transmission()
        limits: Dict[Tuple[str, str], float] = {}
        for row in frame.itertuples(index=False):
            key = (str(row.from_region), str(row.to_region))
            limits[key] = float(row.limit_mw)
        return limits

    def coverage(self) -> pd.DataFrame:
        """Return validated coverage flags by region and (optionally) year."""

        if _COVERAGE_KEY not in self._frames:
            return pd.DataFrame(columns=['region', 'year', 'covered'])

        df = self[_COVERAGE_KEY]
        df = _validate_columns('coverage', df, ['region', 'covered'])

        df['region'] = df['region'].astype(str)
        df['covered'] = df['covered'].astype(bool)

        if 'year' in df.columns:
            year_series = df['year']
            default_mask = year_series.isna()
            if default_mask.any():
                df.loc[default_mask, 'year'] = -1
            df['year'] = _require_numeric('coverage', 'year', df['year']).astype(int)
        else:
            df = df.assign(year=-1)

        duplicates = df.duplicated(subset=['region', 'year'])
        if duplicates.any():
            dupes = df.loc[duplicates, ['region', 'year']].to_records(index=False)
            raise ValueError(
                'coverage frame contains duplicate region/year combinations: '
                + ', '.join(f'({region}, {year})' for region, year in dupes)
            )

        return df[['region', 'year', 'covered']].reset_index(drop=True)

    def coverage_for_year(self, year: int) -> Dict[str, bool]:
        """Return coverage flags for ``year`` keyed by model region."""

        coverage = self.coverage()
        if coverage.empty:
            return {}

        year = int(year)
        mapping = {
            str(row.region): bool(row.covered)
            for row in coverage.itertuples(index=False)
            if int(row.year) == -1
        }

        for row in coverage.itertuples(index=False):
            if int(row.year) == year:
                mapping[str(row.region)] = bool(row.covered)

        return mapping

    def policy(self) -> PolicySpec:
        """Return the allowance policy specification."""

        df = self[_POLICY_KEY]
        required = [
            'year',
            'cap_tons',
            'floor_dollars',
            'ccr1_trigger',
            'ccr1_qty',
            'ccr2_trigger',
            'ccr2_qty',
            'cp_id',
            'full_compliance',
            'bank0',
            'annual_surrender_frac',
            'carry_pct',
        ]
        df = _validate_columns('policy', df, required)

        df['year'] = _require_numeric('policy', 'year', df['year']).astype(int)
        if df['year'].duplicated().any():
            duplicates = sorted(df.loc[df['year'].duplicated(), 'year'].unique())
            raise ValueError('policy frame contains duplicate years: ' + ', '.join(map(str, duplicates)))

        numeric_columns = [
            'cap_tons',
            'floor_dollars',
            'ccr1_trigger',
            'ccr1_qty',
            'ccr2_trigger',
            'ccr2_qty',
            'bank0',
            'annual_surrender_frac',
            'carry_pct',
        ]
        for column in numeric_columns:
            df[column] = _require_numeric('policy', column, df[column]).astype(float)

        df['cp_id'] = df['cp_id'].astype(str)
        df['full_compliance'] = df['full_compliance'].astype(bool)

        bank_values = df['bank0'].unique()
        if len(bank_values) != 1:
            raise ValueError('policy frame must provide a single bank0 value shared across years')
        bank0 = float(bank_values[0])

        surrender_values = df['annual_surrender_frac'].unique()
        if len(surrender_values) != 1:
            raise ValueError('policy frame must provide a single annual_surrender_frac value shared across years')
        annual_surrender_frac = float(surrender_values[0])

        carry_values = df['carry_pct'].unique()
        if len(carry_values) != 1:
            raise ValueError('policy frame must provide a single carry_pct value shared across years')
        carry_pct = float(carry_values[0])

        index = df['year']
        cap = pd.Series(df['cap_tons'].values, index=index)
        floor = pd.Series(df['floor_dollars'].values, index=index)
        ccr1_trigger = pd.Series(df['ccr1_trigger'].values, index=index)
        ccr1_qty = pd.Series(df['ccr1_qty'].values, index=index)
        ccr2_trigger = pd.Series(df['ccr2_trigger'].values, index=index)
        ccr2_qty = pd.Series(df['ccr2_qty'].values, index=index)
        cp_id = pd.Series(df['cp_id'].values, index=index)

        full_compliance_years = {int(year) for year, flag in zip(df['year'], df['full_compliance']) if flag}

        return PolicySpec(
            cap=cap,
            floor=floor,
            ccr1_trigger=ccr1_trigger,
            ccr1_qty=ccr1_qty,
            ccr2_trigger=ccr2_trigger,
            ccr2_qty=ccr2_qty,
            cp_id=cp_id,
            bank0=bank0,
            full_compliance_years=full_compliance_years,
            annual_surrender_frac=annual_surrender_frac,
            carry_pct=carry_pct,
        )


__all__ = ['Frames', 'PolicySpec']

