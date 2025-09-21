"""Centralised access to validated model input frames."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
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


def _coerce_bool(series: pd.Series, frame: str, column: str) -> pd.Series:
    """Return ``series`` converted to booleans with explicit validation."""

    true_tokens = {"true", "t", "yes", "y", "on", "1"}
    false_tokens = {"false", "f", "no", "n", "off", "0"}

    def convert(value: object) -> bool:
        if pd.isna(value):
            raise ValueError
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, Integral):
            if value in (0, 1):
                return bool(value)
            raise ValueError
        if isinstance(value, Real):
            if value in (0.0, 1.0):
                return bool(int(value))
            raise ValueError
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in true_tokens:
                return True
            if normalised in false_tokens:
                return False
        raise ValueError

    try:
        coerced = series.map(convert)
    except ValueError as exc:
        raise ValueError(
            f"{frame} frame column '{column}' must contain boolean-like values"
        ) from exc

    return coerced.astype(bool)


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
    enabled: bool = True
    ccr1_enabled: bool = True
    ccr2_enabled: bool = True
    control_period_years: int | None = None

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
            enabled=self.enabled,
            ccr1_enabled=self.ccr1_enabled,
            ccr2_enabled=self.ccr2_enabled,
            control_period_length=self.control_period_years,
        )


class Frames(Mapping[str, pd.DataFrame]):
    """Light-weight container offering validated access to model data frames."""

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame] | None = None,
        *,
        carbon_policy_enabled: bool | None = None,
    ):
        self._frames: Dict[str, pd.DataFrame] = {}
        self._carbon_policy_enabled = True if carbon_policy_enabled is None else bool(
            carbon_policy_enabled
        )
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
        return Frames(updated, carbon_policy_enabled=self._carbon_policy_enabled)

    def _optional_frame(self, name: str) -> pd.DataFrame | None:
        """Return a defensive copy of ``name`` if present, otherwise ``None``."""

        key = _normalize_name(name)
        frame = self._frames.get(key)
        if frame is None:
            return None
        if not isinstance(frame, pd.DataFrame):  # pragma: no cover - defensive guard
            raise TypeError(f'frame "{name}" must be provided as a pandas DataFrame')
        return frame.copy(deep=True)

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------
    @property
    def carbon_policy_enabled(self) -> bool:
        """Return the cached carbon policy enabled flag."""

        return bool(self._carbon_policy_enabled)

    # ------------------------------------------------------------------
    # Accessors with schema validation
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Return validated system load indexed by region, year, and hour."""

        df = self['load']
        required = ['region', 'year', 'hour', 'Load']
        df = _validate_columns('Load', df, required)

        df['region'] = df['region'].astype(str)
        df['year'] = _require_numeric('Load', 'year', df['year']).astype(int)
        df['hour'] = _require_numeric('Load', 'hour', df['hour']).astype(int)
        df['Load'] = _require_numeric('Load', 'Load', df['Load']).astype(float)

        return df

    def supply_curve(self) -> pd.DataFrame:
        """Return validated supply curve data for electricity dispatch."""

        df = self['supplycurve']
        required = ['region', 'tech', 'step', 'year', 'SupplyCurve']
        df = _validate_columns('SupplyCurve', df, required)

        df['region'] = df['region'].astype(str)
        df['tech'] = _require_numeric('SupplyCurve', 'tech', df['tech']).astype(int)
        df['step'] = _require_numeric('SupplyCurve', 'step', df['step']).astype(int)
        df['year'] = _require_numeric('SupplyCurve', 'year', df['year']).astype(int)
        df['SupplyCurve'] = _require_numeric(
            'SupplyCurve', 'SupplyCurve', df['SupplyCurve']
        ).astype(float)

        if 'hour' in df.columns:
            df['hour'] = _require_numeric('SupplyCurve', 'hour', df['hour']).astype(int)
        if 'season' in df.columns:
            df['season'] = _require_numeric('SupplyCurve', 'season', df['season']).astype(int)

        return df

    def carbon_cap_group_membership(self) -> pd.DataFrame | None:
        """Return the cap group membership frame if supplied."""

        return self._optional_frame('CarbonCapGroupMembership')

    def carbon_allowance_procurement(self) -> pd.DataFrame | None:
        """Return allowance procurement quantities by cap group and year."""

        frame = self._optional_frame('CarbonAllowanceProcurement')
        if frame is None:
            return None
        if 'CarbonAllowanceProcurement' in frame.columns:
            frame['CarbonAllowanceProcurement'] = _require_numeric(
                'CarbonAllowanceProcurement',
                'CarbonAllowanceProcurement',
                frame['CarbonAllowanceProcurement'],
            ).astype(float)
        return frame

    def carbon_price(self) -> pd.DataFrame | None:
        """Return carbon allowance prices indexed by cap group and year."""

        frame = self._optional_frame('CarbonPrice')
        if frame is None:
            return None
        if 'CarbonPrice' in frame.columns:
            frame['CarbonPrice'] = _require_numeric(
                'CarbonPrice', 'CarbonPrice', frame['CarbonPrice']
            ).astype(float)
        return frame

    def carbon_start_bank(self) -> pd.DataFrame | None:
        """Return the starting allowance bank quantities if provided."""

        frame = self._optional_frame('CarbonStartBank')
        if frame is None:
            return None
        if 'CarbonStartBank' in frame.columns:
            frame['CarbonStartBank'] = _require_numeric(
                'CarbonStartBank', 'CarbonStartBank', frame['CarbonStartBank']
            ).astype(float)
        return frame

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

        df['covered'] = _coerce_bool(df['covered'], 'fuels', 'covered')
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
        df['covered'] = _coerce_bool(df['covered'], 'coverage', 'covered')

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

        df = self[_POLICY_KEY].copy(deep=True)
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
        missing_columns = [column for column in required if column not in df.columns]
        if 'year' in missing_columns:
            raise ValueError("policy frame is missing required columns: year")

        policy_enabled = True
        if 'policy_enabled' in df.columns:
            enabled_series = _coerce_bool(df['policy_enabled'], 'policy', 'policy_enabled')
            unique_enabled = enabled_series.dropna().unique()
            if len(unique_enabled) == 0:
                policy_enabled = bool(self._carbon_policy_enabled)
            elif len(unique_enabled) != 1:
                raise ValueError(
                    'policy frame must provide a single policy_enabled value shared across years'
                )
            else:
                policy_enabled = bool(unique_enabled[0])

        if policy_enabled and missing_columns:
            missing_list = ', '.join(sorted(missing_columns))
            raise ValueError(f'policy frame is missing required columns: {missing_list}')

        if missing_columns:
            for column in missing_columns:
                if column == 'cp_id':
                    df[column] = 'NoPolicy'
                elif column == 'full_compliance':
                    df[column] = False
                elif column == 'bank0':
                    df[column] = 0.0
                elif column == 'annual_surrender_frac':
                    df[column] = 0.0
                elif column == 'carry_pct':
                    df[column] = 1.0
                else:
                    df[column] = 0.0

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
        df['full_compliance'] = _coerce_bool(df['full_compliance'], 'policy', 'full_compliance')

        bank_values = df['bank0'].unique()
        if len(bank_values) == 0:
            bank0 = 0.0
        elif len(bank_values) != 1:
            raise ValueError('policy frame must provide a single bank0 value shared across years')
        else:
            bank0 = float(bank_values[0])

        surrender_values = df['annual_surrender_frac'].unique()
        if len(surrender_values) == 0:
            annual_surrender_frac = 0.0
        elif len(surrender_values) != 1:
            raise ValueError('policy frame must provide a single annual_surrender_frac value shared across years')
        else:
            annual_surrender_frac = float(surrender_values[0])

        carry_values = df['carry_pct'].unique()
        if len(carry_values) == 0:
            carry_pct = 1.0
        elif len(carry_values) != 1:
            raise ValueError('policy frame must provide a single carry_pct value shared across years')
        else:
            carry_pct = float(carry_values[0])

        ccr1_enabled = True
        if 'ccr1_enabled' in df.columns:
            ccr1_series = _coerce_bool(df['ccr1_enabled'], 'policy', 'ccr1_enabled')
            unique_ccr1 = ccr1_series.unique()
            if len(unique_ccr1) != 1:
                raise ValueError(
                    'policy frame must provide a single ccr1_enabled value shared across years'
                )
            ccr1_enabled = bool(unique_ccr1[0])

        ccr2_enabled = True
        if 'ccr2_enabled' in df.columns:
            ccr2_series = _coerce_bool(df['ccr2_enabled'], 'policy', 'ccr2_enabled')
            unique_ccr2 = ccr2_series.unique()
            if len(unique_ccr2) != 1:
                raise ValueError(
                    'policy frame must provide a single ccr2_enabled value shared across years'
                )
            ccr2_enabled = bool(unique_ccr2[0])

        control_period_years = None
        if 'control_period_years' in df.columns:
            cp_numeric = pd.to_numeric(df['control_period_years'], errors='coerce')
            cp_numeric = cp_numeric.dropna()
            if not cp_numeric.empty:
                unique_cp = cp_numeric.unique()
                if len(unique_cp) != 1:
                    raise ValueError(
                        'policy frame must provide a single control_period_years value shared across years'
                    )
                control_candidate = unique_cp[0]
                control_int = int(control_candidate)
                if control_int <= 0:
                    raise ValueError('control_period_years must be a positive integer')
                control_period_years = control_int

        index = df['year']
        cap = pd.Series(df['cap_tons'].values, index=index)
        floor = pd.Series(df['floor_dollars'].values, index=index)
        ccr1_trigger = pd.Series(df['ccr1_trigger'].values, index=index)
        ccr1_qty = pd.Series(df['ccr1_qty'].values, index=index)
        ccr2_trigger = pd.Series(df['ccr2_trigger'].values, index=index)
        ccr2_qty = pd.Series(df['ccr2_qty'].values, index=index)
        cp_id = pd.Series(df['cp_id'].values, index=index)

        full_compliance_years = {int(year) for year, flag in zip(df['year'], df['full_compliance']) if flag}

        self._carbon_policy_enabled = bool(policy_enabled)

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
            enabled=policy_enabled,
            ccr1_enabled=ccr1_enabled,
            ccr2_enabled=ccr2_enabled,
            control_period_years=control_period_years,
        )


__all__ = ['Frames', 'PolicySpec']

