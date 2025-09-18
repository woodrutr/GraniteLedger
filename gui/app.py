"""Streamlit interface for running BlueSky policy simulations.

The interface lazily resolves heavy optional dependencies such as :mod:`pandas`
so command-line environments without the GUI stack can still import the module
and report clear error messages.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import streamlit as st
import tomllib

from definitions import PROJECT_ROOT

try:  # pragma: no cover - optional dependency
    import pandas as _PANDAS_MODULE  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _PANDAS_MODULE = None

try:  # pragma: no cover - optional dependency
    from engine.run_loop import run_end_to_end_from_frames as _RUN_END_TO_END
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _RUN_END_TO_END = None

try:  # pragma: no cover - optional dependency
    from io_loader import Frames as _FRAMES_CLASS
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _FRAMES_CLASS = None

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from io_loader import Frames as FramesType
    import pandas as pd
else:
    FramesType = Any

    if _PANDAS_MODULE is not None:
        pd = cast(Any, _PANDAS_MODULE)
    else:
        class _PandasStub:
            DataFrame = Any
            Series = Any

        pd = cast(Any, _PandasStub())


PANDAS_REQUIRED_MESSAGE = (
    'pandas is required to run the policy simulator. Install pandas to continue.'
)


def _ensure_pandas():
    """Return the pandas module or raise an informative error."""

    if _PANDAS_MODULE is None:
        raise ModuleNotFoundError(PANDAS_REQUIRED_MESSAGE)
    return _PANDAS_MODULE


def _ensure_frames_class() -> type[FramesType]:
    """Return the :class:`Frames` class, validating optional dependencies."""

    if _FRAMES_CLASS is None:
        raise ModuleNotFoundError(PANDAS_REQUIRED_MESSAGE)
    return cast('type[FramesType]', _FRAMES_CLASS)


def _ensure_engine_runner():
    """Return the network runner callable used to solve the market model."""

    if _RUN_END_TO_END is None:
        raise ModuleNotFoundError(PANDAS_REQUIRED_MESSAGE)
    return _RUN_END_TO_END

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT, 'src', 'common', 'run_config.toml')
_DEFAULT_LOAD_MWH = 1_000_000.0
_LARGE_ALLOWANCE_SUPPLY = 1e12

_T = TypeVar('_T')


def _load_config_data(config_source: Any | None = None) -> dict[str, Any]:
    """Return configuration data as a dictionary."""

    if config_source is None:
        with open(DEFAULT_CONFIG_PATH, 'rb') as src:
            return tomllib.load(src)

    if isinstance(config_source, Mapping):
        return dict(config_source)

    if isinstance(config_source, (bytes, bytearray)):
        return tomllib.loads(config_source.decode('utf-8'))

    if isinstance(config_source, (str, Path)):
        path_candidate = Path(config_source)
        if path_candidate.exists():
            with open(path_candidate, 'rb') as src:
                return tomllib.load(src)
        return tomllib.loads(str(config_source))

    if hasattr(config_source, 'read'):
        data = config_source.read()
        if isinstance(data, bytes):
            return tomllib.loads(data.decode('utf-8'))
        return tomllib.loads(str(data))

    raise TypeError(f'Unsupported config source type: {type(config_source)!r}')


def _years_from_config(config: Mapping[str, Any]) -> list[int]:
    """Extract candidate years from the configuration mapping."""

    years: set[int] = set()
    raw_years = config.get('years')

    if isinstance(raw_years, (list, tuple, set)):
        for entry in raw_years:
            if isinstance(entry, Mapping) and 'year' in entry:
                try:
                    years.add(int(entry['year']))
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    years.add(int(entry))
                except (TypeError, ValueError):
                    continue
    elif raw_years not in (None, ''):
        try:
            years.add(int(raw_years))
        except (TypeError, ValueError):
            pass

    if not years:
        start = config.get('start_year')
        end = config.get('end_year')
        try:
            if start is not None and end is not None:
                start_val = int(start)
                end_val = int(end)
                step = 1 if end_val >= start_val else -1
                years.update(range(start_val, end_val + step, step))
            elif start is not None:
                years.add(int(start))
            elif end is not None:
                years.add(int(end))
        except (TypeError, ValueError):
            years = set()

    return sorted(years)


def _select_years(
    base_years: Iterable[int],
    start_year: int | None,
    end_year: int | None,
) -> list[int]:
    """Return a sorted list of simulation years respecting bounds."""

    years = sorted({int(year) for year in base_years}) if base_years else []

    def _ensure_int(value: int | None) -> int | None:
        if value is None:
            return None
        return int(value)

    start = _ensure_int(start_year)
    end = _ensure_int(end_year)

    if start is not None and end is not None and end < start:
        raise ValueError('end_year must be greater than or equal to start_year')

    if start is not None and end is not None:
        selected = [year for year in years if start <= year <= end]
        if not selected:
            selected = list(range(start, end + 1))
        years = selected
    elif start is not None:
        selected = [year for year in years if year >= start]
        years = selected or [start]
    elif end is not None:
        selected = [year for year in years if year <= end]
        years = selected or [end]

    if not years:
        raise ValueError('No simulation years specified')

    return sorted({int(year) for year in years})


def _coerce_year_value_map(
    entry: Any,
    years: Iterable[int],
    *,
    cast: Callable[[Any], _T],
    default: _T,
) -> dict[int, _T]:
    """Normalise TOML year/value structures into a mapping."""

    values: dict[int, _T] = {}

    if isinstance(entry, Mapping):
        iterator = entry.items()
    elif isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
        iterator = []
        for item in entry:
            if isinstance(item, Mapping) and 'year' in item:
                iterator.append((item.get('year'), item.get('value', item.get('amount'))))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                iterator.append((item[0], item[1]))
    elif entry is not None:
        try:
            coerced = cast(entry)
        except (TypeError, ValueError):
            coerced = cast(default)
        return {int(year): coerced for year in years}
    else:
        iterator = []

    for year, raw_value in iterator:
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue
        try:
            values[year_int] = cast(raw_value)
        except (TypeError, ValueError):
            continue

    result: dict[int, _T] = {}
    for year in years:
        year_int = int(year)
        result[year_int] = values.get(year_int, cast(default))
    return result


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            return float(value.strip())
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_str(value: Any, default: str = 'default') -> str:
    if value in (None, ''):
        return default
    return str(value)


def _coerce_year_set(value: Any, fallback: Iterable[int]) -> set[int]:
    years: set[int] = set()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        for entry in value:
            try:
                years.add(int(entry))
            except (TypeError, ValueError):
                continue
    elif value not in (None, ''):
        try:
            years.add(int(value))
        except (TypeError, ValueError):
            pass

    if not years:
        years = {int(year) for year in fallback}
    return years


def _build_policy_frame(
    config: Mapping[str, Any],
    years: Iterable[int],
    carbon_policy_enabled: bool,
) -> pd.DataFrame:
    """Construct the policy frame consumed by :class:`io_loader.Frames`."""

    pd = _ensure_pandas()
    years_list = sorted(int(year) for year in years)
    if not years_list:
        raise ValueError('No years supplied for policy frame')

    market_cfg = config.get('allowance_market')
    if not isinstance(market_cfg, Mapping):
        market_cfg = {}

    if carbon_policy_enabled:
        cap_map = _coerce_year_value_map(market_cfg.get('cap'), years_list, cast=float, default=0.0)
        floor_map = _coerce_year_value_map(market_cfg.get('floor'), years_list, cast=float, default=0.0)
        ccr1_trigger_map = _coerce_year_value_map(
            market_cfg.get('ccr1_trigger'), years_list, cast=float, default=0.0
        )
        ccr1_qty_map = _coerce_year_value_map(
            market_cfg.get('ccr1_qty'), years_list, cast=float, default=0.0
        )
        ccr2_trigger_map = _coerce_year_value_map(
            market_cfg.get('ccr2_trigger'), years_list, cast=float, default=0.0
        )
        ccr2_qty_map = _coerce_year_value_map(
            market_cfg.get('ccr2_qty'), years_list, cast=float, default=0.0
        )
        cp_id_map = _coerce_year_value_map(
            market_cfg.get('cp_id'), years_list, cast=lambda v: _coerce_str(v, 'CP1'), default='CP1'
        )
        bank0 = _coerce_float(market_cfg.get('bank0'), default=0.0)
        surrender_frac = _coerce_float(market_cfg.get('annual_surrender_frac'), default=1.0)
        carry_pct = _coerce_float(market_cfg.get('carry_pct'), default=1.0)
        full_compliance_years = _coerce_year_set(
            market_cfg.get('full_compliance_years'), fallback=[years_list[-1]]
        )
    else:
        cap_map = {year: float(_LARGE_ALLOWANCE_SUPPLY) for year in years_list}
        floor_map = {year: 0.0 for year in years_list}
        ccr1_trigger_map = {year: 0.0 for year in years_list}
        ccr1_qty_map = {year: 0.0 for year in years_list}
        ccr2_trigger_map = {year: 0.0 for year in years_list}
        ccr2_qty_map = {year: 0.0 for year in years_list}
        cp_id_map = {year: 'NoPolicy' for year in years_list}
        bank0 = _LARGE_ALLOWANCE_SUPPLY
        surrender_frac = 0.0
        carry_pct = 1.0
        full_compliance_years = {years_list[-1]}

    records: list[dict[str, Any]] = []
    for year in years_list:
        records.append(
            {
                'year': year,
                'cap_tons': float(cap_map[year]),
                'floor_dollars': float(floor_map[year]),
                'ccr1_trigger': float(ccr1_trigger_map[year]),
                'ccr1_qty': float(ccr1_qty_map[year]),
                'ccr2_trigger': float(ccr2_trigger_map[year]),
                'ccr2_qty': float(ccr2_qty_map[year]),
                'cp_id': str(cp_id_map[year]),
                'full_compliance': year in full_compliance_years,
                'bank0': float(bank0),
                'annual_surrender_frac': float(surrender_frac),
                'carry_pct': float(carry_pct),
            }
        )

    return pd.DataFrame(records)


def _default_units() -> pd.DataFrame:
    pd = _ensure_pandas()

    data = [
        {
            'unit_id': 'wind-1',
            'fuel': 'wind',
            'region': 'default',
            'cap_mw': 50.0,
            'availability': 0.5,
            'hr_mmbtu_per_mwh': 0.0,
            'vom_per_mwh': 0.0,
            'fuel_price_per_mmbtu': 0.0,
            'ef_ton_per_mwh': 0.0,
        },
        {
            'unit_id': 'coal-1',
            'fuel': 'coal',
            'region': 'default',
            'cap_mw': 80.0,
            'availability': 0.9,
            'hr_mmbtu_per_mwh': 9.0,
            'vom_per_mwh': 1.5,
            'fuel_price_per_mmbtu': 1.8,
            'ef_ton_per_mwh': 1.0,
        },
        {
            'unit_id': 'gas-1',
            'fuel': 'gas',
            'region': 'default',
            'cap_mw': 70.0,
            'availability': 0.85,
            'hr_mmbtu_per_mwh': 7.0,
            'vom_per_mwh': 2.0,
            'fuel_price_per_mmbtu': 2.5,
            'ef_ton_per_mwh': 0.45,
        },
    ]
    return pd.DataFrame(data)


def _default_fuels() -> pd.DataFrame:
    pd = _ensure_pandas()

    return pd.DataFrame(
        [
            {'fuel': 'wind', 'covered': False},
            {'fuel': 'coal', 'covered': True},
            {'fuel': 'gas', 'covered': True},
        ]
    )


def _default_transmission() -> pd.DataFrame:
    pd = _ensure_pandas()

    return pd.DataFrame(columns=['from_region', 'to_region', 'limit_mw'])


def _build_default_frames(years: Iterable[int]) -> FramesType:
    pd = _ensure_pandas()
    frames_cls = _ensure_frames_class()

    demand_records = [
        {'year': int(year), 'region': 'default', 'demand_mwh': float(_DEFAULT_LOAD_MWH)}
        for year in years
    ]
    base_frames = {
        'units': _default_units(),
        'demand': pd.DataFrame(demand_records),
        'fuels': _default_fuels(),
        'transmission': _default_transmission(),
    }
    return frames_cls(base_frames)


def _ensure_years_in_demand(frames: FramesType, years: Iterable[int]) -> FramesType:
    if not years:
        return frames

    pd = _ensure_pandas()
    demand = frames.demand()
    if demand.empty:
        raise ValueError('Demand frame is empty; cannot infer loads for requested years')

    existing_years = {int(year) for year in demand['year'].unique()}
    target_years = {int(year) for year in years}
    missing = sorted(target_years - existing_years)
    if not missing:
        return frames

    averages = demand.groupby('region')['demand_mwh'].mean()
    new_rows: list[dict[str, Any]] = []
    for year in missing:
        for region, value in averages.items():
            new_rows.append({'year': year, 'region': region, 'demand_mwh': float(value)})

    demand_updated = pd.concat([demand, pd.DataFrame(new_rows)], ignore_index=True)
    demand_updated = demand_updated.sort_values(['year', 'region']).reset_index(drop=True)
    return frames.with_frame('demand', demand_updated)


def _write_outputs_to_temp(outputs) -> tuple[Path, dict[str, bytes]]:
    temp_dir = Path(tempfile.mkdtemp(prefix='bluesky_gui_'))
    outputs.to_csv(temp_dir)
    csv_files: dict[str, bytes] = {}
    for csv_path in temp_dir.glob('*.csv'):
        csv_files[csv_path.name] = csv_path.read_bytes()
    return temp_dir, csv_files


def run_policy_simulation(
    config_source: Any | None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    carbon_policy_enabled: bool = True,
    frames: FramesType | Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Execute the allowance engine and return structured results."""

    try:
        config = _load_config_data(config_source)
    except Exception as exc:  # pragma: no cover - defensive path
        return {'error': f'Unable to load configuration: {exc}'}

    try:
        base_years = _years_from_config(config)
        years = _select_years(base_years, start_year, end_year)
    except Exception as exc:
        return {'error': f'Invalid year selection: {exc}'}

    try:
        _ensure_pandas()
        frames_cls = _ensure_frames_class()
        runner = _ensure_engine_runner()
    except ModuleNotFoundError as exc:
        return {'error': str(exc)}

    try:
        frames_obj = frames_cls.coerce(frames) if frames is not None else _build_default_frames(years)
        frames_obj = _ensure_years_in_demand(frames_obj, years)
        policy_frame = _build_policy_frame(config, years, carbon_policy_enabled)
        frames_obj = frames_obj.with_frame('policy', policy_frame)

        outputs = runner(frames_obj, years=years, price_initial=0.0)
        temp_dir, csv_files = _write_outputs_to_temp(outputs)

        result = {
            'annual': outputs.annual.copy(),
            'emissions_by_region': outputs.emissions_by_region.copy(),
            'price_by_region': outputs.price_by_region.copy(),
            'flows': outputs.flows.copy(),
            'csv_files': csv_files,
            'temp_dir': temp_dir,
            'years': years,
        }
        return result
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.exception('Policy simulation failed')
        return {'error': str(exc)}


def _cleanup_session_temp_dirs() -> None:
    temp_dirs = st.session_state.get('temp_dirs', [])
    for path_str in temp_dirs:
        try:
            shutil.rmtree(path_str, ignore_errors=True)
        except Exception:  # pragma: no cover - best effort cleanup
            continue
    st.session_state['temp_dirs'] = []


def _render_results(result: dict[str, Any]) -> None:
    if 'error' in result:
        st.error(result['error'])
        return

    annual = result['annual']
    if not annual.empty:
        chart_data = annual.set_index('year')
        st.subheader('Allowance market results')
        col_price, col_emissions, col_bank = st.columns(3)
        with col_price:
            st.markdown('**Allowance price ($/ton)**')
            st.line_chart(chart_data[['p_co2']])
        with col_emissions:
            st.markdown('**Emissions (tons)**')
            st.line_chart(chart_data[['emissions_tons']])
        with col_bank:
            st.markdown('**Bank balance (tons)**')
            st.line_chart(chart_data[['bank']])

        st.markdown('---')
        st.dataframe(annual, use_container_width=True)
    else:
        st.info('No annual results to display.')

    st.subheader('Download outputs')
    for filename, content in sorted(result['csv_files'].items()):
        st.download_button(
            label=f'Download {filename}',
            data=content,
            file_name=filename,
            mime='text/csv',
        )

    temp_dir = result.get('temp_dir')
    if temp_dir:
        st.caption(f'Temporary files saved to {temp_dir}')


def main() -> None:
    st.set_page_config(page_title='BlueSky Policy Simulator', layout='wide')
    st.title('BlueSky Policy Simulator')
    st.write('Upload a run configuration and execute the annual allowance market engine.')

    st.session_state.setdefault('last_result', None)
    st.session_state.setdefault('temp_dirs', [])

    with st.sidebar:
        st.header('Configuration')
        uploaded = st.file_uploader('Run configuration (TOML)', type='toml')
        if uploaded is not None:
            config_source = uploaded.getvalue()
            config_label = uploaded.name
        else:
            config_source = DEFAULT_CONFIG_PATH
            config_label = DEFAULT_CONFIG_PATH.name
        st.caption(f'Using configuration: {config_label}')

        try:
            config_preview = _load_config_data(config_source)
            candidate_years = _years_from_config(config_preview)
        except Exception as exc:  # pragma: no cover - defensive path
            candidate_years = []
            st.error(f'Failed to read configuration: {exc}')

        if candidate_years:
            year_min = min(candidate_years)
            year_max = max(candidate_years)
        else:
            year_min, year_max = 2025, 2030

        if year_min == year_max:
            start_year = st.number_input('Simulation year', value=int(year_min), step=1, format='%d')
            end_year = int(start_year)
        else:
            start_year, end_year = st.slider(
                'Simulation years',
                min_value=int(year_min),
                max_value=int(year_max),
                value=(int(year_min), int(year_max)),
            )

        carbon_policy_enabled = st.checkbox('Enable carbon policy', value=True)
        run_clicked = st.button('Run', type='primary')

    result = st.session_state.get('last_result')

    if run_clicked:
        _cleanup_session_temp_dirs()
        with st.spinner('Running simulation...'):
            result = run_policy_simulation(
                config_source,
                start_year=int(start_year),
                end_year=int(end_year),
                carbon_policy_enabled=bool(carbon_policy_enabled),
            )
        if 'temp_dir' in result:
            st.session_state['temp_dirs'] = [str(result['temp_dir'])]
        st.session_state['last_result'] = result

    if result:
        _render_results(result)
    else:
        st.info('Use the sidebar to configure and run the simulation.')


if __name__ == '__main__':  # pragma: no cover - exercised via streamlit runtime
    main()
