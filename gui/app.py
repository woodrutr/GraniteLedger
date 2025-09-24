"""Streamlit interface for running BlueSky policy simulations.

The GUI assumes that core dependencies such as :mod:`pandas` are installed.
"""

from __future__ import annotations

import copy
import io
import importlib.util
import logging
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar
import sys

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            'Python 3.11+ or the tomli package is required to read TOML configuration files.'
        ) from exc

try:
    from main.definitions import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - fallback for packaged app execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

if importlib.util.find_spec('streamlit') is not None:  # pragma: no cover - optional dependency
    import streamlit as st  # type: ignore[import-not-found]
else:  # pragma: no cover - optional dependency
    st = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from engine.run_loop import run_end_to_end_from_frames as _RUN_END_TO_END
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _RUN_END_TO_END = None

try:
    from io_loader import Frames
except ModuleNotFoundError:  # pragma: no cover - fallback when root not on sys.path
    sys.path.append(str(PROJECT_ROOT))
    from io_loader import Frames

FramesType = Frames

from src.models.electricity.scripts.technology_metadata import (
    TECH_ID_TO_LABEL,
    get_technology_label,
    resolve_technology_key,
)

STREAMLIT_REQUIRED_MESSAGE = (
    'streamlit is required to run the policy simulator UI. Install streamlit to continue.'
)

ENGINE_RUNNER_REQUIRED_MESSAGE = (
    'engine.run_loop.run_end_to_end_from_frames is required to run the policy simulator UI.'
)


def _ensure_engine_runner():
    """Return the network runner callable used to solve the market model."""

    if _RUN_END_TO_END is None:
        raise ModuleNotFoundError(ENGINE_RUNNER_REQUIRED_MESSAGE)
    return _RUN_END_TO_END


def _ensure_streamlit() -> None:
    """Raise an informative error when the GUI stack is unavailable."""

    if st is None:
        raise ModuleNotFoundError(STREAMLIT_REQUIRED_MESSAGE)


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT, 'src', 'common', 'run_config.toml')
_DEFAULT_LOAD_MWH = 1_000_000.0
_LARGE_ALLOWANCE_SUPPLY = 1e12
_ALL_REGION_IDENTIFIERS = tuple(range(1, 26))
_GENERAL_REGIONS_NORMALIZED_KEY = 'general_regions_normalized_selection'

_T = TypeVar('_T')


SIDEBAR_SECTIONS: list[tuple[str, bool]] = [
    ('General config', False),
    ('Carbon policy', False),
    ('Electricity dispatch', False),
    ('Incentives / credits', False),
    ('Outputs', False),
]

SIDEBAR_STYLE = """
<style>
.sidebar-module {
    border: 1px solid var(--secondary-background-color);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
}
.sidebar-module.disabled {
    opacity: 0.5;
}
</style>
"""


@dataclass
class GeneralConfigResult:
    """Container for user-selected general configuration settings."""

    config_label: str
    config_source: Any
    run_config: dict[str, Any]
    candidate_years: list[int]
    start_year: int
    end_year: int
    selected_years: list[int]
    regions: list[int | str]


@dataclass
class CarbonModuleSettings:
    """Record of carbon policy sidebar selections."""

    enabled: bool
    enable_floor: bool
    enable_ccr: bool
    ccr1_enabled: bool
    ccr2_enabled: bool
    banking_enabled: bool
    control_period_years: int | None
    errors: list[str] = field(default_factory=list)


@dataclass
class DispatchModuleSettings:
    """Record of electricity dispatch sidebar selections."""

    enabled: bool
    mode: str
    capacity_expansion: bool
    reserve_margins: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class IncentivesModuleSettings:
    """Record of incentive and credit selections."""

    enabled: bool
    production_credits: list[dict[str, Any]]
    investment_credits: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)


@dataclass
class OutputsModuleSettings:
    """Record of output management selections."""

    enabled: bool
    directory: str
    show_csv_downloads: bool
    errors: list[str] = field(default_factory=list)


@contextmanager
def _sidebar_panel(container: Any, enabled: bool):
    """Render a styled sidebar panel with optional greyed-out state."""

    class_name = 'sidebar-module disabled' if not enabled else 'sidebar-module'
    container.markdown(f"<div class=\"{class_name}\">", unsafe_allow_html=True)
    try:
        with container:
            yield container
    finally:
        container.markdown('</div>', unsafe_allow_html=True)


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


def _regions_from_config(config: Mapping[str, Any]) -> list[int | str]:
    """Extract region identifiers from the configuration mapping."""

    raw_regions = config.get('regions')
    regions: list[int | str] = []

    def _normalise(entry: Any) -> int | str:
        if isinstance(entry, bool):
            return int(entry)
        if isinstance(entry, (int, float)):
            return int(entry)
        text = str(entry).strip()
        if not text:
            return 'default'
        try:
            return int(text)
        except (TypeError, ValueError):
            return text

    if isinstance(raw_regions, Mapping):
        iterable: Iterable[Any] = raw_regions.values()
    else:
        iterable = raw_regions  # type: ignore[assignment]

    if isinstance(iterable, Iterable) and not isinstance(iterable, (str, bytes, Mapping)):
        for entry in iterable:
            normalised = _normalise(entry)
            if normalised not in regions:
                regions.append(normalised)
    elif iterable not in (None, ''):
        regions.append(_normalise(iterable))

    if not regions:
        regions = [1]

    return regions


def _normalize_region_labels(
    selected_labels: Iterable[str],
    previous_clean_selection: Iterable[str] | None,
) -> list[str]:
    """Return the cleaned region label selection for the multiselect widget."""

    normalized = [str(entry) for entry in selected_labels]
    if 'All' in normalized and len(normalized) > 1:
        non_all = [entry for entry in normalized if entry != 'All']
        previous_tuple = tuple(str(entry) for entry in (previous_clean_selection or ()))
        if previous_tuple == ('All',) and non_all:
            return non_all
        return ['All']
    return normalized



def _render_general_config_section(
    container: Any,
    *,
    default_source: Any,
    default_label: str,
    default_config: Mapping[str, Any],
) -> GeneralConfigResult:
    """Render general configuration controls and return the selected settings."""

    config_label = default_label
    try:
        base_config = copy.deepcopy(dict(default_config))
    except Exception:
        base_config = dict(default_config)

    uploaded = container.file_uploader(
        'Run configuration (TOML)',
        type='toml',
        key='general_config_upload',
    )
    if uploaded is not None:
        config_label = uploaded.name or 'uploaded_config.toml'
        try:
            base_config = _load_config_data(uploaded.getvalue())
        except Exception as exc:
            container.error(f'Failed to read configuration: {exc}')
            base_config = copy.deepcopy(dict(default_config))
            config_label = default_label
    else:
        config_label = default_label

    container.caption(f'Using configuration: {config_label}')

    candidate_years = _years_from_config(base_config)
    if candidate_years:
        year_min = min(candidate_years)
        year_max = max(candidate_years)
    else:
        try:
            year_min = int(base_config.get('start_year', 2025))
        except (TypeError, ValueError):
            year_min = 2025
        try:
            year_max = int(base_config.get('end_year', year_min))
        except (TypeError, ValueError):
            year_max = year_min
    if year_min > year_max:
        year_min, year_max = year_max, year_min

    def _coerce_year(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)

    start_default = _coerce_year(base_config.get('start_year', year_min), year_min)
    end_default = _coerce_year(base_config.get('end_year', year_max), year_max)
    start_default = max(year_min, min(year_max, start_default))
    end_default = max(year_min, min(year_max, end_default))
    if start_default > end_default:
        start_default, end_default = end_default, start_default

    slider_min_default = int(min(2025, year_min, start_default, end_default))
    slider_max_default = int(
        min(2050, max(2030, year_max, start_default, end_default))
    )

    def _sanitize_year_range(raw_min: Any, raw_max: Any, *, fallback: tuple[int, int]) -> tuple[int, int]:
        fallback_min, fallback_max = fallback
        candidate_min = _coerce_year(raw_min, fallback_min)
        candidate_max = _coerce_year(raw_max, fallback_max)
        candidate_min = max(slider_min_default, min(slider_max_default, candidate_min))
        candidate_max = max(slider_min_default, min(slider_max_default, candidate_max))
        if candidate_min > candidate_max:
            candidate_min, candidate_max = candidate_max, candidate_min
        return int(candidate_min), int(candidate_max)

    slider_min_value = start_default
    slider_max_value = end_default
    slider_bounds = (slider_min_default, slider_max_default)
    slider_key = 'general_year_range_slider'
    bounds_state_key = 'general_year_range_slider_bounds'
    min_numeric_key = 'general_year_range_min_numeric'
    max_numeric_key = 'general_year_range_max_numeric'
    start_input_key = 'general_year_range_min_text'
    end_input_key = 'general_year_range_max_text'
    sync_source_key = 'general_year_range_sync_source'
    slider_default_state = (slider_min_value, slider_max_value)

    if st is not None:
        config_state_key = 'general_config_active_label'
        if (
            st.session_state.get(config_state_key) != config_label
            or st.session_state.get(bounds_state_key) != slider_bounds
        ):
            st.session_state[config_state_key] = config_label
            for reset_key in (
                start_input_key,
                end_input_key,
                min_numeric_key,
                max_numeric_key,
                'general_regions',
                _GENERAL_REGIONS_NORMALIZED_KEY,
                slider_key,
                bounds_state_key,
                sync_source_key,
            ):
                st.session_state.pop(reset_key, None)

        st.session_state.setdefault(bounds_state_key, slider_bounds)
        st.session_state[bounds_state_key] = slider_bounds
        st.session_state.setdefault(slider_key, slider_default_state)
        st.session_state.setdefault(min_numeric_key, slider_default_state[0])
        st.session_state.setdefault(max_numeric_key, slider_default_state[1])
        st.session_state.setdefault(start_input_key, str(slider_default_state[0]))
        st.session_state.setdefault(end_input_key, str(slider_default_state[1]))
        st.session_state.setdefault(sync_source_key, None)

        raw_slider_state = st.session_state.get(slider_key, slider_default_state)
        if not (
            isinstance(raw_slider_state, (tuple, list))
            and len(raw_slider_state) == 2
        ):
            raw_slider_state = slider_default_state
        slider_state = _sanitize_year_range(
            raw_slider_state[0], raw_slider_state[1], fallback=slider_default_state
        )
        if tuple(raw_slider_state) != slider_state:
            st.session_state[slider_key] = slider_state

        numeric_state_raw = (
            st.session_state.get(min_numeric_key, slider_state[0]),
            st.session_state.get(max_numeric_key, slider_state[1]),
        )
        numeric_state = _sanitize_year_range(
            numeric_state_raw[0], numeric_state_raw[1], fallback=slider_state
        )
        if numeric_state != numeric_state_raw:
            st.session_state[min_numeric_key] = numeric_state[0]
            st.session_state[max_numeric_key] = numeric_state[1]
        slider_state = numeric_state

        sync_source = st.session_state.get(sync_source_key)
        start_text_value_raw = st.session_state.get(start_input_key)
        end_text_value_raw = st.session_state.get(end_input_key)
        start_text_value = (
            str(start_text_value_raw)
            if start_text_value_raw is not None
            else str(slider_state[0])
        )
        end_text_value = (
            str(end_text_value_raw)
            if end_text_value_raw is not None
            else str(slider_state[1])
        )

        if sync_source == 'slider':
            desired_start_text = str(slider_state[0])
            desired_end_text = str(slider_state[1])
            if start_text_value != desired_start_text:
                st.session_state[start_input_key] = desired_start_text
                start_text_value = desired_start_text
            if end_text_value != desired_end_text:
                st.session_state[end_input_key] = desired_end_text
                end_text_value = desired_end_text
            st.session_state[sync_source_key] = None
        else:
            sanitized_text_state = _sanitize_year_range(
                start_text_value, end_text_value, fallback=slider_state
            )
            if sanitized_text_state != slider_state:
                slider_state = sanitized_text_state
                st.session_state[slider_key] = slider_state
                st.session_state[min_numeric_key] = slider_state[0]
                st.session_state[max_numeric_key] = slider_state[1]
                sanitized_start_text = str(slider_state[0])
                sanitized_end_text = str(slider_state[1])
                if start_text_value != sanitized_start_text:
                    st.session_state[start_input_key] = sanitized_start_text
                    start_text_value = sanitized_start_text
                if end_text_value != sanitized_end_text:
                    st.session_state[end_input_key] = sanitized_end_text
                    end_text_value = sanitized_end_text
            else:
                desired_start_text = str(slider_state[0])
                desired_end_text = str(slider_state[1])
                if start_text_value != desired_start_text:
                    st.session_state[start_input_key] = desired_start_text
                    start_text_value = desired_start_text
                if end_text_value != desired_end_text:
                    st.session_state[end_input_key] = desired_end_text
                    end_text_value = desired_end_text
            st.session_state[sync_source_key] = None

        start_text_default = start_text_value
        end_text_default = end_text_value
    else:
        slider_state = slider_default_state
        start_text_default = str(slider_state[0])
        end_text_default = str(slider_state[1])

    if hasattr(container, 'text_input'):
        if hasattr(container, 'columns'):
            start_col, end_col = container.columns(2)
        else:
            start_col = container
            end_col = container
        start_text_raw = start_col.text_input('Start year', value=start_text_default, key=start_input_key if st else None)
        end_text_raw = end_col.text_input('End year', value=end_text_default, key=end_input_key if st else None)
    else:
        start_text_raw = start_text_default
        end_text_raw = end_text_default

    if st is None:
        start_numeric = _coerce_year(start_text_raw, slider_state[0])
        end_numeric = _coerce_year(end_text_raw, slider_state[1])
        start_numeric, end_numeric = _sanitize_year_range(
            start_numeric, end_numeric, fallback=slider_state
        )
        slider_state = (start_numeric, end_numeric)
    else:
        start_numeric, end_numeric = slider_state

    slider_kwargs: dict[str, Any] = {
        'min_value': slider_min_default,
        'max_value': slider_max_default,
        'value': slider_state,
        'step': 1,
        'format': '%d',
    }
    if st is not None:
        slider_kwargs['key'] = slider_key

    if hasattr(container, 'slider'):
        slider_value = container.slider('Run years', **slider_kwargs)
        if isinstance(slider_value, tuple):
            slider_min_value, slider_max_value = slider_value
        else:
            slider_min_value = slider_value
            slider_max_value = slider_value
    else:
        slider_min_value, slider_max_value = slider_state

    slider_min_value, slider_max_value = _sanitize_year_range(
        slider_min_value, slider_max_value, fallback=slider_state
    )
    final_slider_state = (slider_min_value, slider_max_value)

    if st is not None:
        st.session_state[min_numeric_key] = slider_min_value
        st.session_state[max_numeric_key] = slider_max_value
        if final_slider_state != tuple(slider_state):
            st.session_state[sync_source_key] = 'slider'
        else:
            st.session_state[sync_source_key] = None
        slider_state = final_slider_state

    start_year = slider_min_value
    end_year = slider_max_value


    region_options = _regions_from_config(base_config)
    default_region_values = list(range(1, 26))
    available_region_values: list[int | str] = []
    seen_region_labels: set[str] = set()

    for region_value in (*default_region_values, *region_options):
        label = str(region_value).strip()
        if not label:
            continue
        if label in seen_region_labels:
            continue
        seen_region_labels.add(label)
        if isinstance(region_value, bool):
            available_region_values.append(int(region_value))
        elif isinstance(region_value, (int, float)):
            available_region_values.append(int(region_value))
        else:
            available_region_values.append(region_value)

    region_labels = ['All'] + [str(value) for value in available_region_values]
    default_selection = ['All']

    if st is not None:  # pragma: no branch - streamlit only when available
        st.session_state.setdefault(
            _GENERAL_REGIONS_NORMALIZED_KEY, list(default_selection)
        )
        previous_clean_selection_raw = st.session_state.get(
            _GENERAL_REGIONS_NORMALIZED_KEY, []
        )
        if isinstance(previous_clean_selection_raw, (list, tuple)):
            previous_clean_selection = tuple(
                str(entry) for entry in previous_clean_selection_raw
            )
        elif isinstance(previous_clean_selection_raw, str):
            previous_clean_selection = (previous_clean_selection_raw,)
        else:
            previous_clean_selection = ()
    else:
        previous_clean_selection = tuple(default_selection)

    selected_regions_raw = list(
        container.multiselect(
            'Regions',
            options=region_labels,
            default=default_selection,
            key='general_regions',
        )
    )

    normalized_selection = _normalize_region_labels(
        selected_regions_raw, previous_clean_selection
    )
    if normalized_selection != selected_regions_raw and st is not None:
        st.session_state['general_regions'] = normalized_selection
    selected_regions_raw = normalized_selection

    if st is not None:
        st.session_state[_GENERAL_REGIONS_NORMALIZED_KEY] = list(selected_regions_raw)

    # Track if "All" was selected explicitly
    all_selected = 'All' in selected_regions_raw

    label_to_value: dict[str, int | str] = {
        str(value): value for value in available_region_values
    }
    if all_selected or not selected_regions_raw:
        selected_regions = list(available_region_values)
    else:
        selected_regions = []
        for entry in selected_regions_raw:
            if entry == 'All':
                continue
            value = label_to_value.get(entry)
            if value is None:
                text = str(entry).strip()
                if not text:
                    continue
                try:
                    value = int(text)
                except ValueError:
                    value = text
            if value not in selected_regions:
                selected_regions.append(value)

    if not selected_regions:
        selected_regions = list(available_region_values)

    run_config = copy.deepcopy(base_config)
    run_config['start_year'] = slider_min_value
    run_config['end_year'] = slider_max_value
    run_config['regions'] = selected_regions
    run_config.setdefault('modules', {})

    try:
        selected_years = _select_years(candidate_years, start_year, end_year)
    except Exception:
        step = 1 if end_year >= start_year else -1
        selected_years = list(range(start_year, end_year + step, step))
    if not selected_years:
        selected_years = [start_year]

    return GeneralConfigResult(
        config_label=config_label,
        config_source=run_config,
        run_config=run_config,
        candidate_years=candidate_years,
        start_year=start_year,
        end_year=end_year,
        selected_years=selected_years,
        regions=selected_regions,
    )


def _render_carbon_policy_section(
    container: Any,
    run_config: dict[str, Any],
) -> CarbonModuleSettings:
    """Render the carbon policy module controls."""

    modules = run_config.setdefault('modules', {})
    defaults = modules.get('carbon_policy', {})
    enabled_default = bool(defaults.get('enabled', True))
    enable_floor_default = bool(defaults.get('enable_floor', True))
    enable_ccr_default = bool(defaults.get('enable_ccr', True))
    ccr1_default = bool(defaults.get('ccr1_enabled', True))
    ccr2_default = bool(defaults.get('ccr2_enabled', True))
    banking_default = bool(defaults.get('allowance_banking_enabled', True))
    control_default_raw = defaults.get('control_period_years')
    try:
        control_default = int(control_default_raw)
    except (TypeError, ValueError):
        control_default = 3
    control_override_default = control_default_raw is not None

    enabled = container.toggle('Enable carbon cap', value=enabled_default, key='carbon_enable')

    with _sidebar_panel(container, enabled) as panel:
        enable_floor = panel.checkbox(
            'Enable minimum reserve price',
            value=enable_floor_default,
            disabled=not enabled,
            key='carbon_floor',
        )
        enable_ccr = panel.checkbox(
            'Enable CCR',
            value=enable_ccr_default,
            disabled=not enabled,
            key='carbon_ccr',
        )
        ccr1_enabled = panel.checkbox(
            'Enable CCR tranche 1',
            value=ccr1_default,
            disabled=not (enabled and enable_ccr),
            key='carbon_ccr1',
        )
        ccr2_enabled = panel.checkbox(
            'Enable CCR tranche 2',
            value=ccr2_default,
            disabled=not (enabled and enable_ccr),
            key='carbon_ccr2',
        )
        banking_enabled = panel.checkbox(
            'Enable allowance banking',
            value=banking_default,
            disabled=not enabled,
            key='carbon_banking',
        )
        control_override = panel.checkbox(
            'Specify control period length',
            value=control_override_default,
            disabled=not enabled,
            key='carbon_control_toggle',
        )
        control_period_value = panel.number_input(
            'Control period length (years)',
            min_value=1,
            value=int(control_default if control_default > 0 else 3),
            step=1,
            format='%d',
            key='carbon_control_years',
            disabled=not (enabled and control_override),
        )

    control_period_years = int(control_period_value) if enabled and control_override else None
    if not enabled:
        enable_floor = False
        enable_ccr = False
        ccr1_enabled = False
        ccr2_enabled = False
        banking_enabled = False
        control_period_years = None

    modules['carbon_policy'] = {
        'enabled': bool(enabled),
        'enable_floor': bool(enable_floor),
        'enable_ccr': bool(enable_ccr),
        'ccr1_enabled': bool(ccr1_enabled),
        'ccr2_enabled': bool(ccr2_enabled),
        'allowance_banking_enabled': bool(banking_enabled),
        'control_period_years': control_period_years,
    }

    errors: list[str] = []
    if enabled and not isinstance(run_config.get('allowance_market'), Mapping):
        message = 'Allowance market settings are missing from the configuration.'
        container.error(message)
        errors.append(message)

    return CarbonModuleSettings(
        enabled=bool(enabled),
        enable_floor=bool(enable_floor),
        enable_ccr=bool(enable_ccr),
        ccr1_enabled=bool(ccr1_enabled),
        ccr2_enabled=bool(ccr2_enabled),
        banking_enabled=bool(banking_enabled),
        control_period_years=control_period_years,
        errors=errors,
    )


def _render_dispatch_section(
    container: Any,
    run_config: dict[str, Any],
    frames: FramesType | None,
) -> DispatchModuleSettings:
    """Render the electricity dispatch controls."""

    modules = run_config.setdefault('modules', {})
    defaults = modules.get('electricity_dispatch', {})
    enabled_default = bool(defaults.get('enabled', False))
    mode_default = str(defaults.get('mode', 'single')).lower()
    if mode_default not in {'single', 'network'}:
        mode_default = 'single'
    capacity_default = bool(defaults.get('capacity_expansion', True))
    reserve_default = bool(defaults.get('reserve_margins', True))

    enabled = container.toggle(
        'Enable electricity dispatch',
        value=enabled_default,
        key='dispatch_enable',
    )

    mode_value = mode_default
    capacity_expansion = capacity_default
    reserve_margins = reserve_default
    errors: list[str] = []

    mode_options = {'single': 'Single region', 'network': 'Networked'}

    with _sidebar_panel(container, enabled) as panel:
        mode_label = mode_options.get(mode_default, mode_options['single'])
        mode_selection = panel.selectbox(
            'Dispatch topology',
            options=list(mode_options.values()),
            index=list(mode_options.values()).index(mode_label),
            disabled=not enabled,
            key='dispatch_mode',
        )
        mode_value = 'network' if mode_selection == mode_options['network'] else 'single'
        capacity_expansion = panel.checkbox(
            'Enable capacity expansion',
            value=capacity_default,
            disabled=not enabled,
            key='dispatch_capacity',
        )
        reserve_margins = panel.checkbox(
            'Enforce reserve margins',
            value=reserve_default,
            disabled=not enabled,
            key='dispatch_reserve',
        )

        if enabled:
            if frames is None:
                message = 'Dispatch requires demand and unit data, but no frames are available.'
                panel.error(message)
                errors.append(message)
            else:
                try:
                    demand_df = frames.demand()
                    units_df = frames.units()
                except Exception as exc:
                    message = f'Dispatch data unavailable: {exc}'
                    panel.error(message)
                    errors.append(message)
                else:
                    if demand_df.empty or units_df.empty:
                        message = 'Dispatch requires non-empty demand and unit tables.'
                        panel.error(message)
                        errors.append(message)
        else:
            mode_value = mode_default
            capacity_expansion = False
            reserve_margins = False

    if not enabled:
        mode_value = mode_value or 'single'

    modules['electricity_dispatch'] = {
        'enabled': bool(enabled),
        'mode': mode_value or 'single',
        'capacity_expansion': bool(capacity_expansion),
        'reserve_margins': bool(reserve_margins),
    }

    return DispatchModuleSettings(
        enabled=bool(enabled),
        mode=mode_value or 'single',
        capacity_expansion=bool(capacity_expansion),
        reserve_margins=bool(reserve_margins),
        errors=errors,
    )


def _render_incentives_section(
    container: Any,
    run_config: dict[str, Any],
    frames: FramesType | None,
) -> IncentivesModuleSettings:
    """Render incentive and credit controls."""

    modules = run_config.setdefault('modules', {})
    defaults = modules.get('incentives', {})
    enabled_default = bool(defaults.get('enabled', False))

    incentives_cfg = run_config.get('electricity_incentives')
    production_source: Any | None = None
    investment_source: Any | None = None
    if isinstance(incentives_cfg, Mapping):
        enabled_default = bool(incentives_cfg.get('enabled', enabled_default))
        production_source = incentives_cfg.get('production')
        investment_source = incentives_cfg.get('investment')
    if production_source is None and isinstance(defaults, Mapping):
        production_source = defaults.get('production')
    if investment_source is None and isinstance(defaults, Mapping):
        investment_source = defaults.get('investment')

    def _normalise_config_entries(
        source: Any, *, credit_key: str, limit_key: str
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if isinstance(source, Mapping):
            iterable: Iterable[Any] = [source]
        elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
            iterable = source
        else:
            iterable = []
        for entry in iterable:
            if not isinstance(entry, Mapping):
                continue
            tech_id = resolve_technology_key(entry.get('technology'))
            if tech_id is None:
                continue
            try:
                year_int = int(entry.get('year'))
            except (TypeError, ValueError):
                continue
            credit_val = _coerce_optional_float(entry.get(credit_key))
            if credit_val is None:
                continue
            limit_val = _coerce_optional_float(entry.get(limit_key))
            record: dict[str, Any] = {
                'technology': get_technology_label(tech_id),
                'year': year_int,
                credit_key: float(credit_val),
            }
            if limit_val is not None:
                record[limit_key] = float(limit_val)
            entries.append(record)
        entries.sort(key=lambda item: (str(item['technology']).lower(), int(item['year'])))
        return entries

    existing_production_entries = _normalise_config_entries(
        production_source, credit_key='credit_per_mwh', limit_key='limit_mwh'
    )
    existing_investment_entries = _normalise_config_entries(
        investment_source, credit_key='credit_per_mw', limit_key='limit_mw'
    )

    technology_options: set[str] = {
        get_technology_label(tech_id) for tech_id in sorted(TECH_ID_TO_LABEL)
    }
    for entry in (*existing_production_entries, *existing_investment_entries):
        label = str(entry.get('technology', '')).strip()
        if label:
            technology_options.add(label)
    technology_labels = sorted(technology_options)

    production_credit_col = 'Credit ($/MWh)'
    production_limit_col = 'Limit (MWh)'
    investment_credit_col = 'Credit ($/MW)'
    investment_limit_col = 'Limit (MW)'

    def _build_editor_rows(
        entries: list[dict[str, Any]],
        *,
        credit_key: str,
        limit_key: str,
        credit_label: str,
        limit_label: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            rows.append(
                {
                    'Technology': entry['technology'],
                    'Years': str(entry['year']),
                    credit_label: entry.get(credit_key),
                    limit_label: entry.get(limit_key),
                }
            )
        seen = {str(row.get('Technology')) for row in rows if row.get('Technology')}
        for label in technology_labels:
            if label not in seen:
                rows.append({'Technology': label, 'Years': '', credit_label: None, limit_label: None})
        rows.sort(
            key=lambda row: (
                str(row.get('Technology', '')).lower(),
                str(row.get('Years', '')).lower(),
            )
        )
        return rows

    production_rows_default = _build_editor_rows(
        existing_production_entries,
        credit_key='credit_per_mwh',
        limit_key='limit_mwh',
        credit_label=production_credit_col,
        limit_label=production_limit_col,
    )
    investment_rows_default = _build_editor_rows(
        existing_investment_entries,
        credit_key='credit_per_mw',
        limit_key='limit_mw',
        credit_label=investment_credit_col,
        limit_label=investment_limit_col,
    )

    available_years = _simulation_years_from_config(run_config)
    valid_years_set = {int(year) for year in available_years}

    def _rows_to_config_entries(
        rows: list[Mapping[str, Any]],
        *,
        credit_column: str,
        limit_column: str,
        credit_config_key: str,
        limit_config_key: str,
        context_label: str,
        valid_years: set[int],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        results: dict[tuple[int, int], dict[str, Any]] = {}
        messages: list[str] = []
        for index, row in enumerate(rows, start=1):
            technology_value = row.get('Technology')
            technology_label = (
                str(technology_value).strip() if technology_value not in (None, '') else ''
            )
            if not technology_label:
                continue
            tech_id = resolve_technology_key(technology_label)
            if tech_id is None:
                messages.append(
                    f'{context_label} row {index}: Unknown technology "{technology_label}".'
                )
                continue
            years_value = row.get('Years')
            years, invalid_tokens, out_of_range = _parse_years_field(
                years_value, valid_years=valid_years
            )
            if invalid_tokens:
                tokens_display = ', '.join(
                    sorted({token.strip() for token in invalid_tokens if token.strip()})
                )
                if tokens_display:
                    messages.append(
                        f'{context_label} row {index}: Unable to parse year value(s): {tokens_display}.'
                    )
            if out_of_range:
                years_display = ', '.join(str(year) for year in out_of_range)
                messages.append(
                    f'{context_label} row {index}: Year(s) {years_display} fall outside the selected simulation years.'
                )
            if not years:
                years_text = str(years_value).strip() if isinstance(years_value, str) else ''
                credit_candidate = _coerce_optional_float(row.get(credit_column))
                if years_text or credit_candidate is not None:
                    messages.append(
                        f'{context_label} row {index}: Specify one or more valid years.'
                    )
                continue
            credit_value = _coerce_optional_float(row.get(credit_column))
            if credit_value is None:
                messages.append(f'{context_label} row {index}: Provide a credit value.')
                continue
            limit_value = _coerce_optional_float(row.get(limit_column))
            label = get_technology_label(tech_id)
            for year in years:
                entry = {
                    'technology': label,
                    'year': int(year),
                    credit_config_key: float(credit_value),
                }
                if limit_value is not None:
                    entry[limit_config_key] = float(limit_value)
                results[(tech_id, int(year))] = entry
        ordered = sorted(
            results.values(),
            key=lambda item: (str(item['technology']).lower(), int(item['year'])),
        )
        return ordered, messages

    enabled = container.toggle(
        'Enable incentives and credits',
        value=enabled_default,
        key='incentives_enable',
    )

    errors: list[str] = []
    production_entries = existing_production_entries
    investment_entries = existing_investment_entries

    with _sidebar_panel(container, enabled) as panel:
        panel.caption(
            'Specify technology-specific tax credits that feed the electricity capacity and generation modules.'
        )
        if available_years:
            years_display = ', '.join(str(year) for year in available_years)
            panel.caption(f'Simulation years: {years_display}')
        panel.caption(
            'Enter comma-separated years or ranges (e.g., 2025, 2030-2032). Leave blank to exclude a technology.'
        )

        panel.markdown('**Production tax credits ($/MWh)**')
        production_editor_value = panel.data_editor(
            production_rows_default,
            disabled=not enabled,
            hide_index=True,
            num_rows='dynamic',
            use_container_width=True,
            key='incentives_production_editor',
            column_config={
                'Technology': st.column_config.SelectboxColumn(
                    'Technology', options=technology_labels
                ),
                'Years': st.column_config.TextColumn(
                    'Applicable years',
                    help='Comma-separated years or ranges (e.g., 2025, 2030-2032).',
                ),
                production_credit_col: st.column_config.NumberColumn(
                    production_credit_col,
                    format='$%.2f',
                    min_value=0.0,
                    help='Credit value applied per megawatt-hour.',
                ),
                production_limit_col: st.column_config.NumberColumn(
                    production_limit_col,
                    min_value=0.0,
                    help='Optional annual limit on eligible production (MWh).',
                ),
            },
        )

        panel.markdown('**Investment tax credits ($/MW)**')
        investment_editor_value = panel.data_editor(
            investment_rows_default,
            disabled=not enabled,
            hide_index=True,
            num_rows='dynamic',
            use_container_width=True,
            key='incentives_investment_editor',
            column_config={
                'Technology': st.column_config.SelectboxColumn(
                    'Technology', options=technology_labels
                ),
                'Years': st.column_config.TextColumn(
                    'Applicable years',
                    help='Comma-separated years or ranges (e.g., 2025, 2030-2032).',
                ),
                investment_credit_col: st.column_config.NumberColumn(
                    investment_credit_col,
                    format='$%.2f',
                    min_value=0.0,
                    help='Credit value applied per megawatt of installed capacity.',
                ),
                investment_limit_col: st.column_config.NumberColumn(
                    investment_limit_col,
                    min_value=0.0,
                    help='Optional annual limit on eligible capacity additions (MW).',
                ),
            },
        )

        validation_messages: list[str] = []
        if enabled:
            production_entries, production_messages = _rows_to_config_entries(
                _data_editor_records(production_editor_value),
                credit_column=production_credit_col,
                limit_column=production_limit_col,
                credit_config_key='credit_per_mwh',
                limit_config_key='limit_mwh',
                context_label='Production tax credit',
                valid_years=valid_years_set,
            )
            investment_entries, investment_messages = _rows_to_config_entries(
                _data_editor_records(investment_editor_value),
                credit_column=investment_credit_col,
                limit_column=investment_limit_col,
                credit_config_key='credit_per_mw',
                limit_config_key='limit_mw',
                context_label='Investment tax credit',
                valid_years=valid_years_set,
            )
            validation_messages.extend(production_messages)
            validation_messages.extend(investment_messages)

        for message in validation_messages:
            panel.error(message)
        errors.extend(validation_messages)

        if enabled:
            if frames is None:
                message = 'Incentives require generating unit data.'
                panel.error(message)
                errors.append(message)
            else:
                try:
                    units_df = frames.units()
                except Exception as exc:
                    message = f'Unable to access unit data: {exc}'
                    panel.error(message)
                    errors.append(message)
                else:
                    if units_df.empty:
                        message = 'Incentives require at least one generating unit.'
                        panel.error(message)
                        errors.append(message)

    incentives_record: dict[str, Any] = {'enabled': bool(enabled)}
    if production_entries:
        incentives_record['production'] = copy.deepcopy(production_entries)
    if investment_entries:
        incentives_record['investment'] = copy.deepcopy(investment_entries)

    run_config['electricity_incentives'] = copy.deepcopy(incentives_record)
    modules['incentives'] = copy.deepcopy(incentives_record)

    return IncentivesModuleSettings(
        enabled=bool(enabled),
        production_credits=copy.deepcopy(production_entries),
        investment_credits=copy.deepcopy(investment_entries),
        errors=errors,
    )



def _render_outputs_section(
    container: Any,
    run_config: dict[str, Any],
    last_result: Mapping[str, Any] | None,
) -> OutputsModuleSettings:
    """Render output directory and download controls."""

    modules = run_config.setdefault('modules', {})
    defaults = modules.get('outputs', {})
    enabled_default = bool(defaults.get('enabled', True))
    directory_default = str(defaults.get('directory') or run_config.get('output_name') or 'outputs')
    show_csv_default = bool(defaults.get('show_csv_downloads', True))

    enabled = container.toggle(
        'Enable output management',
        value=enabled_default,
        key='outputs_enable',
    )

    directory_value = directory_default
    show_csv_downloads = show_csv_default
    errors: list[str] = []

    with _sidebar_panel(container, enabled) as panel:
        directory_value = panel.text_input(
            'Output directory name',
            value=directory_default,
            disabled=not enabled,
            key='outputs_directory',
        ).strip()
        show_csv_downloads = panel.checkbox(
            'Show CSV downloads from last run',
            value=show_csv_default,
            disabled=not enabled,
            key='outputs_csv',
        )

        if enabled and not directory_value:
            message = 'Specify an output directory when the outputs module is enabled.'
            panel.error(message)
            errors.append(message)

        csv_files: Mapping[str, Any] | None = None
        if enabled and show_csv_downloads:
            if isinstance(last_result, Mapping):
                csv_files = last_result.get('csv_files')  # type: ignore[assignment]
            if csv_files:
                panel.caption('Download CSV outputs from the most recent run.')
                for filename, content in sorted(csv_files.items()):
                    panel.download_button(
                        label=f'Download {filename}',
                        data=content,
                        file_name=filename,
                        mime='text/csv',
                        key=f'outputs_download_{filename}',
                    )
            else:
                panel.info('No CSV outputs are available yet.')
        elif enabled:
            panel.caption('CSV downloads will be available after the next run.')

    if not directory_value:
        directory_value = directory_default or 'outputs'
    if not enabled:
        show_csv_downloads = False

    run_config['output_name'] = directory_value
    modules['outputs'] = {
        'enabled': bool(enabled),
        'directory': directory_value,
        'show_csv_downloads': bool(show_csv_downloads),
    }

    return OutputsModuleSettings(
        enabled=bool(enabled),
        directory=directory_value,
        show_csv_downloads=bool(show_csv_downloads),
        errors=errors,
    )


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


def _simulation_years_from_config(config: Mapping[str, Any]) -> list[int]:
    """Return an ordered list of simulation years inferred from ``config``."""

    try:
        base_years = _years_from_config(config)
    except Exception:
        base_years = []

    start_raw = config.get('start_year')
    end_raw = config.get('end_year')

    def _to_int(value: Any) -> int | None:
        if value in (None, ''):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    start_year = _to_int(start_raw)
    end_year = _to_int(end_raw)

    years: list[int] = []

    if base_years:
        try:
            years = _select_years(base_years, start_year, end_year)
        except Exception:
            years = [int(year) for year in base_years]
    else:
        if start_year is not None and end_year is not None:
            step = 1 if end_year >= start_year else -1
            years = list(range(start_year, end_year + step, step))
        elif start_year is not None:
            years = [start_year]
        elif end_year is not None:
            years = [end_year]

    return sorted({int(year) for year in years})


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            return float(value.strip())
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_optional_float(value: Any) -> float | None:
    """Return ``value`` coerced to ``float`` when possible."""

    if value in (None, ''):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool_flag(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "off"}:
            return False
    return bool(default)


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


def _parse_years_field(
    value: Any,
    *,
    valid_years: set[int] | None = None,
) -> tuple[list[int], list[str], list[int]]:
    """Parse a free-form year selector into structured values."""

    if value in (None, ''):
        return [], [], []

    text = str(value).strip()
    if not text:
        return [], [], []

    normalized = text.translate({ord(char): None for char in '[]{}()'})
    tokens = [token for token in re.split(r'[;,\s]+', normalized) if token]

    parsed_years: list[int] = []
    invalid_tokens: list[str] = []
    out_of_range: list[int] = []

    valid_set = {int(year) for year in valid_years} if valid_years else set()

    for token in tokens:
        token_str = token.strip()
        if not token_str:
            continue

        if '-' in token_str:
            start_text, end_text = token_str.split('-', 1)
            try:
                start_year = int(start_text.strip())
                end_year = int(end_text.strip())
            except (TypeError, ValueError):
                invalid_tokens.append(token_str)
                continue

            step = 1 if end_year >= start_year else -1
            for year in range(start_year, end_year + step, step):
                if valid_set and year not in valid_set:
                    out_of_range.append(year)
                else:
                    parsed_years.append(year)
            continue

        try:
            year_int = int(token_str)
        except (TypeError, ValueError):
            invalid_tokens.append(token_str)
            continue

        if valid_set and year_int not in valid_set:
            out_of_range.append(year_int)
        else:
            parsed_years.append(year_int)

    parsed_years = sorted({int(year) for year in parsed_years})
    out_of_range = sorted({int(year) for year in out_of_range if year not in parsed_years})

    return parsed_years, invalid_tokens, out_of_range


def _data_editor_records(value: Any) -> list[dict[str, Any]]:
    """Return ``value`` normalised to a list of row dictionaries."""

    if value is None:
        return []

    if hasattr(value, 'to_dict'):
        try:
            records = value.to_dict('records')  # type: ignore[call-arg]
        except Exception:
            records = None
        if isinstance(records, list):
            return [dict(entry) for entry in records if isinstance(entry, Mapping)]

    if isinstance(value, Mapping):
        return [dict(value)]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        records: list[dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, Mapping):
                records.append(dict(entry))
        if records:
            return records

    return []


def _build_policy_frame(
    config: Mapping[str, Any],
    years: Iterable[int],
    carbon_policy_enabled: bool,
    *,
    ccr1_enabled: bool | None = None,
    ccr2_enabled: bool | None = None,
    control_period_years: int | None = None,
    banking_enabled: bool = True,
) -> pd.DataFrame:
    """Construct the policy frame consumed by :class:`io_loader.Frames`."""

    years_list = sorted(int(year) for year in years)
    if not years_list:
        raise ValueError('No years supplied for policy frame')

    market_cfg = config.get('allowance_market')
    if not isinstance(market_cfg, Mapping):
        market_cfg = {}

    bank_flag = bool(carbon_policy_enabled and banking_enabled)

    resolution_raw = market_cfg.get('resolution', 'annual')
    if isinstance(resolution_raw, str):
        resolution = resolution_raw.strip().lower() or 'annual'
    else:
        resolution = str(resolution_raw).strip().lower() or 'annual'
    if resolution not in {'annual', 'daily'}:
        resolution = 'annual'

    if carbon_policy_enabled:
        ccr1_flag = _coerce_bool_flag(market_cfg.get('ccr1_enabled'), default=True)
        ccr2_flag = _coerce_bool_flag(market_cfg.get('ccr2_enabled'), default=True)
        if ccr1_enabled is not None:
            ccr1_flag = bool(ccr1_enabled)
        if ccr2_enabled is not None:
            ccr2_flag = bool(ccr2_enabled)

        control_period = control_period_years
        if control_period is None:
            raw_control = market_cfg.get('control_period_years')
            if raw_control not in (None, ''):
                try:
                    control_period = int(raw_control)
                except (TypeError, ValueError):
                    control_period = None
        if control_period is not None and control_period <= 0:
            control_period = None

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
        if not bank_flag:
            bank0 = 0.0
            carry_pct = 0.0
        full_compliance_years = _coerce_year_set(
            market_cfg.get('full_compliance_years'), fallback=[]
        )
        if not full_compliance_years:
            if control_period:
                full_compliance_years = {
                    year
                    for idx, year in enumerate(years_list, start=1)
                    if idx % control_period == 0
                }
            if not full_compliance_years:
                full_compliance_years = {years_list[-1]}
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
        full_compliance_years = set()
        ccr1_flag = False
        ccr2_flag = False
        control_period = None
        bank_flag = False

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
                'policy_enabled': bool(carbon_policy_enabled),
                'ccr1_enabled': bool(ccr1_flag),
                'ccr2_enabled': bool(ccr2_flag),
                'control_period_years': control_period,
                'bank_enabled': bool(bank_flag),
                'resolution': resolution,
            }
        )

    return pd.DataFrame(records)


def _default_units() -> pd.DataFrame:

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

    return pd.DataFrame(
        [
            {'fuel': 'wind', 'covered': False},
            {'fuel': 'coal', 'covered': True},
            {'fuel': 'gas', 'covered': True},
        ]
    )


def _default_transmission() -> pd.DataFrame:

    return pd.DataFrame(columns=['from_region', 'to_region', 'limit_mw'])


def _build_default_frames(
    years: Iterable[int],
    *,
    carbon_policy_enabled: bool = True,
    banking_enabled: bool = True,
) -> FramesType:
    frames_cls = FramesType

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
    return frames_cls(
        base_frames,
        carbon_policy_enabled=carbon_policy_enabled,
        banking_enabled=banking_enabled,
    )


def _ensure_years_in_demand(frames: FramesType, years: Iterable[int]) -> FramesType:
    if not years:
        return frames

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


def _read_uploaded_dataframe(uploaded_file: Any | None) -> pd.DataFrame | None:
    """Return a DataFrame parsed from ``uploaded_file`` or ``None`` on failure."""

    if uploaded_file is None:
        return None

    try:
        if hasattr(uploaded_file, 'getvalue'):
            raw = uploaded_file.getvalue()
        elif hasattr(uploaded_file, 'read'):
            raw = uploaded_file.read()
        else:
            raw = uploaded_file

        buffer: io.BytesIO | io.StringIO
        if isinstance(raw, bytes):
            buffer = io.BytesIO(raw)
        else:
            buffer = io.StringIO(str(raw))

        df = pd.read_csv(buffer)
    except Exception as exc:
        _ensure_streamlit()
        st.error(f'Unable to read CSV: {exc}')
        return None

    if df.empty:
        _ensure_streamlit()
        st.warning('Uploaded CSV is empty.')

    return df


def _validate_frame_override(
    frames_obj: FramesType,
    frame_name: str,
    df: pd.DataFrame,
) -> tuple[FramesType | None, str | None]:
    """Return a new ``Frames`` object with ``frame_name`` replaced by ``df``."""

    validator_name = frame_name.lower()
    try:
        candidate = frames_obj.with_frame(frame_name, df)
        validator = getattr(candidate, validator_name, None)
        if callable(validator):
            validator()
        else:
            candidate.frame(frame_name)
        return candidate, None
    except Exception as exc:  # pragma: no cover - defensive guard
        return None, str(exc)


def _render_demand_controls(
    frames_obj: FramesType,
    years: Iterable[int],
) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    demand_default = frames_obj.demand()
    if not demand_default.empty:
        st.caption('Current demand assumptions')
        st.dataframe(demand_default, use_container_width=True)
    else:
        st.info('No default demand data found. Provide values via the controls or upload a CSV.')

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None

    target_years = sorted({int(year) for year in years}) if years else []
    if not target_years and not demand_default.empty:
        target_years = sorted({int(year) for year in demand_default['year'].unique()})
    if not target_years:
        target_years = [2025]

    use_manual = st.checkbox('Create demand profile with controls', value=False, key='demand_manual_toggle')
    if use_manual:
        st.caption('Set a baseline load, per-region multipliers, and annual growth to construct demand.')
        if not demand_default.empty:
            first_year = target_years[0]
            base_year_data = demand_default[demand_default['year'] == first_year]
            default_base = float(base_year_data['demand_mwh'].mean()) if not base_year_data.empty else float(_DEFAULT_LOAD_MWH)
        else:
            default_base = float(_DEFAULT_LOAD_MWH)

        base_value = float(
            st.number_input(
                'Baseline demand for the first year (MWh)',
                min_value=0.0,
                value=max(0.0, default_base),
                step=10_000.0,
                format='%0.0f',
            )
        )
        growth_pct = float(
            st.slider(
                'Annual growth rate (%)',
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.25,
                key='demand_growth',
            )
        )

        if not demand_default.empty:
            region_labels = sorted({str(region) for region in demand_default['region'].unique()})
            region_defaults = (
                demand_default[demand_default['year'] == target_years[0]]
                .set_index('region')['demand_mwh']
                .to_dict()
            )
        else:
            region_labels = ['default']
            region_defaults = {}

        manual_records: list[dict[str, Any]] = []
        for region in region_labels:
            default_region_value = float(region_defaults.get(region, base_value or _DEFAULT_LOAD_MWH))
            multiplier_default = 1.0
            if base_value > 0.0:
                multiplier_default = default_region_value / base_value
            multiplier_default = float(max(0.1, min(3.0, multiplier_default)))

            multiplier = float(
                st.slider(
                    f'{region} demand multiplier',
                    min_value=0.1,
                    max_value=3.0,
                    value=multiplier_default,
                    step=0.05,
                    key=f'demand_scale_{region}',
                )
            )

            for index, year in enumerate(target_years):
                growth_factor = (1.0 + growth_pct / 100.0) ** index
                demand_val = base_value * multiplier * growth_factor
                manual_records.append(
                    {
                        'year': int(year),
                        'region': region,
                        'demand_mwh': float(demand_val),
                    }
                )

        manual_df = pd.DataFrame(manual_records)
        manual_note = (
            f'Demand constructed from GUI controls with baseline {base_value:,.0f} MWh, '
            f'growth {growth_pct:0.2f}% across {len(region_labels)} region(s) '
            f'and {len(target_years)} year(s).'
        )

    uploaded = st.file_uploader('Upload demand CSV', type='csv', key='demand_csv')
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info('Uploaded demand CSV overrides manual adjustments.')
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, 'demand', upload_df)
            if candidate is None:
                message = f'Demand CSV invalid: {error}'
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(
                    f'Demand table loaded from {uploaded.name} ({len(upload_df)} row(s)).'
                )

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, 'demand', manual_df)
        if candidate is None:
            message = f'Demand override invalid: {error}'
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def _render_units_controls(frames_obj: FramesType) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    units_default = frames_obj.units()
    if not units_default.empty:
        st.caption('Current generating units')
        st.dataframe(units_default, use_container_width=True)
    else:
        st.info('No generating units are defined. Upload a CSV to provide unit characteristics.')

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox('Edit units inline', value=False, key='units_manual_toggle')
    if edit_inline and not units_default.empty:
        st.caption('Adjust unit properties with the controls below.')
        manual_records: list[dict[str, Any]] = []
        for index, row in units_default.iterrows():
            unit_label = str(row['unit_id'])
            st.markdown(f'**{unit_label}**')
            col_meta = st.columns(3)
            with col_meta[0]:
                unit_id = st.text_input(
                    'Unit ID',
                    value=unit_label,
                    key=f'units_unit_id_{index}',
                ).strip()
                if not unit_id:
                    unit_id = unit_label
            with col_meta[1]:
                region = st.text_input(
                    'Region',
                    value=str(row['region']),
                    key=f'units_region_{index}',
                ).strip()
                if not region:
                    region = str(row['region'])
            with col_meta[2]:
                fuel = st.text_input(
                    'Fuel',
                    value=str(row['fuel']),
                    key=f'units_fuel_{index}',
                ).strip()
                if not fuel:
                    fuel = str(row['fuel'])

            col_perf = st.columns(3)
            with col_perf[0]:
                cap_mw = st.number_input(
                    'Capacity (MW)',
                    min_value=0.0,
                    value=float(row['cap_mw']),
                    step=1.0,
                    key=f'units_cap_{index}',
                )
            with col_perf[1]:
                availability = st.slider(
                    'Availability',
                    min_value=0.0,
                    max_value=1.0,
                    value=float(row['availability']),
                    step=0.01,
                    key=f'units_availability_{index}',
                )
            with col_perf[2]:
                heat_rate = st.number_input(
                    'Heat rate (MMBtu/MWh)',
                    min_value=0.0,
                    value=float(row['hr_mmbtu_per_mwh']),
                    step=0.1,
                    key=f'units_heat_rate_{index}',
                )

            col_cost = st.columns(3)
            with col_cost[0]:
                vom = st.number_input(
                    'VOM ($/MWh)',
                    min_value=0.0,
                    value=float(row['vom_per_mwh']),
                    step=0.1,
                    key=f'units_vom_{index}',
                )
            with col_cost[1]:
                fuel_price = st.number_input(
                    'Fuel price ($/MMBtu)',
                    min_value=0.0,
                    value=float(row['fuel_price_per_mmbtu']),
                    step=0.1,
                    key=f'units_fuel_price_{index}',
                )
            with col_cost[2]:
                emission_factor = st.number_input(
                    'Emission factor (ton/MWh)',
                    min_value=0.0,
                    value=float(row['ef_ton_per_mwh']),
                    step=0.01,
                    key=f'units_ef_{index}',
                )

            manual_records.append(
                {
                    'unit_id': unit_id,
                    'region': region,
                    'fuel': fuel,
                    'cap_mw': float(cap_mw),
                    'availability': float(availability),
                    'hr_mmbtu_per_mwh': float(heat_rate),
                    'vom_per_mwh': float(vom),
                    'fuel_price_per_mmbtu': float(fuel_price),
                    'ef_ton_per_mwh': float(emission_factor),
                }
            )

        manual_df = pd.DataFrame(manual_records)
        manual_note = f'Units modified via GUI controls ({len(manual_records)} unit(s)).'
    elif edit_inline:
        st.info('Upload a units CSV to edit inline.')

    uploaded = st.file_uploader('Upload units CSV', type='csv', key='units_csv')
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info('Uploaded units CSV overrides inline edits.')
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, 'units', upload_df)
            if candidate is None:
                message = f'Units CSV invalid: {error}'
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f'Units loaded from {uploaded.name} ({len(upload_df)} row(s)).')

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, 'units', manual_df)
        if candidate is None:
            message = f'Units override invalid: {error}'
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def _render_fuels_controls(frames_obj: FramesType) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    fuels_default = frames_obj.fuels()
    if not fuels_default.empty:
        st.caption('Current fuel coverage')
        st.dataframe(fuels_default, use_container_width=True)
    else:
        st.info('No fuel data available. Upload a CSV to specify fuel coverage.')

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox('Edit fuel coverage inline', value=False, key='fuels_manual_toggle')
    if edit_inline and not fuels_default.empty:
        st.caption('Toggle coverage and update emission factors as needed.')
        manual_records: list[dict[str, Any]] = []
        has_emission_column = 'co2_ton_per_mmbtu' in fuels_default.columns
        for index, row in fuels_default.iterrows():
            fuel_label = str(row['fuel'])
            col_line = st.columns(3 if has_emission_column else 2)
            with col_line[0]:
                fuel_name = st.text_input(
                    'Fuel',
                    value=fuel_label,
                    key=f'fuels_name_{index}',
                ).strip()
                if not fuel_name:
                    fuel_name = fuel_label
            with col_line[1]:
                covered = st.checkbox(
                    'Covered',
                    value=bool(row['covered']),
                    key=f'fuels_covered_{index}',
                )
            emission_value: float | None = None
            if has_emission_column:
                with col_line[2]:
                    emission_value = float(
                        st.number_input(
                            'CO₂ tons/MMBtu',
                            min_value=0.0,
                            value=float(row.get('co2_ton_per_mmbtu', 0.0)),
                            step=0.01,
                            key=f'fuels_emission_{index}',
                        )
                    )

            record: dict[str, Any] = {'fuel': fuel_name, 'covered': bool(covered)}
            if has_emission_column:
                record['co2_ton_per_mmbtu'] = float(emission_value or 0.0)
            manual_records.append(record)

        manual_df = pd.DataFrame(manual_records)
        manual_note = f'Fuel coverage edited inline ({len(manual_records)} fuel(s)).'
    elif edit_inline:
        st.info('Upload a fuels CSV to edit inline.')

    uploaded = st.file_uploader('Upload fuels CSV', type='csv', key='fuels_csv')
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info('Uploaded fuels CSV overrides inline edits.')
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, 'fuels', upload_df)
            if candidate is None:
                message = f'Fuels CSV invalid: {error}'
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f'Fuels loaded from {uploaded.name} ({len(upload_df)} row(s)).')

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, 'fuels', manual_df)
        if candidate is None:
            message = f'Fuels override invalid: {error}'
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def _render_transmission_controls(
    frames_obj: FramesType,
) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    transmission_default = frames_obj.transmission()
    if not transmission_default.empty:
        st.caption('Current transmission limits')
        st.dataframe(transmission_default, use_container_width=True)
    else:
        st.info('No transmission limits specified. Add entries below or upload a CSV.')

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox('Edit transmission limits inline', value=False, key='transmission_manual_toggle')
    if edit_inline:
        editable = transmission_default.copy()
        if editable.empty:
            editable = pd.DataFrame(columns=['from_region', 'to_region', 'limit_mw'])
        st.caption('Use the table to add or modify directional flow limits (MW).')
        edited = st.data_editor(
            editable,
            num_rows='dynamic',
            use_container_width=True,
            key='transmission_editor',
        )
        if isinstance(edited, pd.DataFrame):
            manual_df = edited.copy()
        else:
            manual_df = pd.DataFrame(edited)
        manual_df = manual_df.dropna(how='all')
        manual_df = manual_df.reindex(columns=['from_region', 'to_region', 'limit_mw'])
        manual_note = f'Transmission table edited inline ({len(manual_df)} record(s)).'

    uploaded = st.file_uploader('Upload transmission CSV', type='csv', key='transmission_csv')
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info('Uploaded transmission CSV overrides inline edits.')
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, 'transmission', upload_df)
            if candidate is None:
                message = f'Transmission CSV invalid: {error}'
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(
                    f'Transmission limits loaded from {uploaded.name} ({len(upload_df)} row(s)).'
                )

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, 'transmission', manual_df)
        if candidate is None:
            message = f'Transmission override invalid: {error}'
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def run_policy_simulation(
    config_source: Any | None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    carbon_policy_enabled: bool = True,
    enable_floor: bool = True,
    enable_ccr: bool = True,
    ccr1_enabled: bool = True,
    ccr2_enabled: bool = True,
    allowance_banking_enabled: bool = True,
    control_period_years: int | None = None,
    dispatch_use_network: bool = False,
    module_config: Mapping[str, Any] | None = None,
    frames: FramesType | Mapping[str, pd.DataFrame] | None = None,
    assumption_notes: Iterable[str] | None = None,
    progress_cb: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, Any]:
    """Execute the allowance engine and return structured results.

    When provided ``progress_cb`` receives progress updates using the
    ``(stage, payload)`` convention emitted by :func:`engine.run_loop.run_end_to_end_from_frames`.
    """

    try:
        config = _load_config_data(config_source)
    except Exception as exc:  # pragma: no cover - defensive path
        return {'error': f'Unable to load configuration: {exc}'}

    modules_section = config.setdefault('modules', {})
    merged_modules: dict[str, dict[str, Any]] = {}
    if isinstance(modules_section, Mapping):
        for name, settings in modules_section.items():
            if isinstance(settings, Mapping):
                merged_modules[str(name)] = dict(settings)

    if isinstance(module_config, Mapping):
        for name, settings in module_config.items():
            if isinstance(settings, Mapping):
                merged_modules[str(name)] = dict(settings)
            else:
                merged_modules[str(name)] = {'value': settings}

    carbon_record = merged_modules.setdefault('carbon_policy', {})
    carbon_record.update(
        {
            'enabled': bool(carbon_policy_enabled),
            'enable_floor': bool(enable_floor),
            'enable_ccr': bool(enable_ccr),
            'ccr1_enabled': bool(ccr1_enabled) if enable_ccr else False,
            'ccr2_enabled': bool(ccr2_enabled) if enable_ccr else False,
            'allowance_banking_enabled': bool(allowance_banking_enabled),
            'control_period_years': control_period_years,
        }
    )
    if not enable_ccr:
        carbon_record['ccr1_enabled'] = False
        carbon_record['ccr2_enabled'] = False

    dispatch_record = merged_modules.setdefault('electricity_dispatch', {})
    dispatch_enabled = bool(dispatch_record.get('enabled')) or bool(dispatch_use_network)
    dispatch_record['enabled'] = dispatch_enabled
    dispatch_record['use_network'] = bool(dispatch_use_network)
    current_mode = str(dispatch_record.get('mode', 'network' if dispatch_use_network else 'single')).lower()
    dispatch_record['mode'] = 'network' if dispatch_use_network else (
        'network' if current_mode == 'network' else 'single'
    )

    try:
        base_years = _years_from_config(config)
        years = _select_years(base_years, start_year, end_year)
    except Exception as exc:
        return {'error': f'Invalid year selection: {exc}'}

    frames_cls = FramesType
    try:
        runner = _ensure_engine_runner()
    except ModuleNotFoundError as exc:
        return {'error': str(exc)}

    if not carbon_policy_enabled:
        enable_floor = False
        enable_ccr = False
        allowance_banking_enabled = False

    carbon_record['enabled'] = bool(carbon_policy_enabled)
    carbon_record['enable_floor'] = bool(enable_floor)
    carbon_record['enable_ccr'] = bool(enable_ccr)
    carbon_record['ccr1_enabled'] = bool(ccr1_enabled) if enable_ccr else False
    carbon_record['ccr2_enabled'] = bool(ccr2_enabled) if enable_ccr else False
    carbon_record['allowance_banking_enabled'] = bool(allowance_banking_enabled)
    carbon_record['control_period_years'] = control_period_years
    if not enable_ccr:
        carbon_record['ccr1_enabled'] = False
        carbon_record['ccr2_enabled'] = False

    try:
        frames_obj = (
            frames_cls.coerce(
                frames,
                carbon_policy_enabled=carbon_policy_enabled,
                banking_enabled=allowance_banking_enabled,
            )
            if frames is not None
            else _build_default_frames(
                years,
                carbon_policy_enabled=carbon_policy_enabled,
                banking_enabled=allowance_banking_enabled,
            )
        )
        frames_obj = _ensure_years_in_demand(frames_obj, years)
        policy_frame = _build_policy_frame(
            config,
            years,
            carbon_policy_enabled,
            ccr1_enabled=ccr1_enabled,
            ccr2_enabled=ccr2_enabled,
            control_period_years=control_period_years,
            banking_enabled=allowance_banking_enabled,
        )
        frames_obj = frames_obj.with_frame('policy', policy_frame)

        outputs = runner(
            frames_obj,
            years=years,
            price_initial=0.0,
            enable_floor=bool(enable_floor),
            enable_ccr=bool(enable_ccr),
            use_network=bool(dispatch_use_network),
            progress_cb=progress_cb,
        )
        temp_dir, csv_files = _write_outputs_to_temp(outputs)

        overrides = [str(note) for note in assumption_notes] if assumption_notes else []

        config['modules'] = copy.deepcopy(merged_modules)

        result = {
            'annual': outputs.annual.copy(),
            'emissions_by_region': outputs.emissions_by_region.copy(),
            'price_by_region': outputs.price_by_region.copy(),
            'flows': outputs.flows.copy(),
            'csv_files': csv_files,
            'temp_dir': temp_dir,
            'years': years,
            'documentation': {'assumption_overrides': overrides},
            'module_config': copy.deepcopy(merged_modules),
            'run_config': copy.deepcopy(config),
        }
        return result
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.exception('Policy simulation failed')
        return {'error': str(exc)}


def _cleanup_session_temp_dirs() -> None:
    _ensure_streamlit()
    temp_dirs = st.session_state.get('temp_dirs', [])
    for path_str in temp_dirs:
        try:
            shutil.rmtree(path_str, ignore_errors=True)
        except Exception:  # pragma: no cover - best effort cleanup
            continue
    st.session_state['temp_dirs'] = []


def _build_run_summary(settings: Mapping[str, Any], *, config_label: str) -> list[tuple[str, str]]:
    """Return human-readable configuration details for confirmation dialogs."""

    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _bool_label(value: bool) -> str:
        return 'Yes' if value else 'No'

    start_year = _as_int(settings.get('start_year'))
    end_year = _as_int(settings.get('end_year'))

    if start_year is None and end_year is None:
        year_display = 'Not specified'
    else:
        if start_year is None:
            start_year = end_year
        if end_year is None:
            end_year = start_year
        if start_year == end_year:
            year_display = f'{start_year}'
        else:
            year_display = f'{start_year} – {end_year}'

    carbon_enabled = bool(settings.get('carbon_policy_enabled', True))
    enable_floor = bool(settings.get('enable_floor', False)) if carbon_enabled else False
    enable_ccr = bool(settings.get('enable_ccr', False)) if carbon_enabled else False
    ccr1_enabled = bool(settings.get('ccr1_enabled', False)) if enable_ccr else False
    ccr2_enabled = bool(settings.get('ccr2_enabled', False)) if enable_ccr else False
    banking_enabled = (
        bool(settings.get('allowance_banking_enabled', False)) if carbon_enabled else False
    )

    control_period = settings.get('control_period_years') if carbon_enabled else None

    modules = settings.get('module_config')
    if isinstance(modules, Mapping):
        carbon_cfg = modules.get('carbon_policy')
        if isinstance(carbon_cfg, Mapping):
            carbon_enabled = bool(carbon_cfg.get('enabled', carbon_enabled))
            enable_floor = bool(carbon_cfg.get('enable_floor', enable_floor)) if carbon_enabled else False
            enable_ccr = bool(carbon_cfg.get('enable_ccr', enable_ccr)) if carbon_enabled else False
            ccr1_enabled = bool(carbon_cfg.get('ccr1_enabled', ccr1_enabled)) if enable_ccr else False
            ccr2_enabled = bool(carbon_cfg.get('ccr2_enabled', ccr2_enabled)) if enable_ccr else False
            banking_enabled = (
                bool(carbon_cfg.get('allowance_banking_enabled', banking_enabled))
                if carbon_enabled
                else False
            )
            control_period = carbon_cfg.get('control_period_years') if carbon_enabled else control_period

        dispatch_cfg = modules.get('electricity_dispatch')
        incentives_cfg = modules.get('incentives')
        outputs_cfg = modules.get('outputs')
    else:
        dispatch_cfg = None
        incentives_cfg = None
        outputs_cfg = None
    if not carbon_enabled:
        control_display = 'Not applicable'
    elif control_period is None:
        control_display = 'Automatic'
    else:
        control_display = str(control_period)

    summary: list[tuple[str, str]] = [
        ('Configuration', config_label),
        ('Simulation years', year_display),
        ('Carbon cap enabled', _bool_label(carbon_enabled)),
        ('Minimum reserve price', _bool_label(enable_floor)),
        ('CCR enabled', _bool_label(enable_ccr)),
        ('CCR tranche 1', _bool_label(ccr1_enabled)),
        ('CCR tranche 2', _bool_label(ccr2_enabled)),
        ('Allowance banking enabled', _bool_label(banking_enabled)),
        ('Control period length', control_display),
    ]

    dispatch_enabled = False
    dispatch_network = bool(settings.get('dispatch_use_network', False))
    dispatch_mode_label = 'Single region'
    capacity_expansion = False
    reserve_margins = False
    if isinstance(dispatch_cfg, Mapping):
        dispatch_enabled = bool(dispatch_cfg.get('enabled', dispatch_enabled))
        dispatch_network = bool(dispatch_cfg.get('use_network', dispatch_network))
        dispatch_mode = str(dispatch_cfg.get('mode', 'network' if dispatch_network else 'single')).lower()
        dispatch_mode_label = 'Networked' if dispatch_network or dispatch_mode == 'network' else 'Single region'
        capacity_expansion = bool(dispatch_cfg.get('capacity_expansion', capacity_expansion))
        reserve_margins = bool(dispatch_cfg.get('reserve_margins', reserve_margins))

    summary.append(('Electricity dispatch module', _bool_label(dispatch_enabled)))
    if dispatch_enabled or dispatch_network:
        summary.append(('Dispatch mode', dispatch_mode_label))
        summary.append(('Capacity expansion', _bool_label(capacity_expansion)))
        summary.append(('Reserve margins', _bool_label(reserve_margins)))

    incentives_enabled = False
    production_entries: list[Mapping[str, Any]] = []
    investment_entries: list[Mapping[str, Any]] = []
    if isinstance(incentives_cfg, Mapping):
        incentives_enabled = bool(incentives_cfg.get('enabled', incentives_enabled))

        def _extract_entries(raw: Any) -> list[Mapping[str, Any]]:
            if isinstance(raw, Mapping):
                candidates = [raw]
            elif isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
                candidates = [entry for entry in raw if isinstance(entry, Mapping)]
            else:
                candidates = []
            normalised: list[Mapping[str, Any]] = []
            for entry in candidates:
                tech_label = str(entry.get('technology', '')).strip()
                try:
                    year_val = int(entry.get('year'))
                except (TypeError, ValueError):
                    continue
                record: dict[str, Any] = {'technology': tech_label, 'year': year_val}
                normalised.append(record)
            return normalised

        production_entries = _extract_entries(incentives_cfg.get('production'))
        investment_entries = _extract_entries(incentives_cfg.get('investment'))

    summary.append(('Incentives module', _bool_label(incentives_enabled)))
    if incentives_enabled:
        summary.append(('Production tax credit entries', str(len(production_entries))))
        if production_entries:
            ptc_technologies = sorted(
                {
                    str(entry.get('technology')).strip()
                    for entry in production_entries
                    if str(entry.get('technology')).strip()
                }
            )
            if ptc_technologies:
                summary.append(('PTC technologies', ', '.join(ptc_technologies)))
        summary.append(('Investment tax credit entries', str(len(investment_entries))))
        if investment_entries:
            itc_technologies = sorted(
                {
                    str(entry.get('technology')).strip()
                    for entry in investment_entries
                    if str(entry.get('technology')).strip()
                }
            )
            if itc_technologies:
                summary.append(('ITC technologies', ', '.join(itc_technologies)))

    outputs_enabled = True
    output_directory = settings.get('output_name', 'outputs')
    show_sidebar_downloads = False
    if isinstance(outputs_cfg, Mapping):
        outputs_enabled = bool(outputs_cfg.get('enabled', outputs_enabled))
        output_directory = outputs_cfg.get('directory', output_directory)
        show_sidebar_downloads = bool(outputs_cfg.get('show_csv_downloads', show_sidebar_downloads))

    summary.append(('Outputs module', _bool_label(outputs_enabled)))
    summary.append(('Output directory', str(output_directory)))
    if outputs_enabled:
        summary.append(('Sidebar CSV downloads', _bool_label(show_sidebar_downloads)))

    return summary


def _render_results(result: dict[str, Any]) -> None:  # pragma: no cover - UI rendering
    _ensure_streamlit()
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

    documentation = result.get('documentation')
    overrides: list[str] = []
    if isinstance(documentation, Mapping):
        overrides = [str(entry) for entry in documentation.get('assumption_overrides', [])]

    st.subheader('Assumption overrides')
    if overrides:
        for note in overrides:
            st.markdown(f'- {note}')
    else:
        st.caption('No assumption overrides were applied in this run.')

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


def main() -> None:  # pragma: no cover - Streamlit entry point
    _ensure_streamlit()
    st.set_page_config(page_title='BlueSky Policy Simulator', layout='wide')
    st.title('BlueSky Policy Simulator')
    st.write('Upload a run configuration and execute the annual allowance market engine.')

    st.session_state.setdefault('last_result', None)
    st.session_state.setdefault('temp_dirs', [])

    module_errors: list[str] = []
    assumption_notes: list[str] = []
    assumption_errors: list[str] = []

    try:
        default_config_data = _load_config_data(DEFAULT_CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - defensive UI path
        default_config_data = {}
        st.warning(f'Unable to load default configuration: {exc}')

    run_config: dict[str, Any] = copy.deepcopy(default_config_data) if default_config_data else {}
    config_label = DEFAULT_CONFIG_PATH.name
    selected_years: list[int] = []
    frames_for_run: FramesType | None = None
    start_year_val = int(run_config.get('start_year', 2025)) if run_config else 2025
    end_year_val = int(run_config.get('end_year', start_year_val)) if run_config else start_year_val

    with st.sidebar:
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

        general_label, general_expanded = SIDEBAR_SECTIONS[0]
        general_expander = st.expander(general_label, expanded=general_expanded)
        general_result = _render_general_config_section(
            general_expander,
            default_source=DEFAULT_CONFIG_PATH,
            default_label=DEFAULT_CONFIG_PATH.name,
            default_config=default_config_data,
        )
        run_config = general_result.run_config
        config_label = general_result.config_label
        candidate_years = general_result.candidate_years
        start_year_val = general_result.start_year
        end_year_val = general_result.end_year
        selected_years = general_result.selected_years

        carbon_label, carbon_expanded = SIDEBAR_SECTIONS[1]
        carbon_expander = st.expander(carbon_label, expanded=carbon_expanded)
        carbon_settings = _render_carbon_policy_section(carbon_expander, run_config)
        module_errors.extend(carbon_settings.errors)

        try:
            frames_for_run = _build_default_frames(
                selected_years or [start_year_val],
                carbon_policy_enabled=carbon_settings.enabled,
                banking_enabled=carbon_settings.banking_enabled,
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            frames_for_run = None
            st.warning(f'Unable to prepare default assumption tables: {exc}')

        dispatch_label, dispatch_expanded = SIDEBAR_SECTIONS[2]
        dispatch_expander = st.expander(dispatch_label, expanded=dispatch_expanded)
        dispatch_settings = _render_dispatch_section(dispatch_expander, run_config, frames_for_run)
        module_errors.extend(dispatch_settings.errors)

        incentives_label, incentives_expanded = SIDEBAR_SECTIONS[3]
        incentives_expander = st.expander(incentives_label, expanded=incentives_expanded)
        incentives_settings = _render_incentives_section(incentives_expander, run_config, frames_for_run)
        module_errors.extend(incentives_settings.errors)

        outputs_label, outputs_expanded = SIDEBAR_SECTIONS[4]
        outputs_expander = st.expander(outputs_label, expanded=outputs_expanded)
        last_result_mapping = st.session_state.get('last_result')
        if not isinstance(last_result_mapping, Mapping):
            last_result_mapping = None
        outputs_settings = _render_outputs_section(
            outputs_expander,
            run_config,
            last_result_mapping,
        )
        module_errors.extend(outputs_settings.errors)

        run_clicked = st.button('Run Model', type='primary', use_container_width=True)

    try:
        selected_years = _select_years(candidate_years, start_year_val, end_year_val)
    except Exception:
        selected_years = selected_years or []
    if not selected_years:
        step = 1 if end_year_val >= start_year_val else -1
        selected_years = list(range(start_year_val, end_year_val + step, step))

    if frames_for_run is None:
        try:
            frames_for_run = _build_default_frames(
                selected_years or [start_year_val],
                carbon_policy_enabled=bool(carbon_settings.enabled),
                banking_enabled=bool(carbon_settings.banking_enabled),
            )
        except Exception as exc:  # pragma: no cover - defensive UI path
            frames_for_run = None
            st.warning(f'Unable to prepare default assumption tables: {exc}')

    if module_errors:
        st.warning('Resolve the module configuration issues highlighted in the sidebar before running the simulation.')

    if frames_for_run is not None:
        st.subheader('Assumption overrides')
        st.caption('Adjust core assumption tables or upload CSV files to override the defaults.')
        demand_tab, units_tab, fuels_tab, transmission_tab = st.tabs(
            ['Demand', 'Units', 'Fuels', 'Transmission']
        )
        with demand_tab:
            frames_for_run, notes, errors = _render_demand_controls(frames_for_run, selected_years)
            assumption_notes.extend(notes)
            assumption_errors.extend(errors)
        with units_tab:
            frames_for_run, notes, errors = _render_units_controls(frames_for_run)
            assumption_notes.extend(notes)
            assumption_errors.extend(errors)
        with fuels_tab:
            frames_for_run, notes, errors = _render_fuels_controls(frames_for_run)
            assumption_notes.extend(notes)
            assumption_errors.extend(errors)
        with transmission_tab:
            frames_for_run, notes, errors = _render_transmission_controls(frames_for_run)
            assumption_notes.extend(notes)
            assumption_errors.extend(errors)

        if assumption_errors:
            st.warning('Resolve the highlighted assumption issues before running the simulation.')
    else:
        st.info(
            'Default assumption tables are unavailable due to a previous error. '
            'Resolve the issue above to edit inputs through the GUI.'
        )

    execute_run = False
    run_inputs: dict[str, Any] | None = None
    pending_run = st.session_state.get('pending_run')

    if run_clicked:
        if assumption_errors or module_errors:
            st.error('Resolve the configuration issues above before running the simulation.')
        else:
            run_inputs_payload = {
                'config_source': copy.deepcopy(run_config),
                'start_year': int(start_year_val),
                'end_year': int(end_year_val),
                'carbon_policy_enabled': bool(carbon_settings.enabled),
                'enable_floor': bool(carbon_settings.enable_floor),
                'enable_ccr': bool(carbon_settings.enable_ccr),
                'ccr1_enabled': bool(carbon_settings.ccr1_enabled),
                'ccr2_enabled': bool(carbon_settings.ccr2_enabled),
                'allowance_banking_enabled': bool(carbon_settings.banking_enabled),
                'control_period_years': carbon_settings.control_period_years,
                'dispatch_use_network': bool(
                    dispatch_settings.enabled and dispatch_settings.mode == 'network'
                ),
                'module_config': copy.deepcopy(run_config.get('modules', {})),
            }
            st.session_state['pending_run'] = {
                'params': run_inputs_payload,
                'summary': _build_run_summary(run_inputs_payload, config_label=config_label),
            }
            pending_run = st.session_state['pending_run']

    if isinstance(pending_run, Mapping):
        confirmation_box = st.warning(
            'You are about to run the model with the following configuration:'
        )
        summary_details = pending_run.get('summary', [])
        if isinstance(summary_details, list) and summary_details:
            summary_lines = '\n'.join(
                f'- **{label}:** {value}' for label, value in summary_details
            )
            confirmation_box.markdown(summary_lines)
        else:
            confirmation_box.markdown('*No configuration details available.*')
        confirmation_box.markdown('**Do you want to continue and run the model?**')
        confirm_col, cancel_col = confirmation_box.columns(2)
        confirm_clicked = confirm_col.button('Confirm Run', type='primary', key='confirm_run')
        cancel_clicked = cancel_col.button('Cancel', key='cancel_run')

        if cancel_clicked:
            st.session_state.pop('pending_run', None)
            pending_run = None
        elif confirm_clicked:
            pending_params = pending_run.get('params')
            if isinstance(pending_params, Mapping):
                run_inputs = dict(pending_params)
                execute_run = True
            st.session_state.pop('pending_run', None)
            pending_run = None

    dispatch_use_network = bool(
        dispatch_settings.enabled and dispatch_settings.mode == 'network'
    )

    if run_inputs is not None:
        run_config = copy.deepcopy(run_inputs.get('config_source', run_config))
        start_year_val = int(run_inputs.get('start_year', start_year_val))
        end_year_val = int(run_inputs.get('end_year', end_year_val))
        dispatch_use_network = bool(
            run_inputs.get('dispatch_use_network', dispatch_use_network)
        )

    result = st.session_state.get('last_result')

    if execute_run:
        _cleanup_session_temp_dirs()
        progress_text = st.empty()
        progress_bar = st.progress(0)
        progress_state: dict[str, Any] = {
            'total_years': 1,
            'current_index': -1,
            'current_year': None,
        }

        def _update_progress(stage: str, payload: Mapping[str, object]) -> None:
            def _as_int(value: object, default: int = 0) -> int:
                try:
                    return int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return default

            def _as_float(value: object) -> float | None:
                try:
                    if value is None:
                        return None
                    return float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None

            if stage == 'run_start':
                total = _as_int(payload.get('total_years'), 0)
                if total <= 0:
                    total = 1
                progress_state['total_years'] = total
                progress_state['current_index'] = -1
                progress_state['current_year'] = None
                progress_bar.progress(0)
                progress_text.text(f'Preparing simulation for {total} year(s)...')
                return

            if stage == 'year_start':
                index = _as_int(payload.get('index'), 0)
                year_val = payload.get('year')
                total = max(progress_state.get('total_years', 1), 1)
                progress_state['current_index'] = index
                progress_state['current_year'] = year_val
                completed_fraction = max(0.0, min(1.0, index / total))
                progress_bar.progress(int(completed_fraction * 100))
                year_display = str(year_val) if year_val is not None else 'N/A'
                progress_text.text(f'Simulating year {year_display} ({index + 1} of {total})')
                return

            if stage == 'iteration':
                year_val = payload.get('year', progress_state.get('current_year'))
                iteration = _as_int(payload.get('iteration'), 0)
                price_val = _as_float(payload.get('price'))
                year_display = str(year_val) if year_val is not None else 'N/A'
                if price_val is not None:
                    progress_text.text(
                        f'Year {year_display}: iteration {iteration} (price ≈ {price_val:,.2f})'
                    )
                else:
                    progress_text.text(f'Year {year_display}: iteration {iteration}')
                return

            if stage == 'year_complete':
                index = _as_int(payload.get('index'), progress_state.get('current_index', -1))
                total = max(progress_state.get('total_years', 1), 1)
                progress_state['current_index'] = index
                year_val = payload.get('year', progress_state.get('current_year'))
                progress_state['current_year'] = year_val
                completed_fraction = max(0.0, min(1.0, (index + 1) / total))
                progress_bar.progress(min(100, int(completed_fraction * 100)))
                price_val = _as_float(payload.get('price'))
                year_display = str(year_val) if year_val is not None else str(index + 1)
                if price_val is not None:
                    progress_text.text(
                        f'Completed year {year_display} of {total} (price {price_val:,.2f})'
                    )
                else:
                    progress_text.text(f'Completed year {year_display} of {total}')
                return

        inputs_for_run = run_inputs or {}
        try:
            result = run_policy_simulation(
                inputs_for_run.get('config_source', run_config),
                start_year=inputs_for_run.get('start_year', start_year_val),
                end_year=inputs_for_run.get('end_year', end_year_val),
                carbon_policy_enabled=bool(
                    inputs_for_run.get('carbon_policy_enabled', carbon_settings.enabled)
                ),
                enable_floor=bool(
                    inputs_for_run.get('enable_floor', carbon_settings.enable_floor)
                ),
                enable_ccr=bool(inputs_for_run.get('enable_ccr', carbon_settings.enable_ccr)),
                ccr1_enabled=bool(
                    inputs_for_run.get('ccr1_enabled', carbon_settings.ccr1_enabled)
                ),
                ccr2_enabled=bool(
                    inputs_for_run.get('ccr2_enabled', carbon_settings.ccr2_enabled)
                ),
                allowance_banking_enabled=bool(
                    inputs_for_run.get('allowance_banking_enabled', carbon_settings.banking_enabled)
                ),
                control_period_years=inputs_for_run.get(
                    'control_period_years', carbon_settings.control_period_years
                ),
                dispatch_use_network=bool(
                    inputs_for_run.get('dispatch_use_network', dispatch_use_network)
                ),
                module_config=inputs_for_run.get(
                    'module_config', run_config.get('modules', {})
                ),
                frames=frames_for_run,
                assumption_notes=assumption_notes,
                progress_cb=_update_progress,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception('Policy simulation failed during execution')
            result = {'error': str(exc)}
        finally:
            progress_bar.empty()
            progress_text.empty()
        if 'temp_dir' in result:
            st.session_state['temp_dirs'] = [str(result['temp_dir'])]
        st.session_state['last_result'] = result

    if result:
        _render_results(result)
    else:
        st.info('Use the sidebar to configure and run the simulation.')


if __name__ == '__main__':  # pragma: no cover - exercised via streamlit runtime
    main()
