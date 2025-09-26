"""Streamlit interface for running BlueSky policy simulations.

The GUI assumes that core dependencies such as :mod:`pandas` are installed.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import io
import importlib.util
import logging
import re
import shutil
import sys
import os
import tempfile
from datetime import date
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar
from uuid import uuid4

import pandas as pd



# -------------------------
# Optional imports / shims
# -------------------------
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Python 3.11+ or the tomli package is required to read TOML configuration files."
        ) from exc

try:
    from main.definitions import PROJECT_ROOT
except ModuleNotFoundError:  # fallback for packaged app execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from gui.module_settings import (
        CarbonModuleSettings,
        DispatchModuleSettings,
        GeneralConfigResult,
        IncentivesModuleSettings,
        OutputsModuleSettings,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from module_settings import (  # type: ignore[import-not-found]
        CarbonModuleSettings,
        DispatchModuleSettings,
        GeneralConfigResult,
        IncentivesModuleSettings,
        OutputsModuleSettings,
    )

try:
    from gui import price_floor
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import price_floor  # type: ignore[import-not-found]

# Region metadata helpers (robust to running as a script)
try:
    from gui.region_metadata import (
        DEFAULT_REGION_METADATA,
        canonical_region_label,
        canonical_region_value,
        region_alias_map,
        region_display_label,
    )
except ModuleNotFoundError:
    try:
        from region_metadata import (  # type: ignore[import-not-found]
            DEFAULT_REGION_METADATA,
            canonical_region_label,
            canonical_region_value,
            region_alias_map,
            region_display_label,
        )
    except ModuleNotFoundError:
        # Safe no-op fallbacks so the UI can still render
        DEFAULT_REGION_METADATA = {}
        region_alias_map = {}

        def canonical_region_label(x: object) -> str:
            return str(x)

        def canonical_region_value(x: object):
            return x

        def region_display_label(x: object) -> str:
            return str(x)


try:
    from gui.rggi import apply_rggi_defaults
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    try:
        from rggi import apply_rggi_defaults  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
        def apply_rggi_defaults(modules: dict[str, Any]) -> None:
            return None


if importlib.util.find_spec("streamlit") is not None:  # pragma: no cover - optional dependency
    import streamlit as st  # type: ignore[import-not-found]
else:  # pragma: no cover - optional dependency
    st = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from engine.run_loop import run_end_to_end_from_frames as _RUN_END_TO_END
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    try:  # pragma: no cover - optional dependency
        from engine.run_loop import run_end_to_end_from_frames as _RUN_END_TO_END
    except ModuleNotFoundError:
        _RUN_END_TO_END = None

try:
    from io_loader import Frames
except ModuleNotFoundError:  # pragma: no cover - fallback when root not on sys.path
    sys.path.append(str(PROJECT_ROOT))
    from io_loader import Frames

FramesType = Frames

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SESSION_RUN_TOKEN_KEY = "_app_session_run_token"
_CURRENT_SESSION_RUN_TOKEN = str(uuid4())
_SCRIPT_ITERATION_KEY = "_app_script_iteration"
_ACTIVE_RUN_ITERATION_KEY = "_app_active_run_iteration"

try:  # pragma: no cover - optional dependency shim
    from src.common.utilities import get_downloads_directory as _get_downloads_directory
except ImportError:  # pragma: no cover - compatibility fallback
    _get_downloads_directory = None

# Tech metadata
try:
    from src.models.electricity.scripts.technology_metadata import (
        TECH_ID_TO_LABEL,
        get_technology_label,
        resolve_technology_key,
    )
except ModuleNotFoundError:
    TECH_ID_TO_LABEL = {}
    def get_technology_label(x: Any) -> str: return str(x)
    def resolve_technology_key(x: Any) -> int | None:
        try:
            return int(x)
        except Exception:
            return None

# -------------------------
# Constants
# -------------------------
STREAMLIT_REQUIRED_MESSAGE = (
    "streamlit is required to run the policy simulator UI. Install streamlit to continue."
)
ENGINE_RUNNER_REQUIRED_MESSAGE = (
    "engine.run_loop.run_end_to_end_from_frames is required to run the policy simulator UI."
)
DEEP_CARBON_UNSUPPORTED_MESSAGE = (
    "The installed simulation engine does not support the deep carbon pricing mode."
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


DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT, "src", "common", "run_config.toml")
_DEFAULT_LOAD_MWH = 1_000_000.0
_LARGE_ALLOWANCE_SUPPLY = 1e12
_GENERAL_REGIONS_NORMALIZED_KEY = "general_regions_normalized_selection"
_ALL_REGIONS_LABEL = "All regions"
_T = TypeVar("_T")

_RUNNER_SIGNATURE: inspect.Signature | None = None
_RUNNER_ACCEPTS_VAR_KEYWORDS: bool | None = None
_RUNNER_SIGNATURE_CACHE_RUNNER: Callable[..., Any] | None = None
_RUNNER_KEYWORD_SUPPORT: dict[str, bool] = {}


SIDEBAR_SECTIONS: list[tuple[str, bool]] = [
    ("General config", False),
    ("Carbon policy", False),
    ("Electricity dispatch", False),
    ("Incentives / credits", False),
    ("Outputs", False),
]

_GENERAL_PRESET_STATE_KEY = "general_config_active_preset"
_GENERAL_PRESET_WIDGET_KEY = "general_config_preset_option"

_GENERAL_CONFIG_PRESETS = [
    ("manual", "Manual configuration", None, False),
    ("rggi", "Eastern Interconnection – RGGI", apply_rggi_defaults, True),
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

_download_directory_fallback_used = False

from dataclasses import dataclass, field
from typing import Any

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
    preset_key: str | None = None
    preset_label: str | None = None
    lock_carbon_controls: bool = False


@dataclass
class CarbonModuleSettings:
    """Record of carbon policy sidebar selections."""

    enabled: bool
    price_enabled: bool
    enable_floor: bool
    enable_ccr: bool
    ccr1_enabled: bool
    ccr2_enabled: bool
    ccr1_price: float | None
    ccr2_price: float | None
    ccr1_escalator_pct: float
    ccr2_escalator_pct: float
    banking_enabled: bool
    coverage_regions: list[str]
    control_period_years: int | None
    price_per_ton: float
    price_escalator_pct: float = 0.0
    initial_bank: float = 0.0
    cap_regions: list[Any] = field(default_factory=list)
    cap_start_value: float | None = None
    cap_reduction_mode: str = "percent"
    cap_reduction_value: float = 0.0
    cap_schedule: dict[int, float] = field(default_factory=dict)
    floor_value: float = 0.0
    floor_escalator_mode: str = "fixed"
    floor_escalator_value: float = 0.0
    floor_schedule: dict[int, float] = field(default_factory=dict)
    price_schedule: dict[int, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class CarbonPolicyConfig:
    """Normalized carbon allowance policy configuration for engine runs."""

    enabled: bool = True
    enable_floor: bool = True
    enable_ccr: bool = True
    ccr1_enabled: bool = True
    ccr2_enabled: bool = True
    ccr1_price: float | None = None
    ccr2_price: float | None = None
    ccr1_escalator_pct: float = 0.0
    ccr2_escalator_pct: float = 0.0
    allowance_banking_enabled: bool = True
    control_period_years: int | None = None
    floor_value: float = 0.0
    floor_escalator_mode: str = "fixed"
    floor_escalator_value: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        enabled: bool | None = None,
        enable_floor: bool | None = None,
        enable_ccr: bool | None = None,
        ccr1_enabled: bool | None = None,
        ccr2_enabled: bool | None = None,
        allowance_banking_enabled: bool | None = None,
        control_period_years: int | None = None,
        floor_value: float | None = None,
        floor_escalator_mode: str | None = None,
        floor_escalator_value: float | None = None,
        ccr1_price: float | None = None,
        ccr2_price: float | None = None,
        ccr1_escalator_pct: float | None = None,
        ccr2_escalator_pct: float | None = None,
    ) -> "CarbonPolicyConfig":
        record = dict(mapping) if isinstance(mapping, Mapping) else {}

        def _coerce_bool(value: Any, default: bool) -> bool:
            return bool(value) if value is not None else default

        enabled_val = _coerce_bool(enabled, bool(record.get('enabled', True)))
        enable_floor_val = _coerce_bool(enable_floor, bool(record.get('enable_floor', True)))
        enable_ccr_val = _coerce_bool(enable_ccr, bool(record.get('enable_ccr', True)))
        ccr1_val = _coerce_bool(ccr1_enabled, bool(record.get('ccr1_enabled', True)))
        ccr2_val = _coerce_bool(ccr2_enabled, bool(record.get('ccr2_enabled', True)))
        banking_val = _coerce_bool(
            allowance_banking_enabled,
            bool(record.get('allowance_banking_enabled', True)),
        )

        control_period_val = _sanitize_control_period(
            control_period_years
            if control_period_years is not None
            else record.get('control_period_years')
        )

        floor_value_raw = floor_value if floor_value is not None else record.get("floor_value")
        floor_value_val = price_floor.parse_currency_value(floor_value_raw, 0.0)
        floor_mode_raw = (
            floor_escalator_mode if floor_escalator_mode is not None else record.get("floor_escalator_mode")
        )
        floor_mode_val = str(floor_mode_raw or "fixed").strip().lower()
        if floor_mode_val not in {"fixed", "percent"}:
            floor_mode_val = "fixed"
        floor_escalator_raw = (
            floor_escalator_value if floor_escalator_value is not None else record.get("floor_escalator_value")
        )
        if floor_mode_val == "percent":
            floor_escalator_val = price_floor.parse_percentage_value(floor_escalator_raw, 0.0)
        else:
            floor_escalator_val = price_floor.parse_currency_value(floor_escalator_raw, 0.0)
        def _coerce_optional(value: Any, fallback: float | None) -> float | None:
            if value in (None, ""):
                return fallback
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        ccr1_price_val = _coerce_optional(
            ccr1_price if ccr1_price is not None else record.get('ccr1_price'),
            None,
        )
        ccr2_price_val = _coerce_optional(
            ccr2_price if ccr2_price is not None else record.get('ccr2_price'),
            None,
        )
        ccr1_escalator_val = _coerce_optional(
            ccr1_escalator_pct
            if ccr1_escalator_pct is not None
            else record.get('ccr1_escalator_pct'),
            0.0,
        ) or 0.0
        ccr2_escalator_val = _coerce_optional(
            ccr2_escalator_pct
            if ccr2_escalator_pct is not None
            else record.get('ccr2_escalator_pct'),
            0.0,
        ) or 0.0

        config = cls(
            enabled=enabled_val,
            enable_floor=enable_floor_val,
            enable_ccr=enable_ccr_val,
            ccr1_enabled=ccr1_val,
            ccr2_enabled=ccr2_val,
            ccr1_price=ccr1_price_val,
            ccr2_price=ccr2_price_val,
            ccr1_escalator_pct=float(ccr1_escalator_val),
            ccr2_escalator_pct=float(ccr2_escalator_val),
            allowance_banking_enabled=banking_val,
            control_period_years=control_period_val,
            floor_value=float(floor_value_val),
            floor_escalator_mode=str(floor_mode_val),
            floor_escalator_value=float(floor_escalator_val),
        )

        if not config.enabled:
            config.disable_cap()
        elif not config.enable_ccr:
            config.ccr1_enabled = False
            config.ccr2_enabled = False

        return config

    def disable_cap(self) -> None:
        """Disable allowance trading mechanics when the cap is inactive."""

        self.enabled = False
        self.enable_floor = False
        self.enable_ccr = False
        self.ccr1_enabled = False
        self.ccr2_enabled = False
        self.ccr1_price = None
        self.ccr2_price = None
        self.ccr1_escalator_pct = 0.0
        self.ccr2_escalator_pct = 0.0
        self.allowance_banking_enabled = False
        self.control_period_years = None

    def disable_for_price(self) -> None:
        """Disable the cap when an exogenous carbon price is active."""

        self.disable_cap()

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""

        return {
            'enabled': bool(self.enabled),
            'enable_floor': bool(self.enable_floor),
            'enable_ccr': bool(self.enable_ccr),
            'ccr1_enabled': bool(self.ccr1_enabled),
            'ccr2_enabled': bool(self.ccr2_enabled),
            'ccr1_price': self.ccr1_price,
            'ccr2_price': self.ccr2_price,
            'ccr1_escalator_pct': float(self.ccr1_escalator_pct),
            'ccr2_escalator_pct': float(self.ccr2_escalator_pct),
            'allowance_banking_enabled': bool(self.allowance_banking_enabled),
            'control_period_years': self.control_period_years,
        }


@dataclass
class RunProgressState:
    """State container for tracking and rendering run progress."""

    stage: str = "idle"
    message: str = ""
    percent_complete: int = 0
    total_years: int = 1
    current_index: int = -1
    current_year: Any | None = None
    log: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """Return the tracker to an initial idle state."""

        self.stage = "idle"
        self.message = ""
        self.percent_complete = 0
        self.total_years = 1
        self.current_index = -1
        self.current_year = None
        self.log.clear()


def _ensure_progress_state() -> RunProgressState:
    """Return the progress tracker stored in the current Streamlit session."""

    if st is None:
        raise ModuleNotFoundError(STREAMLIT_REQUIRED_MESSAGE)

    state = st.session_state.get("_run_progress_state")
    if isinstance(state, RunProgressState):
        return state

    tracker = RunProgressState()
    st.session_state["_run_progress_state"] = tracker
    return tracker


def _reset_progress_state() -> RunProgressState:
    """Reset the session progress tracker and return it."""

    tracker = _ensure_progress_state()
    tracker.reset()
    return tracker


def _trigger_streamlit_rerun() -> bool:
    """Request that Streamlit immediately rerun the script."""

    if st is None:
        return False

    for attr in ("rerun", "experimental_rerun"):
        rerun_fn = getattr(st, attr, None)
        if callable(rerun_fn):
            rerun_fn()
            return True

    return False

def _bounded_percent(value: float | int) -> int:
    """Clamp a numeric percent to the inclusive range [0, 100]."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, int(round(numeric))))


def _progress_update_from_stage(
    stage: str,
    payload: Mapping[str, object],
    state: RunProgressState,
) -> tuple[str, int]:
    """Derive a status message and completion percent for the given progress stage."""

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

    message = state.message or ""
    percent = state.percent_complete

    if stage == "run_start":
        total = _as_int(payload.get("total_years"), 0)
        if total <= 0:
            total = 1
        state.total_years = total
        state.current_index = -1
        state.current_year = None
        message = f"Preparing simulation for {total} year(s)…"
        percent = 0
    elif stage == "year_start":
        index = _as_int(payload.get("index"), 0)
        total = max(state.total_years, 1)
        year_val = payload.get("year")
        state.current_index = index
        state.current_year = year_val
        fraction = max(0.0, min(1.0, index / total))
        percent = int(round(fraction * 100))
        year_display = str(year_val) if year_val is not None else "N/A"
        message = f"Simulating year {year_display} ({index + 1} of {total})"
    elif stage == "iteration":
        iteration = _as_int(payload.get("iteration"), 0)
        year_val = payload.get("year", state.current_year)
        year_display = str(year_val) if year_val is not None else "N/A"
        price_val = _as_float(payload.get("price"))
        if price_val is not None:
            message = (
                f"Year {year_display}: iteration {iteration} "
                f"(price ≈ {price_val:,.2f})"
            )
        else:
            message = f"Year {year_display}: iteration {iteration}"
    elif stage == "year_complete":
        index = _as_int(payload.get("index"), state.current_index)
        total = max(state.total_years, 1)
        year_val = payload.get("year", state.current_year)
        state.current_index = index
        state.current_year = year_val
        fraction = max(0.0, min(1.0, (index + 1) / total))
        percent = int(round(fraction * 100))
        price_val = _as_float(payload.get("price"))
        year_display = str(year_val) if year_val is not None else str(index + 1)
        if price_val is not None:
            message = (
                f"Completed year {year_display} of {total} "
                f"(price {price_val:,.2f})"
            )
        else:
            message = f"Completed year {year_display} of {total}"
    else:
        message = f"{stage.replace('_', ' ').title()}…"

    return message, _bounded_percent(percent)


def _record_progress_log(state: RunProgressState, message: str, stage: str) -> None:
    """Append a readable entry to the progress log, coalescing noisy updates."""

    if not message:
        return

    if stage == "iteration":
        if state.log and "iteration" in state.log[-1]:
            state.log[-1] = message
        else:
            state.log.append(message)
    else:
        state.log.append(message)

    max_entries = 60
    if len(state.log) > max_entries:
        state.log[:] = state.log[-max_entries:]


def _progress_log_markdown(entries: Sequence[str]) -> str:
    """Render the most recent progress entries as a markdown bullet list."""

    recent = list(entries)[-12:]
    return "\n".join(f"- {entry}" for entry in recent)


def _sync_progress_ui(
    state: RunProgressState,
    message_placeholder,
    progress_placeholder,
    log_placeholder,
) -> None:
    """Synchronize the rendered progress widgets with the stored state."""

    message = state.message.strip()
    if message:
        message_placeholder.write(message)
    else:
        message_placeholder.caption("Run a simulation to view progress updates.")

    percent = _bounded_percent(state.percent_complete)
    if state.stage == "idle" and percent == 0 and not state.log:
        progress_placeholder.empty()
    else:
        progress_placeholder.progress(percent)

    if state.log:
        log_placeholder.markdown(_progress_log_markdown(state.log))
    else:
        log_placeholder.caption("Progress updates will appear here during the run.")


@dataclass
class CarbonPriceConfig:
    """Normalized carbon price configuration for engine runs."""

    enabled: bool = False
    price_per_ton: float = 0.0
    escalator_pct: float = 0.0
    schedule: dict[int, float] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Return ``True`` when the price should override the cap."""

        return bool(self.enabled and self.schedule)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        enabled: bool | None = None,
        value: float | None = None,
        schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
        years: Iterable[int] | None = None,
        escalator_pct: float | None = None,
    ) -> "CarbonPriceConfig":
        record = dict(mapping) if isinstance(mapping, Mapping) else {}

        enabled_val = bool(enabled) if enabled is not None else bool(record.get('enabled', False))
        price_raw = value if value is not None else record.get('price_per_ton', record.get('price', 0.0))
        price_value = _coerce_float(price_raw, default=0.0)
        escalator_raw = (
            escalator_pct
            if escalator_pct is not None
            else record.get('price_escalator_pct', record.get('escalator_pct', 0.0))
        )
        escalator_value = _coerce_float(escalator_raw, default=0.0)

        schedule_map = _merge_price_schedules(
            record.get('price_schedule'),
            schedule,
        )

        if not schedule_map and years:
            normalized_years: list[int] = []
            for year in years:
                try:
                    normalized_years.append(int(year))
                except (TypeError, ValueError):
                    continue
            if normalized_years:
                generated_schedule = _build_price_escalator_schedule(
                    price_value,
                    escalator_value,
                    sorted(set(normalized_years)),
                )
                if generated_schedule:
                    schedule_map = generated_schedule

        if schedule_map:
            schedule_map = dict(sorted(schedule_map.items()))

        config = cls(
            enabled=bool(enabled_val),
            price_per_ton=float(price_value),
            escalator_pct=float(escalator_value),
            schedule=schedule_map,
        )

        if not config.active:
            config.schedule = {}
            config.price_per_ton = 0.0
            config.escalator_pct = 0.0

        return config

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""

        payload = {
            'enabled': bool(self.enabled),
            'price_per_ton': float(self.price_per_ton),
            'price_escalator_pct': float(self.escalator_pct),
        }
        if self.schedule:
            payload['price_schedule'] = dict(self.schedule)
        return payload


def _sanitize_control_period(value: Any) -> int | None:
    """Return ``value`` coerced to a positive integer when possible."""

    if value in (None, ''):
        return None
    try:
        period = int(value)
    except (TypeError, ValueError):
        return None
    return period if period > 0 else None


def _normalize_price_schedule(value: Any) -> dict[int, float]:
    """Return a normalized mapping of year to carbon price."""

    if not isinstance(value, Mapping):
        return {}

    entries: list[tuple[int, float]] = []
    for key, raw in value.items():
        if raw in (None, ""):
            continue
        try:
            year = int(key)
        except (TypeError, ValueError):
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        entries.append((year, price))

    if not entries:
        return {}

    entries.sort(key=lambda item: item[0])
    return {year: price for year, price in entries}


def _merge_price_schedules(
    *values: Mapping[int, float] | Mapping[str, Any] | None,
) -> dict[int, float]:
    """Combine candidate schedules, returning a sorted ``{year: price}`` mapping."""

    merged: dict[int, float] = {}
    for candidate in values:
        if not isinstance(candidate, Mapping):
            continue
        merged.update(_normalize_price_schedule(candidate))

    if not merged:
        return {}

    return dict(sorted(merged.items()))


def _expand_or_build_price_schedule(
    schedule: Mapping[int, float] | None,
    years: Iterable[int] | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
    base: float | None = None,
    esc_pct: float | None = None,
) -> dict[int, float]:
    """
    Expand an explicit schedule to all years, or build a schedule with escalator logic.

    - If `schedule` is provided, expand it across the requested `years` with no gaps.
    - If `schedule` is empty, but start/end/base/esc_pct are given, build a schedule
      growing by esc_pct% per year.
    """

    # Case 1: Explicit schedule provided
    if schedule:
        normalized_years: list[int] = []
        if years is not None:
            for entry in years:
                try:
                    normalized_years.append(int(entry))
                except (TypeError, ValueError):
                    continue

        schedule_items = [(int(year), float(price)) for year, price in schedule.items()]
        if not schedule_items:
            return {}

        schedule_items.sort(key=lambda item: item[0])
        if not normalized_years:
            return dict(schedule_items)

        expanded: dict[int, float] = {}
        sorted_years = sorted(dict.fromkeys(normalized_years))
        current_price = schedule_items[0][1]
        index = 0
        total_schedule = len(schedule_items)

        for year in sorted_years:
            while index < total_schedule and schedule_items[index][0] <= year:
                current_price = schedule_items[index][1]
                index += 1
            expanded[year] = float(current_price)
        return expanded

    # Case 2: Build schedule from base + escalator
    try:
        start_year = int(start) if start is not None else None
        end_year = int(end) if end is not None else None
    except (TypeError, ValueError):
        return {}

    if start_year is None or end_year is None:
        return {}

    try:
        base_value = float(base) if base is not None else 0.0
    except (TypeError, ValueError):
        base_value = 0.0

    try:
        escalator_value = float(esc_pct) if esc_pct is not None else 0.0
    except (TypeError, ValueError):
        escalator_value = 0.0

    return _build_price_schedule(start_year, end_year, base_value, escalator_value)



def _build_price_schedule(
    start_year: int,
    end_year: int,
    base_value: float,
    escalator_pct: float,
) -> dict[int, float]:
    """Return a price schedule that grows geometrically each year."""

    try:
        start = int(start_year)
    except (TypeError, ValueError):
        return {}
    try:
        end = int(end_year)
    except (TypeError, ValueError):
        return {}

    try:
        base = float(base_value)
    except (TypeError, ValueError):
        base = 0.0
    try:
        escalator = float(escalator_pct)
    except (TypeError, ValueError):
        escalator = 0.0

    step = 1 if end >= start else -1
    ratio = 1.0 + (escalator or 0.0) / 100.0

    schedule_items: list[tuple[int, float]] = []
    for exponent, year in enumerate(range(start, end + step, step)):
        try:
            factor = ratio ** exponent
        except OverflowError:
            factor = float("inf")
        schedule_items.append((year, round(base * factor, 6)))

    if not schedule_items:
        return {}

    return dict(sorted(schedule_items))


def _build_price_escalator_schedule(
    base_price: float,
    escalator_pct: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a price schedule grown annually by ``escalator_pct``."""

    if years is None:
        return {}

    try:
        base_value = float(base_price)
    except (TypeError, ValueError):
        base_value = 0.0
    try:
        escalator_value = float(escalator_pct)
    except (TypeError, ValueError):
        escalator_value = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    ordered_years = sorted(dict.fromkeys(year_list))
    base_year = ordered_years[0]
    full_schedule = _build_price_schedule(
        base_year,
        ordered_years[-1],
        base_value,
        escalator_value,
    )
    return {year: float(full_schedule.get(year, base_value)) for year in ordered_years}


def _build_cap_reduction_schedule(
    start_value: float,
    reduction_mode: str,
    reduction_value: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a cap schedule reduced each year by the specified rule."""

    if years is None:
        return {}

    try:
        start_amount = float(start_value)
    except (TypeError, ValueError):
        start_amount = 0.0
    try:
        reduction_amount = float(reduction_value)
    except (TypeError, ValueError):
        reduction_amount = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    normalized_mode = (reduction_mode or "").strip().lower()
    if normalized_mode not in {"percent", "fixed"}:
        normalized_mode = "percent"

    schedule: dict[int, float] = {}
    for idx, year in enumerate(sorted(dict.fromkeys(year_list))):
        if normalized_mode == "percent":
            decrement = start_amount * (max(reduction_amount, 0.0) / 100.0) * idx
        else:
            decrement = max(reduction_amount, 0.0) * idx
        value = max(start_amount - decrement, 0.0)
        schedule[year] = float(value)
    return schedule


def _merge_module_dicts(*sections: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Combine multiple module configuration sections into a copy."""

    merged: dict[str, dict[str, Any]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for name, settings in section.items():
            key = str(name)
            if isinstance(settings, Mapping):
                existing = merged.get(key, {})
                combined = dict(existing)
                combined.update(settings)
                merged[key] = combined
            else:
                merged[key] = {'value': settings}
    return merged


# -------------------------
# Utilities
# -------------------------
def _fallback_downloads_directory(app_subdir: str = "GraniteLedger") -> Path:
    base_path = Path.home() / "Downloads"
    if app_subdir:
        base_path = base_path / app_subdir
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Coerce value to float or return default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_downloads_directory(app_subdir: str = "GraniteLedger") -> Path:
    global _download_directory_fallback_used
    if _get_downloads_directory is not None:
        try:
            return _get_downloads_directory(app_subdir=app_subdir)
        except Exception:  # pragma: no cover - defensive
            LOGGER.warning("Falling back to home Downloads directory; helper raised an error.")
    if not _download_directory_fallback_used:
        LOGGER.warning(
            "get_downloads_directory is unavailable; using ~/Downloads for model outputs."
        )
        _download_directory_fallback_used = True
    return _fallback_downloads_directory(app_subdir)


def _ensure_engine_runner():
    if _RUN_END_TO_END is None:
        raise ModuleNotFoundError(ENGINE_RUNNER_REQUIRED_MESSAGE)
    return _RUN_END_TO_END


def _ensure_streamlit() -> None:
    if st is None:
        raise ModuleNotFoundError(STREAMLIT_REQUIRED_MESSAGE)


def _cache_runner_signature(runner: Callable[..., Any]) -> None:
    global _RUNNER_SIGNATURE, _RUNNER_ACCEPTS_VAR_KEYWORDS, _RUNNER_SIGNATURE_CACHE_RUNNER, _RUNNER_KEYWORD_SUPPORT

    if _RUNNER_SIGNATURE_CACHE_RUNNER is not runner:
        _RUNNER_SIGNATURE_CACHE_RUNNER = runner
        _RUNNER_SIGNATURE = None
        _RUNNER_ACCEPTS_VAR_KEYWORDS = None
        _RUNNER_KEYWORD_SUPPORT.clear()

    if _RUNNER_SIGNATURE is not None or _RUNNER_ACCEPTS_VAR_KEYWORDS is not None:
        return
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        _RUNNER_SIGNATURE = None
        _RUNNER_ACCEPTS_VAR_KEYWORDS = True
    else:
        _RUNNER_SIGNATURE = signature
        _RUNNER_ACCEPTS_VAR_KEYWORDS = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )


def _runner_supports_keyword(runner: Callable[..., Any], name: str) -> bool:
    _cache_runner_signature(runner)

    support = _RUNNER_KEYWORD_SUPPORT.get(name)
    if support is not None:
        return support

    if _RUNNER_ACCEPTS_VAR_KEYWORDS or _RUNNER_SIGNATURE is None:
        support = True
    else:
        parameter = _RUNNER_SIGNATURE.parameters.get(name)
        support = parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

    _RUNNER_KEYWORD_SUPPORT[name] = support
    return support


@contextmanager
def _sidebar_panel(container: Any, enabled: bool):
    """Render a styled sidebar panel with optional greyed-out state.

    NOTE: We do NOT `with container:` here. The caller already has the right Streamlit container.
    """
    class_name = "sidebar-module disabled" if not enabled else "sidebar-module"
    container.markdown(f'<div class="{class_name}">', unsafe_allow_html=True)
    try:
        yield container
    finally:
        container.markdown("</div>", unsafe_allow_html=True)


def _load_config_data(config_source: Any | None = None) -> dict[str, Any]:
    if config_source is None:
        with open(DEFAULT_CONFIG_PATH, "rb") as src:
            return tomllib.load(src)

    if isinstance(config_source, Mapping):
        return dict(config_source)

    if isinstance(config_source, (bytes, bytearray)):
        return tomllib.loads(config_source.decode("utf-8"))

    if isinstance(config_source, (str, Path)):
        path_candidate = Path(config_source)
        if path_candidate.exists():
            with open(path_candidate, "rb") as src:
                return tomllib.load(src)
        return tomllib.loads(str(config_source))

    if hasattr(config_source, "read"):
        data = config_source.read()
        if isinstance(data, bytes):
            return tomllib.loads(data.decode("utf-8"))
        return tomllib.loads(str(data))

    raise TypeError(f"Unsupported config source type: {type(config_source)!r}")


def _years_from_config(config: Mapping[str, Any]) -> list[int]:
    years: set[int] = set()
    raw_years = config.get("years")

    if isinstance(raw_years, (list, tuple, set)):
        for entry in raw_years:
            if isinstance(entry, Mapping) and "year" in entry:
                try:
                    years.add(int(entry["year"]))
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    years.add(int(entry))
                except (TypeError, ValueError):
                    continue
    elif raw_years not in (None, ""):
        try:
            years.add(int(raw_years))
        except (TypeError, ValueError):
            pass

    if not years:
        start = config.get("start_year")
        end = config.get("end_year")
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
    years = sorted({int(year) for year in base_years}) if base_years else []

    def _ensure_int(value: int | None) -> int | None:
        if value is None:
            return None
        return int(value)

    start = _ensure_int(start_year)
    end = _ensure_int(end_year)

    if start is not None and end is not None and end < start:
        raise ValueError("end_year must be greater than or equal to start_year")

    if start is not None and end is not None:
        selected = [year for year in years if start <= year <= end]
        complete_range = range(start, end + 1)
        if selected:
            selected_set = set(selected)
            selected_set.update(complete_range)
            selected = sorted(selected_set)
        else:
            selected = list(complete_range)
        years = selected
    elif start is not None:
        selected = [year for year in years if year >= start]
        years = selected or [start]
    elif end is not None:
        selected = [year for year in years if year <= end]
        years = selected or [end]

    if not years:
        raise ValueError("No simulation years specified")

    return sorted({int(year) for year in years})


def _regions_from_config(config: Mapping[str, Any]) -> list[int | str]:
    raw_regions = config.get("regions")
    regions: list[int | str] = []

    def _normalise(entry: Any) -> int | str:
        resolved = canonical_region_value(entry)
        if isinstance(resolved, str):
            text = resolved.strip()
            return text or "default"
        return resolved

    if isinstance(raw_regions, Mapping):
        iterable: Iterable[Any] = raw_regions.values()
    else:
        iterable = raw_regions  # type: ignore[assignment]

    if isinstance(iterable, Iterable) and not isinstance(iterable, (str, bytes, Mapping)):
        for entry in iterable:
            normalised = _normalise(entry)
            if normalised not in regions:
                regions.append(normalised)
    elif iterable not in (None, ""):
        regions.append(_normalise(iterable))

    if not regions:
        regions = [1]

    return regions


def _normalize_region_labels(
    selected_labels: Iterable[str],
    previous_clean_selection: Iterable[str] | None,
) -> list[str]:
    normalized = [str(entry) for entry in selected_labels]
    if "All" in normalized and len(normalized) > 1:
        non_all = [e for e in normalized if e != "All"]
        prev = tuple(str(e) for e in (previous_clean_selection or ()))
        return non_all if prev == ("All",) and non_all else ["All"]
    return normalized


def _normalize_coverage_selection(selection: Any) -> list[str]:
    """Return a normalised list of coverage region labels."""

    if isinstance(selection, Mapping):
        iterable: Iterable[Any] = selection.values()
    elif isinstance(selection, (str, bytes)) or not isinstance(selection, Iterable):
        iterable = [selection]
    else:
        iterable = selection

    normalized: list[str] = []
    for entry in iterable:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
            return ["All"]
        label = canonical_region_label(entry)
        if label.lower() in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
            return ["All"]
        if label not in normalized:
            normalized.append(label)

    if not normalized:
        return ["All"]
    return normalized


def _normalize_cap_region_entries(
    selection: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[list[int | str], dict[str, int | str]]:
    """Return canonical cap region values and an alias map for lookup."""

    normalized: list[int | str] = []
    seen: set[int | str] = set()
    alias_source = {key.lower(): value for key, value in region_alias_map().items()}
    alias_map: dict[str, int | str] = {}

    def _register_alias(key: Any, value: int | str) -> None:
        if key is None:
            return
        text = str(key).strip()
        if not text:
            return
        alias_map.setdefault(text, value)
        alias_map.setdefault(text.lower(), value)

    for region_id, meta in DEFAULT_REGION_METADATA.items():
        _register_alias(region_id, region_id)
        _register_alias(str(region_id), region_id)
        _register_alias(meta.code, region_id)
        _register_alias(meta.code.lower(), region_id)
        _register_alias(meta.label, region_id)
        _register_alias(meta.label.lower(), region_id)
        _register_alias(meta.area, region_id)
        _register_alias(meta.area.lower(), region_id)
        for alias in meta.aliases:
            _register_alias(alias, region_id)
            _register_alias(alias.lower(), region_id)

    encountered_all = False
    unresolved: list[str] = []

    if selection is None:
        return normalized, alias_map

    if isinstance(selection, Mapping):
        iterable: Iterable[Any] = selection.values()
    elif isinstance(selection, (str, bytes)):
        iterable = [selection]
    else:
        iterable = selection

    for entry in iterable:
        if entry in (None, ""):
            continue

        label = canonical_region_label(entry).strip()
        lowered_label = label.lower()
        if lowered_label in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
            encountered_all = True
            continue

        resolved = canonical_region_value(entry)
        if isinstance(resolved, str):
            text = resolved.strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
                encountered_all = True
                continue
            if lowered == "default":
                canonical_value: int | str = "default"
            else:
                match = alias_source.get(lowered)
                if match is None:
                    unresolved.append(label or text or str(entry))
                    continue
                canonical_value = int(match)
        else:
            canonical_value = int(resolved)

        _register_alias(entry, canonical_value)
        _register_alias(label, canonical_value)
        if not isinstance(canonical_value, str):
            _register_alias(str(canonical_value), canonical_value)
        else:
            _register_alias(canonical_value, canonical_value)

        if canonical_value not in seen:
            seen.add(canonical_value)
            normalized.append(canonical_value)

    if unresolved:
        unique_unresolved = list(dict.fromkeys(unresolved))
        if len(unique_unresolved) == 1:
            raise ValueError(f"Unable to resolve cap region '{unique_unresolved[0]}'")
        unresolved_list = ", ".join(f"'{entry}'" for entry in unique_unresolved)
        raise ValueError(f"Unable to resolve cap regions: {unresolved_list}")

    if encountered_all and not normalized:
        return [], alias_map

    return normalized, alias_map


# General Config UI
# -------------------------
def _render_general_config_section(
    container: Any,
    *,
    default_source: Any,
    default_label: str,
    default_config: Mapping[str, Any],
) -> GeneralConfigResult:
    try:
        base_config = copy.deepcopy(dict(default_config))
    except Exception:
        base_config = dict(default_config)

    preset_key = "manual"
    preset_label: str | None = None
    preset_apply: Callable[[dict[str, Any]], None] | None = None
    preset_locks_carbon = False

    if st is not None:
        default_preset_key = st.session_state.get(
            _GENERAL_PRESET_STATE_KEY,
            _GENERAL_CONFIG_PRESETS[0][0],
        )
        preset_labels = [entry[1] for entry in _GENERAL_CONFIG_PRESETS]
        try:
            default_index = next(
                idx
                for idx, entry in enumerate(_GENERAL_CONFIG_PRESETS)
                if entry[0] == default_preset_key
            )
        except StopIteration:
            default_index = 0
        selected_label = container.radio(
            "Configuration preset",
            options=preset_labels,
            index=default_index,
            key=_GENERAL_PRESET_WIDGET_KEY,
            help=(
                "Select a pre-configured scenario or edit the default configuration manually."
            ),
        )
        for key, label, apply_fn, lock_flag in _GENERAL_CONFIG_PRESETS:
            if label == selected_label:
                preset_key = key
                preset_label = label if key != "manual" else None
                preset_apply = apply_fn
                preset_locks_carbon = bool(lock_flag)
                st.session_state[_GENERAL_PRESET_STATE_KEY] = key
                break
    else:
        preset_key, _, preset_apply, lock_flag = _GENERAL_CONFIG_PRESETS[0]
        preset_label = None
        preset_locks_carbon = bool(lock_flag)

    config_label = default_label

    if preset_key == "manual":
        uploaded = container.file_uploader(
            "Run configuration (TOML)",
            type="toml",
            key="general_config_upload",
        )
        if uploaded is not None:
            config_label = uploaded.name or "uploaded_config.toml"
            try:
                base_config = _load_config_data(uploaded.getvalue())
            except Exception as exc:
                container.error(f"Failed to read configuration: {exc}")
                base_config = copy.deepcopy(dict(default_config))
                config_label = default_label
    else:
        config_label = preset_label or default_label

    container.caption(f"Using configuration: {config_label}")
    if preset_key != "manual":
        container.info(
            "Preset values are loaded automatically. Carbon policy settings are locked while this preset is active."
        )

    candidate_years = _years_from_config(base_config)
    current_year = date.today().year
    if candidate_years:
        year_min = min(candidate_years)
        year_max = max(candidate_years)
    else:
        try:
            year_min = int(base_config.get("start_year", current_year) or current_year)
        except (TypeError, ValueError):
            year_min = int(current_year)
        try:
            fallback_end = base_config.get("end_year", year_min + 1)
            year_max = int(fallback_end) if fallback_end not in (None, "") else year_min + 1
        except (TypeError, ValueError):
            year_max = year_min + 1
        if year_max <= year_min:
            year_max = year_min + 1
    if year_min > year_max:
        year_min, year_max = year_max, year_min

    def _coerce_year(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)

    start_default = max(
        year_min,
        min(year_max, _coerce_year(base_config.get("start_year", year_min), year_min)),
    )
    end_default = max(
        year_min,
        min(year_max, _coerce_year(base_config.get("end_year", year_max), year_max)),
    )
    if end_default <= start_default:
        end_default = start_default + 1

    # Hard bounds
    slider_min_dynamic = 2025
    slider_max_dynamic = 2050

    slider_min_value, slider_max_value = container.slider(
        "Simulation Years",
        min_value=slider_min_dynamic,
        max_value=slider_max_dynamic,
        value=(
            max(slider_min_dynamic, min(start_default, slider_max_dynamic - 1)),
            max(
                max(slider_min_dynamic + 1, min(end_default, slider_max_dynamic)),
                slider_min_dynamic + 1,
            ),
        ),
        step=1,
        key="general_year_slider",
    )


    start_year = int(slider_min_value)
    end_year = int(slider_max_value)

    if st is not None:
        st.session_state["start_year_slider"] = start_year
        st.session_state["end_year_slider"] = end_year

    invalid_year_range = start_year >= end_year
    if invalid_year_range:
        container.error("End year must be greater than start year.")

    region_options = _regions_from_config(base_config)
    default_region_values = list(range(1, 26))
    alias_to_value = region_alias_map()
    available_region_values: list[int | str] = []
    value_to_label: dict[int | str, str] = {}
    label_to_value: dict[str, int | str] = {}
    seen_values: set[int | str] = set()

    def _register_region_value(value: int | str) -> None:
        if value in seen_values:
            return
        seen_values.add(value)
        label = region_display_label(value)
        available_region_values.append(value)
        value_to_label[value] = label
        label_to_value[label] = value
        alias_to_value.setdefault(label.lower(), value)
        alias_to_value.setdefault(str(value).strip().lower(), value)
        if isinstance(value, (bool, int)):
            alias_to_value.setdefault(f"region {int(value)}".lower(), value)

    for candidate in (*default_region_values, *region_options):
        resolved = canonical_region_value(candidate)
        if isinstance(resolved, str):
            text = resolved.strip()
            if text:
                _register_region_value(text)
        else:
            _register_region_value(int(resolved))

    region_labels = ["All"] + [value_to_label[v] for v in available_region_values]
    default_selection = ["All"]

    def _canonical_region_label_entry(entry: Any) -> str:
        text = str(entry).strip()
        if not text:
            return text
        if text == "All":
            return "All"
        if text in label_to_value:
            return text
        lookup_key = text.lower()
        value = alias_to_value.get(lookup_key)
        if value is None:
            try:
                value = int(text)
            except ValueError:
                return text
        return value_to_label.get(value, region_display_label(value))

    def _canonicalize_selection(entries: Iterable[Any]) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            label = _canonical_region_label_entry(entry)
            if label and label not in seen:
                canonical.append(label)
                seen.add(label)
        if not canonical:
            canonical = list(default_selection)
        return canonical

    if st is not None:
        st.session_state.setdefault(
            _GENERAL_REGIONS_NORMALIZED_KEY, list(default_selection)
        )
        prev_raw = st.session_state.get(_GENERAL_REGIONS_NORMALIZED_KEY, [])
        if isinstance(prev_raw, (list, tuple)):
            previous_clean_selection = _canonicalize_selection(prev_raw)
        elif isinstance(prev_raw, str):
            previous_clean_selection = _canonicalize_selection([prev_raw])
        else:
            previous_clean_selection = list(default_selection)

        existing_widget_value = st.session_state.get("general_regions")
        if isinstance(existing_widget_value, str):
            existing_entries: Iterable[Any] = [existing_widget_value]
        elif isinstance(existing_widget_value, (list, tuple, set)):
            existing_entries = existing_widget_value
        else:
            existing_entries = []

        if existing_entries:
            canonical_existing = _canonicalize_selection(existing_entries)
        else:
            canonical_existing = previous_clean_selection

        previous_clean_selection = canonical_existing
    else:
        previous_clean_selection = list(default_selection)

    selected_regions_raw = list(
        container.multiselect(
            "Regions",
            options=region_labels,
            default=previous_clean_selection,
            key="general_regions",
        )
    )
    normalized_selection = _normalize_region_labels(
        selected_regions_raw, previous_clean_selection
    )
    canonical_selection: list[str] = []
    seen_labels: set[str] = set()
    for entry in normalized_selection:
        label = _canonical_region_label_entry(entry)
        if label and label not in seen_labels:
            canonical_selection.append(label)
            seen_labels.add(label)
    selected_regions_raw = canonical_selection
    if st is not None:
        st.session_state[_GENERAL_REGIONS_NORMALIZED_KEY] = list(selected_regions_raw)

    all_selected = "All" in selected_regions_raw
    if all_selected or not selected_regions_raw:
        selected_regions = list(available_region_values)
    else:
        selected_regions = []
        for entry in selected_regions_raw:
            if entry == "All":
                continue
            value = label_to_value.get(entry)
            if value is None:
                normalized = str(entry).strip().lower()
                value = alias_to_value.get(normalized)
                if value is None:
                    try:
                        value = int(str(entry).strip())
                    except ValueError:
                        value = str(entry).strip()
            if value not in selected_regions:
                selected_regions.append(value)
    if not selected_regions:
        selected_regions = list(available_region_values)

    run_config = copy.deepcopy(base_config)
    run_config["start_year"] = start_year
    run_config["end_year"] = end_year
    run_config["regions"] = selected_regions
    run_config.setdefault("modules", {})

    active_preset_key: str | None = None
    active_preset_label: str | None = None
    lock_carbon_controls = False
    if preset_key != "manual" and preset_apply is not None:
        try:
            preset_apply(run_config["modules"])
        except Exception as exc:
            container.error(f"Failed to apply preset defaults: {exc}")
        else:
            active_preset_key = preset_key
            active_preset_label = preset_label
            lock_carbon_controls = bool(preset_locks_carbon)

    try:
        selected_years = _select_years(candidate_years, start_year, end_year)
    except Exception:
        selected_years = []
    if selected_years:
        try:
            selected_min = min(int(year) for year in selected_years)
            selected_max = max(int(year) for year in selected_years)
        except ValueError:
            selected_years = []
        else:
            selected_years = list(range(selected_min, selected_max + 1))
    elif not invalid_year_range:
        selected_years = list(range(start_year, end_year + 1))
    else:
        selected_years = []

    return GeneralConfigResult(
        config_label=config_label,
        config_source=run_config,
        run_config=run_config,
        candidate_years=candidate_years,
        start_year=start_year,
        end_year=end_year,
        selected_years=selected_years,
        regions=selected_regions,
        preset_key=active_preset_key,
        preset_label=active_preset_label,
        lock_carbon_controls=lock_carbon_controls,
    )


def _render_carbon_policy_section(
    container: Any,
    run_config: dict[str, Any],
    *,
    years: Iterable[int] | None = None,
    region_options: Iterable[Any] | None = None,
    lock_inputs: bool = False,
) -> CarbonModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("carbon_policy", {}) or {}
    price_defaults = modules.get("carbon_price", {}) or {}
    dispatch_defaults = modules.get("electricity_dispatch", {}) or {}

    # -------------------------
    # Defaults
    # -------------------------
    enabled_default = bool(defaults.get("enabled", True))
    enable_floor_default = bool(defaults.get("enable_floor", True))
    enable_ccr_default = bool(defaults.get("enable_ccr", True))
    ccr1_default = bool(defaults.get("ccr1_enabled", True))
    ccr2_default = bool(defaults.get("ccr2_enabled", True))
    ccr1_price_default = _coerce_optional_float(defaults.get("ccr1_price"))
    ccr2_price_default = _coerce_optional_float(defaults.get("ccr2_price"))
    ccr1_escalator_default = _coerce_float(defaults.get("ccr1_escalator_pct"), 0.0)
    ccr2_escalator_default = _coerce_float(defaults.get("ccr2_escalator_pct"), 0.0)
    banking_default = bool(defaults.get("allowance_banking_enabled", True))
    bank_default = _coerce_float(defaults.get("bank0", 0.0), default=0.0)

    coverage_default = _normalize_coverage_selection(
        defaults.get("coverage_regions", ["All"])
    )

    control_default_raw = defaults.get("control_period_years")
    try:
        control_default = int(control_default_raw)
    except (TypeError, ValueError):
        control_default = 3
    control_override_default = control_default_raw is not None

    # -------------------------
    # Price defaults
    # -------------------------
    price_enabled_default = bool(price_defaults.get("enabled", False))
    price_value_raw = price_defaults.get("price_per_ton", price_defaults.get("price", 0.0))
    price_default = _coerce_float(price_value_raw, default=0.0)
    price_schedule_default = _normalize_price_schedule(price_defaults.get("price_schedule"))
    price_escalator_default = _coerce_float(price_defaults.get("price_escalator_pct", 0.0), 0.0)

    allowance_defaults = modules.get("allowance_market", {}) or {}
    cap_schedule_default = _normalize_price_schedule(defaults.get("cap_schedule"))
    if not cap_schedule_default and isinstance(allowance_defaults, Mapping):
        cap_schedule_default = _normalize_price_schedule(allowance_defaults.get("cap"))

    cap_start_default = _coerce_float(defaults.get("cap_start_value"), 0.0)
    if cap_start_default <= 0.0 and cap_schedule_default:
        try:
            first_year = next(iter(cap_schedule_default))
        except StopIteration:
            cap_start_default = 0.0
        else:
            cap_start_default = float(cap_schedule_default[first_year])
    cap_reduction_mode_default = str(defaults.get("cap_reduction_mode", "percent")).strip().lower()
    if cap_reduction_mode_default not in {"percent", "fixed"}:
        cap_reduction_mode_default = "percent"
    cap_reduction_value_default = _coerce_float(defaults.get("cap_reduction_value"), 0.0)

    floor_schedule_default = _normalize_price_schedule(
        defaults.get("floor_schedule")
    )
    if not floor_schedule_default and isinstance(allowance_defaults, Mapping):
        floor_schedule_default = _normalize_price_schedule(
            allowance_defaults.get("floor")
        )

    floor_value_default = price_floor.parse_currency_value(
        defaults.get("floor_value"), 0.0
    )
    floor_mode_default = str(
        defaults.get("floor_escalator_mode", "fixed")
    ).strip().lower()
    if floor_mode_default not in {"fixed", "percent"}:
        floor_mode_default = "fixed"

    raw_floor_escalator_default = defaults.get("floor_escalator_value")
    if floor_mode_default == "percent":
        floor_escalator_default = price_floor.parse_percentage_value(
            raw_floor_escalator_default, 0.0
        )
    else:
        floor_escalator_default = price_floor.parse_currency_value(
            raw_floor_escalator_default, 0.0
        )

    inferred_floor = price_floor.infer_parameters(
        floor_schedule_default,
        default_base=floor_value_default,
        default_type=floor_mode_default,
        default_escalator=floor_escalator_default,
    )
    floor_value_default = inferred_floor.base_value
    floor_mode_default = inferred_floor.escalation_type
    if floor_mode_default == "percent":
        floor_escalator_default = price_floor.parse_percentage_value(
            raw_floor_escalator_default, inferred_floor.escalation_value
        )
    else:
        floor_escalator_default = price_floor.parse_currency_value(
            raw_floor_escalator_default, inferred_floor.escalation_value
        )

    def _extend_years(source: Iterable[Any] | None) -> None:
        if isinstance(source, Iterable) and not isinstance(source, (str, bytes, Mapping)):
            for entry in source:
                try:
                    year_candidates.append(int(entry))
                except (TypeError, ValueError):
                    continue

    year_candidates: list[int] = []
    _extend_years(years)
    if not year_candidates:
        raw_years = run_config.get("years")
        _extend_years(raw_years if isinstance(raw_years, Iterable) else None)
    if not year_candidates:
        start_raw = run_config.get("start_year")
        end_raw = run_config.get("end_year")
        start_val: int | None = None
        end_val: int | None = None
        try:
            start_val = int(start_raw)
        except (TypeError, ValueError):
            start_val = None
        try:
            end_val = int(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end_val = None
        if start_val is not None:
            if end_val is None:
                end_val = start_val
            step = 1 if end_val >= start_val else -1
            year_candidates.extend(range(start_val, end_val + step, step))
        elif end_val is not None:
            year_candidates.append(end_val)

    active_years = sorted(dict.fromkeys(year_candidates))

    cap_start_default_int = int(round(cap_start_default)) if cap_start_default > 0 else 0
    cap_percent_default = (
        float(cap_reduction_value_default) if cap_reduction_mode_default == "percent" else 0.0
    )
    cap_fixed_default = (
        float(cap_reduction_value_default) if cap_reduction_mode_default == "fixed" else 0.0
    )

    locked = bool(lock_inputs)

    # -------------------------
    # Coverage value map
    # -------------------------
    coverage_value_map: dict[str, Any] = {
        _ALL_REGIONS_LABEL: "All",
        "All": "All",
    }
    for label in coverage_default:
        coverage_value_map.setdefault(label, canonical_region_value(label))
    if region_options is not None:
        for entry in region_options:
            label = canonical_region_label(entry)
            coverage_value_map.setdefault(label, canonical_region_value(entry))
    try:
        alias_lookup = region_alias_map()
    except Exception:
        alias_lookup = {}
    for alias, region_id in alias_lookup.items():
        coverage_value_map.setdefault(alias, region_id)
    for region_id in sorted(DEFAULT_REGION_METADATA):
        label = region_display_label(region_id)
        coverage_value_map.setdefault(label, region_id)
        coverage_value_map.setdefault(str(region_id), region_id)

    # -------------------------
    # Coverage / Regions
    # -------------------------
    region_labels: list[str] = []
    if region_options is not None:
        for entry in region_options:
            label = canonical_region_label(entry).strip() or "default"
            if label not in region_labels:
                region_labels.append(label)
    for label in coverage_default:
        if label != _ALL_REGIONS_LABEL and label not in region_labels:
            region_labels.append(label)
    for region_id in sorted(DEFAULT_REGION_METADATA):
        label = region_display_label(region_id)
        if label not in region_labels:
            region_labels.append(label)
    if not region_labels:
        region_labels = ["default"]

    coverage_choices = [_ALL_REGIONS_LABEL] + sorted(region_labels, key=str)
    if coverage_default == ["All"]:
        coverage_default_display = [_ALL_REGIONS_LABEL]
    else:
        coverage_default_display = [
            label for label in coverage_default if label in coverage_choices
        ] or [_ALL_REGIONS_LABEL]

    coverage_regions = list(coverage_default)

    default_ccr1_price_value = float(ccr1_price_default) if ccr1_price_default is not None else 0.0
    default_ccr2_price_value = float(ccr2_price_default) if ccr2_price_default is not None else 0.0
    default_ccr1_escalator_value = float(ccr1_escalator_default)
    default_ccr2_escalator_value = float(ccr2_escalator_default)
    default_control_year_value = int(control_default if control_default > 0 else 3)
    default_price_value = float(price_default if price_default >= 0.0 else 0.0)

    if locked and st is not None:
        st.session_state["carbon_enable"] = bool(enabled_default)
        st.session_state["carbon_price_enable"] = bool(price_enabled_default)
        st.session_state["carbon_floor"] = bool(enable_floor_default)
        st.session_state["carbon_ccr"] = bool(enable_ccr_default)
        st.session_state["carbon_ccr1"] = bool(ccr1_default)
        st.session_state["carbon_ccr2"] = bool(ccr2_default)
        st.session_state["carbon_banking"] = bool(banking_default)
        st.session_state["carbon_bank0"] = float(bank_default)
        st.session_state["carbon_control_toggle"] = bool(control_override_default)
        st.session_state["carbon_control_years"] = default_control_year_value
        st.session_state["carbon_ccr1_price"] = default_ccr1_price_value
        st.session_state["carbon_ccr1_escalator"] = default_ccr1_escalator_value
        st.session_state["carbon_ccr2_price"] = default_ccr2_price_value
        st.session_state["carbon_ccr2_escalator"] = default_ccr2_escalator_value
        st.session_state["carbon_coverage_regions"] = list(coverage_default_display)
        st.session_state["carbon_price_value"] = default_price_value
        st.session_state["carbon_price_escalator"] = float(price_escalator_default)
        st.session_state["carbon_cap_start"] = int(cap_start_default_int)
        st.session_state["carbon_cap_reduction_mode"] = cap_reduction_mode_default
        st.session_state["carbon_cap_reduction_percent"] = float(cap_percent_default)
        st.session_state["carbon_cap_reduction_fixed"] = float(cap_fixed_default)
        st.session_state["carbon_floor_value_input"] = f"{floor_value_default:.2f}"
        st.session_state["carbon_floor_mode"] = floor_mode_default
        st.session_state["carbon_floor_escalator_input"] = f"{floor_escalator_default:.2f}"

    # -------------------------
    # Session defaults and change tracking
    # -------------------------
    if st is not None:
        price_escalator_default = float(
            st.session_state.setdefault("carbon_price_escalator", float(price_escalator_default))
        )
        cap_start_default_int = int(
            st.session_state.setdefault("carbon_cap_start", cap_start_default_int)
        )
        cap_reduction_mode_default = str(
            st.session_state.setdefault("carbon_cap_reduction_mode", cap_reduction_mode_default)
        ).strip().lower()
        if cap_reduction_mode_default not in {"percent", "fixed"}:
            cap_reduction_mode_default = "percent"
            st.session_state["carbon_cap_reduction_mode"] = cap_reduction_mode_default
        cap_percent_default = float(
            st.session_state.setdefault("carbon_cap_reduction_percent", cap_percent_default)
        )
        cap_fixed_default = float(
            st.session_state.setdefault("carbon_cap_reduction_fixed", cap_fixed_default)
        )
        floor_value_text_default = st.session_state.setdefault(
            "carbon_floor_value_input", f"{floor_value_default:.2f}"
        )
        floor_value_default = price_floor.parse_currency_value(
            floor_value_text_default, floor_value_default
        )
        floor_mode_default = str(
            st.session_state.setdefault("carbon_floor_mode", floor_mode_default)
        ).strip().lower()
        if floor_mode_default not in {"fixed", "percent"}:
            floor_mode_default = "fixed"
            st.session_state["carbon_floor_mode"] = floor_mode_default
        floor_escalator_text_default = st.session_state.setdefault(
            "carbon_floor_escalator_input", f"{floor_escalator_default:.2f}"
        )
        if floor_mode_default == "percent":
            floor_escalator_default = price_floor.parse_percentage_value(
                floor_escalator_text_default, floor_escalator_default
            )
        else:
            floor_escalator_default = price_floor.parse_currency_value(
                floor_escalator_text_default, floor_escalator_default
            )

    bank_value_default = bank_default
    if st is not None:  # GUI path
        bank_value_default = float(st.session_state.setdefault("carbon_bank0", bank_default))

    def _mark_last_changed(key: str) -> None:
        if st is None:
            return
        st.session_state["carbon_module_last_changed"] = key

    deep_pricing_allowed = bool(dispatch_defaults.get("deep_carbon_pricing", False))
    if st is not None:
        deep_pricing_allowed = bool(
            st.session_state.get("dispatch_deep_carbon", deep_pricing_allowed)
        )

    session_enabled_default = enabled_default
    session_price_default = price_enabled_default
    last_changed = None
    if st is not None:
        session_enabled_default = bool(
            st.session_state.setdefault("carbon_enable", enabled_default)
        )
        session_price_default = bool(
            st.session_state.setdefault("carbon_price_enable", price_enabled_default)
        )
        last_changed = st.session_state.get("carbon_module_last_changed")
        if session_enabled_default and session_price_default:
            if last_changed == "cap":
                session_price_default = False
            else:
                session_enabled_default = False
            st.session_state["carbon_enable"] = session_enabled_default
            st.session_state["carbon_price_enable"] = session_price_default


    # -------------------------
    # Cap vs Price toggles (mutually exclusive)
    # -------------------------
    cap_toggle_disabled = locked or session_price_default
    enabled = container.toggle(
        "Enable carbon cap",
        value=session_enabled_default,
        key="carbon_enable",
        on_change=lambda: _mark_last_changed("cap"),
        disabled=cap_toggle_disabled,
    )
    price_toggle_disabled = locked or bool(enabled)
    price_enabled = container.toggle(
        "Enable carbon price",
        value=session_price_default,
        key="carbon_price_enable",
        on_change=lambda: _mark_last_changed("price"),
        disabled=price_toggle_disabled,
    )

    if price_enabled and enabled:
        enabled = False
    elif enabled and not price_enabled:
        price_enabled = False

    if locked:
        price_enabled = bool(price_enabled_default)
        enabled = bool(enabled_default and not price_enabled)

    cap_schedule: dict[int, float] = dict(cap_schedule_default)
    cap_start_value = float(cap_start_default_int)
    cap_reduction_mode = cap_reduction_mode_default
    cap_reduction_value = (
        cap_percent_default if cap_reduction_mode_default == "percent" else cap_fixed_default
    )
    price_schedule: dict[int, float] = dict(price_schedule_default)
    price_escalator_value = float(price_escalator_default)

    coverage_panel_enabled = (enabled or price_enabled) and not locked
    with _sidebar_panel(container, coverage_panel_enabled) as coverage_panel:
        coverage_selection = coverage_panel.multiselect(
            "Regions covered by carbon policies",
            options=coverage_choices,
            default=coverage_default_display,
            disabled=(not (enabled or price_enabled)) or locked,
            key="carbon_coverage_regions",
            help=(
                "Select the regions subject to the carbon cap or carbon price. "
                "Choose “All regions” to apply the policy across every region."
            ),
        )
        coverage_regions = _normalize_coverage_selection(
            coverage_selection or coverage_default_display
        )

    # -------------------------
    # Carbon Cap Panel
    # -------------------------
    floor_value = float(floor_value_default)
    floor_mode = str(floor_mode_default)
    floor_escalator_value = float(floor_escalator_default)
    floor_schedule: dict[int, float] = dict(floor_schedule_default)

    with _sidebar_panel(container, enabled and not locked) as cap_panel:
        enable_floor = cap_panel.toggle(
            "Enable price floor",
            value=enable_floor_default,
            key="carbon_floor",
            disabled=(not enabled) or locked,
        )
        if enable_floor:
            floor_value_text = cap_panel.text_input(
                "Price floor ($/ton)",
                value=(st.session_state.get("carbon_floor_value_input") if st is not None else f"{floor_value_default:.2f}"),
                key="carbon_floor_value_input",
                disabled=(not enabled) or locked,
                help="Specify the minimum auction clearing price. Values are rounded to two decimals.",
            )
            floor_value = price_floor.parse_currency_value(floor_value_text, floor_value_default)
            floor_mode = cap_panel.radio(
                "Floor escalates by",
                options=("fixed", "percent"),
                index=0 if floor_mode_default == "fixed" else 1,
                format_func=lambda option: (
                    "Fixed amount ($/ton per year)" if option == "fixed" else "Percent (% per year)"
                ),
                key="carbon_floor_mode",
                disabled=(not enabled) or locked,
            )
            escalator_label = (
                "Annual increase ($/ton)" if floor_mode == "fixed" else "Annual increase (%)"
            )
            escalator_value_default = (
                f"{floor_escalator_default:.2f}"
                if st is None
                else st.session_state.get("carbon_floor_escalator_input", f"{floor_escalator_default:.2f}")
            )
            floor_escalator_text = cap_panel.text_input(
                escalator_label,
                value=escalator_value_default,
                key="carbon_floor_escalator_input",
                disabled=(not enabled) or locked,
                help="Set to 0 for a constant floor across all modeled years.",
            )
            if floor_mode == "percent":
                floor_escalator_value = price_floor.parse_percentage_value(
                    floor_escalator_text, floor_escalator_default
                )
            else:
                floor_escalator_value = price_floor.parse_currency_value(
                    floor_escalator_text, floor_escalator_default
                )
            schedule_years = list(active_years)
            if not schedule_years:
                schedule_years = sorted(floor_schedule_default) if floor_schedule_default else []
            if not schedule_years:
                start_year_raw = run_config.get("start_year")
                try:
                    schedule_years = [int(start_year_raw)] if start_year_raw is not None else []
                except (TypeError, ValueError):
                    schedule_years = []
            floor_schedule = price_floor.build_schedule(
                schedule_years,
                floor_value,
                floor_mode,
                floor_escalator_value,
            )
        else:
            floor_schedule = {}
        enable_ccr = cap_panel.toggle(
            "Enable CCR",
            value=enable_ccr_default,
            key="carbon_ccr",
            disabled=(not enabled) or locked,
        )
        ccr1_enabled = cap_panel.toggle(
            "Enable CCR Tier 1",
            value=ccr1_default,
            key="carbon_ccr1",
            disabled=(not (enabled and enable_ccr)) or locked,
        )
        ccr2_enabled = cap_panel.toggle(
            "Enable CCR Tier 2",
            value=ccr2_default,
            key="carbon_ccr2",
            disabled=(not (enabled and enable_ccr)) or locked,
        )

        cap_start_input = cap_panel.number_input(
            "Starting carbon cap (tons)",
            min_value=0,
            value=int(cap_start_default_int),
            step=1000,
            format="%d",
            key="carbon_cap_start",
            disabled=(not enabled) or locked,
        )
        cap_start_value = float(cap_start_input)

        reduction_options = ("percent", "fixed")
        try:
            reduction_index = reduction_options.index(cap_reduction_mode_default)
        except ValueError:
            reduction_index = 0
        cap_reduction_mode = cap_panel.radio(
            "Annual cap adjustment",
            options=reduction_options,
            index=reduction_index,
            format_func=lambda option: (
                "Decrease by % of starting value" if option == "percent" else "Decrease by fixed amount"
            ),
            key="carbon_cap_reduction_mode",
            disabled=(not enabled) or locked,
        )

        if cap_reduction_mode == "percent":
            cap_reduction_percent = float(
                cap_panel.number_input(
                    "Annual reduction (% of starting cap)",
                    min_value=0.0,
                    value=float(cap_percent_default),
                    step=0.1,
                    format="%0.2f",
                    key="carbon_cap_reduction_percent",
                    disabled=(not enabled) or locked,
                )
            )
            cap_reduction_value = cap_reduction_percent
        else:
            cap_reduction_fixed = float(
                cap_panel.number_input(
                    "Annual reduction (tons)",
                    min_value=0.0,
                    value=float(cap_fixed_default),
                    step=1000.0,
                    format="%0.0f",
                    key="carbon_cap_reduction_fixed",
                    disabled=(not enabled) or locked,
                )
            )
            cap_reduction_value = cap_reduction_fixed

        schedule_years = active_years or list(cap_schedule_default.keys())
        if enabled and schedule_years:
            cap_schedule = _build_cap_reduction_schedule(
                cap_start_value,
                cap_reduction_mode,
                cap_reduction_value,
                schedule_years,
            )
        elif enabled:
            cap_schedule = dict(cap_schedule_default)
        else:
            cap_schedule = dict(cap_schedule_default)
        floor_value = float(floor_value_default)
        floor_mode = str(floor_mode_default)
        floor_escalator_value = float(floor_escalator_default)
        floor_schedule = dict(floor_schedule_default) if enable_floor else {}

        if enabled and enable_ccr and ccr1_enabled:
            default_price1 = float(ccr1_price_default) if ccr1_price_default is not None else 0.0
            ccr1_price_value = float(
                cap_panel.number_input(
                    "CCR Tier 1 trigger price ($/ton)",
                    min_value=0.0,
                    value=default_price1,
                    step=1.0,
                    format="%0.2f",
                    key="carbon_ccr1_price",
                    disabled=(not (enabled and enable_ccr and ccr1_enabled)) or locked,
                )
            )
            ccr1_escalator_value = float(
                cap_panel.number_input(
                    "CCR Tier 1 annual escalator (%)",
                    min_value=0.0,
                    value=float(ccr1_escalator_default),
                    step=0.1,
                    format="%0.2f",
                    key="carbon_ccr1_escalator",
                    disabled=(not (enabled and enable_ccr and ccr1_enabled)) or locked,
                )
            )
        else:
            ccr1_price_value = ccr1_price_default if ccr1_price_default is not None else None
            ccr1_escalator_value = float(ccr1_escalator_default)

        if enabled and enable_ccr and ccr2_enabled:
            default_price2 = float(ccr2_price_default) if ccr2_price_default is not None else 0.0
            ccr2_price_value = float(
                cap_panel.number_input(
                    "CCR Tier 2 trigger price ($/ton)",
                    min_value=0.0,
                    value=default_price2,
                    step=1.0,
                    format="%0.2f",
                    key="carbon_ccr2_price",
                    disabled=(not (enabled and enable_ccr and ccr2_enabled)) or locked,
                )
            )
            ccr2_escalator_value = float(
                cap_panel.number_input(
                    "CCR Tier 2 annual escalator (%)",
                    min_value=0.0,
                    value=float(ccr2_escalator_default),
                    step=0.1,
                    format="%0.2f",
                    key="carbon_ccr2_escalator",
                    disabled=(not (enabled and enable_ccr and ccr2_enabled)) or locked,
                )
            )
        else:
            ccr2_price_value = ccr2_price_default if ccr2_price_default is not None else None
            ccr2_escalator_value = float(ccr2_escalator_default)

        banking_enabled = cap_panel.toggle(
            "Enable allowance banking",
            value=banking_default,
            key="carbon_banking",
            disabled=(not enabled) or locked,
        )

        if banking_enabled:
            initial_bank = float(
                cap_panel.number_input(
                    "Initial allowance bank (tons)",
                    min_value=0.0,
                    value=float(bank_value_default if bank_value_default >= 0.0 else 0.0),
                    step=1000.0,
                    format="%f",
                    key="carbon_bank0",
                    disabled=(not (enabled and banking_enabled)) or locked,
                )
            )
        else:
            initial_bank = 0.0

        control_override = cap_panel.toggle(
            "Override control period",
            value=control_override_default,
            key="carbon_control_toggle",
            disabled=(not enabled) or locked,
        )
        control_period_value = cap_panel.number_input(
            "Control period length (years)",
            min_value=1,
            value=int(control_default if control_default > 0 else 3),
            step=1,
            format="%d",
            key="carbon_control_years",
            disabled=(not (enabled and control_override)) or locked,
        )
        control_period_years = (
            _sanitize_control_period(control_period_value)
            if enabled and control_override
            else None
        )

    # -------------------------
    # Carbon Price Panel
    # -------------------------
    with _sidebar_panel(container, price_enabled and not locked) as price_panel:
        price_per_ton = price_panel.number_input(
            "Carbon price ($/ton)",
            min_value=0.0,
            value=float(price_default if price_default >= 0.0 else 0.0),
            step=1.0,
            format="%0.2f",
            key="carbon_price_value",
            disabled=(not price_enabled) or locked,
        )
        price_escalator_value = float(
            price_panel.number_input(
                "Carbon price escalator (% per year)",
                min_value=0.0,
                value=float(price_escalator_default if price_escalator_default >= 0.0 else 0.0),
                step=0.1,
                format="%0.2f",
                key="carbon_price_escalator",
                disabled=(not price_enabled) or locked,
            )
        )

        schedule_years = active_years or list(price_schedule_default.keys())
        if price_enabled and schedule_years:
            price_schedule = _build_price_escalator_schedule(
                price_per_ton,
                price_escalator_value,
                schedule_years,
            )
        elif price_enabled:
            price_schedule = price_schedule_default.copy()
        else:
            price_schedule = {}

    if locked:
        enabled = bool(enabled_default)
        price_enabled = bool(price_enabled_default)
        enable_floor = bool(enable_floor_default)
        enable_ccr = bool(enable_ccr_default)
        ccr1_enabled = bool(ccr1_default)
        ccr2_enabled = bool(ccr2_default)
        banking_enabled = bool(banking_default)
        if banking_enabled:
            initial_bank = float(bank_value_default if bank_value_default >= 0.0 else 0.0)
        else:
            initial_bank = 0.0
        control_override = bool(control_override_default)
        if enabled and control_override:
            control_period_years = _sanitize_control_period(default_control_year_value)
        else:
            control_period_years = None
        coverage_regions = list(coverage_default)
        price_per_ton = default_price_value
        price_escalator_value = float(price_escalator_default)
        price_schedule = price_schedule_default.copy() if price_enabled else {}
        cap_start_value = float(cap_start_default_int)
        cap_reduction_mode = cap_reduction_mode_default
        cap_reduction_value = (
            cap_percent_default if cap_reduction_mode_default == "percent" else cap_fixed_default
        )
        cap_schedule = dict(cap_schedule_default)

    # -------------------------
    # Errors and Return
    # -------------------------
    if not enabled:
        cap_schedule = {}
        floor_schedule = {}

    errors: list[str] = []
    deep_enabled = bool(
        modules.get("electricity_dispatch", {}).get("deep_carbon_pricing", False)
    )
    if st is not None:
        deep_enabled = bool(st.session_state.get("dispatch_deep_carbon", deep_enabled))
    if enabled and price_enabled and not deep_enabled:
        errors.append("Cannot enable both carbon cap and carbon price simultaneously.")

    policy_region_values: list[Any] = []
    if coverage_regions != ["All"]:
        for label in coverage_regions:
            resolved = coverage_value_map.get(label, canonical_region_value(label))
            if isinstance(resolved, str) and resolved.lower() in {"all", "all regions"}:
                policy_region_values = []
                break
            try:
                policy_region_values.append(int(resolved))
            except (TypeError, ValueError):
                policy_region_values.append(resolved)

    carbon_module = modules.setdefault("carbon_policy", {})
    carbon_module.update(
        {
            "enabled": bool(enabled),
            "enable_floor": bool(enabled and enable_floor),
            "enable_ccr": bool(enabled and enable_ccr),
            "ccr1_enabled": bool(enabled and enable_ccr and ccr1_enabled),
            "ccr2_enabled": bool(enabled and enable_ccr and ccr2_enabled),
            "allowance_banking_enabled": bool(enabled and banking_enabled),
            "coverage_regions": list(coverage_regions),
            "cap_start_value": float(cap_start_value),
            "cap_reduction_mode": str(cap_reduction_mode),
            "cap_reduction_value": float(cap_reduction_value),
            "floor_value": float(floor_value),
            "floor_escalator_mode": str(floor_mode),
            "floor_escalator_value": float(floor_escalator_value),
        }
    )

    if control_period_years is None or not enabled:
        carbon_module["control_period_years"] = None
    else:
        carbon_module["control_period_years"] = int(control_period_years)

    if enabled and banking_enabled:
        carbon_module["bank0"] = float(initial_bank)
    else:
        carbon_module["bank0"] = 0.0

    if enabled and cap_schedule:
        carbon_module["cap_schedule"] = dict(cap_schedule)
    else:
        carbon_module.pop("cap_schedule", None)

    if enabled and enable_floor and floor_schedule:
        carbon_module["floor_schedule"] = dict(floor_schedule)
    else:
        carbon_module.pop("floor_schedule", None)
    if policy_region_values:
        carbon_module["regions"] = list(policy_region_values)
    else:
        carbon_module.pop("regions", None)

    allowance_market_module = modules.setdefault("allowance_market", {})
    if enabled and cap_schedule:
        allowance_market_module["cap"] = {
            int(year): float(value) for year, value in cap_schedule.items()
        }
    elif not enabled:
        allowance_market_module.pop("cap", None)

    if enabled and enable_floor and floor_schedule:
        allowance_market_module["floor"] = {
            int(year): float(value) for year, value in floor_schedule.items()
        }
    elif not enabled or not enable_floor:
        allowance_market_module.pop("floor", None)
    price_module = modules.setdefault("carbon_price", {})
    price_module["enabled"] = bool(price_enabled)
    if price_enabled:
        price_module["price_per_ton"] = float(price_per_ton)
        price_module["price_escalator_pct"] = float(price_escalator_value)
        if price_schedule:
            price_module["price_schedule"] = dict(price_schedule)
        else:
            price_module.pop("price_schedule", None)
        price_module["coverage_regions"] = list(coverage_regions)
        if policy_region_values:
            price_module["regions"] = list(policy_region_values)
        else:
            price_module.pop("regions", None)
    else:
        price_module["price_escalator_pct"] = float(price_escalator_value)
        price_module.pop("price_schedule", None)
        price_module.pop("price", None)
        if "price_per_ton" in price_module:
            price_module["price_per_ton"] = float(price_per_ton)
        price_module.pop("coverage_regions", None)
        price_module.pop("regions", None)

    return CarbonModuleSettings(
        enabled=enabled,
        price_enabled=price_enabled,
        enable_floor=enable_floor,
        enable_ccr=enable_ccr,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        ccr1_price=ccr1_price_value if 'ccr1_price_value' in locals() else ccr1_price_default,
        ccr2_price=ccr2_price_value if 'ccr2_price_value' in locals() else ccr2_price_default,
        ccr1_escalator_pct=ccr1_escalator_value if 'ccr1_escalator_value' in locals() else float(ccr1_escalator_default),
        ccr2_escalator_pct=ccr2_escalator_value if 'ccr2_escalator_value' in locals() else float(ccr2_escalator_default),
        banking_enabled=banking_enabled,
        coverage_regions=coverage_regions,
        control_period_years=control_period_years,
        price_per_ton=float(price_per_ton),
        price_escalator_pct=float(price_escalator_value),
        initial_bank=initial_bank,
        cap_regions=policy_region_values,
        cap_start_value=float(cap_start_value),
        cap_reduction_mode=str(cap_reduction_mode),
        cap_reduction_value=float(cap_reduction_value),
        cap_schedule=cap_schedule,
        floor_value=float(floor_value),
        floor_escalator_mode=str(floor_mode),
        floor_escalator_value=float(floor_escalator_value),
        floor_schedule=floor_schedule,
        price_schedule=price_schedule,
        errors=errors,
    )


# -------------------------
# Dispatch UI
# -------------------------
def _render_dispatch_section(
    container: Any,
    run_config: dict[str, Any],
    frames: FramesType | None,
) -> DispatchModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("electricity_dispatch", {})
    enabled_default = bool(defaults.get("enabled", False))
    mode_default = str(defaults.get("mode", "single")).lower()
    if mode_default not in {"single", "network"}:
        mode_default = "single"
    capacity_default = bool(defaults.get("capacity_expansion", True))
    reserve_default = bool(defaults.get("reserve_margins", True))
    deep_default = bool(defaults.get("deep_carbon_pricing", False))

    enabled = container.toggle(
        "Enable electricity dispatch",
        value=enabled_default,
        key="dispatch_enable",
    )

    mode_value = mode_default
    capacity_expansion = capacity_default
    reserve_margins = reserve_default
    deep_carbon_pricing = deep_default
    errors: list[str] = []

    mode_options = {"single": "Single region", "network": "Networked"}

    with _sidebar_panel(container, enabled) as panel:
        mode_label = mode_options.get(mode_default, mode_options["single"])
        mode_selection = panel.selectbox(
            "Dispatch topology",
            options=list(mode_options.values()),
            index=list(mode_options.values()).index(mode_label),
            disabled=not enabled,
            key="dispatch_mode",
        )
        mode_value = "network" if mode_selection == mode_options["network"] else "single"
        capacity_expansion = panel.checkbox(
            "Enable capacity expansion",
            value=capacity_default,
            disabled=not enabled,
            key="dispatch_capacity",
        )
        reserve_margins = panel.checkbox(
            "Enforce reserve margins",
            value=reserve_default,
            disabled=not enabled,
            key="dispatch_reserve",
        )
        default_deep_value = deep_default
        if st is not None:
            default_deep_value = bool(
                st.session_state.get("dispatch_deep_carbon", deep_default)
            )
        deep_carbon_pricing = panel.toggle(
            "Enable deep carbon pricing",
            value=default_deep_value,
            disabled=not enabled,
            key="dispatch_deep_carbon",
            help=(
                "Allows simultaneous use of allowance clearing prices and exogenous "
                "carbon prices when solving dispatch."
            ),
        )

        if enabled:
            if frames is None:
                message = "Dispatch requires demand and unit data, but no frames are available."
                panel.error(message)
                errors.append(message)
            else:
                try:
                    demand_df = frames.demand()
                    units_df = frames.units()
                except Exception as exc:
                    message = f"Dispatch data unavailable: {exc}"
                    panel.error(message)
                    errors.append(message)
                else:
                    if demand_df.empty or units_df.empty:
                        message = "Dispatch requires non-empty demand and unit tables."
                        panel.error(message)
                        errors.append(message)
        else:
            mode_value = mode_default
            capacity_expansion = False
            reserve_margins = False
            deep_carbon_pricing = False

    if not enabled:
        mode_value = mode_value or "single"
        deep_carbon_pricing = False

    modules["electricity_dispatch"] = {
        "enabled": bool(enabled),
        "mode": mode_value or "single",
        "capacity_expansion": bool(capacity_expansion),
        "reserve_margins": bool(reserve_margins),
        "deep_carbon_pricing": bool(deep_carbon_pricing),
    }

    return DispatchModuleSettings(
        enabled=bool(enabled),
        mode=mode_value or "single",
        capacity_expansion=bool(capacity_expansion),
        reserve_margins=bool(reserve_margins),
        deep_carbon_pricing=bool(deep_carbon_pricing),
        errors=errors,
    )


# -------------------------
# Incentives UI
# -------------------------
def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
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


def _parse_years_field(
    value: Any,
    *,
    valid_years: set[int] | None = None,
) -> tuple[list[int], list[str], list[int]]:
    if value in (None, ""):
        return [], [], []

    text = str(value).strip()
    if not text:
        return [], [], []

    normalized = text.translate({ord(char): None for char in "[]{}()"})
    tokens = [token for token in re.split(r"[;,\s]+", normalized) if token]

    parsed_years: list[int] = []
    invalid_tokens: list[str] = []
    out_of_range: list[int] = []

    valid_set = {int(year) for year in valid_years} if valid_years else set()

    for token in tokens:
        token_str = token.strip()
        if not token_str:
            continue

        if "-" in token_str:
            start_text, end_text = token_str.split("-", 1)
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
    if value is None:
        return []

    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")  # type: ignore[call-arg]
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


def _simulation_years_from_config(config: Mapping[str, Any]) -> list[int]:
    try:
        base_years = _years_from_config(config)
    except Exception:
        base_years = []

    start_raw = config.get("start_year")
    end_raw = config.get("end_year")

    def _to_int(value: Any) -> int | None:
        if value in (None, ""):
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


def _render_incentives_section(
    container: Any,
    run_config: dict[str, Any],
    frames: FramesType | None,
) -> IncentivesModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("incentives", {})
    enabled_default = bool(defaults.get("enabled", False))

    incentives_cfg = run_config.get("electricity_incentives")
    production_source: Any | None = None
    investment_source: Any | None = None
    if isinstance(incentives_cfg, Mapping):
        enabled_default = bool(incentives_cfg.get("enabled", enabled_default))
        production_source = incentives_cfg.get("production")
        investment_source = incentives_cfg.get("investment")
    if production_source is None and isinstance(defaults, Mapping):
        production_source = defaults.get("production")
    if investment_source is None and isinstance(defaults, Mapping):
        investment_source = defaults.get("investment")

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
            tech_id = resolve_technology_key(entry.get("technology"))
            if tech_id is None:
                continue
            try:
                year_int = int(entry.get("year"))
            except (TypeError, ValueError):
                continue
            credit_val = _coerce_optional_float(entry.get(credit_key))
            if credit_val is None:
                continue
            limit_val = _coerce_optional_float(entry.get(limit_key))
            record: dict[str, Any] = {
                "technology": get_technology_label(tech_id),
                "year": year_int,
                credit_key: float(credit_val),
            }
            if limit_val is not None:
                record[limit_key] = float(limit_val)
            entries.append(record)
        entries.sort(key=lambda item: (str(item["technology"]).lower(), int(item["year"])))
        return entries

    existing_production_entries = _normalise_config_entries(
        production_source, credit_key="credit_per_mwh", limit_key="limit_mwh"
    )
    existing_investment_entries = _normalise_config_entries(
        investment_source, credit_key="credit_per_mw", limit_key="limit_mw"
    )

    technology_options: set[str] = {
        get_technology_label(tech_id) for tech_id in sorted(TECH_ID_TO_LABEL or {})
    }
    for entry in (*existing_production_entries, *existing_investment_entries):
        label = str(entry.get("technology", "")).strip()
        if label:
            technology_options.add(label)
    if not technology_options:
        technology_options = {"Coal", "Gas", "Wind", "Solar"}
    technology_labels = sorted(technology_options)

    production_credit_col = "Credit ($/MWh)"
    production_limit_col = "Limit (MWh)"
    investment_credit_col = "Credit ($/MW)"
    investment_limit_col = "Limit (MW)"
    selection_column = "Apply Credit"

    def _build_editor_rows(
        entries: list[dict[str, Any]],
        *,
        credit_key: str,
        limit_key: str,
        credit_label: str,
        limit_label: str,
        selection_label: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            rows.append(
                {
                    selection_label: True,
                    "Technology": entry["technology"],
                    "Years": str(entry["year"]),
                    credit_label: entry.get(credit_key),
                    limit_label: entry.get(limit_key),
                }
            )
        seen = {str(row.get("Technology")) for row in rows if row.get("Technology")}
        for label in technology_labels:
            if label not in seen:
                rows.append(
                    {
                        selection_label: False,
                        "Technology": label,
                        "Years": "",
                        credit_label: None,
                        limit_label: None,
                    }
                )
        rows.sort(
            key=lambda row: (
                str(row.get("Technology", "")).lower(),
                str(row.get("Years", "")).lower(),
            )
        )
        return rows

    production_rows_default: list[dict[str, Any]] = _build_editor_rows(
        existing_production_entries,
        credit_key="credit_per_mwh",
        limit_key="limit_mwh",
        credit_label=production_credit_col,
        limit_label=production_limit_col,
        selection_label=selection_column,
    )
    investment_rows_default: list[dict[str, Any]] = _build_editor_rows(
        existing_investment_entries,
        credit_key="credit_per_mw",
        limit_key="limit_mw",
        credit_label=investment_credit_col,
        limit_label=investment_limit_col,
        selection_label=selection_column,
    )

    production_column_order: list[str] = [
        selection_column,
        "Technology",
        "Years",
        production_credit_col,
        production_limit_col,
    ]
    investment_column_order: list[str] = [
        selection_column,
        "Technology",
        "Years",
        investment_credit_col,
        investment_limit_col,
    ]

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
        selection_column: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        results: dict[tuple[int, int], dict[str, Any]] = {}
        messages: list[str] = []
        for index, row in enumerate(rows, start=1):
            if selection_column and selection_column in row:
                include_row = _coerce_bool_flag(row.get(selection_column), default=False)
                if not include_row:
                    continue
            technology_value = row.get("Technology")
            technology_label = (
                str(technology_value).strip() if technology_value not in (None, "") else ""
            )
            if not technology_label:
                continue
            tech_id = resolve_technology_key(technology_label)
            if tech_id is None:
                messages.append(f'{context_label} row {index}: Unknown technology "{technology_label}".')
                continue
            years_value = row.get("Years")
            years, invalid_tokens, out_of_range = _parse_years_field(
                years_value, valid_years=valid_years
            )
            if invalid_tokens:
                tokens_display = ", ".join(
                    sorted({token.strip() for token in invalid_tokens if token.strip()})
                )
                if tokens_display:
                    messages.append(
                        f"{context_label} row {index}: Unable to parse year value(s): {tokens_display}."
                    )
            if out_of_range:
                years_display = ", ".join(str(year) for year in out_of_range)
                messages.append(
                    f"{context_label} row {index}: Year(s) {years_display} fall outside the selected simulation years."
                )
            if not years:
                years_text = str(years_value).strip() if isinstance(years_value, str) else ""
                credit_candidate = _coerce_optional_float(row.get(credit_column))
                if years_text or credit_candidate is not None:
                    messages.append(f"{context_label} row {index}: Specify one or more valid years.")
                continue
            credit_value = _coerce_optional_float(row.get(credit_column))
            if credit_value is None:
                messages.append(f"{context_label} row {index}: Provide a credit value.")
                continue
            limit_value = _coerce_optional_float(row.get(limit_column))
            label = get_technology_label(tech_id)
            for year in years:
                entry = {
                    "technology": label,
                    "year": int(year),
                    credit_config_key: float(credit_value),
                }
                if limit_value is not None:
                    entry[limit_config_key] = float(limit_value)
                results[(tech_id, int(year))] = entry
        ordered = sorted(
            results.values(),
            key=lambda item: (str(item["technology"]).lower(), int(item["year"])),
        )
        return ordered, messages

    enabled = container.toggle(
        "Enable incentives and credits",
        value=enabled_default,
        key="incentives_enable",
    )

    errors: list[str] = []
    production_entries = existing_production_entries
    investment_entries = existing_investment_entries

    with _sidebar_panel(container, enabled) as panel:
        panel.caption(
            "Specify technology-specific tax credits that feed the electricity capacity and generation modules."
        )
        if available_years:
            years_display = ", ".join(str(year) for year in available_years)
            panel.caption(f"Simulation years: {years_display}")
        panel.caption(
            "Enter comma-separated years or ranges (e.g., 2025, 2030-2032). "
            "Leave blank to exclude a technology."
        )

        panel.markdown("**Production tax credits ($/MWh)**")
        production_editor_value = panel.data_editor(
            production_rows_default,
            disabled=not enabled,
            hide_index=True,
            num_rows="dynamic",
            width="stretch",  # Streamlit >= 1.38: replaces use_container_width
            key="incentives_production_editor",
            column_order=production_column_order,
            column_config={
                selection_column: st.column_config.CheckboxColumn(
                    "Apply Credit",
                    help=(
                        "Select to apply production tax credits for this technology. "
                        "Unchecked technologies default to $0 incentives across all years."
                    ),
                    default=False,
                ),
                "Technology": st.column_config.SelectboxColumn(
                    "Technology", options=technology_labels
                ),
                "Years": st.column_config.TextColumn(
                    "Applicable years",
                    help="Comma-separated years or ranges (e.g., 2025, 2030-2032).",
                ),
                production_credit_col: st.column_config.NumberColumn(
                    production_credit_col,
                    format="$%.2f",
                    min_value=0.0,
                    help="Credit value applied per megawatt-hour.",
                ),
                production_limit_col: st.column_config.NumberColumn(
                    production_limit_col,
                    min_value=0.0,
                    help="Optional annual limit on eligible production (MWh).",
                ),
            },
        )

        panel.markdown("**Investment tax credits ($/MW)**")
        investment_editor_value = panel.data_editor(
            investment_rows_default,
            disabled=not enabled,
            hide_index=True,
            num_rows="dynamic",
            width="stretch",  # Streamlit >= 1.38: replaces use_container_width
            key="incentives_investment_editor",
            column_order=investment_column_order,
            column_config={
                selection_column: st.column_config.CheckboxColumn(
                    "Apply Credit",
                    help=(
                        "Select to apply investment tax credits for this technology. "
                        "Unchecked technologies default to $0 incentives across all years."
                    ),
                    default=False,
                ),
                "Technology": st.column_config.SelectboxColumn(
                    "Technology", options=technology_labels
                ),
                "Years": st.column_config.TextColumn(
                    "Applicable years",
                    help="Comma-separated years or ranges (e.g., 2025, 2030-2032).",
                ),
                investment_credit_col: st.column_config.NumberColumn(
                    investment_credit_col,
                    format="$%.2f",
                    min_value=0.0,
                    help="Credit value applied per megawatt of installed capacity.",
                ),
                investment_limit_col: st.column_config.NumberColumn(
                    investment_limit_col,
                    min_value=0.0,
                    help="Optional annual limit on eligible capacity additions (MW).",
                ),
            },
        )


        validation_messages: list[str] = []
        if enabled:
            production_entries, production_messages = _rows_to_config_entries(
                _data_editor_records(production_editor_value),
                credit_column=production_credit_col,
                limit_column=production_limit_col,
                credit_config_key="credit_per_mwh",
                limit_config_key="limit_mwh",
                context_label="Production tax credit",
                valid_years=valid_years_set,
                selection_column=selection_column,
            )
            investment_entries, investment_messages = _rows_to_config_entries(
                _data_editor_records(investment_editor_value),
                credit_column=investment_credit_col,
                limit_column=investment_limit_col,
                credit_config_key="credit_per_mw",
                limit_config_key="limit_mw",
                context_label="Investment tax credit",
                valid_years=valid_years_set,
                selection_column=selection_column,
            )
            validation_messages.extend(production_messages)
            validation_messages.extend(investment_messages)

        for message in validation_messages:
            panel.error(message)
        errors.extend(validation_messages)

        if enabled:
            if frames is None:
                message = "Incentives require generating unit data."
                panel.error(message)
                errors.append(message)
            else:
                try:
                    units_df = frames.units()
                except Exception as exc:
                    message = f"Unable to access unit data: {exc}"
                    panel.error(message)
                    errors.append(message)
                else:
                    if units_df.empty:
                        message = "Incentives require at least one generating unit."
                        panel.error(message)
                        errors.append(message)

    incentives_record: dict[str, Any] = {"enabled": bool(enabled)}
    if production_entries:
        incentives_record["production"] = copy.deepcopy(production_entries)
    if investment_entries:
        incentives_record["investment"] = copy.deepcopy(investment_entries)

    run_config["electricity_incentives"] = copy.deepcopy(incentives_record)
    modules["incentives"] = copy.deepcopy(incentives_record)

    return IncentivesModuleSettings(
        enabled=bool(enabled),
        production_credits=copy.deepcopy(production_entries),
        investment_credits=copy.deepcopy(investment_entries),
        errors=errors,
    )


# -------------------------
# Outputs UI
# -------------------------
def _render_outputs_section(
    container: Any,
    run_config: dict[str, Any],
    last_result: Mapping[str, Any] | None,
) -> OutputsModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("outputs", {})
    enabled_default = bool(defaults.get("enabled", True))
    directory_default = str(defaults.get("directory") or run_config.get("output_name") or "outputs")
    show_csv_default = bool(defaults.get("show_csv_downloads", True))

    downloads_root = get_downloads_directory()

    enabled = container.toggle(
        "Enable output management",
        value=enabled_default,
        key="outputs_enable",
    )

    directory_value = directory_default
    show_csv_downloads = show_csv_default
    errors: list[str] = []

    with _sidebar_panel(container, enabled) as panel:
        directory_value = panel.text_input(
            "Output directory name",
            value=directory_default,
            disabled=not enabled,
            key="outputs_directory",
        ).strip()
        show_csv_downloads = panel.checkbox(
            "Show CSV downloads from last run",
            value=show_csv_default,
            disabled=not enabled,
            key="outputs_csv",
        )

        resolved_directory = downloads_root if not directory_value else downloads_root / directory_value
        panel.caption(f"Outputs will be saved to {resolved_directory}")

        if enabled and not directory_value:
            message = "Specify an output directory when the outputs module is enabled."
            panel.error(message)
            errors.append(message)

        csv_files: Mapping[str, Any] | None = None
        if enabled and show_csv_downloads:
            if isinstance(last_result, Mapping):
                csv_files = last_result.get("csv_files")  # type: ignore[assignment]
            if csv_files:
                panel.caption("Download CSV outputs from the most recent run.")
                for filename, content in sorted(csv_files.items()):
                    panel.download_button(
                        label=f"Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/csv",
                        key=f"outputs_download_{filename}",
                    )
            else:
                panel.info("No CSV outputs are available yet.")
        elif enabled:
            panel.caption("CSV downloads will be available after the next run.")

    if not directory_value:
        directory_value = directory_default or "outputs"
    if not enabled:
        show_csv_downloads = False

    run_config["output_name"] = directory_value
    resolved_directory = downloads_root if not directory_value else downloads_root / directory_value
    modules["outputs"] = {
        "enabled": bool(enabled),
        "directory": directory_value,
        "show_csv_downloads": bool(show_csv_downloads),
        "resolved_path": str(resolved_directory),  # config serialization
    }

    return OutputsModuleSettings(
        enabled=bool(enabled),
        directory=directory_value,
        resolved_path=resolved_directory,  # keep as Path in memory
        show_csv_downloads=bool(show_csv_downloads),
        errors=errors,
    )


# -------------------------
# Frames + runner helpers
# -------------------------
def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            return float(value.strip())
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_str(value: Any, default: str = "default") -> str:
    if value in (None, ""):
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
    elif value not in (None, ""):
        try:
            years.add(int(value))
        except (TypeError, ValueError):
            pass
    if not years:
        years = {int(year) for year in fallback}
    return years


def _coerce_year_value_map(
    entry: Any,
    years: Iterable[int],
    *,
    cast: Callable[[Any], _T],
    default: _T,
) -> dict[int, _T]:
    values: dict[int, _T] = {}

    if isinstance(entry, Mapping):
        iterator = entry.items()
    elif isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
        iterator = []
        for item in entry:
            if isinstance(item, Mapping) and "year" in item:
                iterator.append((item.get("year"), item.get("value", item.get("amount"))))
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
    years_list = sorted(int(year) for year in years)
    if not years_list:
        raise ValueError("No years supplied for policy frame")

    market_cfg = config.get("allowance_market")
    if not isinstance(market_cfg, Mapping):
        market_cfg = {}

    bank_flag = bool(carbon_policy_enabled and banking_enabled)

    resolution_raw = market_cfg.get("resolution", "annual")
    if isinstance(resolution_raw, str):
        resolution = resolution_raw.strip().lower() or "annual"
    else:
        resolution = str(resolution_raw).strip().lower() or "annual"
    if resolution not in {"annual", "daily"}:
        resolution = "annual"

    if carbon_policy_enabled:
        ccr1_flag = _coerce_bool_flag(market_cfg.get("ccr1_enabled"), default=True)
        ccr2_flag = _coerce_bool_flag(market_cfg.get("ccr2_enabled"), default=True)
        if ccr1_enabled is not None:
            ccr1_flag = bool(ccr1_enabled)
        if ccr2_enabled is not None:
            ccr2_flag = bool(ccr2_enabled)

        control_period = control_period_years
        if control_period is None:
            raw_control = market_cfg.get("control_period_years")
            if raw_control not in (None, ""):
                try:
                    control_period = int(raw_control)
                except (TypeError, ValueError):
                    control_period = None
        if control_period is not None and control_period <= 0:
            control_period = None

        cap_map = _coerce_year_value_map(market_cfg.get("cap"), years_list, cast=float, default=0.0)
        floor_map = _coerce_year_value_map(market_cfg.get("floor"), years_list, cast=float, default=0.0)
        ccr1_trigger_map = _coerce_year_value_map(
            market_cfg.get("ccr1_trigger"), years_list, cast=float, default=0.0
        )
        ccr1_qty_map = _coerce_year_value_map(
            market_cfg.get("ccr1_qty"), years_list, cast=float, default=0.0
        )
        ccr2_trigger_map = _coerce_year_value_map(
            market_cfg.get("ccr2_trigger"), years_list, cast=float, default=0.0
        )
        ccr2_qty_map = _coerce_year_value_map(
            market_cfg.get("ccr2_qty"), years_list, cast=float, default=0.0
        )
        cp_id_map = _coerce_year_value_map(
            market_cfg.get("cp_id"), years_list, cast=lambda v: _coerce_str(v, "CP1"), default="CP1"
        )
        bank0 = _coerce_float(market_cfg.get("bank0"), default=0.0)
        surrender_frac = _coerce_float(market_cfg.get("annual_surrender_frac"), default=1.0)
        carry_pct = _coerce_float(market_cfg.get("carry_pct"), default=1.0)
        if not bank_flag:
            bank0 = 0.0
            carry_pct = 0.0
        full_compliance_years = _coerce_year_set(
            market_cfg.get("full_compliance_years"), fallback=[]
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
        cp_id_map = {year: "NoPolicy" for year in years_list}
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
                "year": year,
                "cap_tons": float(cap_map[year]),
                "floor_dollars": float(floor_map[year]),
                "ccr1_trigger": float(ccr1_trigger_map[year]),
                "ccr1_qty": float(ccr1_qty_map[year]),
                "ccr2_trigger": float(ccr2_trigger_map[year]),
                "ccr2_qty": float(ccr2_qty_map[year]),
                "cp_id": str(cp_id_map[year]),
                "full_compliance": year in full_compliance_years,
                "bank0": float(bank0),
                "annual_surrender_frac": float(surrender_frac),
                "carry_pct": float(carry_pct),
                "policy_enabled": bool(carbon_policy_enabled),
                "ccr1_enabled": bool(ccr1_flag),
                "ccr2_enabled": bool(ccr2_flag),
                "control_period_years": control_period,
                "bank_enabled": bool(bank_flag),
                "resolution": "annual" if resolution not in {"annual", "daily"} else resolution,
            }
        )

    return pd.DataFrame(records)


def _available_regions_from_frames(frames: FramesType) -> list[str]:
    """Return an ordered list of region labels present in ``frames``."""

    regions: list[str] = []

    try:
        demand = frames.demand()
        if not demand.empty and 'region' in demand.columns:
            for value in demand['region']:
                label = str(value)
                if label not in regions:
                    regions.append(label)
    except Exception:  # pragma: no cover - defensive guard
        pass

    try:
        units = frames.units()
        if not units.empty and 'region' in units.columns:
            for value in units['region']:
                label = str(value)
                if label not in regions:
                    regions.append(label)
    except Exception:  # pragma: no cover - defensive guard
        pass

    if not regions:
        regions = ['default']

    return regions


def _build_coverage_frame(
    frames: FramesType,
    coverage_regions: Iterable[str] | None,
) -> pd.DataFrame | None:
    """Construct a coverage table aligning regions with enabled status."""

    if coverage_regions is None:
        return None

    normalized = _normalize_coverage_selection(coverage_regions)
    cover_all = normalized == ["All"]

    regions = _available_regions_from_frames(frames)
    ordered = list(dict.fromkeys(regions))
    for label in normalized:
        if label != "All" and label not in ordered:
            ordered.append(label)

    records = [
        {
            'region': region,
            'covered': True if cover_all else region in normalized,
        }
        for region in ordered
    ]

    return pd.DataFrame(records)


def _default_units() -> pd.DataFrame:
    data = [
        {
            "unit_id": "wind-1",
            "fuel": "wind",
            "region": "default",
            "cap_mw": 50.0,
            "availability": 0.5,
            "hr_mmbtu_per_mwh": 0.0,
            "vom_per_mwh": 0.0,
            "fuel_price_per_mmbtu": 0.0,
            "ef_ton_per_mwh": 0.0,
        },
        {
            "unit_id": "coal-1",
            "fuel": "coal",
            "region": "default",
            "cap_mw": 80.0,
            "availability": 0.9,
            "hr_mmbtu_per_mwh": 9.0,
            "vom_per_mwh": 1.5,
            "fuel_price_per_mmbtu": 1.8,
            "ef_ton_per_mwh": 1.0,
        },
        {
            "unit_id": "gas-1",
            "fuel": "gas",
            "region": "default",
            "cap_mw": 70.0,
            "availability": 0.85,
            "hr_mmbtu_per_mwh": 7.0,
            "vom_per_mwh": 2.0,
            "fuel_price_per_mmbtu": 2.5,
            "ef_ton_per_mwh": 0.45,
        },
    ]
    return pd.DataFrame(data)


def _default_fuels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fuel": "wind", "covered": False},
            {"fuel": "coal", "covered": True},
            {"fuel": "gas", "covered": True},
        ]
    )


def _default_transmission() -> pd.DataFrame:
    return pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])


def _build_default_frames(
    years: Iterable[int],
    *,
    carbon_policy_enabled: bool = True,
    banking_enabled: bool = True,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
) -> FramesType:
    frames_cls = FramesType
    demand_records = [
        {"year": int(year), "region": "default", "demand_mwh": float(_DEFAULT_LOAD_MWH)}
        for year in years
    ]
    base_frames = {
        "units": _default_units(),
        "demand": pd.DataFrame(demand_records),
        "fuels": _default_fuels(),
        "transmission": _default_transmission(),
    }
    return frames_cls(
        base_frames,
        carbon_policy_enabled=carbon_policy_enabled,
        banking_enabled=banking_enabled,
        carbon_price_schedule=carbon_price_schedule,
    )


def _ensure_years_in_demand(frames: FramesType, years: Iterable[int]) -> FramesType:
    if not years:
        return frames

    demand = frames.demand()
    if demand.empty:
        raise ValueError("Demand frame is empty; cannot infer loads for requested years")

    existing_years = {int(year) for year in demand["year"].unique()}
    target_years = {int(year) for year in years}
    missing = sorted(target_years - existing_years)
    if not missing:
        return frames

    averages = demand.groupby("region")["demand_mwh"].mean()
    new_rows: list[dict[str, Any]] = []
    for year in missing:
        for region, value in averages.items():
            new_rows.append({"year": year, "region": region, "demand_mwh": float(value)})

    demand_updated = pd.concat([demand, pd.DataFrame(new_rows)], ignore_index=True)
    demand_updated = demand_updated.sort_values(["year", "region"]).reset_index(drop=True)
    return frames.with_frame("demand", demand_updated)


def _temporary_output_directory(prefix: str = "bluesky_gui_") -> Path:
    """Create a writable temporary directory for engine CSV outputs.

    Some execution environments (notably restricted containers) provide a
    read-only ``/tmp``.  ``tempfile.mkdtemp`` raises :class:`PermissionError`
    in those cases which previously caused CSV exports to silently fail.  To
    keep the download buttons working we attempt a small set of candidate
    locations and fall back to a project specific directory under the current
    working directory or the user's home directory.
    """

    candidates: list[Path] = []

    override = os.environ.get("GRANITELEDGER_TMPDIR")
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(Path(tempfile.gettempdir()))
    candidates.append(Path.cwd() / ".graniteledger" / "tmp")

    home = Path.home()
    if home:
        candidates.append(home / ".graniteledger" / "tmp")

    tried: list[tuple[Path, Exception]] = []
    seen: set[Path] = set()
    for base_dir in candidates:
        if base_dir in seen:
            continue
        seen.add(base_dir)

        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            tried.append((base_dir, exc))
            continue

        try:
            return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_dir)))
        except OSError as exc:
            tried.append((base_dir, exc))
            continue

    error_detail = "; ".join(f"{path}: {exc}" for path, exc in tried) or "no candidates available"
    raise RuntimeError(f"Unable to create temporary output directory ({error_detail}).")


def _write_outputs_to_temp(outputs) -> tuple[Path, dict[str, bytes]]:
    temp_dir = _temporary_output_directory()
    # Expect outputs to expose to_csv(target_dir)
    if hasattr(outputs, "to_csv"):
        try:
            outputs.to_csv(temp_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise TypeError("Runner outputs object does not implement to_csv(Path).")
    csv_files: dict[str, bytes] = {}
    for csv_path in temp_dir.glob("*.csv"):
        csv_files[csv_path.name] = csv_path.read_bytes()
    return temp_dir, csv_files


def _extract_output_dataframe(outputs: Any, names: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame from ``outputs`` matching one of ``names``.

    The engine historically exposed results as :class:`EngineOutputs` with
    attributes named ``annual``, ``emissions_by_region`` and so on.  Some
    development branches temporarily renamed these attributes which broke the
    GUI.  This helper provides a resilient lookup that supports both the
    canonical names and any temporary aliases.  When a name cannot be resolved
    an empty DataFrame is returned so the UI can still render informative
    placeholders instead of failing outright.
    """

    for name in names:
        candidate: Any | None = None
        if hasattr(outputs, name):
            candidate = getattr(outputs, name)
        elif isinstance(outputs, Mapping):
            candidate = outputs.get(name)

        if isinstance(candidate, pd.DataFrame):
            return candidate
        if candidate is None:
            continue

        if isinstance(candidate, pd.Series):
            return candidate.to_frame().reset_index(drop=False)

        if isinstance(candidate, Mapping):
            # ``pd.DataFrame`` cannot coerce dictionaries of scalars directly – a
            # frequent pattern for single-region dispatch results.  Attempt an
            # index-oriented conversion before falling back to the generic
            # constructor so we can still surface the data in the UI.
            try:
                coerced = pd.DataFrame(candidate)
            except Exception:
                try:
                    coerced = pd.DataFrame.from_dict(candidate, orient="index")
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.warning(
                        "Unable to coerce mapping output field '%s' to a DataFrame.",
                        name,
                    )
                    continue
                else:
                    return coerced.reset_index(drop=False)
            else:
                return coerced

        try:
            coerced = pd.DataFrame(candidate)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "Unable to coerce engine output field '%s' to a DataFrame.", name
            )
            continue
        else:
            return coerced

    LOGGER.warning(
        "Engine runner outputs missing expected field(s): %s", ", ".join(names)
    )
    return pd.DataFrame()


def _normalize_dispatch_price_frame(
    price_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Return a price DataFrame with best-effort column normalisation.

    Engine refactors occasionally rename the dispatch price fields or return
    mappings that are awkward to coerce into :class:`pandas.DataFrame`
    instances.  The GUI previously assumed the canonical ``['year', 'region',
    'price']`` schema which caused otherwise valid single-region outputs to be
    treated as empty.  This helper performs a defensive normalisation step so
    the UI can render whatever data is available while signalling missing
    columns to the caller.
    """

    if not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return pd.DataFrame(), {"year": False, "region": False, "price": False}

    df = price_df.copy()

    # Promote index labels to columns when possible.  Many historical outputs
    # stored the region name in the index rather than an explicit column.
    if df.index.name or (getattr(df.index, "names", None) and any(df.index.names)):
        df = df.reset_index(drop=False)

    alias_map: dict[str, tuple[str, ...]] = {
        "year": ("year", "period", "calendar_year"),
        "region": ("region", "regions", "zone", "market", "node", "index"),
        "price": (
            "price",
            "value",
            "cost",
            "marginal_cost",
            "dispatch_price",
            "dispatch_cost",
        ),
    }

    rename_map: dict[str, str] = {}
    lower_lookup = {col.lower(): col for col in df.columns}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            column = lower_lookup.get(alias.lower())
            if column is not None:
                rename_map[column] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # When the price column is missing but only a single numeric column is
    # available, assume it represents the dispatch price.
    if "price" not in df.columns:
        numeric_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_columns) == 1:
            df = df.rename(columns={numeric_columns[0]: "price"})

    # If region data is absent but the DataFrame now contains a generic
    # ``index`` column from reset_index(), interpret it as the region label.
    if "region" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "region"})

    field_flags = {key: (key in df.columns) for key in ("year", "region", "price")}
    return df, field_flags


def _read_uploaded_dataframe(uploaded_file: Any | None) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    try:
        if hasattr(uploaded_file, "getvalue"):
            raw = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
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
        st.error(f"Unable to read CSV: {exc}")
        return None

    if df.empty:
        _ensure_streamlit()
        st.warning("Uploaded CSV is empty.")

    return df


def _validate_frame_override(
    frames_obj: FramesType,
    frame_name: str,
    df: pd.DataFrame,
) -> tuple[FramesType | None, str | None]:
    validator_name = frame_name.lower()
    try:
        candidate = frames_obj.with_frame(frame_name, df)
        validator = getattr(candidate, validator_name, None)
        if callable(validator):
            validator()
        else:
            candidate.frame(frame_name)
        return candidate, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


# -------------------------
# Assumptions editor tabs
# -------------------------
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
        st.caption("Current demand assumptions")
        st.dataframe(demand_default, width="stretch")
    else:
        st.info("No default demand data found. Provide values via the controls or upload a CSV.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None

    target_years = sorted({int(year) for year in years}) if years else []
    if not target_years and not demand_default.empty:
        target_years = sorted({int(year) for year in demand_default["year"].unique()})
    if not target_years:
        target_years = [int(date.today().year)]

    use_manual = st.checkbox("Create demand profile with controls", value=False, key="demand_manual_toggle")
    if use_manual:
        st.caption("Set a baseline load, per-region multipliers, and annual growth to construct demand.")
        if not demand_default.empty:
            first_year = target_years[0]
            base_year_data = demand_default[demand_default["year"] == first_year]
            default_base = float(base_year_data["demand_mwh"].mean()) if not base_year_data.empty else float(_DEFAULT_LOAD_MWH)
        else:
            default_base = float(_DEFAULT_LOAD_MWH)

        base_value = float(
            st.number_input(
                "Baseline demand for the first year (MWh)",
                min_value=0.0,
                value=max(0.0, default_base),
                step=10_000.0,
                format="%0.0f",
            )
        )
        growth_pct = float(
            st.slider(
                "Annual growth rate (%)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.25,
                key="demand_growth",
            )
        )

        if not demand_default.empty:
            region_labels = sorted({str(region) for region in demand_default["region"].unique()})
            region_defaults = (
                demand_default[demand_default["year"] == target_years[0]]
                .set_index("region")["demand_mwh"]
                .to_dict()
            )
        else:
            region_labels = ["default"]
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
                    f"{region} demand multiplier",
                    min_value=0.1,
                    max_value=3.0,
                    value=multiplier_default,
                    step=0.05,
                    key=f"demand_scale_{region}",
                )
            )

            for index, year in enumerate(target_years):
                growth_factor = (1.0 + growth_pct / 100.0) ** index
                demand_val = base_value * multiplier * growth_factor
                manual_records.append(
                    {
                        "year": int(year),
                        "region": region,
                        "demand_mwh": float(demand_val),
                    }
                )

        manual_df = pd.DataFrame(manual_records)
        manual_note = (
            f"Demand constructed from GUI controls with baseline {base_value:,.0f} MWh, "
            f"growth {growth_pct:0.2f}% across {len(region_labels)} region(s) "
            f"and {len(target_years)} year(s)."
        )

    uploaded = st.file_uploader("Upload demand CSV", type="csv", key="demand_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded demand CSV overrides manual adjustments.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "demand", upload_df)
            if candidate is None:
                message = f"Demand CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f"Demand table loaded from {uploaded.name} ({len(upload_df)} row(s)).")

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "demand", manual_df)
        if candidate is None:
            message = f"Demand override invalid: {error}"
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
        st.caption("Current generating units")
        st.dataframe(units_default, width="stretch")
    else:
        st.info("No generating units are defined. Upload a CSV to provide unit characteristics.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox("Edit units inline", value=False, key="units_manual_toggle")
    if edit_inline and not units_default.empty:
        st.caption("Adjust unit properties with the controls below.")
        manual_records: list[dict[str, Any]] = []
        for index, row in units_default.iterrows():
            unit_label = str(row["unit_id"])
            st.markdown(f"**{unit_label}**")
            col_meta = st.columns(3)
            with col_meta[0]:
                unit_id = st.text_input(
                    "Unit ID",
                    value=unit_label,
                    key=f"units_unit_id_{index}",
                ).strip() or unit_label
            with col_meta[1]:
                region = st.text_input(
                    "Region",
                    value=str(row["region"]),
                    key=f"units_region_{index}",
                ).strip() or str(row["region"])
            with col_meta[2]:
                fuel = st.text_input(
                    "Fuel",
                    value=str(row["fuel"]),
                    key=f"units_fuel_{index}",
                ).strip() or str(row["fuel"])

            col_perf = st.columns(3)
            with col_perf[0]:
                cap_mw = st.number_input(
                    "Capacity (MW)",
                    min_value=0.0,
                    value=float(row["cap_mw"]),
                    step=1.0,
                    key=f"units_cap_{index}",
                )
            with col_perf[1]:
                availability = st.slider(
                    "Availability",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(row["availability"]),
                    step=0.01,
                    key=f"units_availability_{index}",
                )
            with col_perf[2]:
                heat_rate = st.number_input(
                    "Heat rate (MMBtu/MWh)",
                    min_value=0.0,
                    value=float(row["hr_mmbtu_per_mwh"]),
                    step=0.1,
                    key=f"units_heat_rate_{index}",
                )

            col_cost = st.columns(3)
            with col_cost[0]:
                vom = st.number_input(
                    "VOM ($/MWh)",
                    min_value=0.0,
                    value=float(row["vom_per_mwh"]),
                    step=0.1,
                    key=f"units_vom_{index}",
                )
            with col_cost[1]:
                fuel_price = st.number_input(
                    "Fuel price ($/MMBtu)",
                    min_value=0.0,
                    value=float(row["fuel_price_per_mmbtu"]),
                    step=0.1,
                    key=f"units_fuel_price_{index}",
                )
            with col_cost[2]:
                emission_factor = st.number_input(
                    "Emission factor (ton/MWh)",
                    min_value=0.0,
                    value=float(row["ef_ton_per_mwh"]),
                    step=0.01,
                    key=f"units_ef_{index}",
                )

            manual_records.append(
                {
                    "unit_id": unit_id,
                    "region": region,
                    "fuel": fuel,
                    "cap_mw": float(cap_mw),
                    "availability": float(availability),
                    "hr_mmbtu_per_mwh": float(heat_rate),
                    "vom_per_mwh": float(vom),
                    "fuel_price_per_mmbtu": float(fuel_price),
                    "ef_ton_per_mwh": float(emission_factor),
                }
            )

        manual_df = pd.DataFrame(manual_records)
        manual_note = f"Units modified via GUI controls ({len(manual_records)} unit(s))."
    elif edit_inline:
        st.info("Upload a units CSV to edit inline.")

    uploaded = st.file_uploader("Upload units CSV", type="csv", key="units_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded units CSV overrides inline edits.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "units", upload_df)
            if candidate is None:
                message = f"Units CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f"Units loaded from {uploaded.name} ({len(upload_df)} row(s)).")

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "units", manual_df)
        if candidate is None:
            message = f"Units override invalid: {error}"
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
        st.caption("Current fuel coverage")
        st.dataframe(fuels_default, width="stretch")
    else:
        st.info("No fuel data available. Upload a CSV to specify fuel coverage.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox("Edit fuel coverage inline", value=False, key="fuels_manual_toggle")
    if edit_inline and not fuels_default.empty:
        st.caption("Toggle coverage and update emission factors as needed.")
        manual_records: list[dict[str, Any]] = []
        has_emission_column = "co2_ton_per_mmbtu" in fuels_default.columns
        for index, row in fuels_default.iterrows():
            fuel_label = str(row["fuel"])
            col_line = st.columns(3 if has_emission_column else 2)
            with col_line[0]:
                fuel_name = st.text_input(
                    "Fuel",
                    value=fuel_label,
                    key=f"fuels_name_{index}",
                ).strip() or fuel_label
            with col_line[1]:
                covered = st.checkbox(
                    "Covered",
                    value=bool(row["covered"]),
                    key=f"fuels_covered_{index}",
                )
            emission_value: float | None = None
            if has_emission_column:
                with col_line[2]:
                    emission_value = float(
                        st.number_input(
                            "CO₂ tons/MMBtu",
                            min_value=0.0,
                            value=float(row.get("co2_ton_per_mmbtu", 0.0)),
                            step=0.01,
                            key=f"fuels_emission_{index}",
                        )
                    )

            record: dict[str, Any] = {"fuel": fuel_name, "covered": bool(covered)}
            if has_emission_column:
                record["co2_ton_per_mmbtu"] = float(emission_value or 0.0)
            manual_records.append(record)

        manual_df = pd.DataFrame(manual_records)
        manual_note = f"Fuel coverage edited inline ({len(manual_records)} fuel(s))."
    elif edit_inline:
        st.info("Upload a fuels CSV to edit inline.")

    uploaded = st.file_uploader("Upload fuels CSV", type="csv", key="fuels_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded fuels CSV overrides inline edits.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "fuels", upload_df)
            if candidate is None:
                message = f"Fuels CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f"Fuels loaded from {uploaded.name} ({len(upload_df)} row(s)).")

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "fuels", manual_df)
        if candidate is None:
            message = f"Fuels override invalid: {error}"
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
        st.caption("Current transmission limits")
        st.dataframe(transmission_default, width="stretch")
    else:
        st.info("No transmission limits specified. Add entries below or upload a CSV.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox("Edit transmission limits inline", value=False, key="transmission_manual_toggle")
    if edit_inline:
        editable = transmission_default.copy()
        if editable.empty:
            editable = pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])
        st.caption("Use the table to add or modify directional flow limits (MW).")
        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            width="stretch",
            key="transmission_editor",
        )
        manual_df = (edited.copy() if isinstance(edited, pd.DataFrame) else pd.DataFrame(edited)).dropna(how="all")
        manual_df = manual_df.reindex(columns=["from_region", "to_region", "limit_mw"])
        manual_note = f"Transmission table edited inline ({len(manual_df)} record(s))."

    uploaded = st.file_uploader("Upload transmission CSV", type="csv", key="transmission_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded transmission CSV overrides inline edits.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "transmission", upload_df)
            if candidate is None:
                message = f"Transmission CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(
                    f"Transmission limits loaded from {uploaded.name} ({len(upload_df)} row(s))."
                )

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "transmission", manual_df)
        if candidate is None:
            message = f"Transmission override invalid: {error}"
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


# -------------------------
# Runner
# -------------------------
def _build_run_summary(
    params: Mapping[str, Any] | None,
    *,
    config_label: str | None = None,
) -> list[tuple[str, str]]:
    summary: list[tuple[str, str]] = []

    if config_label:
        summary.append(("Configuration", config_label))

    if not isinstance(params, Mapping):
        return summary

    def _coerce_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _enabled_label(flag: object, *, true: str = "Enabled", false: str = "Disabled") -> str:
        return true if bool(flag) else false

    start_year = _coerce_int(params.get("start_year"))
    end_year = _coerce_int(params.get("end_year"))

    if start_year is not None and end_year is not None:
        if start_year == end_year:
            summary.append(("Simulation years", str(start_year)))
        else:
            total_years = max(0, end_year - start_year + 1)
            years_label = f"{start_year}–{end_year}"
            if total_years > 0:
                years_label = f"{years_label} ({total_years} year(s))"
            summary.append(("Simulation years", years_label))
    elif start_year is not None:
        summary.append(("Simulation start year", str(start_year)))
    elif end_year is not None:
        summary.append(("Simulation end year", str(end_year)))

    carbon_enabled = params.get("carbon_policy_enabled")
    summary.append(("Carbon policy", _enabled_label(carbon_enabled)))

    if carbon_enabled:
        summary.append(("Price floor", _enabled_label(params.get("enable_floor"))))
        ccr_enabled = params.get("enable_ccr")
        summary.append(("Cost containment reserve", _enabled_label(ccr_enabled)))
        if ccr_enabled:
            ccr_triggers: list[str] = []
            if params.get("ccr1_enabled"):
                ccr_triggers.append("CCR1")
            if params.get("ccr2_enabled"):
                ccr_triggers.append("CCR2")
            if ccr_triggers:
                summary.append(("CCR triggers", ", ".join(ccr_triggers)))
        summary.append(
            (
                "Allowance banking",
                _enabled_label(params.get("allowance_banking_enabled"), true="Allowed", false="Not allowed"),
            )
        )
        control_period = _coerce_int(params.get("control_period_years"))
        if control_period:
            summary.append(("Control period", f"{control_period} year(s)"))

    dispatch_network = params.get("dispatch_use_network")
    if dispatch_network is not None:
        summary.append(
            (
                "Electricity dispatch",
                "Network" if bool(dispatch_network) else "Zonal",
            )
        )

    capacity_toggle = params.get("dispatch_capacity_expansion")
    if capacity_toggle is not None:
        summary.append(
            (
                "Capacity expansion",
                _enabled_label(capacity_toggle, true="Enabled", false="Disabled"),
            )
        )

    module_config = params.get("module_config")
    if isinstance(module_config, Mapping):
        enabled_modules: list[str] = []
        disabled_modules: list[str] = []
        for raw_name, settings in module_config.items():
            name = str(raw_name)
            if isinstance(settings, Mapping):
                enabled = settings.get("enabled", True)
            else:
                enabled = bool(settings)
            label = name.replace("_", " ").strip().title() or name
            if bool(enabled):
                enabled_modules.append(label)
            else:
                disabled_modules.append(label)
        if enabled_modules:
            summary.append(("Modules enabled", ", ".join(sorted(enabled_modules))))
        if disabled_modules:
            summary.append(("Modules disabled", ", ".join(sorted(disabled_modules))))

    return summary

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
    ccr1_price: float | None = None,
    ccr2_price: float | None = None,
    ccr1_escalator_pct: float | None = None,
    ccr2_escalator_pct: float | None = None,
    allowance_banking_enabled: bool = True,
    initial_bank: float = 0.0,
    coverage_regions: Iterable[str] | None = None,
    control_period_years: int | None = None,
    cap_regions: Sequence[Any] | None = None,
    carbon_price_enabled: bool | None = None,
    carbon_price_value: float | None = None,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    carbon_price_escalator_pct: float | None = None,
    carbon_cap_start_value: float | None = None,
    carbon_cap_reduction_mode: str | None = None,
    carbon_cap_reduction_value: float | None = None,
    carbon_cap_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    price_floor_value: float | None = None,
    price_floor_escalator_mode: str | None = None,
    price_floor_escalator_value: float | None = None,
    price_floor_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    dispatch_use_network: bool = False,
    dispatch_capacity_expansion: bool | None = None,
    deep_carbon_pricing: bool = False,
    module_config: Mapping[str, Any] | None = None,
    frames: FramesType | Mapping[str, pd.DataFrame] | None = None,
    assumption_notes: Iterable[str] | None = None,
    progress_cb: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, Any]:




    policy_override = bool(carbon_policy_enabled)
    if carbon_price_enabled is None:
        price_override: bool | None = None
    else:
        price_override = bool(carbon_price_enabled)

    try:
        config = _load_config_data(config_source)
    except Exception as exc:  # pragma: no cover
        return {"error": f"Unable to load configuration: {exc}"}

    config.setdefault("modules", {})


    try:
        base_years = _years_from_config(config)
        years = _select_years(base_years, start_year, end_year)
    except Exception as exc:
        return {"error": f"Invalid year selection: {exc}"}

    merged_modules = _merge_module_dicts(config.get("modules"), module_config)
    dispatch_defaults = merged_modules.get("electricity_dispatch", {})
    if deep_carbon_pricing is None:
        deep_carbon_flag = bool(dispatch_defaults.get("deep_carbon_pricing", False))
    else:
        deep_carbon_flag = bool(deep_carbon_pricing)

    carbon_policy_cfg = CarbonPolicyConfig.from_mapping(
        merged_modules.get("carbon_policy"),
        enabled=policy_override,
        enable_floor=enable_floor,
        enable_ccr=enable_ccr,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        ccr1_price=ccr1_price,
        ccr2_price=ccr2_price,
        ccr1_escalator_pct=ccr1_escalator_pct,
        ccr2_escalator_pct=ccr2_escalator_pct,
        allowance_banking_enabled=allowance_banking_enabled,
        control_period_years=control_period_years,
    )

    price_cfg = CarbonPriceConfig.from_mapping(
        merged_modules.get("carbon_price"),
        enabled=price_override,
        value=carbon_price_value,
        schedule=carbon_price_schedule,
        years=years,
        escalator_pct=carbon_price_escalator_pct,
    )

    # Expand any provided carbon price schedule across all modeled years
    if price_cfg.schedule:
        expanded_schedule = _expand_or_build_price_schedule(
            price_cfg.schedule,
            years,
        )
        price_cfg.schedule = expanded_schedule if expanded_schedule else {}

    # Track whether cap constraints were explicitly requested
    explicit_cap_request = (coverage_regions is not None) or (cap_regions is not None)

    if price_cfg.active:
        if (
            carbon_policy_cfg.enabled
            and policy_override
            and explicit_cap_request
            and not deep_carbon_pricing
        ):
            return {
                "error": "Cannot enable both carbon cap and carbon price simultaneously."
            }
        if not deep_carbon_pricing:
            carbon_policy_cfg.disable_for_price()

    normalized_coverage = _normalize_coverage_selection(
        coverage_regions
        if coverage_regions is not None
        else merged_modules.get("carbon_policy", {}).get("coverage_regions", ["All"])
    )



    policy_enabled = bool(carbon_policy_cfg.enabled)
    floor_flag = bool(policy_enabled and carbon_policy_cfg.enable_floor)
    ccr_flag = bool(
        policy_enabled
        and carbon_policy_cfg.enable_ccr
        and (carbon_policy_cfg.ccr1_enabled or carbon_policy_cfg.ccr2_enabled)
    )
    banking_flag = bool(policy_enabled and carbon_policy_cfg.allowance_banking_enabled)

    raw_initial_bank = _coerce_float(initial_bank, default=0.0)

    market_cfg = config.get("allowance_market")
    bank0_from_config: float | None = None
    if isinstance(market_cfg, Mapping):
        bank0_from_config = _coerce_float(market_cfg.get("bank0"), default=0.0)
        if not banking_flag and bank0_from_config > 0.0:
            LOGGER.warning(
                "Allowance banking disabled; ignoring initial bank of %.3f tons.",
                bank0_from_config,
            )
            try:
                market_cfg["bank0"] = 0.0  # type: ignore[index]
            except Exception:  # pragma: no cover - best effort for immutable mappings
                pass

    if not banking_flag and raw_initial_bank > 0.0:
        LOGGER.warning(
            "Allowance banking disabled; ignoring requested initial bank of %.3f tons.",
            raw_initial_bank,
        )
        raw_initial_bank = 0.0

    carbon_record = merged_modules.setdefault("carbon_policy", {})
    initial_bank_value = raw_initial_bank if banking_flag else 0.0

    cap_start_value_norm: float | None
    if carbon_cap_start_value is not None:
        try:
            cap_start_value_norm = float(carbon_cap_start_value)
        except (TypeError, ValueError):
            cap_start_value_norm = None
    else:
        existing_cap_start = carbon_record.get("cap_start_value")
        try:
            cap_start_value_norm = (
                float(existing_cap_start) if existing_cap_start is not None else None
            )
        except (TypeError, ValueError):
            cap_start_value_norm = None

    raw_mode = (
        carbon_cap_reduction_mode
        if carbon_cap_reduction_mode is not None
        else carbon_record.get("cap_reduction_mode")
    )
    cap_reduction_mode_norm = str(raw_mode or "percent").strip().lower()
    if cap_reduction_mode_norm not in {"percent", "fixed"}:
        cap_reduction_mode_norm = "percent"

    if carbon_cap_reduction_value is not None:
        try:
            cap_reduction_value_norm = float(carbon_cap_reduction_value)
        except (TypeError, ValueError):
            cap_reduction_value_norm = 0.0
    else:
        existing_reduction = carbon_record.get("cap_reduction_value", 0.0)
        try:
            cap_reduction_value_norm = float(existing_reduction)
        except (TypeError, ValueError):
            cap_reduction_value_norm = 0.0

    floor_base_value = price_floor.parse_currency_value(
        price_floor_value if price_floor_value is not None else carbon_policy_cfg.floor_value,
        carbon_policy_cfg.floor_value,
    )
    floor_mode_norm = str(
        price_floor_escalator_mode
        if price_floor_escalator_mode is not None
        else carbon_policy_cfg.floor_escalator_mode
    ).strip().lower()
    if floor_mode_norm not in {"fixed", "percent"}:
        floor_mode_norm = "fixed"
    escalator_input = (
        price_floor_escalator_value
        if price_floor_escalator_value is not None
        else carbon_policy_cfg.floor_escalator_value
    )
    if floor_mode_norm == "percent":
        floor_escalator_norm = price_floor.parse_percentage_value(
            escalator_input, carbon_policy_cfg.floor_escalator_value
        )
    else:
        floor_escalator_norm = price_floor.parse_currency_value(
            escalator_input, carbon_policy_cfg.floor_escalator_value
        )
    provided_floor_schedule = price_floor_schedule
    if not isinstance(provided_floor_schedule, Mapping) or not provided_floor_schedule:
        provided_floor_schedule = carbon_record.get("floor_schedule")
    if (
        (not isinstance(provided_floor_schedule, Mapping) or not provided_floor_schedule)
        and isinstance(market_cfg, Mapping)
    ):
        provided_floor_schedule = market_cfg.get("floor")
    provided_schedule_map: dict[int, float] = {}
    if isinstance(provided_floor_schedule, Mapping):
        for year_key, value in provided_floor_schedule.items():
            try:
                year_int = int(year_key)
                provided_schedule_map[year_int] = float(value)
            except (TypeError, ValueError):
                continue
    baseline_floor_schedule: dict[int, float] = {}
    if policy_enabled and floor_flag:
        baseline_floor_schedule = price_floor.build_schedule(
            years,
            floor_base_value,
            floor_mode_norm,
            floor_escalator_norm,
        )
    if provided_schedule_map:
        baseline_floor_schedule.update(provided_schedule_map)
    floor_schedule_map = dict(sorted(baseline_floor_schedule.items())) if baseline_floor_schedule else {}
    carbon_record.update(
        {
            "enabled": policy_enabled,
            "enable_floor": floor_flag,
            "enable_ccr": ccr_flag,
            "ccr1_enabled": bool(carbon_policy_cfg.ccr1_enabled) if ccr_flag else False,
            "ccr2_enabled": bool(carbon_policy_cfg.ccr2_enabled) if ccr_flag else False,
            "allowance_banking_enabled": banking_flag,
            "coverage_regions": normalized_coverage,
            "control_period_years": (
                carbon_policy_cfg.control_period_years if policy_enabled else None
            ),
            "bank0": initial_bank_value,
            "cap_start_value": cap_start_value_norm,
            "cap_reduction_mode": cap_reduction_mode_norm,
            "cap_reduction_value": float(cap_reduction_value_norm),
            "floor_value": float(floor_base_value),
            "floor_escalator_mode": str(floor_mode_norm),
            "floor_escalator_value": float(floor_escalator_norm),
        }
    )
    # Carbon price schedule will be populated after years are finalised below
    if floor_schedule_map:
        carbon_record["floor_schedule"] = dict(sorted(floor_schedule_map.items()))
    else:
        carbon_record.pop("floor_schedule", None)

    merged_modules["carbon_price"] = price_cfg.as_dict()

    raw_run_years = []
    try:
        raw_run_years = [int(year) for year in years]
    except Exception:
        raw_run_years = []
    run_years_sorted = sorted(dict.fromkeys(raw_run_years))

    provided_price_schedule: dict[int, float] = {}
    if isinstance(price_cfg.schedule, Mapping):
        for year, value in price_cfg.schedule.items():
            try:
                provided_price_schedule[int(year)] = float(value)
            except (TypeError, ValueError):
                continue

    price_schedule_map: dict[int, float] = {}
    if price_cfg.active:
        if run_years_sorted:
            start_year = run_years_sorted[0]
            end_year = run_years_sorted[-1]
        elif provided_price_schedule:
            start_year = min(provided_price_schedule)
            end_year = max(provided_price_schedule)
        else:
            start_year = end_year = None

        if start_year is not None and end_year is not None:
            base_year = min(provided_price_schedule) if provided_price_schedule else start_year
            base_price = provided_price_schedule.get(base_year, price_cfg.price_per_ton)
            try:
                base_price_float = float(base_price)
            except (TypeError, ValueError):
                base_price_float = 0.0

            growth_pct = float(price_cfg.escalator_pct)

            # Back-adjust if provided base year is later than start_year
            if base_year is not None and base_year != start_year:
                ratio = 1.0 + (growth_pct or 0.0) / 100.0
                try:
                    steps = base_year - start_year
                    if ratio != 0.0:
                        base_price_float = base_price_float / (ratio ** steps)
                except OverflowError:
                    base_price_float = 0.0

            generated_schedule = _build_price_schedule(
                start_year,
                end_year,
                base_price_float,
                growth_pct,
            )

            # Lock schedule to run_years if explicitly provided
            if run_years_sorted:
                generated_schedule = {
                    year: generated_schedule.get(year, base_price_float)
                    for year in run_years_sorted
                }

            # Overlay provided overrides
            for year, value in provided_price_schedule.items():
                generated_schedule[int(year)] = float(value)

            price_schedule_map = dict(sorted(generated_schedule.items()))

    price_active = bool(price_cfg.active and price_schedule_map)


    cap_schedule_map: dict[int, float] = {}
    if isinstance(carbon_cap_schedule, Mapping):
        for year_key, value in carbon_cap_schedule.items():
            try:
                year_int = int(year_key)
                cap_schedule_map[year_int] = float(value)
            except (TypeError, ValueError):
                continue
    if policy_enabled and not cap_schedule_map and cap_start_value_norm is not None:
        cap_schedule_map = _build_cap_reduction_schedule(
            cap_start_value_norm,
            cap_reduction_mode_norm,
            cap_reduction_value_norm,
            years,
        )
    if cap_schedule_map:
        carbon_record["cap_schedule"] = dict(sorted(cap_schedule_map.items()))
    else:
        carbon_record.pop("cap_schedule", None)
    allowance_market_record = merged_modules.setdefault("allowance_market", {})
    allowance_market_record["enabled"] = policy_enabled
    allowance_market_record["bank0"] = float(initial_bank_value)
    allowance_market_record["ccr1_enabled"] = bool(
        carbon_policy_cfg.ccr1_enabled if ccr_flag else False
    )
    allowance_market_record["ccr2_enabled"] = bool(
        carbon_policy_cfg.ccr2_enabled if ccr_flag else False
    )
    if policy_enabled and cap_schedule_map:
        allowance_market_record["cap"] = dict(sorted(cap_schedule_map.items()))
    elif not policy_enabled:
        allowance_market_record.pop("cap", None)

    if policy_enabled and floor_schedule_map:
        allowance_market_record["floor"] = dict(sorted(floor_schedule_map.items()))
    elif not policy_enabled or not floor_flag:
        allowance_market_record.pop("floor", None)
    existing_allowance_cfg = config.get("allowance_market")
    if isinstance(existing_allowance_cfg, Mapping):
        allowance_config = dict(existing_allowance_cfg)
    else:
        allowance_config = {}
    allowance_config["enabled"] = policy_enabled
    allowance_config["bank0"] = float(initial_bank_value)
    allowance_config["ccr1_enabled"] = bool(
        carbon_policy_cfg.ccr1_enabled if ccr_flag else False
    )
    allowance_config["ccr2_enabled"] = bool(
        carbon_policy_cfg.ccr2_enabled if ccr_flag else False
    )
    if policy_enabled and cap_schedule_map:
        allowance_config["cap"] = dict(sorted(cap_schedule_map.items()))
    elif not policy_enabled:
        allowance_config.pop("cap", None)
    if policy_enabled and floor_schedule_map:
        allowance_config["floor"] = dict(sorted(floor_schedule_map.items()))
    elif not policy_enabled or not floor_flag:
        allowance_config.pop("floor", None)
    config["allowance_market"] = allowance_config
    try:
        normalized_regions, cap_region_aliases = _normalize_cap_region_entries(cap_regions)
    except ValueError as exc:
        return {"error": str(exc)}

    if normalized_regions:
        carbon_record["regions"] = list(normalized_regions)

    if not normalized_regions and normalized_coverage and normalized_coverage != ["All"]:
        normalized_regions = list(normalized_coverage)
        if normalized_regions:
            carbon_record["regions"] = list(normalized_regions)

    config["modules"] = merged_modules

    dispatch_record = merged_modules.setdefault("electricity_dispatch", {})
    capacity_setting = dispatch_record.get("capacity_expansion")
    if dispatch_capacity_expansion is not None:
        capacity_flag = bool(dispatch_capacity_expansion)
    elif capacity_setting is not None:
        capacity_flag = bool(capacity_setting)
    else:
        capacity_flag = True
    dispatch_record["capacity_expansion"] = capacity_flag
    dispatch_record["use_network"] = bool(dispatch_use_network)
    dispatch_record["deep_carbon_pricing"] = bool(deep_carbon_pricing)

    if capacity_flag:
        config["sw_expansion"] = 1
    else:
        config["sw_expansion"] = 0
        if config.get("sw_rm") not in (None, 0, False):
            config["sw_rm"] = 0


    def _coerce_year_range(start: int | None, end: int | None) -> list[int]:
        if start is None and end is None:
            return []
        if start is None:
            start = end
        if end is None:
            end = start
        assert start is not None and end is not None
        step = 1 if end >= start else -1
        return list(range(int(start), int(end) + step, step))

    years = _coerce_year_range(start_year, end_year)
    if not years:
        years = _years_from_config(config)
    if not years:
        fallback_year = start_year or end_year
        if fallback_year is not None:
            years = [int(fallback_year)]
        else:
            years = [int(date.today().year)]

    normalized_years = sorted({int(year) for year in years})
    if normalized_years:
        year_start = normalized_years[0]
        year_end = normalized_years[-1]
        years = list(range(year_start, year_end + 1))
    else:
        current_year = int(date.today().year)
        years = [current_year]

    config["years"] = list(years)
    config["start_year"] = int(years[0])
    config["end_year"] = int(years[-1])

    provided_price_schedule: dict[int, float] = {}
    if isinstance(price_cfg.schedule, Mapping):
        for year, value in price_cfg.schedule.items():
            try:
                provided_price_schedule[int(year)] = float(value)
            except (TypeError, ValueError):
                continue

    price_schedule_map: dict[int, float] = {}
    if price_cfg.enabled and years:
        run_years_sorted = sorted(dict.fromkeys(int(year) for year in years))
        if run_years_sorted:
            start_year_schedule = run_years_sorted[0]
            end_year_schedule = run_years_sorted[-1]
            base_year = (
                min(provided_price_schedule)
                if provided_price_schedule
                else start_year_schedule
            )
            base_price = provided_price_schedule.get(base_year, price_cfg.price_per_ton)
            try:
                base_price_float = float(base_price)
            except (TypeError, ValueError):
                base_price_float = 0.0

            growth_pct = float(price_cfg.escalator_pct)
            if base_year is not None and base_year != start_year_schedule:
                ratio = 1.0 + (growth_pct or 0.0) / 100.0
                try:
                    steps = base_year - start_year_schedule
                    if ratio != 0.0:
                        base_price_float = base_price_float / (ratio ** steps)
                except OverflowError:
                    base_price_float = 0.0

            generated_schedule = _build_price_schedule(
                start_year_schedule,
                end_year_schedule,
                base_price_float,
                growth_pct,
            )

            combined_schedule = dict(sorted(generated_schedule.items()))
            for year, value in provided_price_schedule.items():
                try:
                    combined_schedule[int(year)] = float(value)
                except (TypeError, ValueError):
                    continue

            combined_items = sorted(combined_schedule.items())
            if combined_items:
                first_price = float(combined_items[0][1])
            else:
                first_price = float(base_price_float)
            filled_schedule: dict[int, float] = {}
            last_price: float | None = None
            index = 0
            total_items = len(combined_items)

            for year in run_years_sorted:
                while index < total_items and combined_items[index][0] <= year:
                    last_price = float(combined_items[index][1])
                    index += 1
                if last_price is None:
                    last_price = first_price
                filled_schedule[year] = float(last_price)

            price_schedule_map = filled_schedule

    price_active = bool(price_cfg.enabled and price_schedule_map)
    if price_active:
        price_cfg.schedule = dict(price_schedule_map)
    else:
        price_cfg.schedule = {}

    merged_modules["carbon_price"] = price_cfg.as_dict()

    carbon_price_for_frames: Mapping[int, float] | None = (
        price_schedule_map if price_active else None
    )

    if frames is None:
        frames_obj = _build_default_frames(
            years,
            carbon_policy_enabled=bool(policy_enabled),
            banking_enabled=bool(allowance_banking_enabled),
            carbon_price_schedule=carbon_price_for_frames,
        )
        demand_years: set[int] = set(years)
    else:
        frames_obj = Frames.coerce(
            frames,
            carbon_policy_enabled=bool(policy_enabled),
            banking_enabled=bool(allowance_banking_enabled),
            carbon_price_schedule=carbon_price_for_frames,
        )
        try:
            demand_years = {int(year) for year in frames_obj.demand()["year"].unique()}
        except Exception as exc:
            LOGGER.exception("Unable to read demand data from supplied frames")
            return {"error": f"Invalid demand data: {exc}"}

    requested_years = {int(year) for year in years}
    if frames is not None and demand_years and requested_years:
        if not demand_years.intersection(requested_years):
            sorted_requested = ", ".join(str(year) for year in sorted(requested_years))
            sorted_available = ", ".join(str(year) for year in sorted(demand_years))
            return {
                "error": (
                    "No demand data is available for the requested simulation years. "
                    f"Demand data covers years [{sorted_available}], but the run requested "
                    f"[{sorted_requested}]. Update the configuration or provide start_year/"
                    "end_year values that match the demand data."
                )
            }

    try:
        frames_obj = _ensure_years_in_demand(frames_obj, years)
    except Exception as exc:
        LOGGER.exception("Unable to normalise demand frame for requested years")
        return {"error": str(exc)}

    region_label_map: dict[str, Any] = {}

    def _register_region_alias(alias: Any, value: Any) -> None:
        if alias is None:
            return
        text = str(alias).strip()
        if not text:
            return
        region_label_map.setdefault(text, value)
        region_label_map.setdefault(text.lower(), value)

    for alias, canonical_value in cap_region_aliases.items():
        _register_region_alias(alias, canonical_value)

    for region in normalized_regions:
        _register_region_alias(region, region)
        _register_region_alias(canonical_region_label(region), region)
        if not isinstance(region, str):
            _register_region_alias(str(region), region)

    def _ingest_region_values(values: Sequence[Any] | pd.Series | None) -> None:
        if values is None:
            return
        if isinstance(values, pd.Series):
            iterable = values.dropna().unique()
        else:
            iterable = values
        for value in iterable:
            if value is None:
                continue
            if pd.isna(value):
                continue
            resolved_value = canonical_region_value(value)
            if isinstance(resolved_value, str):
                resolved_text = resolved_value.strip() or str(value)
                _register_region_alias(value, resolved_text)
                _register_region_alias(resolved_text, resolved_text)
            else:
                resolved_int = int(resolved_value)
                _register_region_alias(value, resolved_int)
                _register_region_alias(canonical_region_label(resolved_int), resolved_int)
                _register_region_alias(str(resolved_int), resolved_int)

    demand_region_labels: set[str] = set()
    try:
        demand_df = frames_obj.demand()
    except Exception:
        demand_df = None
    if demand_df is not None and not demand_df.empty:
        _ingest_region_values(demand_df["region"])
        demand_region_labels = {str(region) for region in demand_df["region"].unique()}

    existing_coverage_df: pd.DataFrame | None = None
    for frame_name in ("units", "coverage"):
        try:
            frame_candidate = frames_obj.optional_frame(frame_name)
        except Exception:
            frame_candidate = None
        if frame_candidate is not None and not frame_candidate.empty and "region" in frame_candidate.columns:
            _ingest_region_values(frame_candidate["region"])
            if frame_name == "coverage":
                existing_coverage_df = frame_candidate.copy()

    coverage_selection = list(normalized_coverage or [])
    cover_all = coverage_selection == ["All"]
    coverage_labels = (
        {str(label) for label in coverage_selection if str(label) and str(label) != "All"}
        if not cover_all
        else set()
    )
    for label in coverage_labels:
        resolved_value = canonical_region_value(label)
        if isinstance(resolved_value, str):
            resolved_text = resolved_value.strip() or label
            _register_region_alias(label, resolved_text)
            _register_region_alias(resolved_text, resolved_text)
        else:
            resolved_int = int(resolved_value)
            _register_region_alias(label, resolved_int)
            _register_region_alias(canonical_region_label(resolved_int), resolved_int)
            _register_region_alias(str(resolved_int), resolved_int)

    coverage_region_values: set[int | str] = set()
    for label in coverage_labels:
        resolved_value = canonical_region_value(label)
        if isinstance(resolved_value, str):
            coverage_region_values.add(resolved_value.strip() or label)
        else:
            coverage_region_values.add(int(resolved_value))

    if not demand_region_labels:
        demand_region_labels = set(region_label_map) or set(coverage_labels)

    demand_region_values: set[int | str] = set()
    for label in demand_region_labels:
        resolved_value = canonical_region_value(label)
        if isinstance(resolved_value, str):
            demand_region_values.add(resolved_value.strip() or label)
        else:
            demand_region_values.add(int(resolved_value))

    normalized_existing: pd.DataFrame | None = None
    existing_keys: set[tuple[str, int]] = set()
    if existing_coverage_df is not None and not existing_coverage_df.empty:
        normalized_existing = existing_coverage_df.copy()
        if not isinstance(normalized_existing.index, pd.RangeIndex):
            normalized_existing = normalized_existing.reset_index(drop=True)
        index_names = getattr(normalized_existing.index, "names", None) or []
        if "region" not in normalized_existing.columns and "region" in index_names:
            normalized_existing = normalized_existing.reset_index()
        if "region" not in normalized_existing.columns:
            normalized_existing = normalized_existing.assign(region=pd.Series(dtype=object))
        if "year" not in normalized_existing.columns:
            normalized_existing = normalized_existing.assign(year=-1)
        if "covered" not in normalized_existing.columns:
            normalized_existing = normalized_existing.assign(covered=False)
        normalized_existing = normalized_existing.loc[:, ["region", "year", "covered"]]
        normalized_existing["year"] = pd.to_numeric(
            normalized_existing["year"], errors="coerce"
        ).fillna(-1).astype(int)
        normalized_existing["covered"] = normalized_existing["covered"].astype(bool)
        existing_keys: set[tuple[int | str, int]] = set()
        for region, year in zip(normalized_existing["region"], normalized_existing["year"]):
            resolved_region = canonical_region_value(region)
            if isinstance(resolved_region, str):
                normalized_region = resolved_region.strip() or str(region)
            else:
                normalized_region = int(resolved_region)
            existing_keys.add((normalized_region, int(year)))

    available_region_values: set[int | str] = set(region_label_map.values())
    available_region_values |= coverage_region_values
    available_region_values |= demand_region_values
    if not available_region_values:
        available_region_values = {region for region, _ in existing_keys}

    coverage_records: list[dict[str, Any]] = []
    seen_new_keys: set[tuple[int | str, int]] = set()
    sorted_regions = sorted(
        available_region_values,
        key=lambda value: canonical_region_label(value).lower(),
    )
    for region_value in sorted_regions:
        key = (region_value, -1)
        if key in existing_keys or key in seen_new_keys:
            continue
        coverage_records.append(
            {
                "region": region_value,
                "year": -1,
                "covered": True if cover_all else region_value in coverage_region_values,
            }
        )
        seen_new_keys.add(key)

    if coverage_records:
        coverage_df = pd.DataFrame(coverage_records, columns=["region", "year", "covered"])
    else:
        coverage_df = pd.DataFrame(columns=["region", "year", "covered"])
    if normalized_existing is not None:
        coverage_df = pd.concat([normalized_existing, coverage_df], ignore_index=True)
    coverage_df = coverage_df.sort_values(["region", "year"]).reset_index(drop=True)
    frames_obj = frames_obj.with_frame("coverage", coverage_df)

    if normalized_regions:
        config_regions = list(dict.fromkeys(list(config.get("regions", [])) + normalized_regions))
        config["regions"] = config_regions

        cap_group_cfg = config.get('carbon_cap_groups')
        if isinstance(cap_group_cfg, list):
            if cap_group_cfg:
                first_entry = dict(cap_group_cfg[0])
                first_entry.setdefault('name', first_entry.get('name', 'default'))
                first_entry['regions'] = list(normalized_regions)
                cap_group_cfg[0] = first_entry
            else:
                cap_group_cfg.append({'name': 'default', 'regions': list(normalized_regions), 'cap': 'none'})
        elif isinstance(cap_group_cfg, Mapping):
            updated_groups = {}
            applied = False
            for key, value in cap_group_cfg.items():
                entry = dict(value) if isinstance(value, Mapping) else {}
                if not applied:
                    entry['regions'] = list(normalized_regions)
                    applied = True
                updated_groups[str(key)] = entry
            if not applied:
                updated_groups['default'] = {'regions': list(normalized_regions), 'cap': 'none'}
            config['carbon_cap_groups'] = updated_groups
        else:
            config['carbon_cap_groups'] = [{'name': 'default', 'regions': list(normalized_regions), 'cap': 'none'}]

    policy_frame = _build_policy_frame(
        config,
        years,
        bool(policy_enabled),
        ccr1_enabled=bool(ccr1_enabled),
        ccr2_enabled=bool(ccr2_enabled),
        control_period_years=control_period_years,
        banking_enabled=bool(allowance_banking_enabled),
    )
    frames_obj = frames_obj.with_frame('policy', policy_frame)

    policy_bank0 = 0.0
    if isinstance(policy_frame, pd.DataFrame) and not policy_frame.empty and "bank0" in policy_frame.columns:
        try:
            bank_series = pd.to_numeric(policy_frame["bank0"], errors="coerce")
        except Exception:  # pragma: no cover - defensive
            bank_series = None
        if bank_series is not None and not bank_series.empty:
            first_bank = bank_series.iloc[0]
            if pd.notna(first_bank):
                policy_bank0 = float(first_bank)

    # Apply config override if needed
    if bank0_from_config is not None:
        try:
            bank0_val = float(bank0_from_config)
        except Exception:  # pragma: no cover - defensive
            bank0_val = 0.0

        if banking_flag and bank0_val > 0.0 and policy_bank0 <= 0.0:
            policy_bank0 = bank0_val
        elif not policy_bank0:
            policy_bank0 = bank0_val


    runner = _ensure_engine_runner()
    supports_deep = True
    legacy_signature = False
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):  # pragma: no cover - builtin or C-accelerated callables
        supports_deep = True
    else:
        params = signature.parameters
        if "deep_carbon_pricing" in params:
            supports_deep = True
        else:
            has_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            )
            if has_var_kwargs:
                supports_deep = True
            else:
                supports_deep = False
                modern_keywords = {"tol", "max_iter", "relaxation", "price_cap"}
                legacy_signature = modern_keywords.issubset(params.keys())

    if legacy_signature and deep_carbon_pricing:
        if not _runner_supports_keyword(runner, "deep_carbon_pricing"):
            return {
                "error": (
                    "Deep carbon pricing requires an updated engine. "
                    "Please upgrade engine.run_loop.run_end_to_end_from_frames."
                )
            }
        supports_deep = True

    if not supports_deep:
        supports_deep = _runner_supports_keyword(runner, "deep_carbon_pricing")

    if not supports_deep and deep_carbon_pricing:
        return {"error": DEEP_CARBON_UNSUPPORTED_MESSAGE}

    enable_floor_flag = bool(policy_enabled and carbon_policy_cfg.enable_floor)
    enable_ccr_flag = bool(
        policy_enabled
        and carbon_policy_cfg.enable_ccr
        and (carbon_policy_cfg.ccr1_enabled or carbon_policy_cfg.ccr2_enabled)
    )
    if price_active:
        price_value_raw = (
            carbon_price_value
            if carbon_price_value is not None
            else price_cfg.price_per_ton
        )
        try:
            runner_price_value = float(price_value_raw)
        except (TypeError, ValueError):
            runner_price_value = float(price_cfg.price_per_ton)
    else:
        runner_price_value = 0.0

    runner_kwargs: dict[str, Any] = {
        "years": years,
        "price_initial": 0.0,
        "enable_floor": enable_floor_flag,
        "enable_ccr": enable_ccr_flag,
        "use_network": bool(dispatch_use_network),
        "capacity_expansion": bool(capacity_flag),
        "carbon_price_schedule": price_schedule_map if price_active else None,
        "carbon_price_value": runner_price_value,
        "deep_carbon_pricing": bool(deep_carbon_pricing),
        "progress_cb": progress_cb,
    }

    if _runner_supports_keyword(runner, "report_by_technology"):
        runner_kwargs["report_by_technology"] = True

    if not _runner_supports_keyword(runner, "capacity_expansion"):
        runner_kwargs.pop("capacity_expansion", None)

    if not _runner_supports_keyword(runner, "carbon_price_value"):
        runner_kwargs.pop("carbon_price_value", None)

    if not _runner_supports_keyword(runner, "deep_carbon_pricing"):
        if deep_carbon_pricing:
            return {
                "error": (
                    "Deep carbon pricing requires an updated engine. "
                    "Please upgrade engine.run_loop.run_end_to_end_from_frames."
                )
            }
        runner_kwargs.pop("deep_carbon_pricing", None)

    try:
        outputs = runner(frames_obj, **runner_kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Policy simulation failed")
        return {"error": str(exc)}

    if isinstance(outputs, Mapping):
        capacity_df = outputs.get("capacity_by_technology")
        generation_df = outputs.get("generation_by_technology")
    else:
        capacity_df = getattr(outputs, "capacity_by_technology", None)
        generation_df = getattr(outputs, "generation_by_technology", None)

    if capacity_df is None:
        LOGGER.warning(
            "Runner outputs missing capacity_by_technology frame; capacity charts will not be displayed."
        )
    if generation_df is None:
        LOGGER.warning(
            "Runner outputs missing generation_by_technology frame; generation charts will not be displayed."
        )


    temp_dir, csv_files = _write_outputs_to_temp(outputs)

    documentation = {
        "assumption_overrides": list(assumption_notes or []),
    }


    annual_df = _extract_output_dataframe(
        outputs, ['annual', 'annual_results', 'annual_output', 'annual_outputs']
    )
    emissions_df = _extract_output_dataframe(
        outputs, ['emissions_by_region', 'emissions', 'emissions_region']
    )
    raw_price_df = _extract_output_dataframe(
        outputs, ['price_by_region', 'dispatch_price_by_region', 'region_prices']
    )
    price_df, price_flags = _normalize_dispatch_price_frame(raw_price_df)
    flows_df = _extract_output_dataframe(
        outputs, ['flows', 'network_flows', 'flows_by_region']
    )

    if isinstance(annual_df, pd.DataFrame):
        annual_df = annual_df.copy()
        if not banking_flag:
            annual_df['bank'] = 0.0
        elif not annual_df.empty and {'allowances_available', 'emissions_tons'}.issubset(annual_df.columns):
            try:
                year_order = pd.to_numeric(annual_df['year'], errors='coerce')
            except Exception:  # pragma: no cover - defensive
                year_order = None
            if year_order is not None:
                ordered_index = year_order.sort_values(kind='mergesort').index
            else:
                ordered_index = annual_df.index
            bank_running = policy_bank0 if banking_flag else 0.0
            bank_values: dict[Any, float] = {}
            for idx in ordered_index:
                try:
                    allowances_total = float(annual_df.at[idx, 'allowances_available'])
                except Exception:
                    allowances_total = 0.0
                try:
                    emissions_value = float(annual_df.at[idx, 'emissions_tons'])
                except Exception:
                    emissions_value = 0.0
                bank_running = max(bank_running + allowances_total - emissions_value, 0.0)
                bank_values[idx] = bank_running
            annual_df['bank'] = annual_df.index.map(lambda idx: bank_values.get(idx, 0.0))

    if isinstance(csv_files, Mapping) and isinstance(annual_df, pd.DataFrame):
        try:
            csv_files = dict(csv_files)
            csv_files['annual.csv'] = annual_df.to_csv(index=False).encode('utf-8')
        except Exception:  # pragma: no cover - defensive
            pass

    result: dict[str, Any] = {
        'annual': annual_df,
        'emissions_by_region': emissions_df,
        'price_by_region': price_df,
        'flows': flows_df,
        'module_config': merged_modules,
        'config': config,
        'csv_files': csv_files,
        'temp_dir': temp_dir,
        'documentation': documentation,
    }
    result['_price_output_type'] = 'allowance' if policy_enabled else 'carbon'
    result['_price_field_flags'] = price_flags
    if normalized_regions:
        result['cap_regions'] = list(normalized_regions)

    optional_frames = {
        'capacity_by_technology': ['capacity_by_technology'],
        'generation_by_technology': ['generation_by_technology'],
    }
    for key, aliases in optional_frames.items():
        frame = _extract_output_dataframe(outputs, aliases)
        if isinstance(frame, pd.DataFrame):
            result[key] = frame

    return result

    # Carbon price config

      
def _extract_result_frame(
    result: Mapping[str, Any],
    key: str,
    *,
    csv_name: str | None = None,
) -> pd.DataFrame | None:
    """Return a DataFrame from `result` or load it from cached CSV bytes."""
    frame = result.get(key)
    if isinstance(frame, pd.DataFrame):
        return frame

    csv_files = result.get('csv_files')
    if isinstance(csv_files, Mapping):
        filename = csv_name or f'{key}.csv'
        raw = csv_files.get(filename)
        if isinstance(raw, (bytes, bytearray)):
            try:
                return pd.read_csv(io.BytesIO(raw))
            except Exception:  # pragma: no cover - defensive guard
                return None
    return None


def _render_technology_section(
    frame: pd.DataFrame | None,
    *,
    section_title: str,
    candidate_columns: list[tuple[str, str]],
) -> None:
    """Render charts summarising technology-level output data."""
    _ensure_streamlit()
    st.subheader(section_title)

    if frame is None or frame.empty:
        st.caption(f'{section_title} data not available for this run.')
        return

    if 'technology' not in frame.columns:
        st.caption('Technology detail unavailable; displaying raw data instead.')
        st.dataframe(frame, width="stretch")
        return

    value_col: str | None = None
    value_label = ''
    for column, label in candidate_columns:
        if column in frame.columns:
            value_col = column
            value_label = label
            break

    if value_col is None:
        numeric_cols = frame.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
            value_label = numeric_cols[0]
        else:
            st.caption('No numeric values available to chart; showing raw data.')
            st.dataframe(frame, width="stretch")
            return

    display_frame = frame.copy()
    if 'year' in display_frame.columns:
        display_frame['year'] = pd.to_numeric(display_frame['year'], errors='coerce')
        display_frame = display_frame.dropna(subset=['year'])

    if display_frame.empty:
        st.caption('No valid year entries available; showing raw data.')
        st.dataframe(frame, width="stretch")
        return

    display_frame = display_frame.sort_values(['year', 'technology'])
    pivot = display_frame.pivot_table(
        index='year',
        columns='technology',
        values=value_col,
        aggfunc='sum',
    )

    if pivot.empty:
        st.caption('No data available to chart; showing raw data.')
        st.dataframe(frame, width="stretch")
        return

    st.line_chart(pivot)

    latest_year = pivot.index.max()
    latest_totals = pivot.loc[latest_year].fillna(0.0)
    latest_df = latest_totals.to_frame(name=value_label)
    latest_df.index.name = 'technology'
    st.caption(f'Latest year visualised: {latest_year}')
    st.bar_chart(latest_df)


def _cleanup_session_temp_dirs() -> None:
    _ensure_streamlit()
    temp_dirs = st.session_state.get('temp_dirs', [])
    for path_str in temp_dirs:
        try:
            shutil.rmtree(path_str, ignore_errors=True)
        except Exception:  # pragma: no cover - best effort cleanup
            continue
    st.session_state['temp_dirs'] = []


def _reset_run_state_on_reload() -> None:
    try:
        _ensure_streamlit()
    except ModuleNotFoundError:  # pragma: no cover - GUI dependency missing
        return

    previous_token = st.session_state.get(_SESSION_RUN_TOKEN_KEY)
    if previous_token != _CURRENT_SESSION_RUN_TOKEN:
        st.session_state[_SESSION_RUN_TOKEN_KEY] = _CURRENT_SESSION_RUN_TOKEN
        st.session_state['run_in_progress'] = False
        st.session_state.pop('pending_run', None)


def _advance_script_iteration() -> int:
    """Increment and return the current Streamlit rerun iteration counter."""

    try:
        _ensure_streamlit()
    except ModuleNotFoundError:  # pragma: no cover - GUI dependency missing
        return 0

    current = int(st.session_state.get(_SCRIPT_ITERATION_KEY, 0)) + 1
    st.session_state[_SCRIPT_ITERATION_KEY] = current
    return current


def _recover_stuck_run_state(current_iteration: int) -> None:
    """Clear stale run state flags left behind by interrupted executions."""

    try:
        _ensure_streamlit()
    except ModuleNotFoundError:  # pragma: no cover - GUI dependency missing
        return

    if not st.session_state.get('run_in_progress'):
        st.session_state.pop(_ACTIVE_RUN_ITERATION_KEY, None)
        return

    active_iteration = st.session_state.get(_ACTIVE_RUN_ITERATION_KEY)
    stale_state = not isinstance(active_iteration, int) or active_iteration < current_iteration
    if stale_state:
        LOGGER.warning('Detected stale run_in_progress flag; resetting run state')
        st.session_state['run_in_progress'] = False
        st.session_state.pop('pending_run', None)
        st.session_state.pop(_ACTIVE_RUN_ITERATION_KEY, None)


def _build_run_summary(settings: Mapping[str, Any], *, config_label: str) -> list[tuple[str, str]]:
    """Return human-readable configuration details for confirmation dialogs."""

    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _bool_label(value: bool) -> str:
        return "Yes" if value else "No"

    start_year = _as_int(settings.get("start_year"))
    end_year = _as_int(settings.get("end_year"))

    if start_year is None and end_year is None:
        year_display = "Not specified"
    else:
        if start_year is None:
            start_year = end_year
        if end_year is None:
            end_year = start_year
        if start_year == end_year:
            year_display = f"{start_year}"
        else:
            year_display = f"{start_year} – {end_year}"

    carbon_enabled = bool(settings.get("carbon_policy_enabled", True))
    enable_floor = bool(settings.get("enable_floor", False)) if carbon_enabled else False
    enable_ccr = bool(settings.get("enable_ccr", False)) if carbon_enabled else False
    ccr1_enabled = bool(settings.get("ccr1_enabled", False)) if enable_ccr else False
    ccr2_enabled = bool(settings.get("ccr2_enabled", False)) if enable_ccr else False
    banking_enabled = (
        bool(settings.get("allowance_banking_enabled", False)) if carbon_enabled else False
    )

    control_period = settings.get("control_period_years") if carbon_enabled else None
    if not carbon_enabled:
        control_display = "Not applicable"
    elif control_period is None:
        control_display = "Automatic"
    else:
        control_display = str(control_period)

    price_enabled = bool(settings.get("carbon_price_enabled", False)) if carbon_enabled else False
    price_value = _as_float(settings.get("carbon_price_value")) if price_enabled else None
    price_schedule_raw = settings.get("carbon_price_schedule") if price_enabled else None
    schedule_entries: list[tuple[int, float]] = []
    if isinstance(price_schedule_raw, Mapping):
        for year_key, value in price_schedule_raw.items():
            year_val = _as_int(year_key)
            price_val = _as_float(value)
            if year_val is None or price_val is None:
                continue
            schedule_entries.append((year_val, price_val))
    schedule_entries.sort(key=lambda item: item[0])

    if not price_enabled:
        price_display = "Disabled"
    elif schedule_entries:
        first_year, first_price = schedule_entries[0]
        if len(schedule_entries) == 1:
            price_display = f"Schedule: {first_year} → ${first_price:,.2f}/ton"
        else:
            last_year, last_price = schedule_entries[-1]
            price_display = (
                f"Schedule ({len(schedule_entries)} entries): "
                f"{first_year} → ${first_price:,.2f}/ton, "
                f"{last_year} → ${last_price:,.2f}/ton"
            )
    elif price_value is not None:
        price_display = f"Flat ${price_value:,.2f}/ton"
    else:
        price_display = "Enabled (no price specified)"

    dispatch_network = bool(settings.get("dispatch_use_network", False))

    return [
        ("Configuration", config_label),
        ("Simulation years", year_display),
        ("Carbon cap enabled", _bool_label(carbon_enabled)),
        ("Minimum reserve price", _bool_label(enable_floor)),
        ("CCR enabled", _bool_label(enable_ccr)),
        ("CCR tranche 1", _bool_label(ccr1_enabled)),
        ("CCR tranche 2", _bool_label(ccr2_enabled)),
        ("Allowance banking enabled", _bool_label(banking_enabled)),
        ("Control period length", control_display),
        ("Carbon price", price_display),
        ("Dispatch uses network", _bool_label(dispatch_network)),
    ]



def _render_results(result: Mapping[str, Any]) -> None:
    """Render charts and tables summarising the latest run results."""
    _ensure_streamlit()

    if 'error' in result:
        st.error(result['error'])
        return

    annual = result.get('annual')
    if not isinstance(annual, pd.DataFrame):
        annual = pd.DataFrame()

    display_annual = annual.copy()
    chart_data = pd.DataFrame()
    if not display_annual.empty and 'year' in display_annual.columns:
        display_annual['year'] = pd.to_numeric(display_annual['year'], errors='coerce')
        display_annual = display_annual.dropna(subset=['year'])
        display_annual = display_annual.sort_values('year')
        chart_data = display_annual.set_index('year')
    elif not display_annual.empty:
        chart_data = display_annual.copy()

    price_output_type = str(result.get('_price_output_type') or 'allowance')

    if price_output_type == 'carbon':
        carbon_field_aliases = {
            'allowance_price': 'p_co2_all',
            'allowance_price_exogenous_component': 'p_co2_exc',
            'allowance_price_effective': 'p_co2_eff',
        }

        if not display_annual.empty:
            display_annual = display_annual.rename(columns=carbon_field_aliases)
            if 'p_co2' not in display_annual.columns and 'p_co2_all' in display_annual.columns:
                display_annual['p_co2'] = display_annual['p_co2_all']

        if not chart_data.empty:
            chart_data = chart_data.rename(columns=carbon_field_aliases)
            if 'p_co2' not in chart_data.columns and 'p_co2_all' in chart_data.columns:
                chart_data['p_co2'] = chart_data['p_co2_all']

    if price_output_type == 'carbon':
        price_tab_label = 'Carbon price'
        price_section_title = 'Carbon price results'
        price_series_label = 'Carbon price ($/ton)'
        price_missing_caption = 'Carbon price data unavailable for this run.'
    else:
        price_tab_label = 'Allowance price'
        price_section_title = 'Allowance market results'
        price_series_label = 'Allowance clearing price ($/ton)'
        price_missing_caption = 'Allowance clearing price data unavailable for this run.'

    price_chart_column: str | None = None
    if not chart_data.empty and 'p_co2' in chart_data.columns:
        chart_data = chart_data.rename(columns={'p_co2': price_series_label})
        price_chart_column = price_series_label

    display_price_table = display_annual.copy()

    if price_output_type == 'carbon':
        if 'p_co2' in display_price_table.columns:
            display_price_table = display_price_table.rename(columns={'p_co2': price_series_label})
        else:
            rename_map = {'allowance_price': price_series_label}
            if 'allowance_price_exogenous_component' in display_price_table.columns:
                rename_map['allowance_price_exogenous_component'] = 'p_co2_exc'
            if 'allowance_price_effective' in display_price_table.columns:
                rename_map['allowance_price_effective'] = 'p_co2_eff'
            display_price_table = display_price_table.rename(columns=rename_map)

        if price_series_label in display_price_table.columns and 'p_co2_all' not in display_price_table.columns:
            display_price_table['p_co2_all'] = display_price_table[price_series_label]

        allowed_price_columns = [
            'year',
            price_series_label,
            'p_co2',
            'p_co2_all',
            'p_co2_exc',
            'p_co2_eff',
            'emissions_tons',
        ]
        display_price_table = display_price_table.filter(items=allowed_price_columns)
    else:
        # Allowance output path
        if 'p_co2' in display_price_table.columns:
            display_price_table = display_price_table.rename(
                columns={'p_co2': price_series_label}
            )


    emissions_df = result.get('emissions_by_region')
    if not isinstance(emissions_df, pd.DataFrame):
        emissions_df = pd.DataFrame()

    price_df = result.get('price_by_region')
    if not isinstance(price_df, pd.DataFrame):
        price_df = pd.DataFrame()
        price_flags = {'year': False, 'region': False, 'price': False}
    else:
        price_df = price_df.copy()
        price_flags = result.get(
            '_price_field_flags', {'year': True, 'region': True, 'price': True}
        )

    flows_df = result.get('flows')
    if not isinstance(flows_df, pd.DataFrame):
        flows_df = pd.DataFrame()

    st.caption('Visualisations reflect the most recent model run.')

    show_bank_tab = price_output_type != 'carbon'
    tab_labels = [price_tab_label, 'Emissions']
    if show_bank_tab:
        tab_labels.append('Allowance bank')
    tab_labels.append('Dispatch costs')

    tabs = st.tabs(tab_labels)
    tab_iter = iter(tabs)
    price_tab = next(tab_iter)
    emissions_tab = next(tab_iter)
    bank_tab = next(tab_iter) if show_bank_tab else None
    dispatch_tab = next(tab_iter)

    with price_tab:
        st.subheader(price_section_title)
        if display_annual.empty:
            st.info('No annual results to display.')
        else:
            if price_chart_column and price_chart_column in chart_data.columns:
                st.markdown(f'**{price_series_label}**')
                st.line_chart(chart_data[[price_chart_column]])
            else:
                st.caption(price_missing_caption)

            st.markdown('---')
            st.dataframe(display_price_table, width="stretch")

    with emissions_tab:
        st.subheader('Emissions overview')
        if display_annual.empty and emissions_df.empty:
            st.info('No emissions data available for this run.')
        else:
            if not chart_data.empty and 'emissions_tons' in chart_data.columns:
                st.markdown('**Total emissions (tons)**')
                st.line_chart(chart_data[['emissions_tons']])
                st.bar_chart(chart_data[['emissions_tons']])
            elif not display_annual.empty:
                st.caption('Total emissions data unavailable for this run.')

            if not emissions_df.empty:
                display_emissions = emissions_df.copy()
                display_emissions['year'] = pd.to_numeric(
                    display_emissions['year'], errors='coerce'
                )
                display_emissions = display_emissions.dropna(subset=['year'])

                if 'region' in display_emissions.columns:
                    emissions_pivot = display_emissions.pivot_table(
                        index='year',
                        columns='region',
                        values='emissions_tons',
                        aggfunc='sum',
                    ).sort_index()
                    st.markdown('**Emissions by region**')
                    st.line_chart(emissions_pivot)

                    if not emissions_pivot.empty:
                        latest_year = emissions_pivot.index.max()
                        latest_totals = emissions_pivot.loc[latest_year].fillna(0.0)
                        latest_df = latest_totals.to_frame(name='emissions_tons')
                        latest_df.index.name = 'region'
                        st.caption(f'Latest year visualised: {latest_year}')
                        st.bar_chart(latest_df)
                else:
                    st.caption('Regional emissions data unavailable; showing raw table below.')
                    st.dataframe(display_emissions, width="stretch")
            elif not display_annual.empty:
                st.caption('No regional emissions data available for this run.')

    if bank_tab is not None:
        with bank_tab:
            st.subheader('Allowance bank balance')
            if display_annual.empty:
                st.info('No annual results to display.')
            elif 'bank' in chart_data.columns:
                st.markdown('**Bank balance (tons)**')
                st.line_chart(chart_data[['bank']])
                st.bar_chart(chart_data[['bank']])
            else:
                st.caption('Allowance bank data unavailable for this run.')

    with dispatch_tab:
        st.subheader('Dispatch costs and network results')
        if price_df.empty and flows_df.empty:
            st.info('No dispatch outputs are available for this run.')
        else:
            if not price_df.empty:
                if all(price_flags.get(key, False) for key in ('year', 'region', 'price')):
                    display_price = price_df.copy()
                    display_price['year'] = pd.to_numeric(
                        display_price['year'], errors='coerce'
                    )
                    display_price = display_price.dropna(subset=['year'])

                    if 'region' in display_price.columns:
                        price_pivot = display_price.pivot_table(
                            index='year',
                            columns='region',
                            values='price',
                            aggfunc='mean',
                        ).sort_index()
                        st.markdown('**Dispatch costs by region ($/MWh)**')
                        st.line_chart(price_pivot)

                        if not price_pivot.empty:
                            latest_year = price_pivot.index.max()
                            latest_totals = price_pivot.loc[latest_year].fillna(0.0)
                            latest_df = latest_totals.to_frame(name='price')
                            latest_df.index.name = 'region'
                            st.caption(f'Latest year visualised: {latest_year}')
                            st.bar_chart(latest_df)
                    else:
                        st.caption(
                            'Regional dispatch cost data unavailable; showing raw table below.'
                        )
                        st.dataframe(display_price, width="stretch")
                else:
                    missing = [key for key, present in price_flags.items() if not present]
                    if missing:
                        missing_display = ', '.join(sorted(missing))
                        st.caption(
                            'Dispatch price data missing expected column(s): '
                            f"{missing_display}. Showing available data below."
                        )
                    st.dataframe(price_df, width="stretch")

            if not flows_df.empty:
                st.markdown('---')
                st.markdown('**Interregional energy flows (MWh)**')
                st.dataframe(flows_df, width="stretch")
            elif price_df.empty:
                st.caption('No dispatch network data available for this run.')

    # --- Technology sections ---
    capacity_df = _extract_result_frame(result, 'capacity_by_technology')
    _render_technology_section(
        capacity_df,
        section_title='Capacity by technology',
        candidate_columns=[
            ('capacity_mw', 'Capacity (MW)'),
            ('capacity', 'Capacity'),
            ('value', 'Capacity'),
        ],
    )

    generation_df = _extract_result_frame(result, 'generation_by_technology')
    _render_technology_section(
        generation_df,
        section_title='Generation by technology',
        candidate_columns=[
            ('generation_mwh', 'Generation (MWh)'),
            ('generation', 'Generation'),
            ('value', 'Generation'),
        ],
    )

    # --- Assumption overrides ---
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

    # --- Downloads ---
    st.subheader('Download outputs')
    csv_files = result.get('csv_files')
    if isinstance(csv_files, Mapping) and csv_files:
        for filename, content in sorted(csv_files.items()):
            st.download_button(
                label=f'Download {filename}',
                data=content,
                file_name=filename,
                mime='text/csv',
            )
    else:
        st.caption('No CSV outputs are available for download.')

    temp_dir = result.get('temp_dir')
    if temp_dir:
        st.caption(f'Temporary files saved to {temp_dir}')


def _render_outputs_panel(last_result: Mapping[str, Any] | None) -> None:
    """Render the main outputs panel with charts for the latest run."""
    _ensure_streamlit()
    if not isinstance(last_result, Mapping) or not last_result:
        st.caption('Run the model to populate this panel with results.')
        return
    _render_results(last_result)


def main() -> None:
    """Streamlit entry point."""
    _ensure_streamlit()
    st.set_page_config(page_title='Granite Ledger Policy Simulator', layout='wide')
    st.title('Granite Ledger Policy Simulator')
    st.write('Upload a run configuration and execute the annual allowance market engine.')
    st.session_state.setdefault('last_result', None)
    st.session_state.setdefault('temp_dirs', [])
    st.session_state.setdefault('run_in_progress', False)
    current_iteration = _advance_script_iteration()
    _reset_run_state_on_reload()
    _recover_stuck_run_state(current_iteration)

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
    candidate_years: list[int] = []
    frames_for_run: FramesType | None = None
    current_year = date.today().year
    start_year_val = int(run_config.get('start_year', current_year)) if run_config else int(current_year)
    default_end_year = start_year_val + 1
    end_year_val = int(run_config.get('end_year', default_end_year)) if run_config else int(default_end_year)
    if end_year_val <= start_year_val:
        end_year_val = start_year_val + 1

    carbon_settings = CarbonModuleSettings(
        enabled=False,
        price_enabled=False,
        enable_floor=False,
        enable_ccr=False,
        ccr1_enabled=False,
        ccr2_enabled=False,
        ccr1_price=None,
        ccr2_price=None,
        ccr1_escalator_pct=0.0,
        ccr2_escalator_pct=0.0,
        banking_enabled=False,
        coverage_regions=["All"],
        control_period_years=None,
        price_per_ton=0.0,
        price_escalator_pct=0.0,
        initial_bank=0.0,
        cap_regions=[],
        cap_start_value=None,
        cap_reduction_mode="percent",
        cap_reduction_value=0.0,
        cap_schedule={},
        floor_value=0.0,
        floor_escalator_mode="fixed",
        floor_escalator_value=0.0,
        floor_schedule={},
        price_schedule={},
        errors=[],
    )


    dispatch_settings = DispatchModuleSettings(
        enabled=False,
        mode='single',
        capacity_expansion=False,
        reserve_margins=False,
        deep_carbon_pricing=False,
    )
    incentives_settings = IncentivesModuleSettings(
        enabled=False,
        production_credits=[],
        investment_credits=[],
    )
    output_directory_raw = run_config.get('output_name') if run_config else None
    output_directory = str(output_directory_raw) if output_directory_raw else 'outputs'
    downloads_root = get_downloads_directory()
    resolved_output_path = downloads_root if not output_directory else downloads_root / output_directory
    outputs_settings = OutputsModuleSettings(
        enabled=False,
        directory=output_directory,
        resolved_path=resolved_output_path,
        show_csv_downloads=False,
    )
    run_clicked = False
    run_in_progress = False

    
    
    with st.sidebar:
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

        last_result_mapping = st.session_state.get("last_result")
        if not isinstance(last_result_mapping, Mapping):
            last_result_mapping = None

        (inputs_tab,) = st.tabs(["Inputs"])

        with inputs_tab:
            # -------- General --------
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

            if start_year_val >= end_year_val:
                year_error = "Simulation end year must be greater than start year."
                if year_error not in module_errors:
                    module_errors.append(year_error)

            # -------- Carbon --------
            carbon_label, carbon_expanded = SIDEBAR_SECTIONS[1]
            carbon_expander = st.expander(carbon_label, expanded=carbon_expanded)
            carbon_settings = _render_carbon_policy_section(
                carbon_expander,
                run_config,
                region_options=general_result.regions,
                lock_inputs=general_result.lock_carbon_controls,
            )
            module_errors.extend(carbon_settings.errors)

            # Prepare default frames (defensive)
            try:
                frames_for_run = _build_default_frames(
                    selected_years
                    or list(range(int(start_year_val), int(end_year_val) + 1)),
                    carbon_policy_enabled=bool(
                        carbon_settings.enabled and not carbon_settings.price_enabled
                    ),
                    banking_enabled=bool(carbon_settings.banking_enabled),
                    carbon_price_schedule=(
                        carbon_settings.price_schedule if carbon_settings.price_enabled else None
                    ),
                )
            except Exception as exc:  # pragma: no cover
                frames_for_run = None
                st.warning(f"Unable to prepare default assumption tables: {exc}")

            # -------- Dispatch --------
            dispatch_label, dispatch_expanded = SIDEBAR_SECTIONS[2]
            dispatch_expander = st.expander(dispatch_label, expanded=dispatch_expanded)
            dispatch_settings = _render_dispatch_section(
                dispatch_expander, run_config, frames_for_run
            )
            module_errors.extend(dispatch_settings.errors)

            # -------- Incentives --------
            incentives_label, incentives_expanded = SIDEBAR_SECTIONS[3]
            incentives_expander = st.expander(incentives_label, expanded=incentives_expanded)
            incentives_settings = _render_incentives_section(
                incentives_expander,
                run_config,
                frames_for_run,
            )
            module_errors.extend(incentives_settings.errors)

            # -------- Outputs --------
            outputs_label, outputs_expanded = SIDEBAR_SECTIONS[4]
            outputs_expander = st.expander(outputs_label, expanded=outputs_expanded)
            outputs_settings = _render_outputs_section(
                outputs_expander,
                run_config,
                last_result_mapping,
            )
            module_errors.extend(outputs_settings.errors)

            # -------- Assumptions --------
            st.divider()
            inputs_header = st.container()
            inputs_header.subheader("Assumption overrides")
            inputs_header.caption(
                "Adjust core assumption tables or upload CSV files to override the defaults."
            )
            if frames_for_run is not None:
                demand_tab, units_tab, fuels_tab, transmission_tab = st.tabs(
                    ["Demand", "Units", "Fuels", "Transmission"]
                )
                with demand_tab:
                    frames_for_run, notes, errors = _render_demand_controls(
                        frames_for_run, selected_years
                    )
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
                    st.warning(
                        "Resolve the highlighted assumption issues before running the simulation."
                    )
            else:
                st.info(
                    "Default assumption tables are unavailable due to a previous error. "
                    "Resolve the issue above to edit inputs through the GUI."
                )

            run_clicked = st.button("Run Model", type="primary", use_container_width=True)

    # Finalize selected years defensively
    if st is not None:
        try:
            start_year_state = int(st.session_state.get("start_year_slider", start_year_val))
        except (TypeError, ValueError):
            start_year_state = int(start_year_val)
        try:
            end_year_state = int(st.session_state.get("end_year_slider", end_year_val))
        except (TypeError, ValueError):
            end_year_state = int(end_year_val)
        start_year_val = start_year_state
        end_year_val = end_year_state

    default_years: list[int] = []
    if start_year_val < end_year_val:
        default_years = list(range(int(start_year_val), int(end_year_val) + 1))
    elif start_year_val == end_year_val:
        default_years = [int(start_year_val)]

    try:
        selected_years = _select_years(candidate_years, start_year_val, end_year_val)
    except Exception:
        selected_years = []

    if selected_years:
        try:
            selected_min = min(int(year) for year in selected_years)
            selected_max = max(int(year) for year in selected_years)
        except ValueError:
            selected_years = []
        else:
            selected_years = list(range(selected_min, selected_max + 1))
    else:
        selected_years = list(default_years)

    # Ensure frames if earlier failed
    if frames_for_run is None:
        try:
            frames_for_run = _build_default_frames(
                selected_years
                or default_years
                or [int(start_year_val)],
                carbon_policy_enabled=bool(
                    carbon_settings.enabled and not carbon_settings.price_enabled
                ),
                banking_enabled=bool(carbon_settings.banking_enabled),
                carbon_price_schedule=(
                    carbon_settings.price_schedule if carbon_settings.price_enabled else None
                ),
            )
        except Exception as exc:  # pragma: no cover
            frames_for_run = None
            st.warning(f"Unable to prepare default assumption tables: {exc}")

    if module_errors:
        st.warning(
            "Resolve the module configuration issues highlighted in the sidebar before running the simulation."
        )

    # ---- Run orchestration state ----
    execute_run = False
    run_inputs: dict[str, Any] | None = None

    run_in_progress = bool(st.session_state.get("run_in_progress"))

    def _collect_run_blocking_errors() -> list[str]:
        blocking: list[str] = []
        for message in itertools.chain(assumption_errors, module_errors):
            if not message:
                continue
            text = str(message).strip()
            if text and text not in blocking:
                blocking.append(text)
        return blocking

    # Build the payload that actually drives the engine
    dispatch_use_network = bool(
        dispatch_settings.enabled and dispatch_settings.mode == "network"
    )
    dispatch_deep_carbon = bool(dispatch_settings.deep_carbon_pricing)

    current_run_payload: dict[str, Any] = {
        "config_source": copy.deepcopy(run_config),
        "start_year": int(start_year_val),
        "end_year": int(end_year_val),
        "carbon_policy_enabled": bool(carbon_settings.enabled),
        "enable_floor": bool(carbon_settings.enable_floor),
        "price_floor_value": float(carbon_settings.floor_value),
        "price_floor_escalator_mode": str(carbon_settings.floor_escalator_mode),
        "price_floor_escalator_value": float(carbon_settings.floor_escalator_value),
        "price_floor_schedule": dict(carbon_settings.floor_schedule),
        "enable_ccr": bool(carbon_settings.enable_ccr),
        "ccr1_enabled": bool(carbon_settings.ccr1_enabled),
        "ccr2_enabled": bool(carbon_settings.ccr2_enabled),
        "ccr1_price": float(carbon_settings.ccr1_price)
        if carbon_settings.ccr1_price is not None
        else None,
        "ccr2_price": float(carbon_settings.ccr2_price)
        if carbon_settings.ccr2_price is not None
        else None,
        "ccr1_escalator_pct": float(carbon_settings.ccr1_escalator_pct),
        "ccr2_escalator_pct": float(carbon_settings.ccr2_escalator_pct),
        "allowance_banking_enabled": bool(carbon_settings.banking_enabled),
        "coverage_regions": list(carbon_settings.coverage_regions),
        "cap_regions": list(getattr(carbon_settings, "cap_regions", [])),
        "initial_bank": float(carbon_settings.initial_bank),
        "control_period_years": carbon_settings.control_period_years,
        "carbon_price_enabled": bool(carbon_settings.price_enabled),
        "carbon_price_value": float(carbon_settings.price_per_ton)
        if carbon_settings.price_enabled
        else 0.0,
        "carbon_price_escalator_pct": float(carbon_settings.price_escalator_pct),
        "carbon_price_schedule": (
            dict(carbon_settings.price_schedule) if carbon_settings.price_enabled else {}
        ),
        "carbon_cap_start_value": (
            float(carbon_settings.cap_start_value)
            if carbon_settings.cap_start_value is not None
            else None
        ),
        "carbon_cap_reduction_mode": str(carbon_settings.cap_reduction_mode),
        "carbon_cap_reduction_value": float(carbon_settings.cap_reduction_value),
        "carbon_cap_schedule": dict(carbon_settings.cap_schedule),
        "dispatch_use_network": dispatch_use_network,
        "dispatch_capacity_expansion": bool(
            getattr(dispatch_settings, "capacity_expansion", True)
        ),
        "dispatch_deep_carbon": bool(
            getattr(dispatch_settings, "deep_carbon_pricing", False)
        ),
        "module_config": copy.deepcopy(run_config.get("modules", {})),
        "frames": frames_for_run,
        "assumption_notes": list(assumption_notes),
    }




    def _clone_run_payload(source: Mapping[str, Any]) -> dict[str, Any]:
        base = {k: v for k, v in source.items() if k != "frames"}
        try:
            cloned = copy.deepcopy(base)
        except Exception:  # pragma: no cover
            cloned = dict(base)
        cloned["frames"] = source.get("frames")
        return cloned

    # Handle Run button -> validate and immediately execute
    if run_clicked:
        if run_in_progress:
            st.info("A simulation is already in progress. Wait for it to finish before starting another run.")
        else:
            blocking = _collect_run_blocking_errors()
            if blocking:
                st.error("Resolve the configuration issues above before running the simulation.")
                st.session_state["run_blocking_errors"] = blocking
                st.session_state["run_in_progress"] = False
            else:
                # Transition to execution immediately when the button is clicked
                run_inputs = _clone_run_payload(current_run_payload)
                execute_run = True
                st.session_state.pop("run_blocking_errors", None)
                st.session_state["run_in_progress"] = True
                st.session_state[_ACTIVE_RUN_ITERATION_KEY] = st.session_state.get(
                    _ACTIVE_RUN_ITERATION_KEY, 0
                )
    # Sync dispatch flag for downstream logic
    dispatch_use_network = bool(
        dispatch_settings.enabled and dispatch_settings.mode == "network"
    )
    dispatch_deep_carbon = bool(dispatch_settings.deep_carbon_pricing)

    # Allow downstream to honor confirmed inputs immediately
    if run_inputs is not None:
        run_config = copy.deepcopy(run_inputs.get("config_source", run_config))
        start_year_val = int(run_inputs.get("start_year", start_year_val))
        end_year_val = int(run_inputs.get("end_year", end_year_val))
        dispatch_use_network = bool(
            run_inputs.get("dispatch_use_network", dispatch_use_network)
        )
        if "dispatch_capacity_expansion" in run_inputs:
            dispatch_settings.capacity_expansion = bool(
                run_inputs.get("dispatch_capacity_expansion")
            )
        if "dispatch_deep_carbon" in run_inputs:
            dispatch_settings.deep_carbon_pricing = bool(
                run_inputs.get(
                    "dispatch_deep_carbon",
                    getattr(dispatch_settings, "deep_carbon_pricing", False),
                )
            )

    result = st.session_state.get("last_result")



    # Outputs/progress scaffolding (widgets filled later)
    result = st.session_state.get("last_result")
    inputs_for_run: Mapping[str, Any] = run_inputs or {}
    run_result: Mapping[str, Any] | None = None

    progress_state = _ensure_progress_state()
    progress_section = st.container()
    with progress_section:
        st.subheader("Run progress")
        progress_message_placeholder = st.empty()
        progress_bar_placeholder = st.empty()
        progress_log_placeholder = st.empty()

    _sync_progress_ui(
        progress_state,
        progress_message_placeholder,
        progress_bar_placeholder,
        progress_log_placeholder,
    )

    # --- Execution branch ---
    if execute_run:
        frames_for_execution = inputs_for_run.get("frames", frames_for_run)
        if frames_for_execution is None:
            frames_for_execution = frames_for_run

        # Normalize assumption notes
        assumption_notes_value = inputs_for_run.get("assumption_notes", assumption_notes)
        assumption_notes_for_run: list[str] = []
        if isinstance(assumption_notes_value, Iterable) and not isinstance(
            assumption_notes_value, (str, bytes, Mapping)
        ):
            assumption_notes_for_run = [str(note) for note in assumption_notes_value]
        elif assumption_notes_value not in (None, ""):
            assumption_notes_for_run = [str(assumption_notes_value)]

        try:
            st.session_state["run_in_progress"] = True
            st.session_state[_ACTIVE_RUN_ITERATION_KEY] = st.session_state.get(
                _ACTIVE_RUN_ITERATION_KEY, 0
            )
            _cleanup_session_temp_dirs()

            progress_state = _reset_progress_state()
            progress_state.stage = "initializing"
            progress_state.message = "Initializing simulation…"
            progress_state.percent_complete = 0
            _record_progress_log(progress_state, progress_state.message, progress_state.stage)
            _sync_progress_ui(
                progress_state,
                progress_message_placeholder,
                progress_bar_placeholder,
                progress_log_placeholder,
            )

            def _update_progress(stage: str, payload: Mapping[str, object]) -> None:
                try:
                    message, percent = _progress_update_from_stage(
                        stage, payload, progress_state
                    )
                except Exception:
                    LOGGER.exception("Unable to interpret progress update for stage %s", stage)
                    return
                progress_state.stage = stage
                progress_state.message = message
                progress_state.percent_complete = percent
                _record_progress_log(progress_state, message, stage)
                _sync_progress_ui(
                    progress_state,
                    progress_message_placeholder,
                    progress_bar_placeholder,
                    progress_log_placeholder,
                )

            try:
                run_result = run_policy_simulation(
                    inputs_for_run.get("config_source", run_config),
                    start_year=inputs_for_run.get("start_year", start_year_val),
                    end_year=inputs_for_run.get("end_year", end_year_val),
                    carbon_policy_enabled=bool(
                        inputs_for_run.get("carbon_policy_enabled", carbon_settings.enabled)
                    ),
                    enable_floor=bool(
                        inputs_for_run.get("enable_floor", carbon_settings.enable_floor)
                    ),
                    enable_ccr=bool(inputs_for_run.get("enable_ccr", carbon_settings.enable_ccr)),
                    ccr1_enabled=bool(
                        inputs_for_run.get("ccr1_enabled", carbon_settings.ccr1_enabled)
                    ),
                    ccr2_enabled=bool(
                        inputs_for_run.get("ccr2_enabled", carbon_settings.ccr2_enabled)
                    ),
                    ccr1_price=inputs_for_run.get("ccr1_price", carbon_settings.ccr1_price),
                    ccr2_price=inputs_for_run.get("ccr2_price", carbon_settings.ccr2_price),
                    ccr1_escalator_pct=inputs_for_run.get(
                        "ccr1_escalator_pct", carbon_settings.ccr1_escalator_pct
                    ),
                    ccr2_escalator_pct=inputs_for_run.get(
                        "ccr2_escalator_pct", carbon_settings.ccr2_escalator_pct
                    ),
                    allowance_banking_enabled=bool(
                        inputs_for_run.get(
                            "allowance_banking_enabled", carbon_settings.banking_enabled
                        )
                    ),
                    initial_bank=float(
                        inputs_for_run.get("initial_bank", carbon_settings.initial_bank)
                    ),
                    coverage_regions=inputs_for_run.get(
                        "coverage_regions", carbon_settings.coverage_regions
                    ),
                    control_period_years=inputs_for_run.get(
                        "control_period_years", carbon_settings.control_period_years
                    ),
                    price_floor_value=inputs_for_run.get(
                        "price_floor_value", carbon_settings.floor_value
                    ),
                    price_floor_escalator_mode=inputs_for_run.get(
                        "price_floor_escalator_mode", carbon_settings.floor_escalator_mode
                    ),
                    price_floor_escalator_value=inputs_for_run.get(
                        "price_floor_escalator_value", carbon_settings.floor_escalator_value
                    ),
                    price_floor_schedule=inputs_for_run.get(
                        "price_floor_schedule", carbon_settings.floor_schedule
                    ),
                    cap_regions=inputs_for_run.get(
                        "cap_regions", getattr(carbon_settings, "cap_regions", [])
                    ),
                    carbon_price_enabled=inputs_for_run.get(
                        "carbon_price_enabled", carbon_settings.price_enabled
                    ),
                    carbon_price_value=inputs_for_run.get(
                        "carbon_price_value", carbon_settings.price_per_ton
                    ),
                    carbon_price_schedule=inputs_for_run.get(
                        "carbon_price_schedule", carbon_settings.price_schedule
                    ),
                    dispatch_use_network=bool(
                        inputs_for_run.get("dispatch_use_network", dispatch_use_network)
                    ),
                    dispatch_capacity_expansion=inputs_for_run.get(
                        "dispatch_capacity_expansion",
                        getattr(dispatch_settings, "capacity_expansion", True),
                    ),
                    deep_carbon_pricing=bool(
                        inputs_for_run.get(
                            "dispatch_deep_carbon",
                            getattr(dispatch_settings, "deep_carbon_pricing", False),
                        )
                    ),
                    module_config=inputs_for_run.get(
                        "module_config", run_config.get("modules", {})
                    ),
                    frames=frames_for_execution,
                    assumption_notes=assumption_notes_for_run,
                    progress_cb=_update_progress,
                )

            except Exception as exc:  # defensive guard
                LOGGER.exception("Policy simulation failed during execution")
                run_result = {"error": str(exc)}

        except Exception as exc:  # defensive guard
            LOGGER.exception("Policy simulation failed before execution could complete")
            run_result = {"error": str(exc)}

        finally:
            st.session_state["run_in_progress"] = False
            st.session_state.pop(_ACTIVE_RUN_ITERATION_KEY, None)

            if isinstance(run_result, Mapping):
                if "error" in run_result:
                    progress_state.stage = "error"
                    progress_state.message = f"Simulation failed: {run_result['error']}"
                else:
                    progress_state.stage = "complete"
                    progress_state.percent_complete = 100
                    progress_state.message = "Simulation complete. Outputs updated below."
                    st.session_state["last_result"] = run_result
            else:
                progress_state.stage = "error"
                progress_state.message = "Simulation ended before producing results."

            _record_progress_log(progress_state, progress_state.message, progress_state.stage)
            _sync_progress_ui(
                progress_state,
                progress_message_placeholder,
                progress_bar_placeholder,
                progress_log_placeholder,
            )

    # --- Outputs panel ---
    outputs_container = st.container()
    with outputs_container:
        st.subheader("Model outputs")
        if st.session_state.get("run_in_progress"):
            st.info("Simulation in progress... progress updates appear above.")
        else:
            _render_outputs_panel(st.session_state.get("last_result"))

    # --- Final guidance to user ---
    if isinstance(st.session_state.get("last_result"), Mapping):
        if "error" in st.session_state["last_result"]:
            st.error(st.session_state["last_result"]["error"])
        else:
            st.info(
                "Review the outputs above to explore charts and downloads from the most recent run."
            )
    else:
        st.info("Use the inputs panel to configure and run the simulation.")

if __name__ == "__main__":  # pragma: no cover
    main()

