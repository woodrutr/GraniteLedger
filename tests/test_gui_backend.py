import importlib
import io
import shutil
from collections.abc import Mapping

import pytest

pd = pytest.importorskip("pandas")

from tests.fixtures.dispatch_single_minimal import baseline_frames
from gui.app import run_policy_simulation

streamlit = pytest.importorskip("streamlit")


def _baseline_config() -> dict:
    return {
        "years": [2025, 2026],
        "allowance_market": {
            "cap": {"2025": 500_000.0, "2026": 450_000.0},
            "floor": 5.0,
            "ccr1_trigger": 10.0,
            "ccr1_qty": 0.0,
            "ccr2_trigger": 20.0,
            "ccr2_qty": 0.0,
            "cp_id": "CP1",
            "bank0": 50_000.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "full_compliance_years": [2026],
            "resolution": "annual",
        },
    }


def _frames_for_years(years: list[int]) -> object:
    base = baseline_frames(year=years[0])
    load = float(base.demand()["demand_mwh"].iloc[0])
    demand = pd.DataFrame(
        [{"year": year, "region": "default", "demand_mwh": load} for year in years]
    )
    return base.with_frame("demand", demand)


def _cleanup_temp_dir(result: dict) -> None:
    temp_dir = result.get("temp_dir")
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_backend_generates_outputs(tmp_path):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
    )

    assert "error" not in result
    annual = result["annual"]
    assert not annual.empty
    assert {"p_co2", "emissions_tons", "bank"}.issubset(annual.columns)

    csv_files = result["csv_files"]
    assert {
        "annual.csv",
        "emissions_by_region.csv",
        "price_by_region.csv",
        "flows.csv",
    } <= set(csv_files)
    for content in csv_files.values():
        assert isinstance(content, (bytes, bytearray))

    _cleanup_temp_dir(result)


def test_backend_policy_toggle_affects_price():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    enabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
    )
    disabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
    )

    assert "error" not in enabled
    assert "error" not in disabled

    price_enabled = float(
        enabled["annual"].loc[enabled["annual"]["year"] == 2025, "p_co2"].iloc[0]
    )
    price_disabled = float(
        disabled["annual"].loc[disabled["annual"]["year"] == 2025, "p_co2"].iloc[0]
    )

    assert price_enabled >= 0.0
    assert price_disabled == pytest.approx(0.0)
    assert price_enabled >= price_disabled

    _cleanup_temp_dir(enabled)
    _cleanup_temp_dir(disabled)


def test_backend_disabled_toggle_propagates_flags(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["enabled"] = policy.enabled
        captured["ccr1"] = policy.ccr1_enabled
        captured["ccr2"] = policy.ccr2_enabled
        captured["control"] = policy.control_period_length
        captured["banking"] = policy.banking_enabled
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        ccr1_enabled=False,
        ccr2_enabled=False,
        allowance_banking_enabled=False,
        control_period_years=4,
    )

    assert "error" not in result
    assert captured.get("enabled") is False
    assert captured.get("ccr1") is False
    assert captured.get("ccr2") is False
    assert captured.get("control") is None
    assert captured.get("banking") is False

    _cleanup_temp_dir(result)


def test_backend_dispatch_and_carbon_modules(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["carbon_enabled"] = policy.enabled
        captured["use_network"] = kwargs.get("use_network")
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    module_config = {
        "carbon_policy": {"enabled": True, "allowance_banking_enabled": True},
        "electricity_dispatch": {
            "enabled": True,
            "mode": "network",
            "capacity_expansion": True,
            "reserve_margins": True,
        },
    }

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        dispatch_use_network=True,
        module_config=module_config,
    )

    assert "error" not in result
    assert captured.get("carbon_enabled") is True
    assert captured.get("use_network") is True
    dispatch_cfg = result["module_config"]["electricity_dispatch"]
    assert dispatch_cfg["enabled"] is True
    assert dispatch_cfg["use_network"] is True

    _cleanup_temp_dir(result)


def test_backend_carbon_price_disables_cap(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["carbon_enabled"] = policy.enabled
        captured["price_schedule"] = kwargs.get("carbon_price_schedule")
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2026])

    module_config = {"carbon_price": {"enabled": True, "price_per_ton": 37.0}}

    result = run_policy_simulation(
        config,
        start_year=2026,
        end_year=2026,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=37.0,
        module_config=module_config,
    )

    assert "error" not in result
    assert captured.get("carbon_enabled") is False
    schedule = captured.get("price_schedule")
    assert isinstance(schedule, Mapping)
    assert schedule.get(2026) == pytest.approx(37.0)

    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("enabled") is False
    price_cfg = result["module_config"].get("carbon_price", {})
    assert price_cfg.get("enabled") is True
    assert price_cfg.get("price_per_ton") == pytest.approx(37.0)

    _cleanup_temp_dir(result)


def test_backend_banking_toggle_disables_bank(tmp_path):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        allowance_banking_enabled=False,
    )

    assert "error" not in result
    annual = result["annual"]
    assert annual["bank"].eq(0.0).all()

    _cleanup_temp_dir(result)


def test_backend_returns_error_for_invalid_frames():
    config = _baseline_config()
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames={"demand": pd.DataFrame()},
    )

    assert "error" in result


def test_backend_builds_default_frames(tmp_path):
    config = _baseline_config()
    result = run_policy_simulation(config, start_year=2025, end_year=2025)

    assert "error" not in result
    assert not result["annual"].empty

    _cleanup_temp_dir(result)


def test_build_policy_frame_control_override():
    from gui.app import _build_policy_frame

    config = _baseline_config()
    years = [2025, 2026, 2027]
    frame = _build_policy_frame(
        config,
        years,
        carbon_policy_enabled=True,
        control_period_years=2,
    )

    assert set(frame["year"]) == set(years)
    assert frame["policy_enabled"].all()
    assert frame["control_period_years"].dropna().unique().tolist() == [2]
    assert frame["bank_enabled"].all()


def test_build_policy_frame_disabled_defaults():
    from gui.app import _build_policy_frame

    config = _baseline_config()
    years = [2025]
    frame = _build_policy_frame(config, years, carbon_policy_enabled=False)

    assert not frame["policy_enabled"].any()
    assert frame["cap_tons"].iloc[0] > 0.0
    assert bool(frame["ccr1_enabled"].iloc[0]) is False
    assert frame["bank_enabled"].eq(False).all()


def test_load_config_data_accepts_various_sources(tmp_path):
    from gui.app import _load_config_data

    mapping = {"a": 1}
    assert _load_config_data(mapping) == mapping

    toml_text = "value = 1\n"
    assert _load_config_data(toml_text.encode("utf-8"))["value"] == 1

    temp_file = tmp_path / "config.toml"
    temp_file.write_text("value = 2\n", encoding="utf-8")
    assert _load_config_data(str(temp_file))["value"] == 2

    stream = io.StringIO("value = 3\n")
    assert _load_config_data(stream)["value"] == 3

    with pytest.raises(TypeError):
        _load_config_data(object())


def test_year_and_selection_helpers_cover_branches():
    from gui.app import _years_from_config, _select_years

    config = {"years": [{"year": 2025}, 2026]}
    years = _years_from_config(config)
    assert years == [2025, 2026]

    fallback = {"start_year": 2024, "end_year": 2022}
    fallback_years = _years_from_config(fallback)
    assert fallback_years == [2022, 2023, 2024]

    selected = _select_years(fallback_years, start_year=2023, end_year=2024)
    assert selected == [2023, 2024]

    sparse_years = [2025, 2030]
    expanded = _select_years(sparse_years, start_year=2025, end_year=2030)
    assert expanded == [2025, 2026, 2027, 2028, 2029, 2030]

    with pytest.raises(ValueError):
        _select_years(years, start_year=2026, end_year=2024)
