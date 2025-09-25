import importlib
import io
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from tests.fixtures.dispatch_single_minimal import baseline_frames
from gui.app import run_policy_simulation

streamlit = pytest.importorskip("streamlit")


def _baseline_config() -> dict:
    return {
        "years": [2025, 2026],
        "regions": [1],
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
            "control_period_years": 2,
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


def test_write_outputs_to_temp_falls_back_when_default_unwritable(monkeypatch):
    from gui import app as gui_app

    class DummyOutputs:
        def __init__(self) -> None:
            self.saved_to: Path | None = None

        def to_csv(self, target: Path) -> None:
            self.saved_to = Path(target)
            csv_path = self.saved_to / "dummy.csv"
            csv_path.write_text("value")

    fallback_base = Path.cwd() / ".graniteledger" / "tmp"

    monkeypatch.delenv("GRANITELEDGER_TMPDIR", raising=False)
    monkeypatch.setattr(gui_app.tempfile, "gettempdir", lambda: "/unwritable")

    def fake_mkdtemp(prefix: str, dir: str | None = None) -> str:
        if dir == "/unwritable":
            raise PermissionError("read-only filesystem")
        assert dir == str(fallback_base)
        target_dir = Path(dir) / "fallback"
        target_dir.mkdir(parents=True, exist_ok=False)
        return str(target_dir)

    monkeypatch.setattr(gui_app.tempfile, "mkdtemp", fake_mkdtemp)

    outputs = DummyOutputs()
    temp_dir, csv_files = gui_app._write_outputs_to_temp(outputs)

    try:
        assert outputs.saved_to == temp_dir
        assert temp_dir == fallback_base / "fallback"
        assert csv_files == {"dummy.csv": b"value"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(fallback_base, ignore_errors=True)


def test_backend_generates_outputs(tmp_path):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        cap_regions=[1],
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
        cap_regions=[1],
        frames=frames,
        carbon_policy_enabled=True,
    )
    disabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[1],
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


def test_dispatch_capacity_toggle_updates_config():
    config = _baseline_config()
    frames = _frames_for_years([2025])
    module_config = {
        "electricity_dispatch": {"enabled": True, "capacity_expansion": True}
    }

    disabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[1],
        frames=frames,
        dispatch_capacity_expansion=False,
        module_config=module_config,
    )

    assert "error" not in disabled
    assert disabled["config"].get("sw_expansion") == 0
    dispatch_cfg = disabled["module_config"]["electricity_dispatch"]
    assert dispatch_cfg["capacity_expansion"] is False

    enabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[1],
        frames=frames,
        dispatch_capacity_expansion=True,
        module_config=module_config,
    )

    assert "error" not in enabled
    assert enabled["config"].get("sw_expansion") == 1
    assert enabled["module_config"]["electricity_dispatch"]["capacity_expansion"] is True

    _cleanup_temp_dir(disabled)
    _cleanup_temp_dir(enabled)


def test_backend_handles_renamed_engine_outputs(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    annual = pd.DataFrame([{"year": 2025, "p_co2": 12.0}])
    emissions = pd.DataFrame([{"year": 2025, "region": "default", "emissions_tons": 1.0}])
    prices = pd.DataFrame([{"year": 2025, "region": "default", "price": 45.0}])
    flows = pd.DataFrame(
        [{"year": 2025, "from_region": "A", "to_region": "B", "flow_mwh": 10.0}]
    )

    class FakeOutputs:
        def __init__(self) -> None:
            self.annual_results = annual
            self.emissions = emissions
            self.dispatch_price_by_region = prices
            self.network_flows = flows

        def to_csv(self, outdir):
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            self.annual_results.to_csv(outdir / "annual.csv", index=False)
            self.emissions.to_csv(outdir / "emissions_by_region.csv", index=False)
            self.dispatch_price_by_region.to_csv(outdir / "price_by_region.csv", index=False)
            self.network_flows.to_csv(outdir / "flows.csv", index=False)

    def fake_runner(*args, **kwargs):
        return FakeOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: fake_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
    )

    pd.testing.assert_frame_equal(result["annual"], annual)
    pd.testing.assert_frame_equal(result["emissions_by_region"], emissions)
    pd.testing.assert_frame_equal(result["price_by_region"], prices)
    pd.testing.assert_frame_equal(result["flows"], flows)

    _cleanup_temp_dir(result)


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


def test_backend_control_period_defaults_to_config(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["control"] = policy.control_period_length
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        control_period_years=None,
    )

    assert "error" not in result
    assert captured.get("control") == 2
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("control_period_years") is None

    _cleanup_temp_dir(result)


def test_backend_control_period_override_applies(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["control"] = policy.control_period_length
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        control_period_years=4,
    )

    assert "error" not in result
    assert captured.get("control") == 4
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("control_period_years") == 4

    _cleanup_temp_dir(result)


def test_backend_errors_when_demand_years_do_not_overlap():
    config = _baseline_config()
    frames = _frames_for_years([2030, 2031])

    result = run_policy_simulation(config, frames=frames)

    assert "error" in result
    assert "Demand data covers years" in result["error"]


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
        cap_regions=[1],
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
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("regions") == [1]

    _cleanup_temp_dir(result)


def test_backend_carbon_price_disables_cap(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["carbon_enabled"] = policy.enabled
        captured["control"] = policy.control_period_length
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
    assert captured.get("control") is None
    schedule = captured.get("price_schedule")
    assert isinstance(schedule, Mapping)
    assert schedule.get(2026) == pytest.approx(37.0)

    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("enabled") is False
    assert carbon_cfg.get("control_period_years") is None
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
        cap_regions=[1],
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
        cap_regions=[1],
        frames={"demand": pd.DataFrame()},
    )

    assert "error" in result


def test_backend_preserves_explicit_coverage_overrides(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, pd.DataFrame] = {}

    def capturing_runner(frames, **kwargs):
        captured["coverage"] = frames.coverage()
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])
    coverage = pd.DataFrame(
        [
            {"region": "default", "year": 2025, "covered": False},
            {"region": "other", "year": -1, "covered": True},
        ]
    )
    frames = frames.with_frame("coverage", coverage)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["default"],
        frames=frames,
    )

    assert "error" not in result
    coverage_frame = captured.get("coverage")
    assert coverage_frame is not None
    default_rows = coverage_frame[coverage_frame["region"].astype(str) == "default"]
    assert set(default_rows["year"].astype(int)) == {-1, 2025}
    explicit_value = default_rows.loc[default_rows["year"] == 2025, "covered"].iloc[0]
    default_value = default_rows.loc[default_rows["year"] == -1, "covered"].iloc[0]
    assert bool(explicit_value) is False
    assert bool(default_value) is True

    _cleanup_temp_dir(result)


def test_backend_builds_default_frames(tmp_path):
    config = _baseline_config()
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[1],
    )

    assert "error" not in result
    assert not result["annual"].empty

    _cleanup_temp_dir(result)


def test_backend_coverage_selection_builds_frame(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        captured["coverage_df"] = frames.coverage()
        captured["coverage_map"] = frames.coverage_for_year(2025)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = baseline_frames(year=2025)
    units = frames.units()
    units.loc[units["unit_id"] == "coal-1", "region"] = "north"
    units.loc[units["unit_id"] == "gas-1", "region"] = "south"
    units.loc[units["unit_id"] == "wind-1", "region"] = "north"
    frames = frames.with_frame("units", units)
    demand = pd.DataFrame(
        [
            {"year": 2025, "region": "north", "demand_mwh": 250_000.0},
            {"year": 2025, "region": "south", "demand_mwh": 250_000.0},
        ]
    )
    frames = frames.with_frame("demand", demand)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        coverage_regions=["north"],
        dispatch_use_network=True,
    )

    assert "error" not in result
    coverage_df = captured.get("coverage_df")
    assert isinstance(coverage_df, pd.DataFrame)
    assert {"north", "south"}.issubset(set(coverage_df["region"]))
    north_flag = bool(
        coverage_df.loc[coverage_df["region"] == "north", "covered"].iloc[0]
    )
    south_flag = bool(
        coverage_df.loc[coverage_df["region"] == "south", "covered"].iloc[0]
    )
    assert north_flag is True
    assert south_flag is False
    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("regions") == ["north"]

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
