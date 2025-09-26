import importlib
import logging
import io
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from tests.fixtures.dispatch_single_minimal import baseline_frames
from gui.app import (
    DEEP_CARBON_UNSUPPORTED_MESSAGE,
    _build_price_schedule,
    run_policy_simulation,
)

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


def _assert_price_schedule(result: Mapping[str, Any], expected: Mapping[int, float]) -> None:
    annual = result["annual"].set_index("year")
    for year, price in expected.items():
        assert annual.loc[year, "p_co2"] == pytest.approx(
            price, rel=0.0, abs=1e-9
        )


def _emissions_by_year(result: Mapping[str, Any]) -> pd.Series:
    """Return the annual emissions indexed by year for convenience."""

    annual = result["annual"]
    assert isinstance(annual, pd.DataFrame)
    series = annual.set_index("year")["emissions_tons"].astype(float)
    series.index = series.index.astype(int)
    return series


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
        disabled["annual"].loc[
            disabled["annual"]["year"] == 2025, "p_co2"
        ].iloc[0]
    )

    assert price_enabled >= 0.0
    assert price_disabled == pytest.approx(0.0)
    assert price_enabled >= price_disabled

    _cleanup_temp_dir(enabled)
    _cleanup_temp_dir(disabled)


def test_backend_marks_allowance_price_output():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[1],
        frames=frames,
        carbon_policy_enabled=True,
    )

    assert result.get('_price_output_type') == 'allowance'

    _cleanup_temp_dir(result)


def test_backend_returns_technology_frames(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    capacity_df = pd.DataFrame(
        [
            {"year": 2025, "technology": "wind", "capacity_mw": 10.0},
            {"year": 2025, "technology": "solar", "capacity_mw": 5.0},
        ]
    )
    generation_df = pd.DataFrame(
        [
            {"year": 2025, "technology": "wind", "generation_mwh": 25.0},
            {"year": 2025, "technology": "solar", "generation_mwh": 15.0},
        ]
    )

    class StubOutputs:
        def __init__(self) -> None:
            self.annual = pd.DataFrame(
                [
                    {
                        "year": 2025,
                        "allowance_price": 0.0,
                        "emissions_tons": 0.0,
                        "bank": 0.0,
                    }
                ]
            )
            self.emissions_by_region = pd.DataFrame(
                [{"year": 2025, "region": "default", "emissions_tons": 0.0}]
            )
            self.price_by_region = pd.DataFrame(
                [{"year": 2025, "region": "default", "price": 0.0}]
            )
            self.flows = pd.DataFrame([{"from": "A", "to": "B", "value": 0.0}])
            self.capacity_by_technology = capacity_df
            self.generation_by_technology = generation_df

        def to_csv(self, target: Path) -> None:
            self.annual.to_csv(target / "annual.csv", index=False)
            self.emissions_by_region.to_csv(
                target / "emissions_by_region.csv", index=False
            )
            self.price_by_region.to_csv(target / "price_by_region.csv", index=False)
            self.flows.to_csv(target / "flows.csv", index=False)

    def stub_runner(frames_obj, **kwargs):
        assert kwargs.get("capacity_expansion") is True
        return StubOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: stub_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[1],
        frames=frames,
        dispatch_capacity_expansion=True,
    )

    assert "capacity_by_technology" in result
    assert isinstance(result["capacity_by_technology"], pd.DataFrame)
    assert "generation_by_technology" in result
    assert isinstance(result["generation_by_technology"], pd.DataFrame)

    _cleanup_temp_dir(result)


def test_backend_marks_carbon_price_output():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=15.0,
    )

    assert result.get('_price_output_type') == 'carbon'

    _cleanup_temp_dir(result)


def test_cap_region_alias_resolution_collapses_duplicates():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["NYCW", "nyc", "8"],
        frames=frames,
    )

    assert "error" not in result
    assert result.get("cap_regions") == [8]
    carbon_cfg = result["config"]["modules"]["carbon_policy"]
    assert carbon_cfg.get("regions") == [8]

    _cleanup_temp_dir(result)


def test_cap_region_all_selection_collapses_to_empty():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["All"],
        frames=frames,
    )

    assert "error" not in result
    assert result.get("cap_regions") in (None, [])
    carbon_cfg = result["config"]["modules"]["carbon_policy"]
    assert carbon_cfg.get("regions") in (None, [])

    _cleanup_temp_dir(result)


def test_cap_region_unknown_label_errors():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["Atlantis"],
        frames=frames,
    )

    assert "error" in result
    assert "Unable to resolve cap region" in result["error"]


def test_render_results_carbon_price_hides_allowance_columns(monkeypatch):
    from gui import app as gui_app

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=25.0,
    )

    assert result.get('_price_output_type') == 'carbon'

    class DummyTab:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyStreamlit:
        def __init__(self) -> None:
            self.dataframes = []
            self.tab_labels = []

        def error(self, *args, **kwargs):
            return None

        def caption(self, *args, **kwargs):
            return None

        def subheader(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def line_chart(self, *args, **kwargs):
            return None

        def bar_chart(self, *args, **kwargs):
            return None

        def dataframe(self, frame, **kwargs):
            self.dataframes.append(frame)
            return None

        def tabs(self, labels):
            self.tab_labels.append(list(labels))
            return [DummyTab() for _ in labels]

        def download_button(self, *args, **kwargs):
            return None

    dummy_st = DummyStreamlit()
    monkeypatch.setattr(gui_app, "st", dummy_st)

    gui_app._render_results(result)

    assert dummy_st.tab_labels, "Expected result tabs to be rendered"
    assert "Allowance bank" not in dummy_st.tab_labels[0]

    price_tables = [
        frame
        for frame in dummy_st.dataframes
        if isinstance(frame, pd.DataFrame) and "Carbon price ($/ton)" in frame.columns
    ]

    assert price_tables, "Expected carbon price table to be rendered"

    allowed_columns = {
        "year",
        "Carbon price ($/ton)",
        "p_co2_all",
        "p_co2_exc",
        "p_co2_eff",
        "emissions_tons",
    }
    disallowed_columns = {"allowances_total", "bank"}

    for table in price_tables:
        assert set(table.columns).issubset(allowed_columns)
        assert disallowed_columns.isdisjoint(table.columns)

    _cleanup_temp_dir(result)


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


def test_backend_handles_legacy_runner_without_deep_kw(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    annual = pd.DataFrame([{"year": 2025, "p_co2": 12.0}])
    emissions = pd.DataFrame([{"year": 2025, "region": "default", "emissions_tons": 1.0}])
    prices = pd.DataFrame([{"year": 2025, "region": "default", "price": 45.0}])
    flows = pd.DataFrame(
        [{"year": 2025, "from_region": "A", "to_region": "B", "flow_mwh": 10.0}]
    )

    called: dict[str, bool] = {}

    class LegacyOutputs:
        def __init__(self) -> None:
            self.annual = annual
            self.emissions_by_region = emissions
            self.price_by_region = prices
            self.flows = flows

        def to_csv(self, target: Path) -> None:
            target = Path(target)
            target.mkdir(parents=True, exist_ok=True)
            self.annual.to_csv(target / "annual.csv", index=False)
            self.emissions_by_region.to_csv(target / "emissions_by_region.csv", index=False)
            self.price_by_region.to_csv(target / "price_by_region.csv", index=False)
            self.flows.to_csv(target / "flows.csv", index=False)

    def legacy_runner(
        frames,
        *,
        years=None,
        price_initial=0.0,
        enable_floor=True,
        enable_ccr=True,
        use_network=False,
        carbon_price_schedule=None,
        progress_cb=None,
    ):
        called["executed"] = True
        return LegacyOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: legacy_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        deep_carbon_pricing=False,
    )

    assert "error" not in result
    assert called.get("executed") is True
    pd.testing.assert_frame_equal(result["annual"], annual)
    pd.testing.assert_frame_equal(result["emissions_by_region"], emissions)
    pd.testing.assert_frame_equal(result["price_by_region"], prices)
    pd.testing.assert_frame_equal(result["flows"], flows)

    _cleanup_temp_dir(result)


def test_backend_rejects_deep_mode_when_runner_lacks_kw(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])
    called: dict[str, bool] = {}

    def legacy_runner(
        frames,
        *,
        years=None,
        price_initial=0.0,
        enable_floor=True,
        enable_ccr=True,
        use_network=False,
        carbon_price_schedule=None,
        progress_cb=None,
    ):
        called["executed"] = True
        return {}

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: legacy_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        deep_carbon_pricing=True,
        carbon_price_enabled=True,
        carbon_price_value=10.0,
    )

    assert result.get("error") == DEEP_CARBON_UNSUPPORTED_MESSAGE
    assert "executed" not in called


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
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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


def test_backend_enforces_carbon_mode_exclusivity(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["policy_enabled"] = policy.enabled
        captured["price_schedule"] = kwargs.get("carbon_price_schedule")
        captured["price_value"] = kwargs.get("carbon_price_value")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        carbon_policy_enabled=True,
        carbon_price_enabled=True,
        carbon_price_value=15.0,
        carbon_price_escalator_pct=0.0,
    )

    assert "error" not in result
    assert captured.get("policy_enabled") is False
    schedule = captured.get("price_schedule")
    assert isinstance(schedule, Mapping)
    assert schedule  # non-empty schedule when price enabled
    assert captured.get("price_value") == pytest.approx(15.0)

    module_config = result["module_config"]
    policy_cfg = module_config["carbon_policy"]
    price_cfg = module_config["carbon_price"]
    assert not policy_cfg.get("enabled")
    assert price_cfg.get("enabled")
    assert sum(1 for flag in (policy_cfg.get("enabled"), price_cfg.get("enabled")) if flag) == 1

    _cleanup_temp_dir(result)


def test_backend_control_period_defaults_to_config(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["control"] = policy.control_period_length
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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


def test_backend_carbon_price_reduces_emissions():
    config = _baseline_config()
    config["allowance_market"]["cap"] = {"2025": 1_000_000.0, "2026": 1_000_000.0}
    config["allowance_market"]["floor"] = 0.0
    frames = _frames_for_years([2025, 2026])

    baseline = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        carbon_price_enabled=False,
    )
    priced = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        carbon_price_enabled=True,
        carbon_price_value=100.0,
        carbon_price_escalator_pct=0.0,
    )

    assert "error" not in baseline
    assert "error" not in priced

    baseline_emissions = _emissions_by_year(baseline)
    priced_emissions = _emissions_by_year(priced)

    for year in baseline_emissions.index:
        assert priced_emissions.loc[year] < baseline_emissions.loc[year]

    _cleanup_temp_dir(baseline)
    _cleanup_temp_dir(priced)


def test_backend_carbon_price_schedule_lowers_future_emissions():
    config = _baseline_config()
    config["allowance_market"]["cap"] = {}
    frames = _frames_for_years([2024, 2025])

    result = run_policy_simulation(
        config,
        start_year=2024,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_schedule={2024: 0.0, 2025: 100.0},
        carbon_price_escalator_pct=0.0,
    )

    assert "error" not in result

    annual = result["annual"].set_index("year")
    first_year = int(annual.index.min())
    later_year = int(annual.index.max())

    assert later_year > first_year
    assert annual.loc[later_year, "emissions_tons"] < annual.loc[first_year, "emissions_tons"]

    _cleanup_temp_dir(result)


def test_backend_control_period_override_applies(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["control"] = policy.control_period_length
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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


def test_price_schedule_forward_fills_without_cap_or_banking():
    years = list(range(2025, 2031))
    schedule = {2025: 45.0, 2030: 48.15}
    expected = {
        2025: 45.0,
        2026: 45.0,
        2027: 45.0,
        2028: 45.0,
        2029: 45.0,
        2030: 48.15,
    }

    scenarios = [
        {"carbon_policy_enabled": False, "allowance_banking_enabled": True},
        {"carbon_policy_enabled": False, "allowance_banking_enabled": False},
    ]

    for options in scenarios:
        config = _baseline_config()
        frames = _frames_for_years(years)
        result = run_policy_simulation(
            config,
            start_year=2025,
            end_year=2030,
            frames=frames,
            carbon_price_enabled=True,
            carbon_price_schedule=schedule,
            **options,
        )

        assert "error" not in result
        _assert_price_schedule(result, expected)
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
        captured["deep_carbon_pricing"] = kwargs.get("deep_carbon_pricing")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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
    assert captured.get("deep_carbon_pricing") is False
    assert captured.get("capacity_expansion") is True
    assert captured.get("report_by_technology") is True
    dispatch_cfg = result["module_config"]["electricity_dispatch"]
    assert dispatch_cfg["enabled"] is True
    assert dispatch_cfg["use_network"] is True
    assert dispatch_cfg.get("deep_carbon_pricing") is False
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("regions") == [1]

    _cleanup_temp_dir(result)


def test_backend_canonicalizes_cap_region_aliases():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    alias_entries = [
        "NYISO",
        "Region 9 – NYUP (Northeast / MidAtlantic, NYISO Upstate)",
        9,
        "nyup",
    ]

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=alias_entries,
        frames=frames,
    )

    assert "error" not in result
    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("regions") == [9]
    assert result.get("cap_regions") == [9]

    _cleanup_temp_dir(result)


def test_backend_rejects_unknown_cap_region():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["Atlantis"],
        frames=frames,
    )

    assert "error" in result
    assert "Atlantis" in str(result["error"])


def test_backend_mutual_exclusion_without_deep():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        cap_regions=[1],
        carbon_price_enabled=True,
        carbon_price_value=25.0,
        deep_carbon_pricing=False,
    )

    assert result.get("error") == "Cannot enable both carbon cap and carbon price simultaneously."


def test_backend_deep_carbon_combines_prices(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        captured["deep_carbon_pricing"] = kwargs.get("deep_carbon_pricing")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        cap_regions=[1],
        carbon_price_enabled=True,
        carbon_price_value=15.0,
        deep_carbon_pricing=True,
    )

    assert "error" not in result
    assert captured.get("deep_carbon_pricing") is True

    annual = result["annual"]
    row = annual.loc[annual["year"] == 2025].iloc[0]
    allowance_price = float(row["p_co2_all"])
    exogenous_price = float(row["p_co2_exc"])
    effective_price = float(row["p_co2_eff"])

    assert row["p_co2"] == pytest.approx(allowance_price)
    assert exogenous_price == pytest.approx(15.0)
    assert effective_price == pytest.approx(allowance_price + exogenous_price)

    dispatch_cfg = result["module_config"].get("electricity_dispatch", {})
    assert dispatch_cfg.get("deep_carbon_pricing") is True

    _cleanup_temp_dir(result)


def test_backend_reports_missing_deep_support(monkeypatch):
    def legacy_runner(
        frames,
        *,
        years=None,
        price_initial=0.0,
        tol=1e-3,
        max_iter=25,
        relaxation=0.5,
        enable_floor=True,
        enable_ccr=True,
        price_cap=1000.0,
        use_network=False,
        carbon_price_schedule=None,
        progress_cb=None,
    ):
        raise AssertionError("legacy runner should not be invoked when unsupported")


    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: legacy_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        cap_regions=[1],
        carbon_price_enabled=True,
        carbon_price_value=20.0,
        deep_carbon_pricing=True,
    )

    assert result.get("error") == (
        "Deep carbon pricing requires an updated engine. "
        "Please upgrade engine.run_loop.run_end_to_end_from_frames."
    )

def test_backend_carbon_price_disables_cap(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["carbon_enabled"] = policy.enabled
        captured["control"] = policy.control_period_length
        captured["price_schedule"] = kwargs.get("carbon_price_schedule")
        captured["price_value"] = kwargs.get("carbon_price_value")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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
    assert captured.get("price_value") == pytest.approx(37.0)

    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("enabled") is False
    assert carbon_cfg.get("control_period_years") is None
    price_cfg = result["module_config"].get("carbon_price", {})
    assert price_cfg.get("enabled") is True
    assert price_cfg.get("price_per_ton") == pytest.approx(37.0)

    _cleanup_temp_dir(result)


def test_backend_banking_toggle_disables_bank(tmp_path, caplog):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    with caplog.at_level(logging.WARNING):
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
    assert any("Allowance banking disabled" in record.message for record in caplog.records)

    _cleanup_temp_dir(result)


def test_backend_updates_allowance_market_config():
    config = _baseline_config()
    config["allowance_market"]["cap"] = {}
    frames = _frames_for_years([2025, 2026])

    schedule = {2025: 345_000.0, 2026: 320_000.0}

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        cap_regions=[1],
        carbon_cap_schedule=schedule,
        initial_bank=1234.0,
        allowance_banking_enabled=True,
        ccr1_enabled=True,
        ccr2_enabled=False,
    )

    assert "error" not in result

    allowance_module = result["module_config"]["allowance_market"]
    allowance_config = result["config"]["allowance_market"]
    expected_schedule = {year: float(value) for year, value in schedule.items()}

    assert allowance_module["enabled"] is True
    assert allowance_config["enabled"] is True
    assert allowance_module["cap"] == expected_schedule
    assert allowance_config["cap"] == expected_schedule
    assert allowance_module["bank0"] == pytest.approx(1234.0)
    assert allowance_config["bank0"] == pytest.approx(1234.0)
    assert allowance_module["ccr1_enabled"] is True
    assert allowance_module["ccr2_enabled"] is False
    assert allowance_config["ccr1_enabled"] is True
    assert allowance_config["ccr2_enabled"] is False

    _cleanup_temp_dir(result)


def test_backend_zeroes_allowance_bank_when_disabled_config():
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    schedule = {2025: 400_000.0, 2026: 390_000.0}

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        cap_regions=[1],
        carbon_cap_schedule=schedule,
        initial_bank=9876.0,
        allowance_banking_enabled=False,
    )

    assert "error" not in result

    allowance_module = result["module_config"]["allowance_market"]
    allowance_config = result["config"]["allowance_market"]

    assert allowance_module["bank0"] == pytest.approx(0.0)
    assert allowance_config["bank0"] == pytest.approx(0.0)
    assert allowance_module["enabled"] is True
    assert allowance_config["enabled"] is True

    _cleanup_temp_dir(result)


def test_backend_builds_price_schedule_for_run_years(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        captured["carbon_price_schedule"] = kwargs.get("carbon_price_schedule")
        captured["carbon_price_value"] = kwargs.get("carbon_price_value")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026, 2027, 2028, 2029, 2030])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2030,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=45.0,
        carbon_price_escalator_pct=7.0,
    )

    assert "error" not in result

    schedule = captured.get("carbon_price_schedule")
    assert isinstance(schedule, Mapping)
    expected = _build_price_schedule(2025, 2030, 45.0, 7.0)
    assert schedule == expected
    assert captured.get("carbon_price_value") == pytest.approx(45.0)

    _cleanup_temp_dir(result)


def test_backend_bank_column_tracks_allowances(tmp_path):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        cap_regions=[1],
        frames=frames,
        allowance_banking_enabled=True,
    )


    assert "error" not in result
    annual = result["annual"].copy()
    assert not annual.empty

    annual = annual.sort_values("year").reset_index(drop=True)
    bank0 = float(config["allowance_market"]["bank0"])
    expected = []
    bank_prev = bank0
    for _, row in annual.iterrows():
        allowances_total = float(row["allowances_available"])
        emissions = float(row["emissions_tons"])
        bank_prev = max(bank_prev + allowances_total - emissions, 0.0)
        expected.append(bank_prev)

    assert annual["bank"].tolist() == pytest.approx(expected)

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
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
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
