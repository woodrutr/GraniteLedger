from __future__ import annotations

"""GUI backend tests (skipped automatically if Streamlit is not installed)."""

import pytest

from tests.fixtures.dispatch_single_minimal import baseline_frames
from gui.app import run_policy_simulation

streamlit = pytest.importorskip("streamlit")
pd = pytest.importorskip("pandas")

def _baseline_config() -> dict:
    return {
        'years': [2025, 2026],
        'allowance_market': {
            'cap': {'2025': 500_000.0, '2026': 450_000.0},
            'floor': 5.0,
            'ccr1_trigger': 10.0,
            'ccr1_qty': 0.0,
            'ccr2_trigger': 20.0,
            'ccr2_qty': 0.0,
            'cp_id': 'CP1',
            'bank0': 50_000.0,
            'annual_surrender_frac': 1.0,
            'carry_pct': 1.0,
            'full_compliance_years': [2026],
        },
    }


def _frames_for_years(years: list[int]) -> object:
    base = baseline_frames(year=years[0])
    load = float(base.demand()['demand_mwh'].iloc[0])
    demand = pd.DataFrame(
        [
            {'year': year, 'region': 'default', 'demand_mwh': load}
            for year in years
        ]
    )
    return base.with_frame('demand', demand)


def _cleanup_temp_dir(result: dict) -> None:
    temp_dir = result.get('temp_dir')
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

    assert 'error' not in result
    annual = result['annual']
    assert not annual.empty
    assert {'p_co2', 'emissions_tons', 'bank'}.issubset(annual.columns)

    csv_files = result['csv_files']
    assert {'annual.csv', 'emissions_by_region.csv', 'price_by_region.csv', 'flows.csv'} <= set(csv_files)
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

    assert 'error' not in enabled
    assert 'error' not in disabled

    price_enabled = float(
        enabled['annual'].loc[enabled['annual']['year'] == 2025, 'p_co2'].iloc[0]
    )
    price_disabled = float(
        disabled['annual'].loc[disabled['annual']['year'] == 2025, 'p_co2'].iloc[0]
    )

    assert price_enabled >= 0.0
    assert price_disabled == pytest.approx(0.0)
    assert price_enabled >= price_disabled

    _cleanup_temp_dir(enabled)
    _cleanup_temp_dir(disabled)


def test_backend_returns_error_for_invalid_frames():
    config = _baseline_config()
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames={'demand': pd.DataFrame()},
    )

    assert 'error' in result


def test_backend_reports_missing_pandas(monkeypatch):
    config = _baseline_config()

    # Simulate an environment where pandas is not available after import time.
    monkeypatch.setattr('gui.app._PANDAS_MODULE', None)

    result = run_policy_simulation(config, start_year=2025, end_year=2025)

    assert 'error' in result
    assert 'pandas' in result['error'].lower()
