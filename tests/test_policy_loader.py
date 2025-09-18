from __future__ import annotations

import importlib

import pytest

pd = pytest.importorskip('pandas')

policy_loader = importlib.import_module('config.policy_loader')
load_annual_policy = policy_loader.load_annual_policy
series_from_year_map = policy_loader.series_from_year_map


def test_series_from_year_map_basic():
    cfg = {
        'years': [2027, 2026, 2025],
        'cap': {2025: 100, 2026: 90, 2027: 80},
    }

    series = series_from_year_map(cfg, 'cap')

    assert isinstance(series, pd.Series)
    assert list(series.index) == [2025, 2026, 2027]
    assert series.loc[2026] == pytest.approx(90.0)
    assert not series.attrs.get('fill_forward', False)


def test_series_from_year_map_fill_forward_floor():
    cfg = {
        'years': [2025, 2026, 2027],
        'floor': {'values': {2025: 4.0, 2027: 5.0}, 'fill': 'forward'},
    }

    series = series_from_year_map(cfg, 'floor')

    assert list(series.index) == [2025, 2026, 2027]
    assert series.loc[2026] == pytest.approx(4.0)
    assert series.loc[2027] == pytest.approx(5.0)
    assert series.attrs['fill_forward'] is True


def test_series_from_year_map_fill_forward_not_allowed_for_other_keys():
    cfg = {
        'years': [2025, 2026],
        'ccr1_trigger': {'values': {2025: 7.0}, 'fill': 'forward'},
    }

    with pytest.raises(ValueError) as exc:
        series_from_year_map(cfg, 'ccr1_trigger')

    assert 'ccr1_trigger' in str(exc.value)
    assert 'Fill-forward' in str(exc.value)


def test_series_from_year_map_reports_missing_years():
    cfg = {
        'years': [2025, 2026, 2027],
        'cap': {2025: 100.0, 2027: 80.0},
    }

    with pytest.raises(ValueError) as exc:
        series_from_year_map(cfg, 'cap')

    message = str(exc.value)
    assert 'cap' in message
    assert '2026' in message


def test_load_annual_policy_builds_series_and_validates_alignment():
    cfg = {
        'years': [2025, 2026, 2027],
        'cap': {2025: 100.0, 2026: 90.0, 2027: 95.0},
        'floor': {'values': {2025: 4.0}, 'fill_forward': True},
        'ccr1_trigger': {2025: 7.0, 2026: 7.0, 2027: 7.0},
        'ccr1_qty': {2025: 30.0, 2026: 30.0, 2027: 30.0},
        'ccr2_trigger': {2025: 13.0, 2026: 13.0, 2027: 13.0},
        'ccr2_qty': {2025: 60.0, 2026: 60.0, 2027: 60.0},
        'cp_id': {2025: 'CP1', 2026: 'CP1', 2027: 'CP1'},
        'bank0': 15.0,
        'full_compliance_years': [2027],
        'annual_surrender_frac': 0.5,
        'carry_pct': 1.0,
    }

    policy = load_annual_policy(cfg)

    expected_years = [2025, 2026, 2027]
    for series in [
        policy.cap,
        policy.floor,
        policy.ccr1_trigger,
        policy.ccr1_qty,
        policy.ccr2_trigger,
        policy.ccr2_qty,
    ]:
        assert list(series.index) == expected_years

    assert policy.floor.loc[2026] == pytest.approx(4.0)
    assert policy.floor.loc[2027] == pytest.approx(4.0)
    assert policy.cap.loc[2025] == pytest.approx(100.0)
    assert policy.ccr2_qty.loc[2027] == pytest.approx(60.0)
    assert policy.bank0 == pytest.approx(15.0)
    assert policy.full_compliance_years == {2027}



def test_load_annual_policy_flags_missing_series_years():
    cfg = {
        'years': [2025, 2026, 2027],
        'cap': {2025: 100.0, 2026: 90.0, 2027: 95.0},
        'floor': {2025: 4.0, 2026: 4.0, 2027: 4.0},
        'ccr1_trigger': {2025: 7.0, 2026: 7.0, 2027: 7.0},
        'ccr1_qty': {2025: 30.0, 2026: 30.0, 2027: 30.0},
        'ccr2_trigger': {2025: 13.0, 2026: 13.0, 2027: 13.0},
        'ccr2_qty': {2025: 60.0, 2027: 60.0},
        'cp_id': {2025: 'CP1', 2026: 'CP1', 2027: 'CP1'},
    }

    with pytest.raises(ValueError) as exc:
        load_annual_policy(cfg)

    assert 'ccr2_qty' in str(exc.value)
    assert '2026' in str(exc.value)
