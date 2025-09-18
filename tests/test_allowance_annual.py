from __future__ import annotations

import importlib

import pytest

pd = pytest.importorskip('pandas')

AllowanceAnnual = importlib.import_module('policy.allowance_annual').AllowanceAnnual
_fixtures = importlib.import_module('tests.fixtures.annual_minimal')
LinearDispatch = _fixtures.LinearDispatch
policy_for_shortage = _fixtures.policy_for_shortage
policy_three_year = _fixtures.policy_three_year
run_annual_fixed_point = importlib.import_module('engine.run_loop').run_annual_fixed_point


def test_clear_year_respects_floor_and_tracks_bank():
    policy = policy_three_year()
    market = AllowanceAnnual(policy)

    result = market.clear_year(2025, emissions_tons=80.0, bank_prev=policy.bank0)

    assert result['p_co2'] == pytest.approx(policy.floor.loc[2025])
    assert result['ccr1_issued'] == pytest.approx(0.0)
    assert result['ccr2_issued'] == pytest.approx(0.0)
    assert result['surrendered'] == pytest.approx(0.5 * 80.0)
    expected_bank = policy.bank0 + policy.cap.loc[2025] - result['surrendered']
    assert result['bank_new'] == pytest.approx(expected_bank)
    assert result['available_allowances'] == pytest.approx(policy.bank0 + policy.cap.loc[2025])
    assert result['obligation_new'] == pytest.approx(0.5 * 80.0)
    assert not result['shortage_flag']


def test_ccr_tranches_and_shortage_flag():
    policy = policy_three_year()
    market = AllowanceAnnual(policy)

    first = market.clear_year(2025, emissions_tons=130.0, bank_prev=policy.bank0)
    assert first['p_co2'] == pytest.approx(policy.ccr1_trigger.loc[2025])
    assert first['ccr1_issued'] == pytest.approx(20.0)
    assert first['ccr2_issued'] == pytest.approx(0.0)

    second = market.clear_year(2026, emissions_tons=220.0, bank_prev=first['bank_new'])
    assert second['p_co2'] == pytest.approx(policy.ccr2_trigger.loc[2026])
    assert second['ccr1_issued'] == pytest.approx(30.0)
    assert second['ccr2_issued'] == pytest.approx(35.0)
    assert not second['shortage_flag']

    shortage_policy = policy_for_shortage()
    shortage_market = AllowanceAnnual(shortage_policy)
    shortage = shortage_market.clear_year(2025, emissions_tons=400.0, bank_prev=shortage_policy.bank0)
    assert shortage['shortage_flag']
    assert shortage['ccr1_issued'] == pytest.approx(shortage_policy.ccr1_qty.loc[2025])
    assert shortage['ccr2_issued'] == pytest.approx(shortage_policy.ccr2_qty.loc[2025])
    assert shortage['surrendered'] == pytest.approx(shortage['available_allowances'])


def test_true_up_full_compliance_year():
    policy = policy_three_year()
    market = AllowanceAnnual(policy)

    bank = policy.bank0
    market.clear_year(2025, emissions_tons=130.0, bank_prev=bank)
    bank = market.bank_history[2025]
    market.clear_year(2026, emissions_tons=220.0, bank_prev=bank)
    bank = market.bank_history[2026]
    year_result = market.clear_year(2027, emissions_tons=150.0, bank_prev=bank)

    assert year_result['p_co2'] == pytest.approx(policy.floor.loc[2027])
    summary = market.finalize_period_if_needed(2027)
    assert summary['finalized']
    assert summary['surrendered_additional'] == pytest.approx(250.0)
    assert summary['bank_final'] == pytest.approx(35.0)
    assert summary['remaining_obligation'] == pytest.approx(0.0)
    assert not summary['shortage_flag']


def test_true_up_shortage_if_bank_insufficient():
    policy = policy_for_shortage()
    market = AllowanceAnnual(policy)

    market.clear_year(2025, emissions_tons=400.0, bank_prev=policy.bank0)
    summary = market.finalize_period_if_needed(2025)
    assert summary['finalized']
    assert summary['bank_final'] == pytest.approx(0.0)
    assert summary['remaining_obligation'] > 0.0
    assert summary['shortage_flag']


def test_run_loop_iterates_fixed_point():
    policy = policy_three_year()
    dispatch = LinearDispatch(base={2025: 260.0, 2026: 240.0, 2027: 180.0}, slope={2025: 5.0, 2026: 4.0, 2027: 3.0})

    outputs = run_annual_fixed_point(policy, dispatch, years=[2025, 2026, 2027], price_initial=0.0)

    assert outputs[2025]['p_co2'] >= policy.ccr2_trigger.loc[2025]
    assert outputs[2025]['iterations'] >= 1
    assert 'finalize' in outputs[2027]
    assert outputs[2027]['finalize']['finalized']
    assert outputs[2027]['finalize']['bank_final'] >= 0.0
