from __future__ import annotations

import importlib
import pytest

pytest.importorskip("pandas")

from policy.allowance_annual import (
    AllowanceMarketState,
    allowance_initial_state,
    clear_year,
    finalize_period_if_needed,
)
from engine.run_loop import run_annual_fixed_point

fixtures = importlib.import_module("tests.fixtures.annual_minimal")
LinearDispatch = fixtures.LinearDispatch
policy_for_shortage = fixtures.policy_for_shortage
policy_three_year = fixtures.policy_three_year


def test_clear_year_respects_floor_and_tracks_bank():
    policy = policy_three_year()
    state = allowance_initial_state()
    result, state = clear_year(
        policy,
        state,
        2025,
        emissions_tons=80.0,
        bank_prev=policy.bank0,
    )

    assert result["p_co2"] == pytest.approx(policy.floor.loc[2025])
    assert result["ccr1_issued"] == pytest.approx(0.0)
    assert result["ccr2_issued"] == pytest.approx(0.0)
    assert result["surrendered"] == pytest.approx(0.5 * 80.0)
    expected_bank = policy.bank0 + policy.cap.loc[2025] - result["surrendered"]
    assert result["bank_new"] == pytest.approx(expected_bank)
    assert result["available_allowances"] == pytest.approx(policy.bank0 + policy.cap.loc[2025])
    assert result["obligation_new"] == pytest.approx(0.5 * 80.0)
    assert not result["shortage_flag"]
    assert isinstance(state, AllowanceMarketState)
    assert state.bank_history[2025] == pytest.approx(result["bank_new"])
    assert state.year_records[2025]["p_co2"] == pytest.approx(result["p_co2"])


def test_ccr_tranches_and_shortage_flag():
    policy = policy_three_year()
    state = allowance_initial_state()

    first, state = clear_year(
        policy,
        state,
        2025,
        emissions_tons=130.0,
        bank_prev=policy.bank0,
    )
    assert first["p_co2"] == pytest.approx(policy.ccr1_trigger.loc[2025])
    assert first["ccr1_issued"] == pytest.approx(20.0)
    assert first["ccr2_issued"] == pytest.approx(0.0)
    assert state.bank_history[2025] == pytest.approx(first["bank_new"])

    second, state = clear_year(
        policy,
        state,
        2026,
        emissions_tons=220.0,
        bank_prev=state.bank_history[2025],
    )
    assert second["p_co2"] == pytest.approx(policy.ccr2_trigger.loc[2026])
    assert second["ccr1_issued"] == pytest.approx(30.0)
    assert second["ccr2_issued"] == pytest.approx(35.0)
    assert not second["shortage_flag"]
    assert state.bank_history[2026] == pytest.approx(second["bank_new"])

    shortage_policy = policy_for_shortage()
    shortage_state = allowance_initial_state()
    shortage, shortage_state = clear_year(
        shortage_policy,
        shortage_state,
        2025,
        emissions_tons=400.0,
        bank_prev=shortage_policy.bank0,
    )
    assert shortage["shortage_flag"]
    assert shortage["ccr1_issued"] == pytest.approx(shortage_policy.ccr1_qty.loc[2025])
    assert shortage["ccr2_issued"] == pytest.approx(shortage_policy.ccr2_qty.loc[2025])
    assert shortage["surrendered"] == pytest.approx(shortage["available_allowances"])
    assert shortage_state.bank_history[2025] == pytest.approx(shortage["bank_new"])


def test_ccr_modules_can_be_disabled():
    policy = policy_three_year()
    policy.ccr1_enabled = False
    policy.ccr2_enabled = False
    state = allowance_initial_state()
    result, state = clear_year(
        policy,
        state,
        2025,
        emissions_tons=220.0,
        bank_prev=policy.bank0,
    )

    assert result["ccr1_issued"] == pytest.approx(0.0)
    assert result["ccr2_issued"] == pytest.approx(0.0)
    assert result["p_co2"] == pytest.approx(policy.floor.loc[2025])
    assert result["shortage_flag"]
    assert state.bank_history[2025] == pytest.approx(result["bank_new"])


def test_true_up_full_compliance_year():
    policy = policy_three_year()
    state = allowance_initial_state()

    bank = policy.bank0
    _, state = clear_year(policy, state, 2025, emissions_tons=130.0, bank_prev=bank)
    bank = state.bank_history[2025]
    _, state = clear_year(policy, state, 2026, emissions_tons=220.0, bank_prev=bank)
    bank = state.bank_history[2026]
    year_result, state = clear_year(policy, state, 2027, emissions_tons=150.0, bank_prev=bank)

    assert year_result["p_co2"] == pytest.approx(policy.floor.loc[2027])
    summary, state = finalize_period_if_needed(policy, state, 2027)
    assert summary["finalized"]
    assert summary["surrendered_additional"] == pytest.approx(250.0)
    assert summary["bank_final"] == pytest.approx(35.0)
    assert summary["remaining_obligation"] == pytest.approx(0.0)
    assert not summary["shortage_flag"]
    assert state.bank_history[2027] == pytest.approx(summary["bank_final"])
    assert 2027 in state.finalized_results


def test_true_up_shortage_if_bank_insufficient():
    policy = policy_for_shortage()
    state = allowance_initial_state()

    _, state = clear_year(policy, state, 2025, emissions_tons=400.0, bank_prev=policy.bank0)
    summary, state = finalize_period_if_needed(policy, state, 2025)
    assert summary["finalized"]
    assert summary["bank_final"] == pytest.approx(0.0)
    assert summary["remaining_obligation"] > 0.0
    assert summary["shortage_flag"]
    assert state.bank_history[2025] == pytest.approx(summary["bank_final"])


def test_run_loop_iterates_fixed_point():
    policy = policy_three_year()
    dispatch = LinearDispatch(
        base={2025: 260.0, 2026: 240.0, 2027: 180.0},
        slope={2025: 5.0, 2026: 4.0, 2027: 3.0},
    )

    outputs = run_annual_fixed_point(policy, dispatch, years=[2025, 2026, 2027], price_initial=0.0)

    assert outputs[2025]["p_co2"] >= policy.ccr2_trigger.loc[2025]
    assert outputs[2025]["iterations"] >= 1
    assert "finalize" in outputs[2027]
    assert outputs[2027]["finalize"]["finalized"]
    assert outputs[2027]["finalize"]["bank_final"] >= 0.0


def test_run_loop_skips_when_policy_disabled():
    policy = policy_three_year()
    policy.enabled = False
    dispatch = LinearDispatch(base={2025: 200.0}, slope=0.0)

    outputs = run_annual_fixed_point(policy, dispatch, years=[2025], price_initial=25.0)

    summary = outputs[2025]
    assert summary["p_co2"] == pytest.approx(0.0)
    assert summary["iterations"] == 0
    assert summary["surrendered"] == pytest.approx(0.0)
