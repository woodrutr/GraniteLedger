from __future__ import annotations

import importlib
import pytest

pd = pytest.importorskip("pandas")

run_end_to_end_from_frames = importlib.import_module(
    "engine.run_loop"
).run_end_to_end_from_frames
baseline_frames = importlib.import_module(
    "tests.fixtures.dispatch_single_minimal"
).baseline_frames

YEARS = [2025, 2026, 2027]


def _policy_frame(
    floor: float = 0.0,
    cap_scale: float = 1.0,
    *,
    carry_pct: float = 1.0,
    annual_surrender_frac: float = 0.5,
    policy_enabled: bool = True,
) -> pd.DataFrame:
    """Return a policy frame scaled to the dispatch fixture demand levels."""

    caps_base = [800_000.0, 780_000.0, 760_000.0]
    caps = [cap * cap_scale for cap in caps_base]
    ccr1_qty = 120_000.0 * cap_scale
    ccr2_qty = 180_000.0 * cap_scale
    bank0 = 200_000.0 * cap_scale
    records = []
    for idx, year in enumerate(YEARS):
        records.append(
            {
                "year": year,
                "cap_tons": caps[idx],
                "floor_dollars": float(floor),
                "ccr1_trigger": 10.0,
                "ccr1_qty": ccr1_qty,
                "ccr2_trigger": 18.0,
                "ccr2_qty": ccr2_qty,
                "cp_id": "CP1",
                "full_compliance": year == YEARS[-1],
                "bank0": bank0,
                "annual_surrender_frac": float(annual_surrender_frac),
                "carry_pct": float(carry_pct),
                "policy_enabled": bool(policy_enabled),
            }
        )
    return pd.DataFrame(records)


def _three_year_frames(
    loads: list[float] | None = None,
    *,
    floor: float = 0.0,
    cap_scale: float = 1.0,
    carry_pct: float = 1.0,
    annual_surrender_frac: float = 0.5,
    policy_enabled: bool = True,
):
    """Build Frames with three years of demand and a configurable policy."""

    if loads is None:
        loads = [1_000_000.0, 950_000.0, 900_000.0]

    frames = baseline_frames(year=YEARS[0], load_mwh=loads[0])
    demand = pd.DataFrame(
        [
            {"year": year, "region": "default", "demand_mwh": float(load)}
            for year, load in zip(YEARS, loads)
        ]
    )
    frames = frames.with_frame("demand", demand)

    units = frames.units()
    units.loc[units["fuel"] == "gas", "cap_mw"] = 200.0
    frames = frames.with_frame("units", units)
    frames = frames.with_frame(
        "policy",
        _policy_frame(
            floor=floor,
            cap_scale=cap_scale,
            carry_pct=carry_pct,
            annual_surrender_frac=annual_surrender_frac,
            policy_enabled=policy_enabled,
        ),
    )
    return frames


@pytest.fixture
def three_year_outputs():
    frames = _three_year_frames()
    return run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )


def test_three_year_control_period_converges(three_year_outputs):
    """The coupled dispatch/allowance engine should converge quickly."""

    iterations = three_year_outputs.annual["iterations"]
    assert not iterations.empty
    assert int(iterations.max()) <= 10


def test_bank_non_negative_after_compliance(three_year_outputs):
    """Final true-up should not create a negative allowance bank."""

    cp_year = YEARS[-1]
    bank = three_year_outputs.annual.loc[
        three_year_outputs.annual["year"] == cp_year, "bank"
    ].iloc[0]
    assert bank >= -1e-9


def test_emissions_decline_with_stricter_policy():
    """Raising the floor or lowering the cap should reduce emissions."""

    base_frames = _three_year_frames(carry_pct=0.0, annual_surrender_frac=1.0)
    base_outputs = run_end_to_end_from_frames(
        base_frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    base_emissions = base_outputs.annual["emissions_tons"].sum()

    higher_floor_outputs = run_end_to_end_from_frames(
        _three_year_frames(floor=14.0, carry_pct=0.0, annual_surrender_frac=1.0),
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert higher_floor_outputs.annual["emissions_tons"].sum() < base_emissions

    lower_cap_outputs = run_end_to_end_from_frames(
        _three_year_frames(cap_scale=0.35, carry_pct=0.0, annual_surrender_frac=1.0),
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert lower_cap_outputs.annual["emissions_tons"].sum() < base_emissions


def test_disabled_policy_produces_zero_price():
    frames = _three_year_frames()
    frames = frames.with_frame("policy", _policy_frame(policy_enabled=False))

    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=5.0,
        tol=1e-4,
        relaxation=0.8,
    )

    assert outputs.annual["p_co2"].eq(0.0).all()
    assert outputs.annual["surrendered"].eq(0.0).all()
    assert outputs.annual["bank"].eq(0.0).all()


def test_bank_never_negative_across_years():
    frames = _three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    assert outputs.annual["bank"].min() >= -1e-9


def test_bank_accumulates_when_emissions_below_cap():
    low_loads = [400_000.0, 380_000.0, 360_000.0]
    frames = _three_year_frames(loads=low_loads)
    policy = frames.policy().to_policy()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    initial_bank = float(policy.bank0)
    banks = outputs.annual.set_index("year")["bank"]
    assert banks.iloc[0] >= initial_bank
    assert banks.is_monotonic_increasing


def test_bank_disabled_yields_zero_balances():
    frames = _three_year_frames()
    policy = _policy_frame()
    policy["bank_enabled"] = False
    frames = frames.with_frame("policy", policy)

    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    assert outputs.annual["bank"].eq(0.0).all()


def test_ccr_trigger_increases_allowances():
    loads = [600_000.0, 600_000.0, 600_000.0]
    frames = _three_year_frames(loads=loads)
    policy_df = _policy_frame(cap_scale=0.15)
    policy_df['bank0'] = 0.0
    policy_df['ccr1_qty'] = 500_000.0
    frames = frames.with_frame("policy", policy_df)
    policy = frames.policy().to_policy()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    annual = outputs.annual.set_index("year")
    first_year = YEARS[0]
    cap = float(policy.cap.loc[first_year])
    bank0 = float(policy.bank0)
    available = float(annual.loc[first_year, "available_allowances"])
    allowances_issued = available - bank0

    assert allowances_issued > cap
    assert annual.loc[first_year, "p_co2"] == pytest.approx(policy.ccr1_trigger.loc[first_year], rel=1e-4)


def test_compliance_true_up_reconciles_obligations():
    frames = _three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    annual = outputs.annual.set_index("year")
    first_two = annual.loc[YEARS[:-1], "obligation"]
    assert (first_two > 0.0).all()

    final_year = YEARS[-1]
    final_obligation = float(annual.loc[final_year, "obligation"])
    assert final_obligation <= 1e-6
    surrendered_final = float(annual.loc[final_year, "surrendered"])
    required_fraction = 0.5 * float(annual.loc[final_year, "emissions_tons"])
    assert surrendered_final > required_fraction


def test_control_period_mass_balance():
    frames = _three_year_frames()
    policy = frames.policy().to_policy()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    annual = outputs.annual.set_index("year")
    bank_prev = []
    for idx, year in enumerate(YEARS):
        if idx == 0:
            bank_prev.append(float(policy.bank0))
        else:
            bank_prev.append(float(annual.iloc[idx - 1]["bank"]))

    bank_prev_series = pd.Series(bank_prev, index=YEARS)
    allowances_minted = annual["available_allowances"]
    allowances_total = annual["allowances_total"]
    pd.testing.assert_series_equal(
        allowances_total,
        bank_prev_series + allowances_minted,
        check_names=False,
        rtol=1e-9,
        atol=1e-6,
    )

    total_supply = float(policy.bank0) + allowances_minted.sum()
    total_surrendered = annual["surrendered"].sum()
    ending_bank = float(annual.iloc[-1]["bank"])
    remaining = float(annual.iloc[-1]["obligation"])

    assert total_supply == pytest.approx(total_surrendered + ending_bank + remaining)
