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
