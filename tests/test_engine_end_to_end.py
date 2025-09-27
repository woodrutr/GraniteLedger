from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

engine_run_loop = importlib.import_module("engine.run_loop")
run_end_to_end_from_frames = engine_run_loop.run_end_to_end_from_frames
ANNUAL_OUTPUT_COLUMNS = engine_run_loop.ANNUAL_OUTPUT_COLUMNS
baseline_frames = importlib.import_module(
    "tests.fixtures.dispatch_single_minimal"
).baseline_frames
prep = importlib.import_module(
    "src.models.electricity.scripts.preprocessor"
)

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
                "resolution": "annual",
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


def test_progress_callback_reports_each_year():
    frames = _three_year_frames()
    events: list[tuple[str, dict[str, object]]] = []

    def _capture(stage: str, payload: Mapping[str, object]) -> None:
        events.append((stage, dict(payload)))

    run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        progress_cb=_capture,
    )

    stages = [stage for stage, _ in events]
    assert stages.count("run_start") == 1
    assert stages.count("year_start") == len(YEARS)
    assert stages.count("year_complete") == len(YEARS)
    assert "iteration" in stages

    first_year_payload = next(payload for stage, payload in events if stage == "year_start")
    assert int(first_year_payload.get("index", -1)) == 0
    assert int(first_year_payload.get("total_years", 0)) == len(YEARS)

    final_payload = next(payload for stage, payload in reversed(events) if stage == "year_complete")
    assert int(final_payload.get("index", -1)) == len(YEARS) - 1
    assert "price" in final_payload
    assert "iterations" in final_payload


def test_debug_logging_includes_full_metrics(caplog):
    frames = _three_year_frames()

    with caplog.at_level(logging.DEBUG, logger="engine.run_loop"):
        run_end_to_end_from_frames(
            frames,
            years=YEARS,
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.8,
        )

    prefix = "allowance_year_metrics "
    metrics: list[dict[str, object]] = []
    for record in caplog.records:
        if record.name != "engine.run_loop":
            continue
        message = record.getMessage()
        if not message.startswith(prefix):
            continue
        metrics.append(json.loads(message[len(prefix) :]))

    assert len(metrics) == len(YEARS)
    required_fields = {
        "price_raw",
        "reserve_cap",
        "ecr_trigger",
        "ccr1_trigger",
        "ccr2_trigger",
        "reserve_budget",
        "reserve_withheld",
        "ccr1_release",
        "ccr2_release",
        "bank_in",
        "bank_out",
        "available_allowances",
        "emissions",
        "shortage_flag",
    }
    for payload in metrics:
        missing = required_fields.difference(payload)
        assert not missing, f"missing fields: {sorted(missing)}"


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
    assert outputs.annual["surrender"].eq(0.0).all()
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
    policy_df["bank0"] = 0.0
    policy_df["ccr1_qty"] = 500_000.0
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
    available = float(annual.loc[first_year, "allowances_minted"])
    allowances_issued = available - bank0

    assert allowances_issued > cap
    assert annual.loc[first_year, "p_co2"] == pytest.approx(
        policy.ccr1_trigger.loc[first_year], rel=1e-4
    )


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
    surrendered_final = float(annual.loc[final_year, "surrender"])
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
    allowances_minted = annual["allowances_minted"]
    allowances_total = annual["allowances_available"]
    pd.testing.assert_series_equal(
        allowances_total,
        bank_prev_series + allowances_minted,
        check_names=False,
        rtol=1e-9,
        atol=1e-6,
    )

    total_supply = float(policy.bank0) + allowances_minted.sum()
    total_surrendered = annual["surrender"].sum()
    ending_bank = float(annual.iloc[-1]["bank"])
    remaining = float(annual.iloc[-1]["obligation"])

    assert total_supply == pytest.approx(total_surrendered + ending_bank + remaining)


def test_annual_output_schema_matches_spec():
    frames = _three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    assert list(outputs.annual.columns) == ANNUAL_OUTPUT_COLUMNS


def test_allowance_price_column_matches_allowance_component():
    frames = _three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    annual = outputs.annual
    assert "allowance_price" in annual.columns
    pd.testing.assert_series_equal(
        annual["allowance_price"],
        annual["p_co2_all"],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_annual_output_csv_schema(tmp_path):
    frames = _three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    outdir = tmp_path / "results"
    outputs.to_csv(outdir)

    annual_csv = pd.read_csv(outdir / "annual.csv")
    assert list(annual_csv.columns) == ANNUAL_OUTPUT_COLUMNS


def test_daily_resolution_matches_annual_totals():
    base_frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    annual_policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 10_000_000.0,
                "floor_dollars": 0.0,
                "ccr1_trigger": 0.0,
                "ccr1_qty": 0.0,
                "ccr2_trigger": 0.0,
                "ccr2_qty": 0.0,
                "cp_id": "CP1",
                "full_compliance": True,
                "bank0": 0.0,
                "annual_surrender_frac": 1.0,
                "carry_pct": 1.0,
                "policy_enabled": True,
                "bank_enabled": True,
                "resolution": "annual",
            }
        ]
    )
    frames_annual = base_frames.with_frame("policy", annual_policy)

    annual_outputs = run_end_to_end_from_frames(
        frames_annual,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    demand_total = float(base_frames.demand()["demand_mwh"].iloc[0])
    demand_period = pd.DataFrame(
        [
            {"year": 2025001, "region": "default", "demand_mwh": demand_total / 2.0},
            {"year": 2025002, "region": "default", "demand_mwh": demand_total / 2.0},
        ]
    )

    frames_daily = base_frames.with_frame("demand", demand_period)

    daily_records = []
    for period in (1, 2):
        period_key = 2025 * 1000 + period
        daily_records.append(
            {
                "year": period_key,
                "cap_tons": 5_000_000.0,
                "floor_dollars": 0.0,
                "ccr1_trigger": 0.0,
                "ccr1_qty": 0.0,
                "ccr2_trigger": 0.0,
                "ccr2_qty": 0.0,
                "cp_id": "CP1",
                "full_compliance": period == 2,
                "bank0": 0.0,
                "annual_surrender_frac": 1.0,
                "carry_pct": 1.0,
                "policy_enabled": True,
                "bank_enabled": True,
                "resolution": "daily",
            }
        )

    daily_policy = pd.DataFrame(daily_records)
    frames_daily = frames_daily.with_frame("policy", daily_policy)

    daily_outputs = run_end_to_end_from_frames(
        frames_daily,
        years=[2025001, 2025002],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    annual_df = annual_outputs.annual.set_index("year").sort_index()
    daily_df = daily_outputs.annual.set_index("year").sort_index()

    pd.testing.assert_index_equal(daily_df.index, annual_df.index)
    pd.testing.assert_frame_equal(daily_df, annual_df, check_exact=False, atol=1e-6, rtol=1e-9)

    def _sorted(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        return df.sort_values(columns).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        _sorted(daily_outputs.emissions_by_region, ["year", "region"]),
        _sorted(annual_outputs.emissions_by_region, ["year", "region"]),
        check_exact=False,
        atol=1e-6,
        rtol=1e-9,
    )
    pd.testing.assert_frame_equal(
        _sorted(daily_outputs.price_by_region, ["year", "region"]),
        _sorted(annual_outputs.price_by_region, ["year", "region"]),
        check_exact=False,
        atol=1e-6,
        rtol=1e-9,
    )
    pd.testing.assert_frame_equal(
        _sorted(daily_outputs.flows, ["year", "from_region", "to_region"]),
        _sorted(annual_outputs.flows, ["year", "from_region", "to_region"]),
        check_exact=False,
        atol=1e-6,
        rtol=1e-9,
    )


def test_engine_outputs_expose_emissions_mappings() -> None:
    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 1_000_000.0,
                "floor_dollars": 0.0,
                "ccr1_trigger": 0.0,
                "ccr1_qty": 0.0,
                "ccr2_trigger": 0.0,
                "ccr2_qty": 0.0,
                "cp_id": "CP1",
                "full_compliance": True,
                "bank0": 0.0,
                "annual_surrender_frac": 1.0,
                "carry_pct": 1.0,
                "policy_enabled": True,
                "resolution": "annual",
            }
        ]
    )
    frames = frames.with_frame("policy", policy)
    outputs = run_end_to_end_from_frames(
        frames,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    totals = dict(outputs.emissions_total)
    assert 2025 in totals
    total_value = float(totals[2025])
    assert total_value == pytest.approx(outputs.annual.loc[outputs.annual['year'] == 2025, 'emissions_tons'].iloc[0])

    regional_map = {region: dict(years) for region, years in outputs.emissions_by_region_map.items()}
    assert 'default' in regional_map
    assert 2025 in regional_map['default']
    regional_value = float(regional_map['default'][2025])
    df_value = float(
        outputs.emissions_by_region.loc[
            (outputs.emissions_by_region['year'] == 2025)
            & (outputs.emissions_by_region['region'] == 'default'),
            'emissions_tons',
        ].iloc[0]
    )
    assert regional_value == pytest.approx(df_value)


def test_engine_audits_pass_for_three_year_run(three_year_outputs):
    audits = three_year_outputs.audits
    assert isinstance(audits, dict)
    assert audits["emissions"]["passed"]
    assert audits["generation_capacity"]["passed"]
    assert audits["cost"]["passed"]
    assert float(audits["emissions"].get("max_region_gap", 1.0)) <= 1e-6
    assert float(audits["emissions"].get("max_fuel_gap", 1.0)) <= 1e-6


def test_generation_and_capacity_tables_available(three_year_outputs):
    generation_df = three_year_outputs.generation_by_fuel
    capacity_df = three_year_outputs.capacity_by_fuel
    assert not generation_df.empty
    assert not capacity_df.empty
    capacity_margins = three_year_outputs.audits["generation_capacity"]["capacity_margin"]
    assert capacity_margins
    assert all(float(margin) >= -1e-6 for margin in capacity_margins.values())
    assert isinstance(
        three_year_outputs.audits["generation_capacity"].get("stranded_units"), list
    )


def test_cost_breakdown_reconciles_components(three_year_outputs):
    cost_df = three_year_outputs.cost_by_fuel
    assert not cost_df.empty
    residual = (
        cost_df["total_cost"]
        - (
            cost_df["variable_cost"]
            + cost_df["allowance_cost"]
            + cost_df["carbon_price_cost"]
        )
    ).abs()
    assert float(residual.max()) <= 1e-6


def test_emissions_by_fuel_matches_totals(three_year_outputs):
    fuel_totals = (
        three_year_outputs.emissions_by_fuel.groupby("year")["emissions_tons"].sum()
    )
    annual_totals = (
        three_year_outputs.annual.set_index("year")["emissions_tons"].astype(float)
    )
    combined_years = sorted(set(fuel_totals.index) | set(annual_totals.index))
    max_gap = (
        fuel_totals.reindex(combined_years, fill_value=0.0)
        - annual_totals.reindex(combined_years, fill_value=0.0)
    ).abs().max()
    assert float(max_gap) <= 1e-6


def test_engine_outputs_include_zero_rows_for_empty_regions() -> None:
    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    demand = frames.demand()
    demand_extra = pd.DataFrame(
        [{"year": 2025, "region": "unused", "demand_mwh": 0.0}]
    )
    demand = pd.concat([demand, demand_extra], ignore_index=True)
    frames = frames.with_frame("demand", demand)
    policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 1_000_000.0,
                "floor_dollars": 0.0,
                "ccr1_trigger": 0.0,
                "ccr1_qty": 0.0,
                "ccr2_trigger": 0.0,
                "ccr2_qty": 0.0,
                "cp_id": "CP1",
                "full_compliance": True,
                "bank0": 0.0,
                "annual_surrender_frac": 1.0,
                "carry_pct": 1.0,
                "policy_enabled": True,
                "resolution": "annual",
            }
        ]
    )
    frames = frames.with_frame("policy", policy)

    outputs = run_end_to_end_from_frames(
        frames,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    region_map = {region: dict(years) for region, years in outputs.emissions_by_region_map.items()}
    assert 'unused' in region_map
    assert region_map['unused'][2025] == pytest.approx(0.0)

    unused_rows = outputs.emissions_by_region[
        (outputs.emissions_by_region['year'] == 2025)
        & (outputs.emissions_by_region['region'] == 'unused')
    ]
    assert not unused_rows.empty
    assert unused_rows['emissions_tons'].iloc[0] == pytest.approx(0.0)


def test_zero_cap_policy_still_enforced():
    frames = _three_year_frames()

    zero_cap_settings = SimpleNamespace(
        years=YEARS,
        start_year=YEARS[0],
        carbon_cap=0.0,
        carbon_policy_enabled=True,
        carbon_allowance_start_bank=0.0,
    )

    default_policy = prep._default_policy_frame(zero_cap_settings)
    zero_cap_policy = _policy_frame(cap_scale=0.0)
    zero_cap_policy["policy_enabled"] = list(default_policy["policy_enabled"])
    zero_cap_policy["annual_surrender_frac"] = 1.0
    assert zero_cap_policy["policy_enabled"].all()

    frames = frames.with_frame("policy", zero_cap_policy)
    frames_store = prep.FrameStore(
        frames, carbon_policy_enabled=prep._is_carbon_policy_enabled(zero_cap_settings)
    )
    frames = frames_store.to_frames()
    assert frames.carbon_policy_enabled

    policy_spec = frames.policy()
    assert policy_spec.enabled

    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    annual = outputs.annual.set_index("year")
    assert (annual["obligation"] > 0.0).any()
    assert annual["p_co2"].gt(0.0).any()
