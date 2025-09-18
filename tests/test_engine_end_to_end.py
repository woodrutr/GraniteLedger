import pytest
from engine.run_loop import run_end_to_end_from_frames
from tests.fixtures.dispatch_single_minimal import baseline_frames

pd = pytest.importorskip("pandas")


def _three_year_demand() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2025, "region": "default", "demand_mwh": 1_200_000.0},
            {"year": 2026, "region": "default", "demand_mwh": 1_050_000.0},
            {"year": 2027, "region": "default", "demand_mwh": 900_000.0},
        ]
    )


def _policy_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for year in (2025, 2026, 2027):
        records.append(
            {
                "year": year,
                "cap_tons": 100.0 if year == 2025 else (90.0 if year == 2026 else 250.0),
                "floor_dollars": 4.0,
                "ccr1_trigger": 7.0,
                "ccr1_qty": 30.0,
                "ccr2_trigger": 13.0,
                "ccr2_qty": 60.0,
                "cp_id": "CP1",
                "full_compliance": year == 2027,
                "bank0": 10.0,
                "annual_surrender_frac": 0.5,
                "carry_pct": 1.0,
            }
        )
    return pd.DataFrame(records)


def test_end_to_end_runs_from_frames() -> None:
    """The end-to-end engine runner should operate on frame containers."""

    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    frames = frames.with_frame("demand", _three_year_demand())
    frames = frames.with_frame("policy", _policy_frame())

    results = run_end_to_end_from_frames(
        frames,
        years=[2025, 2026, 2027],
        price_initial=0.0,
        tol=1e-4,
    )

    assert set(results) == {2025, 2026, 2027}
    assert all("emissions" in year_result for year_result in results.values())
    assert results[2027]["finalize"]["finalized"]
