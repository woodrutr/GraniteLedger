"""Tests for the :mod:`io.frames_api` helpers."""

from __future__ import annotations

import importlib
import pytest

pd = pytest.importorskip("pandas")

run_fixed_point_from_frames = importlib.import_module(
    "engine.run_loop"
).run_fixed_point_from_frames
Frames = importlib.import_module("io_loader").Frames
policy_frame_three_year = importlib.import_module(
    "tests.fixtures.annual_minimal"
).policy_frame_three_year
baseline_frames = importlib.import_module(
    "tests.fixtures.dispatch_single_minimal"
).baseline_frames


def test_demand_validation_requires_unique_pairs() -> None:
    """Duplicate region-year entries should raise a clear validation error."""

    frames = Frames(
        {
            "demand": pd.DataFrame(
                [
                    {"year": 2030, "region": "north", "demand_mwh": 10.0},
                    {"year": 2030, "region": "north", "demand_mwh": 12.0},
                ]
            )
        }
    )

    with pytest.raises(ValueError, match="duplicate year/region"):  # demand validation
        frames.demand()


def test_region_coverage_defaults_and_overrides() -> None:
    """Coverage defaults should apply unless a year-specific override exists."""

    coverage = pd.DataFrame(
        [
            {"region": "north", "covered": True},
            {"region": "south", "covered": False},
            {"region": "south", "year": 2035, "covered": True},
        ]
    )

    frames = Frames({"coverage": coverage})

    default = frames.coverage_for_year(2030)
    override = frames.coverage_for_year(2035)

    assert default["north"] is True
    assert default["south"] is False
    assert override["south"] is True


def test_boolean_columns_accept_common_tokens() -> None:
    """Boolean validation should understand mixed types and strings."""

    fuels = pd.DataFrame(
        [
            {"fuel": "gas", "covered": "true"},
            {"fuel": "coal", "covered": "0"},
            {"fuel": "wind", "covered": 1},
        ]
    )

    coverage = pd.DataFrame(
        [
            {"region": "north", "covered": "false"},
            {"region": "south", "covered": 1, "year": 2030},
            {"region": "east", "covered": 0, "year": 2031},
        ]
    )

    policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 100.0,
                "floor_dollars": 3.0,
                "ccr1_trigger": 1.0,
                "ccr1_qty": 1.0,
                "ccr2_trigger": 2.0,
                "ccr2_qty": 2.0,
                "cp_id": "A",
                "full_compliance": "yes",
                "bank0": 0.0,
                "annual_surrender_frac": 0.5,
                "carry_pct": 1.0,
            },
            {
                "year": 2026,
                "cap_tons": 110.0,
                "floor_dollars": 3.0,
                "ccr1_trigger": 1.1,
                "ccr1_qty": 1.1,
                "ccr2_trigger": 2.1,
                "ccr2_qty": 2.1,
                "cp_id": "B",
                "full_compliance": 0,
                "bank0": 0.0,
                "annual_surrender_frac": 0.5,
                "carry_pct": 1.0,
            },
        ]
    )

    frames = Frames({"fuels": fuels, "coverage": coverage, "policy": policy})

    fuels_result = frames.fuels()
    assert fuels_result["covered"].dtype == bool
    assert fuels_result.set_index("fuel")["covered"].to_dict() == {
        "gas": True,
        "coal": False,
        "wind": True,
    }

    coverage_result = frames.coverage()
    assert coverage_result["covered"].dtype == bool
    coverage_lookup = {
        (row.region, row.year): row.covered for row in coverage_result.itertuples(index=False)
    }
    assert coverage_lookup[("north", -1)] is False
    assert coverage_lookup[("south", 2030)] is True
    assert coverage_lookup[("east", 2031)] is False

    policy_spec = frames.policy()
    assert policy_spec.full_compliance_years == {2025}


def test_boolean_columns_reject_invalid_tokens() -> None:
    """Unknown boolean tokens should raise a validation error."""

    frames = Frames(
        {
            "fuels": pd.DataFrame(
                [
                    {"fuel": "gas", "covered": "maybe"},
                ]
            )
        }
    )

    with pytest.raises(ValueError, match="boolean-like values"):
        frames.fuels()


def test_frame_helper_methods_provide_copies_and_defaults() -> None:
    """Ensure helper accessors return defensive copies and optional defaults."""

    base = pd.DataFrame({"value": [1.0, 2.0]})
    frames = Frames({"Example": base})

    assert frames.has_frame("example") is True

    retrieved = frames.frame("EXAMPLE")
    assert retrieved.equals(base)
    assert retrieved is not base

    optional_existing = frames.optional_frame("example")
    assert optional_existing.equals(base)
    assert optional_existing is not base

    assert frames.optional_frame("missing") is None

    default_df = pd.DataFrame({"value": []})
    assert frames.optional_frame("missing", default=default_df) is default_df


def test_policy_spec_round_trip() -> None:
    """The policy accessor should convert to an :class:`RGGIPolicyAnnual`."""

    frames = Frames({"policy": policy_frame_three_year()})

    spec = frames.policy()
    policy = spec.to_policy()

    assert policy.bank0 == pytest.approx(10.0)
    assert policy.annual_surrender_frac == pytest.approx(0.5)
    assert policy.carry_pct == pytest.approx(1.0)
    assert policy.full_compliance_years == {2027}
    assert list(policy.cap.index) == [2025, 2026, 2027]
    assert policy.enabled is True
    assert policy.ccr1_enabled is True
    assert policy.ccr2_enabled is True
    assert policy.control_period_length is None
    assert policy.banking_enabled is True


def test_fixed_point_runs_from_frames() -> None:
    """The engine should operate directly on in-memory frame data."""

    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    demand = pd.DataFrame(
        [
            {"year": 2025, "region": "default", "demand_mwh": 1_200_000.0},
            {"year": 2026, "region": "default", "demand_mwh": 1_050_000.0},
            {"year": 2027, "region": "default", "demand_mwh": 900_000.0},
        ]
    )
    frames = frames.with_frame("demand", demand)
    frames = frames.with_frame("policy", policy_frame_three_year())

    results = run_fixed_point_from_frames(
        frames,
        years=[2025, 2026, 2027],
        price_initial=0.0,
        tol=1e-4,
    )

    assert set(results) == {2025, 2026, 2027}
    assert all("emissions" in year_result for year_result in results.values())
    assert results[2027]["finalize"]["finalized"]


def test_policy_spec_respects_optional_columns() -> None:
    base = policy_frame_three_year()
    base['full_compliance'] = [False, False, False]
    base['policy_enabled'] = [False, False, False]
    base['ccr1_enabled'] = [False, False, False]
    base['ccr2_enabled'] = [True, True, True]
    base['control_period_years'] = [2, 2, 2]
    base['bank_enabled'] = [False, False, False]

    frames = Frames({"policy": base})
    policy = frames.policy().to_policy()

    assert policy.enabled is False
    assert policy.ccr1_enabled is False
    assert policy.ccr2_enabled is True
    assert policy.control_period_length == 2
    assert policy.full_compliance_years == {2026}
    assert policy.banking_enabled is False


def test_policy_disabled_allows_minimal_columns() -> None:
    frames = Frames(
        {
            "policy": pd.DataFrame(
                [
                    {"year": 2025, "policy_enabled": False},
                    {"year": 2026, "policy_enabled": False},
                ]
            )
        }
    )

    spec = frames.policy()
    policy = spec.to_policy()

    assert spec.enabled is False
    assert policy.enabled is False
    assert list(policy.cap.index) == [2025, 2026]
    assert policy.cap.eq(0.0).all()
    assert policy.bank0 == pytest.approx(0.0)
    assert policy.banking_enabled is False
