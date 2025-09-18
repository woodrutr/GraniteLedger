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
            'demand': pd.DataFrame(
                [
                    {'year': 2030, 'region': 'north', 'demand_mwh': 10.0},
                    {'year': 2030, 'region': 'north', 'demand_mwh': 12.0},
                ]
            )
        }
    )

    with pytest.raises(ValueError, match='duplicate year/region'):  # demand validation
        frames.demand()


def test_region_coverage_defaults_and_overrides() -> None:
    """Coverage defaults should apply unless a year-specific override exists."""

    coverage = pd.DataFrame(
        [
            {'region': 'north', 'covered': True},
            {'region': 'south', 'covered': False},
            {'region': 'south', 'year': 2035, 'covered': True},
        ]
    )

    frames = Frames({'coverage': coverage})

    default = frames.coverage_for_year(2030)
    override = frames.coverage_for_year(2035)

    assert default['north'] is True
    assert default['south'] is False
    assert override['south'] is True


def test_policy_spec_round_trip() -> None:
    """The policy accessor should convert to an :class:`RGGIPolicyAnnual`."""

    frames = Frames({'policy': policy_frame_three_year()})

    spec = frames.policy()
    policy = spec.to_policy()

    assert policy.bank0 == pytest.approx(10.0)
    assert policy.annual_surrender_frac == pytest.approx(0.5)
    assert policy.carry_pct == pytest.approx(1.0)
    assert policy.full_compliance_years == {2027}
    assert list(policy.cap.index) == [2025, 2026, 2027]


def test_fixed_point_runs_from_frames() -> None:
    """The engine should operate directly on in-memory frame data."""

    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    demand = pd.DataFrame(
        [
            {'year': 2025, 'region': 'default', 'demand_mwh': 1_200_000.0},
            {'year': 2026, 'region': 'default', 'demand_mwh': 1_050_000.0},
            {'year': 2027, 'region': 'default', 'demand_mwh': 900_000.0},
        ]
    )
    frames = frames.with_frame('demand', demand)
    frames = frames.with_frame('policy', policy_frame_three_year())

    results = run_fixed_point_from_frames(
        frames,
        years=[2025, 2026, 2027],
        price_initial=0.0,
        tol=1e-4,
    )

    assert set(results) == {2025, 2026, 2027}
    assert all('emissions' in year_result for year_result in results.values())
    assert results[2027]['finalize']['finalized']
