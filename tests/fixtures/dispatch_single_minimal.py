"""Fixtures supporting single-region dispatch tests."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from io_loader import Frames

HOURS_PER_YEAR = 8760.0


def baseline_units() -> pd.DataFrame:
    """Return a deterministic three-unit system used across the tests."""

    data = [
        {
            "unit_id": "wind-1",
            "fuel": "wind",
            "region": "default",
            "cap_mw": 50.0,
            "availability": 0.5,
            "hr_mmbtu_per_mwh": 0.0,
            "vom_per_mwh": 0.0,
            "fuel_price_per_mmbtu": 0.0,
            "ef_ton_per_mwh": 0.0,
        },
        {
            "unit_id": "coal-1",
            "fuel": "coal",
            "region": "default",
            "cap_mw": 80.0,
            "availability": 0.9,
            "hr_mmbtu_per_mwh": 9.0,
            "vom_per_mwh": 1.5,
            "fuel_price_per_mmbtu": 1.8,
            "ef_ton_per_mwh": 1.0,
        },
        {
            "unit_id": "gas-1",
            "fuel": "gas",
            "region": "default",
            "cap_mw": 70.0,
            "availability": 0.85,
            "hr_mmbtu_per_mwh": 7.0,
            "vom_per_mwh": 2.0,
            "fuel_price_per_mmbtu": 2.5,
            "ef_ton_per_mwh": 0.45,
        },
    ]

    return pd.DataFrame(data)


def baseline_frames(year: int = 2030, load_mwh: float = 1_000_000.0) -> Frames:
    """Construct frames with the baseline unit data and supplied load."""

    demand = pd.DataFrame(
        [{"year": year, "region": "default", "demand_mwh": float(load_mwh)}]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "wind", "covered": False},
            {"fuel": "coal", "covered": True},
            {"fuel": "gas", "covered": True},
        ]
    )
    transmission = pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])

    return Frames(
        {
            "units": baseline_units(),
            "demand": demand,
            "fuels": fuels,
            "transmission": transmission,
        }
    )


def infeasible_frames(year: int = 2030) -> Frames:
    """Frames with load exceeding the total available generation."""

    base = baseline_frames(year=year)
    units = base.units()
    total_cap = float((units["cap_mw"] * units["availability"] * HOURS_PER_YEAR).sum())
    demand = base.demand()
    demand.loc[demand["year"] == year, "demand_mwh"] = total_cap + 10_000.0

    return base.with_frame("demand", demand)
