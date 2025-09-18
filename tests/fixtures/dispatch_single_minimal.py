"""Fixtures supporting single-region dispatch tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

HOURS_PER_YEAR = 8760.0


@dataclass
class Frames:
    """Minimal container mimicking the frames input expected by the solver."""

    units: pd.DataFrame
    load_mwh: Dict[int, float]


def baseline_units() -> pd.DataFrame:
    """Return a deterministic three-unit system used across the tests."""

    data = [
        {
            "unit_id": "wind-1",
            "fuel": "wind",
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

    return Frames(units=baseline_units(), load_mwh={year: load_mwh})


def infeasible_frames(year: int = 2030) -> Frames:
    """Frames with load exceeding the total available generation."""

    units = baseline_units()
    total_cap = float((units["cap_mw"] * units["availability"] * HOURS_PER_YEAR).sum())
    # Exceed total capacity by ten thousand megawatt-hours.
    excessive_load = total_cap + 10_000.0

    return Frames(units=units, load_mwh={year: excessive_load})

