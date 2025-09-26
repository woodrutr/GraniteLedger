from __future__ import annotations

import io
from typing import Mapping, Sequence

import pandas as pd

NAType = type(pd.NA)

try:  # pragma: no cover - fallback when executed as a script
    from gui.region_metadata import (
        canonical_region_label,
        canonical_region_value,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from region_metadata import (  # type: ignore[import-not-found]
        canonical_region_label,
        canonical_region_value,
    )


_EMISSIONS_FRAME_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("emissions_by_region", "emissions_by_region.csv"),
    ("emissions", "emissions.csv"),
    ("emissions_region", "emissions_region.csv"),
)


def _canonical_region_id(value: object) -> int | NAType:
    """Return the canonical integer region identifier or ``pd.NA`` when unknown."""

    canonical = canonical_region_value(value)
    if isinstance(canonical, bool):
        canonical = int(canonical)
    if isinstance(canonical, (int, float)) and not isinstance(canonical, bool):
        return int(canonical)
    return pd.NA


def _canonical_region_label(value: object) -> str | NAType:
    """Return the label for a canonical region identifier or ``pd.NA``."""

    if pd.isna(value):
        return pd.NA
    try:
        canonical_id = int(value)
    except (TypeError, ValueError):
        return pd.NA
    return canonical_region_label(canonical_id)


def load_emissions_data(result: Mapping[str, object] | None) -> pd.DataFrame:
    """Return the emissions-by-region frame from ``result`` when available."""

    if not isinstance(result, Mapping):
        return pd.DataFrame()

    frame: pd.DataFrame | None = None
    for key, _ in _EMISSIONS_FRAME_CANDIDATES:
        candidate = result.get(key)
        if isinstance(candidate, pd.DataFrame):
            frame = candidate.copy()
            break

    if frame is None:
        csv_files = result.get("csv_files")
        if isinstance(csv_files, Mapping):
            for key, csv_name in _EMISSIONS_FRAME_CANDIDATES:
                raw = csv_files.get(csv_name) or csv_files.get(f"{key}.csv")
                if isinstance(raw, (bytes, bytearray)):
                    try:
                        frame = pd.read_csv(io.BytesIO(raw))
                    except Exception:  # pragma: no cover - best effort load
                        continue
                    else:
                        break

    if frame is None or frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    if "emissions_tons" not in frame.columns:
        return pd.DataFrame()

    frame["emissions_tons"] = pd.to_numeric(frame["emissions_tons"], errors="coerce")
    frame = frame.dropna(subset=["emissions_tons"])

    if "region" in frame.columns:
        frame["region_canonical"] = frame["region"].apply(_canonical_region_id)
        frame["region_label"] = frame["region_canonical"].apply(_canonical_region_label)
    else:
        frame["region_canonical"] = pd.NA
        frame["region_label"] = pd.NA

    return frame


def region_selection_options(emissions_df: pd.DataFrame) -> list[tuple[str, int]]:
    """Return (label, value) pairs for canonical regions present in ``emissions_df``."""

    if emissions_df.empty or "region_canonical" not in emissions_df.columns:
        return []

    options: list[tuple[str, int]] = []
    seen: set[int] = set()
    for canonical, label in zip(
        emissions_df["region_canonical"], emissions_df.get("region_label", [])
    ):
        if pd.isna(canonical):
            continue
        try:
            canonical_id = int(canonical)
        except (TypeError, ValueError):
            continue
        if canonical_id in seen:
            continue
        seen.add(canonical_id)
        display_label = (
            str(label)
            if isinstance(label, str) and label
            else canonical_region_label(canonical_id)
        )
        options.append((display_label, canonical_id))

    options.sort(key=lambda item: item[0].lower())
    return options


def filter_emissions_by_regions(
    emissions_df: pd.DataFrame, selected_regions: Sequence[int | str] | None
) -> pd.DataFrame:
    """Return a copy of ``emissions_df`` filtered to the canonical ``selected_regions``."""

    if emissions_df.empty or "region_canonical" not in emissions_df.columns:
        return emissions_df.copy()

    if not selected_regions:
        return emissions_df.copy()

    resolved: set[int] = set()
    for value in selected_regions:
        canonical = canonical_region_value(value)
        if isinstance(canonical, bool):
            canonical = int(canonical)
        if isinstance(canonical, (int, float)) and not isinstance(canonical, bool):
            resolved.add(int(canonical))

    if not resolved:
        return emissions_df.iloc[0:0].copy()

    return emissions_df[emissions_df["region_canonical"].isin(resolved)].copy()


def summarize_emissions_totals(emissions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate emissions totals by canonical region label for bar chart visualisation."""

    if emissions_df.empty or "emissions_tons" not in emissions_df.columns:
        return pd.DataFrame(columns=["emissions_tons"])

    working = emissions_df.copy()
    if "region_label" not in working.columns:
        working["region_label"] = working.get("region", "region")

    grouped = (
        working.dropna(subset=["region_label"])
        .groupby("region_label")
        ["emissions_tons"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )

    if grouped.empty:
        return pd.DataFrame(columns=["emissions_tons"])

    summary = grouped.to_frame()
    summary.index.name = "region"
    return summary
