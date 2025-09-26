from __future__ import annotations

import io
from typing import Mapping, Sequence

import pandas as pd

try:  # pragma: no cover - fallback when executed as a script
    from gui.region_metadata import (
        DEFAULT_REGION_METADATA,
        canonical_region_label,
        canonical_region_value,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from region_metadata import (  # type: ignore[import-not-found]
        DEFAULT_REGION_METADATA,
        canonical_region_label,
        canonical_region_value,
    )


_EMISSIONS_FRAME_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("emissions_by_region", "emissions_by_region.csv"),
    ("emissions", "emissions.csv"),
    ("emissions_region", "emissions_region.csv"),
)


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
        frame["region_canonical"] = frame["region"].apply(canonical_region_value)
        frame["region_label"] = frame["region_canonical"].apply(canonical_region_label)
    else:
        frame["region_canonical"] = pd.NA
        frame["region_label"] = pd.NA

    return frame


def region_selection_options(emissions_df: pd.DataFrame) -> list[tuple[str, int | str]]:
    """Return (label, value) pairs for all canonical regions present in ``emissions_df``."""

    if emissions_df.empty or "region_canonical" not in emissions_df.columns:
        return []

    options: list[tuple[str, int | str]] = []
    seen: set[int | str] = set()
    for canonical in emissions_df["region_canonical"]:
        if pd.isna(canonical):
            continue
        canonical_value = canonical_region_value(canonical)
        if isinstance(canonical_value, str) and canonical_value.strip().lower() == "default":
            continue
        if canonical_value in seen:
            continue
        seen.add(canonical_value)
        display_label = canonical_region_label(canonical_value)
        options.append((display_label, canonical_value))

    options.sort(key=lambda item: item[0].lower())
    if not options:
        fallback = [
            (metadata.label, metadata.id)
            for metadata in DEFAULT_REGION_METADATA.values()
        ]
        fallback.sort(key=lambda item: item[0].lower())
        return fallback
    return options


def filter_emissions_by_regions(
    emissions_df: pd.DataFrame, selected_regions: Sequence[int | str] | None
) -> pd.DataFrame:
    """Return a copy of ``emissions_df`` filtered to the canonical ``selected_regions``."""

    if emissions_df.empty or "region_canonical" not in emissions_df.columns:
        return emissions_df.copy()

    if not selected_regions:
        return emissions_df.copy()

    resolved: set[int | str] = set()
    for value in selected_regions:
        resolved.add(canonical_region_value(value))

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
