"""Generation standard policy definitions for dispatch."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING, cast

try:  # pragma: no cover - exercised when pandas missing
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before continuing."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for policy.generation_standard; install it with `pip install pandas`."
        )


def _coerce_enabled_regions(enabled: Iterable[str] | None) -> frozenset[str] | None:
    """Return ``enabled`` coerced to a frozenset of normalised region labels."""

    if enabled is None:
        return None

    normalized = [str(region) for region in enabled if region is not None]
    return frozenset(normalized)


def _read_requirement_csv(path: str | Path) -> pd.DataFrame:
    """Read a requirement CSV with years in the first column."""

    _ensure_pandas()

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        return pd.DataFrame()

    working = dataframe.copy(deep=True)
    year_column = working.columns[0]
    working = working.rename(columns={year_column: "year"})
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    if working["year"].isna().any():
        raise ValueError(f"{Path(path)} contains non-numeric year values")
    working["year"] = working["year"].astype(int)

    value_columns = [column for column in working.columns if column != "year"]
    for column in value_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.fillna(0.0)
    working[value_columns] = working[value_columns].astype(float)
    working[value_columns] = working[value_columns].where(
        working[value_columns].notna(), 0.0
    )

    renamed_columns = ["year"] + [str(column).strip() for column in value_columns]
    working.columns = renamed_columns
    working = working.set_index("year").sort_index()
    return working


def _normalize_year_table(table: pd.DataFrame, *, fill_value: float) -> pd.DataFrame:
    """Return ``table`` keyed by integer years with numeric region values."""

    _ensure_pandas()

    if not isinstance(table, pd.DataFrame):
        raise TypeError("generation standard tables must be provided as pandas DataFrames")

    if table.empty:
        empty = table.copy(deep=True)
        empty.index = pd.Index([], name="year")
        empty.columns = [str(column) for column in empty.columns]
        return empty.astype(float)

    working = table.copy(deep=True)
    if "year" in working.columns:
        working["year"] = pd.to_numeric(working["year"], errors="coerce")
        if working["year"].isna().any():
            raise ValueError("generation standard tables must specify integer years")
        working["year"] = working["year"].astype(int)
        working = working.set_index("year")
    else:
        working.index = pd.to_numeric(working.index, errors="coerce")
        if working.index.isna().any():
            raise ValueError("generation standard tables must specify integer years")
        working.index = working.index.astype(int)

    if working.index.duplicated().any():
        duplicates = sorted(working.index[working.index.duplicated()].unique())
        raise ValueError(f"generation standard tables contain duplicate years: {duplicates}")

    working = working.sort_index()
    working.index.name = "year"

    working.columns = [str(column).strip() for column in working.columns]
    numeric = working.apply(pd.to_numeric, errors="coerce").fillna(fill_value)

    return numeric.astype(float)


def _prepare_capacity_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise capacity requirements from ``table`` in megawatts."""

    if table is None:
        return None
    normalized = _normalize_year_table(table, fill_value=0.0)
    return normalized.clip(lower=0.0)


def _prepare_share_table(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise generation share requirements from ``table`` as fractions."""

    if table is None:
        return None
    normalized = _normalize_year_table(table, fill_value=0.0)
    if not normalized.empty and (normalized > 1.0).any().any():
        normalized = normalized / 100.0
    return normalized.clip(lower=0.0, upper=1.0)


@dataclass(frozen=True)
class TechnologyStandard:
    """Requirements for a specific technology across regions."""

    technology: str
    capacity_table: pd.DataFrame | None = None
    generation_table: pd.DataFrame | None = None
    enabled_regions: frozenset[str] | None = None
    technology_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _ensure_pandas()

        object.__setattr__(self, "technology", str(self.technology))
        object.__setattr__(self, "technology_key", str(self.technology).strip().lower())

        normalized_enabled = _coerce_enabled_regions(self.enabled_regions)
        object.__setattr__(self, "enabled_regions", normalized_enabled)

        normalized_capacity = _prepare_capacity_table(self.capacity_table)
        normalized_generation = _prepare_share_table(self.generation_table)

        object.__setattr__(self, "capacity_table", normalized_capacity)
        object.__setattr__(self, "generation_table", normalized_generation)

    @classmethod
    def from_csvs(
        cls,
        technology: str,
        *,
        capacity_csv: str | Path | None = None,
        generation_csv: str | Path | None = None,
        enabled_regions: Iterable[str] | None = None,
    ) -> TechnologyStandard:
        """Build a :class:`TechnologyStandard` from CSV inputs."""

        capacity_table = _read_requirement_csv(capacity_csv) if capacity_csv else None
        generation_table = (
            _read_requirement_csv(generation_csv) if generation_csv else None
        )

        return cls(
            technology=technology,
            capacity_table=capacity_table,
            generation_table=generation_table,
            enabled_regions=_coerce_enabled_regions(enabled_regions),
        )

    def regions(self) -> Sequence[str]:
        """Return regions covered by this standard."""

        if self.enabled_regions is not None:
            return sorted(self.enabled_regions)

        regions: set[str] = set()
        if self.capacity_table is not None:
            regions.update(str(column) for column in self.capacity_table.columns)
        if self.generation_table is not None:
            regions.update(str(column) for column in self.generation_table.columns)
        return sorted(regions)

    def _region_enabled(self, region: str) -> bool:
        if self.enabled_regions is None:
            return True
        return str(region) in self.enabled_regions

    def capacity_requirement(self, year: int, region: str) -> float:
        """Return the minimum required capacity in megawatts for ``region``."""

        if not self._region_enabled(region):
            return 0.0
        if self.capacity_table is None or self.capacity_table.empty:
            return 0.0
        try:
            value = float(self.capacity_table.loc[int(year), str(region)])
        except KeyError:
            return 0.0
        return max(0.0, value)

    def generation_share(self, year: int, region: str) -> float:
        """Return the minimum required generation share as a fraction."""

        if not self._region_enabled(region):
            return 0.0
        if self.generation_table is None or self.generation_table.empty:
            return 0.0
        try:
            value = float(self.generation_table.loc[int(year), str(region)])
        except KeyError:
            return 0.0
        return min(max(0.0, value), 1.0)


@dataclass(frozen=True)
class TechnologyRegionRequirement:
    """Technology requirement resolved to a specific region and year."""

    technology: str
    region: str
    capacity_mw: float = 0.0
    generation_share: float = 0.0
    technology_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        technology = str(self.technology)
        region = str(self.region)
        capacity = max(0.0, float(self.capacity_mw))
        share = max(0.0, float(self.generation_share))
        if share > 1.0:
            share = 1.0

        key = self.technology_key or technology.strip().lower()

        object.__setattr__(self, "technology", technology)
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "capacity_mw", capacity)
        object.__setattr__(self, "generation_share", share)
        object.__setattr__(self, "technology_key", key)


class GenerationStandardPolicy:
    """Container managing generation standards across technologies."""

    def __init__(self, standards: Iterable[TechnologyStandard] | None = None) -> None:
        self._standards: dict[str, TechnologyStandard] = {}
        if standards:
            for standard in standards:
                self.add_standard(standard)

    def add_standard(self, standard: TechnologyStandard) -> None:
        """Register ``standard`` with the policy."""

        if not isinstance(standard, TechnologyStandard):
            raise TypeError("standard must be a TechnologyStandard instance")

        key = standard.technology_key
        if key in self._standards:
            raise ValueError(
                f"Technology '{standard.technology}' already has a registered generation standard"
            )
        self._standards[key] = standard

    def technologies(self) -> Sequence[str]:
        """Return the list of technology labels managed by the policy."""

        return [standard.technology for standard in self._standards.values()]

    def requirements_for_year(self, year: int) -> list[TechnologyRegionRequirement]:
        """Return resolved requirements for ``year`` across regions."""

        requirements: list[TechnologyRegionRequirement] = []
        for standard in self._standards.values():
            for region in standard.regions():
                capacity = standard.capacity_requirement(year, region)
                share = standard.generation_share(year, region)
                if capacity <= 0.0 and share <= 0.0:
                    continue
                requirements.append(
                    TechnologyRegionRequirement(
                        technology=standard.technology,
                        region=region,
                        capacity_mw=capacity,
                        generation_share=share,
                        technology_key=standard.technology_key,
                    )
                )
        return requirements


__all__ = [
    "GenerationStandardPolicy",
    "TechnologyRegionRequirement",
    "TechnologyStandard",
]
