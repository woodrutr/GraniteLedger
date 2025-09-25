"""Metadata and helpers for mapping region identifiers to human-friendly labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RegionMetadata:
    """Static information describing one of the default model regions."""

    id: int
    code: str
    area: str
    label: str
    aliases: tuple[str, ...] = ()


# The default region definitions used across the prototype rely on the
# 25-region dataset bundled with the electricity and hydrogen inputs.  The
# ``NERCRegion`` codes and their broader geography are documented in the
# hydrogen interactive development sample data set.  The entries below combine
# that information with the numeric identifiers present in ``regions.csv`` so
# the GUI can present meaningful names instead of bare integers.【F:input/hydrogen/all_regions/regions.csv†L1-L26】【F:sample/hydrogen_interactive_dev/input/regions.csv†L1-L34】
DEFAULT_REGION_METADATA: Mapping[int, RegionMetadata] = {
    1: RegionMetadata(
        id=1,
        code="TRE",
        area="South / WSCentral",
        label="Region 1 – TRE (South / WSCentral)",
    ),
    2: RegionMetadata(
        id=2,
        code="FRCC",
        area="South / SAtlantic",
        label="Region 2 – FRCC (South / SAtlantic)",
    ),
    3: RegionMetadata(
        id=3,
        code="SPPC",
        area="Midwest / WNCentral",
        label="Region 3 – SPPC (Midwest / WNCentral)",
    ),
    4: RegionMetadata(
        id=4,
        code="MISC",
        area="Midwest / ENCentral",
        label="Region 4 – MISC (Midwest / ENCentral)",
    ),
    5: RegionMetadata(
        id=5,
        code="MISE",
        area="Midwest / ENCentral",
        label="Region 5 – MISE (Midwest / ENCentral)",
    ),
    6: RegionMetadata(
        id=6,
        code="MISS",
        area="South / ESCentral",
        label="Region 6 – MISS (South / ESCentral)",
    ),
    7: RegionMetadata(
        id=7,
        code="ISNE",
        area="Northeast / NewEngland",
        label="Region 7 – ISNE (Northeast / NewEngland)",
    ),
    8: RegionMetadata(
        id=8,
        code="NYCW",
        area="Northeast / MidAtlantic",
        label="Region 8 – NYCW (Northeast / MidAtlantic, NYISO Downstate)",
        aliases=("nyiso downstate", "nyc"),
    ),
    9: RegionMetadata(
        id=9,
        code="NYUP",
        area="Northeast / MidAtlantic",
        label="Region 9 – NYUP (Northeast / MidAtlantic, NYISO Upstate)",
        aliases=("nyiso", "nyiso upstate"),
    ),
    10: RegionMetadata(
        id=10,
        code="PJME",
        area="Northeast / MidAtlantic",
        label="Region 10 – PJME (Northeast / MidAtlantic)",
    ),
    11: RegionMetadata(
        id=11,
        code="PJMW",
        area="Midwest / ENCentral",
        label="Region 11 – PJMW (Midwest / ENCentral)",
    ),
    12: RegionMetadata(
        id=12,
        code="PJMC",
        area="Midwest / ENCentral",
        label="Region 12 – PJMC (Midwest / ENCentral)",
    ),
    13: RegionMetadata(
        id=13,
        code="PJMD",
        area="South / SAtlantic",
        label="Region 13 – PJMD (South / SAtlantic)",
    ),
    14: RegionMetadata(
        id=14,
        code="SRCA",
        area="South / SAtlantic",
        label="Region 14 – SRCA (South / SAtlantic)",
    ),
    15: RegionMetadata(
        id=15,
        code="SRSE",
        area="South / ESCentral",
        label="Region 15 – SRSE (South / ESCentral)",
    ),
    16: RegionMetadata(
        id=16,
        code="SRCE",
        area="South / ESCentral",
        label="Region 16 – SRCE (South / ESCentral)",
    ),
    17: RegionMetadata(
        id=17,
        code="SPPS",
        area="South / WSCentral",
        label="Region 17 – SPPS (South / WSCentral)",
    ),
    18: RegionMetadata(
        id=18,
        code="SPPC",
        area="Midwest / WNCentral",
        label="Region 18 – SPPC (Midwest / WNCentral)",
    ),
    19: RegionMetadata(
        id=19,
        code="SPPN",
        area="Midwest / WNCentral",
        label="Region 19 – SPPN (Midwest / WNCentral)",
    ),
    20: RegionMetadata(
        id=20,
        code="SRSG",
        area="West / Mountain",
        label="Region 20 – SRSG (West / Mountain)",
    ),
    21: RegionMetadata(
        id=21,
        code="CANO",
        area="West / Pacific",
        label="Region 21 – CANO (West / Pacific)",
    ),
    22: RegionMetadata(
        id=22,
        code="CASO",
        area="West / Pacific",
        label="Region 22 – CASO (West / Pacific)",
    ),
    23: RegionMetadata(
        id=23,
        code="NWPP",
        area="West / Pacific",
        label="Region 23 – NWPP (West / Pacific)",
    ),
    24: RegionMetadata(
        id=24,
        code="RMRG",
        area="West / Mountain",
        label="Region 24 – RMRG (West / Mountain)",
    ),
    25: RegionMetadata(
        id=25,
        code="BASN",
        area="West / Mountain",
        label="Region 25 – BASN (West / Mountain)",
    ),
}

# The additional NYISO aliases use the prototype baseload processing script that
# recodes NYISO balancing authority zones into their ``NYIS`` identifiers, which
# provides the upstream naming convention for the downstate and upstate
# groupings.【F:sample/baseload_data_pipeline/scripts/make_BASR.R†L146-L193】


def region_metadata(region_id: int) -> RegionMetadata | None:
    """Return the metadata entry for ``region_id`` if it exists."""

    return DEFAULT_REGION_METADATA.get(int(region_id))


def region_display_label(value: int | str) -> str:
    """Return a human-readable label for ``value`` suitable for GUI display."""

    if isinstance(value, (bool, int)):
        meta = region_metadata(int(value))
        if meta is not None:
            return meta.label
        return f"Region {int(value)}"
    return str(value)


def _build_alias_map() -> dict[str, int]:
    alias_map: dict[str, int] = {}
    for region_id, meta in DEFAULT_REGION_METADATA.items():
        alias_map[str(region_id)] = region_id
        alias_map[meta.code.lower()] = region_id
        alias_map[meta.label.lower()] = region_id
        alias_map[meta.area.lower()] = region_id
        for alias in meta.aliases:
            alias_map[alias.lower()] = region_id
    return alias_map


_REGION_ALIAS_MAP = _build_alias_map()


def region_alias_map() -> dict[str, int]:
    """Return a copy of the alias map linking strings to canonical region IDs."""

    return dict(_REGION_ALIAS_MAP)


def canonical_region_value(value: Any) -> int | str:
    """Resolve ``value`` into a canonical region identifier when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)

    text = str(value).strip()
    if not text:
        return text

    try:
        return int(text)
    except ValueError:
        pass

    normalized = text.lower()
    match = _REGION_ALIAS_MAP.get(normalized)
    if match is not None:
        return match
    return text


def canonical_region_label(value: Any) -> str:
    """Return the preferred label for ``value`` after canonicalization."""

    resolved = canonical_region_value(value)
    return region_display_label(resolved)

