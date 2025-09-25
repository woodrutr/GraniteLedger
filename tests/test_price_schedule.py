from __future__ import annotations

from gui.app import CarbonPriceConfig, _merge_price_schedules, _normalize_price_schedule


def test_normalize_price_schedule_handles_malformed_entries() -> None:
    schedule = {
        '2024': '25.5',
        '2022': None,
        2023.0: '10',
        'bad': '4',
        2021: '',
        2020.5: 7.25,
    }

    normalized = _normalize_price_schedule(schedule)

    assert normalized == {2020: 7.25, 2023: 10.0, 2024: 25.5}
    assert list(normalized) == [2020, 2023, 2024]


def test_merge_price_schedules_overrides_and_sorts() -> None:
    base = {'2025': '5', '2024': '3'}
    override = {2026: 7, '2024': '4'}

    merged = _merge_price_schedules(base, override)

    assert merged == {2024: 4.0, 2025: 5.0, 2026: 7.0}
    assert list(merged) == [2024, 2025, 2026]


def test_carbon_price_config_builds_sorted_schedule_from_years() -> None:
    config = CarbonPriceConfig.from_mapping(
        {},
        enabled=True,
        value=12.5,
        schedule=None,
        years=[2023, '2021', 'invalid', 2022.0],
    )

    assert config.schedule == {2021: 12.5, 2022: 12.5, 2023: 12.5}
    assert list(config.schedule) == [2021, 2022, 2023]
