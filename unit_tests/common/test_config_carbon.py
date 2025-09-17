"""Tests for configuration carbon policy handling."""

from pathlib import Path

import pytest

pytest.importorskip('pandas')

from definitions import PROJECT_ROOT
from src.common.config_setup import Config_settings, SHORT_TON_TO_METRIC_TON


def test_carbon_cap_groups_from_table(tmp_path):
    """Carbon cap groups defined in the table should populate the default group."""

    source_config = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    config_contents = source_config.read_text()
    config_contents = config_contents.replace(
        'cap = "none" # Set to "none" for no cap',
        'cap = 1234.5 # numeric cap for test',
        1,
    )

    temp_config_path = tmp_path / 'run_config.toml'
    temp_config_path.write_text(config_contents)

    settings = Config_settings(temp_config_path, test=True)

    assert settings.default_cap_group is not None
    assert settings.carbon_cap_groups[0] is settings.default_cap_group
    assert settings.carbon_cap == pytest.approx(1234.5)
    assert settings.default_cap_group.allowance_procurement == settings.carbon_allowance_procurement
    assert settings.default_cap_group.bank_enabled == settings.carbon_allowance_bank_enabled
    assert (
        settings.default_cap_group.allow_borrowing
        == settings.carbon_allowance_allow_borrowing
    )


def test_legacy_carbon_keys_create_default_group(tmp_path):
    """Legacy carbon policy keys should build a default group representation."""

    source_config = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    lines = source_config.read_text().splitlines()
    filtered_lines = []
    skip_cap_group = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[[carbon_cap_groups]]'):
            skip_cap_group = True
            continue
        if skip_cap_group:
            if stripped == '':
                skip_cap_group = False
            continue
        filtered_lines.append(line)

    short_ton_cap = 5432.1
    legacy_block = [
        '',
        '[carbon_policy]',
        f'carbon_cap = {short_ton_cap}',
        '',
        'carbon_allowance_procurement = { "2025" = 1.5 }',
        'carbon_allowance_start_bank = 2.75',
        'carbon_allowance_bank_enabled = false',
        'carbon_allowance_allow_borrowing = true',
    ]
    config_contents = '\n'.join(filtered_lines + legacy_block) + '\n'

    temp_config_path = tmp_path / 'run_config.toml'
    temp_config_path.write_text(config_contents)

    settings = Config_settings(temp_config_path, test=True)

    assert len(settings.carbon_cap_groups) == 1
    group = settings.default_cap_group
    assert group is not None
    assert group.cap == pytest.approx(short_ton_cap * SHORT_TON_TO_METRIC_TON)
    assert group.allowance_procurement == {2025: 1.5}
    assert group.start_bank == pytest.approx(2.75)
    assert group.bank_enabled is False
    assert group.allow_borrowing is True
    assert group.regions == tuple(settings.regions)
    assert settings.carbon_cap == group.cap
    assert settings.carbon_allowance_procurement == group.allowance_procurement
    assert settings.carbon_allowance_start_bank == group.start_bank
    assert settings.carbon_allowance_bank_enabled == group.bank_enabled
    assert settings.carbon_allowance_allow_borrowing == group.allow_borrowing
