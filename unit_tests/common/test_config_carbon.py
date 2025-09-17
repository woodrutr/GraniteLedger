"""Tests for configuration carbon policy handling."""

from pathlib import Path

import pytest

pytest.importorskip('pandas')

from definitions import PROJECT_ROOT
from src.common.config_setup import Config_settings, SHORT_TON_TO_METRIC_TON


SOURCE_CONFIG = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')


def _config_without_top_level_carbon_cap() -> str:
    """Return the base config text without any top-level carbon_cap entries."""

    filtered_lines = [
        line
        for line in SOURCE_CONFIG.read_text().splitlines()
        if not line.strip().startswith('carbon_cap =')
    ]
    return '\n'.join(filtered_lines) + '\n'


def test_carbon_cap_converts_short_tons_to_metric(tmp_path):
    """Carbon cap values in short tons should be converted to metric tons."""

    short_ton_cap = 1234.5
    temp_config_path = tmp_path / 'run_config.toml'
    config_contents = _config_without_top_level_carbon_cap()
    config_contents += f"carbon_cap = {short_ton_cap}\n"
    temp_config_path.write_text(config_contents)

    settings = Config_settings(temp_config_path, test=True)

    assert settings.carbon_cap == pytest.approx(
        short_ton_cap * SHORT_TON_TO_METRIC_TON
    )


def test_carbon_cap_uses_carbon_policy_section_when_present(tmp_path):
    """Values defined in the carbon policy section should not be overwritten."""

    temp_config_path = tmp_path / 'run_config.toml'
    config_contents = _config_without_top_level_carbon_cap()
    short_ton_cap = 4321.0
    config_contents += '\n[carbon_policy]\n'
    config_contents += f'carbon_cap = {short_ton_cap}\n'
    temp_config_path.write_text(config_contents)

    settings = Config_settings(temp_config_path, test=True)

    assert settings.carbon_cap == pytest.approx(
        short_ton_cap * SHORT_TON_TO_METRIC_TON
    )


def test_carbon_policy_section_without_cap_falls_back_to_root(tmp_path):
    """Root-level caps should be used when the carbon policy table omits them."""

    temp_config_path = tmp_path / 'run_config.toml'
    config_contents = _config_without_top_level_carbon_cap()
    config_contents += '\n[carbon_policy]\n'
    config_contents += 'carbon_allowance_start_bank = 5\n'
    root_level_cap = 2789.0
    config_contents += f"\ncarbon_cap = {root_level_cap}\n"
    temp_config_path.write_text(config_contents)

    settings = Config_settings(temp_config_path, test=True)

    assert settings.carbon_cap == pytest.approx(
        root_level_cap * SHORT_TON_TO_METRIC_TON
    )


@pytest.mark.parametrize('sentinel_value', [' none ', 'NULL', ''])
def test_carbon_cap_sentinels_map_to_none(tmp_path, sentinel_value):
    """Sentinel values should consistently map to ``None`` after normalization."""

    temp_config_path = tmp_path / 'run_config.toml'
    config_contents = _config_without_top_level_carbon_cap()
    config_contents += f'carbon_cap = "{sentinel_value}"\n'
    temp_config_path.write_text(config_contents)

    settings = Config_settings(temp_config_path, test=True)

    assert settings.carbon_cap is None
