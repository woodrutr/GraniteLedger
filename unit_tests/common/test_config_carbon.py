"""Tests for configuration carbon policy handling."""

from pathlib import Path

import pytest

pytest.importorskip('pandas')

from definitions import PROJECT_ROOT
from src.common.config_setup import Config_settings, SHORT_TON_TO_METRIC_TON


def test_carbon_cap_converts_short_tons_to_metric(tmp_path):
    """Carbon cap values in short tons should be converted to metric tons."""

    source_config = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    config_contents = source_config.read_text()

    short_ton_cap = 1234.5
    temp_config_path = tmp_path / 'run_config.toml'
    updated_config = config_contents.replace(
        'carbon_cap = "none"', f'carbon_cap = {short_ton_cap}'
    )
    temp_config_path.write_text(updated_config)

    settings = Config_settings(temp_config_path, test=True)

    assert settings.carbon_cap == pytest.approx(
        short_ton_cap * SHORT_TON_TO_METRIC_TON
    )
