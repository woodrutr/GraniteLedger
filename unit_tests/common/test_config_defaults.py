from __future__ import annotations

import textwrap

import pytest

pytest.importorskip('pandas')

from src.common.config_setup import Config_settings
from src.models.electricity.scripts import preprocessor as elec_preprocessor


def test_missing_electric_switches_use_defaults(tmp_path):
    """Missing electricity switches should fall back to safe defaults."""

    config_text = textwrap.dedent(
        """
        default_mode = "elec"
        electricity = true
        hydrogen = false
        residential = false
        tol = 0.1
        force_10 = false
        max_iter = 1
        years = [2025]
        regions = [7]
        scale_load = "annual"
        h2_data_folder = "input/hydrogen/all_regions"
        """
    ).strip()

    config_path = tmp_path / "trimmed_config.toml"
    config_path.write_text(config_text)

    settings = Config_settings(config_path, test=True)

    assert settings.sw_temporal == "default"
    assert settings.sw_agg_years == 0
    assert settings.sw_trade == 0
    assert settings.sw_expansion == 0
    assert settings.sw_rm == 0
    assert settings.sw_ramp == 0
    assert settings.sw_reserves == 0
    assert settings.sw_learning == 0

    sets = elec_preprocessor.Sets(settings)

    assert sets.sw_trade == 0
    assert sets.sw_expansion == 0
    assert sets.sw_rm == 0
    assert sets.sw_ramp == 0
    assert sets.sw_reserves == 0
    assert sets.sw_learning == 0
    assert sets.sw_agg_years == 0
