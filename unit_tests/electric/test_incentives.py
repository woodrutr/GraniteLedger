"""Tests for the electricity technology incentives placeholder module."""

from __future__ import annotations

import pandas as pd

from src.models.electricity.scripts.incentives import TechnologyIncentives


def _sorted(config_section):
    return sorted(config_section, key=lambda entry: (entry['technology'], entry['year']))


def test_incentives_config_round_trip():
    config = {
        'production': [
            {
                'technology': 'Solar',
                'year': 2025,
                'credit_per_mwh': 20.0,
                'limit_mwh': 5000.0,
            }
        ],
        'investment': [
            {
                'technology': 'Wind Onshore',
                'year': 2030,
                'credit_per_mw': 75000.0,
            }
        ],
    }

    incentives = TechnologyIncentives.from_config(config)

    assert len(list(incentives)) == 2

    round_trip = incentives.to_config()
    assert set(round_trip) == {'production', 'investment'}
    assert _sorted(round_trip['production']) == _sorted(config['production'])
    assert _sorted(round_trip['investment']) == _sorted(config['investment'])


def test_incentives_to_frames_structure():
    config = {
        'production': [
            {'technology': 'Solar', 'year': 2025, 'credit_per_mwh': 10.0},
            {'technology': 'Wind Onshore', 'year': 2026, 'credit_per_mwh': 15.5},
        ],
        'investment': [
            {'technology': 'Wind Onshore', 'year': 2026, 'credit_per_mw': 60000.0, 'limit_mw': 400.0}
        ],
    }

    incentives = TechnologyIncentives.from_config(config)
    frames = incentives.to_frames()

    assert set(frames) == {'TechnologyIncentiveCredit', 'TechnologyIncentiveLimit'}

    credit_df = frames['TechnologyIncentiveCredit']
    limit_df = frames['TechnologyIncentiveLimit']

    pd.testing.assert_index_equal(credit_df.columns, pd.Index(['tech', 'year', 'incentive_type', 'credit_per_unit']))
    pd.testing.assert_index_equal(limit_df.columns, pd.Index(['tech', 'year', 'incentive_type', 'limit_value']))

    assert set(credit_df['incentive_type']) == {'production', 'investment'}
    assert credit_df['credit_per_unit'].dtype.kind == 'f'


def test_incentives_from_table_rows_filters_invalid():
    rows = [
        {
            'type': 'Production',
            'technology': 'Solar',
            'year': 2025,
            'credit_value': 12.5,
            'limit_value': '',
            'limit_units': 'MWh',
        },
        {
            'type': 'Investment',
            'technology': 'Unknown',
            'year': 2025,
            'credit_value': 50000,
            'limit_value': 100.0,
            'limit_units': 'MW',
        },
        {
            'type': 'Production',
            'technology': 'Solar',
            'year': '',
            'credit_value': 5.0,
            'limit_value': 10.0,
            'limit_units': 'MWh',
        },
    ]

    incentives = TechnologyIncentives.from_table_rows(rows)

    records = list(incentives)
    assert len(records) == 1
    record = records[0]
    assert record.credit_type == 'production'
    assert record.year == 2025
    assert record.limit_value is None
