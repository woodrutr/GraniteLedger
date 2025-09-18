from logging import getLogger
from pathlib import Path

import importlib
import pytest
from definitions import PROJECT_ROOT

pd = pytest.importorskip("pandas")

config_setup = importlib.import_module("src.common.config_setup")
utilities = importlib.import_module("src.models.electricity.scripts.utilities")
runner = importlib.import_module("src.models.electricity.scripts.runner")
prep = importlib.import_module("src.models.electricity.scripts.preprocessor")

logger = getLogger(__name__)


def test_years_set():
    """test to ensure the years set is injested properly"""
    years = [2030, 2031, 2042]
    regions = [7, 8]

    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = regions
    settings.years = years

    elec_model = runner.run_elec_model(settings, solve=False)

    # quick helper for annualizations....
    # quick test....  the aggregate weight of all the rep hours must = 8760
    assert (
        sum(utilities.annual_count(t, elec_model) for t in elec_model.hour) == 8760
    ), 'Annualized hours do not add up!'

    # the xor of the sets should be empty...
    assert len(elec_model.year ^ set(years)) == 0, 'some diff in expected sets'


def test_hours_set():
    """test to ensure the hours set is injested properly"""

    def get_tot_load(sw_temporal):
        """sum total load using hours and dayweights"""
        years = [2025]
        regions = [7]
        config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
        settings = config_setup.Config_settings(config_path, test=True)
        settings.regions = regions
        settings.years = years
        settings.sw_temporal = sw_temporal
        all_frames, setin = prep.preprocessor(prep.Sets(settings))

        tot_load1 = pd.merge(
            all_frames['Load'].reset_index(), all_frames['MapHourDay'].reset_index(), on='hour'
        )
        tot_load1 = pd.merge(tot_load1, all_frames['WeightDay'], on='day')
        tot_load1.loc[:, 'tot_load'] = tot_load1['Load'] * tot_load1['WeightDay']
        sum_load = round(sum(tot_load1.tot_load), 0)
        return sum_load

    # total load for 4 days, 1 hour per day
    tot_load_d4h1 = get_tot_load('d4h1')
    # total load for 8 days, 12 hours per day
    tot_load_d8h12 = get_tot_load('d8h12')

    # check that sum of load matches regardless of hours per day
    assert tot_load_d4h1 == tot_load_d8h12, 'some diff in hours sets'


def test_default_allowance_override_applies_to_remaining_groups():
    """Ensure default overrides update every group without an explicit override."""

    allowances = pd.DataFrame(
        {
            'cap_group': ['rggi', 'rggi', 'non_rggi', 'non_rggi'],
            'year': [2025, 2030, 2025, 2030],
            'CarbonAllowanceProcurement': [5.0, 6.0, 5.0, 6.0],
        }
    )
    overrides = {
        'rggi': {2025: 7.0},
        '__default__': {2025: 3.0, 2030: 4.0},
    }

    updated = prep.apply_allowance_overrides(
        allowances.copy(), overrides, ['rggi', 'non_rggi']
    )

    def allowance_value(group: str, year: int) -> float:
        series = updated[
            (updated['cap_group'] == group) & (updated['year'] == year)
        ]['CarbonAllowanceProcurement']
        assert not series.empty
        return float(series.iloc[0])

    assert allowance_value('rggi', 2025) == pytest.approx(7.0)
    assert allowance_value('non_rggi', 2025) == pytest.approx(3.0)
    assert allowance_value('non_rggi', 2030) == pytest.approx(4.0)


@pytest.mark.usefixtures('minimal_carbon_policy_inputs')
def test_carbon_cap_group_tables():
    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = [7, 8]
    settings.years = [2025, 2030]

    all_frames, setin = prep.preprocessor(prep.Sets(settings))

    membership = all_frames['CarbonCapGroupMembership'].reset_index()
    assert set(membership['region']) == set(settings.regions)
    assert set(membership['cap_group']) == {'national'}

    allowance_groups = all_frames['CarbonAllowanceProcurementByCapGroup'].reset_index()
    assert set(allowance_groups['cap_group']) == {'national'}
    weights = setin.WeightYear.set_index('year')['WeightYear']
    for _, row in allowance_groups.iterrows():
        year = int(row['year'])
        expected = settings.carbon_allowance_procurement.get(year, 0.0) * float(
            weights.get(year, 1)
        )
        assert row['CarbonAllowanceProcurement'] == pytest.approx(expected)

    price_groups = all_frames['CarbonAllowancePriceByCapGroup'].reset_index()
    assert set(price_groups['cap_group']) == {'national'}
    base_prices = (
        all_frames['CarbonAllowancePrice'].reset_index()
        if 'CarbonAllowancePrice' in all_frames
        else pd.DataFrame(columns=['year', 'CarbonPrice'])
    )
    base_price_lookup = {
        int(row['year']): float(row['CarbonPrice']) for _, row in base_prices.iterrows()
    }
    for _, row in price_groups.iterrows():
        year = int(row['year'])
        assert row['CarbonPrice'] == pytest.approx(base_price_lookup.get(year, 0.0))

    assert set(setin.cap_groups) == {'national'}
    assert setin.cap_group_membership.index.names == ['cap_group', 'region']
    assert setin.carbon_allowance_by_cap_group.index.names == ['cap_group', 'year']
    assert setin.carbon_price_by_cap_group.index.names == ['cap_group', 'year']


@pytest.mark.usefixtures('minimal_carbon_policy_inputs')
def test_carbon_cap_group_region_override():
    config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    settings = config_setup.Config_settings(config_path, test=True)
    settings.regions = [7, 8]
    settings.years = [2025, 2030]
    settings.carbon_cap_groups = settings._normalize_carbon_cap_groups(
        {'national': {'regions': [7]}}
    )

    all_frames, setin = prep.preprocessor(prep.Sets(settings))

    membership = all_frames['CarbonCapGroupMembership'].reset_index()
    assert set(membership['cap_group']) == {'national'}
    assert set(membership['region']) == {7}

    indexed_membership = setin.cap_group_membership.reset_index()
    assert set(indexed_membership['region']) == {7}
    assert set(setin.cap_groups) == {'national'}

