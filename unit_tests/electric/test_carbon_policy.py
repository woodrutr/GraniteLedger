import pytest
import pyomo.environ as pyo

from src.models.electricity.scripts.runner import record_allowance_emission_prices


def build_allowance_model(
    years,
    cap_groups,
    allowances,
    start_bank,
    region_to_group,
    banking_enabled=True,
    allow_borrowing=False,
):
    model = pyo.ConcreteModel()
    model.year = pyo.Set(initialize=years, ordered=True)
    model.cap_group = pyo.Set(initialize=cap_groups, ordered=True)
    model.group_year_index = pyo.Set(
        initialize=[(group, year) for group in cap_groups for year in years], dimen=2
    )
    model.region = pyo.Set(initialize=list(region_to_group.keys()), ordered=True)
    model.region_to_group = {region: region_to_group[region] for region in model.region}
    model.cap_group_regions = {
        group: {region for region, assigned in model.region_to_group.items() if assigned == group}
        for group in cap_groups
    }
    model.CarbonAllowanceProcurement = pyo.Param(
        model.group_year_index,
        initialize=lambda m, g, y: allowances.get((g, y), 0.0),
    )
    model.CarbonStartBank = pyo.Param(
        model.cap_group, initialize=lambda m, g: start_bank.get(g, 0.0)
    )
    model.banking_enabled = banking_enabled
    model.prev_year_lookup = {
        year: (years[idx - 1] if idx > 0 else None) for idx, year in enumerate(years)
    }
    model.first_year = years[0] if years else None

    model.allowance_purchase = pyo.Var(
        model.group_year_index, domain=pyo.NonNegativeReals
    )
    bank_domain = pyo.Reals if allow_borrowing else pyo.NonNegativeReals
    model.allowance_bank = pyo.Var(model.group_year_index, domain=bank_domain)
    model.year_emissions = pyo.Var(model.group_year_index, domain=pyo.NonNegativeReals)
    model.region_emissions = pyo.Var(
        model.region, model.year, domain=pyo.NonNegativeReals
    )

    def incoming_bank(m, group, year):
        prev_year = m.prev_year_lookup[year]
        carryover = (
            m.allowance_bank[(group, prev_year)]
            if (m.banking_enabled and prev_year is not None)
            else 0
        )
        start = m.CarbonStartBank[group] if year == m.first_year else 0
        return carryover + start

    model.allowance_purchase_limit = pyo.Constraint(
        model.group_year_index,
        rule=lambda m, g, y: m.allowance_purchase[(g, y)]
        <= m.CarbonAllowanceProcurement[(g, y)],
    )
    model.group_emissions_balance = pyo.Constraint(
        model.group_year_index,
        rule=lambda m, g, y: m.year_emissions[(g, y)]
        == sum(m.region_emissions[region, y] for region in m.cap_group_regions.get(g, set())),
    )
    model.allowance_bank_balance = pyo.Constraint(
        model.group_year_index,
        rule=lambda m, g, y: m.allowance_bank[(g, y)]
        == incoming_bank(m, g, y)
        + m.allowance_purchase[(g, y)]
        - m.year_emissions[(g, y)],
    )
    model.cap_group_allowance_emissions_limit = pyo.Constraint(
        model.group_year_index,
        rule=lambda m, g, y: m.year_emissions[(g, y)]
        <= m.allowance_purchase[(g, y)] + incoming_bank(m, g, y),
    )
    return model


def test_allowance_bank_balance():
    years = [2025, 2030]
    cap_groups = ['rggi', 'non_rggi']
    allowances = {
        ('rggi', 2025): 10.0,
        ('rggi', 2030): 12.0,
        ('non_rggi', 2025): 8.0,
        ('non_rggi', 2030): 9.0,
    }
    start_bank = {'rggi': 2.0, 'non_rggi': 1.0}
    region_to_group = {'RegionRGGI': 'rggi', 'RegionOther': 'non_rggi'}
    model = build_allowance_model(
        years,
        cap_groups,
        allowances,
        start_bank,
        region_to_group,
    )

    model.allowance_purchase['rggi', 2025].set_value(4.0)
    model.allowance_purchase['rggi', 2030].set_value(6.0)
    model.allowance_purchase['non_rggi', 2025].set_value(5.0)
    model.allowance_purchase['non_rggi', 2030].set_value(5.0)

    model.region_emissions['RegionRGGI', 2025].set_value(3.0)
    model.region_emissions['RegionRGGI', 2030].set_value(7.0)
    model.region_emissions['RegionOther', 2025].set_value(4.0)
    model.region_emissions['RegionOther', 2030].set_value(3.0)

    model.year_emissions['rggi', 2025].set_value(3.0)
    model.year_emissions['rggi', 2030].set_value(7.0)
    model.year_emissions['non_rggi', 2025].set_value(4.0)
    model.year_emissions['non_rggi', 2030].set_value(3.0)

    model.allowance_bank['rggi', 2025].set_value(3.0)
    model.allowance_bank['rggi', 2030].set_value(2.0)
    model.allowance_bank['non_rggi', 2025].set_value(2.0)
    model.allowance_bank['non_rggi', 2030].set_value(4.0)

    for key in model.group_year_index:
        balance = model.allowance_bank_balance[key]
        assert pytest.approx(0.0) == pyo.value(balance.body)
        emissions = model.group_emissions_balance[key]
        assert pytest.approx(0.0) == pyo.value(emissions.body)

    for key in model.group_year_index:
        limit = model.cap_group_allowance_emissions_limit[key]
        assert pyo.value(limit.body) <= 1e-9


def test_emission_limit_detects_shortfall():
    years = [2025, 2030]
    cap_groups = ['rggi', 'non_rggi']
    allowances = {
        ('rggi', 2025): 5.0,
        ('rggi', 2030): 5.0,
        ('non_rggi', 2025): 5.0,
        ('non_rggi', 2030): 5.0,
    }
    region_to_group = {'RegionRGGI': 'rggi', 'RegionOther': 'non_rggi'}
    model = build_allowance_model(
        years,
        cap_groups,
        allowances,
        start_bank={'rggi': 0.0, 'non_rggi': 0.0},
        region_to_group=region_to_group,
        banking_enabled=True,
        allow_borrowing=True,
    )

    model.allowance_purchase['rggi', 2025].set_value(5.0)
    model.allowance_purchase['rggi', 2030].set_value(5.0)
    model.allowance_purchase['non_rggi', 2025].set_value(5.0)
    model.allowance_purchase['non_rggi', 2030].set_value(5.0)

    model.region_emissions['RegionRGGI', 2025].set_value(6.0)
    model.region_emissions['RegionRGGI', 2030].set_value(4.0)
    model.region_emissions['RegionOther', 2025].set_value(3.0)
    model.region_emissions['RegionOther', 2030].set_value(4.0)

    model.year_emissions['rggi', 2025].set_value(6.0)
    model.year_emissions['rggi', 2030].set_value(4.0)
    model.year_emissions['non_rggi', 2025].set_value(3.0)
    model.year_emissions['non_rggi', 2030].set_value(4.0)

    model.allowance_bank['rggi', 2025].set_value(-1.0)
    model.allowance_bank['rggi', 2030].set_value(0.0)
    model.allowance_bank['non_rggi', 2025].set_value(2.0)
    model.allowance_bank['non_rggi', 2030].set_value(3.0)

    shortfall = pyo.value(
        model.cap_group_allowance_emissions_limit['rggi', 2025].body
    )
    assert shortfall > 0

    assert pytest.approx(0.0) == pyo.value(
        model.allowance_bank_balance['rggi', 2025].body
    )
    assert pyo.value(model.cap_group_allowance_emissions_limit['rggi', 2030].body) <= 1e-9
    assert pyo.value(
        model.cap_group_allowance_emissions_limit['non_rggi', 2025].body
    ) <= 1e-9


def test_non_member_regions_do_not_use_rggi_allowances():
    years = [2025]
    cap_groups = ['rggi', 'non_rggi']
    allowances = {('rggi', 2025): 5.0, ('non_rggi', 2025): 6.0}
    region_to_group = {'RegionRGGI': 'rggi', 'RegionOther': 'non_rggi'}
    model = build_allowance_model(
        years,
        cap_groups,
        allowances,
        start_bank={'rggi': 0.0, 'non_rggi': 0.0},
        region_to_group=region_to_group,
    )

    model.allowance_purchase['rggi', 2025].set_value(5.0)
    model.allowance_purchase['non_rggi', 2025].set_value(6.0)

    model.region_emissions['RegionRGGI', 2025].set_value(4.0)
    model.region_emissions['RegionOther', 2025].set_value(6.0)

    model.year_emissions['rggi', 2025].set_value(4.0)
    model.year_emissions['non_rggi', 2025].set_value(6.0)

    model.allowance_bank['rggi', 2025].set_value(1.0)
    model.allowance_bank['non_rggi', 2025].set_value(0.0)

    assert pytest.approx(0.0) == pyo.value(
        model.group_emissions_balance['rggi', 2025].body
    )
    assert pytest.approx(0.0) == pyo.value(
        model.group_emissions_balance['non_rggi', 2025].body
    )

    rggi_limit = pyo.value(
        model.cap_group_allowance_emissions_limit['rggi', 2025].body
    )
    assert rggi_limit == pytest.approx(4.0 - 5.0)

    assert model.year_emissions['rggi', 2025].value == pytest.approx(4.0)
    assert model.year_emissions['non_rggi', 2025].value == pytest.approx(6.0)


def test_reported_carbon_price_matches_marginal_cost():
    baseline_emissions = 10.0
    baseline_allowance = 8.0
    abatement_cost = 25.0
    tighten = 0.5
    solver = pyo.SolverFactory('appsi_highs')
    if not solver.available(False):
        pytest.skip('appsi_highs solver is not available in this environment')

    def solve_toy_model(allowance: float) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.cap_group = pyo.Set(initialize=['system'])
        model.year = pyo.Set(initialize=[2025])
        model.cap_group_year_index = pyo.Set(initialize=[('system', 2025)])
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.abatement = pyo.Var(domain=pyo.NonNegativeReals)
        model.emissions = pyo.Var(domain=pyo.NonNegativeReals)
        model.allowance_purchase = pyo.Var(
            model.cap_group_year_index, domain=pyo.NonNegativeReals
        )

        model.emissions_balance = pyo.Constraint(
            expr=model.emissions == baseline_emissions - model.abatement
        )
        model.allowance_purchase_limit = pyo.Constraint(
            model.cap_group_year_index,
            rule=lambda m, g, y: m.allowance_purchase[(g, y)] <= allowance,
        )
        model.cap_group_allowance_emissions_limit = pyo.Constraint(
            model.cap_group_year_index,
            rule=lambda m, g, y: m.emissions <= m.allowance_purchase[(g, y)],
        )
        model.total_cost = pyo.Objective(expr=abatement_cost * model.abatement)

        solver.solve(model)
        record_allowance_emission_prices(model)
        return model

    baseline_model = solve_toy_model(baseline_allowance)
    base_cost = pyo.value(baseline_model.total_cost)
    base_price = baseline_model.carbon_prices.get(('system', 2025))
    assert base_price is not None and base_price > 0.0

    tightened_model = solve_toy_model(baseline_allowance - tighten)
    tightened_cost = pyo.value(tightened_model.total_cost)

    delta_cost = tightened_cost - base_cost
    expected_cost = base_price * tighten

    assert delta_cost == pytest.approx(expected_cost, rel=1e-8, abs=1e-8)
