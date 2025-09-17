import pytest
import pyomo.environ as pyo

from src.models.electricity.scripts.runner import record_allowance_emission_prices


def build_allowance_model(
    years,
    allowances,
    start_bank,
    banking_enabled=True,
    allow_borrowing=False,
    cap_groups=('system',),
    membership=None,
):
    model = pyo.ConcreteModel()
    cap_groups = tuple(cap_groups)
    model.year = pyo.Set(initialize=years, ordered=True)
    model.cap_group = pyo.Set(initialize=cap_groups, ordered=True)
    model.cap_group_year = pyo.Set(
        initialize=[(g, y) for g in cap_groups for y in years],
        dimen=2,
        ordered=True,
    )
    model.banking_enabled = banking_enabled
    model.prev_year_lookup = {
        year: (years[idx - 1] if idx > 0 else None)
        for idx, year in enumerate(years)
    }

    allowances_map = {}
    for key, value in allowances.items():
        if isinstance(key, tuple):
            group, year = key
        else:
            group, year = cap_groups[0], key
        allowances_map[(group, int(year))] = float(value)
    model.CarbonAllowanceProcurement = pyo.Param(
        model.cap_group_year,
        initialize=allowances_map,
        default=0.0,
        mutable=True,
    )

    if isinstance(start_bank, dict):
        start_bank_map = {
            (group, int(year)): float(value) for (group, year), value in start_bank.items()
        }
    else:
        first_year = years[0]
        start_bank_map = {
            (group, first_year): float(start_bank) for group in cap_groups
        }
    model.CarbonStartBank = pyo.Param(
        model.cap_group_year,
        initialize=start_bank_map,
        default=0.0,
        mutable=True,
    )

    if membership is None:
        membership = {(cap_groups[0], 'region'): 1.0}
    membership_map = {
        (group, region): float(value) for (group, region), value in membership.items()
    }
    model.cap_group_region = pyo.Set(
        initialize=list(membership_map.keys()), dimen=2, ordered=True
    )
    model.CarbonCapGroupMembership = pyo.Param(
        model.cap_group_region, initialize=membership_map, default=0.0
    )

    bank_domain = pyo.Reals if allow_borrowing else pyo.NonNegativeReals
    model.allowance_purchase = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    model.allowance_bank = pyo.Var(model.cap_group_year, domain=bank_domain)
    model.year_emissions = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )

    def incoming_bank(m, group, year):
        prev_year = m.prev_year_lookup[year]
        carryover = (
            m.allowance_bank[(group, prev_year)]
            if (m.banking_enabled and prev_year is not None)
            else 0
        )
        return carryover + m.CarbonStartBank[(group, year)]

    model.allowance_purchase_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_purchase[g, y]
        <= m.CarbonAllowanceProcurement[g, y],
    )
    model.allowance_bank_balance = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.allowance_bank[g, y]
        == incoming_bank(m, g, y)
        + m.allowance_purchase[g, y]
        - m.year_emissions[g, y],
    )
    model.allowance_emissions_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, g, y: m.year_emissions[g, y]
        <= m.allowance_purchase[g, y] + incoming_bank(m, g, y),
    )

    return model


def test_allowance_bank_balance():
    years = [2025, 2030]
    allowances = {2025: 10.0, 2030: 12.0}
    model = build_allowance_model(years, allowances, start_bank=2.0)

    key_2025 = ('system', 2025)
    key_2030 = ('system', 2030)
    model.allowance_purchase[key_2025].set_value(4.0)
    model.allowance_purchase[key_2030].set_value(6.0)
    model.year_emissions[key_2025].set_value(3.0)
    model.year_emissions[key_2030].set_value(7.0)
    model.allowance_bank[key_2025].set_value(3.0)
    model.allowance_bank[key_2030].set_value(2.0)

    balance_2025 = model.allowance_bank_balance[key_2025]
    balance_2030 = model.allowance_bank_balance[key_2030]
    assert pytest.approx(0.0) == pyo.value(balance_2025.body)
    assert pytest.approx(0.0) == pyo.value(balance_2030.body)

    incoming_2025 = pyo.value(model.CarbonStartBank[key_2025])
    incoming_2030 = model.allowance_bank[key_2025].value
    assert model.year_emissions[key_2025].value <= (
        model.allowance_purchase[key_2025].value + incoming_2025
    )
    assert model.year_emissions[key_2030].value <= (
        model.allowance_purchase[key_2030].value + incoming_2030
    )


def test_emission_limit_detects_shortfall():
    years = [2025, 2030]
    allowances = {2025: 5.0, 2030: 5.0}
    model = build_allowance_model(
        years,
        allowances,
        start_bank=0.0,
        banking_enabled=True,
        allow_borrowing=True,
    )

    key_2025 = ('system', 2025)
    key_2030 = ('system', 2030)
    model.allowance_purchase[key_2025].set_value(5.0)
    model.allowance_purchase[key_2030].set_value(5.0)
    model.year_emissions[key_2025].set_value(6.0)
    model.year_emissions[key_2030].set_value(4.0)
    model.allowance_bank[key_2025].set_value(-1.0)
    model.allowance_bank[key_2030].set_value(0.0)

    incoming_2025 = pyo.value(model.CarbonStartBank[key_2025])
    incoming_2030 = model.allowance_bank[key_2025].value
    assert model.year_emissions[key_2025].value > (
        model.allowance_purchase[key_2025].value + incoming_2025
    )
    assert pytest.approx(0.0) == pyo.value(
        model.allowance_bank_balance[key_2025].body
    )
    assert model.year_emissions[key_2030].value <= (
        model.allowance_purchase[key_2030].value + incoming_2030
    )


def test_reported_carbon_price_matches_marginal_cost():
    baseline_emissions = 10.0
    baseline_allowance = 8.0
    abatement_cost = 25.0
    tighten = 0.5
    solver = pyo.SolverFactory('appsi_highs')
    if not solver.available(False):
        pytest.skip('appsi_highs solver not available in test environment')

    def solve_toy_model(allowance: float) -> pyo.ConcreteModel:
        model = pyo.ConcreteModel()
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        model.abatement = pyo.Var(domain=pyo.NonNegativeReals)
        model.emissions = pyo.Var(domain=pyo.NonNegativeReals)
        model.allowance_purchase = pyo.Var(domain=pyo.NonNegativeReals)

        model.emissions_balance = pyo.Constraint(
            expr=model.emissions == baseline_emissions - model.abatement
        )
        model.allowance_purchase_limit = pyo.Constraint(
            expr=model.allowance_purchase <= allowance
        )
        model.allowance_emissions_limit = pyo.Constraint(
            expr=model.emissions <= model.allowance_purchase
        )
        model.total_cost = pyo.Objective(expr=abatement_cost * model.abatement)

        solver.solve(model)
        record_allowance_emission_prices(model)
        return model

    baseline_model = solve_toy_model(baseline_allowance)
    base_cost = pyo.value(baseline_model.total_cost)
    base_price = baseline_model.carbon_prices.get(None)
    assert base_price is not None and base_price > 0.0

    tightened_model = solve_toy_model(baseline_allowance - tighten)
    tightened_cost = pyo.value(tightened_model.total_cost)

    delta_cost = tightened_cost - base_cost
    expected_cost = base_price * tighten

    assert delta_cost == pytest.approx(expected_cost, rel=1e-8, abs=1e-8)
