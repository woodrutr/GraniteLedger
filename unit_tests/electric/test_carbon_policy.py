import pytest

pyo = pytest.importorskip(
    'pyomo.environ', reason='Pyomo is required for carbon policy electricity tests'
)

from src.models.electricity.scripts.runner import record_allowance_emission_prices


def build_grouped_allowance_model(
    years,
    cap_groups,
    allowances,
    emissions,
    start_bank,
    prices=None,
    banking_enabled=True,
    allow_borrowing=False,
):
    model = pyo.ConcreteModel()
    model.year = pyo.Set(initialize=years, ordered=True)
    model.cap_group = pyo.Set(initialize=cap_groups, ordered=True)
    model.allowance_index = model.cap_group * model.year

    def allowance_init(m, g, y):
        return allowances.get((g, y), 0.0)

    model.CarbonAllowanceProcurement = pyo.Param(
        model.cap_group,
        model.year,
        initialize=allowance_init,
        default=0.0,
    )

    if isinstance(start_bank, dict):
        def start_init(m, g):
            return start_bank.get(g, 0.0)
    else:
        def start_init(m, g):
            return start_bank

    model.CarbonStartBank = pyo.Param(model.cap_group, initialize=start_init)
    model.banking_enabled = banking_enabled
    model.prev_year_lookup = {
        year: (years[idx - 1] if idx > 0 else None)
        for idx, year in enumerate(years)
    }

    model.allowance_purchase = pyo.Var(
        model.allowance_index, domain=pyo.NonNegativeReals
    )
    bank_domain = pyo.Reals if allow_borrowing else pyo.NonNegativeReals
    model.allowance_bank = pyo.Var(model.allowance_index, domain=bank_domain)
    model.allowance_emissions = pyo.Var(
        model.allowance_index, domain=pyo.NonNegativeReals
    )
    model.year_emissions = pyo.Var(model.year, domain=pyo.NonNegativeReals)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    def incoming_bank(m, g, year):
        prev_year = m.prev_year_lookup[year]
        carryover = (
            m.allowance_bank[g, prev_year]
            if (m.banking_enabled and prev_year is not None)
            else 0
        )
        start = pyo.value(m.CarbonStartBank[g]) if year == years[0] else 0
        return carryover + start

    model.incoming_bank = incoming_bank

    model.allowance_purchase_limit = pyo.Constraint(
        model.allowance_index,
        rule=lambda m, g, y: m.allowance_purchase[g, y]
        <= m.CarbonAllowanceProcurement[g, y],
    )
    model.allowance_bank_balance = pyo.Constraint(
        model.allowance_index,
        rule=lambda m, g, y: m.allowance_bank[g, y]
        == incoming_bank(m, g, y)
        + m.allowance_purchase[g, y]
        - m.allowance_emissions[g, y],
    )
    model.allowance_emissions_limit = pyo.Constraint(
        model.allowance_index,
        rule=lambda m, g, y: m.allowance_emissions[g, y]
        <= m.allowance_purchase[g, y] + incoming_bank(m, g, y),
    )
    model.emissions_assignment = pyo.Constraint(
        model.allowance_index,
        rule=lambda m, g, y: m.allowance_emissions[g, y]
        == emissions.get((g, y), 0.0),
    )
    model.year_emissions_balance = pyo.Constraint(
        model.year,
        rule=lambda m, y: m.year_emissions[y]
        == sum(m.allowance_emissions[g, y] for g in m.cap_group),
    )

    if prices is not None:
        def price_init(m, g, y):
            return prices.get((g, y), 0.0)

        model.CarbonPrice = pyo.Param(
            model.cap_group,
            model.year,
            initialize=price_init,
            default=0.0,
        )
        model.total_cost = pyo.Objective(
            expr=sum(
                model.CarbonPrice[g, y] * model.allowance_purchase[g, y]
                for g in model.cap_group
                for y in model.year
            )
        )

    return model


def test_group_allowance_bank_balance():
    years = [2025, 2030]
    cap_groups = ['rggi', 'other']
    allowances = {
        ('rggi', 2025): 3.0,
        ('rggi', 2030): 4.0,
        ('other', 2025): 0.0,
        ('other', 2030): 0.0,
    }
    emissions = {
        ('rggi', 2025): 3.0,
        ('rggi', 2030): 4.0,
        ('other', 2025): 1.0,
        ('other', 2030): 1.0,
    }
    start_bank = {'rggi': 0.0, 'other': 3.0}

    model = build_grouped_allowance_model(
        years, cap_groups, allowances, emissions, start_bank
    )

    expected_purchase = {
        ('rggi', 2025): 3.0,
        ('rggi', 2030): 4.0,
        ('other', 2025): 0.0,
        ('other', 2030): 0.0,
    }
    expected_bank = {
        ('rggi', 2025): 0.0,
        ('rggi', 2030): 0.0,
        ('other', 2025): 2.0,
        ('other', 2030): 1.0,
    }

    for (group, year), value in expected_purchase.items():
        model.allowance_purchase[group, year].set_value(value)
        model.allowance_emissions[group, year].set_value(
            emissions[(group, year)]
        )
        model.allowance_bank[group, year].set_value(expected_bank[group, year])
    for year in years:
        model.year_emissions[year].set_value(
            sum(emissions[(group, year)] for group in cap_groups)
        )

    for idx in model.allowance_bank_balance:
        assert pytest.approx(0.0) == pyo.value(
            model.allowance_bank_balance[idx].body
        )

    incoming_other_2030 = pyo.value(model.incoming_bank(model, 'other', 2030))
    incoming_rggi_2030 = pyo.value(model.incoming_bank(model, 'rggi', 2030))
    assert incoming_other_2030 == pytest.approx(2.0)
    assert incoming_rggi_2030 == pytest.approx(0.0)

    for group in cap_groups:
        for year in years:
            incoming = pyo.value(model.incoming_bank(model, group, year))
            assert (
                model.allowance_emissions[group, year].value
                <= model.allowance_purchase[group, year].value + incoming + 1e-9
            )


def test_non_member_group_does_not_use_rggi_allowances():
    years = [2025]
    cap_groups = ['rggi', 'other']
    allowances = {('rggi', 2025): 5.0, ('other', 2025): 10.0}
    emissions = {('rggi', 2025): 5.0, ('other', 2025): 0.0}
    start_bank = {'rggi': 0.0, 'other': 0.0}
    prices = {('rggi', 2025): 50.0, ('other', 2025): 0.0}

    model = build_grouped_allowance_model(
        years,
        cap_groups,
        allowances,
        emissions,
        start_bank,
        prices=prices,
        banking_enabled=False,
    )

    model.allowance_purchase['rggi', 2025].set_value(0.0)
    model.allowance_purchase['other', 2025].set_value(5.0)
    model.allowance_bank['rggi', 2025].set_value(0.0)
    model.allowance_bank['other', 2025].set_value(5.0)
    model.allowance_emissions['rggi', 2025].set_value(emissions[('rggi', 2025)])
    model.allowance_emissions['other', 2025].set_value(emissions[('other', 2025)])
    incoming_rggi = pyo.value(model.incoming_bank(model, 'rggi', 2025))
    assert emissions[('rggi', 2025)] > (
        model.allowance_purchase['rggi', 2025].value + incoming_rggi
    )

    model.allowance_purchase['rggi', 2025].set_value(emissions[('rggi', 2025)])
    model.allowance_purchase['other', 2025].set_value(0.0)
    model.allowance_bank['rggi', 2025].set_value(0.0)
    model.allowance_bank['other', 2025].set_value(0.0)
    incoming_rggi = pyo.value(model.incoming_bank(model, 'rggi', 2025))
    assert emissions[('rggi', 2025)] == pytest.approx(
        model.allowance_purchase['rggi', 2025].value + incoming_rggi
    )


def test_reported_carbon_price_matches_marginal_cost():
    baseline_emissions = 10.0
    baseline_allowance = 8.0
    abatement_cost = 25.0
    tighten = 0.5
    def cost(allowance: float) -> float:
        abatement = max(0.0, baseline_emissions - allowance)
        return abatement_cost * abatement

    baseline_cost = cost(baseline_allowance)
    tightened_cost = cost(baseline_allowance - tighten)
    expected_price = abatement_cost
    assert tightened_cost - baseline_cost == pytest.approx(
        expected_price * tighten
    )

    model = pyo.ConcreteModel()
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.emissions = pyo.Var()
    model.allowance_purchase = pyo.Var()
    model.allowance_emissions_limit = pyo.Constraint(
        expr=model.emissions <= model.allowance_purchase
    )
    model.dual[model.allowance_emissions_limit] = -expected_price
    record_allowance_emission_prices(model)
    assert model.carbon_prices.get(None) == pytest.approx(expected_price)
