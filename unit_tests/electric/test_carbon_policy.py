import pyomo.environ as pyo
import pytest


def build_allowance_model(
    years,
    cap_groups,
    allowances,
    start_bank,
    membership,
    banking_enabled=None,
    allow_borrowing=None,
    region_emissions=None,
):
    model = pyo.ConcreteModel()
    model.year = pyo.Set(initialize=years, ordered=True)
    model.cap_group = pyo.Set(initialize=cap_groups, ordered=True)
    model.cap_group_year = pyo.Set(
        initialize=list(allowances.keys()), dimen=2, ordered=True
    )
    regions = sorted({region for (_, region) in membership.keys()})
    model.region = pyo.Set(initialize=regions, ordered=True)

    model.prev_year_lookup = {
        year: (years[idx - 1] if idx > 0 else None) for idx, year in enumerate(years)
    }

    banking_enabled = banking_enabled or {group: True for group in cap_groups}
    allow_borrowing = allow_borrowing or {group: False for group in cap_groups}
    region_emissions = region_emissions or {
        (region, year): 0.0 for region in regions for year in years
    }

    model.bank_enabled = banking_enabled
    model.allow_borrowing_by_group = allow_borrowing

    model.CarbonAllowanceProcurement = pyo.Param(
        model.cap_group_year,
        initialize=allowances,
        default=0.0,
    )
    model.CarbonStartBank = pyo.Param(
        model.cap_group, initialize=start_bank, default=0.0
    )
    model.CarbonCapGroupMembership = pyo.Param(
        model.cap_group,
        model.region,
        initialize=membership,
        default=0,
    )
    model.region_emissions = pyo.Param(
        model.region,
        model.year,
        initialize=region_emissions,
        default=0.0,
    )

    model.allowance_purchase = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )
    bank_domain = (
        pyo.Reals if any(allow_borrowing.values()) else pyo.NonNegativeReals
    )
    model.allowance_bank = pyo.Var(model.cap_group_year, domain=bank_domain)
    model.year_emissions = pyo.Var(
        model.cap_group_year, domain=pyo.NonNegativeReals
    )

    def incoming_bank(m, cap_group, year):
        prev_year = m.prev_year_lookup[year]
        carryover = (
            m.allowance_bank[(cap_group, prev_year)]
            if (m.bank_enabled.get(cap_group, True) and prev_year is not None)
            else 0
        )
        start = m.CarbonStartBank[cap_group] if year == years[0] else 0
        return carryover + start

    model.allowance_purchase_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, cap_group, y: m.allowance_purchase[(cap_group, y)]
        <= m.CarbonAllowanceProcurement[(cap_group, y)],
    )

    model.allowance_bank_balance = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, cap_group, y: m.allowance_bank[(cap_group, y)]
        == incoming_bank(m, cap_group, y)
        + m.allowance_purchase[(cap_group, y)]
        - m.year_emissions[(cap_group, y)],
    )

    model.allowance_emissions_limit = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, cap_group, y: m.year_emissions[(cap_group, y)]
        <= m.allowance_purchase[(cap_group, y)] + incoming_bank(m, cap_group, y),
    )

    model.year_emissions_balance = pyo.Constraint(
        model.cap_group_year,
        rule=lambda m, cap_group, y: m.year_emissions[(cap_group, y)]
        == sum(
            m.CarbonCapGroupMembership[(cap_group, region)]
            * m.region_emissions[(region, y)]
            for region in m.region
        ),
    )

    non_borrow_pairs = [
        (cap_group, y)
        for (cap_group, y) in model.cap_group_year
        if not model.allow_borrowing_by_group.get(cap_group, False)
    ]
    if non_borrow_pairs and bank_domain is pyo.Reals:
        model.allowance_bank_nonneg = pyo.Constraint(
            non_borrow_pairs,
            rule=lambda m, cap_group, y: m.allowance_bank[(cap_group, y)] >= 0,
        )

    return model


def test_allowance_bank_balance():
    years = [2025, 2030]
    groups = ['system']
    allowances = {
        ('system', 2025): 10.0,
        ('system', 2030): 12.0,
    }
    start_bank = {'system': 2.0}
    membership = {('system', 'system_region'): 1}
    region_emissions = {
        ('system_region', 2025): 3.0,
        ('system_region', 2030): 7.0,
    }

    model = build_allowance_model(
        years,
        groups,
        allowances,
        start_bank,
        membership,
        region_emissions=region_emissions,
    )

    model.allowance_purchase[('system', 2025)].set_value(4.0)
    model.allowance_purchase[('system', 2030)].set_value(6.0)
    model.year_emissions[('system', 2025)].set_value(3.0)
    model.year_emissions[('system', 2030)].set_value(7.0)
    model.allowance_bank[('system', 2025)].set_value(3.0)
    model.allowance_bank[('system', 2030)].set_value(2.0)

    balance_2025 = model.allowance_bank_balance[('system', 2025)]
    balance_2030 = model.allowance_bank_balance[('system', 2030)]
    assert pytest.approx(0.0) == pyo.value(balance_2025.body)
    assert pytest.approx(0.0) == pyo.value(balance_2030.body)

    incoming_2025 = pyo.value(model.CarbonStartBank['system'])
    incoming_2030 = model.allowance_bank[('system', 2025)].value
    assert model.year_emissions[('system', 2025)].value <= (
        model.allowance_purchase[('system', 2025)].value + incoming_2025
    )
    assert model.year_emissions[('system', 2030)].value <= (
        model.allowance_purchase[('system', 2030)].value + incoming_2030
    )


def test_emission_limit_detects_shortfall():
    years = [2025, 2030]
    groups = ['system']
    allowances = {
        ('system', 2025): 5.0,
        ('system', 2030): 5.0,
    }
    start_bank = {'system': 0.0}
    membership = {('system', 'system_region'): 1}
    region_emissions = {
        ('system_region', 2025): 6.0,
        ('system_region', 2030): 4.0,
    }

    model = build_allowance_model(
        years,
        groups,
        allowances,
        start_bank,
        membership,
        banking_enabled={'system': True},
        allow_borrowing={'system': True},
        region_emissions=region_emissions,
    )

    model.allowance_purchase[('system', 2025)].set_value(5.0)
    model.allowance_purchase[('system', 2030)].set_value(5.0)
    model.year_emissions[('system', 2025)].set_value(6.0)
    model.year_emissions[('system', 2030)].set_value(4.0)
    model.allowance_bank[('system', 2025)].set_value(-1.0)
    model.allowance_bank[('system', 2030)].set_value(0.0)

    incoming_2025 = pyo.value(model.CarbonStartBank['system'])
    incoming_2030 = model.allowance_bank[('system', 2025)].value
    assert model.year_emissions[('system', 2025)].value > (
        model.allowance_purchase[('system', 2025)].value + incoming_2025
    )
    assert pytest.approx(0.0) == pyo.value(
        model.allowance_bank_balance[('system', 2025)].body
    )
    assert model.year_emissions[('system', 2030)].value <= (
        model.allowance_purchase[('system', 2030)].value + incoming_2030
    )


def test_group_membership_limits_emissions_to_members():
    years = [2025]
    groups = ['rggi', 'rest']
    allowances = {('rggi', 2025): 2.0, ('rest', 2025): 3.0}
    start_bank = {'rggi': 0.0, 'rest': 0.0}
    membership = {
        ('rggi', 'rggi_region'): 1,
        ('rggi', 'rest_region'): 0,
        ('rest', 'rggi_region'): 0,
        ('rest', 'rest_region'): 1,
    }
    region_emissions = {
        ('rggi_region', 2025): 2.0,
        ('rest_region', 2025): 3.0,
    }

    model = build_allowance_model(
        years,
        groups,
        allowances,
        start_bank,
        membership,
        region_emissions=region_emissions,
    )

    model.allowance_purchase[('rggi', 2025)].set_value(2.0)
    model.allowance_purchase[('rest', 2025)].set_value(3.0)
    model.allowance_bank[('rggi', 2025)].set_value(0.0)
    model.allowance_bank[('rest', 2025)].set_value(0.0)

    # If non-members were counted, emissions would appear too high
    model.year_emissions[('rggi', 2025)].set_value(5.0)
    model.year_emissions[('rest', 2025)].set_value(3.0)
    imbalance = pyo.value(model.year_emissions_balance[('rggi', 2025)].body)
    assert imbalance != pytest.approx(0.0)

    # With correct grouping, constraint residual should vanish
    model.year_emissions[('rggi', 2025)].set_value(2.0)
    balanced = pyo.value(model.year_emissions_balance[('rggi', 2025)].body)
    assert balanced == pytest.approx(0.0)

    limit_residual = pyo.value(
        model.allowance_emissions_limit[('rggi', 2025)].body
    )
    assert limit_residual <= pytest.approx(0.0)
