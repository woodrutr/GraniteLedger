import pytest
import pyomo.environ as pyo


def build_allowance_model(
    years,
    allowances,
    start_bank,
    banking_enabled=True,
    allow_borrowing=False,
):
    model = pyo.ConcreteModel()
    model.year = pyo.Set(initialize=years, ordered=True)
    model.CarbonAllowanceProcurement = pyo.Param(model.year, initialize=allowances)
    model.CarbonStartBank = pyo.Param(initialize=start_bank)
    model.banking_enabled = banking_enabled
    model.prev_year_lookup = {
        year: (years[idx - 1] if idx > 0 else None)
        for idx, year in enumerate(years)
    }

    model.allowance_purchase = pyo.Var(model.year, domain=pyo.NonNegativeReals)
    bank_domain = pyo.Reals if allow_borrowing else pyo.NonNegativeReals
    model.allowance_bank = pyo.Var(model.year, domain=bank_domain)
    model.year_emissions = pyo.Var(model.year, domain=pyo.NonNegativeReals)

    def incoming_bank(m, year):
        prev_year = m.prev_year_lookup[year]
        carryover = (
            m.allowance_bank[prev_year]
            if (m.banking_enabled and prev_year is not None)
            else 0
        )
        start = m.CarbonStartBank if year == years[0] else 0
        return carryover + start

    model.allowance_purchase_limit = pyo.Constraint(
        model.year,
        rule=lambda m, y: m.allowance_purchase[y] <= m.CarbonAllowanceProcurement[y],
    )
    model.allowance_bank_balance = pyo.Constraint(
        model.year,
        rule=lambda m, y: m.allowance_bank[y]
        == incoming_bank(m, y) + m.allowance_purchase[y] - m.year_emissions[y],
    )
    model.allowance_emissions_limit = pyo.Constraint(
        model.year,
        rule=lambda m, y: m.year_emissions[y]
        <= m.allowance_purchase[y] + incoming_bank(m, y),
    )

    return model


def test_allowance_bank_balance():
    years = [2025, 2030]
    allowances = {2025: 10.0, 2030: 12.0}
    model = build_allowance_model(years, allowances, start_bank=2.0)

    model.allowance_purchase[2025].set_value(4.0)
    model.allowance_purchase[2030].set_value(6.0)
    model.year_emissions[2025].set_value(3.0)
    model.year_emissions[2030].set_value(7.0)
    model.allowance_bank[2025].set_value(3.0)
    model.allowance_bank[2030].set_value(2.0)

    balance_2025 = model.allowance_bank_balance[2025]
    balance_2030 = model.allowance_bank_balance[2030]
    assert pytest.approx(0.0) == pyo.value(balance_2025.body)
    assert pytest.approx(0.0) == pyo.value(balance_2030.body)

    incoming_2025 = model.CarbonStartBank.value
    incoming_2030 = model.allowance_bank[2025].value
    assert model.year_emissions[2025].value <= (
        model.allowance_purchase[2025].value + incoming_2025
    )
    assert model.year_emissions[2030].value <= (
        model.allowance_purchase[2030].value + incoming_2030
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

    model.allowance_purchase[2025].set_value(5.0)
    model.allowance_purchase[2030].set_value(5.0)
    model.year_emissions[2025].set_value(6.0)
    model.year_emissions[2030].set_value(4.0)
    model.allowance_bank[2025].set_value(-1.0)
    model.allowance_bank[2030].set_value(0.0)

    incoming_2025 = model.CarbonStartBank.value
    incoming_2030 = model.allowance_bank[2025].value
    assert model.year_emissions[2025].value > (
        model.allowance_purchase[2025].value + incoming_2025
    )
    assert pytest.approx(0.0) == pyo.value(model.allowance_bank_balance[2025].body)
    assert model.year_emissions[2030].value <= (
        model.allowance_purchase[2030].value + incoming_2030
    )
