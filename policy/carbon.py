"""Carbon policy helpers for applying allowance market rules."""
from __future__ import annotations

from typing import Any, Mapping


class CarbonPolicyError(ValueError):
    """Error raised when the carbon policy configuration is invalid."""


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    """Return ``value`` coerced to ``float`` with ``default`` fallback."""

    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CarbonPolicyError(f"Unable to coerce value {value!r} to float") from exc


def _required_float(mapping: Mapping[str, Any], key: str) -> float:
    """Return a required float entry from ``mapping``."""

    if key not in mapping:
        raise CarbonPolicyError(f"Carbon policy configuration missing required key '{key}'")
    return _coerce_float(mapping[key])


def _validate_trigger(
    *,
    enabled: bool,
    trigger_key: str,
    quantity_key: str,
    config: Mapping[str, Any],
) -> tuple[float, float]:
    """Validate and return CCR trigger settings when ``enabled``."""

    trigger = config.get(trigger_key)
    quantity = config.get(quantity_key)
    if not enabled:
        return 0.0, 0.0
    if trigger in (None, ""):
        raise CarbonPolicyError(
            f"Configuration enables CCR but does not supply '{trigger_key}'."
        )
    if quantity in (None, ""):
        raise CarbonPolicyError(
            f"Configuration enables CCR but does not supply '{quantity_key}'."
        )
    return float(trigger), float(quantity)


def apply_carbon_policy(
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a new allowance state after applying carbon policy rules.

    Parameters
    ----------
    state:
        Mapping describing the market state with at least the keys ``emissions``
        and ``bank_balance``. Optional keys ``allowances`` and ``price`` provide
        the allowances minted before policy adjustments and the observed market
        price.
    config:
        Mapping defining the policy settings such as cap, floor, CCR triggers
        and banking behavior.

    Returns
    -------
    dict[str, Any]
        New immutable mapping containing the updated state with allowances,
        surrendered tons, bank balance, price and CCR issuance.

    Raises
    ------
    CarbonPolicyError
        If the configuration is inconsistent with the enabled options.
    """

    if not isinstance(state, Mapping):
        raise CarbonPolicyError("state must be a mapping")
    if not isinstance(config, Mapping):
        raise CarbonPolicyError("config must be a mapping")

    enabled = bool(config.get("enabled", True))
    enable_floor = bool(config.get("enable_floor", False))
    enable_ccr = bool(config.get("enable_ccr", False))
    banking_enabled = bool(config.get("allowance_banking_enabled", False))

    emissions = _coerce_float(state.get("emissions", 0.0))
    bank_previous = _coerce_float(state.get("bank_balance", 0.0))
    base_price = _coerce_float(state.get("price", 0.0))

    if bank_previous < 0.0:
        raise CarbonPolicyError("Bank balance cannot be negative")
    if emissions < 0.0:
        raise CarbonPolicyError("Emissions cannot be negative")

    result_price = base_price
    ccr1_issued = 0.0
    ccr2_issued = 0.0

    if not enabled:
        allowances_available = _coerce_float(state.get("allowances", emissions))
        total_allowances = bank_previous + allowances_available
        surrendered = min(emissions, total_allowances)
        remaining_bank = max(0.0, total_allowances - emissions) if banking_enabled else 0.0
        shortage = emissions > total_allowances
        return {
            "emissions": emissions,
            "price": result_price,
            "allowances_minted": allowances_available,
            "total_allowances": total_allowances,
            "surrendered": surrendered,
            "bank_balance": remaining_bank,
            "shortage": shortage,
            "ccr1_issued": ccr1_issued,
            "ccr2_issued": ccr2_issued,
        }

    cap = _required_float(config, "cap")
    if cap < 0.0:
        raise CarbonPolicyError("Cap must be non-negative")

    price_floor = float(config.get("price_floor", config.get("floor", 0.0)))
    if enable_floor:
        result_price = max(result_price, price_floor)

    ccr1_enabled = bool(config.get("ccr1_enabled", enable_ccr)) and enable_ccr
    ccr2_enabled = bool(config.get("ccr2_enabled", enable_ccr)) and enable_ccr

    ccr1_trigger, ccr1_qty = _validate_trigger(
        enabled=ccr1_enabled,
        trigger_key="ccr1_trigger_price",
        quantity_key="ccr1_quantity",
        config=config,
    )
    ccr2_trigger, ccr2_qty = _validate_trigger(
        enabled=ccr2_enabled,
        trigger_key="ccr2_trigger_price",
        quantity_key="ccr2_quantity",
        config=config,
    )

    allowances_minted = min(cap, _coerce_float(state.get("allowances", cap)))
    if allowances_minted < 0.0:
        raise CarbonPolicyError("Allowances cannot be negative")

    if ccr1_enabled and result_price >= ccr1_trigger:
        ccr1_issued = ccr1_qty
        allowances_minted += ccr1_qty
    if ccr2_enabled and result_price >= ccr2_trigger:
        ccr2_issued = ccr2_qty
        allowances_minted += ccr2_qty

    total_allowances = allowances_minted + bank_previous
    surrendered = min(emissions, total_allowances)
    shortage = emissions > total_allowances
    remaining_bank = 0.0
    if banking_enabled:
        remaining_bank = max(0.0, total_allowances - emissions)

    return {
        "emissions": emissions,
        "price": result_price,
        "allowances_minted": allowances_minted,
        "total_allowances": total_allowances,
        "surrendered": surrendered,
        "bank_balance": remaining_bank,
        "shortage": shortage,
        "ccr1_issued": ccr1_issued,
        "ccr2_issued": ccr2_issued,
    }


__all__ = ["apply_carbon_policy", "CarbonPolicyError"]
