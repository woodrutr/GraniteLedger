"""Carbon policy helpers for applying allowance market rules."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


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
    trigger_keys: Sequence[str],
    quantity_keys: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[float, float]:
    """Validate and return CCR trigger settings when ``enabled``."""

    if not enabled:
        return 0.0, 0.0

    trigger_value: Any = None
    for key in trigger_keys:
        trigger_value = config.get(key, trigger_value)
        if trigger_value not in (None, ""):
            break
    if trigger_value in (None, ""):
        preferred = trigger_keys[0]
        raise CarbonPolicyError(
            "Configuration enables CCR but does not supply "
            f"'{preferred}' (or an accepted alias)."
        )

    quantity_value: Any = None
    for key in quantity_keys:
        quantity_value = config.get(key, quantity_value)
        if quantity_value not in (None, ""):
            break
    if quantity_value in (None, ""):
        preferred = quantity_keys[0]
        raise CarbonPolicyError(
            "Configuration enables CCR but does not supply "
            f"'{preferred}' (or an accepted alias)."
        )

    try:
        trigger_float = float(trigger_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise CarbonPolicyError(
            f"CCR trigger price must be numeric (received {trigger_value!r})."
        ) from exc

    try:
        quantity_float = float(quantity_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise CarbonPolicyError(
            f"CCR quantity must be numeric (received {quantity_value!r})."
        ) from exc

    if quantity_float < 0.0:
        raise CarbonPolicyError("CCR quantity must be non-negative")

    return trigger_float, quantity_float


def _extract_year_hint(value: Any) -> int | None:
    """Return a four-digit year parsed from ``value`` when possible."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() and len(stripped) >= 4:
            try:
                return int(stripped[:4])
            except ValueError:  # pragma: no cover - defensive guard
                return None
        digits = "".join(ch for ch in stripped if ch.isdigit())
        if len(digits) >= 4:
            try:
                return int(digits[:4])
            except ValueError:  # pragma: no cover - defensive guard
                return None
    return None


def _resolve_reserve_price(
    state: Mapping[str, Any], config: Mapping[str, Any]
) -> float | None:
    """Return the applicable reserve price for ``state`` if configured."""

    reserve_entry = config.get("reserve_price")
    if reserve_entry in (None, ""):
        return None

    if isinstance(reserve_entry, Mapping):
        candidate_sources: tuple[Mapping[str, Any], ...] = (state, config)
        candidate_keys = (
            "year",
            "compliance_year",
            "period",
            "period_label",
            "control_period",
        )
        candidates: list[Any] = []
        for mapping in candidate_sources:
            for key in candidate_keys:
                value = mapping.get(key)
                if value is not None:
                    candidates.append(value)

        for candidate in candidates:
            lookup_keys: list[Any] = []
            if isinstance(candidate, str):
                stripped = candidate.strip()
                if stripped:
                    lookup_keys.append(stripped)
            if candidate is not None:
                lookup_keys.append(candidate)
            year_hint = _extract_year_hint(candidate)
            if year_hint is not None:
                lookup_keys.append(year_hint)
                lookup_keys.append(str(year_hint))

            for key in lookup_keys:
                if key in reserve_entry:
                    return _coerce_float(reserve_entry[key])
        return None

    return _coerce_float(reserve_entry)


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
    banking_enabled = bool(
        config.get("allowance_banking_enabled", config.get("banking_enabled", False))
    )

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

    reserve_price_value = _resolve_reserve_price(state, config)
    if reserve_price_value is not None:
        result_price = max(result_price, reserve_price_value)

    ccr1_enabled = bool(config.get("ccr1_enabled", enable_ccr)) and enable_ccr
    ccr2_enabled = bool(config.get("ccr2_enabled", enable_ccr)) and enable_ccr

    ccr1_trigger, ccr1_qty = _validate_trigger(
        enabled=ccr1_enabled,
        trigger_keys=("ccr1_trigger_price", "ccr1_price"),
        quantity_keys=("ccr1_quantity", "ccr1_qty"),
        config=config,
    )
    ccr2_trigger, ccr2_qty = _validate_trigger(
        enabled=ccr2_enabled,
        trigger_keys=("ccr2_trigger_price", "ccr2_price"),
        quantity_keys=("ccr2_quantity", "ccr2_qty"),
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
