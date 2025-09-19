"""Allowance supply primitives with CCR and floor toggles."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AllowanceSupply:
    """Representation of the allowance supply curve for a single year."""

    cap: float
    floor: float
    ccr1_trigger: float
    ccr1_qty: float
    ccr2_trigger: float
    ccr2_qty: float
    enable_floor: bool = True
    enable_ccr: bool = True

    def available_allowances(self, price: float) -> float:
        """Return the allowances available at ``price`` including CCR supply."""

        allowances = float(self.cap)
        if self.enable_ccr:
            if price >= self.ccr1_trigger:
                allowances += float(self.ccr1_qty)
            if price >= self.ccr2_trigger:
                allowances += float(self.ccr2_qty)
        return allowances

    def enforce_floor(self, price: float) -> float:
        """Apply the price floor if enabled."""

        if self.enable_floor:
            return max(float(price), float(self.floor))
        return float(price)


__all__ = ["AllowanceSupply"]
