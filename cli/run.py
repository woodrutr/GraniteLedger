from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import typer

from engine.run_loop import run_end_to_end_from_frames
from engine.outputs import EngineOutputs
from gui.app import (
    _build_default_frames,
    _build_policy_frame,
    _ensure_years_in_demand,
    _load_config_data,
    _years_from_config,
)

app = typer.Typer(help='Run the annual allowance market engine end-to-end.')


def _parse_years_option(value: str) -> list[int]:
    """Parse the ``--years`` option into a sorted list of integers."""

    years: set[int] = set()
    for part in value.split(','):
        token = part.strip()
        if not token:
            continue
        if '-' in token:
            start_str, end_str = token.split('-', 1)
            try:
                start = int(start_str.strip())
                end = int(end_str.strip())
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid year range '{token}'") from exc
            step = 1 if end >= start else -1
            years.update(range(start, end + step, step))
        else:
            try:
                years.add(int(token))
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid year '{token}'") from exc

    if not years:
        raise ValueError('No valid years supplied')

    return sorted(years)


def _resolve_years(years_option: str | None, config: Mapping[str, Any]) -> list[int]:
    """Determine the simulation years from CLI overrides or configuration."""

    if years_option:
        return _parse_years_option(years_option)

    years = _years_from_config(config)
    if not years:
        raise ValueError('No simulation years specified; supply --years to override the config')
    return years


def _coerce_bool(value: Any, *, default: bool = True) -> bool:
    """Interpret booleans from common TOML representations."""

    if value in (None, ''):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(int(value))
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', 't', 'yes', 'y', '1', 'on'}:
            return True
        if normalized in {'false', 'f', 'no', 'n', '0', 'off'}:
            return False
    return default


def _extract_policy_flags(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return policy enablement flags derived from the configuration mapping."""

    market_cfg = config.get('allowance_market')
    if not isinstance(market_cfg, Mapping):
        market_cfg = {}

    carbon_enabled = _coerce_bool(market_cfg.get('enabled'), default=True)
    ccr1_enabled = _coerce_bool(market_cfg.get('ccr1_enabled'), default=True)
    ccr2_enabled = _coerce_bool(market_cfg.get('ccr2_enabled'), default=True)

    banking_key = 'allowance_banking_enabled'
    if banking_key not in market_cfg:
        banking_key = 'bank_enabled'
    banking_enabled = _coerce_bool(market_cfg.get(banking_key), default=True)

    if not carbon_enabled:
        banking_enabled = False

    control_period: int | None
    raw_control = market_cfg.get('control_period_years')
    if raw_control in (None, ''):
        control_period = None
    else:
        try:
            control_period = int(raw_control)
        except (TypeError, ValueError):
            control_period = None
        if control_period is not None and control_period <= 0:
            control_period = None

    floor_mode = str(market_cfg.get('floor_escalator_mode', 'fixed')).strip().lower()
    if floor_mode not in {'percent', 'fixed'}:
        floor_mode = 'fixed'

    floor_value = market_cfg.get('floor_escalator_value')
    try:
        floor_growth = float(floor_value)
    except (TypeError, ValueError):
        floor_growth = 0.0

    ccr1_growth = market_cfg.get('ccr1_escalator_pct', 0.0)
    try:
        ccr1_growth = float(ccr1_growth)
    except (TypeError, ValueError):
        ccr1_growth = 0.0

    ccr2_growth = market_cfg.get('ccr2_escalator_pct', 0.0)
    try:
        ccr2_growth = float(ccr2_growth)
    except (TypeError, ValueError):
        ccr2_growth = 0.0

    return {
        'carbon_policy_enabled': carbon_enabled,
        'ccr1_enabled': ccr1_enabled,
        'ccr2_enabled': ccr2_enabled,
        'banking_enabled': banking_enabled,
        'control_period_years': control_period,
        'floor_escalator_mode': floor_mode,
        'floor_escalator_value': floor_growth,
        'ccr1_escalator_pct': ccr1_growth,
        'ccr2_escalator_pct': ccr2_growth,
    }


def _run_engine(
    config: Mapping[str, Any],
    years: Sequence[int],
    *,
    use_network: bool = False,
) -> EngineOutputs:
    """Execute the dispatch/allowance engine for ``years`` using ``config``."""

    flags = _extract_policy_flags(config)
    carbon_enabled = bool(flags['carbon_policy_enabled'])
    ccr1_enabled = bool(flags['ccr1_enabled'])
    ccr2_enabled = bool(flags['ccr2_enabled'])
    banking_enabled = bool(flags['banking_enabled'])
    control_period = flags.get('control_period_years')

    frames = _build_default_frames(
        years,
        carbon_policy_enabled=carbon_enabled,
        banking_enabled=banking_enabled,
    )
    frames = _ensure_years_in_demand(frames, years)
    policy_frame = _build_policy_frame(
        config,
        years,
        carbon_enabled,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        control_period_years=control_period,
        banking_enabled=banking_enabled,
        floor_escalator_mode=flags.get('floor_escalator_mode'),
        floor_escalator_value=flags.get('floor_escalator_value'),
        ccr1_escalator_pct=flags.get('ccr1_escalator_pct'),
        ccr2_escalator_pct=flags.get('ccr2_escalator_pct'),
    )
    frames = frames.with_frame('policy', policy_frame)

    enable_ccr = carbon_enabled and (ccr1_enabled or ccr2_enabled)

    return run_end_to_end_from_frames(
        frames,
        years=years,
        price_initial=0.0,
        enable_floor=carbon_enabled,
        enable_ccr=enable_ccr,
        use_network=use_network,
    )


def _write_outputs(outputs: EngineOutputs, out_dir: Path) -> None:
    """Persist model outputs to ``out_dir`` using the CLI naming convention."""

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(
        out_dir,
        annual_filename='allowance.csv',
        emissions_filename='emissions.csv',
        price_filename='prices.csv',
        flows_filename='flows.csv',
    )


@app.command()
def main(
    config: Path | None = typer.Option(
        None,
        '--config',
        '-c',
        help='Path to the TOML configuration file (defaults to run_config.toml).',
    ),
    years: str | None = typer.Option(
        None,
        '--years',
        help='Override simulation years (e.g. "2025,2026" or "2025-2027").',
    ),
    use_network: bool = typer.Option(
        False,
        '--use-network',
        help='Enable network dispatch with transmission constraints.',
    ),
    out: Path = typer.Option(
        Path('output'),
        '--out',
        '-o',
        help='Directory where CSV outputs will be written.',
    ),
) -> None:
    """Run the policy model end-to-end and export CSV outputs."""

    try:
        config_data = _load_config_data(config)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f'Failed to load configuration: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    try:
        simulation_years = _resolve_years(years, config_data)
    except Exception as exc:
        typer.secho(f'Invalid year selection: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(2)

    try:
        outputs = _run_engine(config_data, simulation_years, use_network=use_network)
    except ModuleNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(3)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f'Model execution failed: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(3)

    try:
        _write_outputs(outputs, out)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f'Failed to write outputs: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(4)

    typer.secho(
        f'Saved allowance results for years {", ".join(map(str, simulation_years))} to {out.resolve()}',
        fg=typer.colors.GREEN,
    )


if __name__ == '__main__':  # pragma: no cover - CLI entry point
    app()
