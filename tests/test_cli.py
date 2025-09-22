from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from main.definitions import PROJECT_ROOT


def _tiny_config() -> str:
    return """
years = [2025, 2026]

[allowance_market]
enabled = true
ccr1_enabled = false
ccr2_enabled = false
bank0 = 100000.0
annual_surrender_frac = 1.0
carry_pct = 1.0

[allowance_market.cap]
"2025" = 500000.0
"2026" = 480000.0

[allowance_market.floor]
"2025" = 5.0
"2026" = 7.0
""".strip()


def test_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    config_path.write_text(_tiny_config())

    output_dir = tmp_path / 'outputs'
    cmd = [
        sys.executable,
        '-m',
        'cli.run',
        '--config',
        str(config_path),
        '--years',
        '2025-2026',
        '--out',
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_files = ['allowance.csv', 'emissions.csv', 'prices.csv', 'flows.csv']
    for name in expected_files:
        csv_path = output_dir / name
        assert csv_path.exists(), (
            f"Expected {name} to be generated.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert csv_path.stat().st_size > 0, f'{name} should not be empty'
