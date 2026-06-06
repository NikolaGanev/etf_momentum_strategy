from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>> RUN:", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"Command failed with code {r.returncode}: {' '.join(cmd)}")


def main() -> None:
    root = project_root()
    python = sys.executable  # uses your venv interpreter

    # 1) Update data
    run([python, str(root / "src" / "run_step2.py")])

    # 2) Update regime proxies (topology overlay)
    run([python, str(root / "src" / "run_stepR1_etf_regime_proxies.py")])

    # 3) Build weights (TS-mom + vol target + regime)
    run([python, str(root / "src" / "run_step7_ts_mom.py")])

    # 4) Backtest snapshot (sanity check)
    run([python, str(root / "src" / "run_step9_ts_mom.py")])

    print("\n✅ DAILY PIPELINE COMPLETE (2 → R1 → 7 → 9)\n")


if __name__ == "__main__":
    main()