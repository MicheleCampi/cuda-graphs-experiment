#!/usr/bin/env python3
"""Sanity-check that per-phase energy (ADR-012) landed in the steady JSON.

Run this against results-energy-tight/ after the first H100 rep to confirm
the --metrics-endpoint scrape populated phase_energy before spending the
rest of the budget. It reports, per {tag}.json RunResult, whether
inferscope_steady_json.phase_energy is present and prints its key figures.
It does NOT aggregate or compare — that analysis runs on the full dataset.
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results-energy-tight"


def main() -> int:
    files = sorted(p for p in RES.glob("*.json") if p.name != "bench_tmp.json")
    if not files:
        print(f"no result files in {RES}")
        return 1

    present, absent, skipped = 0, 0, 0
    for f in files:
        d = json.loads(f.read_text())
        # Only RunResult records carry inferscope_steady_json; skip raw bench.
        if "inferscope_steady_json" not in d:
            skipped += 1
            continue
        steady = d.get("inferscope_steady_json") or {}
        pe = steady.get("phase_energy")
        if not isinstance(pe, dict):
            absent += 1
            print(f"  ABSENT  {f.name}  (no phase_energy in steady report)")
            continue
        present += 1
        div = pe.get("phase_energy_divergence")
        ept = pe.get("energy_prefill_by_time_mj")
        epk = pe.get("energy_prefill_by_tokens_mj")
        div_str = f"{div:.4f}" if isinstance(div, float) else "n/a"
        print(
            f"  OK      {f.name}  "
            f"prefill_by_time={ept}mJ  prefill_by_tokens={epk}mJ  "
            f"divergence={div_str}"
        )

    print(
        f"\nsummary: {present} with phase_energy, {absent} without, "
        f"{skipped} non-RunResult skipped"
    )
    # Non-zero exit if every RunResult lacked phase_energy: the scrape
    # silently did nothing, which is the failure this guard exists to catch.
    if present == 0 and absent > 0:
        print("FAIL: no run carried phase_energy — check the scrape endpoint.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
