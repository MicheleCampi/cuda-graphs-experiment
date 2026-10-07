# Validation Provenance — ADR-012 (per-phase energy attribution)

## What this validates

ADR-012 promotes per-phase energy attribution (NVML counters) from
`Accepted (pending empirical validation)` to empirically validated.
This file is the canonical provenance record for the H100 PCIe run that
closes that gate. The dataset itself lives at `results-energy-tight/`
(committed in this repo); it is **not** duplicated here.

## Hardware — read this first

- **GPU: 1× NVIDIA H100 PCIe (~350W envelope).**
- This **supersedes** an earlier energy run on **H100 SXM5 (~700W)**,
  recorded at commit `92cf980` and replaced at commit `2e93ef0`.
- SXM5 and PCIe have different power envelopes, so **absolute joules and
  tokens/joule are NOT comparable across the two**. The PCIe run is the
  canonical one because it matches the hardware of the original 40-run
  throughput experiment (also 1× H100 PCIe). The independent-metrics
  argument — throughput and NVML energy converging on the same 32B
  inversion — only holds on identical hardware.
- The JSON files carry **no internal hardware tag**. Hardware provenance
  exists only in this file and in commit message `2e93ef0`. Do not move,
  rename, or re-run these files without preserving that link.

## Software stack

| Component       | Version                          |
|-----------------|----------------------------------|
| Driver          | 580.105.08                       |
| CUDA            | 13.0                             |
| Lambda image    | lambda-stack-24-04 (24.4.4-2141) |
| vLLM            | 0.23.0 (cu13 build)              |
| inferscope      | 62f03d6, reconstructed (see below) |

The inferscope commit was not recorded at run time. 62f03d6 (2026-06-27)
added per-phase energy to the `--sample-only` window, which these JSON carry,
so the binary was built at or after it. The only later commit on main before
the data commit 2e93ef0 (2026-06-28 12:27 UTC) is c0bebb9, which adds one
`#[test]` function to `crates/inferscope/src/cli.rs` and builds the same
binary. This assumes the binary was built from main without local changes,
and that commit dates reflect when the commits existed.

## Experiment matrix

2 models × 2 conditions × 3 reps = **12 runs**.

- Models: `Qwen2.5-7B-Instruct`, `Qwen2.5-32B-Instruct`
- Conditions: `graphs_on`, `graphs_off` (CUDA graphs / enforce_eager)
- Profile: `sustained`
- Steady-state sampling window (`STEADY_SAMPLE_SECS`, calibrated to real
  bench duration at commit `751a11d`): **7B = 60s, 32B = 400s**
  (real bench ~35s / ~300s respectively).

## Integrity — verified file-by-file

All 12 JSON in `results-energy-tight/Qwen2.5-*.json`:

- `energy_source: "counter"` — real NVML energy counters under production
  load (not power-integration fallback).
- `"error": null` — no failed reps.

## Headline finding (aggregated -- re-derived from 12 JSON via `aggregate_energy.py`)
Token denominator = `bench_json.total_output_tokens` = **64000** on all 12
files (500 prompts x 128 max output, capped; no early-EOS), so the 7B-vs-32B
cross-model denominator is clean and tokenizer-independent. Energy =
`inferscope_steady_json.gpu.energy_millijoules`. All guard-rails pass:
both apportionments sum to total energy; divergence re-derived == stored.

tokens/joule (mean of 3 reps), graphs_on vs graphs_off:
- **7B:** 4.660 vs 4.646 tok/J (**+0.3%**); energy 13735 vs 13775 J (−0.3%).
  CUDA graphs marginally better.
- **32B (saturated):** 0.790 vs 0.804 tok/J (**−1.8%**); energy 81046 vs
  79584 J (+1.8%). CUDA graphs **worse** under saturation.

The 32B penalty is a **power** effect, not a time effect: mean power
199.3->203.0 W (+1.9%) at ~equal active fraction -- graphs run the saturated
32B hotter without compensating in throughput.

This converges with the original 40-run throughput result, where graphs_on
inverted on the 32B under 7.6x overload (−5.15%). Two fully independent
metrics (throughput + NVML energy), same sign, same conclusion.

Per-phase divergence, stable and model-dependent across reps:
7B ~−0.69, 32B ~−0.60.

### Active-fraction caveat (window vs bench duration)
The steady window is fixed; the bench fills only part of it, so absolute
tok/J is diluted by idle tail -- but symmetrically within each model, so the
**relative** on/off claim holds. `active = bench_duration / window`:
- **32B:** symmetric (off 0.72 / on 0.75) -- clean comparison.
- **7B:** off 0.72 / on 0.65 (graphs shorten TPOT -> shorter bench -> more
  idle tail on the on-side). This penalizes graphs_on, yet it still ties/wins.
  The +0.3% is therefore a floor, not an idle-inflated artifact.

## Source data

- Dataset: `results-energy-tight/Qwen2.5-{7B,32B}-Instruct_graphs_{on,off}_sustained_rep{1,2,3}.json`
- Canonical commit: `2e93ef0`
- Aggregation script: `aggregate_energy.py` (guard-railed re-derivation)
- Window calibration: `751a11d`
