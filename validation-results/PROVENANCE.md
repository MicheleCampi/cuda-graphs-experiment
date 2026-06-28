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
| vLLM            | 0.23.0 (cu13 build)              |

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

## Headline finding (from run; full aggregation pending — see analysis)

tokens/joule, graphs_on vs graphs_off:

- **7B:** 4.660 vs 4.646 tok/J (+0.3%), energy ~equal. CUDA graphs
  marginally better.
- **32B (saturated):** 0.790 vs 0.804 tok/J (−1.7%); energy 81046 vs
  79584 J (+1.8%). CUDA graphs **worse** under saturation.

This converges with the original 40-run throughput result, where
graphs_on inverted on the 32B under 7.6× overload (−5.15%). Two
independent metrics (throughput + NVML energy), same conclusion.

Per-phase divergence, stable and model-dependent across reps:
7B ~−0.69, 32B ~−0.60.

> Numbers above are as reported from the run. Detailed re-derivation from
> the 12 JSON (token-denominator comparability 7B vs 32B included) is the
> aggregate analysis step, committed separately.

## Source data

- Dataset: `results-energy-tight/Qwen2.5-{7B,32B}-Instruct_graphs_{on,off}_sustained_rep{1,2,3}.json`
- Canonical commit: `2e93ef0`
- Window calibration: `751a11d`
