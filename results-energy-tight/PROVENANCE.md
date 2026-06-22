# Energy re-run (tight window) provenance

**Supersedes the absolute tokens/joule of `results-energy/`** for the
energy axis. Same hardware, same software, same matrix — only the
sampling window changed.

**Why this run exists:** the wide-window run (`results-energy/`, 150s/900s)
measured counter energy over the whole attach window, which includes a
long idle tail after the bench. Because `--sample-only` emits aggregated
metrics, not raw per-tick samples, that tail cannot be trimmed post-hoc
(verified in is-report/resource_report.rs and is-sysmon/gpu_nvidia.rs:
`energy_millijoules = counter_end - counter_start` over the full window).
Tight windows (7B=45s, 32B=150s ≈ bench + margin) make the counter delta
equal the bench energy directly — clean absolutes, no idle subtraction,
no P_idle estimate.

**Hardware:** NVIDIA H100 80GB HBM3 (SXM5), Lambda Cloud, single GPU,
driver 580.105.08. (Same SXM5 as results-energy/; NOT the PCIe of results/.)
**Software:** vLLM 0.23.0, inferscope HEAD 6ca675f, --features gpu-nvidia.
Energy via NVML hardware counter (ADR-010), energy_source=counter on all 12.

**Results (clean, window=bench, 3 reps each):**
- 7B:  tok/J on=4.18 off=4.21  Δ=-0.70% (negligible) | throughput Δ=+23.2%
- 32B: tok/J on=0.845 off=0.888 Δ=-4.83%            | throughput Δ=-4.12%
- 7B: graphs on buy throughput at no per-token energy cost.
- 32B under overload: graphs on lose on BOTH axes (throughput AND energy).
Max bench fill 74.2% of window — no truncation on any cell.

**Date:** 2026-06-22
