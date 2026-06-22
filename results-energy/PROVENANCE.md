# Energy re-run provenance

**Hardware:** NVIDIA H100 80GB HBM3 (SXM5), Lambda Cloud, single GPU
**NB:** differs from the original 40-run matrix in `results/`, which ran on H100 **PCIe**.
SXM5 has higher memory bandwidth and TDP than PCIe; absolute decode latency and
throughput here are NOT directly comparable to `results/`. The energy axis
(tokens/joule) is the new measurement; throughput/cold-start findings in the
article remain sourced from the original PCIe runs.

**Software:** vLLM 0.23.0, inferscope HEAD 6ca675f (rustls, MSRV 1.85),
built `--features gpu-nvidia`. Energy via NVML hardware counter (ADR-010),
`energy_source: "counter"` on all 12 runs.

**Method:** inferscope `--sample-only --pid <EngineCore> --gpu` (ADR-009) attached
over the full bench window. Windows: 7B=150s, 32B=900s (deliberately wide;
actual bench fill ~12%, trimmed in analysis via sample elapsed_ns).
EngineCore PID resolved per run; attach to APIServer refused by design.

**Matrix:** 2 models (Qwen2.5-7B/32B-Instruct) x 2 conditions (graphs on/off)
x sustained (num_prompts=500, max_concurrency=64) x 3 reps = 12 runs.

**Date:** 2026-06-22
