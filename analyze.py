#!/usr/bin/env python3
"""Aggregate the 40-run CUDA graphs matrix into per-cell stats + break-even."""
import json, glob, statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path.home() / "cuda-graphs-experiment" / "results"
files = sorted(f for f in RES.glob("*.json") if "bench_tmp" not in f.name)

cells = defaultdict(list)
for f in files:
    d = json.loads(f.read_text())
    key = (d["model"].split("/")[-1], d["condition"], d["profile"])
    cells[key].append(d)

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return (None, None)
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)

def agg(runs):
    bj = [r.get("bench_json") or {} for r in runs]
    gj = [((r.get("inferscope_json") or {}).get("gpu") or {}) for r in runs]
    return {
        "n": len(runs),
        "cold_start_s": mean_std([r.get("cold_start_s") for r in runs]),
        "output_throughput": mean_std([b.get("output_throughput") for b in bj]),
        "total_token_throughput": mean_std([b.get("total_token_throughput") for b in bj]),
        "mean_ttft_ms": mean_std([b.get("mean_ttft_ms") for b in bj]),
        "p99_ttft_ms": mean_std([b.get("p99_ttft_ms") for b in bj]),
        "mean_tpot_ms": mean_std([b.get("mean_tpot_ms") for b in bj]),
        "failed": mean_std([b.get("failed") for b in bj]),
        "gpu_util_mean_pct": mean_std([g.get("utilization_mean_percent") for g in gj]),
        "gpu_power_mean_w": mean_std([(g.get("power_mean_milliwatts") or 0)/1000 for g in gj]),
    }

models = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]
profiles = ["sporadic", "sustained"]

print("="*78)
print("CUDA GRAPHS TRADE-OFF - aggregated (mean over 5 reps; std in parens)")
print("="*78)

results = {}
for m in models:
    for p in profiles:
        results[(m,p)] = (agg(cells[(m,"graphs_on",p)]), agg(cells[(m,"graphs_off",p)]))

def fmt(ms):
    v,s = ms
    return f"{v:8.2f} (+-{s:5.2f})" if v is not None else "    n/a"

for m in models:
    print(f"\n##### {m} #####")
    for p in profiles:
        on, off = results[(m,p)]
        print(f"\n  --- profile: {p} ---")
        print(f"  {'metric':<24} {'graphs_ON':>18} {'graphs_OFF':>18}")
        for label, key in [
            ("cold_start_s","cold_start_s"),
            ("output_throughput tok/s","output_throughput"),
            ("total_token_thru tok/s","total_token_throughput"),
            ("mean_ttft_ms","mean_ttft_ms"),
            ("p99_ttft_ms","p99_ttft_ms"),
            ("mean_tpot_ms","mean_tpot_ms"),
            ("gpu_util_mean_pct","gpu_util_mean_pct"),
            ("gpu_power_mean_w","gpu_power_mean_w"),
            ("failed_requests","failed"),
        ]:
            print(f"  {label:<24} {fmt(on[key]):>18} {fmt(off[key]):>18}")

print("\n" + "="*78)
print("BREAK-EVEN ANALYSIS (sustained profile)")
print("="*78)
for m in models:
    on, off = results[(m,"sustained")]
    cold_on, cold_off = on["cold_start_s"][0], off["cold_start_s"][0]
    thr_on, thr_off = on["output_throughput"][0], off["output_throughput"][0]
    d_cold = cold_on - cold_off
    out_tokens = 128
    t_on  = out_tokens / thr_on  if thr_on  else float('inf')
    t_off = out_tokens / thr_off if thr_off else float('inf')
    d_per_req = t_off - t_on
    print(f"\n  {m}:")
    print(f"    cold_start  ON={cold_on:.2f}s  OFF={cold_off:.2f}s  -> ON extra startup = {d_cold:+.2f}s")
    print(f"    throughput  ON={thr_on:.1f}  OFF={thr_off:.1f} tok/s")
    print(f"    per-req serve ON={t_on*1000:.1f}ms OFF={t_off*1000:.1f}ms -> ON saves {d_per_req*1000:+.2f}ms/req")
    if d_per_req > 0 and d_cold > 0:
        print(f"    => BREAK-EVEN at ~{d_cold/d_per_req:.0f} requests")
    elif d_cold <= 0:
        print(f"    => ON has no startup penalty (dCold<=0): ON wins immediately")
    else:
        print(f"    => ON slower at steady-state too: never breaks even here")
