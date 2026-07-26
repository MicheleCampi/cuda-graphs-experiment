#!/usr/bin/env python3
"""
Deep analysis of the CUDA graphs trade-off experiment (40 runs).
Regenerates the full evidence chain behind the article's thesis:
  1. per-cell aggregates (8 cells, mean+-pstdev over 5 reps)
  2. raw 5-rep distributions (rules out noise: ON vs OFF separation)
  3. TPOT improvement per cell (graphs always speed per-token decode)
  4. wall-clock / completed / req-throughput (same work, ON slower under load)
  5. GPU power+util from inferscope (physical signature of wasted work)
  6. dThroughput vs queue-overload trend (two-factor causal model)
  7. break-even (sustained only, per-cell cold start) - superseded by 8
  8. published cold-start and break-even: cold start pooled across profiles with
     the AOT-cache-contaminated replicate excluded, break-even in both load
     regimes. These are the figures that appear in the article.
All numbers trace to results/*.json. See PROTOCOL.md for design.
"""
import json, glob, statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct"]
PROFILES = ["sporadic", "sustained"]
# KV-cache max concurrency measured at serve time (max_model_len=4096):
CAP = {"Qwen2.5-7B-Instruct": 31.8, "Qwen2.5-32B-Instruct": 8.45}
CONC = {"sporadic": 1, "sustained": 64}
OUT_TOKENS = 128  # random-output-len, fixed

def load():
    runs = defaultdict(list)
    for f in RES.glob("*.json"):
        if "bench_tmp" in f.name:
            continue
        d = json.loads(f.read_text())
        runs[(d["model"].split("/")[-1], d["condition"], d["profile"])].append(d)
    return runs

def ms(xs):
    xs = [x for x in xs if x is not None]
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0) if xs else (None, None)

def bench(runs):  # list of bench_json dicts
    return [r.get("bench_json") or {} for r in runs]

def gpu(runs):
    return [((r.get("inferscope_json") or {}).get("gpu") or {}) for r in runs]

def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

R = load()

hr("1. PER-CELL AGGREGATES (mean +- pstdev, n=5)")
for m in MODELS:
    print(f"\n##### {m} #####")
    for p in PROFILES:
        on, off = R[(m,"graphs_on",p)], R[(m,"graphs_off",p)]
        bo, bf = bench(on), bench(off)
        print(f"\n  --- {p} (overload {CONC[p]/CAP[m]:.1f}x) ---")
        print(f"  {'metric':<22}{'ON':>16}{'OFF':>16}")
        rows = [
            ("cold_start_s", [r.get("cold_start_s") for r in on], [r.get("cold_start_s") for r in off]),
            ("out_thru tok/s", [b.get("output_throughput") for b in bo], [b.get("output_throughput") for b in bf]),
            ("mean_ttft_ms", [b.get("mean_ttft_ms") for b in bo], [b.get("mean_ttft_ms") for b in bf]),
            ("mean_tpot_ms", [b.get("mean_tpot_ms") for b in bo], [b.get("mean_tpot_ms") for b in bf]),
            ("duration_s", [b.get("duration") for b in bo], [b.get("duration") for b in bf]),
            ("completed", [b.get("completed") for b in bo], [b.get("completed") for b in bf]),
        ]
        for name, o, f in rows:
            mo, so = ms(o); mf, sf = ms(f)
            print(f"  {name:<22}{mo:8.2f}(±{so:5.2f}){mf:8.2f}(±{sf:5.2f})")

hr("2. RAW 5-REP DISTRIBUTIONS (output_throughput) - rules out noise")
for m in MODELS:
    for p in PROFILES:
        on = sorted(b.get("output_throughput") for b in bench(R[(m,"graphs_on",p)]))
        off = sorted(b.get("output_throughput") for b in bench(R[(m,"graphs_off",p)]))
        sep = "SEPARATED" if (min(on) > max(off) or max(on) < min(off)) else "OVERLAP"
        print(f"  {m:24} {p:10} ON{[round(x,1) for x in on]} OFF{[round(x,1) for x in off]} -> {sep}")

hr("3. TPOT IMPROVEMENT FROM ON (decode-only) - graphs always help compute")
for m in MODELS:
    for p in PROFILES:
        o = st.mean(b.get("mean_tpot_ms") for b in bench(R[(m,"graphs_on",p)]))
        f = st.mean(b.get("mean_tpot_ms") for b in bench(R[(m,"graphs_off",p)]))
        print(f"  {m:24} {p:10} TPOT ON={o:6.2f} OFF={f:6.2f} -> ON {100*(f-o)/f:+5.2f}% faster decode")

hr("4. SAME WORK, DIFFERENT WALL-CLOCK (sustained)")
for m in MODELS:
    bo, bf = bench(R[(m,"graphs_on","sustained")]), bench(R[(m,"graphs_off","sustained")])
    do, df = st.mean(b.get("duration") for b in bo), st.mean(b.get("duration") for b in bf)
    co, cf = st.mean(b.get("completed") for b in bo), st.mean(b.get("completed") for b in bf)
    to, tf = st.mean(b.get("total_output_tokens") for b in bo), st.mean(b.get("total_output_tokens") for b in bf)
    print(f"  {m:24} ON: {do:.1f}s {co:.0f}req {to:.0f}tok | OFF: {df:.1f}s {cf:.0f}req {tf:.0f}tok | ON +{100*(do-df)/df:.1f}% wall-clock")

hr("5. GPU POWER/UTIL FROM INFERSCOPE (cold-start probe) - physical signature")
for m in MODELS:
    for p in PROFILES:
        go, gf = gpu(R[(m,"graphs_on",p)]), gpu(R[(m,"graphs_off",p)])
        uo = st.mean(g.get("utilization_mean_percent") for g in go)
        uf = st.mean(g.get("utilization_mean_percent") for g in gf)
        po = st.mean((g.get("power_mean_milliwatts") or 0)/1000 for g in go)
        pf = st.mean((g.get("power_mean_milliwatts") or 0)/1000 for g in gf)
        print(f"  {m:24} {p:10} util ON={uo:5.1f}% OFF={uf:5.1f}% | power ON={po:6.1f}W OFF={pf:6.1f}W")

hr("6. dTHROUGHPUT vs QUEUE-OVERLOAD (two-factor model)")
print(f"  {'model':24} {'profile':10} {'overload':>9} {'dThru%':>9}")
rows = []
for m in MODELS:
    for p in PROFILES:
        o = st.mean(b.get("output_throughput") for b in bench(R[(m,"graphs_on",p)]))
        f = st.mean(b.get("output_throughput") for b in bench(R[(m,"graphs_off",p)]))
        rows.append((CONC[p]/CAP[m], m, p, 100*(o-f)/f))
for ovl, m, p, d in rows:
    print(f"  {m:24} {p:10} {ovl:8.1f}x {d:+8.2f}%")

hr("7. BREAK-EVEN (sustained, per-cell cold start) - SUPERSEDED BY SECTION 8")
for m in MODELS:
    on, off = R[(m,"graphs_on","sustained")], R[(m,"graphs_off","sustained")]
    cold_on = st.mean(r.get("cold_start_s") for r in on)
    cold_off = st.mean(r.get("cold_start_s") for r in off)
    thr_on = st.mean(b.get("output_throughput") for b in bench(on))
    thr_off = st.mean(b.get("output_throughput") for b in bench(off))
    d_cold = cold_on - cold_off
    t_on, t_off = OUT_TOKENS/thr_on, OUT_TOKENS/thr_off
    d_per_req = t_off - t_on
    print(f"\n  {m}: extra startup ON = {d_cold:+.2f}s | per-req ON saves {d_per_req*1000:+.2f}ms")
    if d_per_req > 0 and d_cold > 0:
        print(f"    -> break-even at ~{d_cold/d_per_req:.0f} requests")
    elif d_cold <= 0:
        print(f"    -> ON wins immediately (no startup penalty)")
    else:
        print(f"    -> never breaks even (ON slower at steady-state too)")


hr("8. PUBLISHED COLD-START + BREAK-EVEN (pooled, contamination excluded)")
# Cold start is measured before the bench, so the load profile does not affect it:
# the two profiles are replicates of the same quantity and are pooled (n=10).
# EXCLUSION: rep1 of each campaign pays a cold torch.compile that writes vLLM's AOT
# cache (7B sporadic ON r1 = 51.76s vs ~36.6s for reps 2-5); later starts load it.
# Only that replicate is contaminated in this matrix; it is dropped, not averaged in.
# Readiness is polled on a ~2s grid, so figures are quantised and rounded to 1s.
EXCLUDE = {("Qwen2.5-7B-Instruct", "graphs_on", "sporadic", 1)}

def cold_pooled(m, cond):
    vals = []
    for p in PROFILES:
        for r in R[(m, cond, p)]:
            if (m, cond, p, r.get("rep")) in EXCLUDE:
                continue
            v = r.get("cold_start_s")
            if v is not None:
                vals.append(v)
    return vals

print("\n  cold start, pooled over profiles (published table):")
cold = {}
for m in MODELS:
    on, off = cold_pooled(m, "graphs_on"), cold_pooled(m, "graphs_off")
    cold[m] = (st.mean(on), st.mean(off))
    print(f"    {m:24} ON {st.mean(on):5.1f}s (n={len(on):2d})  "
          f"OFF {st.mean(off):5.1f}s (n={len(off):2d})  -> +{st.mean(on)-st.mean(off):.1f}s")

print("\n  break-even by load regime (published table):")
print("    sustained: gain is aggregate throughput, amortised across a saturated batch")
print("    sporadic : gain is per-request latency at concurrency 1 (TPOT x output tokens)")
for m in MODELS:
    d_cold = cold[m][0] - cold[m][1]
    for p in PROFILES:
        on, off = R[(m, "graphs_on", p)], R[(m, "graphs_off", p)]
        if p == "sustained":
            thr_on = st.mean(b.get("output_throughput") for b in bench(on))
            thr_off = st.mean(b.get("output_throughput") for b in bench(off))
            save_s = OUT_TOKENS/thr_off - OUT_TOKENS/thr_on
        else:
            tp_on = st.mean(b.get("mean_tpot_ms") for b in bench(on))
            tp_off = st.mean(b.get("mean_tpot_ms") for b in bench(off))
            save_s = (tp_off - tp_on) * OUT_TOKENS / 1000.0
        verdict = (f"~{d_cold/save_s:>6.0f} requests" if save_s > 0
                   else "     never (ON slower at steady state)")
        print(f"    {m:24} {p:10} saves {save_s*1000:+8.1f} ms/req  "
              f"| startup +{d_cold:5.1f}s -> {verdict}")
