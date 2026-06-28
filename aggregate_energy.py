import json, glob
from collections import defaultdict

FILES = sorted(glob.glob('results-energy-tight/Qwen2.5-*_graphs_*_sustained_rep*.json'))
assert len(FILES) == 12, f"attesi 12 file, trovati {len(FILES)}"

rows = []
fails = []
for f in FILES:
    d = json.load(open(f))
    s = d['inferscope_steady_json']; b = d['bench_json']; pe = s['phase_energy']
    energy_mj = s['gpu']['energy_millijoules']
    out_tok   = b['total_output_tokens']
    gen_delta = pe['generation_tokens_delta']
    tok_j     = out_tok / (energy_mj / 1e3)
    active    = b['duration'] / d['steady_window_secs']
    pf_t, pf_k = pe['energy_prefill_by_time_mj'], pe['energy_prefill_by_tokens_mj']
    dec_t, dec_k = pe['energy_decode_by_time_mj'], pe['energy_decode_by_tokens_mj']
    div_calc  = (pf_t - pf_k) / energy_mj
    div_store = pe['phase_energy_divergence']

    # guard-rail
    if out_tok != 64000:                         fails.append(f"{f}: out_tok={out_tok} != 64000")
    if out_tok != gen_delta:                     fails.append(f"{f}: out_tok {out_tok} != gen_delta {gen_delta}")
    if abs((pf_t+dec_t) - energy_mj) > 1:        fails.append(f"{f}: by_time sum {pf_t+dec_t} != energy {energy_mj}")
    if abs((pf_k+dec_k) - energy_mj) > 1:        fails.append(f"{f}: by_tok sum {pf_k+dec_k} != energy {energy_mj}")
    if abs(div_calc - div_store) > 1e-6:         fails.append(f"{f}: div_calc {div_calc:.6f} != stored {div_store:.6f}")

    rows.append(dict(model=d['model'].split('/')[-1], cond=d['condition'], rep=d['rep'],
                     energy_j=energy_mj/1e3, out_tok=out_tok, tok_j=tok_j, div=div_calc,
                     active=active, power_w=s['gpu']['power_mean_milliwatts']/1e3,
                     util=s['gpu']['utilization_mean_percent'], dur=b['duration']))

print("=== GUARD-RAIL ===")
print("TUTTI OK" if not fails else "\n".join(fails))

print("\n=== PER-FILE ===")
print(f"{'model':22}{'cond':11}{'rep':4}{'energy_J':>10}{'tok/J':>9}{'div':>9}{'active':>8}{'pow_W':>8}{'util':>6}")
for r in sorted(rows, key=lambda x:(x['model'],x['cond'],x['rep'])):
    print(f"{r['model']:22}{r['cond']:11}{r['rep']:<4}{r['energy_j']:>10.1f}{r['tok_j']:>9.3f}{r['div']:>9.3f}{r['active']:>8.2f}{r['power_w']:>8.1f}{r['util']:>6}")

print("\n=== AGGREGATO (media 3 rep) ===")
agg = defaultdict(list)
for r in rows: agg[(r['model'],r['cond'])].append(r)
mean = lambda xs,k: sum(x[k] for x in xs)/len(xs)
summary = {}
for (m,c),xs in sorted(agg.items()):
    summary[(m,c)] = dict(tok_j=mean(xs,'tok_j'), energy_j=mean(xs,'energy_j'),
                          div=mean(xs,'div'), active=mean(xs,'active'), power_w=mean(xs,'power_w'))
    s = summary[(m,c)]
    print(f"{m:22}{c:11} tok/J={s['tok_j']:.3f}  E_J={s['energy_j']:.0f}  div={s['div']:.3f}  active={s['active']:.2f}  pow_W={s['power_w']:.1f}")

print("\n=== EFFETTO GRAPHS on/off ===")
for m in sorted(set(r['model'] for r in rows)):
    on, off = summary[(m,'graphs_on')], summary[(m,'graphs_off')]
    d_tok = (on['tok_j']-off['tok_j'])/off['tok_j']*100
    d_e   = (on['energy_j']-off['energy_j'])/off['energy_j']*100
    print(f"{m:22} tok/J on={on['tok_j']:.3f} off={off['tok_j']:.3f} ({d_tok:+.1f}%) | E on={on['energy_j']:.0f} off={off['energy_j']:.0f} ({d_e:+.1f}%) | div on={on['div']:.3f} off={off['div']:.3f}")
