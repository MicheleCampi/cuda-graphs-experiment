# CUDA graphs trade-off in vLLM

Measured answer to a question the tuning guides skip: CUDA graphs always speed up
per-token decode, but do they always speed up the *server*? On a saturated 32B they
do not - the sign of the end-to-end throughput delta flips while decode stays faster.

Engine: vLLM 0.23.0, single H100 80GB PCIe (Lambda), driver 580.105.08 / CUDA 13.0.
Models: Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct, bf16, `--max-model-len 4096`.
One variable: `enforce_eager`. Matrix: 2 models x 2 conditions x 2 profiles x 5 reps,
plus a 12-run energy campaign on the sustained cells.

## Headline numbers

| model | profile | queue overload | delta throughput (ON vs OFF) | delta decode (TPOT) |
|---|---|---|---|---|
| 7B  | sporadic  | 0.0x | +8.51% | +8.27% faster |
| 32B | sporadic  | 0.1x | +3.21% | +3.40% faster |
| 7B  | sustained | 2.0x | +4.75% | +7.43% faster |
| 32B | sustained | 7.6x | **-5.15%** | +5.48% faster |

Decode is faster in all four comparisons. Throughput is not. The loss sits above the
kernel, in scheduling and batching.

## Reproducing

```
python3 deep_analysis.py     # throughput, TPOT, wall-clock, overload trend, break-even
python3 aggregate_energy.py  # tok/J and per-phase energy, with reconciliation guard-rails
```

Both read the committed JSON; no GPU required. `deep_analysis.py` section 8 produces the
cold-start and break-even figures as published (section 7 is the older per-cell view, kept
and marked superseded). `aggregate_energy.py` fails loudly if the token denominator or the
per-phase energy split do not reconcile.

## What is in here

- `results/` - 40 matrix runs (JSON) plus the vLLM server logs
- `results-energy-tight/` - 12 energy runs (H100 PCIe, ADR-012), the published energy set
- `results-energy/` - earlier energy campaign, superseded, kept for provenance
- `validation-results/PROVENANCE.md` - hardware, driver and CUDA build
- `PROTOCOL.md` - hypotheses and design, written before the runs
- `RESULTS.md` - generated output of `deep_analysis.py`
- `AUDIT-2026-07-26.md` - pre-publication audit of the article against these files

The server logs are tracked deliberately: engine-init phase timings (graph capture,
`torch.compile`, AOT cache state) exist nowhere else, and several published figures
depend on that decomposition.

## Known limits

- **Cold start is quantised.** Readiness is polled on a ~2s grid, so cold-start values
  move in ~2s steps and per-cell spreads reflect the grid, not physical variance. The
  reported deltas (7s and 16s) are several times the step.
- **The startup delta is not graph capture alone.** `--enforce-eager` also disables the
  compiled path. Capture accounts for 49% of the 7B delta and 61% of the 32B delta; most
  of the rest is `torch.compile`.
- **One replicate is excluded from the cold-start table.** The first start of each campaign
  pays a cold compilation that writes vLLM's AOT cache (7B sporadic ON rep1 = 51.76s against
  ~36.6s for reps 2-5). It is dropped, not averaged in; the exclusion is in the code.
- **The energy window is fixed per model and is not all useful.** It brackets the bench,
  so it also covers a ~15s head while the benchmark loads its dataset, plus a drain tail.
  On the 32B, generation spans ~290s of 400s; on the 7B it can be as little as 36s of 60s.
  The 7B energy row is therefore read as no measurable effect rather than a signed number.
- **n=5 per cell, single card, single run of the matrix.** The throughput distributions
  are cleanly separated, but this is one machine.

## License

Apache-2.0.
