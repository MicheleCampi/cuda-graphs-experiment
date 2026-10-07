# Provenance — `results-energy-tight/`

The twelve JSON files in this directory are the **H100 PCIe** energy run
committed at `2e93ef0` (2026-06-28). Their provenance — hardware, software
stack, window calibration, integrity checks and the figures re-derived with
`aggregate_energy.py` — is recorded in
[`validation-results/PROVENANCE.md`](../validation-results/PROVENANCE.md),
which is the canonical record.

Until this commit, this file described an earlier run on an **H100 SXM5**,
committed at `92cf980` (2026-06-22). Its data were replaced at `2e93ef0` and
this file was not updated with them, so it gave the wrong GPU, date, sampling
windows (45 s / 150 s; the files here use 60 s / 400 s), inferscope commit
and figures for the files beside it. The SXM5 run is still in the history at
`92cf980`. Its tokens/joule are not comparable with the PCIe ones: see
"Hardware — read this first" in the canonical record.
