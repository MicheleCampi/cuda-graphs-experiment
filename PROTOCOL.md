# CUDA Graphs Trade-off — Experiment Protocol

## Research question
CUDA graphs in vLLM trade startup cost for steady-state speed. This experiment
measures both halves on real hardware and derives the **break-even**: after how
many served requests does the steady-state gain repay the extra cold-start cost,
and how does that break-even change with model size?

This is the question a scale-to-zero operator actually faces: enable CUDA graphs
or not? If pods cycle often, the startup penalty may never be repaid. If traffic
is sustained, it pays off. No published measurement of this break-even exists.

## Hypothesis (falsifiable)
H1: enabling CUDA graphs increases cold-start time by a measurable factor.
H2: enabling CUDA graphs reduces inter-token latency (ITL) and/or raises
    sustained throughput at steady state.
H3: a break-even exists, expressed in number of served requests, beyond which
    ON is the better choice.
H4: the break-even shifts with model size (7B vs 32B).

Any of these can be refuted by the data; that is the point.

## Variable under test
`enforce_eager`:
  - True  -> CUDA graphs OFF (eager execution)
  - False -> CUDA graphs ON (capture + replay)
This is the single experimental lever. Everything else held constant.

## Fixed factors (held constant within a model)
- Same model weights, same GPU, same vLLM version, same sampling params.
- Same request workload (fixed prompt set, fixed max_tokens).
- Single H100 on Lambda. No tensor parallelism (TP=1) to avoid confounding
  multi-GPU variables. One variable at a time.

## Models
- Qwen2.5-7B-Instruct
- Qwen2.5-32B-Instruct
Same family to isolate the size variable (no cross-architecture noise).

## Metrics

### Cost side (startup)
- cold_start_seconds: time from `vllm serve` process start to first token
  served on the first request. Measured ON vs OFF.

### Gain side (steady state)
- throughput_tok_s: sustained output tokens/sec under load, ON vs OFF.
- itl_ms: inter-token latency (median, p99), ON vs OFF.
- ttft_ms: time to first token at steady state (warm), ON vs OFF.
Measured with inferscope (client-side timing) + per-device GPU sampling.

### Derived
- break_even_requests = startup_penalty_seconds / per_request_gain_seconds
  where per_request_gain_seconds is the steady-state time saved per request
  by CUDA graphs. Computed per model.

## Validity controls
- N repetitions per condition (not a single run). Report mean + variance.
- Cold-start and steady-state measured separately and explicitly.
- Steady-state measured only after a warmup window (warmup excluded).
- Zero failed requests required for a run to count.
- Identical workload across ON/OFF for the same model.

## Run matrix
2 models x 2 conditions (ON/OFF) x N repetitions x load profile(s).
N and load profiles to be sized to keep GPU cost bounded (decided before
launch, not improvised on the paid node).

## Tooling
- inferscope: per-token client timing, per-device GPU (--sample-only attach).
- vllm bench serve (or equivalent) for sustained load generation.
- vLLM version pinned and recorded at run time (verify on node, do not assume).

## Deliverables
- ANALYSIS.md with the measured break-even per model.
- Article (editorial calendar slot TBD): "CUDA graphs in vLLM: I measured the
  startup-vs-steady-state trade-off. Here's the break-even for scale-to-zero."
- Ties to the cold-start article (art 11), which flagged CUDA graphs as a
  warmup lever without quantifying it. This quantifies it.

## Decisions (resolved before launch)
- N repetitions: 5 per condition. Below 3 = no statistics; 5 gives credible
  mean + variance without exploding GPU cost.
- Load profile: two-level sweep — sporadic (scale-to-zero-like) and sustained.
  The break-even depends on traffic pattern; measuring both turns the finding
  from "here is the break-even" into "here is how it shifts with traffic",
  the full decision map.
- Run matrix: 2 models x 2 conditions x 2 load profiles x 5 reps = 40
  steady-state runs, plus cold-start measured separately per (model, condition).

## Still to verify on the node (not assumptions)
- Exact load generator flags (check `vllm bench serve --help` on the node).
- Whether 32B fits on one H100 at bf16 with adequate KV cache headroom;
  if too tight, consider a quantized variant or smaller second model.
- vLLM version on the node, pinned and recorded at run time.
