#!/usr/bin/env python3
"""
CUDA graphs energy re-run orchestrator.
Extends the CUDA graphs trade-off experiment with a THIRD axis: energy
efficiency (tokens/joule). The original 40-run matrix measured the GPU only
during the first single request (idle window); here inferscope --sample-only
(ADR-009) is attached to the EngineCore for the WHOLE bench window, with the
ADR-010 NVML energy counter active, so the energy reflects the actual load.

Reduced matrix: 2 models x 2 conditions x sustained x 3 reps = 12 runs.
Writes to results-energy/ to avoid colliding with the original results/.
"""
import json
import os
import subprocess
import time
import signal
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# VERIFY_ON_NODE — confirm these against the live node before the paid run.
# ---------------------------------------------------------------------------
VLLM_SERVE_CMD = ["vllm", "serve"]
BENCH_CMD = ["vllm", "bench", "serve"]
INFERSCOPE_BIN = os.path.expanduser("~/inferscope/target/release/inferscope")
ENDPOINT = "http://localhost:8000"
HEALTH_PATH = "/v1/models"
MODELS = [
    {"id": "Qwen/Qwen2.5-7B-Instruct",  "served_name": "Qwen/Qwen2.5-7B-Instruct"},
    {"id": "Qwen/Qwen2.5-32B-Instruct", "served_name": "Qwen/Qwen2.5-32B-Instruct"},
]
CONDITIONS = [
    {"name": "graphs_on",  "enforce_eager": False},
    {"name": "graphs_off", "enforce_eager": True},
]
# Reduced to sustained only — the energy question lives under load.
LOAD_PROFILES = [
    {"name": "sustained", "num_prompts": 500, "max_concurrency": 64},
]
N_REPS = 3
WARMUP_REQUESTS = 5
COLD_START_TIMEOUT_S = 600
COLD_START_MAX_TOKENS = 16
MAX_MODEL_LEN = 4096

# inferscope --sample-only window per model. MUST exceed the real bench
# duration: if the window ends mid-load, the energy counter covers only part
# of the work while tokens come from the full bench -> tokens/joule inflated.
# 32B sustained is severely overloaded (~7.6x in the original run) and is the
# cell where the throughput sign inverts (the contrarian finding); truncating
# it would falsify exactly that result, so its window is deliberately wide.
# Excess tail is near-idle and is trimmed in analysis via sample elapsed_ns.
# Validate against bench.duration on rep1; a runtime WARNING fires if the bench
# fills >90% of the window.
STEADY_SAMPLE_SECS = {"7B": 150, "32B": 900}
# Delay between inferscope window-open and bench start, so NVML opens the
# energy window before the first token. Recorded for analysis alignment.
STEADY_ATTACH_OFFSET_S = 0.5

RESULTS_DIR = Path(os.path.expanduser("~/cuda-graphs-experiment/results-energy"))


@dataclass
class RunResult:
    model: str
    condition: str
    profile: str
    rep: int
    cold_start_s: float | None = None
    inferscope_json: dict | None = None          # cold-start single-request probe
    inferscope_steady_json: dict | None = None   # WHOLE-bench-window energy (ADR-009/010)
    steady_window_secs: float | None = None       # --duration-secs used
    steady_attach_offset_s: float | None = None   # inferscope-start -> bench-start gap
    bench_json: dict | None = None
    vllm_version: str | None = None
    error: str | None = None


def wait_until_ready(timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(ENDPOINT + HEALTH_PATH, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def start_vllm(model_id: str, enforce_eager: bool) -> subprocess.Popen:
    cmd = VLLM_SERVE_CMD + [model_id, "--max-model-len", str(MAX_MODEL_LEN)]
    if enforce_eager:
        cmd.append("--enforce-eager")
    log = open(RESULTS_DIR / "vllm_server.log", "ab")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def find_enginecore_pid(api_proc: subprocess.Popen, retries: int = 30,
                        delay_s: float = 1.0) -> int | None:
    """Resolve the EngineCore PID for THIS run, scoped to its process group."""
    try:
        pgid = os.getpgid(api_proc.pid)
    except Exception:
        return None
    for _ in range(retries):
        out = subprocess.run(
            ["pgrep", "-g", str(pgid), "-f", "VLLM::EngineCore"],
            capture_output=True, text=True,
        )
        pids = [int(x) for x in out.stdout.split()]
        if pids:
            return pids[0]
        time.sleep(delay_s)
    return None


def stop_vllm(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def measure_cold_start(proc: subprocess.Popen, served_name: str):
    """
    Time server-start -> first token, capture a single-request inferscope probe,
    and resolve the EngineCore PID (reused for the steady-state attach — the PID
    is stable for the life of the server; warmup does not restart it).
    Returns (cold_start_seconds, inferscope_json, engine_pid).
    """
    t0 = time.time()
    if not wait_until_ready(COLD_START_TIMEOUT_S):
        return None, None, None
    engine_pid = find_enginecore_pid(proc)
    sample_pid = engine_pid if engine_pid is not None else proc.pid
    cmd = [
        INFERSCOPE_BIN,
        "--endpoint", ENDPOINT,
        "--model", served_name,
        "--prompt", "Explain what a Kalman filter does in two sentences.",
        "--max-tokens", str(COLD_START_MAX_TOKENS),
        "--pid", str(sample_pid),
        "--gpu",
        "--json",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        cold_start = time.time() - t0
        js = json.loads(out.stdout) if out.stdout.strip() else None
        return cold_start, js, engine_pid
    except Exception:
        return time.time() - t0, None, engine_pid


def warmup(served_name: str) -> None:
    for _ in range(WARMUP_REQUESTS):
        try:
            subprocess.run(
                [INFERSCOPE_BIN, "--endpoint", ENDPOINT, "--model", served_name,
                 "--prompt", "warmup", "--max-tokens", "8"],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass


def measure_steady_state(served_name: str, profile: dict, engine_pid: int | None,
                         duration_secs: int):
    """
    Run vllm bench serve at the load profile WHILE inferscope --sample-only is
    attached to the EngineCore for the whole window (energy via ADR-010).
    Returns (bench_json, inferscope_steady_json).

    inferscope --sample-only is a fixed-duration timer with no signal handling
    (main.rs: tokio::time::sleep(duration)), so it is NOT killed at bench end;
    it is launched first, brackets the bench, and exits on its own timer,
    printing the resource+GPU JSON to stdout (collected via communicate()).
    """
    if engine_pid is None:
        # Refuse to attach to the APIServer parent: it carries no CUDA context,
        # so its GPU samples are near-idle — the exact artefact this re-run fixes.
        return None, None

    out_path = RESULTS_DIR / "bench_tmp.json"
    out_path.unlink(missing_ok=True)

    infer_cmd = [
        INFERSCOPE_BIN,
        "--sample-only",
        "--pid", str(engine_pid),
        "--duration-secs", str(duration_secs),
        "--gpu",
    ]
    bench_cmd = BENCH_CMD + [
        "--model", served_name,
        "--base-url", ENDPOINT,
        "--dataset-name", "random",
        "--random-input-len", "1024",
        "--random-output-len", "128",
        "--num-prompts", str(profile["num_prompts"]),
        "--max-concurrency", str(profile["max_concurrency"]),
        "--save-result", "--result-filename", str(out_path),
    ]

    infer_proc = None
    bench_json = None
    inferscope_steady_json = None
    try:
        infer_proc = subprocess.Popen(infer_cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
        time.sleep(STEADY_ATTACH_OFFSET_S)  # open the NVML window before load
        subprocess.run(bench_cmd, capture_output=True, text=True,
                       timeout=1800, check=True)
        if out_path.exists():
            bench_json = json.loads(out_path.read_text())
    except Exception:
        bench_json = None
    finally:
        if infer_proc is not None:
            try:
                # sample-only exits on its own timer; wait it out, collect JSON.
                infer_out, _ = infer_proc.communicate(timeout=duration_secs + 60)
                if infer_out and infer_out.strip():
                    inferscope_steady_json = json.loads(infer_out)
            except Exception:
                try:
                    infer_proc.kill()
                except Exception:
                    pass
    return bench_json, inferscope_steady_json


def record_vllm_version() -> str | None:
    try:
        out = subprocess.run(["vllm", "--version"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vllm_version = record_vllm_version()
    for model in MODELS:
        model_key = "32B" if "32B" in model["id"] else "7B"
        duration_secs = STEADY_SAMPLE_SECS[model_key]
        for cond in CONDITIONS:
            for profile in LOAD_PROFILES:
                for rep in range(1, N_REPS + 1):
                    tag = f"{model['id'].split('/')[-1]}_{cond['name']}_{profile['name']}_rep{rep}"
                    out_file = RESULTS_DIR / f"{tag}.json"
                    if out_file.exists():
                        print(f"skip (exists): {tag}")
                        continue
                    print(f"RUN: {tag}  (window={duration_secs}s)")
                    res = RunResult(model=model["id"], condition=cond["name"],
                                    profile=profile["name"], rep=rep,
                                    vllm_version=vllm_version)
                    proc = None
                    try:
                        proc = start_vllm(model["id"], cond["enforce_eager"])
                        res.cold_start_s, res.inferscope_json, engine_pid = \
                            measure_cold_start(proc, model["served_name"])
                        if res.cold_start_s is None:
                            res.error = "server did not become ready"
                        elif engine_pid is None:
                            res.error = ("could not resolve EngineCore PID; "
                                         "refusing to attach to APIServer for energy")
                        else:
                            warmup(model["served_name"])
                            res.steady_window_secs = float(duration_secs)
                            res.steady_attach_offset_s = STEADY_ATTACH_OFFSET_S
                            res.bench_json, res.inferscope_steady_json = \
                                measure_steady_state(model["served_name"], profile,
                                                     engine_pid, duration_secs)
                            if res.bench_json is None:
                                res.error = "bench produced no result"
                            elif res.inferscope_steady_json is None:
                                res.error = "steady-state inferscope attach produced no result"
                            else:
                                bd = res.bench_json.get("duration")
                                if bd is not None and bd >= duration_secs * 0.9:
                                    print(f"  WARNING: bench duration {bd:.1f}s >= 90% of "
                                          f"window {duration_secs}s — energy window may be "
                                          f"truncated; raise STEADY_SAMPLE_SECS['{model_key}']")
                    except Exception as e:
                        res.error = f"{type(e).__name__}: {e}"
                    finally:
                        if proc is not None:
                            stop_vllm(proc)
                        time.sleep(10)
                    out_file.write_text(json.dumps(asdict(res), indent=2))
                    print(f"  saved: {out_file.name}  cold_start={res.cold_start_s}  err={res.error}")
    print("done. results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
