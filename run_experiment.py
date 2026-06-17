#!/usr/bin/env python3
"""
CUDA graphs trade-off experiment orchestrator.
Design-first: see PROTOCOL.md. Measures cold-start cost vs steady-state gain
of CUDA graphs (enforce_eager on/off) across models and load profiles.

Run on the GPU node. Everything marked VERIFY_ON_NODE must be confirmed
against the actual node before the paid run (vllm bench flags, model fit).
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
VLLM_SERVE_CMD = ["vllm", "serve"]          # confirm entrypoint on node
BENCH_CMD = ["vllm", "bench", "serve"]      # VERIFY: exact subcommand + flags
INFERSCOPE_BIN = os.path.expanduser("~/inferscope/target/release/inferscope")
ENDPOINT = "http://localhost:8000"          # VERIFY: vllm default serve port on node
HEALTH_PATH = "/v1/models"                  # readiness probe path
# Models — VERIFY 32B fits one H100 at bf16 with KV headroom; else quantize/smaller.
MODELS = [
    {"id": "Qwen/Qwen2.5-7B-Instruct",  "served_name": "Qwen/Qwen2.5-7B-Instruct"},
    {"id": "Qwen/Qwen2.5-32B-Instruct", "served_name": "Qwen/Qwen2.5-32B-Instruct"},
]
CONDITIONS = [
    {"name": "graphs_on",  "enforce_eager": False},   # CUDA graphs ON  (default)
    {"name": "graphs_off", "enforce_eager": True},    # CUDA graphs OFF (eager)
]
# Load profiles — VERIFY flag names against `vllm bench serve --help` on node.
LOAD_PROFILES = [
    {"name": "sporadic",  "num_prompts": 50,   "max_concurrency": 1},
    {"name": "sustained", "num_prompts": 500,  "max_concurrency": 64},
]
N_REPS = 5
WARMUP_REQUESTS = 5                  # excluded from steady-state measurement
COLD_START_TIMEOUT_S = 600          # 32B load can be slow; generous bound
COLD_START_MAX_TOKENS = 16          # tiny: we time first token, not generation
RESULTS_DIR = Path(os.path.expanduser("~/cuda-graphs-experiment/results"))


@dataclass
class RunResult:
    model: str
    condition: str
    profile: str
    rep: int
    cold_start_s: float | None = None
    inferscope_json: dict | None = None       # single-request timing + resources
    bench_json: dict | None = None            # steady-state aggregate
    vllm_version: str | None = None           # recorded at run time, not assumed
    error: str | None = None


def wait_until_ready(timeout_s: int) -> bool:
    """Poll the health path until the server answers or timeout."""
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
    """Launch vllm serve in the background. Returns the process handle."""
    cmd = VLLM_SERVE_CMD + [model_id]
    if enforce_eager:
        cmd.append("--enforce-eager")
    # VERIFY_ON_NODE: any extra flags needed for the node (dtype, gpu-mem-util).
    log = open(RESULTS_DIR / "vllm_server.log", "ab")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)


def stop_vllm(proc: subprocess.Popen) -> None:
    """Terminate the server process group cleanly."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def measure_cold_start(proc: subprocess.Popen, served_name: str) -> tuple[float | None, dict | None]:
    """
    Time from server start to first served token, then capture a single-request
    inferscope probe (timing + /proc resources via --pid).
    Returns (cold_start_seconds, inferscope_json).
    """
    t0 = time.time()
    if not wait_until_ready(COLD_START_TIMEOUT_S):
        return None, None
    # First real request, profiled by inferscope, with resource sampling on the
    # vllm pid. Per-device GPU comes from inferscope's NVML side (ADR-007).
    cmd = [
        INFERSCOPE_BIN,
        "--endpoint", ENDPOINT,
        "--model", served_name,
        "--prompt", "Explain what a Kalman filter does in two sentences.",
        "--max-tokens", str(COLD_START_MAX_TOKENS),
        "--pid", str(proc.pid),
        "--json",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        cold_start = time.time() - t0   # includes warmup paid on first request
        js = json.loads(out.stdout) if out.stdout.strip() else None
        return cold_start, js
    except Exception:
        return time.time() - t0, None


def warmup(served_name: str) -> None:
    """Send throwaway requests so steady-state is measured warm. Excluded from data."""
    for _ in range(WARMUP_REQUESTS):
        try:
            subprocess.run(
                [INFERSCOPE_BIN, "--endpoint", ENDPOINT, "--model", served_name,
                 "--prompt", "warmup", "--max-tokens", "8"],
                capture_output=True, timeout=60,
            )
        except Exception:
            pass


def measure_steady_state(served_name: str, profile: dict) -> dict | None:
    """
    Run vllm bench serve at the given load profile, return parsed aggregate JSON.
    VERIFY_ON_NODE: exact flag names — this is the most likely thing to need
    adjustment against `vllm bench serve --help`.
    """
    out_path = RESULTS_DIR / "bench_tmp.json"
    cmd = BENCH_CMD + [
        "--model", served_name,
        "--base-url", ENDPOINT,
        "--num-prompts", str(profile["num_prompts"]),
        "--max-concurrency", str(profile["max_concurrency"]),
        "--save-result", "--result-filename", str(out_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=True)
        if out_path.exists():
            return json.loads(out_path.read_text())
    except Exception:
        return None
    return None


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
        for cond in CONDITIONS:
            for profile in LOAD_PROFILES:
                for rep in range(1, N_REPS + 1):
                    tag = f"{model['id'].split('/')[-1]}_{cond['name']}_{profile['name']}_rep{rep}"
                    out_file = RESULTS_DIR / f"{tag}.json"
                    if out_file.exists():
                        print(f"skip (exists): {tag}")
                        continue
                    print(f"RUN: {tag}")
                    res = RunResult(model=model["id"], condition=cond["name"],
                                    profile=profile["name"], rep=rep,
                                    vllm_version=vllm_version)
                    proc = None
                    try:
                        proc = start_vllm(model["id"], cond["enforce_eager"])
                        res.cold_start_s, res.inferscope_json = measure_cold_start(
                            proc, model["served_name"])
                        if res.cold_start_s is None:
                            res.error = "server did not become ready"
                        else:
                            warmup(model["served_name"])
                            res.bench_json = measure_steady_state(
                                model["served_name"], profile)
                            if res.bench_json is None:
                                res.error = "bench produced no result"
                    except Exception as e:
                        res.error = f"{type(e).__name__}: {e}"
                    finally:
                        if proc is not None:
                            stop_vllm(proc)
                        time.sleep(10)   # let GPU memory settle before next run
                    out_file.write_text(json.dumps(asdict(res), indent=2))
                    print(f"  saved: {out_file.name}  cold_start={res.cold_start_s}  err={res.error}")
    print("done. results in", RESULTS_DIR)


if __name__ == "__main__":
    main()
