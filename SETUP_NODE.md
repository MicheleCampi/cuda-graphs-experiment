# Node setup - resume notes (cold-prepared)

> **Historical note.** These are the working notes written *before* the matrix ran,
> kept as a record of how the node was prepared and what was still unknown at that
> point. Everything flagged here as open was resolved during the run: the 32B does
> fit on one H100 at bf16 with `--max-model-len 4096`, inferscope was rebuilt with
> `--features gpu-nvidia`, and `run_experiment.py` resolves and samples the EngineCore
> PID rather than the APIServer parent. Read `README.md` and `PROTOCOL.md` for the
> state of the experiment as published.


Lambda node: 1x H100 80GB PCIe, $3.29/hr. Lambda Stack 22.04.
Driver 580.105.08, CUDA 13.0. IP changes per instance — get fresh from dashboard.
Flow: enter node FROM the VM (ssh -i ~/.ssh/runpod_optimdev ubuntu@<IP>), rsync from VM.

## State at last stop
- vLLM 0.23.0 installed in ~/vllm-env (venv). Confirmed working, 7B serves on :8000.
- inferscope compiled BUT WITHOUT gpu-nvidia feature -> gpu_timeline: null.
- vllm bench serve flags confirmed for 0.23.0 (see run_experiment.py).
- Serve port confirmed: 8000.
- 32B fit on one H100: NOT yet tested.

## ROOT CAUSE of gpu_timeline: null (diagnosed cold)
inferscope was built with plain `cargo build --release` — no features.
`gpu-nvidia` is opt-in (default = []), so NVML sampler is NOT in the binary.
Past validations (L4/H100/A40) used the Docker image, which ships the feature.
RUNBOOK.md line 72/90 documents exactly this. Not a driver/NVML problem.

## FIX SEQUENCE (run on the node, GPU on)
# 1. build deps for gpu-nvidia (bindgen needs nvml header + clang) — RUNBOOK line 215
sudo apt-get install -y libnvidia-ml-dev clang
#    VERIFY: if libnvidia-ml-dev name not found, `apt search nvidia-ml` for the right pkg.
#    clang is almost certainly needed (bindgen).

# 2. rebuild inferscope WITH the feature
cd ~/inferscope && cargo build --release --features gpu-nvidia

# 3. with vLLM 7B already serving on :8000, find the EngineCore PID (not APIServer)
#    from vllm logs: the "(EngineCore pid=XXXX)" line — that's the GPU process.
#    then verify GPU is now captured:
~/inferscope/target/release/inferscope --endpoint http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct --prompt test --max-tokens 16 \
  --pid <ENGINECORE_PID> --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('gpu:', d.get('gpu'))"
#    Expect: gpu is NOT null (shows util/power). If still null -> then debug NVML runtime.

## PID note for the orchestrator
vLLM 0.23.0 runs TWO processes: APIServer (parent) + EngineCore (does GPU work).
run_experiment.py currently passes proc.pid (the parent/APIServer).
For correct GPU/resource sampling it must target the EngineCore PID.
FIX in script before full matrix: parse the EngineCore pid from vllm stdout/log,
or discover it (child of the serve process actually using the GPU).

## After GPU sampling works
- dry-run 7B end-to-end via run_experiment.py (one combo), validate JSON output.
- then full matrix (idempotent, resumes if interrupted).
- 32B: first launch tells if it fits on one H100 (~64GB weights / 80GB). If OOM,
  drop to 7B-only or a quantized 32B.
- rsync results back to VM, then SHUT DOWN the node (dashboard).

## Cost reminder
~$3.29/hr. Full run ~3-5h. SHUT DOWN when done — idle node still bills.
