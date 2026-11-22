#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${1:-runs/reddit_2022_8h}"
HOURS="${2:-8.0}"
SEED="${3:-2022}"
LLM_MODEL="${LLM_MODEL:-qwen3-vl-30b-a3b-instruct-mlx}"

echo "============================================================"
echo "  place-sim — 1:1 Reddit r/place 2022 Run (${HOURS}h)"
echo "  Canvas: 1000x1000 (16c) -> 2000x1000 (24c) -> 2000x2000 (32c) -> Whiteout"
echo "  Run dir: ${RUN_DIR}"
echo "  Model:   ${LLM_MODEL} + FLUX.1-schnell Pixel-Art LoRA"
echo "  Started: $(date)"
echo "============================================================"

# 1. Ensure LM Studio has the target MLX VLM loaded with 4096 context to keep unified memory lean
LMS_BIN="${HOME}/.lmstudio/bin/lms"
if [ ! -x "${LMS_BIN}" ] && command -v lms >/dev/null 2>&1; then
  LMS_BIN="$(command -v lms)"
fi
if [ -x "${LMS_BIN}" ]; then
  if ! "${LMS_BIN}" ps 2>/dev/null | grep -q "${LLM_MODEL}"; then
    echo "Loading ${LLM_MODEL} in LM Studio (context=4096)..."
    "${LMS_BIN}" load "${LLM_MODEL}" -y --gpu max --context-length 4096 || true
  fi
fi

# 2. Run headless 1:1 r/place 2022 simulation under caffeinate -i
caffeinate -i uv run pixel-battle run \
  --preset reddit2022 \
  --headless \
  --policy neural \
  --tps 0 \
  --max-hours "${HOURS}" \
  --llm-model "${LLM_MODEL}" \
  --warmup-flux \
  --seed "${SEED}" \
  --run-dir "${RUN_DIR}"

echo ""
echo "============================================================"
echo "  Simulation finished at $(date)"
echo "  Rendering native 2K (2000x2000) 60fps MP4 timelapses..."
echo "============================================================"

# 3. Render 2K 60fps timelapse with 4-Act dynamic camera & chronicle HUD (120s + 4s end hold)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/timelapse_2k_hud.mp4" \
  --fps 60 \
  --duration 120 \
  --scale 1 \
  --hold 4 \
  --curve 1.35 \
  --crf 18

# 4. Render 2K 60fps clean edge-to-edge timelapse with dynamic 4-Act camera (90s + 4s end hold)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/timelapse_2k_clean.mp4" \
  --fps 60 \
  --duration 90 \
  --scale 1 \
  --hold 4 \
  --curve 1.35 \
  --crf 18 \
  --no-hud

# 5. Render snappy 20-second 2K highlight cut (great for social / GIF preview)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/timelapse_20s_clean.mp4" \
  --fps 60 \
  --duration 20 \
  --scale 1 \
  --hold 2 \
  --curve 1.35 \
  --crf 19 \
  --no-hud

# 6. Write concise run summary
uv run python - "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
from pixel_battle.events import read_events_header

rd = Path(sys.argv[1])
w, h, total = read_events_header(rd / "events.bin")
chron = [json.loads(l) for l in (rd / "chronicle.jsonl").read_text().splitlines() if l.strip()]
kinds = {}
for c in chron:
    kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1

state_path = rd / "checkpoint" / "state.json"
pop = {}
ai_pct = 0.0
if state_path.exists():
    st = json.loads(state_path.read_text())
    pop = st.get("population", {})
    facs = [f for f in st.get("factions", []) if not f.get("is_void")]
    if facs:
        ai_pct = 100.0 * sum(1 for f in facs if f.get("ai_origin")) / len(facs)

lines = [
    f"place-sim 1:1 Reddit r/place 2022 run summary ({rd})",
    f"Canvas: {w}x{h} | Total placements: {total:,}",
    f"Unique users: {pop.get('total_unique_users', 0):,} | One-shot: {pop.get('one_shot_pct', 0.0):.1f}%",
    f"AI-origin faction share: {ai_pct:.1f}%",
    f"Chronicle events: {len(chron)} | Breakdown: {kinds}",
    "Outputs:",
    f"  - {rd / 'final_peak_2k.png'} (peak colorful canvas before Whiteout)",
    f"  - {rd / 'final_2k.png'} (post-Whiteout final state)",
    f"  - {rd / 'timelapse_2k_hud.mp4'} ({(rd / 'timelapse_2k_hud.mp4').stat().st_size / 1e6:.1f} MB)",
    f"  - {rd / 'timelapse_2k_clean.mp4'} ({(rd / 'timelapse_2k_clean.mp4').stat().st_size / 1e6:.1f} MB)",
    f"  - {rd / 'timelapse_20s_clean.mp4'} ({(rd / 'timelapse_20s_clean.mp4').stat().st_size / 1e6:.1f} MB)",
]
summary = "\n".join(lines)
(rd / "summary.txt").write_text(summary + "\n", encoding="utf-8")
print("\n" + summary)
PY

echo ""
echo "Done at $(date)."
