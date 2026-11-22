#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${1:-runs/long_run}"
HOURS="${2:-8.0}"
SEED="${3:-2028}"

echo "============================================================"
echo "  pixel-battle — ${HOURS}h headless run"
echo "  Run dir: ${RUN_DIR}"
echo "  Started: $(date)"
echo "============================================================"

# 1. Run headless simulation under caffeinate -i to prevent macOS idle sleep
caffeinate -i uv run pixel-battle run \
  --headless \
  --policy neural \
  --tps 0 \
  --max-hours "${HOURS}" \
  --users 36000 \
  --max-factions 200 \
  --chaos-ticks 140 \
  --discovery-interval 45 \
  --epoch-interval 110 \
  --seed "${SEED}" \
  --run-dir "${RUN_DIR}"

echo ""
echo "============================================================"
echo "  Simulation finished at $(date)"
echo "  Rendering 2K 60fps MP4 timelapses..."
echo "============================================================"

# 2. Render 2K 60fps timelapse with chronicle HUD ticker (120s + 4s end hold)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/timelapse_2k_hud.mp4" \
  --fps 60 \
  --duration 120 \
  --scale 4 \
  --hold 4 \
  --curve 1.45 \
  --crf 18

# 3. Render 2K 60fps clean edge-to-edge timelapse (90s + 4s end hold, no HUD)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/timelapse_2k_clean.mp4" \
  --fps 60 \
  --duration 90 \
  --scale 4 \
  --hold 4 \
  --curve 1.45 \
  --crf 18 \
  --no-hud

# 4. Write a concise run summary
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

lines = [
    f"pixel-battle run summary ({rd})",
    f"Canvas: {w}x{h} | Total placements: {total:,}",
    f"Chronicle events: {len(chron)} | Breakdown: {kinds}",
    "Outputs:",
    f"  - {rd / 'timelapse_2k_hud.mp4'} ({(rd / 'timelapse_2k_hud.mp4').stat().st_size / 1e6:.1f} MB)",
    f"  - {rd / 'timelapse_2k_clean.mp4'} ({(rd / 'timelapse_2k_clean.mp4').stat().st_size / 1e6:.1f} MB)",
    f"  - {rd / 'final_2k.png'}",
]
summary = "\n".join(lines)
(rd / "summary.txt").write_text(summary + "\n", encoding="utf-8")
print("\n" + summary)
PY

echo ""
echo "Done at $(date)."
