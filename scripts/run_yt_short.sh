#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${1:-runs/yt_short_360x640_niche}"
TICKS="${2:-9000}"
SEED="${3:-2027}"
BANK_DIR="${4:-${RUN_DIR}/art_bank}"
CONCEPTS="${CONCEPTS:-480}"
FLUX_COUNT="${FLUX_COUNT:-380}"
LLM_MODEL="${LLM_MODEL:-qwen3-vl-30b-a3b-instruct-mlx}"

echo "============================================================"
echo "  place-sim — Deterministic YouTube Short Pipeline (Rev 2.3)"
echo "  Canvas:   360x640 fixed (exact 9:16 twin of 512x512, no expansion, no whiteout)"
echo "  Exports:  1080p (3x -> 1080x1920) & 4K (6x -> 2160x3840) @ 60fps"
echo "  Scale:    ${TICKS} step-locked ticks | Landmarks First (95) -> Niche-Carvers (285 active)"
echo "  Rivals:   AI-curated natural rival pairs + focused border wars (<= 16 concurrent)"
echo "  Chaos:    White-claiming free agents + gentle burnout turnover (zero white gaps)"
echo "  Art Bank: ${BANK_DIR}"
echo "  Run dir:  ${RUN_DIR}"
echo "  Started:  $(date)"
echo "============================================================"

mkdir -p "${RUN_DIR}"

# Stage 1 & 2: Decoupled Qwen3-VL solo -> Unload -> MLX FLUX solo (zero Metal swap thrashing)
if [[ ! -f "${BANK_DIR}/manifest.json" ]]; then
  caffeinate -i uv run pixel-battle forge-bank \
    --out-dir "${BANK_DIR}" \
    --concepts "${CONCEPTS}" \
    --flux-count "${FLUX_COUNT}" \
    --size-mult 0.85 \
    --llm-model "${LLM_MODEL}" \
    --seed "${SEED}"
fi

# Clean any white sticker bubbles (idempotent via sentinel) & ensure AI-curated natural rival pairs exist
uv run python - "${BANK_DIR}" "${LLM_MODEL}" <<'PY'
import sys
from pathlib import Path
from pixel_battle.art.bank import clean_art_bank_bubbles, curate_rival_pairs

bank_dir = Path(sys.argv[1])
llm_model = sys.argv[2]
sentinel = bank_dir / ".bubbles_cleaned"
if not sentinel.exists():
    cleaned = clean_art_bank_bubbles(bank_dir, keep_fraction=0.02)
    sentinel.write_text(f"cleaned={cleaned}\n", encoding="utf-8")
    print(f"Art Bank bubble check: cleaned {cleaned} white-oval cutouts.", flush=True)
if not (bank_dir / "rivals.json").exists():
    pairs = curate_rival_pairs(bank_dir, llm_model=llm_model, target_pairs=42, manage_lms=True)
    print(f"Curated {len(pairs)} natural rival pairs -> {bank_dir / 'rivals.json'}", flush=True)
PY

echo ""
echo "============================================================"
echo "  Art Bank ready at $(date)"
echo "  Stage 3: Running step-locked MPS pixel battle (${TICKS} ticks)..."
echo "============================================================"

# Stage 3: Pure step-locked deterministic MPS battle pulling from pre-forged Art Bank
caffeinate -i uv run pixel-battle run \
  --preset ytshort \
  --sync \
  --headless \
  --policy neural \
  --tps 0 \
  --max-ticks "${TICKS}" \
  --art-bank "${BANK_DIR}" \
  --no-flux-art \
  --llm-url "http://127.0.0.1:9/v1" \
  --seed "${SEED}" \
  --run-dir "${RUN_DIR}"

echo ""
echo "============================================================"
echo "  Simulation finished at $(date)"
echo "  Stage 4: Rendering 4K (2160x3840) 60fps YouTube Short MP4..."
echo "============================================================"

# Primary YouTube Short: 4K Vertical 2160x3840 @ 60fps (55s timelapse + 3s final hold = 58s)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/short_4k_60fps_58s.mp4" \
  --fps 60 \
  --duration 55 \
  --scale 6 \
  --hold 3 \
  --curve 1.35 \
  --crf 17 \
  --no-hud \
  --no-dynamic-camera

# Lightweight 1080p Vertical 1080x1920 @ 60fps export (exact 3x nearest-neighbor)
caffeinate -i uv run pixel-battle render "${RUN_DIR}" \
  --out "${RUN_DIR}/short_1080p_60fps_58s.mp4" \
  --fps 60 \
  --duration 55 \
  --scale 3 \
  --hold 3 \
  --curve 1.35 \
  --crf 17 \
  --no-hud \
  --no-dynamic-camera

# Write concise run summary
uv run python - "${RUN_DIR}" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
from pixel_battle.events import read_events_header

rd = Path(sys.argv[1])
w, h, total = read_events_header(rd / "events.bin")
chron = [json.loads(l) for l in (rd / "chronicle.jsonl").read_text().splitlines() if l.strip()]
kinds = Counter(c["kind"] for c in chron)
founded = [c for c in chron if c.get("kind") == "founded"]
modes = Counter(c.get("art_mode") for c in founded)
scale = 2160 // max(1, w)

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
    f"place-sim Deterministic 4K YouTube Short summary ({rd})",
    f"Canvas: {w}x{h} fixed ({scale}x -> 4K {w*scale}x{h*scale}) | Total placements: {total:,}",
    f"Unique users: {pop.get('total_unique_users', 0):,} | One-shot: {pop.get('one_shot_pct', 0.0):.1f}%",
    f"Total founded: {len(founded)} | AI-origin share: {ai_pct:.1f}%",
    f"Art modes: {dict(modes)}",
    f"Chronicle events: {len(chron)} | Breakdown: {dict(kinds)}",
    "Outputs:",
    f"  - {rd / 'final_4k.png'} (4K {w*scale}x{h*scale} final canvas)",
    f"  - {rd / 'short_4k_60fps_58s.mp4'} ({(rd / 'short_4k_60fps_58s.mp4').stat().st_size / 1e6:.1f} MB)",
    f"  - {rd / 'short_1080p_60fps_58s.mp4'} ({(rd / 'short_1080p_60fps_58s.mp4').stat().st_size / 1e6:.1f} MB)",
]
summary = "\n".join(lines)
(rd / "summary.txt").write_text(summary + "\n", encoding="utf-8")
print("\n" + summary)
PY

echo ""
echo "Done at $(date)."
