"""Direct events.bin -> ffmpeg crisp 60fps MP4 timelapse renderer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pixel_battle.events import mmap_events
from pixel_battle.palette import PALETTE_RGB, WHITE_IDX


def _load_chronicle(run_dir: Path) -> list[dict]:
    path = run_dir / "chronicle.jsonl"
    if not path.exists():
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def render_timelapse(
    run_dir: Path,
    out_path: Path,
    fps: int = 60,
    duration_sec: float = 60.0,
    scale: int = 2,
    hold_end_sec: float = 3.0,
    show_hud: bool = True,
    crf: int = 18,
    curve: float = 1.35,
) -> Path:
    """Stream frames from events.bin directly into ffmpeg for crisp nearest-neighbor MP4 output.

    curve > 1.0 (e.g. 1.35-1.50) gives primordial chaos & early faction formation rich screen time
    before accelerating smoothly through multi-hour empire wars.
    """
    run_dir = Path(run_dir)
    events_path = run_dir / "events.bin"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing {events_path}")

    width, height, events = mmap_events(events_path)
    total = len(events)
    if total == 0:
        raise RuntimeError("events.bin contains 0 pixel placements.")

    chronicle = _load_chronicle(run_dir)
    chron_idx = 0
    latest_headline = "Primordial chaos begins..."

    total_frames = max(1, int(round(fps * duration_sec)))
    hold_frames = max(0, int(round(fps * hold_end_sec)))

    t_lin = np.linspace(0.0, 1.0, total_frames + 1)
    frame_cutoffs = np.round((t_lin ** max(0.5, curve)) * total).astype(np.int64)

    out_w = width * scale
    hud_h = 44 if show_hud else 0
    out_h = height * scale + hud_h
    # H.264 yuv420p requires even dimensions
    out_w += out_w % 2
    out_h += out_h % 2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "medium",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None

    grid = np.full((height, width), WHITE_IDX, dtype=np.uint8)
    frame_buf = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas_view = frame_buf[: height * scale, : width * scale]

    last_frame_bytes: bytes = b""
    try:
        for f_idx in range(total_frames):
            s = int(frame_cutoffs[f_idx])
            e = int(frame_cutoffs[f_idx + 1])
            if e > s:
                chunk = events[s:e]
                grid[chunk["y"], chunk["x"]] = chunk["color"]
                cur_tick = int(chunk["tick"][-1])
            else:
                cur_tick = int(events[max(0, s - 1)]["tick"])

            # Advance chronicle headline up to cur_tick
            while chron_idx < len(chronicle) and int(chronicle[chron_idx].get("tick", 0)) <= cur_tick:
                entry = chronicle[chron_idx]
                if entry.get("kind") != "system":
                    latest_headline = str(entry.get("summary", ""))[:110]
                chron_idx += 1

            rgb = PALETTE_RGB[grid]
            if scale == 1:
                canvas_view[:, :] = rgb
            else:
                # Ultra-fast exact nearest-neighbor repeat in NumPy
                canvas_view[:, :] = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

            if show_hud:
                hud_img = Image.new("RGB", (out_w, hud_h), (14, 18, 28))
                draw = ImageDraw.Draw(hud_img)
                draw.line([(0, 0), (out_w, 0)], fill=(45, 58, 84), width=1)
                stats_str = f"TICK {cur_tick:,}  |  {e:,} PIXELS"
                draw.text((14, 6), stats_str, fill=(56, 189, 248))
                draw.text((14, 24), latest_headline, fill=(226, 232, 240))
                frame_buf[height * scale : height * scale + hud_h, :out_w] = np.asarray(hud_img)

            last_frame_bytes = frame_buf.tobytes()
            proc.stdin.write(last_frame_bytes)

        # Hold final masterpiece frame
        for _ in range(hold_frames):
            proc.stdin.write(last_frame_bytes)

        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (code {ret}):\n{stderr[-1000:]}")
    finally:
        if proc.poll() is None:
            proc.kill()

    return out_path
