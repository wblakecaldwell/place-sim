"""Direct events.bin -> ffmpeg crisp 60fps MP4 timelapse renderer with 4-Act dynamic camera pull-back."""

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


def _extract_viewport_schedule(
    chronicle: list[dict], full_w: int, full_h: int
) -> tuple[int, int, int, int, list[tuple[int, int, int, int, int]]]:
    """Return (init_w, init_h, init_act, init_pal, [(tick, w, h, act, pal), ...]) from chronicle."""
    init_w, init_h, init_act, init_pal = full_w, full_h, 3, 32
    schedule: list[tuple[int, int, int, int, int]] = []
    for entry in chronicle:
        kind = entry.get("kind", "")
        if kind == "system" and "active_w" in entry and "active_h" in entry:
            init_w = int(entry["active_w"])
            init_h = int(entry["active_h"])
            init_act = int(entry.get("act", 1 if init_w < full_w else 3))
            init_pal = int(entry.get("palette_size", 16 if init_act == 1 else 32))
        elif kind in ("expand_canvas", "whiteout") and "active_w" in entry and "active_h" in entry:
            schedule.append(
                (
                    int(entry.get("tick", 0)),
                    int(entry["active_w"]),
                    int(entry["active_h"]),
                    int(entry.get("act", 4 if kind == "whiteout" else 2)),
                    int(entry.get("palette_size", 1 if kind == "whiteout" else 32)),
                )
            )
    schedule.sort(key=lambda item: item[0])
    return init_w, init_h, init_act, init_pal, schedule


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
    dynamic_camera: bool = True,
) -> Path:
    """Stream frames from events.bin directly into ffmpeg for crisp nearest-neighbor MP4 output.

    When dynamic_camera=True and the run contains 4-Act canvas expansions (1000x1000 -> 2000x1000 ->
    2000x2000 -> Whiteout), the camera smoothly pulls back to frame each active viewport cleanly.
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

    init_w, init_h, cur_act, cur_pal, vp_schedule = _extract_viewport_schedule(chronicle, width, height)
    use_dyn_cam = dynamic_camera and (init_w < width or init_h < height or len(vp_schedule) > 0)
    vp_idx = 0
    tgt_w, tgt_h = float(init_w if use_dyn_cam else width), float(init_h if use_dyn_cam else height)
    cam_w, cam_h = tgt_w, tgt_h

    total_frames = max(1, int(round(fps * duration_sec)))
    hold_frames = max(0, int(round(fps * hold_end_sec)))

    t_lin = np.linspace(0.0, 1.0, total_frames + 1)
    frame_cutoffs = np.round((t_lin ** max(0.5, curve)) * total).astype(np.int64)

    canvas_out_w = width * scale
    canvas_out_h = height * scale
    out_w = canvas_out_w
    hud_h = 48 if show_hud else 0
    out_h = canvas_out_h + hud_h
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
    canvas_view = frame_buf[:canvas_out_h, :canvas_out_w]
    bg_rgb = np.array([12, 15, 22], dtype=np.uint8)

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
                    latest_headline = str(entry.get("summary", ""))[:125]
                chron_idx += 1

            # Advance dynamic camera viewport targets
            while vp_idx < len(vp_schedule) and vp_schedule[vp_idx][0] <= cur_tick:
                _, nw, nh, nact, npal = vp_schedule[vp_idx]
                tgt_w, tgt_h = float(nw), float(nh)
                cur_act, cur_pal = nact, npal
                vp_idx += 1

            if use_dyn_cam:
                # Smooth exponential pull-back across expansion milestones
                cam_w += (tgt_w - cam_w) * 0.14
                cam_h += (tgt_h - cam_h) * 0.14
                if abs(tgt_w - cam_w) < 2.0:
                    cam_w = tgt_w
                if abs(tgt_h - cam_h) < 2.0:
                    cam_h = tgt_h
                cw = max(16, min(width, int(round(cam_w))))
                ch = max(16, min(height, int(round(cam_h))))
                if cw == width and ch == height:
                    rgb = PALETTE_RGB[grid]
                    if scale == 1:
                        canvas_view[:, :] = rgb
                    else:
                        canvas_view[:, :] = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
                else:
                    sub_rgb = PALETTE_RGB[grid[:ch, :cw]]
                    fit_scale = min(canvas_out_w / float(cw), canvas_out_h / float(ch))
                    fit_w = max(2, min(canvas_out_w, int(round(cw * fit_scale))))
                    fit_h = max(2, min(canvas_out_h, int(round(ch * fit_scale))))
                    ox = (canvas_out_w - fit_w) // 2
                    oy = (canvas_out_h - fit_h) // 2
                    canvas_view[:, :] = bg_rgb
                    resized = np.asarray(
                        Image.fromarray(sub_rgb, mode="RGB").resize((fit_w, fit_h), Image.Resampling.NEAREST)
                    )
                    canvas_view[oy : oy + fit_h, ox : ox + fit_w] = resized
            else:
                rgb = PALETTE_RGB[grid]
                if scale == 1:
                    canvas_view[:, :] = rgb
                else:
                    canvas_view[:, :] = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)

            if show_hud:
                hud_img = Image.new("RGB", (out_w, hud_h), (14, 18, 28))
                draw = ImageDraw.Draw(hud_img)
                draw.line([(0, 0), (out_w, 0)], fill=(45, 58, 84), width=1)
                act_tag = f"ACT {cur_act} ({int(round(tgt_w))}x{int(round(tgt_h))} · {'WHITEOUT' if cur_pal == 1 else f'{cur_pal} COLORS'})"
                stats_str = f"{act_tag}   |   TICK {cur_tick:,}   |   {e:,} PIXELS"
                draw.text((14, 6), stats_str, fill=(56, 189, 248))
                draw.text((14, 26), latest_headline, fill=(226, 232, 240))
                frame_buf[canvas_out_h : canvas_out_h + hud_h, :out_w] = np.asarray(hud_img)

            last_frame_bytes = frame_buf.tobytes()
            proc.stdin.write(last_frame_bytes)

        # Hold final frame
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
