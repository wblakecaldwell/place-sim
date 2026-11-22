"""Compact binary pixel event logging (v2: 16 bytes/pixel with uint32 user_id & uint16 faction; v1 read-compatible) and JSONL chronicle."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

# Binary header: magic(4s) + version(H) + width(H) + height(H) + reserved(H) = 12 bytes
HEADER_MAGIC = b"PXBT"
HEADER_STRUCT = struct.Struct("<4sHHHH")
HEADER_VERSION_V1 = 1
HEADER_VERSION = 2

# Version 1 record (legacy read/resume compat): 12 bytes
EVENT_DTYPE_V1 = np.dtype(
    [
        ("tick", "<u4"),
        ("x", "<u2"),
        ("y", "<u2"),
        ("color", "u1"),
        ("faction", "u1"),
        ("user_id", "<u2"),
    ]
)
assert EVENT_DTYPE_V1.itemsize == 12

# Version 2 record: tick(u4), x(u2), y(u2), color(u1), reserved(u1), faction(u2), user_id(u4) = 16 bytes
# faction = 65535 denotes free agent (-1); supports 65,534 lifetime factions and 4.29B unique users
EVENT_DTYPE_V2 = np.dtype(
    [
        ("tick", "<u4"),
        ("x", "<u2"),
        ("y", "<u2"),
        ("color", "u1"),
        ("reserved", "u1"),
        ("faction", "<u2"),
        ("user_id", "<u4"),
    ]
)
assert EVENT_DTYPE_V2.itemsize == 16

EVENT_DTYPE = EVENT_DTYPE_V2


def _dtype_for_version(version: int) -> np.dtype:
    if version == HEADER_VERSION_V1:
        return EVENT_DTYPE_V1
    if version == HEADER_VERSION:
        return EVENT_DTYPE_V2
    raise ValueError(f"Unsupported events version: {version}")


class EventLogger:
    """Append-only binary pixel event writer (v2 default, v1 resume-compatible) + JSONL chronicle."""

    def __init__(self, run_dir: Path, width: int, height: int, resume: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.bin"
        self.chronicle_path = self.run_dir / "chronicle.jsonl"
        self.width = width
        self.height = height
        self.total_events = 0
        self.version = HEADER_VERSION
        self._dtype = EVENT_DTYPE_V2

        if resume and self.events_path.exists():
            with open(self.events_path, "rb") as fh:
                raw = fh.read(HEADER_STRUCT.size)
            magic, ver, _, _, _ = HEADER_STRUCT.unpack(raw)
            if magic == HEADER_MAGIC and ver in (HEADER_VERSION_V1, HEADER_VERSION):
                self.version = int(ver)
                self._dtype = _dtype_for_version(self.version)
            size = self.events_path.stat().st_size
            self.total_events = max(0, (size - HEADER_STRUCT.size) // self._dtype.itemsize)
            self._bin_fh = open(self.events_path, "ab", buffering=1024 * 256)
        else:
            self._bin_fh = open(self.events_path, "wb", buffering=1024 * 256)
            self._bin_fh.write(HEADER_STRUCT.pack(HEADER_MAGIC, self.version, width, height, 0))
            self._bin_fh.flush()

        mode = "a" if resume else "w"
        self._chron_fh = open(self.chronicle_path, mode, encoding="utf-8", buffering=1)
        self.recent_chronicle: list[dict[str, Any]] = []
        if resume and self.chronicle_path.exists():
            try:
                with open(self.chronicle_path, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()[-120:]
                self.recent_chronicle = [json.loads(line) for line in lines if line.strip()]
            except Exception:
                self.recent_chronicle = []

    def log_pixels(
        self,
        tick: int,
        xs: np.ndarray,
        ys: np.ndarray,
        colors: np.ndarray,
        factions: np.ndarray,
        user_ids: np.ndarray,
    ) -> None:
        n = len(xs)
        if n == 0:
            return
        buf = np.empty(n, dtype=self._dtype)
        buf["tick"] = tick
        buf["x"] = xs.astype(np.uint16, copy=False)
        buf["y"] = ys.astype(np.uint16, copy=False)
        buf["color"] = colors.astype(np.uint8, copy=False)
        if self.version == HEADER_VERSION_V1:
            f_i32 = factions.astype(np.int32, copy=False)
            f_u8 = np.where(f_i32 < 0, 255, np.clip(f_i32, 0, 254)).astype(np.uint8)
            buf["faction"] = f_u8
            buf["user_id"] = (user_ids & 0xFFFF).astype(np.uint16, copy=False)
        else:
            buf["reserved"] = 0
            f_i32 = factions.astype(np.int32, copy=False)
            f_u16 = np.where(f_i32 < 0, 65535, np.clip(f_i32, 0, 65534)).astype(np.uint16)
            buf["faction"] = f_u16
            buf["user_id"] = (user_ids & 0xFFFFFFFF).astype(np.uint32, copy=False)
        self._bin_fh.write(buf.tobytes())
        self.total_events += n

    def log_chronicle(self, tick: int, kind: str, summary: str, **details: Any) -> dict[str, Any]:
        entry = {
            "ts": round(time.time(), 2),
            "tick": int(tick),
            "kind": kind,
            "summary": summary,
            **details,
        }
        self._chron_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.recent_chronicle.append(entry)
        if len(self.recent_chronicle) > 200:
            self.recent_chronicle = self.recent_chronicle[-200:]
        return entry

    def flush(self) -> None:
        self._bin_fh.flush()
        self._chron_fh.flush()

    def close(self) -> None:
        try:
            self.flush()
            self._bin_fh.close()
            self._chron_fh.close()
        except Exception:
            pass


def read_events_header_full(events_path: Path) -> tuple[int, int, int, int]:
    """Return (version, width, height, total_event_count) from events.bin."""
    with open(events_path, "rb") as fh:
        raw = fh.read(HEADER_STRUCT.size)
    magic, version, width, height, _ = HEADER_STRUCT.unpack(raw)
    if magic != HEADER_MAGIC:
        raise ValueError(f"Invalid events file magic: {magic!r}")
    dtype = _dtype_for_version(version)
    file_size = events_path.stat().st_size
    count = max(0, (file_size - HEADER_STRUCT.size) // dtype.itemsize)
    return int(version), int(width), int(height), int(count)


def read_events_header(events_path: Path) -> tuple[int, int, int]:
    """Return (width, height, total_event_count) from events.bin."""
    _, width, height, count = read_events_header_full(events_path)
    return width, height, count


def mmap_events(events_path: Path) -> tuple[int, int, np.memmap]:
    """Memory-map events.bin (v1 or v2) for fast zero-copy timelapse rendering."""
    version, width, height, count = read_events_header_full(events_path)
    dtype = _dtype_for_version(version)
    arr = np.memmap(
        events_path,
        dtype=dtype,
        mode="r",
        offset=HEADER_STRUCT.size,
        shape=(count,),
    )
    return width, height, arr
