"""Decoupled 2-stage Art Bank Forger (Qwen3-VL solo -> MLX FLUX solo) & zero-repeat runtime ArtBank loader."""

from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pixel_battle.art.flux_art import FluxPixelArtGenerator
from pixel_battle.art.manager import ArtManager
from pixel_battle.brain import AIFactionIdea, LocalVLMBrain
from pixel_battle.palette import PALETTE_TIER_3_32, resolve_color
from pixel_battle.subreddits import SubredditSampler

log = logging.getLogger(__name__)


@dataclass
class ArtBankEntry:
    id: int
    name: str
    category: str
    art_mode: str              # "flux_art" | "flux_comic" | "banner"
    size_tier: str             # "micro" | "medium" | "macro"
    aspect: str                # "square" | "wide" | "tall"
    fluid_cutout: bool
    timezone_region: str
    language: str
    slogan: str
    art_prompt: str
    banner_text: str
    primary_color: int
    secondary_color: int
    accent_color: int
    w: int
    h: int
    blueprint_file: str        # relative path to .npy ("" for banner entries rendered at runtime)


def _dims_for_bank_entry(
    tier: str, art_mode: str, aspect: str, size_mult: float, rng: random.Random
) -> tuple[int, int]:
    """Compute crisp pixel-art dimensions for an Art Bank blueprint scaled by size_mult."""
    if tier == "micro":
        if art_mode == "banner":
            w, h = rng.choice([(36, 16), (42, 18)])
        elif aspect == "wide":
            w, h = rng.choice([(34, 24), (38, 26)])
        elif aspect == "tall":
            w, h = rng.choice([(24, 34), (26, 38)])
        else:
            d = rng.choice([24, 28, 32])
            w, h = d, d
    elif tier == "medium":
        if art_mode == "flux_comic":
            # Keep FLUX comics near-square so 2x2 512x512 comic pages fit intact without slicing panels in half
            w, h = rng.choice([(62, 54), (68, 58)])
        elif art_mode == "banner":
            w, h = rng.choice([(68, 26), (78, 30)])
        elif aspect == "wide":
            w, h = rng.choice([(62, 44), (70, 48)])
        elif aspect == "tall":
            w, h = rng.choice([(42, 58), (46, 64)])
        else:
            d = rng.choice([46, 52, 58])
            w, h = d, d
    else:  # macro
        if art_mode == "flux_comic":
            w, h = rng.choice([(76, 66), (84, 72)])
        elif art_mode == "banner":
            w, h = rng.choice([(96, 36), (112, 40)])
        elif aspect == "wide":
            w, h = rng.choice([(92, 62), (104, 68)])
        elif aspect == "tall":
            w, h = rng.choice([(64, 90), (72, 100)])
        else:
            d = rng.choice([72, 82, 90])
            w, h = d, d

    return max(18, int(round(w * size_mult))), max(16, int(round(h * size_mult)))


def _lms_bin() -> str | None:
    cand = Path.home() / ".lmstudio" / "bin" / "lms"
    if cand.exists():
        return str(cand)
    return None


def _lms_ensure_loaded(model: str, context: int = 4096) -> None:
    lms = _lms_bin()
    if not lms:
        return
    try:
        out = subprocess.run([lms, "ps"], capture_output=True, text=True, timeout=15)
        if model in out.stdout and str(context) in out.stdout:
            return
        if model in out.stdout:
            subprocess.run([lms, "unload", model], capture_output=True, timeout=30)
        log.info("Loading %s in LM Studio (context=%d)...", model, context)
        subprocess.run(
            [lms, "load", model, "-y", "--gpu", "max", "--context-length", str(context)],
            capture_output=True,
            timeout=120,
        )
    except Exception as exc:
        log.warning("Could not manage LM Studio model load (%s); continuing.", exc)


def _lms_unload(model: str) -> None:
    lms = _lms_bin()
    if not lms:
        return
    try:
        log.info("Unloading %s from LM Studio so MLX FLUX gets 100%% of Unified Memory...", model)
        subprocess.run([lms, "unload", model], capture_output=True, timeout=30)
        time.sleep(2.0)
    except Exception as exc:
        log.warning("Could not unload LM Studio model (%s).", exc)


def forge_art_bank(
    bank_dir: Path,
    total_concepts: int = 480,
    target_flux: int = 380,
    size_mult: float = 0.85,
    llm_url: str = "http://localhost:1234/v1",
    llm_model: str = "qwen3-vl-30b-a3b-instruct-mlx",
    keychain_service: str = "place-sim",
    flux_model_repo: str = "dhairyashil/FLUX.1-schnell-mflux-4bit",
    flux_lora_repo: str = "hliu143/dr-diffusion-flux-1-s-pixel-art-lora",
    flux_lora_file: str = "pixelart_schnell_v1.safetensors",
    flux_lora_scale: float = 0.95,
    flux_steps: int = 4,
    manage_lms: bool = True,
    seed: int = 2026,
) -> Path:
    """Decoupled 2-stage Art Bank creator: Stage 1 (Qwen3-VL solo) -> Unload -> Stage 2 (MLX FLUX solo)."""
    bank_dir = Path(bank_dir)
    bp_dir = bank_dir / "blueprints"
    bp_dir.mkdir(parents=True, exist_ok=True)

    concepts_path = bank_dir / "concepts.json"
    manifest_path = bank_dir / "manifest.json"
    rng = random.Random(seed)
    subreddits = SubredditSampler(seed=seed)

    # ------------------------------------------------------------------
    # Stage 1: Brainstorm unique community concepts with Qwen3-VL SOLO
    # ------------------------------------------------------------------
    concepts: list[dict[str, Any]] = []
    if concepts_path.exists():
        try:
            concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
        except Exception:
            concepts = []

    seen_names = {str(c.get("name", "")).strip().lower() for c in concepts if c.get("name")}
    if len(concepts) < total_concepts:
        if manage_lms:
            _lms_ensure_loaded(llm_model, context=4096)
        brain = LocalVLMBrain(ollama_url=llm_url, vlm_model=llm_model, keychain_service=keychain_service)
        if not brain.check_health():
            brain.close()
            raise RuntimeError(f"Local LLM not reachable at {llm_url} for Stage 1 concept brainstorming.")

        print(f"Stage 1/2: Brainstorming {total_concepts} unique communities with {llm_model} (have {len(concepts)})...", flush=True)
        t_s1 = time.perf_counter()
        tz_cycle = ("AMERICAS", "EUROPE", "ASIA_PACIFIC", "LATAM", "GLOBAL")
        stalls = 0
        while len(concepts) < total_concepts and stalls < 8:
            need = min(16, total_concepts - len(concepts) + 4)
            prime_tz = tz_cycle[len(concepts) % len(tz_cycle)]
            sub_hints = [subreddits.sample(size_tier="medium").name for _ in range(8)]
            existing_recent = [c["name"] for c in concepts[-60:]]
            batch: list[AIFactionIdea] = brain.brainstorm_faction_batch(
                count=need,
                act=3,
                virtual_hour=float((len(concepts) * 3) % 72),
                prime_tz=prime_tz,
                unlocked_colors=PALETTE_TIER_3_32,
                existing_names=existing_recent,
                subreddit_hints=sub_hints,
            )
            added = 0
            for idea in batch:
                nm = idea.name.strip()[:28]
                if not nm or nm.lower() in seen_names:
                    continue
                seen_names.add(nm.lower())
                # Convert ai_code into flux_art so we get maximum crisp visual FLUX variety (zero generic shields!)
                mode = idea.art_mode
                if mode == "ai_code":
                    mode = "flux_art"
                concepts.append(
                    {
                        "name": nm,
                        "category": idea.category or "art",
                        "art_mode": mode,
                        "aspect": idea.aspect or "square",
                        "fluid_cutout": False if mode == "flux_comic" else True,
                        "timezone_region": idea.timezone_region or prime_tz,
                        "language": idea.language or "en",
                        "slogan": (idea.slogan or f"{nm} unites!")[:90],
                        "art_prompt": idea.art_prompt[:180],
                        "banner_text": (idea.banner_text or nm[:12])[:14],
                        "primary_color": resolve_color(idea.primary_color, rng.randint(2, 30)),
                        "secondary_color": resolve_color(idea.secondary_color, rng.randint(2, 30)),
                        "accent_color": resolve_color(idea.accent_color, 6),
                    }
                )
                added += 1
            if added == 0:
                stalls += 1
            else:
                stalls = 0
                concepts_path.write_text(json.dumps(concepts, indent=2), encoding="utf-8")
                elapsed = time.perf_counter() - t_s1
                print(f"  [Stage 1] {len(concepts):3d}/{total_concepts} unique concepts ({elapsed:.0f}s)", flush=True)
        brain.close()

    # Unload LLM from LM Studio so MLX FLUX gets 100% of Unified Memory with zero swap thrashing!
    if manage_lms:
        _lms_unload(llm_model)

    # ------------------------------------------------------------------
    # Stage 2: Forge crisp pixel-art blueprints with MLX FLUX SOLO
    # ------------------------------------------------------------------
    entries: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []

    done_names = {str(e.get("name", "")).strip().lower() for e in entries}
    flux_done = sum(1 for e in entries if e.get("art_mode") in ("flux_art", "flux_comic") and e.get("blueprint_file"))

    print(
        f"Stage 2/2: Forging {target_flux} unique FLUX pixel artworks solo on MLX (have {flux_done})...",
        flush=True,
    )
    flux = FluxPixelArtGenerator(
        model_repo=flux_model_repo,
        lora_repo=flux_lora_repo,
        lora_file=flux_lora_file,
        lora_scale=flux_lora_scale,
        steps=flux_steps,
    )
    if not flux.ensure_loaded():
        raise RuntimeError("Failed to load local MLX FLUX.1-schnell Pixel-Art LoRA for Stage 2.")

    t_s2 = time.perf_counter()
    for idx, c in enumerate(concepts):
        nm = c["name"]
        if nm.lower() in done_names:
            continue

        # Balanced multi-scale distribution: ~42% micro, ~38% medium, ~20% macro
        r_tier = rng.random()
        if r_tier < 0.42:
            tier = "micro"
        elif r_tier < 0.80:
            tier = "medium"
        else:
            tier = "macro"

        mode = c["art_mode"]
        if mode == "flux_comic" and tier == "micro":
            tier = "medium"

        # Once target_flux is met, remaining concepts become unique multilingual banners
        if mode in ("flux_art", "flux_comic") and flux_done >= target_flux:
            mode = "banner"

        w, h = _dims_for_bank_entry(tier, mode, c["aspect"], size_mult, rng)
        bp_rel = ""

        if mode in ("flux_art", "flux_comic"):
            is_comic = mode == "flux_comic"
            prompt_str = (
                f"3-panel horizontal pixel art comic strip with black borders and cute expressive characters, {c['art_prompt']}"
                if is_comic
                else c["art_prompt"]
            )
            t0 = time.perf_counter()
            bp = flux.generate_pixel_art(
                prompt=prompt_str,
                target_w=w,
                target_h=h,
                strip_bg=(not is_comic) and bool(c.get("fluid_cutout", True)),
                add_outline=not is_comic,
                allowed_colors=PALETTE_TIER_3_32,
                seed=seed + idx * 131,
            )
            dt_img = time.perf_counter() - t0
            if bp is None or not ArtManager._is_valid_blueprint(bp, art_mode=mode):
                log.warning("Skipping invalid FLUX blueprint for '%s'", nm)
                continue
            eid = len(entries)
            bp_rel = f"blueprints/bp_{eid:04d}.npy"
            np.save(bank_dir / bp_rel, bp.astype(np.int16))
            flux_done += 1
            if flux_done % 10 == 0 or flux_done <= 5:
                avg_s = (time.perf_counter() - t_s2) / max(1, flux_done)
                rem_m = (max(0, target_flux - flux_done) * avg_s) / 60.0
                print(
                    f"  [Stage 2] FLUX {flux_done:3d}/{target_flux} | '{nm}' ({w}x{h}, {tier}/{mode}) "
                    f"in {dt_img:.1f}s | avg={avg_s:.1f}s | ETA={rem_m:.1f}m",
                    flush=True,
                )

        entry = ArtBankEntry(
            id=len(entries),
            name=nm,
            category=c["category"],
            art_mode=mode,
            size_tier=tier,
            aspect=c["aspect"],
            fluid_cutout=bool(c.get("fluid_cutout", True)),
            timezone_region=c["timezone_region"],
            language=c["language"],
            slogan=c["slogan"],
            art_prompt=c["art_prompt"],
            banner_text=c["banner_text"],
            primary_color=int(c["primary_color"]),
            secondary_color=int(c["secondary_color"]),
            accent_color=int(c["accent_color"]),
            w=w,
            h=h,
            blueprint_file=bp_rel,
        )
        entries.append(asdict(entry))
        done_names.add(nm.lower())
        if len(entries) % 10 == 0 or flux_done >= target_flux:
            manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(
        f"Art Bank ready -> {bank_dir} ({len(entries)} total entries: {flux_done} unique FLUX + "
        f"{len(entries) - flux_done} multilingual banners) in {(time.perf_counter() - t_s2)/60.0:.1f}m.",
        flush=True,
    )
    return bank_dir


class ArtBank:
    """Zero-repeat runtime loader for a pre-forged Art Bank directory (with optional AI-curated natural rival pairs)."""

    def __init__(self, bank_dir: Path, seed: int = 2026, visual_only: bool = False) -> None:
        self.bank_dir = Path(bank_dir)
        manifest_path = self.bank_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Art Bank manifest not found: {manifest_path}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = [ArtBankEntry(**item) for item in raw]
        if visual_only:
            # Strict AI-visual-only mode (REQ-ART-018): exclude Python-rendered bitmap wordmark banners!
            entries = [e for e in entries if bool(e.blueprint_file)]
        self.entries: list[ArtBankEntry] = entries
        self._by_name: dict[str, ArtBankEntry] = {e.name.lower(): e for e in self.entries}
        self.rival_map: dict[str, tuple[str, str]] = {}  # lower_name -> (partner_lower_name, reason)
        rivals_path = self.bank_dir / "rivals.json"
        if rivals_path.exists():
            try:
                pairs = json.loads(rivals_path.read_text(encoding="utf-8"))
                for p in pairs:
                    a = str(p.get("a", "")).strip().lower()
                    b = str(p.get("b", "")).strip().lower()
                    why = str(p.get("reason", "natural rivals")).strip()[:40]
                    if a in self._by_name and b in self._by_name and a != b:
                        self.rival_map[a] = (b, why)
                        self.rival_map[b] = (a, why)
            except Exception as exc:
                log.warning("Could not load rivals.json (%s)", exc)

        rng = random.Random(seed)
        # Interleave rival-paired entries early so visible rivalries ignite right from Day-0!
        rival_entries = [e for e in self.entries if e.name.lower() in self.rival_map]
        other_entries = [e for e in self.entries if e.name.lower() not in self.rival_map]
        rng.shuffle(rival_entries)
        rng.shuffle(other_entries)
        self.entries = rival_entries + other_entries
        self._used_ids: set[int] = set()
        log.info(
            "Loaded ArtBank from %s (%d unique entries, %d rival-paired)",
            self.bank_dir,
            len(self.entries),
            len(self.rival_map),
        )

    def remaining(self) -> int:
        return len(self.entries) - len(self._used_ids)

    def _load_entry_bp(self, entry: ArtBankEntry) -> np.ndarray | None:
        if entry.blueprint_file:
            bp_path = self.bank_dir / entry.blueprint_file
            if bp_path.exists():
                return np.load(bp_path).astype(np.int16)
        return None

    def pop_by_name(self, name_lower: str) -> tuple[ArtBankEntry, np.ndarray | None] | None:
        """Pop a specific unused ArtBankEntry by lowercase name (used to spawn a rival right beside its partner)."""
        entry = self._by_name.get(name_lower.lower())
        if entry is None or entry.id in self._used_ids:
            return None
        self._used_ids.add(entry.id)
        return entry, self._load_entry_bp(entry)

    def get_unused_rival(self, name: str) -> tuple[str, str] | None:
        """Return (partner_name_lower, rivalry_reason) if this entry has an unspawned natural rival."""
        info = self.rival_map.get(name.strip().lower())
        if info is None:
            return None
        partner_lower, why = info
        p_ent = self._by_name.get(partner_lower)
        if p_ent is None or p_ent.id in self._used_ids:
            return None
        return partner_lower, why

    def pop_next(
        self,
        preferred_tier: str | None = None,
        existing_names: set[str] | None = None,
    ) -> tuple[ArtBankEntry, np.ndarray | None] | None:
        """Pop the next unused unique ArtBankEntry (preferring requested size tier)."""
        avoid = existing_names or set()
        for strict in (True, False):
            for entry in self.entries:
                if entry.id in self._used_ids:
                    continue
                if entry.name.lower() in avoid:
                    self._used_ids.add(entry.id)
                    continue
                if strict and preferred_tier and entry.size_tier != preferred_tier:
                    continue
                self._used_ids.add(entry.id)
                return entry, self._load_entry_bp(entry)
        return None

    def pop_best_fit(
        self,
        target_w: int,
        target_h: int,
        existing_names: set[str] | None = None,
        max_oversize: float = 1.45,
    ) -> tuple[ArtBankEntry, np.ndarray] | None:
        """Pop the unused AI-generated visual blueprint whose dimensions best fit an unclaimed whitespace niche."""
        avoid = existing_names or set()
        best_entry: ArtBankEntry | None = None
        best_score = 1e18
        for entry in self.entries:
            if entry.id in self._used_ids or not entry.blueprint_file:
                continue
            if entry.name.lower() in avoid:
                self._used_ids.add(entry.id)
                continue
            if entry.w > int(round(target_w * max_oversize)) or entry.h > int(round(target_h * max_oversize)):
                continue
            # Prefer artworks that fill the niche snugly with matching aspect ratio
            size_err = abs(entry.w - target_w) + abs(entry.h - target_h)
            over_pen = max(0, entry.w - target_w) * 1.5 + max(0, entry.h - target_h) * 1.5
            score = size_err + over_pen
            if score < best_score:
                best_score = score
                best_entry = entry
        if best_entry is None:
            return None
        bp = self._load_entry_bp(best_entry)
        if bp is None:
            self._used_ids.add(best_entry.id)
            return None
        self._used_ids.add(best_entry.id)
        return best_entry, bp


def strip_white_sticker_bubble(bp: np.ndarray, allow_keep: bool = False) -> np.ndarray:
    """Strip lumpy outer black-bordered white sticker ovals while keeping interior white & tight silhouettes."""
    from collections import deque
    from pixel_battle.art.diffusion import _add_1px_silhouette_outline, _remove_isolated_specks
    from pixel_battle.palette import BLACK_IDX, WHITE_IDX

    if allow_keep:
        return bp
    h, w = bp.shape
    valid = bp >= 0
    n_valid = int(valid.sum())
    if n_valid < 40:
        return bp
    # Only target blueprints with a significant pure-white field (>= 34% of valid pixels)
    white_mask = bp == WHITE_IDX
    if float(white_mask.sum()) / max(1, n_valid) < 0.34:
        return bp

    # Detect outer 1px black outline sitting adjacent to transparency (-1) or image edge
    ext = ~valid
    adj_ext = np.zeros((h, w), dtype=bool)
    adj_ext[0, :] = True
    adj_ext[-1, :] = True
    adj_ext[:, 0] = True
    adj_ext[:, -1] = True
    if h > 1:
        adj_ext[1:, :] |= ext[:-1, :]
        adj_ext[:-1, :] |= ext[1:, :]
    if w > 1:
        adj_ext[:, 1:] |= ext[:, :-1]
        adj_ext[:, :-1] |= ext[:, 1:]

    # Require a black outer sticker ring touching exterior before peeling
    out = bp.copy()
    outer_black = (out == BLACK_IDX) & adj_ext
    if int(outer_black.sum()) < 10:
        return bp
    out[outer_black] = -1

    # Flood-fill ONLY pure WHITE_IDX (preserving light-gray/gray shading on clouds, castles, bowls, etc.)
    light_bg = out == WHITE_IDX
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for y in range(h):
        for x in range(w):
            if not light_bg[y, x]:
                continue
            touches = (
                y == 0 or y == h - 1 or x == 0 or x == w - 1
                or out[y - 1, x] < 0 or out[y + 1, x] < 0
                or out[y, x - 1] < 0 or out[y, x + 1] < 0
            )
            if touches:
                visited[y, x] = True
                q.append((y, x))

    while q:
        cy, cx = q.popleft()
        out[cy, cx] = -1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and light_bg[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    out = _remove_isolated_specks(out)
    surviving = int((out >= 0).sum())
    if surviving < 28:
        return bp

    # Crop tightly around the surviving artwork (+1px padding for crisp 1px black outline)
    ys, xs = np.where(out >= 0)
    y0 = max(0, int(ys.min()) - 1)
    y1 = min(h, int(ys.max()) + 2)
    x0 = max(0, int(xs.min()) - 1)
    x1 = min(w, int(xs.max()) + 2)
    cropped = out[y0:y1, x0:x1].copy()
    cropped = _add_1px_silhouette_outline(cropped, BLACK_IDX)
    return cropped


def clean_art_bank_bubbles(bank_dir: Path, keep_fraction: float = 0.02, seed: int = 2026) -> int:
    """Clean white sticker-bubble halos across an existing Art Bank in-place (keeping a tiny optional fraction)."""
    bank_dir = Path(bank_dir)
    manifest_path = bank_dir / "manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    cleaned = 0
    for e in entries:
        rel = e.get("blueprint_file")
        if not rel or e.get("art_mode") != "flux_art":
            continue
        p = bank_dir / rel
        if not p.exists():
            continue
        bp = np.load(p).astype(np.int16)
        keep_this = rng.random() < keep_fraction
        new_bp = strip_white_sticker_bubble(bp, allow_keep=keep_this)
        if new_bp.shape != bp.shape or not np.array_equal(new_bp, bp):
            np.save(p, new_bp.astype(np.int16))
            e["h"], e["w"] = int(new_bp.shape[0]), int(new_bp.shape[1])
            cleaned += 1
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return cleaned


def curate_rival_pairs(
    bank_dir: Path,
    llm_url: str = "http://localhost:1234/v1",
    llm_model: str = "qwen3-vl-30b-a3b-instruct-mlx",
    keychain_service: str = "place-sim",
    target_pairs: int = 40,
    manage_lms: bool = True,
) -> list[dict[str, str]]:
    """Ask local Qwen3-VL to pair up visually/culturally natural rival communities from an existing Art Bank."""
    from pydantic import BaseModel, Field

    class RivalPairItem(BaseModel):
        a: str = Field(..., min_length=1, max_length=32)
        b: str = Field(..., min_length=1, max_length=32)
        reason: str = Field(..., min_length=3, max_length=48)

    class RivalPairBatch(BaseModel):
        pairs: list[RivalPairItem] = Field(..., min_length=4, max_length=55)

    bank_dir = Path(bank_dir)
    entries = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    flux_entries = [e for e in entries if e.get("blueprint_file")]
    if len(flux_entries) < 10:
        return []

    if manage_lms:
        _lms_ensure_loaded(llm_model, context=4096)

    brain = LocalVLMBrain(ollama_url=llm_url, vlm_model=llm_model, keychain_service=keychain_service)
    if not brain.check_health():
        brain.close()
        return []

    # Send concise catalog chunks so it easily fits inside 4096 context
    valid_names = {e["name"].lower(): e["name"] for e in flux_entries}
    used: set[str] = set()
    all_pairs: list[dict[str, str]] = []
    chunk_size = 95
    for start in range(0, len(flux_entries), chunk_size):
        if len(all_pairs) >= target_pairs:
            break
        chunk = flux_entries[start : start + chunk_size]
        lines = [
            f"- {e['name']} ({e['category']}): {e['art_prompt'][:55]}"
            for e in chunk
        ]
        sys_p = (
            "You match playful, visually recognizable rival pairs for an r/place pixel-art battle. "
            "Examples of great rivalries viewers immediately get: Coffee vs Tea/Soda, Taco vs Ramen/Sushi, "
            "Cats vs Dogs, Retro vs Cyberpunk, Fire vs Water/Ice, Wizards vs Robots, Ninjas vs Knights, "
            "Sun vs Moon, Street Food vs Fine Dining, Cute vs Spooky. "
            "Return ONLY valid JSON matching the schema. Use EXACT faction names from the list, each name at most once."
        )
        usr_p = (
            f"Pick 14 natural rival pairs from these communities (exact names, no duplicates):\n"
            + "\n".join(lines)
        )
        res = brain._chat_json(sys_p, usr_p, image=None, schema_cls=RivalPairBatch, temperature=0.7, max_tokens=1600)
        if res is not None:
            for item in res.pairs:
                al = item.a.strip().lower()
                bl = item.b.strip().lower()
                if al in valid_names and bl in valid_names and al != bl and al not in used and bl not in used:
                    used.add(al)
                    used.add(bl)
                    all_pairs.append({"a": valid_names[al], "b": valid_names[bl], "reason": item.reason.strip()})
                    if len(all_pairs) >= target_pairs:
                        break
    brain.close()
    if manage_lms:
        _lms_unload(llm_model)

    (bank_dir / "rivals.json").write_text(json.dumps(all_pairs, indent=2), encoding="utf-8")
    return all_pairs
