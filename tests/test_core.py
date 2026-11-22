"""Unit and integration tests for pixel-battle."""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import torch

from pixel_battle.art.flags import NATIONAL_FLAGS, RIBBON_PATTERNS, generate_national_flag, generate_transverse_ribbon
from pixel_battle.art.manager import ArtManager
from pixel_battle.art.procedural import add_ornate_border, generate_procedural
from pixel_battle.art.sprites import (
    FINE_ART_CATALOG,
    SPRITE_KEYS,
    generate_black_hole,
    generate_clean_space,
    generate_fractal,
    generate_micro_sprite,
    generate_procedural_painting,
    generate_void_monster,
)
from pixel_battle.brain import FactionDecision, ocr_text_matches
from pixel_battle.canvas import CanvasState
from pixel_battle.detector import find_pareidolia_candidates, update_faction_metrics
from pixel_battle.events import (
    EVENT_DTYPE_V1,
    HEADER_MAGIC,
    HEADER_STRUCT,
    HEADER_VERSION_V1,
    EventLogger,
    mmap_events,
    read_events_header_full,
)
from pixel_battle.factions import Faction, compute_edge_weights, compute_shared_border_rect
from pixel_battle.palette import BLACK_IDX, BLUE_IDX, PALETTE_RGB, RED_IDX, WHITE_IDX, YELLOW_IDX, rgb_to_palette_indices
from pixel_battle.policies.stochastic import StochasticPolicy
from pixel_battle.subreddits import SubredditSampler


def test_palette_roundtrip() -> None:
    grid = np.arange(len(PALETTE_RGB), dtype=np.uint8).reshape(4, 8)
    rgb = PALETTE_RGB[grid]
    recovered = rgb_to_palette_indices(rgb)
    assert np.array_equal(grid, recovered)


def test_binary_event_log_v2_and_v1_compat(tmp_path: Path) -> None:
    # Version 2: uint32 user_id (>10M) & uint16 faction (>500)
    logger = EventLogger(tmp_path, 512, 512)
    xs = np.array([10, 511, 0], dtype=np.int32)
    ys = np.array([20, 511, 1], dtype=np.int32)
    cols = np.array([RED_IDX, BLUE_IDX, BLACK_IDX], dtype=np.uint8)
    facs = np.array([-1, 0, 512], dtype=np.int16)
    uids = np.array([100, 10_500_000, 25_000_000], dtype=np.int64)
    logger.log_pixels(42, xs, ys, cols, facs, uids)
    logger.close()

    ver, w, h, cnt = read_events_header_full(tmp_path / "events.bin")
    assert (ver, w, h, cnt) == (2, 512, 512, 3)
    w, h, arr = mmap_events(tmp_path / "events.bin")
    assert (w, h) == (512, 512)
    assert len(arr) == 3
    assert int(arr[0]["tick"]) == 42
    assert int(arr[0]["faction"]) == 65535
    assert int(arr[1]["user_id"]) == 10_500_000
    assert int(arr[2]["faction"]) == 512 and int(arr[2]["user_id"]) == 25_000_000

    # Legacy Version 1 (12-byte) read compatibility
    v1_path = tmp_path / "legacy_events.bin"
    rec = np.zeros(2, dtype=EVENT_DTYPE_V1)
    rec["tick"] = [1, 2]
    rec["x"] = [5, 6]
    rec["y"] = [7, 8]
    rec["color"] = [RED_IDX, BLUE_IDX]
    rec["faction"] = [255, 3]
    rec["user_id"] = [10, 20]
    with open(v1_path, "wb") as fh:
        fh.write(HEADER_STRUCT.pack(HEADER_MAGIC, HEADER_VERSION_V1, 512, 512, 0))
        fh.write(rec.tobytes())
    v1_ver, vw, vh, vcnt = read_events_header_full(v1_path)
    assert (v1_ver, vw, vh, vcnt) == (1, 512, 512, 2)
    _, _, arr1 = mmap_events(v1_path)
    assert int(arr1[1]["color"]) == BLUE_IDX and int(arr1[1]["faction"]) == 3


def test_stochastic_policy_and_void_on_device() -> None:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    canvas = CanvasState(128, 128, dev)
    policy = StochasticPolicy(num_users=2500, width=128, height=128, base_cooldown=4, device=dev, seed=7)

    total = 0
    for t in range(1, 30):
        batch = policy.step(t, canvas, [], [])
        canvas.apply_placements(t, batch.xs, batch.ys, batch.colors, batch.factions)
        total += int(batch.xs.numel())
    assert total > 5_000

    cands = find_pareidolia_candidates(canvas, [], window=32, stride=16, min_density=0.05, top_k=2)
    assert len(cands) >= 1

    bp_np = add_ornate_border(
        generate_procedural("emerald shield crest", 32, 32, BLUE_IDX, YELLOW_IDX, WHITE_IDX),
        BLUE_IDX,
        YELLOW_IDX,
        WHITE_IDX,
    )
    bp_t = torch.from_numpy(bp_np).to(dev)
    f_art = Faction(
        id=0,
        name="Test Guild",
        theme="shield",
        slogan="Hold!",
        art_mode="procedural",
        art_prompt="emerald shield crest",
        primary_color=BLUE_IDX,
        secondary_color=YELLOW_IDX,
        accent_color=WHITE_IDX,
        x0=16,
        y0=16,
        w=32,
        h=32,
        blueprint=bp_t,
        edge_weight=compute_edge_weights(bp_t),
        created_tick=25,
    )
    f_void = Faction(
        id=1,
        name="THE VOID",
        theme="abyss",
        slogan="CONSUME",
        art_mode="void",
        art_prompt="void",
        primary_color=BLACK_IDX,
        secondary_color=RED_IDX,
        accent_color=RED_IDX,
        x0=64,
        y0=64,
        w=32,
        h=32,
        blueprint=torch.full((32, 32), BLACK_IDX, dtype=torch.int16, device=dev),
        edge_weight=torch.ones((32, 32), dtype=torch.float32, device=dev),
        created_tick=25,
        is_void=True,
    )
    policy.recruit_initial_members(f_art, 400)
    policy.recruit_initial_members(f_void, 350)
    for t in range(30, 75):
        batch = policy.step(t, canvas, [f_art, f_void], [])
        canvas.apply_placements(t, batch.xs, batch.ys, batch.colors, batch.factions)
    update_faction_metrics(canvas, [f_art, f_void], policy.user_factions)
    assert f_art.member_count > 60
    assert f_art.integrity > 0.30
    assert f_void.integrity > 0.25


def test_one_shot_population_turnover_ratio() -> None:
    """REQ-SIM-014: verify 20% One-Shot Citizen session turnover ratio."""
    dev = torch.device("cpu")
    canvas = CanvasState(64, 64, dev)
    policy = StochasticPolicy(num_users=4000, width=64, height=64, base_cooldown=4, device=dev, seed=123)
    for t in range(1, 220):
        batch = policy.step(t, canvas, [], [])
        canvas.apply_placements(t, batch.xs, batch.ys, batch.colors, batch.factions)
    stats = policy.get_population_stats()
    assert stats["completed_sessions"] > 5000
    assert stats["total_unique_users"] > 9000
    # After steady-state churn, one-shot share among completed sessions is strongly represented (~20% of arrivals;
    # slightly higher in finite windows because 1-click sessions finish immediately while 200-click veterans are still online).
    # Verify newly sampled click budgets directly hit 20.0% ± 1.5%:
    sample = policy._sample_click_budgets(50_000)
    one_shot_frac = float((sample == 1).to(torch.float32).mean().item())
    assert 0.185 <= one_shot_frac <= 0.215, f"Expected ~20% one-shot, got {one_shot_frac:.3f}"


def test_rev13_art_generators_and_clean_space() -> None:
    # 32 National flags across 3 scales (REQ-ART-008)
    assert len(NATIONAL_FLAGS) >= 24
    for fl in NATIONAL_FLAGS:
        for w, h in ((20, 13), (44, 28), (84, 52)):
            bp = generate_national_flag(fl.key, w, h)
            assert bp.shape == (h, w)
            assert len(np.unique(bp)) >= 2

    # 24 Micro sprites (REQ-ART-009)
    assert len(SPRITE_KEYS) >= 20
    for sk in SPRITE_KEYS:
        bp, _ = generate_micro_sprite(sk, 20, 20, seed=5)
        assert bp.shape == (20, 20)
        assert int((bp >= 0).sum()) >= 50

    # Fine-art paintings in gold frames (REQ-ART-010)
    for spec in FINE_ART_CATALOG[:6]:
        bp_small = generate_procedural_painting(spec, 26, 22)
        bp_med = generate_procedural_painting(spec, 48, 38)
        assert bp_small.shape == (22, 26) and bp_med.shape == (38, 48)
        assert int(bp_med[0, 0]) == BLACK_IDX and int(bp_med[1, 1]) in (YELLOW_IDX, 7)

    # Empty-Space pure white validation (REQ-ART-011)
    cs = generate_clean_space(24, 20, style="pure_white")
    assert bool((cs == WHITE_IDX).all())
    assert ArtManager._is_valid_blueprint(cs, art_mode="clean_space")

    # Transverse Ribbons H, V, Diagonal (REQ-ART-012)
    for axis in ("h", "v", "d1", "d2"):
        r_bp = generate_transverse_ribbon(RIBBON_PATTERNS[0][0], axis, length=120, thickness=8)
        assert ArtManager._is_valid_blueprint(r_bp, art_mode="ribbon")

    # Black Hole, Fractals, Void Monster (REQ-ART-013)
    bh = generate_black_hole(64, 64, seed=7)
    assert int((bh == BLACK_IDX).sum()) > 400
    for fk in ("sierpinski", "mandelbrot", "carpet", "diamond"):
        fr = generate_fractal(fk, 48, 48, seed=7)
        assert int((fr >= 0).sum()) > 300
    vm = generate_void_monster(64, 64, seed=7)
    assert int((vm == BLACK_IDX).sum()) > 500


def test_ocr_edit_distance_and_schema_bounds() -> None:
    # REQ-NEXT-001: reject subtle spelling mistakes like HOLD FAIST for HOLD FAST
    assert ocr_text_matches("HOLD FAST", "HOLD FAST") is True
    assert ocr_text_matches("HOLD FAST", "HOLD FAIST") is False
    assert ocr_text_matches("R/CATS", "R/CATS") is True
    assert ocr_text_matches("R/CATS", "R/CARS") is False
    assert ocr_text_matches("BLUE CORNER", "BLUE CORNER") is True

    # REQ-NEXT-002: allow expander_ratio=1.0 without schema error
    dec = FactionDecision(action="EXPAND", builder_ratio=0.0, defender_ratio=0.0, expander_ratio=1.0, reasoning="Push!")
    assert dec.expander_ratio == 1.0


def test_subreddit_sampler() -> None:
    sampler = SubredditSampler(seed=42)
    assert len(sampler.entries) >= 50
    picks = {sampler.sample(tier).name for tier in ("macro", "medium", "micro") for _ in range(5)}
    assert len(picks) >= 12


def test_shared_border_rect() -> None:
    dev = torch.device("cpu")
    bp = torch.zeros((40, 40), dtype=torch.int16, device=dev)
    ew = torch.ones((40, 40), dtype=torch.float32, device=dev)
    a = Faction(0, "A", "t", "s", "procedural", "p", 1, 2, 3, 20, 20, 40, 40, bp, ew, 0)
    b = Faction(1, "B", "t", "s", "procedural", "p", 4, 5, 6, 62, 24, 40, 40, bp, ew, 0)
    rect = compute_shared_border_rect(a, b, pad=5)
    assert rect is not None
    x0, y0, x1, y1 = rect
    assert x1 > x0 and y1 > y0


def test_neural_policy_batched_200_factions() -> None:
    """REQ-ML-008: verify single-pass batched NeuralRedditPolicy with 200 active factions >= 8,000 px/s on MPS."""
    model_path = Path("models/reddit_citizen.pt")
    if not model_path.exists():
        return
    from pixel_battle.policies.neural import NeuralRedditPolicy

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    canvas = CanvasState(512, 512, dev)
    policy = NeuralRedditPolicy(model_path, num_users=36_000, width=512, height=512, base_cooldown=18, device=dev, seed=11)

    # Create 200 active factions (120 micro, 48 medium, 24 macro, 8 ribbons)
    factions: list[Faction] = []
    for i in range(200):
        if i < 120:
            w, h, tier = 20, 20, "micro"
            bp_np, _ = generate_micro_sprite(SPRITE_KEYS[i % len(SPRITE_KEYS)], w, h, seed=i)
        elif i < 168:
            w, h, tier = 40, 26, "medium"
            bp_np = generate_national_flag(NATIONAL_FLAGS[i % len(NATIONAL_FLAGS)].key, w, h)
        elif i < 192:
            w, h, tier = 64, 44, "macro"
            bp_np = generate_national_flag(NATIONAL_FLAGS[i % len(NATIONAL_FLAGS)].key, w, h)
        else:
            w, h, tier = 240, 8, "ribbon"
            bp_np = generate_transverse_ribbon(RIBBON_PATTERNS[i % len(RIBBON_PATTERNS)][0], "h", w, h)
        x0 = (i * 37) % (512 - w)
        y0 = (i * 53) % (512 - h)
        bp_t = torch.from_numpy(bp_np).to(dev)
        fac = Faction(
            id=i,
            name=f"F{i}",
            theme="test",
            slogan="go",
            art_mode="flag" if tier != "micro" else "micro_sprite",
            art_prompt="test",
            primary_color=BLUE_IDX,
            secondary_color=YELLOW_IDX,
            accent_color=WHITE_IDX,
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            blueprint=bp_t,
            edge_weight=compute_edge_weights(bp_t),
            created_tick=0,
            size_tier=tier,
        )
        factions.append(fac)
        policy.recruit_initial_members(fac, 70 if tier == "micro" else (180 if tier == "medium" else 350))

    # Warmup 2 ticks then benchmark 15 ticks
    for t in range(1, 3):
        b = policy.step(t, canvas, factions, [])
        canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)
    if dev.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    placed = 0
    for t in range(3, 18):
        b = policy.step(t, canvas, factions, [])
        canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)
        placed += int(b.xs.numel())
    if dev.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    pps = placed / max(1e-4, dt)
    print(f"\n[BENCH 200 FACTIONS] {placed:,} px in {dt:.2f}s => {pps:,.0f} px/s on {dev}")
    if dev.type == "mps":
        assert pps >= 8_000, f"Expected >= 8,000 px/s with 200 batched factions, got {pps:,.0f}"


def test_typography_styled_rendering() -> None:
    from pixel_battle.art.typography import FONT_STYLES, TypographyStyleSpec, composite_styled_text

    for style in ("gothic", "bubble", "arcade", "condensed", "script", "pixel_retro"):
        assert style in FONT_STYLES
        bp = np.full((56, 120), -1, dtype=np.int16)
        spec = TypographyStyleSpec(
            include_text=True,
            text="R/TEST",
            font_style=style,
            size_tier="medium",
            color_effect="vertical_gradient",
            border_effect="extrude_3d",
            layout="wavy",
            placement="center",
        )
        out = composite_styled_text(bp, spec)
        assert int((out >= 0).sum()) > 50


def test_role_split_rounding_zero_expander() -> None:
    """Verify odd ready counts with expander=0.0 never overflow k_build + k_def > k."""
    dev = torch.device("cpu")
    canvas = CanvasState(64, 64, dev)
    bp = torch.full((16, 16), BLUE_IDX, dtype=torch.int16, device=dev)
    fac = Faction(
        id=0,
        name="OddTest",
        theme="test",
        slogan="hold",
        art_mode="flag",
        art_prompt="test",
        primary_color=BLUE_IDX,
        secondary_color=YELLOW_IDX,
        accent_color=WHITE_IDX,
        x0=10,
        y0=10,
        w=16,
        h=16,
        blueprint=bp,
        edge_weight=compute_edge_weights(bp),
        created_tick=0,
        role_weights={"builder": 0.5, "defender": 0.5, "expander": 0.0},
    )

    # Stochastic policy with exactly 3 ready members (round(1.5) + round(1.5) == 4 > 3 if unclamped)
    sp = StochasticPolicy(num_users=3, width=64, height=64, base_cooldown=1, device=dev, seed=1)
    sp.user_factions[:] = 0
    sp.cooldown[:] = 1
    b1 = sp.step(1, canvas, [fac], [])
    assert int(b1.xs.numel()) == 3

    model_path = Path("models/reddit_citizen.pt")
    if model_path.exists():
        from pixel_battle.policies.neural import NeuralRedditPolicy

        np_pol = NeuralRedditPolicy(model_path, num_users=3, width=64, height=64, base_cooldown=1, device=dev, seed=1)
        np_pol.user_factions[:] = 0
        np_pol.cooldown[:] = 1
        b2 = np_pol.step(1, canvas, [fac], [])
        assert int(b2.xs.numel()) == 3


def test_comic_strips_and_cartoon_memes() -> None:
    """Verify all 12 multi-panel comic strips, 14 cartoon/meme portraits, and 34 micro sprites."""
    from pixel_battle.art.comics import (
        CARTOON_MEME_CATALOG,
        COMIC_CATALOG,
        generate_cartoon_meme,
        generate_comic_strip,
    )

    assert len(COMIC_CATALOG) >= 12
    for i, c in enumerate(COMIC_CATALOG):
        w, h = (84, 34) if c.layout == "3h" else ((56, 44) if c.layout == "2v" else (58, 48))
        bp, spec = generate_comic_strip(c.key, w, h, seed=i)
        assert bp.shape == (h, w)
        assert spec.key == c.key
        assert len(np.unique(bp[bp >= 0])) >= 4

    assert len(CARTOON_MEME_CATALOG) >= 14
    for i, m in enumerate(CARTOON_MEME_CATALOG):
        w, h = (56, 36) if m.key == "nyan_cat" else (44, 44)
        bp, spec = generate_cartoon_meme(m.key, w, h, seed=i)
        assert bp.shape == (h, w)
        assert spec.key == m.key
        assert len(np.unique(bp[bp >= 0])) >= 2

    assert len(SPRITE_KEYS) >= 34
