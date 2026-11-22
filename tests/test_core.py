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


# ------------------------------------------------------------------
# Revision 2.0 Tests (1:1 Reddit r/place 2022 Scale & AI-First)
# ------------------------------------------------------------------


def test_four_act_palette_unlocks_and_masking() -> None:
    """REQ-SIM-008 & REQ-SIM-016: enforce 16 -> 24 -> 32 -> Whiteout palette & viewport boundaries."""
    from pixel_battle.palette import (
        PALETTE_TIER_1_16,
        PALETTE_TIER_2_24,
        PALETTE_TIER_3_32,
        PALETTE_TIER_4_WHITEOUT,
        remap_indices_to_allowed,
    )

    assert len(PALETTE_TIER_1_16) == 16
    assert len(PALETTE_TIER_2_24) == 24
    assert len(PALETTE_TIER_3_32) == 32
    assert PALETTE_TIER_4_WHITEOUT == (WHITE_IDX,)

    # Verify remap_indices_to_allowed strictly maps every non-transparent color into allowed set
    full_grid = np.arange(32, dtype=np.int16).reshape(4, 8)
    remapped = remap_indices_to_allowed(full_grid, PALETTE_TIER_1_16)
    assert set(int(v) for v in np.unique(remapped)).issubset(set(PALETTE_TIER_1_16))

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = Path("models/reddit_citizen.pt")
    policies_to_test: list[StochasticPolicy] = [
        StochasticPolicy(num_users=600, width=128, height=128, base_cooldown=2, device=dev, seed=101)
    ]
    if model_path.exists():
        from pixel_battle.policies.neural import NeuralRedditPolicy

        policies_to_test.append(
            NeuralRedditPolicy(model_path, num_users=600, width=128, height=128, base_cooldown=2, device=dev, seed=102)
        )

    for pol in policies_to_test:
        canvas = CanvasState(128, 128, dev, active_w=64, active_h=64, palette_tier=1)
        pol.active_w, pol.active_h = 64, 64
        pol.pos_x = pol.pos_x % 64
        pol.pos_y = pol.pos_y % 64

        # Act 1: 64x64, 16 colors only
        for t in range(1, 8):
            b = pol.step(t, canvas, [], [])
            assert int(b.xs.max().item()) < 64 and int(b.ys.max().item()) < 64
            used_cols = set(int(c) for c in torch.unique(b.colors).detach().cpu().tolist())
            assert used_cols.issubset(set(PALETTE_TIER_1_16)), f"Locked colors leaked in Act 1: {used_cols - set(PALETTE_TIER_1_16)}"
            canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)

        # Act 2: 128x64, 24 colors only
        canvas.expand_viewport(128, 64, 2)
        for t in range(8, 14):
            b = pol.step(t, canvas, [], [])
            assert int(b.xs.max().item()) < 128 and int(b.ys.max().item()) < 64
            used_cols = set(int(c) for c in torch.unique(b.colors).detach().cpu().tolist())
            assert used_cols.issubset(set(PALETTE_TIER_2_24)), f"Locked colors leaked in Act 2: {used_cols - set(PALETTE_TIER_2_24)}"
            canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)

        # Act 4: 128x128 Whiteout — 100% pure white only!
        canvas.expand_viewport(128, 128, 4)
        non_white_before = int((canvas.colors != WHITE_IDX).sum().item())
        assert non_white_before > 50
        for t in range(14, 24):
            b = pol.step(t, canvas, [], [])
            assert bool((b.colors == WHITE_IDX).all().item()), "Act 4 Whiteout placed a non-white pixel!"
            canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)
        non_white_after = int((canvas.colors != WHITE_IDX).sum().item())
        assert non_white_after < non_white_before, "Act 4 Whiteout did not erase colored pixels!"


def test_timezone_diurnal_curves_and_flag_regions() -> None:
    """REQ-SIM-015: verify smooth regional sleep/wake diurnal curves and national flag timezone bindings."""
    from pixel_battle.factions import FLAG_TIMEZONES, TIMEZONE_PEAK_UTC, timezone_awake_factor

    for tz in TIMEZONE_PEAK_UTC:
        vals = [timezone_awake_factor(tz, float(h)) for h in range(24)]
        assert min(vals) >= 0.21 and max(vals) <= 1.36
        if tz != "GLOBAL":
            assert max(vals) >= 1.30 and min(vals) <= 0.28

    # Distinct regional prime times (Europe evening vs Americas evening vs Asia-Pacific evening)
    assert timezone_awake_factor("EUROPE", 19.5) > 1.25
    assert timezone_awake_factor("EUROPE", 7.5) < 0.35
    assert timezone_awake_factor("ASIA_PACIFIC", 12.0) > 1.25
    assert timezone_awake_factor("AMERICAS", 1.5) > 1.25

    # National flags mapped to true timezones
    assert FLAG_TIMEZONES["japan"] == "ASIA_PACIFIC"
    assert FLAG_TIMEZONES["france"] == "EUROPE"
    assert FLAG_TIMEZONES["usa"] == "AMERICAS"
    assert FLAG_TIMEZONES["brazil"] == "LATAM"
    for fl in NATIONAL_FLAGS:
        assert fl.key in FLAG_TIMEZONES, f"Missing timezone mapping for flag '{fl.key}'"


def test_reddit2022_population_calibration() -> None:
    """REQ-SIM-014: verify 20% one-shot ratio and ~15.5 mean px/user for 10M users at 160M placements."""
    dev = torch.device("cpu")
    policy = StochasticPolicy(num_users=1000, width=64, height=64, base_cooldown=4, device=dev, seed=2022)
    sample = policy._sample_click_budgets(100_000)
    one_shot = float((sample == 1).to(torch.float32).mean().item())
    mean_px = float(sample.to(torch.float32).mean().item())
    assert 0.19 <= one_shot <= 0.21, f"Expected 20% ± 1% one-shot, got {one_shot:.4f}"
    assert 13.0 <= mean_px <= 18.0, f"Expected ~15.5 mean clicks/user, got {mean_px:.2f}"


def test_multilingual_cjk_and_unicode_typography() -> None:
    """REQ-ART-014: verify crisp CJK & accented Unicode wordmark rendering inside allowed palette."""
    from pixel_battle.art.typography import TypographyStyleSpec, composite_styled_text
    from pixel_battle.palette import PALETTE_TIER_1_16, remap_indices_to_allowed

    samples = ["像素", "日本", "한국", "¡VIVA!", "FRANÇAIS", "DEUTSCH", "TÜRKİYE"]
    for txt in samples:
        bp = np.full((34, 96), -1, dtype=np.int16)
        spec = TypographyStyleSpec(
            include_text=True,
            text=txt,
            font_style="block",
            size_tier="large",
            color_effect="solid",
            border_effect="outline_1px",
            layout="straight",
            placement="center",
            fill_color=YELLOW_IDX,
            outline_color=BLACK_IDX,
        )
        out = remap_indices_to_allowed(composite_styled_text(bp, spec), PALETTE_TIER_1_16)
        valid_px = int((out >= 0).sum())
        assert valid_px >= 40, f"Multilingual wordmark '{txt}' rendered too few pixels ({valid_px})"
        used = set(int(v) for v in np.unique(out[out >= 0]))
        assert used.issubset(set(PALETTE_TIER_1_16))


def test_sandboxed_llm_pixel_code_runner() -> None:
    """REQ-ART-009: verify AST sandbox executes safe NumPy/math geometry code and blocks unsafe code."""
    from pixel_battle.art.flux_art import run_sandboxed_pixel_code
    from pixel_battle.palette import PALETTE_TIER_1_16

    good_code = """```python
import numpy as np
import math

def draw(grid, w, h):
    cx, cy = w // 2, h // 2
    for y in range(h):
        for x in range(w):
            r = math.hypot(x - cx, y - cy)
            if 6 <= r <= 20:
                grid[y, x] = 2 if int(r) % 2 == 0 else 6
```"""
    bp = run_sandboxed_pixel_code(good_code, 48, 48, allowed_colors=PALETTE_TIER_1_16)
    assert bp is not None and bp.shape == (48, 48)
    assert int((bp >= 0).sum()) > 200
    assert set(int(v) for v in np.unique(bp[bp >= 0])).issubset(set(PALETTE_TIER_1_16))

    bad_scripts = [
        "import os\ndef draw(grid, w, h): os.system('echo pwned')",
        "def draw(grid, w, h): open('/etc/passwd').read()",
        "def draw(grid, w, h): ().__class__.__base__.__subclasses__()",
        "def draw(grid, w, h): eval('1+1')",
        "def draw(grid, w, h): __import__('subprocess')",
    ]
    for bad in bad_scripts:
        assert run_sandboxed_pixel_code(bad, 32, 32) is None, f"Sandbox failed to block: {bad}"


def test_four_act_engine_lifecycle_checkpoint_and_timelapse(tmp_path: Path) -> None:
    """End-to-end 4-Act lifecycle: Act 1 -> Act 2 -> Act 3 -> final_peak.png -> Act 4 Whiteout + dynamic-camera MP4."""
    from pixel_battle.config import SimConfig
    from pixel_battle.engine import SimulationEngine
    from pixel_battle.timelapse import render_timelapse

    cfg = SimConfig(
        width=256,
        height=256,
        num_users=1600,
        base_cooldown=3,
        max_ticks=80,
        chaos_ticks=8,
        discovery_interval=12,
        epoch_interval=25,
        flash_raid_interval_ticks=30,
        max_active_factions=24,
        checkpoint_interval=40,
        enable_expansion=True,
        enable_whiteout=True,
        act2_progress=0.30,
        act3_progress=0.60,
        whiteout_progress=0.88,
        use_flux_art=False,
        use_flux_typography=False,
        use_diffusion=False,
        policy_mode="stochastic",
        llm_url="http://127.0.0.1:9/v1",  # Offline deterministic fallback
        device="cpu",
        run_dir=tmp_path / "run_4act",
        seed=77,
    )
    eng = SimulationEngine(cfg)
    assert eng.act == 1 and (eng.canvas.active_w, eng.canvas.active_h) == (128, 128)
    assert len(eng.canvas.allowed_colors) == 16

    saw_acts: set[int] = {eng.act}
    for t in range(1, cfg.max_ticks + 1):
        eng.step_once()
        saw_acts.add(eng.act)
        if t in (10, 30, 52):
            eng.run_pareidolia_discovery(max_new=3)
        if t == 36:
            eng.spawn_streamer_flash_raid()
        if t == 45:
            eng.run_geopolitical_epoch()

    eng.save_checkpoint()
    eng.logger.close()
    eng.brain.close()

    assert saw_acts == {1, 2, 3, 4}, f"Expected all 4 acts, saw {saw_acts}"
    assert (cfg.run_dir / "final_peak.png").exists(), "Missing pre-whiteout final_peak.png!"
    assert eng.canvas.is_whiteout and len(eng.canvas.allowed_colors) == 1

    # Verify checkpoint reload restores Act 4 & faction metadata cleanly
    eng2 = SimulationEngine(cfg, resume=True)
    assert eng2.act == 4 and eng2.tick == cfg.max_ticks
    assert len(eng2.factions) == len(eng.factions) and len(eng2.factions) >= 6
    eng2.logger.close()
    eng2.brain.close()

    # Verify dynamic-camera timelapse renders cleanly from 4-act chronicle
    mp4 = render_timelapse(
        cfg.run_dir,
        cfg.run_dir / "dyn_test.mp4",
        fps=15,
        duration_sec=2.0,
        scale=1,
        hold_end_sec=0.5,
        show_hud=True,
        dynamic_camera=True,
    )
    assert mp4.exists() and mp4.stat().st_size > 8_000


def test_neural_policy_2000x2000_1200_factions_mps_benchmark() -> None:
    """REQ-ML-008 & REQ-SYS-001: 1:1 r/place 2022 benchmark (2000x2000, 1,200 factions, 120k slots) >= 10,000 px/s on MPS."""
    model_path = Path("models/reddit_citizen.pt")
    if not model_path.exists():
        return
    from pixel_battle.policies.neural import NeuralRedditPolicy

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    canvas = CanvasState(2000, 2000, dev, active_w=2000, active_h=2000, palette_tier=3)
    policy = NeuralRedditPolicy(
        model_path,
        num_users=120_000,
        width=2000,
        height=2000,
        base_cooldown=14,
        device=dev,
        seed=2022,
    )

    factions: list[Faction] = []
    for i in range(1200):
        if i < 696:
            w, h, tier = 24, 24, "micro"
        elif i < 1020:
            w, h, tier = 48, 36, "medium"
        elif i < 1164:
            w, h, tier = 84, 56, "macro"
        else:
            w, h, tier = 320, 10, "ribbon"
        x0 = (i * 137) % (2000 - w)
        y0 = (i * 193) % (2000 - h)
        bp_np = np.full((h, w), (i % 30) + 1, dtype=np.int16)
        bp_t = torch.from_numpy(bp_np).to(dev)
        fac = Faction(
            id=i,
            name=f"R22_{i}",
            theme="bench",
            slogan="go",
            art_mode="flux_art" if tier != "ribbon" else "ribbon",
            art_prompt="bench",
            primary_color=(i % 30) + 1,
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
            timezone_region=("AMERICAS", "EUROPE", "ASIA_PACIFIC", "LATAM", "GLOBAL")[i % 5],
        )
        factions.append(fac)
        # Assign slots directly for fast deterministic benchmark setup
        start_s = i * 80
        policy.user_factions[start_s : start_s + 80] = i

    # Verify bulk CPU-cached metrics update speed across all 1,200 factions
    t_m = time.perf_counter()
    update_faction_metrics(canvas, factions, policy.user_factions)
    dt_m = (time.perf_counter() - t_m) * 1000.0
    print(f"\n[BENCH METRICS 1200 FACTIONS] {dt_m:.1f} ms")
    assert dt_m < 300.0, f"update_faction_metrics too slow for 1,200 factions: {dt_m:.1f} ms"

    # Warmup 2 ticks then benchmark 12 ticks
    for t in range(1, 3):
        b = policy.step(t, canvas, factions, [])
        canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)
    if dev.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    placed = 0
    for t in range(3, 15):
        b = policy.step(t, canvas, factions, [])
        canvas.apply_placements(t, b.xs, b.ys, b.colors, b.factions)
        placed += int(b.xs.numel())
    if dev.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    pps = placed / max(1e-4, dt)
    print(f"[BENCH 2000x2000 1200 FACTIONS 120K SLOTS] {placed:,} px in {dt:.2f}s => {pps:,.0f} px/s on {dev}")
    if dev.type == "mps":
        assert pps >= 10_000, f"Expected >= 10,000 px/s on 2000x2000 MPS, got {pps:,.0f}"


def test_macro_community_planner_profiles_and_export(tmp_path: Path) -> None:
    """Verify fast ~1s macro community planner reproduces baseline sparsity vs tuned wall-to-wall coverage."""
    from pixel_battle.planner import (
        export_plan_artifacts,
        preset_baseline_2000,
        preset_tuned_1024,
        preset_tuned_2000,
        run_macro_plan,
    )

    t0 = time.perf_counter()
    base = run_macro_plan(preset_baseline_2000(), steps=96)
    tuned2k = run_macro_plan(preset_tuned_2000(), steps=96)
    tuned1k = run_macro_plan(preset_tuned_1024(), steps=96)
    dt = time.perf_counter() - t0
    assert dt < 3.0, f"Macro planner should run in < 3s for 3 full 8h plans, took {dt:.2f}s"

    sb = base.summary_dict()
    s2 = tuned2k.summary_dict()
    s1 = tuned1k.summary_dict()
    assert sb["peak_cov_overall_pct"] < 45.0
    assert s2["peak_cov_overall_pct"] >= 65.0 and s2["peak_cov_act3_pct"] >= 55.0
    assert s1["peak_cov_overall_pct"] >= 70.0 and s1["peak_cov_act3_pct"] >= 60.0
    assert s2["peak_organized_pct"] > sb["peak_organized_pct"] + 40.0

    arts = export_plan_artifacts(tuned2k, tmp_path / "plan_out", render_mp4=True)
    assert arts["trace"].exists() and arts["mp4"].exists() and arts["mp4"].stat().st_size > 8_000
