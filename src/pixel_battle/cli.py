"""CLI entrypoints for running the pixel battle, testing local AI models, and rendering timelapses."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import uvicorn

from pixel_battle.config import SimConfig, apply_preset
from pixel_battle.engine import SimulationEngine, resolve_device
from pixel_battle.server import create_app
from pixel_battle.timelapse import render_timelapse


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / datetime.now().strftime("battle_%Y%m%d_%H%M%S")
    cfg = SimConfig(run_dir=run_dir)
    apply_preset(cfg, args.preset)

    # Apply explicit CLI overrides on top of preset defaults
    if args.width is not None:
        cfg.width = args.width
    if args.height is not None:
        cfg.height = args.height
    if args.users is not None:
        cfg.num_users = args.users
    if args.max_factions is not None:
        cfg.max_active_factions = args.max_factions
    if args.chaos_ticks is not None:
        cfg.chaos_ticks = args.chaos_ticks
    if args.discovery_interval is not None:
        cfg.discovery_interval = args.discovery_interval
    if args.epoch_interval is not None:
        cfg.epoch_interval = args.epoch_interval
    if args.flash_raid_interval is not None:
        cfg.flash_raid_interval_ticks = args.flash_raid_interval

    cfg.max_ticks = args.max_ticks
    cfg.max_hours = args.max_hours
    cfg.target_tps = args.tps
    cfg.llm_url = args.llm_url or args.ollama_url or cfg.llm_url
    cfg.llm_model = args.llm_model or args.vlm_model or cfg.llm_model
    cfg.ollama_url = cfg.llm_url
    cfg.vlm_model = cfg.llm_model
    cfg.keychain_service = args.keychain_service
    cfg.use_flux_art = not args.no_flux_art
    cfg.use_flux_typography = not args.no_flux_typography
    cfg.use_diffusion = args.use_diffusion
    cfg.enable_expansion = not args.no_expansion
    cfg.enable_whiteout = not args.no_whiteout
    cfg.enable_timezones = not args.no_timezones
    cfg.enable_flash_raids = not args.no_flash_raids
    cfg.enable_burnout = not args.no_burnout
    cfg.policy_mode = args.policy
    cfg.neural_model_path = Path(args.neural_model)
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.host = args.host
    cfg.port = args.port

    engine = SimulationEngine(cfg, resume=args.resume)
    latest_link = Path("runs/latest")
    if run_dir.parent == Path("runs") and run_dir.name != "latest":
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name)
        except OSError:
            pass

    if (cfg.use_flux_art or cfg.use_diffusion) and (args.warmup_flux or args.warmup_diffusion):
        print("Warming up local MLX FLUX.1-schnell + Pixel-Art LoRA...")
        engine.art.warmup()

    engine.start()
    print(f"\nPixel Battle ({cfg.preset}) running in: {run_dir}")
    print(
        f"Canvas: {engine.canvas.active_w}x{engine.canvas.active_h} -> {cfg.width}x{cfg.height} "
        f"(palette={len(engine.canvas.allowed_colors)}c, 4-act={'ON' if cfg.enable_expansion else 'OFF'}) | "
        f"Online Slots: {cfg.num_users:,} | Max Factions: {cfg.max_active_factions:,}"
    )
    print(f"Device: {engine.device} | Policy: {engine.policy_name}")
    print(f"Local VLM ({cfg.llm_model} @ {cfg.llm_url}): {'ONLINE' if engine.ollama_ready else 'OFFLINE (fallback)'}")
    print(f"Art engine: {'MLX FLUX.1-schnell Pixel-Art LoRA + LLM Code + Unicode Typography' if cfg.use_flux_art else 'Procedural / Offline'}")

    if args.headless:
        def _handle_sig(_signum, _frame):
            engine.stop_requested = True

        signal.signal(signal.SIGINT, _handle_sig)
        signal.signal(signal.SIGTERM, _handle_sig)
        try:
            while not engine.stop_requested and not engine.is_finished():
                time.sleep(5.0)
                st = engine.get_ui_state()
                elapsed_m = (time.time() - engine.start_wall_time) / 60.0
                lim = f"{cfg.max_hours:.1f}h" if cfg.max_hours > 0 else f"{cfg.max_ticks}"
                pop = st.get("population", {})
                u_tot = pop.get("total_unique_users", cfg.num_users)
                os_pct = pop.get("one_shot_pct", 20.0)
                n_fac = st.get("total_active_factions", len(st["factions"]))
                ai_pct = st.get("ai_faction_pct", 0.0)
                print(
                    f"[{elapsed_m:5.1f}m | Act {st['act']} Hr {st['virtual_hour']:4.1f}/72 | "
                    f"{st['active_w']}x{st['active_h']} {st['palette_size']:2d}c | t={st['tick']:6d}/{lim}] "
                    f"{st['total_placements']:11,d} px | "
                    f"{st['pixels_per_sec']:6,d} px/s | "
                    f"users={u_tot:,} ({os_pct:.1f}% 1-shot) | "
                    f"{n_fac:4d} fac ({ai_pct:.0f}% AI) | "
                    f"{st['brain_status']}",
                    flush=True,
                )
        finally:
            engine.stop()
        if engine.sim_error is not None:
            print(f"Simulation aborted with error: {engine.sim_error}", file=sys.stderr)
            return 1
        return 0

    print(f"Live Dashboard -> http://{cfg.host}:{cfg.port}\nPress Ctrl+C to save checkpoint & stop.\n")
    app = create_app(engine)
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
    finally:
        print("\nSaving checkpoint & closing event log...")
        engine.stop()
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "timelapse.mp4"
    print(
        f"Rendering timelapse from {run_dir} -> {out_path} "
        f"({args.fps} fps, {args.duration}s, scale={args.scale}x, dynamic_camera={not args.no_dynamic_camera})..."
    )
    t0 = time.perf_counter()
    res = render_timelapse(
        run_dir=run_dir,
        out_path=out_path,
        fps=args.fps,
        duration_sec=args.duration,
        scale=args.scale,
        hold_end_sec=args.hold,
        show_hud=not args.no_hud,
        crf=args.crf,
        curve=args.curve,
        dynamic_camera=not args.no_dynamic_camera,
    )
    dt = time.perf_counter() - t0
    print(f"Done in {dt:.1f}s -> {res}")
    return 0


def _cmd_train_reddit(args: argparse.Namespace) -> int:
    from pixel_battle.ml.train import train_reddit_citizen_model

    out = train_reddit_citizen_model(
        data_dir=Path(args.data_dir),
        out_model=Path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        patch_size=args.patch_size,
        early_samples=args.early_samples,
        peak_samples=args.peak_samples,
        device_spec=args.device,
        seed=args.seed,
    )
    print(f"\nTrained RedditCitizenNet v2 saved -> {out}")
    print("Run `uv run pixel-battle run` (auto-loads neural policy) or pass `--policy neural`!")
    return 0


def _cmd_test_stack(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image
    from pixel_battle.art.manager import ArtManager
    from pixel_battle.brain import LocalVLMBrain
    from pixel_battle.palette import PALETTE_RGB, PALETTE_TIER_1_16

    dev = resolve_device("auto")
    print(f"1. PyTorch device: {dev}")

    out_dir = Path("runs/stack_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_url = args.llm_url
    llm_model = args.llm_model
    brain = LocalVLMBrain(ollama_url=llm_url, vlm_model=llm_model, keychain_service=args.keychain_service)
    healthy = brain.check_health()
    print(f"2. Local VLM ({llm_model} @ {llm_url}) health: {'OK' if healthy else 'NOT FOUND'}")
    if healthy:
        t0 = time.perf_counter()
        ideas = brain.brainstorm_faction_batch(count=4, act=1, virtual_hour=6.0, unlocked_colors=PALETTE_TIER_1_16)
        print(f"   Brainstormed {len(ideas)} factions in {time.perf_counter()-t0:.2f}s:")
        for idea in ideas:
            print(f"     - {idea.name} [{idea.timezone_region}/{idea.language}, {idea.art_mode}, fluid={idea.fluid_cutout}]")

        code = brain.write_geometric_code("Spiral Lattice", "concentric spiral diamonds", 48, 48, PALETTE_TIER_1_16)
        print(f"   Geometric code generated ({len(code)} chars)")

    cfg = SimConfig(use_flux_art=not args.no_flux_art, use_diffusion=False)
    art = ArtManager(cfg, dev)
    if not args.no_flux_art:
        print("3. Testing MLX FLUX.1-schnell + Pixel-Art LoRA (fluid die-cut silhouette)...")
        t0 = time.perf_counter()
        bp, eng = art.create_blueprint(
            "flux_art",
            "cheerful tuxedo cat wearing a tiny golden wizard hat, crisp 16-bit pixel art sprite",
            64,
            64,
            15,
            6,
            0,
            "CatWizard",
            allowed_colors=PALETTE_TIER_1_16,
            fluid_cutout=True,
        )
        rgb = PALETTE_RGB[np.where(bp < 0, 0, bp)]
        Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST).save(out_dir / "flux_pixel_sample.png")
        print(f"   FLUX pixel art ({eng}, {time.perf_counter()-t0:.2f}s) saved -> {out_dir / 'flux_pixel_sample.png'}")

    brain.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pixel-battle", description="Local AI/ML Reddit r/place simulation")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Start the live pixel battle simulation")
    p_run.add_argument("--preset", choices=["standard", "large", "reddit2022"], default="standard", help="Scale preset")
    p_run.add_argument("--width", type=int, default=None)
    p_run.add_argument("--height", type=int, default=None)
    p_run.add_argument("--users", type=int, default=None)
    p_run.add_argument("--max-ticks", type=int, default=8_000)
    p_run.add_argument("--max-hours", type=float, default=0.0, help="If > 0, run for this many wall-clock hours cleanly")
    p_run.add_argument("--chaos-ticks", type=int, default=None)
    p_run.add_argument("--discovery-interval", type=int, default=None)
    p_run.add_argument("--epoch-interval", type=int, default=None)
    p_run.add_argument("--flash-raid-interval", type=int, default=None)
    p_run.add_argument("--max-factions", type=int, default=None)
    p_run.add_argument("--tps", type=float, default=18.0, help="Target ticks/sec (0 = uncapped)")
    p_run.add_argument("--llm-url", default=None, help="OpenAI-compatible local LLM endpoint (e.g. http://localhost:1234/v1)")
    p_run.add_argument("--llm-model", default=None, help="Local VLM model name (e.g. qwen3-vl-30b-a3b-instruct-mlx)")
    p_run.add_argument("--keychain-service", default="place-sim", help="macOS Keychain service name for local API token")
    p_run.add_argument("--ollama-url", default=None, help="Legacy alias for --llm-url")
    p_run.add_argument("--vlm-model", default=None, help="Legacy alias for --llm-model")
    p_run.add_argument("--no-flux-art", action="store_true", help="Disable MLX FLUX.1-schnell Pixel-Art LoRA")
    p_run.add_argument("--no-flux-typography", action="store_true", help="Disable FLUX wordmark mode")
    p_run.add_argument("--use-diffusion", action="store_true", help="Enable legacy SDXL-Turbo fallback")
    p_run.add_argument("--no-diffusion", action="store_true", help="Legacy flag (SDXL-Turbo is off by default)")
    p_run.add_argument("--warmup-flux", action="store_true", help="Preload MLX FLUX Pixel-Art LoRA before starting ticks")
    p_run.add_argument("--warmup-diffusion", action="store_true", help="Legacy alias for --warmup-flux")
    p_run.add_argument("--no-expansion", action="store_true", help="Disable 4-act canvas expansion and start at full size")
    p_run.add_argument("--no-whiteout", action="store_true", help="Disable Act 4 Whiteout finale")
    p_run.add_argument("--no-timezones", action="store_true", help="Disable regional timezone diurnal sleep waves")
    p_run.add_argument("--no-flash-raids", action="store_true", help="Disable regional streamer flash-raids")
    p_run.add_argument("--no-burnout", action="store_true", help="Disable community burnout turnover")
    p_run.add_argument("--policy", choices=["auto", "neural", "stochastic"], default="auto", help="Pixel painting policy")
    p_run.add_argument("--neural-model", default="models/reddit_citizen.pt", help="Path to trained RedditCitizenNet weights")
    p_run.add_argument("--device", default="auto")
    p_run.add_argument("--run-dir", default=None)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--host", default="127.0.0.1")
    p_run.add_argument("--port", type=int, default=8765)
    p_run.add_argument("--headless", action="store_true", help="Run without web dashboard")
    p_run.set_defaults(func=_cmd_run)

    p_ren = sub.add_parser("render", help="Render a crisp 60fps MP4 timelapse from a run directory")
    p_ren.add_argument("run_dir", help="Run directory containing events.bin")
    p_ren.add_argument("--out", default=None, help="Output MP4 file path")
    p_ren.add_argument("--fps", type=int, default=60)
    p_ren.add_argument("--duration", type=float, default=60.0, help="Video duration in seconds (excluding end hold)")
    p_ren.add_argument("--scale", type=int, default=1, help="Nearest-neighbor upscale factor (1 on 2000x2000 = native 2K)")
    p_ren.add_argument("--hold", type=float, default=3.0, help="Final freeze-frame hold seconds")
    p_ren.add_argument("--no-hud", action="store_true", help="Disable bottom chronicle HUD bar")
    p_ren.add_argument("--no-dynamic-camera", action="store_true", help="Disable 4-act dynamic camera pull-back")
    p_ren.add_argument("--crf", type=int, default=18)
    p_ren.add_argument("--curve", type=float, default=1.35, help="Pacing exponent (>1.0 slows early chaos, 1.0=linear)")
    p_ren.set_defaults(func=_cmd_render)

    p_train = sub.add_parser("train-reddit", help="Download r/place 2022 logs and train RedditCitizenNet v2 on PyTorch MPS")
    p_train.add_argument("--data-dir", default="data/reddit_place")
    p_train.add_argument("--out", default="models/reddit_citizen.pt")
    p_train.add_argument("--epochs", type=int, default=8)
    p_train.add_argument("--batch-size", type=int, default=512)
    p_train.add_argument("--lr", type=float, default=2.5e-3)
    p_train.add_argument("--hidden-dim", type=int, default=80)
    p_train.add_argument("--patch-size", type=int, default=21)
    p_train.add_argument("--early-samples", type=int, default=320_000)
    p_train.add_argument("--peak-samples", type=int, default=160_000)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=_cmd_train_reddit)

    p_test = sub.add_parser("test-stack", help="Verify MPS GPU, local Qwen3-VL LM Studio API, and MLX FLUX Pixel-Art LoRA")
    p_test.add_argument("--llm-url", default="http://localhost:1234/v1")
    p_test.add_argument("--llm-model", default="qwen3-vl-30b-a3b-instruct-mlx")
    p_test.add_argument("--keychain-service", default="place-sim")
    p_test.add_argument("--no-flux-art", action="store_true")
    p_test.set_defaults(func=_cmd_test_stack)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
