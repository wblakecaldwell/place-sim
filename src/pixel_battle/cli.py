"""CLI entrypoints for running the pixel battle, testing local models, and rendering timelapses."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import uvicorn

from pixel_battle.config import SimConfig
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
    cfg = SimConfig(
        width=args.width,
        height=args.height,
        num_users=args.users,
        max_ticks=args.max_ticks,
        max_hours=args.max_hours,
        chaos_ticks=args.chaos_ticks,
        discovery_interval=args.discovery_interval,
        epoch_interval=args.epoch_interval,
        max_active_factions=args.max_factions,
        target_tps=args.tps,
        ollama_url=args.ollama_url,
        vlm_model=args.vlm_model,
        use_diffusion=not args.no_diffusion,
        use_flux_typography=not args.no_flux_typography,
        policy_mode=args.policy,
        neural_model_path=Path(args.neural_model),
        device=args.device,
        run_dir=run_dir,
        seed=args.seed,
        host=args.host,
        port=args.port,
    )

    engine = SimulationEngine(cfg, resume=args.resume)
    # Keep a convenient symlink runs/latest -> current run
    latest_link = Path("runs/latest")
    if run_dir.parent == Path("runs") and run_dir.name != "latest":
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name)
        except OSError:
            pass

    if cfg.use_diffusion and args.warmup_diffusion:
        print("Warming up local SDXL-Turbo + nerijs/pixel-art-xl on MPS...")
        engine.art.warmup()

    engine.start()
    print(f"\nPixel Battle running in: {run_dir}")
    print(f"Canvas: {cfg.width}x{cfg.height} | Citizens: {cfg.num_users:,} | Device: {engine.device} | Policy: {engine.policy_name}")
    print(f"Ollama ({cfg.vlm_model}): {'ONLINE' if engine.ollama_ready else 'OFFLINE (using fallback)'}")
    print(f"Art engine: {'Hybrid (SDXL-Turbo pixel-art img2img + Procedural)' if cfg.use_diffusion else 'Procedural only'}")

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
                tc = st.get("tier_counts", {})
                tier_str = f"M:{tc.get('macro',0)}/m:{tc.get('medium',0)}/u:{tc.get('micro',0)}/R:{tc.get('ribbon',0)}"
                print(
                    f"[{elapsed_m:5.1f}m | t={st['tick']:6d} / {lim}] "
                    f"{st['total_placements']:10,d} px | "
                    f"{st['pixels_per_sec']:6,d} px/s | "
                    f"users={u_tot:,} ({os_pct:.1f}% 1-shot) | "
                    f"{len(st['factions']):3d} fac ({tier_str}) | "
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
    print(f"Rendering timelapse from {run_dir} -> {out_path} ({args.fps} fps, {args.duration}s, scale={args.scale}x)...")
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
    from pixel_battle.palette import PALETTE_RGB

    dev = resolve_device("auto")
    print(f"1. PyTorch device: {dev}")

    out_dir = Path("runs/stack_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test Ollama qwen2.5vl:7b vision structured output
    brain = LocalVLMBrain(vlm_model=args.vlm_model)
    healthy = brain.check_health()
    print(f"2. Ollama ({args.vlm_model}) health: {'OK' if healthy else 'NOT FOUND'}")
    if healthy:
        # Feed a synthetic messy green/yellow blob for pareidolia test
        arr = np.full((56, 56, 3), 255, dtype=np.uint8)
        arr[16:40, 14:42] = PALETTE_RGB[9]   # green
        arr[12:20, 22:34] = PALETTE_RGB[6]   # yellow crown-ish blob
        arr[22:26, 20:24] = PALETTE_RGB[1]   # eyes
        arr[22:26, 32:36] = PALETTE_RGB[1]
        blob_img = Image.fromarray(arr).resize((336, 336), Image.Resampling.NEAREST)
        t0 = time.perf_counter()
        prop, theme = brain.pareidolia_discover(blob_img, ["green", "yellow", "black"], [], [])
        print(f"   Pareidolia ({time.perf_counter()-t0:.2f}s): {prop.name} [{prop.art_mode} | {prop.banner_text}] ({theme}) -> \"{prop.art_prompt}\"")
    brain.close()

    # Test Hybrid Art (procedural + optional diffusion txt2img & img2img)
    cfg = SimConfig(use_diffusion=not args.no_diffusion)
    art = ArtManager(cfg, dev)
    t0 = time.perf_counter()
    proc_bp, _ = art.create_blueprint("procedural", "regal emerald castle shield", 64, 64, 9, 6, 0, "TEST")
    proc_img = Image.fromarray(PALETTE_RGB[np.where(proc_bp < 0, 0, proc_bp)]).resize((256, 256), Image.Resampling.NEAREST)
    proc_img.save(out_dir / "procedural_sample.png")
    print(f"3. Procedural art ({time.perf_counter()-t0:.3f}s) saved -> {out_dir / 'procedural_sample.png'}")

    if not args.no_diffusion:
        print("4. Loading SDXL-Turbo + nerijs/pixel-art-xl (first run downloads weights from Hugging Face)...")
        t0 = time.perf_counter()
        diff_bp, _ = art.create_blueprint("diffusion", "cute green pixel frog wearing golden crown", 64, 64, 9, 6, 0, "FROG")
        diff_rgb = PALETTE_RGB[np.where(diff_bp < 0, 0, diff_bp)]
        Image.fromarray(diff_rgb).resize((256, 256), Image.Resampling.NEAREST).save(out_dir / "diffusion_txt2img.png")
        print(f"   txt2img ({time.perf_counter()-t0:.2f}s) saved -> {out_dir / 'diffusion_txt2img.png'}")

        # Punch a Void scar on the right half and test img2img morphing
        scarred = diff_rgb.copy()
        scarred[10:50, 42:62] = PALETTE_RGB[1]
        t0 = time.perf_counter()
        morph_bp = art.morph_blueprint(
            "diffusion",
            "royal pixel frog king with glowing dark shadow portal on right side",
            scarred,
            64,
            64,
            9,
            6,
            21,
            stage=2,
            strength=0.82,
        )
        morph_rgb = PALETTE_RGB[np.where(morph_bp < 0, 0, morph_bp)]
        Image.fromarray(morph_rgb).resize((256, 256), Image.Resampling.NEAREST).save(out_dir / "diffusion_morph.png")
        print(f"   img2img morph ({time.perf_counter()-t0:.2f}s) saved -> {out_dir / 'diffusion_morph.png'}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pixel-battle", description="Local AI/ML r/place simulation")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Start the live pixel battle simulation")
    p_run.add_argument("--width", type=int, default=512)
    p_run.add_argument("--height", type=int, default=512)
    p_run.add_argument("--users", type=int, default=36_000)
    p_run.add_argument("--max-ticks", type=int, default=8_000)
    p_run.add_argument("--max-hours", type=float, default=0.0, help="If > 0, run for this many wall-clock hours cleanly")
    p_run.add_argument("--chaos-ticks", type=int, default=140)
    p_run.add_argument("--discovery-interval", type=int, default=45)
    p_run.add_argument("--epoch-interval", type=int, default=110)
    p_run.add_argument("--max-factions", type=int, default=200)
    p_run.add_argument("--tps", type=float, default=18.0, help="Target ticks/sec (0 = uncapped)")
    p_run.add_argument("--ollama-url", default="http://localhost:11434")
    p_run.add_argument("--vlm-model", default="qwen2.5vl:7b")
    p_run.add_argument("--no-diffusion", action="store_true", help="Disable SDXL-Turbo diffusion (procedural only)")
    p_run.add_argument("--no-flux-typography", action="store_true", help="Disable 4-bit FLUX.1-schnell MLX wordmarks")
    p_run.add_argument("--warmup-diffusion", action="store_true", help="Load diffusion model before starting ticks")
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
    p_ren.add_argument("--scale", type=int, default=2, help="Nearest-neighbor upscale factor (2 => 1024x1024)")
    p_ren.add_argument("--hold", type=float, default=3.0, help="Final freeze-frame hold seconds")
    p_ren.add_argument("--no-hud", action="store_true", help="Disable bottom chronicle HUD bar")
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

    p_test = sub.add_parser("test-stack", help="Verify MPS GPU, Ollama Qwen2.5-VL, procedural art, and pixel-art diffusion")
    p_test.add_argument("--vlm-model", default="qwen2.5vl:7b")
    p_test.add_argument("--no-diffusion", action="store_true")
    p_test.set_defaults(func=_cmd_test_stack)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
