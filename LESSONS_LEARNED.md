# Lessons Learned — `place-sim` (`pixel-battle`)

Engineering, visual, and ML post-mortems captured across early local simulation runs on Apple Silicon.

---

## What Worked Well

1. **Local 4-bit MLX `FLUX.1-schnell` + Pixel-Art LoRA (`pixelart_schnell_v1.safetensors`)**
   - Generates expressive, one-of-a-kind pixel art (characters, mascots, food, comics, storefronts, vehicles, landmarks) with clean fluid non-rectangular cutouts (`-1` transparent exterior) quantized strictly to the active r/place palette via CIE-LAB.
2. **Local `Qwen3-VL-30B-A3B` via LM Studio with Structured JSON Schema**
   - Switching faction concept batches from free-form text to strict JSON schema eliminated malformed responses and produced diverse multilingual/international communities cleanly.
   - Reading the local API Bearer token from macOS Keychain (`place-sim`) or `LMSTUDIO_API_KEY` keeps secrets completely out of tracked repository files.
3. **Trained MPS Citizen Policy (`RedditCitizenNet`) + Real Session Turnover**
   - Behavioral cloning on official Reddit r/place 2022 placement logs achieved `98.6%` blueprint-free defender accuracy (`def_free@1`) and `100.0%` goal color accuracy (`goal_col@1`).
   - Single-pass cross-faction GPU batching sustains `10,000–15,000 px/s` on Apple Silicon MPS while simulating ~15.5 mean pixels/user and `20%` one-shot citizens.
4. **Exact 4K Nearest-Neighbor Upscaling for YouTube Shorts**
   - Rendering a fixed 9:16 pixel grid (`720x1280` at `3x` or `540x960` at `4x`) directly to `2160x3840` @ `60 fps` MP4 forces YouTube's high-bitrate 2160p60 VP9/AV1 pipeline so pixel art stays razor-sharp instead of blurring under 1080p H.264 compression.
5. **Fast Macro Community Planner (`pixel-battle plan`)**
   - Running a coarse-grid 72-virtual-hour macro simulation in `< 1s` catches coverage, burnout, and recruitment balance bugs before spending hours on a full render.
6. **Frame-0 Poster Hold for GitHub README GIFs**
   - Placing the completed canvas on Frame 0 (with seamless loop from the end frame) guarantees GitHub's paused GIF preview displays the finished artwork instead of a blank white frame.

---

## What Went Wrong & Root Causes

### 1. Wall-Clock Async Execution Degraded Quality Under Load
- **Symptom:** Running the pixel loop on a wall-clock timer (`--max-hours`) while AI brainstorming and FLUX art ran asynchronously in a background thread meant any GPU slowdown directly degraded output: early runs had `0` factions through tick `455` and only `6` factions through tick `2,053`, then resorted to low-quality fallback templates to catch up.
- **Root Cause:** Hardware speed was coupled to simulation logic.
- **Fix:** Offline/video runs must use **deterministic, step-locked execution (`--sync`)** driven strictly by simulated ticks (`--max-ticks`). All scheduled concept creation, art forging, and epoch turns must finish before advancing to `tick + 1`. Hardware speed should only change render duration, never visual quality.

### 2. Apple Silicon Unified Memory & Metal Swap Thrashing
- **Symptom:** FLUX generation speed swung wildly between **`6–7s` per sprite** (`1.5s/it`) and **`100–240s` per sprite** (`60s/it`).
- **Root Causes:**
  1. Running PyTorch MPS + LM Studio `Qwen3-VL` (~18 GB) + MLX `FLUX.1-schnell` (~12 GB) concurrently exceeded comfortable wired GPU memory on a 48 GB machine, causing macOS to page FLUX weights to SSD swap.
  2. Varying FLUX generation dimensions (`512x512`, `384x576`, `576x384`, `640x320`) triggered ~45s MLX Metal shader recompiles mid-run.
  3. LM Studio's default 10-minute idle TTL auto-reloaded the model at `8192` context instead of lean `4096`, doubling KV-cache memory pressure mid-run.
- **Fixes:**
  - Lock FLUX generation to a **single fixed `512x512` MLX shape** and center-crop to target aspect ratio before quantization.
  - Decouple heavy GPU stages via `pixel-battle forge-bank`: run **Stage 1 (Qwen3-VL solo)** -> unload LLM -> **Stage 2 (MLX FLUX solo)** -> **Stage 3 (PyTorch MPS battle solo)**. Zero swap thrashing, ~5–7s per FLUX piece, and 100% deterministic playback.

### 3. Unorganized Free Agents Carpeted Empty White Space in Noise
- **Symptom:** Unclaimed background areas filled up with orange/teal/black TV-static confetti before factions could claim them.
- **Root Causes:**
  1. Free agents painting stray pixels outward from faction edges created new "painted" pixels that attracted more free agents, spreading noise outward like mold.
  2. Completed factions converted idle builders into `_expand_perimeter` expanders that sprayed random primary/secondary halo dust outside their silhouettes.
- **Fix:** Real r/place participants do not carpet blank canvas with random static. Keep untouched empty space pure white (`#FFFFFF`), stop perimeter halo splatter on completed art, and direct chaotic players strictly at **active factions** via random vandalism inside artworks and small-scale coordinated **Chaos Squads** (30–150 players swarming a tight patch for 50–100 ticks).

### 4. Generic Fallback Spam ("Shields & Sierpinski Clones")
- **Symptom:** In an 8-hour run of `4,551` founded factions, only `889` (`19%`) were unique FLUX art; hundreds of identical quartered star-shields, dozens of identical Sierpinski triangles, and ~28 copies of each micro-sprite appeared side-by-side.
- **Root Cause:** Throttling FLUX per wave pushed overflow spawns into `procedural.py` (9 motifs defaulting to a quartered shield), `fractal` (4 shapes), and `micro_sprite` (34 icons) without lifetime uniqueness caps.
- **Fix:** Retire generic procedural shields/medallions/spirals as bulk fallbacks, enforce `<= 1` lifetime uniqueness for every flag, fractal, comic, meme, painting, and micro-sprite, and supply the vast majority of factions (`~70–80%`) from unique FLUX artworks + unique multilingual wordmarks.

### 5. Slow Start ("Empty Site Nobody Knew About") & Excessive Burnout
- **Symptom:** Early minutes looked sparsely populated; later, `2,115 / 4,551` factions collapsed due to premature burnout, continuously punching holes in dense territory.
- **Fix:** Add a **Day-0 Launch Wave** (`--day0-factions`) where prepped communities storm the canvas during Ticks `1–50` (mirroring real r/place Discord/subreddit launches), and gate burnout to high canvas coverage (`>= 85–88%`) with longer virtual-hour staying power.

### 6. Canvas Expansions vs. Short-Form Vertical Video
- **Symptom:** 4-Act dynamic camera pull-back (`540x960 -> 1080x960 -> 1080x1920`) works well for long desktop Reddit 2022 replays, but in a 58-second mobile YouTube Short it zooms out too far to follow individual pixel battles.
- **Fix:** Keep YouTube Shorts on a **fixed 9:16 canvas** (`720x1280` @ `3x` or `540x960` @ `4x` -> 4K `2160x3840`) with `--no-expansion` and `--no-whiteout`.

### 7. Additional Technical Bugs Fixed Along the Way
- **All-black neural citizen v1:** Oversampling Void placements and applying `FREE` dropout to `VOID` biased `RedditCitizenNet` toward black. Fixed by preserving natural r/place color balance.
- **Recruitment pinned at ~14% organized share:** ~8,500 organized sessions expired every 20 ticks while `social_step` capped recruitment at `1,500`. Fixed by scaling recruitment budget and member caps dynamically to track `target_organized_share`.
- **Top-edge spawn piling:** `find_open_exodus_spot` broke zero-occupancy ties deterministically at `(0, 0)`, and overly large corner/edge bonuses left hollow interiors. Fixed with randomized candidate sampling and tight neighbor-packing (`find_neighbor_spawn_spot`).
- **LLM `ai_code` `NameError`:** Generated Python scripts referenced color names (`DARK_BROWN`, etc.) not present in sandbox globals. Fixed by injecting all palette color constants into the AST sandbox.

---

### 8. Pixel-Grid Area vs. Video Upscale Resolution & the `512x512` Sweet Spot
- **Symptom:** The early `512x512` square run (`battle_comics_memes`) looked densely packed, crisp, and alive, while `720x1280` (`921,600 px`, `3.5x` larger) left `54%` pure white space and even `540x960` (`518,400 px`, `2.0x` larger) looked under-built (`31 px/px` vs `337 px/px` on `512x512`).
- **Root Cause:** Confusing video export resolution (`4K = 2160x3840`) with simulation pixel-grid resolution. Doubling or tripling grid area while shrinking sprites to `0.85x` and cutting total ticks to `6,000` gave each faction `14x` fewer pixel placements (`~30k` vs `~418k` in `512x512`).
- **Fix:** Match the `512x512` (`262k px`) sweet spot in 9:16 using **`360 x 640` (`230,400 px`)** upscaled **`6x` exact nearest-neighbor -> 4K (`2160 x 3840` @ 60fps)**, keep bold **`1.0x`** artwork sizing, and run **`12,000` ticks** so every faction gets deep pixel density (`>= 120–160 px/canvas-px`) to finish, defend, and evolve crisply.

### 9. Withholding Factions Across 85% of the Run Left Half-White Mid-Videos
- **Symptom:** Spreading `target_total_factions` linearly across `85%` of total ticks (`progress / 0.85`) withheld ~450 factions until mid/late video, so the canvas still had huge white gaps 10–25 seconds into a 58-second Short.
- **Root Cause:** Real r/place communities do not trickle in smoothly over 72 hours — prepped groups rush and claim almost the entire canvas inside the **first simulated hour**, after which the battle is about defending borders, filling tiny cracks, expanding, fighting rivals, and conquering collapsed turf.
- **Fix:** Front-load the initial land grab so `~220–240` factions land inside the **first ~4–5% of ticks** (`Ticks 1–480`, simulated Hours `0–3`), eliminating empty white background early while keeping replacement waves ready whenever older factions burn out or fall.

### 10. People Love Canvas Corners & Edges First
- **Symptom:** Replacing corner-first seeding with purely interior anchor hubs ignored one of the most iconic human behaviors on r/place: communities fiercely race for the **4 corners and 4 outer walls** because edges only require defending 2–3 sides instead of 4.
- **Fix:** In `_pick_spawn_slot`, explicitly seed the **4 corners (0px flush)** and **all 4 walls (0px flush)** during the very first opening waves alongside interior anchor hubs, and snap nearby placements flush against borders (`_snap_to_walls`).

### 11. White Sticker Bubbles, Diagonal Ribbon Ghost Stripes, & Raid Confetti Speckle
- **Symptom A (White Sticker Bubbles):** ~70 FLUX sprites rendered inside chunky black-bordered white ovals. **Root Cause:** `diffusion.py`'s flood-strip guard (`> 0.82`) aborted stripping whenever a sprite occupied `< 18%` of the frame. **Fix:** Raised guard to `> 0.95` and added `strip_white_sticker_bubble()` (stripping pure `WHITE_IDX` only, keeping interior grays/whites intact, allowing at most ~2% intentional ovals).
- **Symptom B (Diagonal Ribbon Ghost Stripes & Concentric Rings):** Growing a `d2` diagonal ribbon shifted its `ny0` offset each growth step (painting parallel candy-cane stripes across the board), and expander perimeter rings drew giant concentric circles around diagonal ribbon bounding boxes. **Fix:** Keep ribbons strictly horizontal (`h`) or vertical (`v`) and disable circular perimeter expansion on ribbons.
- **Symptom C (Raid "Confetti" Speckle):** Allowing `455` simultaneous wars where `_step_faction_expanders` sprays random single pixels across entire enemy bounding boxes (including transparent corners) turned contested mid-board art into speckled static. **Fix:** Cap concurrent active wars (`<= 16–20` focused frontlines, prioritizing AI-curated **Natural Rival Pairs**), restrict raids strictly to **opaque enemy pixels (`bp >= 0`) closest to the attacker's shared border** (creating a clean, solid advancing front wave instead of random bbox confetti), and re-enable autonomous gap-closing `EXPAND` / `FORTIFY` during epochs.

### 12. Zero Python-Rendered Word Boxes or Templated Niche Tiles
- **Symptom:** Procedural niche-carver templates (`mini_tag` solid-colored rectangles with a bitmap word inside, plus repeated diamond/stripe tiles) created dozens of generic, artificial-looking Windows-Metro-style boxes across the canvas.
- **Root Cause:** Any Python-rendered bitmap text box or procedural geometric tile repeated across dozens of small factions immediately breaks the illusion of organic human communities.
- **Fix:** **Python must build zero word images or templated box tiles** (`REQ-ART-018`). All word art must originate from local AI image generation (`FLUX`), `ArtBank(visual_only=True)` excludes Python-only `banner` entries, and whitespace niche hunters (`find_whitespace_niche`) fit **real, unique AI-generated FLUX artworks (`pop_best_fit`) or `<= 1`-copy pixel sprites** directly into remaining white pockets.

### 13. Center-Cropping Wide Comics Across `512x512` 2-Row Comic Pages
- **Symptom:** Wide `80x36` FLUX comics (`ComicCats`, `PixelPunch`) appeared "wrapped around" — showing the bottom half/legs of a top panel above a middle gutter and the top half/heads of a bottom panel underneath.
- **Root Cause:** Locking FLUX to `512x512` caused it to draw 2-row comic grids with a horizontal gutter right across `y = 256`. Center-cropping `y = 141..371` for a `2.2:1` comic strip sliced right through the gutter, cutting both rows in half.
- **Fix:** Size `flux_comic` entries near-square (`62x54` to `84x72`) in `bank.py` so 2x2 comic grids stay uncropped, and in `flux_art.py` crop wide non-cutout frames (`ratio >= 1.55`) from `top = 0` (complete top panel row) instead of straddling the middle gutter.
