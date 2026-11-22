# Pixel Battle System Requirements Specification

**Document Standard:** ASD-STE100 (Simplified Technical English)
**Project:** `pixel-battle`
**Revision:** 1.3

---

## 1. Purpose and Scope

This document defines the functional, visual, machine-learning, and operational requirements for `pixel-battle`.
The system simulates a multi-agent pixel Canvas similar to Reddit `r/place`.
The visual density and artwork variety target public Reddit `r/place` 2022 canvas metrics.
All artificial intelligence models run locally on Apple Silicon hardware.

---

## 2. Technical Names (Approved Project Terminology)

Use these exact Technical Names in this specification and in related project documents.

| Technical Name | Definition |
| :--- | :--- |
| **Black Hole** | A circular Void anomaly with a dark core, spiral bands, and dark tendrils. |
| **Blueprint** | A 2D target grid of palette indices (`0` to `31`) or transparent cells (`-1`) for one Faction. |
| **Canvas** | The live 2D pixel grid that Citizens change on each Tick. |
| **Chronicle** | The append-only JSON Lines file (`chronicle.jsonl`) that records Faction and diplomacy events. |
| **Citizen** | One simulated user with a position, a cooldown timer, a loyalty value, and a Faction assignment. |
| **Comic Strip** | A multi-panel pixel Blueprint with black panel borders, white gutters, cartoon figures, and speech bubbles. |
| **Cutout** | A non-rectangular Blueprint whose outer background cells have the transparent value (`-1`). |
| **Empty-Space Faction** | A small Faction whose Blueprint sets pixels to pure white (`0`) to defend blank area. |
| **Epoch** | One strategy cycle in which active Faction leaders examine the Canvas and select actions. |
| **Event Log** | The append-only binary file (`events.bin`) that stores every pixel placement. |
| **Faction** | A group of Citizens that share one Blueprint, one territory, and one diplomacy state. |
| **Fine-Art Painting** | A small or medium pixel rendition of a classic painting inside a gold pixel frame. |
| **Fractal** | A recursive mathematical pixel pattern such as a Sierpinski triangle or Mandelbrot set. |
| **Headless Mode** | An execution mode that runs the simulation without a web dashboard or browser window. |
| **Hype** | A numeric vitality score for a Faction that controls Citizen recruitment and retention. |
| **Macro Faction** | A large Faction whose Blueprint spans `60` to `128` pixels or crosses the full Canvas. |
| **Medium Faction** | A mid-size Faction whose Blueprint spans `30` to `59` pixels on its longer side. |
| **Micro Faction** | A small Faction whose Blueprint spans `12` to `29` pixels on its longer side. |
| **National Flag** | A Blueprint that represents a country flag with accurate stripe and emblem geometry. |
| **One-Shot Citizen** | A Citizen that places `1` pixel during its session and then leaves the Canvas. |
| **Pareidolia** | The process in which a vision model examines an unclaimed pixel cluster and founds a Faction. |
| **Policy** | The GPU module that computes pixel placements for ready Citizens on each Tick. |
| **Rainbow Road** | A curved, multi-color stripe Blueprint that spans a wide area of the Canvas. |
| **Tick** | One discrete simulation step that updates Citizen cooldowns and applies pixel placements. |
| **Transverse Ribbon** | A straight horizontal, vertical, or diagonal multi-color stripe band that extends across the Canvas. |
| **The Void** | A hostile Faction that spreads dark tendrils, Black Holes, Fractals, or abyssal faces into nearby art. |
| **Wordmark** | A stylized pixel-art text emblem or logo banner for one Faction. |

---

## 3. System and Runtime Requirements (`REQ-SYS`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-SYS-001** | The system must run all models locally with zero cloud API calls and zero external API cost. | Block network access after local model cache load. Run the system and make sure that zero external requests occur. | Implemented |
| **REQ-SYS-002** | The system must run on Apple Silicon macOS with PyTorch MPS and Apple MLX below 40 GB peak memory. | Run all 4 local models on Apple Silicon macOS. Make sure that peak memory stays below 40 GB. | Implemented |
| **REQ-SYS-003** | The CLI must accept a Tick limit (`--max-ticks`) and a wall-clock hour limit (`--max-hours`). | Run with `--max-hours 0.04`. Make sure that the process stops normally at the target time. | Implemented |
| **REQ-SYS-004** | The engine must save Canvas, Blueprint, and Faction checkpoints at each checkpoint interval and on `SIGINT`. | Stop a live run with `SIGINT`. Resume with `--resume` and make sure that Tick numbers continue in sequence. | Implemented |
| **REQ-SYS-005** | The batch script must prevent macOS idle sleep with `caffeinate -i` and release unused MPS memory every 250 Ticks. | Run unattended for 6 hours or more. Make sure that Tick counts increase without pause. | Implemented |
| **REQ-SYS-006** | In uncapped mode (`--tps 0`), the loop must pause Ticks at Epoch boundaries until model calls finish. | Run with `--tps 0`. Make sure that every scheduled Pareidolia turn and Epoch turn finishes on schedule. | Implemented |

---

## 4. Canvas and Citizen Simulation Requirements (`REQ-SIM`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-SIM-001** | The Canvas must default to `512 x 512` pixels and must accept custom width and height values. | Start the simulation with default flags. Make sure that the Canvas size is `512 x 512`. | Implemented |
| **REQ-SIM-002** | The Canvas must use the official 32-color `r/place` 2022 palette with index `0` set to pure white (`#FFFFFF`). | Run `test_palette_roundtrip()`. Make sure that all 32 palette colors match after CIE-LAB conversion. | Verified |
| **REQ-SIM-003** | The Canvas must store current color, previous color (`prev_colors`), owner Faction, edit Tick, and activity for each pixel. | Change one pixel color twice. Make sure that `prev_colors` keeps the color from before the latest change. | Implemented |
| **REQ-SIM-004** | The Policy must simulate at least 20,000 Citizens in vectorized GPU batches with individual cooldowns and loyalties. | Run 100 Ticks with 24,000 Citizens. Make sure that GPU batch operations run with zero per-Citizen Python loops. | Implemented |
| **REQ-SIM-005** | Each new run must start from a blank white Canvas with zero Factions and progress through chaos, nucleation, alliances, and wars. | Examine Canvas images at Tick 0, Tick 100, and Tick 500. Make sure that each phase occurs in order. | Implemented |
| **REQ-SIM-006** | Citizens and Factions must reach Canvas corners and edges directly up to coordinates `0` and `512` with zero forced border margin. | Measure pixel coverage on all 4 Canvas edges at Hour 1. Make sure that each edge has `>= 50%` coverage. | Implemented |
| **REQ-SIM-007** | Early Pareidolia scans must check the 4 Canvas corners first, then balance walls and interior space after 10 edge Factions exist. | Examine the first 16 founded Factions. Make sure that all 4 corners and interior regions contain Factions. | Implemented |
| **REQ-SIM-008** | Long runs must decrease Hype on Factions older than 2,800 Ticks so new Factions and wars continue over many hours. | Examine `chronicle.jsonl` from a multi-hour run. Make sure that old Factions decline and new Factions form. | Implemented |
| **REQ-SIM-009** | The engine must support `120` to `240` simultaneous active Factions on a `512 x 512` Canvas to match `placeAtlas` density. | Run a multi-hour simulation and examine `state.json`. Make sure that at least `120` active Factions coexist. | Implemented |
| **REQ-SIM-010** | Active Factions must match `placeAtlas` shares: `55%` to `65%` Micro, `20%` to `28%` Medium, `10%` to `15%` Macro, and `2%` to `6%` Transverse Ribbons. `EXPAND` must apply per-tier size limits. | Measure active Blueprint sizes after Hour 1. Make sure that median size stays near `24 px` and limits hold. | Implemented |
| **REQ-SIM-011** | Initial Citizen recruitment must scale in proportion to Blueprint pixel area, with at least `32,000` concurrent online Citizen slots. | Check member counts across Macro, Medium, and Micro Factions. Make sure that Micro Factions start with `40` to `140` Citizens. | Implemented |
| **REQ-SIM-012** | Pareidolia scans must detect small unclaimed gaps (`12` to `28` pixels). Discovery must allow cameo Micro Factions on wide flags or ribbons. | Examine a mature Canvas image. Make sure that small sprites fill gaps and appear on wide ribbons or flags. | Implemented |
| **REQ-SIM-013** | Population targets must scale from `r/place` 2022 (`10,000,000` users, `2,000,000` One-Shot Citizens) by area ratio `S = (W * H) / 4,000,000`. | Compute targets on `512 x 512` (`S = 0.065536`). Make sure that targets equal `655,360` users and `131,072` One-Shot Citizens. | Implemented |
| **REQ-SIM-014** | Each GPU Citizen slot must track a unique `user_id` and click budget, then cycle to a new `user_id` at zero. New sessions must sample `20%` One-Shot (`1`), `50%` casual (`2` to `12`), `24%` regular (`13` to `70`), and `6%` veteran (`71` to `350`). | Count placements for each `user_id` in `events.bin`. Make sure that One-Shot Citizens equal `20.0% ± 1.0%` of finished sessions. | Verified |
| **REQ-SIM-015** | Community theme selection must sample active non-NSFW subreddits from `data/subreddits/` by subscriber rank to match Reddit category variety. | Examine 40 founded Factions. Make sure that themes span flags, games, anime, memes, fine art, and logos with zero duplicates. | Verified |

---

## 5. Trained Local Citizen Model Requirements (`REQ-ML`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-ML-001** | The engine must separate pixel placement behind the `PixelPolicy` protocol so neural and stochastic Policies swap without engine edits. | Run both `--policy neural` and `--policy stochastic` with the same engine code. | Implemented |
| **REQ-ML-002** | The dataset builder must train on official Reddit `r/place` 2022 logs from Day 1 Hours 0 to 12 and Day 3 peak activity. | Check that 8 chronological chunks (`10, 06, 09, 14, 15, 17, 36, 37`) produce `>= 400,000` samples. | Implemented |
| **REQ-ML-003** | `RedditCitizenNet` must accept a `21 x 21` local patch with current colors, previous colors, recency, optional hindsight goal, and archetype. | Examine checkpoint metadata and run `test_neural_policy_inference()`. | Verified |
| **REQ-ML-004** | `RedditCitizenNet` must predict click position across `441` cells and then predict color across `32` palette classes at that cell. | Examine the model forward pass and check validation metrics (`pos@5 >= 40%`). | Implemented |
| **REQ-ML-005** | Dataset selection must keep natural Reddit color balance with zero extra Void samples and must not apply `FREE` dropout to `VOID`. | Run the 5-archetype color probe on a blue-green patch. Make sure that `BUILDER` black probability is `<= 0.15`. | Implemented |
| **REQ-ML-006** | The trained model must reach `>= 90%` color accuracy for Blueprint-free defenders (`def_free@1`) and `>= 98%` goal color accuracy (`goal_col@1`). | Read saved metrics in `models/reddit_citizen.pt` (`def_free@1 = 98.6%`, `goal_col@1 = 100.0%`). | Verified |
| **REQ-ML-007** | Live neural inference on a `512 x 512` Canvas must maintain `>= 8,000` pixel placements per second on Apple Silicon MPS. | Run a 120-Tick benchmark with `RedditCitizenNet v2`. Make sure that throughput stays `>= 8,000` pixels per second. | Verified |
| **REQ-ML-008** | `NeuralRedditPolicy` must batch ready builders and defenders from all active Factions into one combined GPU forward pass on each Tick. | Benchmark 200 active Factions on Apple Silicon MPS. Make sure that throughput stays `>= 8,000` pixels per second. | Verified |

---

## 6. Faction Intelligence and Diplomacy Requirements (`REQ-DIP`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-DIP-001** | The strategic brain must run local `qwen2.5vl:7b` through Ollama with strict JSON schema checks and deterministic fallback on error. | Run `pixel-battle test-stack`. Make sure that Ollama returns HTTP 200 and schema-valid JSON. | Implemented |
| **REQ-DIP-002** | Pareidolia discovery must send an upscaled PNG crop of an unclaimed cluster and must block prohibited repeated names (`Radiant`, `Whirl`, `Cosmic`). | Examine 20 founded Faction names. Make sure that zero prohibited words and zero exact duplicates appear. | Implemented |
| **REQ-DIP-003** | Each Epoch turn must show the Faction leader an annotated PNG of its territory with labeled neighbors (`YOU`, `ALLY`, `VOID`, neutral). | Examine `render_faction_view_pil()` output. Make sure that `FactionDecision` JSON passes validation. | Implemented |
| **REQ-DIP-004** | The engine must execute `BUILD`, `EXPAND`, `MORPH_ART`, `FORTIFY`, `RAID`, `PROPOSE_ALLIANCE`, `ACCEPT_ALLIANCE`, and `EXODUS`. | Examine `chronicle.jsonl` after a multi-hour run. Make sure that diplomacy, expansion, and war events occur. | Implemented |
| **REQ-DIP-005** | Allied neighbors must place pink and red border hearts along shared seams and must send spare defenders when an ally loses territory. | Examine the final Canvas image and the ally defense routine in `policies/stochastic.py`. | Implemented |
| **REQ-DIP-006** | The Void must appear at intervals, recruit free and low-loyalty Citizens, spread with the `VOID` archetype, and stop after its time limit. | Examine `chronicle.jsonl` for Void start and end events with `>= 500` shadow Citizens. | Implemented |
| **REQ-DIP-007** | Adjacent National Flag Factions must compete through territorial pushes and raids, and must also form border-heart treaties when diplomacy succeeds. | Examine `chronicle.jsonl` from a run with `>= 8` National Flags. Make sure that flag wars and treaties both occur. | Implemented |
| **REQ-DIP-008** | Discovery waves must spawn multiple Factions at once. Micro Factions must cluster into dense multi-sprite alliances without vision-model bottlenecks. | Time one discovery wave and one Epoch with `>= 80` active Factions. Make sure that Epoch time stays `<= 25` seconds. | Implemented |
| **REQ-DIP-009** | Transverse Ribbon Factions must advance their stripe tip across the Canvas, cut through rivals, and yield to allied art. | Examine timelapse frames of a Transverse Ribbon. Make sure that the stripe grows across the Canvas and interacts with neighbors. | Implemented |
| **REQ-DIP-010** | The Void subsystem must rotate across Tendril Void, Black Hole, Fractal, and Void Monster variants so dark events vary across long runs. | Examine `chronicle.jsonl` and snapshots across a multi-hour run. Make sure that multiple Void variants appear. | Implemented |

---

## 7. Visual Art, Organic Shapes, and Color Diversity Requirements (`REQ-ART`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-ART-001** | The system must rotate at least 12 distinct signature color themes across new Factions so all 32 palette colors appear on the Canvas. | Check the Canvas color histogram. Make sure that pinks, purples, blues, reds, greens, and yellows appear. | Implemented |
| **REQ-ART-002** | Diffusion sprites must use `SDXL-Turbo` with `nerijs/pixel-art-xl`, CIE-LAB quantization, border background removal, and a 1-pixel black outline. | Examine generated sprite PNGs. Make sure that pixel edges stay sharp and outer background cells equal `-1`. | Implemented |
| **REQ-ART-003** | Non-flag Factions must use non-rectangular silhouettes such as curved corner floods, Rainbow Roads, free-form hearts, medallions, wavy lattices, and Cutout sprites. | Examine live Canvas frames. Make sure that rectangular gray boxes do not dominate. | Implemented |
| **REQ-ART-004** | Expander Citizens must grow from the active silhouette frontier and must not draw dotted rectangular borders. | Examine `_expand_perimeter()` output on a live Canvas. Make sure that zero dotted rectangular frames appear. | Implemented |
| **REQ-ART-005** | Occupancy checks must mark only non-transparent silhouette pixels (`blueprint >= 0`) so curved artworks fit closely together. | Examine the occupancy mask in `detector.py`. Check curved neighbor placement on the Canvas. | Implemented |
| **REQ-ART-006** | Procedural Factions must stay procedural across all stages. Diffusion img2img morphs must apply a `>= 280` Tick cooldown and a Stage `<= 5` limit. | Check `manager.py` for the procedural lock. Check `engine.py` for cooldown and stage limits. | Implemented |
| **REQ-ART-007** | `FORTIFY` must add one outer border along a Cutout contour and must not add extra rings on repeated fortify turns. | Run `FORTIFY` twice on one Faction. Make sure that the outer border applies only once. | Implemented |
| **REQ-ART-008** | The art engine must provide a procedural library of at least 24 recognizable National Flags that stay sharp at Macro, Medium, and Micro scales. | Generate a preview sheet of all 32 National Flags at 3 scales. Make sure that stripes, crosses, and emblems stay clear. | Verified |
| **REQ-ART-009** | The art engine must generate memes, cartoons, anime faces, and retro sprites with 1-pixel outlines. Sizes must cover Micro (`14` to `26` px) and Medium (`32` to `56` px). | Generate a preview grid of 24 Micro and Medium sprites. Make sure that small sprites stay sharp and clear. | Verified |
| **REQ-ART-010** | The art engine must generate small (`24 x 20` to `36 x 30`) and medium (`44 x 36` to `68 x 52`) Fine-Art Paintings. Each painting must include a gold pixel frame. | Generate a gallery preview of 6 Fine-Art Paintings at small and medium sizes. Make sure that subjects and frames stay clear. | Verified |
| **REQ-ART-011** | The art engine must generate pure-white (`0`) Empty-Space Faction Blueprints (`16 x 16` to `40 x 32`). Blueprint validation must accept pure-white grids. | Place an Empty-Space Faction on a colored patch. Make sure that Citizens restore colored pixels to pure white. | Verified |
| **REQ-ART-012** | The art engine must generate horizontal, vertical, and diagonal Transverse Ribbon Blueprints (`5` to `15` pixels thick). Ribbons must support solid, flag, rainbow, and pattern fills. | Generate horizontal, vertical, and diagonal Transverse Ribbons. Make sure that patterns tile accurately across 512 pixels. | Verified |
| **REQ-ART-013** | The art engine must generate Black Hole Blueprints, Fractal Blueprints (`sierpinski`, `mandelbrot`, `carpet`, `diamond`), and Void Monster faces. | Generate preview PNGs of Black Holes, Fractals, and Void Monsters at multiple scales. Make sure that geometry stays sharp. | Verified |
| **REQ-ART-014** | The art engine must generate multi-panel Comic Strips and pixel cartoon or meme portraits. Active National Flags must stay `<= 18` total (`<= 3` Macro). | Generate previews of all 12 Comic Strips and 14 cartoon portraits. Check that active National Flags stay within limits. | Verified |

---

## 8. Model-Designed Typography Requirements (`REQ-TYP`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-TYP-001** | A local AI model must design the typography of each Faction and must not apply one uniform bitmap font across the Canvas. | Examine `design_typography()` outputs for 6 Factions. Make sure that fonts, colors, and layouts differ. | Implemented |
| **REQ-TYP-002** | Between 30% and 40% of non-banner murals must set `include_text = false` so pure pixel art remains visible without unnecessary text. | Count `pure-art(no-text)` entries across 20 founded Factions. Make sure that the share stays between 25% and 45%. | Implemented |
| **REQ-TYP-003** | **Engine 1 (VLM Multi-Font Studio)** must provide at least 10 font styles, 3 size tiers, 3 casings, 6 fills, 5 borders, and 6 layouts. | Run `test_typography_styled_rendering()` and inspect the output image. | Verified |
| **REQ-TYP-004** | **Engine 2 (FLUX Wordmarks)** must generate pixel Wordmarks with local 4-bit `FLUX.1-schnell` on MLX and check each image with `qwen2.5vl:7b` OCR. | Run `verify_wordmark_ocr()` on valid and invalid samples. Make sure that it accepts accurate text and rejects errors. | Implemented |
| **REQ-TYP-005** | **Engine 3 (Code Flourish Artist)** must use local `qwen2.5:7b-instruct` to produce sandboxed PIL flourish instructions. The sandbox must block imports, file access, and `draw.text()` calls. | Run `generate_code_wordmark()` with valid and unsafe inputs. Make sure that the safety filter blocks unsafe code. | Implemented |
| **REQ-TYP-006** | The engine must store `text_stamped` for each Faction so repeated `FORTIFY` turns must not draw duplicate letters on top of previous text. | Run `FORTIFY` twice on one Faction. Make sure that text appears only once with zero overlap. | Implemented |
| **REQ-TYP-007** | Wordmark truncation must split text at complete word boundaries and must not split a word in the middle. | Render long text inside a narrow Blueprint. Make sure that truncation keeps complete words only. | Implemented |

---

## 9. Live Dashboard and Timelapse Export Requirements (`REQ-UI`, `REQ-VID`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-UI-001** | The system must serve a local web dashboard (`127.0.0.1:8765`) with nearest-neighbor zoom and pan, a Faction list, and a live Chronicle feed. | Run FastAPI `TestClient` checks on `/`, `/api/state`, `/api/canvas.png`, and `/api/heatmap.png`. | Implemented |
| **REQ-UI-002** | The dashboard must provide Pause, Resume, Speed, Discover Faction, Run Epoch, Summon Void, Snapshot PNG, and overlay toggles with Faction boxes off by default. | Examine controls in `server.py` and test each control endpoint. | Implemented |
| **REQ-VID-001** | The Event Log (`events.bin`) must store each placement in a compact fixed-width binary record and must allow memory-mapped reads above 100,000,000 events. | Run `test_binary_event_log_roundtrip()`. Render a timelapse from a multi-hour Event Log without full-file RAM load. | Verified |
| **REQ-VID-002** | The renderer (`pixel-battle render`) must stream frames directly into `ffmpeg` to produce nearest-neighbor 60 fps `H.264` (`yuv420p`) 2K (`scale=4`) MP4 videos. | Render HUD and clean MP4 videos. Make sure that playback works and zero temporary frame files remain on disk. | Implemented |
| **REQ-VID-003** | The renderer must accept a time-curve exponent (`--curve`, default `1.35` to `1.45`) so early chaos and corner claims get clear visible duration. | Render a multi-hour run with `--curve 1.45`. Make sure that early and late phases show balanced speed. | Implemented |
| **REQ-VID-004** | Event Log Version 2 must store `uint32 user_id` and `uint16 faction` in 16 bytes for each placement and must keep read compatibility with Version 1 (`12`-byte) logs. | Write and read Version 2 records with `user_id > 10,000,000` and `faction > 500`. Make sure that Version 1 logs still load. | Verified |

---

## 10. Planned Infrastructure and Enhancement Requirements (`REQ-NEXT`)

Record additional candidate enhancements in this table before implementation starts.

| ID | Requirement | Target Area | Status |
| :--- | :--- | :--- | :--- |
| **REQ-NEXT-001** | `verify_wordmark_ocr()` must use edit distance to reject single-letter errors in Engine 2 FLUX Wordmarks (for example, `HOLD FAIST` for `HOLD FAST`). | `brain.py` | Verified |
| **REQ-NEXT-002** | `FactionDecision` must allow `expander_ratio` up to `1.0` before weight normalization so valid vision-model responses do not cause schema fallback warnings. | `brain.py` | Verified |
| **REQ-NEXT-003** | The CLI must provide a read-only `pixel-battle watch <run_dir>` dashboard mode that tails an active Headless Mode run without stop impact. | `server.py`, `cli.py` | Planned |
| **REQ-NEXT-004** | The renderer must provide an option that adds automatic war and Void zoom highlights or local Kokoro-82M voiceover narration to timelapse exports. | `timelapse.py` | Planned |
| **REQ-NEXT-005** | The Canvas must provide an optional mid-run expansion (for example, `256 x 256` to `512 x 512` at Hour 3) to start a new territory rush. | `canvas.py`, `engine.py` | Planned |

---

## 11. ASD-STE100 Update Rules

Apply these rules when you add or edit requirements in this file:

1. Use active voice and keep each descriptive sentence to 25 words or fewer (20 words or fewer for instructions).
2. State each mandatory requirement with `must` or `must not`.
3. Use approved Technical Names from Section 2 consistently, with one word for one meaning.
4. Do not use contractions, metaphors, vague adjectives, or unapproved `-ing` verb forms.
5. Assign a unique requirement ID, a clear verification method, and a status (`Planned`, `Implemented`, or `Verified`) to every requirement.
