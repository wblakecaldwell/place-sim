# Pixel Battle System Requirements Specification

**Document Standard:** ASD-STE100 (Simplified Technical English)
**Project:** `place-sim` (`pixel-battle`)
**Revision:** 2.0

---

## 1. Purpose and Scope

This document defines the functional, visual, machine-learning, and operational requirements for `place-sim`.
The system simulates a multi-agent pixel Canvas similar to Reddit `r/place`.
The visual density, Canvas expansion, palette unlocks, and user scale target public Reddit `r/place` 2022 metrics.
All artificial intelligence models run locally on Apple Silicon hardware.

---

## 2. Technical Names (Approved Project Terminology)

Use these exact Technical Names in this specification and in related project documents.

| Technical Name | Definition |
| :--- | :--- |
| **Active Viewport** | The unlocked rectangle (`active_w x active_h`) of the Canvas where Citizens can place pixels. |
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
| **Faction** | A group of Citizens that share one Blueprint, one territory, one Timezone Region, and one diplomacy state. |
| **Fine-Art Painting** | A small or medium pixel rendition of a classic painting inside a gold pixel frame. |
| **Flash-Raid** | A temporary coordinated surge of Citizens led by a regional streamer onto established Canvas art. |
| **Fractal** | A recursive mathematical pixel pattern such as a Sierpinski triangle or Mandelbrot set. |
| **Geometric Chaos Faction** | A visionless Faction that draws abstract lines, lattices, spirals, or patterns from AI code. |
| **Headless Mode** | An execution mode that runs the simulation without a web dashboard or browser window. |
| **Hype** | A numeric vitality score for a Faction that controls Citizen recruitment, burnout, and retention. |
| **Macro Faction** | A large Faction whose Blueprint spans `60` to `160` pixels or crosses the Canvas. |
| **Medium Faction** | A mid-size Faction whose Blueprint spans `30` to `59` pixels on its longer side. |
| **Micro Faction** | A small Faction whose Blueprint spans `12` to `29` pixels on its longer side. |
| **National Flag** | A Blueprint that represents a country flag with accurate stripe and emblem geometry. |
| **One-Shot Citizen** | A Citizen that places `1` pixel during its session and then leaves the Canvas. |
| **Pareidolia** | The process in which a vision model examines an unclaimed pixel cluster and founds a Faction. |
| **Policy** | The GPU module that computes pixel placements for ready Citizens on each Tick. |
| **Rainbow Road** | A curved, multi-color stripe Blueprint that spans a wide area of the Canvas. |
| **Tick** | One discrete simulation step that updates Citizen cooldowns and applies pixel placements. |
| **Timezone Region** | One geographic region (`AMERICAS`, `EUROPE`, `ASIA_PACIFIC`, `LATAM`, `GLOBAL`) with a diurnal activity curve. |
| **Transverse Ribbon** | A straight horizontal, vertical, or diagonal multi-color stripe band that extends across the Canvas. |
| **The Void** | A hostile Faction that spreads dark tendrils, Black Holes, Fractals, or abyssal faces into nearby art. |
| **Whiteout Finale** | The final phase in which the active palette locks to pure white (`0`) to erase the Canvas. |
| **Wordmark** | A stylized pixel-art text emblem or logo banner for one Faction. |

---

## 3. System and Runtime Requirements (`REQ-SYS`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-SYS-001** | The system must run all models locally with zero cloud API calls and zero external API cost. | Block network access after local model cache load. Run the system and make sure that zero external requests occur. | Implemented |
| **REQ-SYS-002** | The system must run on Apple Silicon macOS with PyTorch MPS and Apple MLX below 42 GB peak memory. | Run all local models on Apple Silicon macOS. Make sure that peak memory stays below 42 GB. | Implemented |
| **REQ-SYS-003** | The CLI must accept a Tick limit (`--max-ticks`), a wall-clock hour limit (`--max-hours`), and a scale preset (`--preset`). | Run with `--max-hours 0.04`. Make sure that the process stops normally at the target time. | Implemented |
| **REQ-SYS-004** | The engine must save Canvas, Blueprint, and Faction checkpoints at each checkpoint interval and on `SIGINT`. | Stop a live run with `SIGINT`. Resume with `--resume` and make sure that Tick numbers continue in sequence. | Implemented |
| **REQ-SYS-005** | The batch script must prevent macOS idle sleep with `caffeinate -i` and release unused MPS and MLX memory at intervals. | Run unattended for 6 hours or more. Make sure that Tick counts increase without pause. | Implemented |
| **REQ-SYS-006** | The GPU Tick loop must run continuously without pause while background AI concept creation and art generation execute. | Run a discovery wave with live FLUX art. Make sure that GPU placements continue during model calls. | Verified |
| **REQ-SYS-007** | The local AI client must support OpenAI-compatible endpoints (`LM Studio` and `Ollama`) and read tokens from macOS Keychain (`place-sim`) or `LMSTUDIO_API_KEY`. | Store a token in Keychain and call the local server. Make sure that zero secrets exist in repo files. | Verified |

---

## 4. Canvas and Citizen Simulation Requirements (`REQ-SIM`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-SIM-001** | The Canvas must support `512 x 512`, `1024 x 1024`, and `2000 x 2000` presets and must accept custom width and height values. | Start simulations with each preset flag. Make sure that Canvas dimensions match each preset. | Implemented |
| **REQ-SIM-002** | The Canvas must use the official 32-color `r/place` 2022 palette with index `0` set to pure white (`#FFFFFF`). | Run `test_palette_roundtrip()`. Make sure that all 32 palette colors match after CIE-LAB conversion. | Verified |
| **REQ-SIM-003** | The Canvas must store current color, previous color (`prev_colors`), owner Faction, edit Tick, and activity for each pixel. | Change one pixel color twice. Make sure that `prev_colors` keeps the color from before the latest change. | Implemented |
| **REQ-SIM-004** | The Policy must simulate up to 150,000 concurrent Citizens in vectorized GPU batches with individual cooldowns and loyalties. | Run 50 Ticks with 120,000 Citizens on `2000 x 2000`. Make sure that zero per-Citizen Python loops run. | Verified |
| **REQ-SIM-005** | Each new run must start from a blank white Canvas with zero Factions and progress through chaos, nucleation, alliances, wars, and finale. | Examine Canvas images across all 4 acts. Make sure that each phase occurs in order. | Implemented |
| **REQ-SIM-006** | Citizens and Factions must reach Active Viewport corners and edges directly with zero forced border margin. | Measure pixel coverage on all 4 Active Viewport edges. Make sure that each edge has `>= 50%` coverage. | Implemented |
| **REQ-SIM-007** | Early Pareidolia scans must check the 4 Active Viewport corners first, then balance walls and interior space. | Examine the first 16 founded Factions. Make sure that corners and interior regions contain Factions. | Implemented |
| **REQ-SIM-008** | The `reddit2022` preset must run 4 acts: `1000 x 1000` (`16` colors), `2000 x 1000` (`24` colors), `2000 x 2000` (`32` colors), and Whiteout Finale (`1` color). | Run a compressed 4-act test. Make sure that locked colors never appear before unlock and `final_peak.png` saves before Whiteout. | Verified |
| **REQ-SIM-009** | The engine must support up to `1,200` simultaneous active Factions on a `2000 x 2000` Canvas across Micro, Medium, Macro, and Ribbon tiers. | Run a multi-hour `2000 x 2000` simulation and examine `state.json`. Make sure that active Factions scale smoothly across acts. | Verified |
| **REQ-SIM-010** | Active Factions must maintain realistic multi-scale diversity across Micro (`<= 30 px`), Medium (`31` to `64 px`), Macro (`65` to `160 px`), and Transverse Ribbons. | Measure active Blueprint sizes after Act 1. Make sure that all 4 size tiers coexist cleanly. | Implemented |
| **REQ-SIM-011** | Initial Citizen recruitment must scale in proportion to Blueprint pixel area across all concurrent Citizen slots. | Check member counts across Macro, Medium, and Micro Factions. Make sure that larger art recruits more Citizens. | Implemented |
| **REQ-SIM-012** | Pareidolia scans must detect unclaimed gaps across the Active Viewport and allow small cameos on wide flags or ribbons. | Examine a mature Canvas image. Make sure that small sprites fill gaps and appear on wide ribbons or flags. | Implemented |
| **REQ-SIM-013** | The `reddit2022` population curve must average `~15` to `16` placements for each user so `160,000,000` placements reach `~10,000,000` users and `~2,000,000` One-Shot Citizens. | Run a 500,000-placement statistical test. Make sure that mean clicks equal `14` to `18` and One-Shot Citizens equal `20.0% ± 1.0%`. | Verified |
| **REQ-SIM-014** | Each GPU Citizen slot must track a unique `user_id` and click budget, then cycle to a new `user_id` at zero. | Count placements for each `user_id` in `events.bin`. Make sure that finished sessions match target click tiers. | Verified |
| **REQ-SIM-015** | Community theme selection must sample active non-NSFW subreddits and international cultures with a profanity filter active. | Examine 40 founded Factions. Make sure that zero profanity and zero duplicate names appear. | Verified |
| **REQ-SIM-016** | The engine must advance a 72-hour virtual clock across the run and apply diurnal sleep-wake multipliers (`0.22x` to `1.35x`) to each Timezone Region. | Run a unit test across 24 virtual hours. Make sure that `EUROPE`, `AMERICAS`, and `ASIA_PACIFIC` peak at offset hours. | Verified |
| **REQ-SIM-017** | Factions must lose Hype through burnout after peak age. Alliances must cap at `<= 3` partners with timed expiry for territory turnover. | Examine `chronicle.jsonl` across Acts 1 to 3. Make sure that old Factions burn out and new Factions replace them. | Implemented |

---

## 5. Trained Local Citizen Model Requirements (`REQ-ML`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-ML-001** | The engine must separate pixel placement behind the `PixelPolicy` protocol so neural and stochastic Policies swap without engine edits. | Run both `--policy neural` and `--policy stochastic` with the same engine code. | Implemented |
| **REQ-ML-002** | The dataset builder must train on official Reddit `r/place` 2022 logs from Day 1 Hours 0 to 12 and Day 3 peak activity. | Check that 8 chronological chunks produce `>= 400,000` samples. | Implemented |
| **REQ-ML-003** | `RedditCitizenNet` must accept a `21 x 21` local patch with current colors, previous colors, recency, optional hindsight goal, and archetype. | Examine checkpoint metadata and run `test_neural_policy_inference()`. | Verified |
| **REQ-ML-004** | `RedditCitizenNet` must predict click position across `441` cells and then predict color across `32` palette classes masked to unlocked colors. | Run inference with a 16-color mask. Make sure that zero locked color indices appear in output batches. | Verified |
| **REQ-ML-005** | Dataset selection must keep natural Reddit color balance with zero extra Void samples and must not apply `FREE` dropout to `VOID`. | Run the 5-archetype color probe on a blue-green patch. Make sure that `BUILDER` black probability is `<= 0.15`. | Implemented |
| **REQ-ML-006** | The trained model must reach `>= 90%` color accuracy for Blueprint-free defenders (`def_free@1`) and `>= 98%` goal color accuracy (`goal_col@1`). | Read saved metrics in `models/reddit_citizen.pt` (`def_free@1 = 98.6%`, `goal_col@1 = 100.0%`). | Verified |
| **REQ-ML-007** | Live neural inference must maintain `>= 8,000` pixel placements per second on Apple Silicon MPS. | Run a benchmark on Apple Silicon MPS. Make sure that throughput stays `>= 8,000` pixels per second. | Verified |
| **REQ-ML-008** | `NeuralRedditPolicy` must batch ready builders and defenders across active Factions into one combined GPU forward pass on each Tick. | Benchmark `1,200` active Factions on `2000 x 2000` MPS. Make sure that throughput stays `>= 10,000` pixels per second. | Verified |

---

## 6. Faction Intelligence and Diplomacy Requirements (`REQ-DIP`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-DIP-001** | The strategic brain must run a local Vision-Language model (`Qwen3-VL-30B-A3B` via LM Studio or Ollama) with strict JSON validation and fallback on error. | Call the local server with a test prompt and image. Make sure that HTTP 200 and valid JSON return. | Verified |
| **REQ-DIP-002** | Local AI concept creation must invent Faction names, timezones, languages, slogans, and art prompts live with zero duplicate names and playful non-violent culture. | Generate 24 AI-invented Factions. Make sure that names stay unique and span international and pop-culture groups. | Verified |
| **REQ-DIP-003** | Each Epoch turn must show the Faction leader an annotated PNG of its territory with labeled neighbors (`YOU`, `ALLY`, `VOID`, neutral). | Examine `render_faction_view_pil()` output. Make sure that `FactionDecision` JSON passes validation. | Implemented |
| **REQ-DIP-004** | The engine must execute `BUILD`, `EXPAND`, `MORPH_ART`, `FORTIFY`, `RAID`, `PROPOSE_ALLIANCE`, `ACCEPT_ALLIANCE`, and `EXODUS`. | Examine `chronicle.jsonl` after a multi-hour run. Make sure that diplomacy, expansion, and war events occur. | Implemented |
| **REQ-DIP-005** | Allied neighbors must place pink and red border hearts along shared seams and must send spare defenders when an ally loses territory. | Examine the final Canvas image and the ally defense routine in `policies/stochastic.py`. | Implemented |
| **REQ-DIP-006** | The Void must appear at intervals, recruit Citizens, spread with the `VOID` archetype, collapse weak Factions, and stop after its time limit. | Examine `chronicle.jsonl` for Void start, collapse, and end events. | Implemented |
| **REQ-DIP-007** | Adjacent National Flag Factions must compete through territorial pushes and raids, and must also form border-heart treaties when diplomacy succeeds. | Examine `chronicle.jsonl` from a run with National Flags. Make sure that flag wars and treaties both occur. | Implemented |
| **REQ-DIP-008** | Regional Flash-Raids must trigger during prime-time windows, mobilize `3,000` to `8,000` temporary Citizens onto established territory, and disperse after `250` to `450` Ticks. | Run a Flash-Raid test and inspect `chronicle.jsonl`. Make sure that raiders stamp the mural and then disperse. | Verified |
| **REQ-DIP-009** | Transverse Ribbon Factions must advance their stripe tip across the Active Viewport, cut through rivals, and yield to allied art. | Examine timelapse frames of a Transverse Ribbon. Make sure that the stripe grows across the Canvas. | Implemented |
| **REQ-DIP-010** | The Void subsystem must rotate across Tendril Void, Black Hole, Fractal, and Void Monster variants so dark events vary across long runs. | Examine `chronicle.jsonl` and snapshots across a multi-hour run. Make sure that multiple Void variants appear. | Implemented |

---

## 7. AI-First Visual Art, Fluid Cutouts, and Diversity Requirements (`REQ-ART`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-ART-001** | All generated Blueprints must quantize strictly to the currently unlocked Canvas palette tier (`16`, `24`, `32`, or `1` Whiteout color). | Generate art in Acts 1, 2, and 3. Make sure that every pixel index belongs to the unlocked palette. | Verified |
| **REQ-ART-002** | Primary AI artwork must use local 4-bit `FLUX.1-schnell` with a Pixel-Art LoRA on MLX (`mx.clear_cache()` after each image) quantized via CIE-LAB. | Generate 6 test Blueprints with `FluxPixelArtGenerator`. Make sure that edges stay crisp and Metal cache stays bounded. | Verified |
| **REQ-ART-003** | Characters, mascots, superheroes, cartoons, memes, and Wordmarks must use fluid non-rectangular Cutout silhouettes (`-1` exterior) by default. | Inspect 10 AI-generated character and logo Blueprints. Make sure that outer background pixels equal `-1`. | Verified |
| **REQ-ART-004** | Expander Citizens must grow from the active silhouette frontier and must not draw dotted rectangular borders. | Examine `_expand_perimeter()` output on a live Canvas. Make sure that zero dotted rectangular frames appear. | Implemented |
| **REQ-ART-005** | Occupancy checks must mark only non-transparent silhouette pixels (`blueprint >= 0`) so curved artworks fit closely together. | Examine the occupancy mask in `detector.py`. Check curved neighbor placement on the Canvas. | Implemented |
| **REQ-ART-006** | Every named National Flag, Fine-Art Painting, or fallback template must appear at most once on the active Canvas to prevent duplicate stamps. | Inspect `state.json` after `>= 100` Factions form. Make sure that zero duplicate country flags or templates exist. | Implemented |
| **REQ-ART-007** | `FORTIFY` must add one outer border along a Cutout contour and must not add extra rings on repeated fortify turns. | Run `FORTIFY` twice on one Faction. Make sure that the outer border applies only once. | Implemented |
| **REQ-ART-008** | The art engine must provide accurate procedural National Flags (`<= 1` per country, `~8%` to `12%` of Factions), each locked to its exact Timezone Region. | Verify timezone tags and uniqueness across all founded National Flag Factions. | Verified |
| **REQ-ART-009** | At least `80%` of non-flag Factions must originate live from local AI concept creation (`FLUX` Pixel-Art LoRA, LLM-written pixel code, or AI Wordmarks). | Audit `state.json` after a live run. Make sure that AI-generated Factions account for `>= 80%` of non-flag art. | Implemented |
| **REQ-ART-010** | Geometric Chaos Factions must execute sandboxed LLM-written Python (`np` and `math` only) to generate original lines, lattices, spirals, and abstract patterns. | Run `test_sandboxed_geometric_code()` with valid and unsafe scripts. Make sure that valid code draws and unsafe code fails safely. | Verified |
| **REQ-ART-011** | The art engine must generate pure-white (`0`) Empty-Space Faction Blueprints, and Blueprint validation must accept pure-white grids. | Place an Empty-Space Faction on a colored patch. Make sure that Citizens restore colored pixels to pure white. | Verified |
| **REQ-ART-012** | The art engine must generate horizontal, vertical, and diagonal Transverse Ribbon Blueprints that tile cleanly across the Active Viewport. | Generate horizontal, vertical, and diagonal Transverse Ribbons. Make sure that patterns tile cleanly across all pixels. | Verified |
| **REQ-ART-013** | The art engine must generate Black Hole Blueprints, Fractal Blueprints, and Void Monster faces for Void eruptions. | Generate preview PNGs of Black Holes, Fractals, and Void Monsters. Make sure that geometry stays sharp. | Verified |
| **REQ-ART-014** | The engine must provide an option for sparse `4 x 5` pixel mini-crewmate cameos inside wide solid regions with no overwrite of fine details. | Inspect cameo placements in a mature run. Make sure that mini cameos remain tiny and sparse. | Implemented |

---

## 8. Multilingual Model-Designed Typography Requirements (`REQ-TYP`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-TYP-001** | A local AI model must design the typography of each Faction and must not apply one uniform bitmap font across the Canvas. | Examine `design_typography()` outputs for 6 Factions. Make sure that fonts, colors, and layouts differ. | Implemented |
| **REQ-TYP-002** | Multilingual Wordmarks must render Chinese, Japanese, Korean, Spanish, French, German, Portuguese, and Turkish text cleanly with system Unicode pixel fonts. | Run `test_multilingual_typography()` on CJK and accented Latin strings. Make sure that zero missing-glyph boxes appear. | Verified |
| **REQ-TYP-003** | **Engine 1 (Multi-Font Studio)** must provide at least 10 Latin font styles plus CJK Unicode fonts, 3 sizes, 6 fills, 5 borders, and 6 layouts. | Run `test_typography_styled_rendering()` and inspect the output image. | Verified |
| **REQ-TYP-004** | **Engine 2 (FLUX Wordmarks)** must generate illustrated Wordmarks with local 4-bit `FLUX.1-schnell` on MLX and check text accuracy with VLM OCR edit distance. | Run `verify_wordmark_ocr()` on valid and misspelled samples. Make sure that it accepts accurate text and rejects errors. | Verified |
| **REQ-TYP-005** | **Engine 3 (Code Flourish Artist)** must run sandboxed PIL flourish instructions from the local LLM with zero imports, zero file access, and zero `draw.text()` calls. | Run `generate_code_wordmark()` with valid and unsafe inputs. Make sure that the safety filter blocks unsafe code. | Implemented |
| **REQ-TYP-006** | The engine must store `text_stamped` for each Faction so repeated `FORTIFY` turns must not draw duplicate letters on top of previous text. | Run `FORTIFY` twice on one Faction. Make sure that text appears only once with zero overlap. | Implemented |
| **REQ-TYP-007** | Wordmark truncation must split text at complete word boundaries and must not split a word in the middle. | Render long text inside a narrow Blueprint. Make sure that truncation keeps complete words only. | Implemented |

---

## 9. Live Dashboard and Timelapse Export Requirements (`REQ-UI`, `REQ-VID`)

| ID | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **REQ-UI-001** | The system must serve a local web dashboard (`127.0.0.1:8765`) with nearest-neighbor zoom and pan, virtual timezone clock, Faction list, and live Chronicle feed. | Run FastAPI `TestClient` checks on `/`, `/api/state`, `/api/canvas.png`, and `/api/heatmap.png`. | Implemented |
| **REQ-UI-002** | The dashboard must provide Pause, Resume, Speed, Discover Faction, Run Epoch, Summon Void, Snapshot PNG, and overlay toggles with Faction boxes off by default. | Examine controls in `server.py` and test each control endpoint. | Implemented |
| **REQ-VID-001** | The Event Log (`events.bin`) must store each placement in a 16-byte record (`tick`, `x`, `y`, `color`, `faction`, `user_id`). Reads must support above 160,000,000 events. | Run `test_binary_event_log_v2_and_v1_compat()`. Render a timelapse from a large Event Log without full-file RAM load. | Verified |
| **REQ-VID-002** | The renderer (`pixel-battle render`) must stream frames directly into `ffmpeg` to produce nearest-neighbor 60 fps `H.264` (`yuv420p`) 2K MP4 videos. | Render HUD and clean MP4 videos. Make sure that playback works and zero temporary frame files remain on disk. | Implemented |
| **REQ-VID-003** | The renderer must accept a time-curve exponent (`--curve`, default `1.35` to `1.45`) so early chaos and expansion rushes get clear visible duration. | Render a multi-act run with `--curve 1.45`. Make sure that early, expansion, and finale phases show balanced speed. | Implemented |
| **REQ-VID-004** | Event Log Version 2 must keep read compatibility with Version 1 (`12`-byte) logs. | Read a Version 1 Event Log with `mmap_events()`. Make sure that all fields match original values. | Verified |
| **REQ-VID-005** | The renderer must support dynamic camera pull-back across `expand_canvas` milestones (`1000 x 1000` -> `2000 x 1000` -> `2000 x 2000`) and Whiteout Finale playback. | Render a 4-act test run with `--dynamic-camera`. Make sure that the camera expands smoothly at Act 2 and Act 3. | Verified |

---

## 10. ASD-STE100 Update Rules

Apply these rules when you add or edit requirements in this file:

1. Use active voice and keep each descriptive sentence to 25 words or fewer (20 words or fewer for instructions).
2. State each mandatory requirement with `must` or `must not`.
3. Use approved Technical Names from Section 2 consistently, with one word for one meaning.
4. Do not use contractions, metaphors, vague adjectives, or unapproved `-ing` verb forms.
5. Assign a unique requirement ID, a clear verification method, and a status (`Planned`, `Implemented`, or `Verified`) to every requirement.
