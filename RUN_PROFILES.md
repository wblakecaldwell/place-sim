# Run Profiles & Creative Presets — `place-sim`

Creative preferences vary depending on the target video format and storytelling goal. This document separates per-run creative recipes from the permanent system requirements in [REQUIREMENTS.md](REQUIREMENTS.md).

---

## Profile A: `ytshort` — 4K YouTube Short (Fixed Vertical, Day-0 Land Rush)

**Goal:** Crisp, high-energy, immediately readable 58-second 4K vertical Short where prepped communities storm the canvas on Tick 1 and battle wall-to-wall with zero background static and zero repeated shields.

| Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Canvas** | `360 x 640` fixed (`230,400 px`, no expansion) | Exact 9:16 pixel-area twin of the magical `512 x 512` (`262k px`) square; exact `6x` nearest-neighbor upscale to **4K `2160 x 3840` @ 60fps** so every artwork is bold and crisp on phones. |
| **Palette** | Full 32 colors | Rich art variety immediately during the Hour-1 land grab. |
| **Whiteout Finale** | **OFF** (`--no-whiteout`) | Ends with a 3-second freeze-frame on the packed, colorful final masterpiece. |
| **Execution Mode** | **Step-locked (`--sync`)** + pre-forged `art_bank/` | 100% deterministic tick progression; zero hardware-vs-quality race; zero Metal swap thrashing. |
| **Simulation Horizon** | `10,000` ticks (`72,000` online slots -> ~37M+ placements) | `>= 150` placements/pixel across `360 x 640` so every artwork renders razor-sharp, defends cleanly, and evolves over time. |
| **Hour-1 Land Rush (Corners & Edges First)** | `150` Day-0 factions in Ticks `1–60` + rapid fill to `300` by Tick `500` (~5% of run) | First 18 landmarks lock down all 4 corners and 4 walls flush (`0px` margin), then whitespace niche-hunters fit unique AI FLUX artworks and `<= 1`-copy sprites into every remaining white pocket. |
| **Total Unique Factions** | `440` (`~300` active at once + immediate replacements when older factions fall; `1.0x` bold art scale) | Zero generic procedural shields, **zero Python-rendered word boxes/tiles**, stripped white sticker bubbles (`<= 2%`), near-square uncropped FLUX comics, and `<= 1` lifetime copy of every flag, comic, meme, painting, sprite, and fractal. |
| **Natural Rivals & Focused Wars** | AI-curated `rivals.json` + classic rival flags, capped at `<= 16` concurrent active wars | Natural rivals spawn face-to-face and push **coherent solid border fronts** strictly along opaque enemy pixels (zero bounding-box confetti), paired with border-heart alliances, clean 1px fortify contours, and flag expansion. |
| **Ribbons** | Strictly horizontal (`h`) and vertical (`v`), max `3` active | Clean cross-map stripes with zero diagonal ghost-shift artifacts or circular perimeter rings. |
| **Chaos Model** | White-claiming free agents + optional `--targeted-chaos` | Unorganized visitors actively claim stray white specks while `86%` organized share keeps factions dominant (no empty white background and no giant orange noise oceans). |
| **Burnout Gate** | `90%` coverage / `44` virtual hours (`<= 2` collapses/epoch) | Keeps the board packed at `~295–300` active factions all the way to Hour 72 while still allowing natural mid/late turnover. |
| **Exports** | `short_1080p_60fps_58s.mp4` (`3x`, `1080x1920`) & `short_4k_60fps_58s.mp4` (`6x`, `2160x3840`), 60fps, 55s + 3s hold, CRF 17, `--no-hud` | Lightweight 1080p copy plus 4K60 master unlocking YouTube's high-bitrate VP9/AV1 encoder. |

---

## Profile B: `reddit2022` — Faithful 1:1 Reddit r/place 2022 Tribute

**Goal:** Authentic multi-hour historical recreation of the full 72-hour Reddit r/place 2022 lifecycle for widescreen / desktop viewing.

| Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Canvas** | `1000x1000` -> `2000x1000` -> `2000x2000` (4-Act expansion) | Exact 1:1 r/place 2022 geometry with dynamic camera pull-back. |
| **Palette** | `16` -> `24` -> `32` -> `1` (Whiteout) | Matches official r/place 2022 palette unlocks and Whiteout finale. |
| **Population** | `120,000` online slots (~10M–15M unique users, 20% one-shot) | Matches official r/place 2022 participant scale. |
| **Dynamics** | Regional timezones, prime-time streamer flash-raids, alliances, Void cycle, Whiteout | Full 72-hour global community lifecycle. |
| **Export** | `timelapse_2k_clean.mp4` & `timelapse_2k_hud.mp4` (`2000x2000`, 60fps, 90–120s) | Native 2K desktop replay with dynamic camera. |

---

## Profile C: `standard` — Fast Local Sandbox & README GIF Demo

**Goal:** Quick interactive live-dashboard runs (`127.0.0.1:8765`) for testing features or exporting compact 20-second `512x512` GIF previews.

| Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Canvas** | `512 x 512` (`36,000` online slots, up to `200` factions) | Fast iteration on Apple Silicon with live web viewer. |
| **Export** | `assets/timelapse.gif` (Frame 0 = final canvas poster, lossless 32-color delta GIF) | Clean paused preview on GitHub README. |
