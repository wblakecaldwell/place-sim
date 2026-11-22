"""Local Ollama Qwen2.5-VL multimodal brain for pareidolia discovery, color diversity & diplomacy."""

from __future__ import annotations

import base64
import io
import logging
import random
from typing import Literal

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from pixel_battle.art.typography import FONT_STYLES, TypographyStyleSpec
from pixel_battle.factions import DiplomacyMessage, Faction
from pixel_battle.palette import BLACK_IDX, COLOR_NAMES, WHITE_IDX, resolve_color

log = logging.getLogger(__name__)


class TypographyDesign(BaseModel):
    include_text: bool = Field(
        description="True if this faction deliberately puts text/wordmark on its art; False (~35% of murals) for pure iconic art with no words"
    )
    text: str = Field(
        description="Short intentional text (2-12 chars), e.g. 'r/cats', 'HOLD FAST', 'KOI', 'BLUE CORNER', 'PEACE'"
    )
    casing: Literal["upper", "title", "subreddit"] = Field(
        description="'upper' for bold titles, 'title' for elegant script/serif/bubble, 'subreddit' for lowercase r/name"
    )
    font_style: Literal[
        "gothic",
        "block",
        "arcade",
        "bubble",
        "graffiti",
        "condensed",
        "serif",
        "script",
        "pixel_mono",
        "pixel_retro",
    ]
    size_tier: Literal["tiny", "medium", "large"] = Field(
        description="'tiny' for subtle corner tag, 'medium' for mural label, 'large' for hero wordmark"
    )
    color_effect: Literal[
        "solid",
        "vertical_gradient",
        "per_letter_palette",
        "rainbow_letters",
        "glossy_highlight",
        "striped_fill",
    ]
    border_effect: Literal[
        "outline_1px",
        "extrude_3d",
        "double_sticker",
        "neon_glow",
        "none",
    ]
    layout: Literal[
        "straight",
        "wavy",
        "arched",
        "italic_slant",
        "stacked",
        "vertical_side",
    ]
    placement: Literal[
        "bottom_center",
        "top_center",
        "center",
        "bottom_right_tag",
        "top_left_tag",
        "left_vertical",
    ]
    fill_color: str = Field(description="Primary letter color name from allowed palette (be creative — not always white!)")
    fill_color_2: str = Field(description="Second letter color name for gradient/stripes/per-letter effects")
    fill_color_3: str = Field(description="Third letter accent color name")
    outline_color: str = Field(description="Outline color name (e.g. black, dark_blue, dark_purple, burgundy, white)")
    shadow_color: str = Field(description="3D extrusion / neon glow color name")
    use_ribbon: bool = Field(default=False, description="True only if text sits on a solid ribbon bar; False for organic floating letters")
    flux_style_desc: str = Field(
        default="",
        description="Short 5-10 word artistic typography description if rendered as a FLUX illustrated logo (e.g. 'chunky 3D chrome arcade letters with tiny lightning bolts')",
    )


class WordmarkOCRCheck(BaseModel):
    readable: bool = Field(description="True if the pixel-art wordmark is legible, visually crisp, and clearly resembles the intended word")
    read_text: str = Field(description="What letters/word you read in the image")
    quality_score: int = Field(ge=1, le=10, description="Visual design quality score 1-10")


class PareidoliaProposal(BaseModel):
    name: str = Field(description="Fresh, catchy 2-3 word faction name (NO words like Radiant, Whirl, Cosmic, Mystic, Luminous)")
    theme: str = Field(description="Short 2-4 word visual theme")
    banner_text: str = Field(
        description="Short 3-11 char r/place pixel banner tag (UPPERCASE letters/numbers/slash only, e.g. 'R/FROG', 'BLUE CORNER', 'HOLD FAST', 'R/CAT', 'RAMEN')"
    )
    slogan: str = Field(description="Punchy rallying cry under 10 words")
    art_mode: Literal["diffusion", "procedural", "banner"] = Field(
        description="'diffusion' for creatures/icons/items; 'procedural' for flags/shields/rainbows/castles/mosaics; 'banner' for wide horizontal text signs"
    )
    art_prompt: str = Field(
        description="STRICTLY 4 to 10 visual words ONLY (e.g. 'cute bubblegum pink kirby sprite on sky blue'). NO sentences."
    )
    primary_color: str = Field(description="Palette color name from allowed list matching the requested color theme")
    secondary_color: str = Field(description="Contrasting palette color name from allowed list")
    accent_color: str = Field(description="Highlight palette color name from allowed list")


class FactionDecision(BaseModel):
    action: Literal[
        "BUILD",
        "EXPAND",
        "MORPH_ART",
        "FORTIFY",
        "RAID",
        "PROPOSE_ALLIANCE",
        "ACCEPT_ALLIANCE",
        "EXODUS",
    ]
    direction: Literal["NORTH", "SOUTH", "EAST", "WEST", "NONE"] = "NONE"
    target_faction_id: int | None = None
    new_art_prompt: str = Field(
        default="",
        description="ONLY visual nouns/adjectives (4-10 words) for MORPH_ART or EXPAND, keeping your signature palette. NO verbs like 'Expand' or 'Create'.",
    )
    builder_ratio: float = Field(default=0.55, ge=0.0, le=1.0)
    defender_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    expander_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    diplomacy_message: str = Field(
        default="",
        description="Short in-character message (<16 words) to target neighbor or public board",
    )
    reasoning: str = Field(description="One crisp sentence explaining your strategic choice")


def levenshtein_distance(a: str, b: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def ocr_text_matches(expected_text: str, read_text: str) -> bool:
    """Strict edit-distance OCR gate (REQ-NEXT-001): rejects subtle spelling errors like 'HOLD FAIST'."""
    exp = "".join(ch for ch in expected_text.lower() if ch.isalnum())
    got = "".join(ch for ch in read_text.lower() if ch.isalnum())
    if not exp or not got:
        return False
    dist = levenshtein_distance(exp, got)
    if len(exp) <= 8:
        return dist == 0
    return dist <= 1 and exp[0] == got[0] and exp[-1] == got[-1]


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# 12 color palettes spanning all 32 r/place colors to avoid palette drift
COLOR_THEMES: list[tuple[str, str, str, str]] = [
    ("electric blue, light blue, and white", "blue", "light_blue", "white"),
    ("hot pink, magenta, and light pink", "pink", "magenta", "white"),
    ("royal purple, periwinkle, and lavender", "purple", "periwinkle", "lavender"),
    ("bright crimson red, snow white, and dark blue", "dark_red", "white", "dark_blue"),
    ("vibrant full rainbow spectrum on black", "red", "blue", "yellow"),
    ("bubblegum pink, sky blue, and pastel yellow", "light_pink", "light_blue", "pale_yellow"),
    ("deep indigo, bright cyan, and pale purple", "indigo", "light_blue", "pale_purple"),
    ("emerald green, bright lime, and crisp white", "green", "light_green", "white"),
    ("sunshine yellow, cobalt blue, and black", "yellow", "dark_blue", "black"),
    ("magenta, dark purple, and electric teal", "magenta", "dark_purple", "light_teal"),
    ("coral red, bright orange, and pale yellow", "red", "orange", "pale_yellow"),
    ("cobalt dark blue, lavender, and pink", "dark_blue", "lavender", "pink"),
]

_CREATIVE_CATEGORIES: list[str] = [
    "an iconic cute animal mascot or internet meme (shiba doge, smug frog, pixel cat, capybara, penguin, rubber duck, green owl)",
    "a retro video game or indie game character (pink puff kirby, hollow mask knight, slime, arcade ghost, sword, potion, gameboy)",
    "an anime chibi face or colorful cartoon character with expressive pixel eyes",
    "a bold pop-culture meme or funny internet icon (sunglasses cool face, nyan cat, pixel UFO, skull, friendly ghost, pixel crown)",
    "a classic r/place wide horizontal word banner with a catchy subreddit tag or slogan",
    "a cute food or drink icon (strawberry cake, boba milk tea, ramen bowl, pizza slice, sushi, donut)",
    "a space or nature icon (ringed Saturn planet, crescent moon, pixel astronaut, cherry blossom, red mushroom, blue whale)",
    "a bold heraldic shield, rainbow road, pixel castle, or fractal emblem",
]

_FALLBACK_FACTIONS: list[tuple[str, str, str, str, str, str]] = [
    ("Blue Corner", "blue corner empire", "BLUE CORNER", "Defend the corner at all costs!", "procedural", "solid cobalt blue corner banner"),
    ("Pink Kirby Club", "cute pink puff", "R/KIRBY", "Spread pink joy everywhere!", "diffusion", "cute bubblegum pink round mascot smiling on sky blue"),
    ("Pixel Cat Cafe", "purple tabby cat", "R/CATS", "Meow across the whole canvas!", "diffusion", "cute lavender and white pixel cat face with pink ears"),
    ("Rainbow Road", "prism stripes", "RAINBOW", "Unite all 32 colors!", "procedural", "vibrant diagonal rainbow prism stripes"),
    ("Hold The Line", "retro word banner", "HOLD FAST", "Never surrender a single pixel!", "banner", "bold horizontal pixel word banner"),
    ("Indigo Moon", "crescent night", "R/MOON", "Guard the starry night!", "diffusion", "glowing pale yellow crescent moon on deep indigo sky"),
    ("Strawberry Club", "pixel shortcake", "R/CAKE", "Sweetest pixels on the board!", "diffusion", "cute red and white pixel strawberry cake with pink frosting"),
    ("Nordic Frost", "cobalt cross flag", "R/NORDIC", "Hold the northern wall!", "procedural", "crisp cobalt blue and white nordic cross flag"),
    ("Neon Duck Gang", "yellow duck", "R/DUCK", "Quack together, stay together!", "diffusion", "cute bright yellow rubber duck on electric blue water"),
    ("Violet Portal", "lavender vortex", "R/PORTAL", "Enter the purple dimension!", "procedural", "glowing purple and magenta pixel spiral portal"),
    ("Retro Arcade", "pixel heart sign", "PLAY PIXEL", "Insert coin to paint!", "banner", "retro magenta and cyan arcade word banner"),
    ("Boba Guild", "pastel boba tea", "R/BOBA", "Fuel the canvas with boba!", "diffusion", "cute pastel pink and purple pixel boba milk tea cup"),
]

_BANNED_CLICHES = ("radiant", "whirl", "twirl", "luminous", "mystic", "cosmic", "blaze")


class LocalVLMBrain:
    """Interfaces with local Ollama qwen2.5vl:7b using strict JSON schemas & vibrant palette rotation."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vlm_model: str = "qwen2.5vl:7b",
        timeout: float = 45.0,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.vlm_model = vlm_model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._fallback_idx = 0

    def check_health(self) -> bool:
        try:
            r = self._client.get(f"{self.ollama_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(self.vlm_model in n for n in names)
        except Exception:
            return False

    def _chat_json(
        self,
        system: str,
        user_text: str,
        image: Image.Image | None,
        schema_cls: type[BaseModel],
        temperature: float = 0.80,
    ) -> BaseModel | None:
        msg: dict = {"role": "user", "content": user_text}
        if image is not None:
            msg["images"] = [_pil_to_b64_png(image)]

        payload = {
            "model": self.vlm_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                msg,
            ],
            "format": schema_cls.model_json_schema(),
            "options": {
                "temperature": temperature,
                "num_predict": 280,
            },
        }
        try:
            resp = self._client.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = data["message"]["content"]
            return schema_cls.model_validate_json(raw)
        except Exception as exc:
            log.warning("Ollama VLM call failed (%s); using heuristic fallback.", exc)
            return None

    def pareidolia_discover(
        self,
        cluster_img: Image.Image,
        dominant_color_names: list[str],
        existing_names: list[str],
        existing_themes: list[str],
        faction_index: int = 0,
        desired_mode: str = "diffusion",
        is_corner: bool = False,
        is_edge: bool = False,
        subreddit_hint: str = "",
    ) -> tuple[PareidoliaProposal, str]:
        """Inspect an emergent cluster and return (PareidoliaProposal, assigned_color_theme)."""
        theme_desc, def_c1, def_c2, def_c3 = COLOR_THEMES[faction_index % len(COLOR_THEMES)]
        category = _CREATIVE_CATEGORIES[faction_index % len(_CREATIVE_CATEGORIES)]

        loc_hint = (
            "This cluster is right in a CANVAS CORNER — perfect for an iconic corner colony (like Blue Corner, Pink Corner, or a proud corner mascot/flag)!"
            if is_corner
            else "This cluster hugs a CANVAS WALL — great for a crisp banner, flag, or bold sprite!"
            if is_edge
            else "This cluster formed in the open canvas."
        )
        sub_hint = (
            f"Community inspiration: real subreddit r/{subreddit_hint} (design iconic pixel art & banner_text inspired by this community!).\n"
            if subreddit_hint
            else ""
        )

        if desired_mode == "banner":
            mode_hint = (
                f"Set art_mode='banner'. Invent a fun r/place horizontal word banner (like 'r/{subreddit_hint or 'place'}', 'HOLD FAST', 'BLUE TEAM', 'PEACE') "
                f"using the signature color theme: {theme_desc}."
            )
        elif desired_mode == "procedural":
            mode_hint = (
                f"Set art_mode='procedural'. Invent a crisp flag, Nordic cross, rainbow road, heart crest, castle, or mosaic "
                f"using the signature color theme: {theme_desc}."
            )
        else:
            mode_hint = (
                f"Set art_mode='diffusion'. Invent {category} using the signature color theme: {theme_desc}."
            )

        system = (
            "You are a creative r/place community founder. Look at a messy early cluster of pixels "
            "and invent a charming, iconic pixel-art faction with a short punchy subreddit-style banner_text. "
            "Be wildly diverse and colorful! NEVER use cliché adjectives like Radiant, Whirlwind, Cosmic, Mystic, or Luminous."
        )
        recent_themes = ", ".join(existing_themes[-6:]) if existing_themes else "none"
        user_text = (
            f"{loc_hint}\n"
            f"{sub_hint}"
            f"Signature color palette to feature prominently: {theme_desc}.\n"
            f"Colors already visible in cluster: {', '.join(dominant_color_names) or 'mixed'}.\n"
            f"Recent themes already taken (pick something totally different!): {recent_themes}.\n"
            f"Allowed palette names: {', '.join(COLOR_NAMES[1:])}.\n"
            f"{mode_hint}\n"
            "Return JSON with a concise 4-10 word visual art_prompt and a short UPPERCASE banner_text (3-11 chars)."
        )

        res = self._chat_json(system, user_text, cluster_img, PareidoliaProposal, temperature=0.88)
        if isinstance(res, PareidoliaProposal) and res.name.strip():
            low_name = res.name.strip().lower()
            cliche_hit = any(w in low_name for w in _BANNED_CLICHES)
            dup_hit = low_name in {n.lower() for n in existing_names}
            if not dup_hit and not cliche_hit:
                # Enforce requested mode balance & clean short banner text
                res.art_mode = desired_mode  # type: ignore[assignment]
                if not res.banner_text.strip():
                    res.banner_text = f"R/{res.name.split()[0][:7].upper()}"
                return res, theme_desc

        # Diverse vibrant fallback
        idx = self._fallback_idx % len(_FALLBACK_FACTIONS)
        self._fallback_idx += 1
        name, theme, btxt, slogan, mode, prompt = _FALLBACK_FACTIONS[idx]
        if name.lower() in {n.lower() for n in existing_names}:
            name = f"{name} {faction_index + 1}"
        if desired_mode in ("diffusion", "procedural", "banner"):
            mode = desired_mode
        return (
            PareidoliaProposal(
                name=name,
                theme=theme,
                banner_text=btxt,
                slogan=slogan,
                art_mode=mode,  # type: ignore[arg-type]
                art_prompt=f"{prompt} in {theme_desc}",
                primary_color=def_c1,
                secondary_color=def_c2,
                accent_color=def_c3,
            ),
            theme_desc,
        )

    def faction_epoch_turn(
        self,
        faction: Faction,
        view_img: Image.Image,
        neighbors: list[dict],
        inbox: list[DiplomacyMessage],
        public_feed: list[DiplomacyMessage],
        tick: int,
        can_morph: bool = True,
    ) -> FactionDecision:
        """Let a faction's VLM leader inspect its territory + neighbors and choose its next move."""
        system = (
            "You are the strategic leader of a pixel-art faction on a live 512x512 r/place canvas. "
            "Inspect the image (CYAN box = your territory; GREEN = allies; ORANGE = neighbors; RED = Void threat). "
            "Prioritize diplomacy, border-heart alliances, defending borders, fortifying, and dramatic neighbor raids! "
            "If you morph or expand art, strictly preserve your faction's signature colors."
        )

        nbr_text = (
            "\n".join(
                f"- ID={n['id']} '{n['name']}' ({n['relation']}, {n['direction']}, "
                f"dist={n['distance']}px, overlap={n['overlap_pixels']}px, members={n['members']})"
                for n in neighbors[:5]
            )
            if neighbors
            else "None nearby (open frontier!)"
        )
        inbox_text = (
            "\n".join(f"- From {m.sender_name} (ID={m.sender_id}, {m.kind}): \"{m.text}\"" for m in inbox)
            if inbox
            else "No direct messages."
        )
        pub_text = (
            "\n".join(f"- {m.sender_name}: \"{m.text}\"" for m in public_feed[-4:])
            if public_feed
            else "Quiet."
        )
        mem_text = "; ".join(faction.memory[-4:]) if faction.memory else "Just founded."

        morph_rule = (
            "- If integrity >= 0.85 and no active border war: you may MORPH_ART or EXPAND (keep your signature colors!), "
            "OR prefer PROPOSE_ALLIANCE / RAID with nearby neighbors."
            if can_morph
            else "- Morph cooldown is active: DO NOT choose MORPH_ART. Choose PROPOSE_ALLIANCE, ACCEPT_ALLIANCE, FORTIFY, RAID, EXPAND, or BUILD!"
        )

        flag_rivalry_rule = (
            "- You are a NATIONAL FLAG faction! Compete boldly with neighboring rival flags via RAID/EXPAND, "
            "OR sign iconic border-heart peace treaties via PROPOSE_ALLIANCE!\n"
            if faction.category == "flag"
            else "- You are a TRANSVERSE RIBBON faction! Push EXPAND to extend your straight stripe all the way across the canvas!\n"
            if faction.art_mode == "ribbon"
            else ""
        )

        user_text = (
            f"TICK {tick} | FACTION: {faction.name} (ID={faction.id}, tier={faction.size_tier}, cat={faction.category}, banner='{faction.banner_text}', stage={faction.evolution_stage}/5)\n"
            f"Signature colors: {faction.color_theme or 'vibrant'} | Art ({faction.art_mode}): \"{faction.art_prompt}\"\n"
            f"BBox: {faction.w}x{faction.h} at ({faction.x0},{faction.y0}) | members={faction.member_count}, "
            f"integrity={faction.integrity:.0%}, morale={faction.morale:.2f}, churn={faction.recent_churn:.2f}, fortified={faction.has_border_frame}\n"
            f"Pending alliance offers from IDs: {sorted(faction.pending_ally_offers) or 'none'}\n"
            f"Memory: {mem_text}\n\n"
            f"NEIGHBORS:\n{nbr_text}\n\n"
            f"INBOX:\n{inbox_text}\n\n"
            f"PUBLIC FEED:\n{pub_text}\n\n"
            "RULES:\n"
            f"{flag_rivalry_rule}"
            "- If pending alliance offer exists from a non-Void neighbor -> strongly consider ACCEPT_ALLIANCE(target_faction_id).\n"
            "- If integrity < 0.72 -> BUILD or FORTIFY with high defender_ratio.\n"
            "- If close neutral neighbor (< 24px) -> PROPOSE_ALLIANCE (for border hearts!) or RAID (for territory war!).\n"
            f"{morph_rule}\n"
            "- If integrity < 0.28 and members < 160 -> EXODUS.\n"
            "Return JSON!"
        )

        res = self._chat_json(system, user_text, view_img, FactionDecision, temperature=0.8)
        if isinstance(res, FactionDecision):
            return res

        return self._heuristic_fallback_decision(faction, neighbors, can_morph=can_morph)

    def _heuristic_fallback_decision(
        self, faction: Faction, neighbors: list[dict], can_morph: bool = True
    ) -> FactionDecision:
        if faction.pending_ally_offers:
            aid = next(iter(faction.pending_ally_offers))
            return FactionDecision(
                action="ACCEPT_ALLIANCE",
                target_faction_id=aid,
                builder_ratio=0.45,
                defender_ratio=0.30,
                expander_ratio=0.25,
                diplomacy_message="Deal! Let's stitch border hearts together!",
                reasoning="Accepting alliance to secure our border.",
            )
        if faction.integrity < 0.28 and faction.low_integrity_epochs >= 2:
            return FactionDecision(
                action="EXODUS",
                builder_ratio=0.70,
                defender_ratio=0.20,
                expander_ratio=0.10,
                diplomacy_message=f"{faction.name} migrates to open pixels!",
                reasoning="Territory overwhelmed; relocating survivors.",
            )
        if faction.integrity >= 0.80:
            close_neutral = [n for n in neighbors if n["relation"] == "NEUTRAL" and n["distance"] < 28]
            if close_neutral:
                tgt = close_neutral[0]
                if random.random() < 0.55:
                    return FactionDecision(
                        action="PROPOSE_ALLIANCE",
                        target_faction_id=tgt["id"],
                        builder_ratio=0.40,
                        defender_ratio=0.30,
                        expander_ratio=0.30,
                        diplomacy_message=f"Hey {tgt['name']}, truce + border hearts?",
                        reasoning="Proposing peaceful border-heart treaty.",
                    )
                return FactionDecision(
                    action="RAID",
                    target_faction_id=tgt["id"],
                    builder_ratio=0.30,
                    defender_ratio=0.25,
                    expander_ratio=0.45,
                    diplomacy_message=f"{faction.name} challenges {tgt['name']} for the border!",
                    reasoning="Contesting border territory against neighbor.",
                )
            if not faction.has_border_frame:
                return FactionDecision(
                    action="FORTIFY",
                    builder_ratio=0.45,
                    defender_ratio=0.45,
                    expander_ratio=0.10,
                    diplomacy_message=f"Frame & defend {faction.banner_text or faction.name}!",
                    reasoning="Adding ornate pixel border and nameplate.",
                )
            if can_morph and random.random() < 0.5:
                return FactionDecision(
                    action="MORPH_ART",
                    new_art_prompt=f"{faction.art_prompt}, upgraded detailed pixel art in {faction.color_theme}",
                    builder_ratio=0.65,
                    defender_ratio=0.25,
                    expander_ratio=0.10,
                    diplomacy_message=f"{faction.name} evolves to Stage {faction.evolution_stage + 1}!",
                    reasoning="Upgrading artwork in signature color palette.",
                )
            d = random.choice(["NORTH", "SOUTH", "EAST", "WEST"])
            return FactionDecision(
                action="EXPAND",
                direction=d,  # type: ignore[arg-type]
                new_art_prompt=f"expanded {faction.art_prompt} in {faction.color_theme}",
                builder_ratio=0.55,
                defender_ratio=0.25,
                expander_ratio=0.20,
                diplomacy_message=f"{faction.name} pushes {d.lower()}!",
                reasoning=f"Growing territory {d.lower()}.",
            )
        return FactionDecision(
            action="BUILD",
            builder_ratio=0.65,
            defender_ratio=0.25,
            expander_ratio=0.10,
            diplomacy_message=faction.slogan,
            reasoning="Continuing construction of core artwork.",
        )

    def design_typography(
        self,
        name: str,
        theme: str,
        art_mode: str,
        art_prompt: str,
        banner_text: str,
        color_theme: str,
        primary_idx: int,
        secondary_idx: int,
        accent_idx: int,
        faction_index: int = 0,
    ) -> tuple[TypographyStyleSpec, str]:
        """Ask Qwen2.5-VL to design typography (font, colors, 3D effects, layout) for this faction."""
        must_have_text = art_mode == "banner"
        # Nudge ~35% of non-banner factions toward pure iconic art (no text) like real r/place
        want_pure_art = (not must_have_text) and ((faction_index % 3) == 1)
        suggested_font = FONT_STYLES[faction_index % len(FONT_STYLES)]
        color_effects = ("vertical_gradient", "per_letter_palette", "glossy_highlight", "solid", "rainbow_letters", "striped_fill")
        border_effects = ("extrude_3d", "double_sticker", "neon_glow", "outline_1px")
        layouts = ("straight", "wavy", "arched", "italic_slant", "stacked", "vertical_side")
        suggested_ce = color_effects[faction_index % len(color_effects)]
        suggested_be = border_effects[(faction_index // 2) % len(border_effects)]
        suggested_lay = layouts[(faction_index // 3) % len(layouts)]

        system = (
            "You are a pixel-art typographer designing intentional, bespoke lettering for an r/place community. "
            "Every faction has its OWN unique typography personality—medieval gothic gold gradients for kingdoms, "
            "chunky 3D extruded neon arcade letters for gamers, bubbly candy-colored letters for cute mascots, "
            "tiny lowercase r/sub tags tucked in a corner for subtle communities, or NO text at all for pure art!"
        )
        pure_hint = (
            "For this mural, consider setting include_text=false so the artwork shines on its own (unless text is essential)."
            if want_pure_art
            else "This is a dedicated wordmark/logo banner — set include_text=true and size_tier='large'!"
            if must_have_text
            else f"Design a distinct typographic lockup (consider font_style='{suggested_font}', color_effect='{suggested_ce}', border_effect='{suggested_be}', layout='{suggested_lay}')."
        )
        user_text = (
            f"Faction: '{name}' | Theme: '{theme}' | Mode: {art_mode}\n"
            f"Artwork: '{art_prompt}' | Candidate text: '{banner_text}'\n"
            f"Signature palette: {color_theme}\n"
            f"Allowed palette names: {', '.join(COLOR_NAMES)}.\n"
            f"{pure_hint}\n"
            "Pick colorful, creative fill/gradient/outline/3D shadow colors (NOT plain white on black every time!). Return JSON."
        )

        res = self._chat_json(system, user_text, None, TypographyDesign, temperature=0.90)
        if isinstance(res, TypographyDesign):
            inc = True if must_have_text else (False if want_pure_art and not res.include_text else res.include_text)
            if want_pure_art and (faction_index % 3 == 1):
                # Honor pure-art breathing room reliably when VLM still leans yes ~60% of the time
                inc = res.include_text and (random.random() < 0.35)
            engine: Literal["styled", "flux", "code"] = "styled"
            if must_have_text:
                engine = "flux" if (faction_index % 2 == 0) else "code"
            spec = TypographyStyleSpec(
                include_text=inc,
                text=(res.text or banner_text or name[:10]).strip()[:14],
                casing=res.casing,
                font_style=res.font_style,
                size_tier="large" if must_have_text else res.size_tier,
                color_effect=res.color_effect,
                border_effect=res.border_effect,
                layout=res.layout if not (must_have_text and res.layout == "vertical_side") else "wavy",
                placement="center" if must_have_text else res.placement,
                fill_color=resolve_color(res.fill_color, primary_idx),
                fill_color_2=resolve_color(res.fill_color_2, secondary_idx),
                fill_color_3=resolve_color(res.fill_color_3, accent_idx),
                outline_color=resolve_color(res.outline_color, BLACK_IDX),
                shadow_color=resolve_color(res.shadow_color, BLACK_IDX),
                use_ribbon=False if must_have_text else res.use_ribbon,
                ribbon_color=primary_idx,
                engine=engine,
            )
            return spec, res.flux_style_desc.strip()

        # Rich diverse heuristic fallback
        engine = ("flux" if (faction_index % 2 == 0) else "code") if must_have_text else "styled"
        spec = TypographyStyleSpec(
            include_text=must_have_text or (not want_pure_art),
            text=(banner_text or name[:10]).strip()[:14],
            casing=("subreddit" if faction_index % 4 == 0 else ("title" if faction_index % 4 == 1 else "upper")),
            font_style=suggested_font,
            size_tier="large" if must_have_text else ("tiny" if faction_index % 4 == 0 else "medium"),
            color_effect=suggested_ce,  # type: ignore[arg-type]
            border_effect=suggested_be,  # type: ignore[arg-type]
            layout=suggested_lay,  # type: ignore[arg-type]
            placement="center" if must_have_text else ("bottom_right_tag" if faction_index % 4 == 0 else "bottom_center"),
            fill_color=primary_idx if primary_idx != BLACK_IDX else WHITE_IDX,
            fill_color_2=secondary_idx,
            fill_color_3=accent_idx,
            outline_color=BLACK_IDX,
            shadow_color=secondary_idx,
            use_ribbon=False,
            engine=engine,
        )
        return spec, f"{suggested_font} 3D pixel arcade letters with sparkles"

    def verify_wordmark_ocr(self, img: Image.Image, expected_text: str) -> tuple[bool, str]:
        """Use Qwen2.5-VL vision + strict edit-distance gate (REQ-NEXT-001) to verify FLUX wordmarks."""
        system = (
            "You are a strict pixel-art quality reviewer. Transcribe the EXACT letters shown in this pixel-art logo "
            "without auto-correcting typos, and score its visual quality from 1 to 10."
        )
        user_text = (
            f"Intended text: '{expected_text}'.\n"
            "Read the exact characters rendered in the image (do NOT fix typos in read_text!). "
            "Set readable=true ONLY if every letter is clear and sharp. Return JSON."
        )
        res = self._chat_json(system, user_text, img, WordmarkOCRCheck, temperature=0.1)
        if isinstance(res, WordmarkOCRCheck):
            exact_ok = ocr_text_matches(expected_text, res.read_text)
            ok = res.readable and res.quality_score >= 6 and exact_ok
            return ok, f"read='{res.read_text}' score={res.quality_score}/10 match={exact_ok} ok={ok}"
        return False, "ocr_check_failed"

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
