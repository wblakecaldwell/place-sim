"""Local OpenAI-compatible Vision-Language brain (LM Studio MLX / Ollama) for AI-first faction creation, coding & diplomacy."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import re
import subprocess
from typing import Literal

import httpx
from PIL import Image
from pydantic import BaseModel, Field

from pixel_battle.art.typography import FONT_STYLES, TypographyStyleSpec
from pixel_battle.factions import DiplomacyMessage, Faction
from pixel_battle.palette import BLACK_IDX, COLOR_NAMES, WHITE_IDX, resolve_color

log = logging.getLogger(__name__)


def load_local_api_token(keychain_service: str = "place-sim") -> str:
    """Load local LM Studio API token from LMSTUDIO_API_KEY or macOS Keychain (never stored in files)."""
    env_tok = os.environ.get("LMSTUDIO_API_KEY", "").strip()
    if env_tok:
        return env_tok
    if not keychain_service:
        return ""
    try:
        user = os.environ.get("USER", "")
        if not user:
            return ""
        return subprocess.check_output(
            ["security", "find-generic-password", "-a", user, "-s", keychain_service, "-w"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3.0,
        ).strip()
    except Exception:
        return ""


class AIFactionIdea(BaseModel):
    name: str = Field(description="Short distinct faction name (2-24 chars)")
    timezone_region: Literal["AMERICAS", "EUROPE", "ASIA_PACIFIC", "LATAM", "GLOBAL"] = "GLOBAL"
    language: str = Field(default="en", description="ISO language code: en, zh, es, fr, de, ja, ko, pt, tr")
    category: str = Field(default="art", description="e.g. comic, cartoon, movie, superhero, meme, international, logo, geometric, game, anime")
    slogan: str = Field(default="", description="Short rallying slogan in the faction's language")
    art_mode: Literal["flux_art", "flux_comic", "banner", "ai_code"] = "flux_art"
    aspect: Literal["square", "wide", "tall"] = "square"
    fluid_cutout: bool = Field(default=True, description="True for die-cut floating shapes/mascots/logos; False only for framed 3-panel comics")
    banner_text: str = Field(default="", description="Short 2-12 char wordmark text if banner or labeled emblem")
    art_prompt: str = Field(description="Concrete 15-35 word pixel-art visual description")
    primary_color: str = "blue"
    secondary_color: str = "yellow"
    accent_color: str = "white"


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
    """Interfaces with local LM Studio (MLX) or Ollama via OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:1234/v1",
        vlm_model: str = "qwen3-vl-30b-a3b-instruct-mlx",
        timeout: float = 150.0,
        keychain_service: str = "place-sim",
    ) -> None:
        base = ollama_url.rstrip("/")
        # Normalize base URL to OpenAI-compatible /v1 endpoint (works on LM Studio and Ollama)
        if not base.endswith("/v1"):
            self.api_base = f"{base}/v1"
        else:
            self.api_base = base
        self.ollama_url = base
        self.vlm_model = vlm_model
        self.timeout = timeout
        self._api_token = load_local_api_token(keychain_service)
        headers = {"Authorization": f"Bearer {self._api_token}"} if self._api_token else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self._fallback_idx = 0

    def check_health(self) -> bool:
        try:
            r = self._client.get(f"{self.api_base}/models", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            ids = [str(m.get("id", "")) for m in data.get("data", [])]
            if any(self.vlm_model in mid or mid in self.vlm_model for mid in ids):
                return True
            return len(ids) > 0
        except Exception:
            pass
        # Legacy Ollama /api/tags fallback
        try:
            root = self.api_base[:-3] if self.api_base.endswith("/v1") else self.api_base
            r = self._client.get(f"{root}/api/tags", timeout=5.0)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(self.vlm_model in n for n in names) or len(names) > 0
        except Exception:
            return False

    @staticmethod
    def _strip_think_and_fences(raw: str) -> str:
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        m = re.search(r"```(?:json|python)?\s*(.*?)```", clean, flags=re.S)
        if m:
            clean = m.group(1).strip()
        return clean

    def _chat_text(
        self,
        system: str,
        user_text: str,
        image: Image.Image | None = None,
        temperature: float = 0.80,
        max_tokens: int = 800,
    ) -> str | None:
        if image is not None:
            b64 = _pil_to_b64_png(image)
            user_content: object = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            user_content = user_text

        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = self._client.post(f"{self.api_base}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            log.warning("Local VLM call failed (%s); using fallback.", exc)
            return None

    def _chat_json(
        self,
        system: str,
        user_text: str,
        image: Image.Image | None,
        schema_cls: type[BaseModel],
        temperature: float = 0.80,
    ) -> BaseModel | None:
        schema_hint = json.dumps(schema_cls.model_json_schema())
        sys_full = (
            f"{system}\nRespond with ONLY a single valid JSON object matching this schema "
            f"(no extra text, no markdown):\n{schema_hint}"
        )
        raw = self._chat_text(sys_full, user_text, image=image, temperature=temperature, max_tokens=550)
        if not raw:
            return None
        clean = self._strip_think_and_fences(raw)
        # Extract outermost JSON object if extra prose leaked
        if not clean.startswith("{"):
            s = clean.find("{")
            e = clean.rfind("}")
            if s >= 0 and e > s:
                clean = clean[s : e + 1]
        try:
            return schema_cls.model_validate_json(clean)
        except Exception as exc:
            log.warning("VLM JSON parse failed for %s (%s)", schema_cls.__name__, exc)
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

    def brainstorm_faction_batch(
        self,
        count: int = 8,
        act: int = 1,
        virtual_hour: float = 12.0,
        prime_tz: str = "EUROPE",
        unlocked_colors: tuple[int, ...] | list[int] | None = None,
        existing_names: list[str] | None = None,
        subreddit_hints: list[str] | None = None,
    ) -> list[AIFactionIdea]:
        """Ask the local 30B VLM to invent a diverse, non-repeating batch of r/place factions from scratch."""
        col_names = (
            [COLOR_NAMES[c] for c in unlocked_colors if 0 <= c < len(COLOR_NAMES)]
            if unlocked_colors
            else COLOR_NAMES
        )
        avoid = ", ".join((existing_names or [])[-40:]) or "none"
        subs = ", ".join(f"r/{s}" for s in (subreddit_hints or [])[:6]) or "r/gaming, r/comics, r/cats, r/anime"

        system = (
            "You are the creative collective behind a giant live Reddit r/place simulation. "
            "Invent genuinely human, playful, non-violent, culturally diverse pixel-art communities. "
            "NEVER repeat previous names, NEVER use clichés (Radiant, Whirl, Cosmic, Vortex, Luminous), "
            "and keep all text/slogans friendly, witty, and safe for work."
        )
        user_text = (
            f"Current world state: Act {act} (Virtual Hour {virtual_hour:.1f}/72, prime-time region: {prime_tz}).\n"
            f"Unlocked palette ({len(col_names)} colors): {', '.join(col_names)}.\n"
            f"Optional subreddit inspiration: {subs}.\n"
            f"Already taken names (DO NOT repeat or clone): {avoid}.\n\n"
            f"Invent EXACTLY {count} wildly different new factions mixing:\n"
            "- International communities (Chinese zh, Spanish es, French fr, German de, Japanese ja, Korean ko, Portuguese pt, Turkish tr) paired with their matching timezone_region and short native-language text/slogan!\n"
            "- Pop-culture fandoms: TV shows, movies, superheroes, 90s/2000s cartoons, video games, anime, internet memes.\n"
            "- At least 1 original funny 3-panel short pixel comic strip (art_mode='flux_comic', aspect='wide', fluid_cutout=false, describing Panel 1, Panel 2, Panel 3 with black borders & white gutters).\n"
            "- At least 1 stylized multilingual or community text logo (art_mode='banner', aspect='wide', fluid_cutout=true, short 2-10 char banner_text in quotes).\n"
            "- At least 1 visionless Geometric Chaos cult (art_mode='ai_code', fluid_cutout=true, pure abstract lines/lattices/spirals/beams/optical patterns — ZERO characters!).\n"
            "- For ALL characters, mascots, superheroes, cartoons, memes, and logos, set fluid_cutout=true so they float organically without square boxes!\n\n"
            "Return ONLY a JSON array of objects with keys: "
            "name, timezone_region (AMERICAS|EUROPE|ASIA_PACIFIC|LATAM|GLOBAL), language, category, slogan, "
            "art_mode (flux_art|flux_comic|banner|ai_code), aspect (square|wide|tall), fluid_cutout (bool), "
            "banner_text, art_prompt (15-35 vivid visual words), primary_color, secondary_color, accent_color."
        )

        raw = self._chat_text(system, user_text, image=None, temperature=0.92, max_tokens=2400)
        if not raw:
            return []
        clean = self._strip_think_and_fences(raw)
        if not clean.startswith("["):
            s, e = clean.find("["), clean.rfind("]")
            if s >= 0 and e > s:
                clean = clean[s : e + 1]
        try:
            items = json.loads(clean)
            if not isinstance(items, list):
                return []
        except Exception as exc:
            log.warning("brainstorm_faction_batch JSON parse failed (%s)", exc)
            return []

        seen_lower = {n.lower() for n in (existing_names or [])}
        ideas: list[AIFactionIdea] = []
        for obj in items:
            if not isinstance(obj, dict):
                continue
            try:
                idea = AIFactionIdea.model_validate(obj)
            except Exception:
                continue
            nm = idea.name.strip()[:28]
            if not nm or nm.lower() in seen_lower:
                continue
            if any(bad in nm.lower() for bad in _BANNED_CLICHES):
                continue
            # Enforce fluid cutouts for non-comics
            if idea.art_mode != "flux_comic":
                idea.fluid_cutout = True
            seen_lower.add(nm.lower())
            idea.name = nm
            ideas.append(idea)
        return ideas

    def write_geometric_code(
        self,
        name: str,
        art_prompt: str,
        w: int,
        h: int,
        allowed_colors: tuple[int, ...] | list[int] | None = None,
    ) -> str:
        """Ask the local LLM to write a sandboxed Python draw(grid, w, h) script for geometric chaos art."""
        cols = allowed_colors if allowed_colors else list(range(len(COLOR_NAMES)))
        pal_desc = ", ".join(f"{c}={COLOR_NAMES[c]}" for c in cols if 0 <= c < len(COLOR_NAMES))
        system = (
            "You write compact, bug-free Python/NumPy pixel-art geometry functions. "
            "Only use `np` and `math` (do NOT write import statements). Return ONLY a Python code block."
        )
        user_text = (
            f"Write `def draw(grid, w, h):` that paints an original abstract geometric pixel pattern into `grid` "
            f"(a numpy int16 array of shape (h, w) pre-filled with -1 for transparent).\n"
            f"Faction: {name}\n"
            f"Pattern idea: {art_prompt}\n"
            f"Dimensions: {w}x{h}. Allowed color indices: {pal_desc}.\n"
            "Leave outer edges irregular or transparent (-1) where appropriate so it flows fluidly onto the canvas (no boring solid rectangle box!)."
        )
        raw = self._chat_text(system, user_text, image=None, temperature=0.72, max_tokens=850)
        return raw or ""

    def invent_streamer_raid(
        self,
        timezone_region: str,
        virtual_hour: float,
        target_names: list[str],
        unlocked_colors: tuple[int, ...] | list[int] | None = None,
    ) -> AIFactionIdea | None:
        """Invent a regional prime-time streamer flash-raid persona and bold die-cut raid emblem."""
        col_names = (
            [COLOR_NAMES[c] for c in unlocked_colors if 0 <= c < len(COLOR_NAMES)]
            if unlocked_colors
            else COLOR_NAMES
        )
        lang_hint = {
            "EUROPE": "French, Spanish, German, or English",
            "LATAM": "Spanish or Portuguese",
            "ASIA_PACIFIC": "Japanese, Korean, Chinese, or English",
            "AMERICAS": "English",
            "GLOBAL": "English",
        }.get(timezone_region, "English")
        system = (
            "You invent playful, energetic r/place streamer flash-raid mobs (like famous Twitch/YouTube community blitzes). "
            "Keep content fun, safe, and non-violent."
        )
        user_text = (
            f"Prime-time region: {timezone_region} (virtual hour {virtual_hour:.1f}/72, languages: {lang_hint}).\n"
            f"Target territory currently held by: {', '.join(target_names[:4]) or 'established factions'}.\n"
            f"Unlocked colors: {', '.join(col_names)}.\n"
            "Invent 1 bold streamer flash-raid faction with a punchy raid slogan and vivid pixel-art mascot/emblem prompt "
            "(art_mode='flux_art', aspect='square', fluid_cutout=true)."
        )
        res = self._chat_json(system, user_text, image=None, schema_cls=AIFactionIdea, temperature=0.92)
        if isinstance(res, AIFactionIdea):
            res.timezone_region = timezone_region  # type: ignore[assignment]
            res.art_mode = "flux_art"
            res.fluid_cutout = True
            return res
        return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
