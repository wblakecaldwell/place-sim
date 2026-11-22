"""Pixel-perfect Micro Cartoons/Memes/Game Sprites, Fine-Art Museum Frames, Empty-Space, Black Holes, Fractals & Void Monsters."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from pixel_battle.palette import (
    BLACK_IDX,
    BLUE_IDX,
    BURGUNDY_IDX,
    DARK_BLUE_IDX,
    DARK_RED_IDX,
    GREEN_IDX,
    NAME_TO_IDX,
    NUM_COLORS,
    ORANGE_IDX,
    PINK_IDX,
    PURPLE_IDX,
    RED_IDX,
    WHITE_IDX,
    YELLOW_IDX,
)

LIGHT_BLUE = NAME_TO_IDX["light_blue"]
LIGHT_GREEN = NAME_TO_IDX["light_green"]
DARK_GREEN = NAME_TO_IDX["dark_green"]
LIGHT_PINK = NAME_TO_IDX["light_pink"]
PALE_YELLOW = NAME_TO_IDX["pale_yellow"]
INDIGO = NAME_TO_IDX["indigo"]
DARK_PURPLE = NAME_TO_IDX["dark_purple"]
MAGENTA = NAME_TO_IDX["magenta"]
BEIGE = NAME_TO_IDX["beige"]
BROWN = NAME_TO_IDX["brown"]
DARK_BROWN = NAME_TO_IDX["dark_brown"]
GRAY = NAME_TO_IDX["gray"]
DARK_GRAY = NAME_TO_IDX["dark_gray"]
LIGHT_GRAY = NAME_TO_IDX["light_gray"]
TEAL = NAME_TO_IDX["teal"]
LIGHT_TEAL = NAME_TO_IDX["light_teal"]
LAVENDER = NAME_TO_IDX["lavender"]
PERIWINKLE = NAME_TO_IDX["periwinkle"]


# ---------------------------------------------------------------------------
# 1. Pixel-Perfect 1px-Outlined Micro Sprites (Memes, Cartoons, Indie/Retro Games)
# ---------------------------------------------------------------------------
# Legend: '.'=-1 (transparent), 'K'=black, 'W'=white, 'P'=primary, 'S'=secondary,
#         'A'=accent, 'R'=red/blush, 'E'=eye/visor, 'H'=hair/hat, 'B'=skin/beige

@dataclass(frozen=True)
class SpriteTemplate:
    key: str
    category: str  # "game" | "meme" | "anime"
    title: str
    grid: tuple[str, ...]
    palettes: tuple[dict[str, int], ...]


_MICRO_SPRITES: list[SpriteTemplate] = [
    SpriteTemplate(
        "amogus",
        "meme",
        "Sus Crewmate",
        (
            "....KKKKK...",
            "...KPPPPPK..",
            "..KPPKKKKK..",
            ".KKPKWEEEEK.",
            "KPKPKWEEEEK.",
            "KPKPPKKKKK..",
            "KPKPPPPPPK..",
            "KPKPPPPPPK..",
            ".KKPPSSPPK..",
            "..KPPSSPPK..",
            "..KPPKKPPK..",
            "..KPPK.KPK..",
            "..KKKK.KKK..",
        ),
        (
            {"P": RED_IDX, "S": DARK_RED_IDX, "E": LIGHT_BLUE},
            {"P": BLUE_IDX, "S": DARK_BLUE_IDX, "E": LIGHT_BLUE},
            {"P": GREEN_IDX, "S": DARK_GREEN, "E": LIGHT_BLUE},
            {"P": PINK_IDX, "S": MAGENTA, "E": LIGHT_BLUE},
            {"P": YELLOW_IDX, "S": ORANGE_IDX, "E": LIGHT_BLUE},
            {"P": PURPLE_IDX, "S": DARK_PURPLE, "E": LIGHT_BLUE},
            {"P": ORANGE_IDX, "S": RED_IDX, "E": LIGHT_BLUE},
            {"P": LIGHT_TEAL, "S": TEAL, "E": WHITE_IDX},
        ),
    ),
    SpriteTemplate(
        "kirby",
        "game",
        "Pink Puff",
        (
            "....KKKKKK....",
            "..KKPPPPPPKK..",
            ".KPPPPPPPPPPK.",
            ".KPPPKPPKPPPK.",
            "KPPPKWKPKWPPPK",
            "KPPPKPKPKPPPKK",
            "KPPRPPKKPPRPPK",
            "KPPPPPPPPPPPPK",
            ".KPPPPPPPPPPK.",
            "..KPPPPPPPPK..",
            ".KRRKKKKKKRRK.",
            "KRRRRK..KRRRRK",
            ".KKKK....KKKK.",
        ),
        (
            {"P": LIGHT_PINK, "R": PINK_IDX},
            {"P": PINK_IDX, "R": DARK_RED_IDX},
            {"P": YELLOW_IDX, "R": ORANGE_IDX},
            {"P": LIGHT_BLUE, "R": BLUE_IDX},
        ),
    ),
    SpriteTemplate(
        "frog_pepe",
        "meme",
        "Smug Frog",
        (
            "..KKKK..KKKK..",
            ".KPPPPKKPPPPK.",
            "KPPWWKPPKWWPPK",
            "KPPWKKPPKKWPPK",
            "KPPPPPPPPPPPPK",
            "KPPPPPPPPPPPPK",
            ".KPPKKKKKKPPK.",
            ".KPRRRRRRRRPK.",
            "..KPRRRRRRPK..",
            "..KKPPPPPPKK..",
            "...KSSSSSSK...",
            "..KSSSSSSSSK..",
            "..KKKKKKKKKK..",
        ),
        (
            {"P": GREEN_IDX, "S": DARK_BLUE_IDX, "R": RED_IDX},
            {"P": LIGHT_GREEN, "S": BLUE_IDX, "R": PINK_IDX},
            {"P": DARK_GREEN, "S": PURPLE_IDX, "R": DARK_RED_IDX},
        ),
    ),
    SpriteTemplate(
        "shiba_doge",
        "meme",
        "Pixel Doge",
        (
            ".KK........KK.",
            "KPKK......KKPK",
            "KPAPKKKKKKPAPK",
            "KPPPPPPPPPPPPK",
            "KPPPKPPPPKPPPK",
            "KPPPKPPPPKPPPK",
            "KPPWWPPPPWWPPK",
            "KPPWWWKKWWWPPK",
            ".KWWWWKKWWWWK.",
            ".KWWWKKKKWWWK.",
            "..KWWWRRWWWK..",
            "...KWWWWWWK...",
            "....KKKKKK....",
        ),
        (
            {"P": ORANGE_IDX, "A": BEIGE, "R": PINK_IDX},
            {"P": YELLOW_IDX, "A": PALE_YELLOW, "R": RED_IDX},
            {"P": BROWN, "A": BEIGE, "R": PINK_IDX},
        ),
    ),
    SpriteTemplate(
        "cat",
        "meme",
        "Pixel Cat",
        (
            ".KK........KK.",
            "KPKK......KKPK",
            "KPRPKKKKKKPRPK",
            "KPPPPPPPPPPPPK",
            "KPPKEPPPPKEPPK",
            "KPPKKPPPPKKPPK",
            "KPPPPPRRPPPPPK",
            ".KPPPKKKKPPPK.",
            "..KPPWWWWPPK..",
            "..KPPWWWWPPK..",
            "..KPPWWWWPPK..",
            "..KPPK..KPPK..",
            "..KKKK..KKKK..",
        ),
        (
            {"P": ORANGE_IDX, "R": LIGHT_PINK, "E": GREEN_IDX},
            {"P": GRAY, "R": LIGHT_PINK, "E": YELLOW_IDX},
            {"P": DARK_GRAY, "R": PINK_IDX, "E": LIGHT_BLUE},
            {"P": LAVENDER, "R": PINK_IDX, "E": PURPLE_IDX},
            {"P": WHITE_IDX, "R": LIGHT_PINK, "E": BLUE_IDX},
        ),
    ),
    SpriteTemplate(
        "nyan_cat",
        "meme",
        "Nyan Cat",
        (
            "RRRRR.KKKKKKKK..",
            "OOOOOKPPRPPPKKK.",
            "YYYYYKPPRPPPKPKK",
            "GGGGGKPPPRPPKPPK",
            "BBBBBKPPPKPKPPPK",
            "AAAAAKPPPRPKRPPK",
            "RRRRRKPPPPPPPPPK",
            "OOOOO.KKKKKKKKK.",
            ".......KK...KK..",
        ),
        (
            {"P": LIGHT_PINK, "R": RED_IDX, "A": PURPLE_IDX},
            {"P": PALE_YELLOW, "R": PINK_IDX, "A": MAGENTA},
        ),
    ),
    SpriteTemplate(
        "retro_kid",
        "game",
        "Earthbound Kid",
        (
            "....KKKKKK....",
            "...KHHHHHHKK..",
            "..KHHHHHHHHHK.",
            "..KKKKKKHHHHK.",
            "..KBBBBBBBBK..",
            "..KBKBBBBKBK..",
            "..KBRBBBBRBK..",
            "...KBBBBBBK...",
            "..KKPPPPPPKK..",
            ".KBBAAAAAAABK.",
            ".KBBPPPPPPBBK.",
            "...KSSSSSSK...",
            "...KSSKKSSK...",
            "...KKK..KKK...",
        ),
        (
            {"H": RED_IDX, "B": BEIGE, "R": LIGHT_PINK, "P": DARK_BLUE_IDX, "A": YELLOW_IDX, "S": BLUE_IDX},
            {"H": YELLOW_IDX, "B": BEIGE, "R": LIGHT_PINK, "P": RED_IDX, "A": YELLOW_IDX, "S": DARK_BLUE_IDX},
            {"H": GREEN_IDX, "B": BEIGE, "R": LIGHT_PINK, "P": BLUE_IDX, "A": WHITE_IDX, "S": DARK_GRAY},
        ),
    ),
    SpriteTemplate(
        "chibi_anime",
        "anime",
        "Chibi Star",
        (
            "...KKKKKKKK...",
            "..KHHHHHHHHK..",
            ".KHHHAHHHAHHK.",
            "KHHHBBBBBBHHHK",
            "KHHBBEBBBEBBHK",
            "KHHBKEBBBKEBHK",
            "KHHBRBKKBRBBHK",
            "KHHBBRRRRRBBHK",
            "KHHHKBBBBKHHHK",
            ".KHHKPPPPKHHK.",
            ".KHHKPAAPKHHK.",
            "..KKKPPPPKKK..",
            "....KKKKKK....",
        ),
        (
            {"H": LIGHT_TEAL, "A": LIGHT_BLUE, "B": PALE_YELLOW, "E": BLUE_IDX, "R": LIGHT_PINK, "P": DARK_GRAY},
            {"H": LIGHT_PINK, "A": WHITE_IDX, "B": PALE_YELLOW, "E": PURPLE_IDX, "R": PINK_IDX, "P": INDIGO},
            {"H": YELLOW_IDX, "A": WHITE_IDX, "B": BEIGE, "E": GREEN_IDX, "R": LIGHT_PINK, "P": DARK_BLUE_IDX},
            {"H": PURPLE_IDX, "A": LAVENDER, "B": PALE_YELLOW, "E": MAGENTA, "R": LIGHT_PINK, "P": BLACK_IDX},
            {"H": BROWN, "A": ORANGE_IDX, "B": BEIGE, "E": TEAL, "R": LIGHT_PINK, "P": DARK_RED_IDX},
        ),
    ),
    SpriteTemplate(
        "chad_shades",
        "meme",
        "Cool Shades",
        (
            "...KKKKKKKK...",
            "..KHHHHHHHHK..",
            ".KHHHHHHHHHHK.",
            ".KBBBBBBBBBBK.",
            "KKKKKKKKKKKKKK",
            "KKWWKKKKWWKKKK",
            ".KBBBBBBBBBBK.",
            ".KBBBBBBKBBBK.",
            "..KBBWWWWBBK..",
            "..KBBBBBBBBK..",
            "...KSSSSSSK...",
            "...KSSSSSSK...",
            "....KKKKKK....",
        ),
        (
            {"H": BLACK_IDX, "B": BEIGE, "S": GRAY},
            {"H": DARK_GRAY, "B": LIGHT_GRAY, "S": DARK_GRAY},
            {"H": YELLOW_IDX, "B": BEIGE, "S": BLUE_IDX},
        ),
    ),
    SpriteTemplate(
        "penguin",
        "game",
        "Pixel Penguin",
        (
            "....KKKKKK....",
            "...KPPPPPPK...",
            "..KPPWWWWPPK..",
            "..KPWKWWKWPK..",
            "..KPPWAAWPPK..",
            "..KPPWWWWPPK..",
            ".KPPPWWWWPPPK.",
            ".KPPPWWWWPPPK.",
            ".KPPPWWWWPPPK.",
            "..KPPWWWWPPK..",
            "...KKWWWWKK...",
            "..KAAKKKKAAK..",
            "..KKKK..KKKK..",
        ),
        (
            {"P": DARK_GRAY, "A": ORANGE_IDX},
            {"P": DARK_BLUE_IDX, "A": YELLOW_IDX},
            {"P": BLACK_IDX, "A": ORANGE_IDX},
        ),
    ),
    SpriteTemplate(
        "chicken",
        "game",
        "Stardew Chicken",
        (
            ".....KKK......",
            "....KRRRK.....",
            "...KPPPPPK....",
            "..KPKPPPKPK...",
            "..KPPAAAPPK...",
            "..KPPPRPPPK...",
            ".KPPPPPPPPPKK.",
            "KPPPPSSPPPPPPK",
            "KPPPPSSPPPPPK.",
            ".KPPPPPPPPPK..",
            "..KKKKKKKKK...",
            "...KA...AK....",
            "..KAA...AAK...",
        ),
        (
            {"P": WHITE_IDX, "S": LIGHT_GRAY, "R": RED_IDX, "A": ORANGE_IDX},
            {"P": BLUE_IDX, "S": LIGHT_BLUE, "R": RED_IDX, "A": YELLOW_IDX},
            {"P": BROWN, "S": DARK_BROWN, "R": DARK_RED_IDX, "A": ORANGE_IDX},
        ),
    ),
    SpriteTemplate(
        "green_owl",
        "meme",
        "Streak Owl",
        (
            "..KK......KK..",
            ".KPPKKKKKKPPK.",
            "KPPPPPPPPPPPPK",
            "KPPWWKPPKWWPPK",
            "KPPWKKPPKKWPPK",
            "KPPWWKAAKWWPPK",
            "KPPPPAAAAAPPPK",
            "KPPSSSSSSSSPPK",
            "KPPSSSSSSSSPPK",
            ".KPPPPPPPPPPK.",
            "..KPPKKKKPPK..",
            "..KAAK..KAAK..",
            "..KKKK..KKKK..",
        ),
        (
            {"P": LIGHT_GREEN, "S": GREEN_IDX, "A": ORANGE_IDX},
            {"P": PURPLE_IDX, "S": DARK_PURPLE, "A": YELLOW_IDX},
        ),
    ),
    SpriteTemplate(
        "duck",
        "meme",
        "Rubber Duck",
        (
            ".....KKKKK....",
            "....KPPPPPK...",
            "...KPPKPPPPK..",
            ".KKKPPPPPPPK..",
            "KAAAPPPPPPPK..",
            ".KKKPPPPPPK...",
            "...KPPPPPPKKK.",
            "..KPPPPSSPPPPK",
            ".KPPPPSSSSPPPK",
            ".KPPPPPPPPPPK.",
            "..KKKKKKKKKK..",
        ),
        (
            {"P": YELLOW_IDX, "S": ORANGE_IDX, "A": ORANGE_IDX},
            {"P": WHITE_IDX, "S": LIGHT_GRAY, "A": ORANGE_IDX},
            {"P": PINK_IDX, "S": MAGENTA, "A": YELLOW_IDX},
        ),
    ),
    SpriteTemplate(
        "hollow_mask",
        "game",
        "Hollow Knight",
        (
            ".KK........KK.",
            "KWWK......KWWK",
            "KWWWK....KWWWK",
            ".KWWWKKKKWWWK.",
            ".KWWWWWWWWWWK.",
            ".KWWKKWWKKWWK.",
            ".KWWKKWWKKWWK.",
            "..KWWWWWWWWK..",
            "...KKKKKKKK...",
            "..KPPPPPPPPK..",
            ".KPPSSSSSSPPK.",
            ".KPPSSSSSSPPK.",
            "..KKKKKKKKKK..",
        ),
        (
            {"P": DARK_GRAY, "S": GRAY},
            {"P": DARK_BLUE_IDX, "S": BLUE_IDX},
            {"P": DARK_RED_IDX, "S": RED_IDX},
        ),
    ),
    SpriteTemplate(
        "junimo_slime",
        "game",
        "Forest Sprite",
        (
            "......KK......",
            ".....KAK......",
            "...KKKKKKKK...",
            "..KPPPPPPPPK..",
            ".KPPPPPPPPPPK.",
            "KPPPKPPPPKPPPK",
            "KPPPKPPPPKPPPK",
            "KPPRPPKKPPRPPK",
            "KPPPPPPPPPPPPK",
            ".KPPPPPPPPPPK.",
            "..KKPPPPPPKK..",
            "...KK....KK...",
        ),
        (
            {"P": GREEN_IDX, "A": DARK_GREEN, "R": LIGHT_PINK},
            {"P": LIGHT_BLUE, "A": BLUE_IDX, "R": PINK_IDX},
            {"P": YELLOW_IDX, "A": ORANGE_IDX, "R": RED_IDX},
            {"P": PURPLE_IDX, "A": DARK_PURPLE, "R": LIGHT_PINK},
        ),
    ),
    SpriteTemplate(
        "mushroom",
        "game",
        "1-Up Shroom",
        (
            "....KKKKKK....",
            "..KKPPWWPPKK..",
            ".KPPWWWWWWPPK.",
            "KPWWPWWWWPWWPK",
            "KPWWPPPPPPWWPK",
            "KPPPPPPPPPPPPK",
            "KKKKKKKKKKKKKK",
            "..KWWKWWKWWK..",
            "..KWWKWWKWWK..",
            "..KWWWWWWWWK..",
            "...KWWWWWWK...",
            "....KKKKKK....",
        ),
        (
            {"P": RED_IDX},
            {"P": GREEN_IDX},
            {"P": PURPLE_IDX},
            {"P": BLUE_IDX},
        ),
    ),
    SpriteTemplate(
        "pokeball",
        "game",
        "Pixel Orb",
        (
            "....KKKKKK....",
            "..KKPPPPPPKK..",
            ".KPPWPPPPPPPK.",
            "KPPWWPPPPPPPPK",
            "KPPPPPPPPPPPPK",
            "KKKKKKWWKKKKKK",
            "KKKKKWAAWKKKKK",
            "KWWWWKWWKWWWWK",
            "KWWWWWWWWWWWWK",
            ".KWWWWWWWWWWK.",
            "..KKWWWWWWKK..",
            "....KKKKKK....",
        ),
        (
            {"P": RED_IDX, "A": WHITE_IDX},
            {"P": BLUE_IDX, "A": RED_IDX},
            {"P": PURPLE_IDX, "A": LIGHT_PINK},
            {"P": YELLOW_IDX, "A": BLACK_IDX},
        ),
    ),
    SpriteTemplate(
        "ghost_arcade",
        "game",
        "Arcade Ghost",
        (
            "....KKKKKK....",
            "..KKPPPPPPKK..",
            ".KPPPPPPPPPPK.",
            "KPPWWKPPKWWPPK",
            "KPPWKKPPKKWPPK",
            "KPPWWKPPKWWPPK",
            "KPPPPPPPPPPPPK",
            "KPPPPPPPPPPPPK",
            "KPPPPPPPPPPPPK",
            "KPPKPPKKPPKPPK",
            "KPK.KK..KK.KPK",
            "KK..........KK",
        ),
        (
            {"P": RED_IDX},
            {"P": LIGHT_BLUE},
            {"P": PINK_IDX},
            {"P": ORANGE_IDX},
            {"P": PURPLE_IDX},
        ),
    ),
    SpriteTemplate(
        "alien_invader",
        "game",
        "Space Invader",
        (
            "...K......K...",
            "....K....K....",
            "...KKKKKKKK...",
            "..KKPPKKPPKK..",
            ".KPPPPPPPPPPK.",
            "KPPKPPPPPPKPPK",
            "KPPKPPPPPPKPPK",
            "KPKKKKKKKKKKPK",
            "KPK.K....K.KPK",
            "....KK..KK....",
        ),
        (
            {"P": LIGHT_GREEN},
            {"P": LIGHT_BLUE},
            {"P": MAGENTA},
            {"P": YELLOW_IDX},
        ),
    ),
    SpriteTemplate(
        "pixel_sword",
        "game",
        "Pixel Blade",
        (
            "..........KKKK",
            ".........KWWPK",
            "........KWWPPK",
            ".......KWWPPK.",
            "......KWWPPK..",
            "..KK.KWWPPK...",
            ".KAAKWWPPK....",
            "..KAAWPPK.....",
            "...KAAAK......",
            "..KSSKAAK.....",
            ".KSSK.KAAK....",
            "KAAK...KK.....",
            "KKKK..........",
        ),
        (
            {"P": LIGHT_BLUE, "A": YELLOW_IDX, "S": DARK_BROWN},
            {"P": MAGENTA, "A": PALE_YELLOW, "S": DARK_PURPLE},
            {"P": LIGHT_GREEN, "A": ORANGE_IDX, "S": BROWN},
        ),
    ),
    SpriteTemplate(
        "potion",
        "game",
        "Mana Potion",
        (
            "....KKKKKK....",
            "....KAAAAK....",
            "....KKKKKK....",
            ".....KWWK.....",
            "....KWWWWK....",
            "...KWWPPWWK...",
            "..KWWPPPPWWK..",
            ".KWWPPPPPPWWK.",
            ".KWWPSPPPPWWK.",
            ".KWWPPPPPPWWK.",
            "..KPPPPPPPPK..",
            "...KKKKKKKK...",
        ),
        (
            {"P": BLUE_IDX, "S": LIGHT_BLUE, "A": BROWN},
            {"P": RED_IDX, "S": LIGHT_PINK, "A": BROWN},
            {"P": GREEN_IDX, "S": LIGHT_GREEN, "A": BROWN},
            {"P": PURPLE_IDX, "S": LAVENDER, "A": YELLOW_IDX},
        ),
    ),
    SpriteTemplate(
        "ufo",
        "meme",
        "Pixel UFO",
        (
            ".....KKKK.....",
            "...KKWWWWKK...",
            "..KWEEEEEEWK..",
            ".KKKKKKKKKKKK.",
            "KSSSASSSASSSSK",
            "KSSSSSSSSSSSSK",
            ".KKKKKKKKKKKK.",
            "...KAAAAAAK...",
            "..KAAAAAAAAK..",
            ".KAA......AAK.",
        ),
        (
            {"S": GRAY, "E": LIGHT_GREEN, "A": YELLOW_IDX},
            {"S": LIGHT_GRAY, "E": LIGHT_BLUE, "A": LIGHT_TEAL},
            {"S": PURPLE_IDX, "E": PINK_IDX, "A": PALE_YELLOW},
        ),
    ),
    SpriteTemplate(
        "boba_cup",
        "meme",
        "Boba Club",
        (
            ".......KK.....",
            "......KAK.....",
            "......KAK.....",
            "..KKKKKKKKKK..",
            ".KWWWWWWWWWWK.",
            "..KKKKKKKKKK..",
            "..KPPPPPPPPK..",
            "..KPPPPPPPPK..",
            "..KPSSPPSSPK..",
            "..KPPSSPPSSK..",
            "...KPSSPPSSK..",
            "...KKKKKKKK...",
        ),
        (
            {"P": LIGHT_PINK, "S": DARK_BROWN, "A": PINK_IDX},
            {"P": BEIGE, "S": BLACK_IDX, "A": ORANGE_IDX},
            {"P": LIGHT_GREEN, "S": DARK_BROWN, "A": GREEN_IDX},
            {"P": LAVENDER, "S": DARK_PURPLE, "A": PURPLE_IDX},
        ),
    ),
    SpriteTemplate(
        "skull_pixel",
        "meme",
        "Pixel Skull",
        (
            "...KKKKKKKK...",
            "..KWWWWWWWWK..",
            ".KWWWWWWWWWWK.",
            "KWWWWWWWWWWWWK",
            "KWWEEWWWWEEWWK",
            "KWWEEWWWWEEWWK",
            "KWWWWWKWWWWWWK",
            ".KWWWWWWWWWWK.",
            "..KKWWWWWWKK..",
            "...KWKWWKWK...",
            "...KWKWWKWK...",
            "...KKKKKKKK...",
        ),
        (
            {"E": BLACK_IDX},
            {"E": RED_IDX},
            {"E": LIGHT_BLUE},
            {"E": PURPLE_IDX},
        ),
    ),
    SpriteTemplate(
        "pepe_mini",
        "meme",
        "Mini Frog",
        (
            "..KKKK..KKKK..",
            ".KPPPPKKPPPPK.",
            "KPWWKKPPWWKKPK",
            "KPPPPPPPPPPPPK",
            "KPPPPPPPPPPPPK",
            "KPSSPPPPPPSSPK",
            "KPPPRRRRRRPPPK",
            ".KPPPRRRRPPPK.",
            "..KPPPPPPPPK..",
            "...KKKKKKKK...",
        ),
        (
            {"P": GREEN_IDX, "S": LIGHT_GREEN, "R": RED_IDX},
            {"P": LIGHT_GREEN, "S": GREEN_IDX, "R": DARK_RED_IDX},
        ),
    ),
    SpriteTemplate(
        "doge_mini",
        "meme",
        "Mini Doge",
        (
            ".KK........KK.",
            "KPPK......KPPK",
            "KPPPKKKKKKPPPK",
            "KPPPPPPPPPPPPK",
            "KPPKPPPPPPKPPK",
            "KPPWPPSSPPWPPK",
            "KPPSSSKKSSSPPK",
            ".KPSSSSSSSSPK.",
            "..KSSKKKKSSK..",
            "...KKKKKKKK...",
        ),
        (
            {"P": YELLOW_IDX, "S": PALE_YELLOW},
            {"P": ORANGE_IDX, "S": BEIGE},
        ),
    ),
    SpriteTemplate(
        "nyan_mini",
        "meme",
        "Mini Nyan",
        (
            "....KKKKKKKK..",
            "RR.KBBBBBBBBK.",
            "OOKBPPAPPPKPK.",
            "YYKBPPPPPKPPPK",
            "GGKBPPAPPKWKWK",
            "BBKBPPPPPKPPPK",
            "PP.KBBBBBBBBK.",
            "....KKK..KKK..",
        ),
        (
            {"B": BEIGE, "P": LIGHT_PINK, "A": MAGENTA},
            {"B": BEIGE, "P": LIGHT_BLUE, "A": YELLOW_IDX},
        ),
    ),
    SpriteTemplate(
        "duck_knife_mini",
        "meme",
        "Knife Duck",
        (
            "....KKKK......",
            "...KWWWWK.....",
            "..KWWKWWK.....",
            "..KWWWWWOOKKKK",
            ".KKWWWWWWKAAAK",
            "KWWWWWWWWKKKK.",
            "KWWWWWWWWK....",
            ".KWWWWWWK.....",
            "..KOO.OOK.....",
            "..KKK.KKK.....",
        ),
        (
            {"A": LIGHT_GRAY},
            {"A": LIGHT_BLUE},
        ),
    ),
    SpriteTemplate(
        "cool_sun",
        "cartoon",
        "Cool Sun",
        (
            "......KK......",
            "..K..KPPK..K..",
            ".KPKKKKKKKKPK.",
            "..KPPPPPPPPK..",
            "KKPKKKKKKKKPKK",
            "KPPKWKKPKWKKPK",
            "..KPPPPPPPPK..",
            ".KPKPPKKPPKPK.",
            "..K..KPPK..K..",
            "......KK......",
        ),
        (
            {"P": YELLOW_IDX},
            {"P": ORANGE_IDX},
        ),
    ),
    SpriteTemplate(
        "capy_mini",
        "cartoon",
        "Yuzu Capy",
        (
            ".....KKK......",
            "....KAAAK.....",
            "....KKKKK.....",
            "..KKKKKKKKK...",
            ".KPPPPPPPPPK..",
            "KPPKKPPPKKPPK.",
            "KPPPPPPPPSSPK.",
            "KPPPPPPPPSSPK.",
            ".KPPPPPPPPPK..",
            "..KKKKKKKKK...",
        ),
        (
            {"P": BROWN, "S": DARK_BROWN, "A": ORANGE_IDX},
            {"P": BEIGE, "S": BROWN, "A": YELLOW_IDX},
        ),
    ),
    SpriteTemplate(
        "cat_loaf",
        "cartoon",
        "Cat Loaf",
        (
            "..KK....KK....",
            ".KPPK..KPPK...",
            ".KPPKKKKPPKK..",
            "KPPPPPPPPPPPK.",
            "KPPKPPPPKPPPPK",
            "KPPPPRPPPPPSSK",
            "KPPPPPPPPPPSSK",
            "KPPPPPPPPPPPPK",
            ".KKKKKKKKKKKK.",
        ),
        (
            {"P": ORANGE_IDX, "S": BROWN, "R": PINK_IDX},
            {"P": WHITE_IDX, "S": LIGHT_GRAY, "R": PINK_IDX},
            {"P": GRAY, "S": DARK_GRAY, "R": LIGHT_PINK},
        ),
    ),
    SpriteTemplate(
        "shades_cat",
        "meme",
        "Shades Cat",
        (
            ".KK......KK...",
            "KPPK....KPPK..",
            "KPPPKKKKPPPK..",
            "KPPPPPPPPPPK..",
            "KKKKKKKKKKKK..",
            "KKWKKPPKKWKK..",
            "KPPPKKKKPPPK..",
            ".KPPPRRPPPK...",
            "..KPPPPPPK....",
            "...KKKKKK.....",
        ),
        (
            {"P": WHITE_IDX, "R": PINK_IDX},
            {"P": YELLOW_IDX, "R": ORANGE_IDX},
            {"P": LIGHT_BLUE, "R": PINK_IDX},
        ),
    ),
    SpriteTemplate(
        "banana_dance",
        "meme",
        "Pixel Banana",
        (
            ".......KKK....",
            "......KSSK....",
            ".....KPPK.....",
            "....KPPPK.....",
            "...KPPKPPK....",
            "...KPPKPPK....",
            "..KPPPKPPK....",
            "..KPPKKPPK....",
            "...KPPPPK.....",
            "....KSSK......",
            "....KKKK......",
        ),
        (
            {"P": YELLOW_IDX, "S": BROWN},
            {"P": PALE_YELLOW, "S": DARK_BROWN},
        ),
    ),
    SpriteTemplate(
        "troll_mini",
        "meme",
        "Grin Face",
        (
            "..KKKKKKKKK...",
            ".KWWWWWWWWWK..",
            "KWWKKWWWKKWWK.",
            "KWWWWWWWWWWWK.",
            "KWKKKKKKKKKWK.",
            "KWKWKWKWKWKWK.",
            ".KWKKKKKKKKK..",
            "..KWWWWWWWK...",
            "...KKKKKKK....",
        ),
        (
            {"P": WHITE_IDX},
            {"P": PALE_YELLOW},
        ),
    ),
]

SPRITE_BY_KEY: dict[str, SpriteTemplate] = {s.key: s for s in _MICRO_SPRITES}
SPRITE_KEYS: list[str] = [s.key for s in _MICRO_SPRITES]


def generate_micro_sprite(
    sprite_key: str | None = None,
    target_w: int = 20,
    target_h: int = 20,
    seed: int = 42,
    category: str | None = None,
) -> tuple[np.ndarray, SpriteTemplate]:
    """Render a crisp 1px-outlined micro sprite cutout (-1 background) with palette/flip/crown variation."""
    rng = random.Random(seed)
    if sprite_key and sprite_key in SPRITE_BY_KEY:
        tmpl = SPRITE_BY_KEY[sprite_key]
    else:
        pool = [s for s in _MICRO_SPRITES if (category is None or s.category == category)] or _MICRO_SPRITES
        tmpl = rng.choice(pool)

    pal = dict(rng.choice(tmpl.palettes))
    rows = tmpl.grid
    sh = len(rows)
    sw = max(len(r) for r in rows)

    raw = np.full((sh, sw), -1, dtype=np.int16)
    base_map = {
        "K": BLACK_IDX,
        "W": WHITE_IDX,
        "R": pal.get("R", RED_IDX),
        "O": ORANGE_IDX,
        "Y": YELLOW_IDX,
        "G": GREEN_IDX,
        "B": pal.get("B", BLUE_IDX),
        "P": pal.get("P", PINK_IDX),
        "S": pal.get("S", DARK_BLUE_IDX),
        "A": pal.get("A", YELLOW_IDX),
        "E": pal.get("E", LIGHT_BLUE),
        "H": pal.get("H", RED_IDX),
    }
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != ".":
                raw[y, x] = base_map.get(ch, BLACK_IDX)

    if rng.random() < 0.45:
        raw = np.fliplr(raw)

    # Upscale 2x cleanly only if target box is big enough (e.g. 26-32px)
    scale = 2 if (target_w >= sw * 2 and target_h >= sh * 2) else 1
    if scale > 1:
        raw = np.repeat(np.repeat(raw, scale, axis=0), scale, axis=1)
        sh, sw = raw.shape

    out_h = max(sh, target_h)
    out_w = max(sw, target_w)
    out = np.full((out_h, out_w), -1, dtype=np.int16)
    oy = (out_h - sh) // 2
    ox = (out_w - sw) // 2
    out[oy : oy + sh, ox : ox + sw] = raw

    # Optional tiny pixel heart or crown accent above sprite (~20% chance if space allows)
    if oy >= 3 and rng.random() < 0.20:
        cx = out_w // 2
        out[oy - 2, max(0, cx - 2) : min(out_w, cx + 3)] = YELLOW_IDX
        out[oy - 3, max(0, cx - 2)] = YELLOW_IDX
        out[oy - 3, cx] = YELLOW_IDX
        out[oy - 3, min(out_w - 1, cx + 2)] = YELLOW_IDX
        out[oy - 1, max(0, cx - 2) : min(out_w, cx + 3)] = BLACK_IDX

    # Trim to exact target_w x target_h centered if needed
    if out.shape[0] > target_h or out.shape[1] > target_w:
        cy0 = max(0, (out.shape[0] - target_h) // 2)
        cx0 = max(0, (out.shape[1] - target_w) // 2)
        out = out[cy0 : cy0 + target_h, cx0 : cx0 + target_w]
    return out, tmpl


# ---------------------------------------------------------------------------
# 2. Empty-Space / Clean-Canvas Purist Factions (REQ-ART-011)
# ---------------------------------------------------------------------------

def generate_clean_space(w: int, h: int, style: str = "pure_white", seed: int = 0) -> np.ndarray:
    """Generate an intentional pure-white blank-canvas enclave defending negative space."""
    bp = np.full((h, w), WHITE_IDX, dtype=np.int16)
    if style == "corner_ticks" and w >= 14 and h >= 14:
        # Subtle 1px minimalist corner bracket marks in light_gray
        L = min(4, w // 4, h // 4)
        bp[0, :L] = LIGHT_GRAY
        bp[:L, 0] = LIGHT_GRAY
        bp[0, -L:] = LIGHT_GRAY
        bp[:L, -1] = LIGHT_GRAY
        bp[-1, :L] = LIGHT_GRAY
        bp[-L:, 0] = LIGHT_GRAY
        bp[-1, -L:] = LIGHT_GRAY
        bp[-L:, -1] = LIGHT_GRAY
    return bp


# ---------------------------------------------------------------------------
# 3. Fine-Art Paintings & Gold Museum Frames (REQ-ART-010)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FineArtSpec:
    key: str
    name: str
    artist: str
    prompt: str
    color_theme: str
    primary: int
    secondary: int
    accent: int


FINE_ART_CATALOG: list[FineArtSpec] = [
    FineArtSpec("starry_night", "Starry Night", "Van Gogh", "Van Gogh Starry Night swirling blue sky yellow crescent moon dark cypress tree village", "deep cobalt blue, bright yellow, and light blue", DARK_BLUE_IDX, YELLOW_IDX, LIGHT_BLUE),
    FineArtSpec("great_wave", "The Great Wave", "Hokusai", "Hokusai Great Wave off Kanagawa crashing blue ocean wave white foam Mount Fuji cream sky", "dark blue, crisp white, and pale yellow", DARK_BLUE_IDX, WHITE_IDX, BLUE_IDX),
    FineArtSpec("the_scream", "The Scream", "Munch", "Edvard Munch The Scream orange red swirling sunset sky bridge dark blue fjord screaming figure", "bright orange, dark red, and dark blue", ORANGE_IDX, DARK_RED_IDX, DARK_BLUE_IDX),
    FineArtSpec("sunflowers", "Sunflowers", "Van Gogh", "Van Gogh bright yellow and orange sunflowers bouquet in golden vase on pale yellow background", "bright yellow, orange, and pale yellow", YELLOW_IDX, ORANGE_IDX, BROWN),
    FineArtSpec("pearl_earring", "Pearl Earring", "Vermeer", "Vermeer Girl with a Pearl Earring portrait blue yellow headscarf dark background glowing pearl", "dark blue, yellow, and black", DARK_BLUE_IDX, YELLOW_IDX, BLACK_IDX),
    FineArtSpec("mona_lisa", "Mona Lisa", "Da Vinci", "Leonardo Da Vinci Mona Lisa portrait smiling woman dark robe warm Renaissance landscape", "brown, dark green, and beige", BROWN, DARK_GREEN, BEIGE),
    FineArtSpec("milkmaid", "The Milkmaid", "Vermeer", "Vermeer The Milkmaid yellow bodice blue apron pouring milk near bright sunlit wall", "yellow, dark blue, and pale yellow", YELLOW_IDX, DARK_BLUE_IDX, PALE_YELLOW),
    FineArtSpec("wanderer", "Sea of Fog", "Friedrich", "Caspar David Friedrich Wanderer above the Sea of Fog dark silhouette on rocky peak white mist", "dark gray, light blue, and white", DARK_GRAY, LIGHT_BLUE, WHITE_IDX),
    FineArtSpec("water_lilies", "Water Lilies", "Monet", "Claude Monet Water Lilies pond pastel green bridge pink floating blossoms blue reflections", "green, light blue, and light pink", GREEN_IDX, LIGHT_BLUE, LIGHT_PINK),
    FineArtSpec("the_kiss", "The Kiss", "Klimt", "Gustav Klimt The Kiss golden mosaic geometric robes warm yellow orange pattern on dark background", "yellow, orange, and dark brown", YELLOW_IDX, ORANGE_IDX, DARK_BROWN),
]


def wrap_museum_frame(inner: np.ndarray) -> np.ndarray:
    """Wrap a full-bleed pixel painting inside a 2-3px ornate gold/brown museum pixel frame."""
    h, w = inner.shape
    out = inner.copy()
    # Replace any stray transparent pixels inside painting with dark_blue/dark_brown
    out = np.where(out < 0, DARK_BROWN, out)
    if h < 12 or w < 12:
        return out
    # Outer 1px black edge
    out[0, :] = BLACK_IDX
    out[-1, :] = BLACK_IDX
    out[:, 0] = BLACK_IDX
    out[:, -1] = BLACK_IDX
    # Gold gilded molding ring (alternating yellow & orange/brown)
    yy, xx = np.mgrid[0:h, 0:w]
    gold = np.where(((xx + yy) % 2) == 0, YELLOW_IDX, ORANGE_IDX)
    out[1, 1:-1] = gold[1, 1:-1]
    out[-2, 1:-1] = gold[-2, 1:-1]
    out[1:-1, 1] = gold[1:-1, 1]
    out[1:-1, -2] = gold[1:-1, -2]
    if h >= 22 and w >= 22:
        # Inner dark_brown bevel shadow
        out[2, 2:-2] = DARK_BROWN
        out[-3, 2:-2] = DARK_BROWN
        out[2:-2, 2] = DARK_BROWN
        out[2:-2, -3] = DARK_BROWN
        # Corner gold rosettes
        for cy in (1, h - 3):
            for cx in (1, w - 3):
                out[cy : cy + 2, cx : cx + 2] = PALE_YELLOW
    return out


def generate_procedural_painting(spec: FineArtSpec, w: int, h: int, seed: int = 42) -> np.ndarray:
    """Crisp procedural fine-art miniature (used directly for tiny paintings or as fallback)."""
    bp = np.full((h, w), DARK_BLUE_IDX, dtype=np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = xx / max(1.0, w - 1)
    ny = yy / max(1.0, h - 1)

    if spec.key == "starry_night":
        # Deep indigo/blue swirling sky + bright yellow moon & stars + dark cypress silhouette
        swirl = np.sin(nx * 10.0 + np.cos(ny * 7.0) * 2.2) + np.cos((nx - 0.5) * 8.0 + ny * 9.0)
        bp[:, :] = DARK_BLUE_IDX
        bp[swirl > 0.35] = BLUE_IDX
        bp[swirl > 1.05] = LIGHT_BLUE
        # Crescent moon top-right
        mcx, mcy = w * 0.78, h * 0.25
        md = np.sqrt((xx - mcx) ** 2 + (yy - mcy) ** 2)
        bp[md <= max(2.2, min(w, h) * 0.16)] = YELLOW_IDX
        bp[md <= max(1.2, min(w, h) * 0.11)] = PALE_YELLOW
        # Glowing spiral stars
        for sx, sy in ((0.28, 0.28), (0.52, 0.24), (0.42, 0.44)):
            sd = np.sqrt((xx - w * sx) ** 2 + (yy - h * sy) ** 2)
            bp[sd <= max(1.5, min(w, h) * 0.08)] = YELLOW_IDX
            bp[sd <= max(0.8, min(w, h) * 0.04)] = WHITE_IDX
        # Rolling dark hills + tiny warm village windows
        hill = ny > (0.68 + 0.07 * np.sin(nx * 6.5))
        bp[hill] = INDIGO
        bp[ny > 0.78] = DARK_BROWN
        # Towering cypress tree left
        cypress = (np.abs(nx - 0.20) < (0.09 * (ny ** 0.7) + 0.02 * np.sin(ny * 18.0))) & (ny > 0.18)
        bp[cypress] = BLACK_IDX
    elif spec.key == "great_wave":
        # Cream sky -> sweeping dark-blue wave curling right -> white foam crest -> Mt Fuji
        bp[:, :] = PALE_YELLOW
        bp[ny < 0.24] = BEIGE
        # Mt Fuji in distance right-center
        fuji = (ny > 0.56) & (np.abs(nx - 0.68) < (ny - 0.52) * 0.9)
        bp[fuji] = DARK_BLUE_IDX
        bp[fuji & (ny < 0.64)] = WHITE_IDX
        # Giant sweeping wave arc
        crest_y = 0.28 + 0.34 * ((nx - 0.38) ** 2) * 3.5 + 0.06 * np.sin(nx * 14.0)
        wave = (ny >= crest_y) & (nx <= 0.78 + 0.15 * ny)
        bp[wave] = BLUE_IDX
        bp[wave & (ny > crest_y + 0.12)] = DARK_BLUE_IDX
        foam = wave & (ny <= crest_y + 0.08)
        bp[foam] = WHITE_IDX
        bp[ny > 0.78] = DARK_BLUE_IDX
    elif spec.key == "the_scream":
        # Fiery wavy orange/red sunset sky -> dark blue river -> diagonal brown bridge -> figure
        wave = np.sin(nx * 9.0 + ny * 3.0) * 0.06
        bp[ny < 0.45 + wave] = ORANGE_IDX
        bp[(ny >= 0.12 + wave) & (ny < 0.26 + wave)] = RED_IDX
        bp[(ny >= 0.26 + wave) & (ny < 0.36 + wave)] = YELLOW_IDX
        bp[ny >= 0.45 + wave] = DARK_BLUE_IDX
        bridge = ny > (0.50 + (0.65 - nx) * 0.55)
        bp[bridge] = BROWN
        bp[bridge & (np.abs(ny - (0.50 + (0.65 - nx) * 0.55)) < 0.05)] = DARK_BROWN
        # Screaming figure center-bottom
        head = ((nx - 0.52) / 0.10) ** 2 + ((ny - 0.62) / 0.13) ** 2 <= 1.0
        body = (np.abs(nx - 0.52) < 0.12) & (ny >= 0.72)
        bp[body] = BLACK_IDX
        bp[head] = BEIGE
        bp[(np.abs(nx - 0.52) < 0.03) & (np.abs(ny - 0.66) < 0.04)] = BLACK_IDX
    elif spec.key == "sunflowers":
        bp[:, :] = PALE_YELLOW
        bp[ny > 0.68] = ORANGE_IDX
        # Vase
        vase = ((nx - 0.5) / 0.19) ** 2 + ((ny - 0.74) / 0.22) ** 2 <= 1.0
        bp[vase] = BROWN
        bp[vase & (ny < 0.70)] = YELLOW_IDX
        # Bright yellow/orange sunflower blooms
        for fx, fy, r in ((0.34, 0.38, 0.13), (0.54, 0.28, 0.14), (0.66, 0.44, 0.13), (0.46, 0.48, 0.12)):
            d = np.sqrt((nx - fx) ** 2 + (ny - fy) ** 2)
            bp[d <= r] = YELLOW_IDX
            bp[d <= r * 0.72] = ORANGE_IDX
            bp[d <= r * 0.42] = DARK_BROWN
    elif spec.key == "pearl_earring":
        # Dark background, face, blue headscarf, yellow drape, and white pearl
        bp[:, :] = BLACK_IDX
        # Shoulders/coat
        coat = (ny > 0.68) & (np.abs(nx - 0.50) < 0.34)
        bp[coat] = BROWN
        bp[coat & (nx < 0.45)] = ORANGE_IDX
        # Blue turban headscarf + flowing yellow tail
        turban = ((nx - 0.52) / 0.25) ** 2 + ((ny - 0.36) / 0.22) ** 2 <= 1.0
        bp[turban] = DARK_BLUE_IDX
        bp[turban & (ny < 0.34)] = BLUE_IDX
        tail = (nx > 0.62) & (nx < 0.78) & (ny > 0.38) & (ny < 0.74)
        bp[tail] = YELLOW_IDX
        # Face turning over shoulder
        face = ((nx - 0.45) / 0.16) ** 2 + ((ny - 0.46) / 0.19) ** 2 <= 1.0
        bp[face] = BEIGE
        bp[face & (nx < 0.43)] = PALE_YELLOW
        # Eyes, lips, and white pearl
        bp[(np.abs(nx - 0.40) < 0.04) & (np.abs(ny - 0.44) < 0.03)] = DARK_BROWN
        bp[(np.abs(nx - 0.50) < 0.04) & (np.abs(ny - 0.44) < 0.03)] = DARK_BROWN
        bp[(np.abs(nx - 0.44) < 0.05) & (np.abs(ny - 0.55) < 0.025)] = RED_IDX
        bp[np.sqrt((nx - 0.57) ** 2 + (ny - 0.56) ** 2) <= 0.055] = WHITE_IDX
    elif spec.key == "mona_lisa":
        # Soft hazy green-blue landscape sky, dark hair & robe, warm smiling Renaissance portrait
        bp[ny < 0.36] = LIGHT_BLUE
        bp[(ny >= 0.36) & (ny < 0.56)] = TEAL
        bp[ny >= 0.56] = DARK_GREEN
        # Dark veil/hair framing face + dark brown robe
        robe = (ny > 0.62) & (np.abs(nx - 0.50) < 0.34)
        bp[robe] = DARK_BROWN
        hair = ((nx - 0.50) / 0.22) ** 2 + ((ny - 0.44) / 0.28) ** 2 <= 1.0
        bp[hair] = BLACK_IDX
        face = ((nx - 0.50) / 0.13) ** 2 + ((ny - 0.38) / 0.17) ** 2 <= 1.0
        bp[face] = BEIGE
        neck = (np.abs(nx - 0.50) < 0.09) & (ny >= 0.50) & (ny <= 0.68)
        bp[neck] = BEIGE
        bp[(np.abs(nx - 0.45) < 0.03) & (np.abs(ny - 0.36) < 0.025)] = DARK_BROWN
        bp[(np.abs(nx - 0.55) < 0.03) & (np.abs(ny - 0.36) < 0.025)] = DARK_BROWN
        bp[(np.abs(nx - 0.50) < 0.05) & (np.abs(ny - 0.45) < 0.02)] = DARK_BROWN
    elif spec.key == "wanderer":
        # Misty sky -> white fog sea -> dramatic dark rocky crag + lone black silhouette overlooking peaks
        bp[ny < 0.42] = LIGHT_BLUE
        bp[(ny >= 0.42) & (ny < 0.72)] = WHITE_IDX
        bp[(ny >= 0.54) & (ny < 0.68) & (np.sin(nx * 10.0) > 0.2)] = LIGHT_GRAY
        rock = ny > (0.64 + 0.30 * np.abs(nx - 0.50))
        bp[rock] = DARK_GRAY
        bp[ny > (0.74 + 0.25 * np.abs(nx - 0.50))] = BLACK_IDX
        # Standing wanderer silhouette
        torso = (np.abs(nx - 0.50) < 0.08) & (ny >= 0.38) & (ny <= 0.74)
        head = ((nx - 0.50) / 0.05) ** 2 + ((ny - 0.33) / 0.06) ** 2 <= 1.0
        bp[torso | head] = BLACK_IDX
    elif spec.key == "water_lilies":
        # Pastel blue/green pond + arched green Japanese bridge + pink/white floating blossoms
        bp[:, :] = BLUE_IDX
        bp[ny < 0.32] = DARK_GREEN
        bp[(ny >= 0.22) & (ny < 0.45)] = GREEN_IDX
        arch = np.abs(ny - (0.36 + 0.16 * ((nx - 0.5) ** 2) * 4.0)) < 0.07
        bp[arch] = LIGHT_GREEN
        for lx, ly in ((0.22, 0.72), (0.44, 0.80), (0.70, 0.70), (0.82, 0.84), (0.34, 0.64)):
            d = np.sqrt(((nx - lx) / 0.09) ** 2 + ((ny - ly) / 0.05) ** 2)
            bp[d <= 1.0] = DARK_GREEN
            bp[d <= 0.45] = LIGHT_PINK
    else:
        # Milkmaid / Klimt The Kiss — rich golden/blue Renaissance composition
        bp[:, :] = spec.accent
        bp[ny > 0.52] = spec.secondary
        d = np.sqrt(((nx - 0.5) / 0.26) ** 2 + ((ny - 0.44) / 0.30) ** 2)
        bp[d <= 1.0] = spec.primary
        bp[(d <= 0.95) & (((xx + yy) % 3) == 0)] = YELLOW_IDX
        bp[((nx - 0.50) / 0.11) ** 2 + ((ny - 0.32) / 0.12) ** 2 <= 1.0] = BEIGE

    return wrap_museum_frame(bp)


# ---------------------------------------------------------------------------
# 4. Black Holes, Fractals & Void Monsters (REQ-ART-013)
# ---------------------------------------------------------------------------

def generate_black_hole(w: int, h: int, seed: int = 42) -> np.ndarray:
    """Circular Black Hole event horizon with swirling purple/orange/white accretion spiral & dark tendrils."""
    rng = random.Random(seed)
    bp = np.full((h, w), -1, dtype=np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx = (xx - cx) / max(1.0, w * 0.5)
    dy = (yy - cy) / max(1.0, h * 0.5)
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)

    # Logarithmic accretion spiral arms (purple/indigo/orange/white glow)
    n_arms = rng.choice([2, 3, 4])
    phase = rng.uniform(0, 6.28)
    spiral = np.sin(n_arms * ang - 6.5 * r + phase)
    outer_disk = (r <= 0.94) & ((spiral > 0.15) | (r <= 0.58))

    accent_scheme = rng.choice(["violet", "solar", "crimson"])
    if accent_scheme == "violet":
        c_outer, c_mid, c_hot = DARK_PURPLE, PURPLE_IDX, LAVENDER
    elif accent_scheme == "solar":
        c_outer, c_mid, c_hot = DARK_RED_IDX, ORANGE_IDX, YELLOW_IDX
    else:
        c_outer, c_mid, c_hot = BURGUNDY_IDX, DARK_RED_IDX, PINK_IDX

    bp[outer_disk] = BLACK_IDX
    bp[outer_disk & (spiral > 0.48) & (r > 0.34) & (r < 0.86)] = c_outer
    bp[outer_disk & (spiral > 0.72) & (r > 0.36) & (r < 0.72)] = c_mid
    # Glowing photon ring right around the event horizon
    photon_ring = (r >= 0.31) & (r <= 0.42)
    bp[photon_ring] = c_mid
    bp[(r >= 0.33) & (r <= 0.38) & (spiral > -0.2)] = c_hot
    bp[(r >= 0.34) & (r <= 0.37) & (spiral > 0.55)] = WHITE_IDX
    # Pitch-black circular singularity core
    bp[r < 0.32] = BLACK_IDX
    return bp


def generate_fractal(
    kind: str,
    w: int,
    h: int,
    palette: tuple[int, int, int, int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate crisp mathematical Fractals (Sierpinski Triangle, Mandelbrot/Julia, Carpet, Diamond)."""
    rng = random.Random(seed)
    if palette is None:
        palette = rng.choice(
            [
                (BLACK_IDX, DARK_PURPLE, PURPLE_IDX, LAVENDER),      # Void Purple Fractal
                (BLACK_IDX, DARK_BLUE_IDX, LIGHT_BLUE, WHITE_IDX),    # Electric Mandelbrot
                (BLACK_IDX, DARK_GREEN, LIGHT_GREEN, YELLOW_IDX),     # Neon Matrix Fractal
                (BLACK_IDX, BURGUNDY_IDX, RED_IDX, ORANGE_IDX),       # Abyssal Fire Fractal
                (BLACK_IDX, MAGENTA, PINK_IDX, PALE_YELLOW),          # Synthwave Fractal
            ]
        )
    c0, c1, c2, c3 = palette
    bp = np.full((h, w), -1, dtype=np.int16)
    k = kind.lower().strip()

    if k == "sierpinski":
        # Centered Sierpinski triangle cutout
        size = min(w, h)
        oy = (h - size) // 2
        ox = (w - size) // 2
        for dy in range(size):
            # Centered equilateral row width = dy + 1
            x_start = ox + (size - 1 - dy) // 2
            for dx in range(dy + 1):
                px = x_start + dx
                py = oy + dy
                if 0 <= py < h and 0 <= px < w:
                    if (dx & (dy - dx)) == 0:
                        band = (dy * 3) // max(1, size)
                        bp[py, px] = (c3, c2, c1)[min(2, band)]
                    else:
                        bp[py, px] = c0
        return bp

    if k == "mandelbrot":
        # Pixel Mandelbrot / Julia set cutout
        yy, xx = np.mgrid[0:h, 0:w]
        use_julia = rng.random() < 0.45
        if use_julia:
            zr = (xx / max(1.0, w - 1) - 0.5) * 2.9
            zi = (yy / max(1.0, h - 1) - 0.5) * 2.9
            cr, ci = rng.choice([(-0.7, 0.27015), (-0.4, 0.6), (0.285, 0.01)])
        else:
            cr = (xx / max(1.0, w - 1)) * 2.7 - 2.05
            ci = (yy / max(1.0, h - 1)) * 2.5 - 1.25
            zr = np.zeros_like(cr)
            zi = np.zeros_like(ci)
        iters = np.zeros((h, w), dtype=np.int32)
        max_it = 24
        for i in range(1, max_it + 1):
            alive = (zr * zr + zi * zi) <= 4.0
            zr_n = zr * zr - zi * zi + cr
            zi_n = 2.0 * zr * zi + ci
            zr = np.where(alive, zr_n, zr)
            zi = np.where(alive, zi_n, zi)
            iters = np.where(alive, i, iters)
        # Keep the fractal boundary and immediate halo on a transparent background
        bp[iters >= 3] = c1
        bp[iters >= 6] = c2
        bp[iters >= 10] = c3
        bp[iters >= max_it - 1] = c0
        return bp

    if k == "carpet":
        # Recursive Sierpinski Carpet (square fractal)
        size = min(w, h)
        oy = (h - size) // 2
        ox = (w - size) // 2
        for dy in range(size):
            for dx in range(size):
                x, y = dx, dy
                hole_depth = 0
                for depth in range(1, 5):
                    if (x % 3 == 1) and (y % 3 == 1):
                        hole_depth = depth
                        break
                    x //= 3
                    y //= 3
                if hole_depth == 0:
                    bp[oy + dy, ox + dx] = c0
                elif hole_depth == 1:
                    bp[oy + dy, ox + dx] = c3
                elif hole_depth == 2:
                    bp[oy + dy, ox + dx] = c2
                else:
                    bp[oy + dy, ox + dx] = c1
        return bp

    # Recursive diamond / sacred-geometry fractal cutout
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx = np.abs(xx - cx)
    dy = np.abs(yy - cy)
    man = dx / max(1.0, w * 0.48) + dy / max(1.0, h * 0.48)
    inside = man <= 1.0
    xi = dx.astype(np.int32)
    yi = dy.astype(np.int32)
    pat = (xi ^ yi) % 8
    bp[inside] = c0
    bp[inside & (pat < 4)] = c1
    bp[inside & (pat < 2)] = c2
    bp[inside & (pat == 0)] = c3
    bp[inside & (man > 0.90)] = BLACK_IDX
    return bp


def generate_void_monster(w: int, h: int, seed: int = 42) -> np.ndarray:
    """Abyssal Void Monster silhouette with jagged black tendrils, glowing crimson/white eyes & teeth."""
    rng = random.Random(seed)
    bp = np.full((h, w), -1, dtype=np.int16)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx = (xx - cx) / max(1.0, w * 0.48)
    dy = (yy - cy) / max(1.0, h * 0.48)
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)

    # Jagged irregular monster silhouette
    edge = 0.78 + 0.18 * np.sin(7.0 * ang + seed) + 0.10 * np.cos(13.0 * ang)
    body = r <= edge
    bp[body] = BLACK_IDX
    bp[body & (r > edge - 0.12)] = BURGUNDY_IDX

    # Glowing sinister eyes
    eye_col = rng.choice([RED_IDX, DARK_RED_IDX, PURPLE_IDX])
    for sx in (-0.28, 0.28):
        ex = cx + sx * w * 0.48
        ey = cy - 0.12 * h * 0.48
        ed = np.sqrt(((xx - ex) / max(2.0, w * 0.11)) ** 2 + ((yy - ey) / max(1.5, h * 0.08)) ** 2)
        bp[ed <= 1.0] = eye_col
        bp[ed <= 0.45] = WHITE_IDX

    # Jagged grinning mouth with pixel teeth
    mouth = (np.abs(dy - 0.26) < (0.14 - 0.18 * (dx ** 2))) & (np.abs(dx) < 0.52)
    bp[mouth] = DARK_RED_IDX
    teeth = mouth & (((xx // max(1, w // 22)) % 2) == 0) & (np.abs(dy - 0.26) > 0.04)
    bp[teeth] = WHITE_IDX
    return bp
