"""Real Reddit active subreddit directory sampler (from data/subreddits/retained_active_subreddits.csv)."""

from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

_SKIP_PATTERNS = re.compile(
    r"^(ask|explainlike|amithe|tifu|legaladvice|personalfinance|nostupid|relationship|"
    r"techsupport|unpopularopinion|changemyview|confession|offmychest|trueoffmychest|"
    r"advice|jobs|career|resumes|tax|insurance|creditcards|mortgage|studentloans|"
    r"news|worldnews|politics|conservative|liberal|economics|wallstreet|stocks|"
    r"help|support|questions|answers|mod|meta|circlejerk|copypasta|outoftheloop|"
    r"whatisthis|tipofmytongue|findareddit|rant|vent|depression|anxiety|adhd|"
    r"medical|diagnose|skincare|haircare|dating|tinder|hinge|bumble|wedding|"
    r"parenting|pregnancy|babybumps|beyondthebump|twox|mensrights|aita|"
    r"antiwork|workreform|recruiting|cscareer|learnprogramming|homework|"
    r"todayilearned|showerthoughts|mildlyinfuriating|mildlyinteresting|lifeprotips|"
    r"getmotivated|upliftingnews|nottheonion|bestof|subredditdrama|iamverysmart|"
    r"trashy|publicfreakout|facepalm|cringe|insanepeople|choosingbeggars)",
    re.IGNORECASE,
)

# Built-in fallback if data/subreddits CSV hasn't been downloaded yet
_FALLBACK_SUBS: list[tuple[str, int]] = [
    ("gaming", 45_000_000), ("Art", 22_000_000), ("cats", 8_000_000), ("anime", 12_000_000),
    ("Minecraft", 8_200_000), ("pokemon", 4_800_000), ("space", 27_000_000), ("memes", 35_000_000),
    ("PixelArt", 2_100_000), ("StardewValley", 2_300_000), ("HollowKnight", 800_000),
    ("OnePiece", 4_100_000), ("zelda", 2_500_000), ("NintendoSwitch", 6_500_000),
    ("formula1", 4_200_000), ("MechanicalKeyboards", 1_300_000), ("osugame", 360_000),
    ("earthbound", 120_000), ("celestegame", 160_000), ("OMORI", 210_000), ("Pikmin", 150_000),
    ("Kirby", 260_000), ("Terraria", 1_500_000), ("factorio", 420_000), ("bicycling", 460_000),
    ("linux", 1_600_000), ("rust", 320_000), ("python", 1_300_000), ("ghibli", 450_000),
    ("RetroGaming", 520_000), ("capybara", 140_000), ("frogs", 240_000), ("duck", 95_000),
    ("penguin", 65_000), ("Boba", 45_000), ("ramen", 310_000), ("bonsai", 320_000),
    ("mycology", 720_000), ("astronomy", 2_900_000), ("cyberpunkgame", 1_800_000),
    ("EldenRing", 3_400_000), ("Undertale", 640_000), ("Deltarune", 420_000),
    ("Vocaloid", 180_000), ("Hololive", 1_100_000), ("Bluey", 220_000), ("gravityfalls", 540_000),
    ("adventuretime", 610_000), ("Parakeets", 19_600), ("retrocomputing", 19_600),
    ("Synthwave", 130_000), ("Vaporwave", 280_000), ("origami", 160_000), ("lego", 1_400_000),
]


@dataclass
class SubredditEntry:
    name: str
    subscribers: int
    rank: int


class SubredditSampler:
    """Power-law sampler over real active non-NSFW subreddits for diverse r/place community themes."""

    def __init__(self, data_dir: Path | None = None, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.entries: list[SubredditEntry] = []
        self._used: set[str] = set()

        base = Path(data_dir) if data_dir is not None else Path("data/subreddits")
        candidates = [
            base / "retained_active_subreddits.csv",
            base / "top_25000_subreddits.csv",
            Path(__file__).resolve().parents[2] / "data" / "subreddits" / "retained_active_subreddits.csv",
            Path(__file__).resolve().parents[2] / "data" / "subreddits" / "top_25000_subreddits.csv",
        ]
        loaded_path: Path | None = None
        for p in candidates:
            if p.exists():
                loaded_path = p
                break

        if loaded_path is not None:
            try:
                with loaded_path.open("r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    rank = 0
                    for row in reader:
                        sub = (row.get("subreddit") or "").strip()
                        if not sub or len(sub) < 2 or len(sub) > 22:
                            continue
                        if not re.match(r"^[A-Za-z0-9_]+$", sub):
                            continue
                        if _SKIP_PATTERNS.search(sub):
                            continue
                        # Avoid crude words in automatic community names
                        low = sub.lower()
                        if any(
                            bad in low
                            for bad in (
                                "porn", "nsfw", "gore", "drug", "weed", "420",
                                "sex", "nude", "hentai", "fuck", "shit", "bitch",
                                "dick", "cock", "pussy",
                            )
                        ):
                            continue
                        try:
                            subs = int(float(row.get("subscribers") or 0))
                        except ValueError:
                            subs = 0
                        if subs < 3_000:
                            continue
                        rank += 1
                        self.entries.append(SubredditEntry(name=sub, subscribers=subs, rank=rank))
            except Exception:
                self.entries = []

        if not self.entries:
            for i, (name, subs) in enumerate(_FALLBACK_SUBS, 1):
                self.entries.append(SubredditEntry(name=name, subscribers=subs, rank=i))

    def sample(self, size_tier: str = "micro", avoid_names: set[str] | None = None) -> SubredditEntry:
        """Sample an unused real subreddit suited to macro/medium/micro tier."""
        n = len(self.entries)
        if size_tier == "macro":
            lo, hi = 0, min(n, max(30, int(n * 0.04)))
        elif size_tier == "medium":
            lo = min(n - 1, max(0, int(n * 0.02)))
            hi = min(n, max(lo + 40, int(n * 0.25)))
        else:
            lo = min(n - 1, max(0, int(n * 0.12)))
            hi = n

        pool = self.entries[lo:hi] or self.entries
        avoid = self._used | (avoid_names or set())

        # Weighted random pick favoring active mid/upper subs within the tier without deterministic repeats
        for _ in range(25):
            # Power-law index bias inside pool
            u = self.rng.random() ** 1.35
            idx = min(len(pool) - 1, int(u * len(pool)))
            cand = pool[idx]
            if cand.name.lower() not in avoid:
                self._used.add(cand.name.lower())
                return cand

        cand = self.rng.choice(pool)
        self._used.add(cand.name.lower())
        return cand

    @staticmethod
    def pretty_label(sub: str) -> str:
        """Convert CamelCase or lowercase subreddit into a clean readable community name."""
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", sub.replace("_", " "))
        return f"r/{sub}" if len(sub) <= 12 else spaced[:18].strip()
