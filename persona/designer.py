"""Design a brand-new persona with Claude: name, look, personality, world.

/design [brief] (or `python -m persona.cli design "brief"`) asks Claude Opus 5.5
for a few distinct concepts. Picking one writes her bible.json, with the casting
pools narrowed to the look Claude described, so /cast produces faces that match
the concept instead of random ones.

Needs ANTHROPIC_API_KEY in .env (console.anthropic.com -> API keys).

Server-side refusal fallbacks are on (fallbacks="default"): if the primary model
declines the request, the API re-runs it on a fallback model inside the same call
instead of returning nothing.
"""
import json
import time
from typing import List

from pydantic import BaseModel, Field

from . import character, config

MODEL = "claude-opus-5-5"
MIN_AGE = 21


class Look(BaseModel):
    ethnicity: str = Field(description="heritage, as a photographer would brief it")
    hair: str = Field(description="colour, length, texture, usual style")
    eyes: str
    face: str = Field(description="face shape and distinctive features, e.g. freckles, dimples")
    build: str = Field(description="height and body type")


class Concept(BaseModel):
    name: str = Field(description="first name she goes by")
    age: int = Field(description=f"between {MIN_AGE} and 35")
    tagline: str = Field(description="one line that sells her concept")
    look: Look
    personality: str = Field(description="3-5 traits, how she talks and acts")
    backstory: str = Field(description="2-3 sentences: job or studies, why she posts")
    home_city: str
    recurring_places: List[str] = Field(description="5 specific places she returns to")
    style: List[str] = Field(description="5 outfit styles she wears")
    caption_voice: str = Field(description="how her captions read: tone, emoji habits, language")
    content_pillars: List[str] = Field(description="3-5 recurring themes of her posts")
    fanvue_angle: str = Field(description="what her subscription page offers fans, tasteful wording")


class Concepts(BaseModel):
    concepts: List[Concept]


SYSTEM = (
    "You design original, fictional virtual-influencer personas for an AI image and video "
    "pipeline. Each persona is an adult woman with a consistent face and a believable life "
    "that can be followed post by post on Instagram and TikTok, plus a subscription page. "
    "Rules: every persona is clearly an adult, between 21 and 35; she must be wholly "
    "original and must not resemble or be named after any real person, celebrity or "
    "existing influencer; make each concept distinct from the others in look, city, "
    "personality and niche. Describe the look concretely enough for a casting director."
)


def _client():
    import anthropic
    if not config.env("ANTHROPIC_API_KEY") and not config.env("ANTHROPIC_AUTH_TOKEN"):
        raise RuntimeError("set ANTHROPIC_API_KEY in .env (console.anthropic.com -> API keys)")
    return anthropic.Anthropic()


def design(brief="", n=3):
    """Returns (path, [concept dicts]). Concepts are saved under PERSONA_HOME/designs/
    so a Telegram button can refer to them by index later."""
    existing = ", ".join(character.list_characters()) or "none"
    ask = (f"Design {n} persona concepts.\n"
           f"Brief from the owner: {brief or 'open, surprise me, but commercially strong'}\n"
           f"Existing personas to stay distinct from: {existing}.")
    response = _client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": ask}],
        output_format=Concepts,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        detail = getattr(response.stop_details, "explanation", "") if response.stop_details else ""
        raise RuntimeError(f"Claude declined the brief. {detail}".strip())
    concepts = [c.model_dump() for c in response.parsed_output.concepts][:n]
    for c in concepts:
        c["age"] = max(MIN_AGE, min(35, int(c["age"])))
    out = config.home() / "designs"
    out.mkdir(exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"brief": brief, "model": response.model, "concepts": concepts},
                               indent=2, ensure_ascii=False))
    return path, concepts


def load(design_name):
    path = config.home() / "designs" / f"{design_name}.json"
    return json.loads(path.read_text())["concepts"]


def summary(c):
    look = c["look"]
    return (f"✨ {c['name']}, {c['age']} — {c['tagline']}\n"
            f"👤 {look['ethnicity']}, {look['hair']}, {look['eyes']}, {look['face']}, {look['build']}\n"
            f"🧠 {c['personality']}\n"
            f"📍 {c['home_city']} · {', '.join(c['recurring_places'][:3])}\n"
            f"👗 {', '.join(c['style'][:4])}\n"
            f"📖 {c['backstory']}\n"
            f"🗣 {c['caption_voice']}\n"
            f"💜 {c['fanvue_angle']}")


def create_from(c):
    """Write a bible from a concept and make her the active persona."""
    char = character.create(c["name"], slug=_free_slug(c["name"]))
    bible = char.bible
    look = c["look"]
    bible.update({
        "name": c["name"],
        "age": c["age"],
        "identity": ", ".join([look["ethnicity"], look["hair"], look["eyes"], look["face"], look["build"]]),
        "tagline": c["tagline"],
        "personality": c["personality"] + ". Caption voice: " + c["caption_voice"],
        "backstory": c["backstory"],
        "home_city": c["home_city"],
        "recurring_places": c["recurring_places"],
        "style": c["style"],
        "content_pillars": c["content_pillars"],
        "fanvue_angle": c["fanvue_angle"],
        # One fixed value per trait: /cast then varies only seed and lighting, so
        # every candidate is a take on the same concept rather than a new person.
        "casting": {k: [v] for k, v in look.items()},
    })
    bible["safety"]["adult_anchor"] = f"an adult woman, {c['age']} years old"
    char.save_bible(bible)
    character.set_active(char.slug)
    return char


def _free_slug(name):
    base = character.slugify(name)
    taken = set(character.list_characters())
    slug, i = base, 2
    while slug in taken:
        slug, i = f"{base}-{i}", i + 1
    return slug
