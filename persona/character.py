"""One persona on disk: her bible, her master refs, everything made of her, her
storyline and what the votes have taught us.

The bible is the only thing written by hand. Edit bible.json freely -- every scene
is built from it on the next run.
"""
import json
import re
import secrets
import time
from pathlib import Path

from . import config

DEFAULT_BIBLE = {
    "name": "",
    "age": 24,
    # Filled in by /cast when you pick a candidate: the look that candidate was
    # generated from. Every prompt carries it next to the reference images, so the
    # text and the pictures agree on who she is.
    "identity": "",
    "personality": "warm, playful, a little sarcastic, outdoorsy, loves coffee and late-night drives",
    "home_city": "Lisbon",
    "recurring_places": ["her small sunlit apartment with plants", "a corner cafe near her flat",
                         "the beach at golden hour", "a rooftop bar with city views", "her gym"],
    "style": ["casual streetwear", "summer dresses", "athleisure", "cozy knitwear", "evening going-out looks"],
    # /cast samples one value from each pool per candidate. Narrow a pool to steer
    # casting; the picked combination becomes `identity`.
    "casting": {
        "ethnicity": ["Southern European", "Eastern European", "Latina", "Scandinavian", "mixed-race",
                      "Middle Eastern", "East Asian", "South Asian"],
        "hair": ["long wavy chestnut hair", "shoulder-length straight black hair", "long honey-blonde hair",
                 "curly dark-brown hair", "copper-red wavy hair", "sleek brunette hair in a high ponytail"],
        "eyes": ["hazel eyes", "green eyes", "dark-brown eyes", "light-blue eyes", "grey eyes"],
        "face": ["soft oval face with light freckles", "high cheekbones and full lips",
                 "heart-shaped face and a small nose", "defined jawline and thick brows",
                 "round face with dimples"],
        "build": ["slim athletic build", "curvy hourglass figure", "petite toned build", "tall slender build"],
    },
    "lanes": {
        # Appended to every scene in that lane. The social lane has to stay safe for
        # TikTok and Instagram. The Fanvue lane's cue is yours to write; it never
        # leaves this machine except to Fanvue itself.
        "social": {"cue": "sexy and confident Instagram model look, glamorous makeup, styled hair, "
                           "figure-flattering revealing outfit, no nudity",
                   "negative": "nudity, nipples, genitals"},
        "fanvue": {"cue": "sensual boudoir photography, intimate and confident mood",
                   "negative": ""},
    },
    # Non-negotiable, prepended to every prompt in both lanes. She is an adult and
    # must always read as one -- persona/faceid.py also rejects any image whose
    # estimated age falls under the floor, whatever the prompt said.
    "safety": {"adult_anchor": "an adult woman in her mid-twenties",
               "negative": "child, childlike, teen, underage, minor, school uniform, young-looking"},
}

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class CharacterError(ValueError):
    pass


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32]
    if not SLUG_RE.match(s or ""):
        raise CharacterError(f"cannot make a slug from {name!r}")
    return s


def list_characters():
    return sorted(p.parent.name for p in config.home().glob("*/bible.json"))


def create(name, slug=None):
    slug = slug or slugify(name)
    path = config.home() / slug
    if (path / "bible.json").exists():
        raise CharacterError(f"{slug} already exists")
    for sub in ("refs", "casting", "items", "outbox/fanvue", "outbox/instagram"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    bible = json.loads(json.dumps(DEFAULT_BIBLE))
    bible["name"] = name
    _write_json(path / "bible.json", bible)
    return Character(slug)


def active():
    settings = config.load_settings()
    slug = config.env("PERSONA_CHARACTER") or settings.get("active")
    if not slug:
        raise CharacterError("no active character -- create one with /new <Name>")
    return Character(slug)


def set_active(slug):
    if slug not in list_characters():
        raise CharacterError(f"no character {slug!r}; have: {', '.join(list_characters()) or 'none'}")
    settings = config.load_settings()
    settings["active"] = slug
    config.save_settings(settings)


def _write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def new_id():
    return secrets.token_hex(4)


class Character:
    def __init__(self, slug):
        self.slug = slug
        self.dir = config.home() / slug
        if not (self.dir / "bible.json").exists():
            raise CharacterError(f"no character {slug!r} under {config.home()}")
        for sub in ("refs", "casting", "items", "outbox/fanvue", "outbox/instagram"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------- bible
    @property
    def bible(self):
        data = json.loads((self.dir / "bible.json").read_text())
        # Missing keys come from the default so an older bible keeps working when a
        # new field is added here.
        merged = json.loads(json.dumps(DEFAULT_BIBLE))
        merged.update(data)
        return merged

    def save_bible(self, bible):
        _write_json(self.dir / "bible.json", bible)

    @property
    def name(self):
        return self.bible.get("name") or self.slug

    # --------------------------------------------------------------------- refs
    @property
    def refs_dir(self):
        return self.dir / "refs"

    def refs(self):
        """Master refs in a stable order: 01-front first, which matters because
        prompts call it <image1>."""
        return sorted(p for p in self.refs_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))

    def has_refs(self):
        return bool(self.refs())

    # -------------------------------------------------------------------- items
    def item_path(self, item_id):
        return self.dir / "items" / f"{item_id}.json"

    def new_item(self, **meta):
        item_id = new_id()
        meta = {"id": item_id, "created": int(time.time()), "status": "new", **meta}
        _write_json(self.item_path(item_id), meta)
        return meta

    def item(self, item_id):
        path = self.item_path(item_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def update_item(self, item_id, **changes):
        meta = self.item(item_id) or {"id": item_id}
        meta.update(changes)
        _write_json(self.item_path(item_id), meta)
        return meta

    def items(self):
        out = []
        for p in (self.dir / "items").glob("*.json"):
            try:
                out.append(json.loads(p.read_text()))
            except ValueError:
                pass
        return sorted(out, key=lambda m: m.get("created", 0))

    # ---------------------------------------------------------------- storyline
    def add_beat(self, text, item_id=None):
        line = {"t": int(time.time()), "beat": text.strip(), "item": item_id}
        with open(self.dir / "storyline.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def story(self, last=12):
        path = self.dir / "storyline.jsonl"
        if not path.exists():
            return []
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return lines[-last:]

    # ------------------------------------------------------------ recent scenes
    def add_recent(self, scenes):
        """Every scene ever planned, one line each, so the writer can be told what not
        to repeat. Synced with the Kaggle dataset like the storyline."""
        with open(self.dir / "recent_scenes.jsonl", "a") as f:
            for sc in scenes:
                f.write(json.dumps({"setting": sc.get("setting", ""), "outfit": sc.get("outfit", ""),
                                    "action": sc.get("action", "")}, ensure_ascii=False) + "\n")

    def recent_scenes(self, last=30):
        path = self.dir / "recent_scenes.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()][-last:]
        return [f"{r['setting']} / {r['outfit']} / {r['action']}" for r in rows]

    # -------------------------------------------------------------------- votes
    def _votes(self):
        path = self.dir / "votes.json"
        return json.loads(path.read_text()) if path.exists() else {"tags": {}, "models": {}}

    def vote(self, item, delta):
        """A 👍/👎 moves every tag on the item and the model that made it. Scenes are
        written with the top tags favoured and the bottom ones avoided, which is all
        "learning from engagement" needs to mean here."""
        votes = self._votes()
        for tag in item.get("tags", []):
            votes["tags"][tag] = votes["tags"].get(tag, 0) + delta
        if item.get("model"):
            votes["models"][item["model"]] = votes["models"].get(item["model"], 0) + delta
        _write_json(self.dir / "votes.json", votes)

    def leaning(self, n=6):
        tags = self._votes()["tags"]
        ranked = sorted(tags.items(), key=lambda kv: kv[1], reverse=True)
        liked = [t for t, s in ranked if s > 0][:n]
        disliked = [t for t, s in reversed(ranked) if s < 0][:n]
        return liked, disliked

    def votes_summary(self):
        return self._votes()


REQUIRED = ("name", "age", "identity", "personality", "home_city", "recurring_places", "style", "casting")


def import_bible(data, slug=None):
    """Create a persona from a complete bible (what the new-persona Claude Code skill
    writes) and make her active. Missing optional keys come from DEFAULT_BIBLE."""
    missing = [k for k in REQUIRED if not data.get(k)]
    if missing:
        raise CharacterError(f"bible is missing: {', '.join(missing)}")
    if int(data["age"]) < 21:
        raise CharacterError("age must be 21 or over")
    taken = set(list_characters())
    base = slug or slugify(data["name"])
    slug, i = base, 2
    while slug in taken:
        slug, i = f"{base}-{i}", i + 1
    char = create(data["name"], slug=slug)
    bible = char.bible
    for k, v in data.items():
        # lanes/safety merge (a bible may set only the social cue); casting replaces,
        # since a leftover default pool would cast a random trait.
        if k in ("lanes", "safety") and isinstance(v, dict):
            bible[k] = {**bible[k], **v}
        else:
            bible[k] = v
    bible["safety"]["adult_anchor"] = f"an adult woman, {int(data['age'])} years old"
    char.save_bible(bible)
    set_active(char.slug)
    return char
