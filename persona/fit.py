"""Which roster girl fits a clip best, for an MPT 🎭 face swap (2026-10-30: "selecting
any of the personas that fit").

The person in the clip is read from one frame: hair colour and length from the
pixels around and below her face, skin tone from the face itself. Each girl's bible
`identity` says the same in words ("platinum blonde long straight hair", "olive
skin"). The best scores win; a tie is broken at random, so the same kind of clip
doesn't always go to the same girl. The swap keeps the clip's body, outfit and
moves, so a girl whose hair and skin are close to the original looks the most real.
"""
import json
import random
import subprocess
from pathlib import Path

from . import config


def log(msg):
    print(f"[persona.fit] {msg}", flush=True)


HAIR_WORDS = {
    "blonde": ("blonde", "platinum", "strawberry"),
    "red": ("red hair", "ginger", "auburn", "copper", "strawberry"),
    "pink": ("pink", "pastel"),
    "black": ("black hair", "jet black", "black"),
    "brown": ("brown", "brunette", "chestnut", "caramel", "honey-brown"),
}
SKIN_WORDS = {
    "light": ("porcelain", "pale", "fair", "freckles", "nordic", "russian", "french", "slavic"),
    "medium": ("olive", "sun-kissed", "caramel", "latina", "italian", "spanish", "persian", "golden"),
    "dark": ("deep brown", "black,", "dark skin", "brown skin", "indian"),
}


def _frame(clip, dest, at=1.0):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(at), "-i", str(clip), "-frames:v", "1", str(dest)], check=True)
    return dest


def _hair_colour(px):
    """The colour most of these HSV pixels vote for."""
    import numpy as np
    h, s_, v = px[:, 0].astype(float), px[:, 1].astype(float), px[:, 2].astype(float)
    votes = {"black": (v < 60), "pink": (v >= 60) & (h >= 140) & (h <= 175) & (s_ > 60),
             "red": (v >= 90) & (h <= 15) & (s_ > 110), "blonde": (v > 150) & (s_ < 120)}
    votes["brown"] = ~(votes["black"] | votes["pink"] | votes["red"] | votes["blonde"])
    return max(votes, key=lambda k: int(np.sum(votes[k]))), votes


def read_person(clip, work):
    """{hair, length, skin} of the main face in the clip, or {} when there is none. Several frames,
    background-coloured pixels left out (2026-11-28: a long platinum-blonde clip shot in profile read
    as "short brown" -- the patch over her forehead was the grey-brown wall -- and Gia's short dark bob
    went on with the original's blonde hair still down her back)."""
    import cv2
    import numpy as np
    from . import faceid
    reads = []
    for at in (0.5, 1.5, 3.0, 4.5):
        img_path = _frame(clip, Path(work) / f"fit-{at}.jpg", at)
        face = faceid._main_face(faceid._faces(img_path))
        if face is None:
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in face.bbox]
        fw, fh = x1 - x0, y1 - y0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        def region(a, b, c, d):
            a, b, c, d = max(0, a), max(0, b), min(w, c), min(h, d)
            return hsv[b:d, a:c].reshape(-1, 3) if c > a and d > b else np.zeros((0, 3), np.uint8)

        # The background: the frame's top corners. Pixels close to it are not hair.
        bg = np.concatenate([region(0, 0, w // 8, h // 10), region(w - w // 8, 0, w, h // 10)]).astype(float)
        bgm = np.median(bg, axis=0) if len(bg) else None

        def not_bg(px):
            if bgm is None or not len(px):
                return px
            d = np.abs(px.astype(float) - bgm)
            d[:, 0] = np.minimum(d[:, 0], 180 - d[:, 0])
            return px[(d[:, 0] > 12) | (d[:, 1] > 40) | (d[:, 2] > 40)]

        # Hair: above the forehead and both sides of the face, wide enough for a head in profile.
        hair = not_bg(np.concatenate([region(x0 - fw // 4, y0 - fh // 2, x1 + fw // 4, y0 + fh // 8),
                                      region(x0 - fw // 2, y0, x0, y1), region(x1, y0, x1 + fw // 2, y1)]))
        skin = region(x0 + fw // 4, y0 + fh // 3, x1 - fw // 4, y1 - fh // 4)
        if len(hair) < 200 or not len(skin):
            continue
        colour, _ = _hair_colour(hair)
        # Long hair: pixels of that same colour beside the neck and shoulders, below the chin.
        below = not_bg(np.concatenate([region(x0 - fw // 2, y1, x0 + fw // 6, y1 + fh),
                                       region(x1 - fw // 6, y1, x1 + fw // 2, y1 + fh)]))
        frac = float(np.mean(_hair_colour(below)[1][colour])) if len(below) else 0.0
        sv = float(np.median(skin[:, 2]))
        reads.append({"hair": colour, "long": frac > 0.3, "skin": "light" if sv > 175 else "medium" if sv > 120 else "dark"})
    if not reads:
        return {}
    pick = lambda k: max({r[k] for r in reads}, key=lambda v: sum(r[k] == v for r in reads))
    return {"hair": pick("hair"), "length": "long" if pick("long") else "short", "skin": pick("skin")}


def _bible(slug):
    p = config.home() / slug / "bible.json"
    return json.loads(p.read_text()) if p.exists() else {}


def hair_matches(person, identity):
    return any(word in identity.lower() for word in HAIR_WORDS.get(person.get("hair"), ()))


def length_matches(person, identity):
    short = any(word in identity.lower() for word in ("bob", "chin-length", "pixie", "short"))
    return (person.get("length") == "short") == short


def score(person, identity):
    t = identity.lower()
    s = 0
    if hair_matches(person, identity):
        s += 4
    # 2026-11-28: Gia's short dark bob on a long platinum-blonde clip -- the original's long hair stayed
    # down her back. Length counts more now, and a girl matching both is what the edit needs.
    s += 1.5 if length_matches(person, identity) else 0
    if any(word in t for word in SKIN_WORDS.get(person.get("skin"), ())):
        s += 2
    return s


def pick(clip, candidates, work):
    """The best-fitting slug of `candidates`, or None when no face was found."""
    try:
        person = read_person(clip, work)
    except Exception as e:
        log(f"could not read the clip's person ({e})")
        return None
    if not person:
        log("no face in the clip: picked at random")
        return None
    scored = [(score(person, _bible(s).get("identity", "")), s) for s in candidates]
    best = max(x for x, _ in scored)
    choice = random.choice([s for x, s in scored if x == best])
    log(f"clip person {person}: {choice} (score {best})")
    ident = _bible(choice).get("identity", "")
    global LAST_HAIR_OK
    LAST_HAIR_OK = hair_matches(person, ident) and length_matches(person, ident)
    return choice


# Whether the last pick's hair matched the clip's in colour and length: when not, the edit keeps the
# clip's own hair (persona/recreate.py), so there are never two heads of hair.
LAST_HAIR_OK = True
