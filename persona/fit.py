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


def read_person(clip, work):
    """{hair, length, skin} of the main face in the clip, or {} when there is none."""
    import cv2
    import numpy as np
    from . import faceid
    for at in (1.0, 0.3, 2.5):
        img_path = _frame(clip, Path(work) / f"fit-{at}.jpg", at)
        face = faceid._main_face(faceid._faces(img_path))
        if face is not None:
            break
    else:
        return {}
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in face.bbox]
    fw, fh = x1 - x0, y1 - y0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def region(a, b, c, d):
        a, b, c, d = max(0, a), max(0, b), min(w, c), min(h, d)
        return hsv[b:d, a:c].reshape(-1, 3) if c > a and d > b else np.zeros((0, 3))

    # Hair: a band over the forehead and strips beside the face.
    hair = np.concatenate([region(x0, y0 - fh // 3, x1, y0 + fh // 10),
                           region(x0 - fw // 3, y0, x0, y1), region(x1, y0, x1 + fw // 3, y1)])
    skin = region(x0 + fw // 4, y0 + fh // 3, x1 - fw // 4, y1 - fh // 4)
    if not len(hair) or not len(skin):
        return {}
    hh, hs, hv = [float(np.median(hair[:, i])) for i in range(3)]
    if hv < 60:
        hair_c = "black"
    elif 140 <= hh <= 175 and hs > 60:
        hair_c = "pink"
    elif hh <= 15 and hs > 110 and hv > 90:
        hair_c = "red"
    elif hv > 150 and hs < 120:
        hair_c = "blonde"
    else:
        hair_c = "brown"
    # Long hair: hair-coloured pixels well below the chin beside the neck.
    below = np.concatenate([region(x0 - fw // 3, y1, x0, y1 + fh), region(x1, y1, x1 + fw // 3, y1 + fh)])
    near = (np.abs(below[:, 2].astype(float) - hv) < 35).mean() if len(below) else 0
    length = "long" if near > 0.45 else "short"
    sv = float(np.median(skin[:, 2]))
    skin_c = "light" if sv > 175 else "medium" if sv > 120 else "dark"
    return {"hair": hair_c, "length": length, "skin": skin_c}


def _bible(slug):
    p = config.home() / slug / "bible.json"
    return json.loads(p.read_text()) if p.exists() else {}


def score(person, identity):
    t = identity.lower()
    s = 0
    if any(word in t for word in HAIR_WORDS.get(person.get("hair"), ())):
        s += 3
    short = any(word in t for word in ("bob", "chin-length", "pixie", "short"))
    # Length is read less surely than colour (a scarf or the neck can pass for hair).
    s += 0.5 if (person.get("length") == "short") == short else 0
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
    return choice
