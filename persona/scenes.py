"""What she does next, and the prompts that render it.

An LLM writes the scene (where she is, what she wears, what she's doing, a caption
and a one-line story beat) from her bible, the last storyline beats and the vote
leaning. It is always asked for a SAFE scene, in both lanes: the Fanvue lane's
extra wording comes from bible.json's lanes.fanvue.cue, written by you, never from
a hosted model. That keeps every LLM call refusal-free and keeps explicit wording
off third-party servers.

If no LLM answers, scenes fall back to random combinations from the bible, so a
run never stalls on captions.
"""
import json
import random
import re
import sys

from . import config

if str(config.REPO) not in sys.path:
    sys.path.insert(0, str(config.REPO))

SHOTS = ["candid smartphone photo", "mirror selfie", "close-up portrait", "half-body shot",
         "full-body shot", "over-the-shoulder candid", "wide shot with her small in the frame"]
MOODS = ["laughing", "relaxed", "confident", "dreamy", "playful", "focused", "flirty smile"]
TIMES = ["morning light", "midday sun", "golden hour", "blue hour", "night, warm lamps", "overcast soft light"]


def log(msg):
    print(f"[persona.scenes] {msg}", flush=True)


GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions"


def _ask_github(prompt):
    """GitHub Models: free inside Actions with the workflow's GITHUB_TOKEN
    (permissions: models: read). A far stronger writer than the 8B fallback."""
    import requests
    token = config.env("GITHUB_MODELS_TOKEN")
    if not token:
        raise RuntimeError("no GITHUB_MODELS_TOKEN")
    r = requests.post(GITHUB_MODELS_URL, timeout=120, headers={"Authorization": f"Bearer {token}"},
                      json={"model": config.env("PERSONA_SCENE_MODEL", "openai/gpt-4.1"),
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 2500, "temperature": 1.0})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _ask(prompt, lane):
    """GitHub Models first (Actions), then llm.ask: HF's router, then Ollama. In the
    Fanvue lane only Ollama is used: its scenes are still SFW, but which posts are
    destined for Fanvue is nobody else's business."""
    import llm
    if lane == "fanvue" or config.flag("PERSONA_LOCAL_LLM_ONLY"):
        return llm._ask_ollama(prompt, max_tokens=900, temperature=0.9)
    if config.env("GITHUB_MODELS_TOKEN"):
        try:
            return _ask_github(prompt)
        except Exception as e:
            log(f"GitHub Models failed ({type(e).__name__}: {str(e)[:150]}); falling back")
    return llm.ask(prompt, max_tokens=1600, temperature=1.0)


def _prompt(char, lane, hint, n):
    b = char.bible
    liked, disliked = char.leaning()
    story = "\n".join(f"- {s['beat']}" for s in char.story(8)) or "- (nothing yet, this is her first post)"
    recent = "\n".join(f"- {r}" for r in char.recent_scenes(30)) or "- (none yet)"
    return f"""You plan photos for {b['name']}, a fictional social-media persona.
Who she is: {b['age']}-year-old woman, {b.get('identity') or 'look not decided yet'}.
Personality: {b['personality']}. Lives in {b['home_city']}.
{('Backstory: ' + b['backstory']) if b.get('backstory') else ''}
{('Content pillars: ' + ', '.join(b['content_pillars'])) if b.get('content_pillars') else ''}
Places she goes: {', '.join(b['recurring_places'])}.
Her styles: {', '.join(b['style'])}.
Her story so far (oldest first):
{story}
People liked: {', '.join(liked) or 'no data yet'}.
People did not like: {', '.join(disliked) or 'no data yet'}.
{('Direction for this batch: ' + hint) if hint else ''}

She is a glamorous, sexy Instagram model and every photo is a flattering shoot of HER,
not a travel photo: she is the subject, filling most of the frame. Think of what top
lifestyle/fashion models post: pool and beach in bikinis, hotel rooms, bedroom and
bathroom mirror selfies, gym sets, rooftop golden hour, night out in a bodycon mini
dress, yacht, balcony, car selfies. Outfits are figure-flattering and revealing
(bikinis, crop tops, mini dresses, low-cut tops, lingerie-inspired tops, leggings),
but there is NO nudity. Poses are confident, model-like and varied: over the
shoulder, hip popped, lying on the bed, arched back, hair flip, sitting on the edge
of the pool, walking toward the camera. Moods: seductive, playful, confident, sultry.
Never a stiff frontal tourist pose, never the same smile every time.

Write {n} NEW scenes. Every one must be a genuinely NEW situation: a different
place, outfit, activity, time of day and camera angle from each other AND from the
recent scenes listed below. Invent them yourself -- anywhere a glamorous young woman
might plausibly be, at home or travelling, indoors or out, any season; surprise the
reader. Use her usual places for at most one in four scenes. Keep
it believable for her life and story. Lean toward what people liked. Do not describe
her face or hair; that is fixed.

Recent scenes -- do NOT repeat these settings or outfits:
{recent}

Return ONLY a JSON array of {n} objects with these keys:
"setting" (where, specific), "outfit" (always with colours and materials, e.g. "emerald satin
mini dress", "white string bikini" -- vary the colours), "action" (what she is doing, pose,
with her hands somewhere simple and visible: on her hip, in her hair, holding a drink),
"shot" (camera framing), "light", "mood", "motion" (one sentence: how she moves in a
5-second video of this photo), "caption" (first-person social caption, max 20 words,
1-2 emojis, no hashtags), "beat" (one past-tense sentence for her storyline),
"tags" (3-5 short lowercase tags like "beach", "golden hour", "summer dress").
"""


# Head angle and expression, rotated across a batch so at most one in four looks
# straight into the lens. Without this every image copied the refs: front-facing,
# same closed-mouth smile.
# Her face sheet: one portrait per head angle / expression, rendered once from her
# master refs (studio.make_sheet) into refs/sheet/<stem>.jpg. Each scene is given
# a gaze, and the sheet portrait for it rides along as its second reference, so a
# profile scene is drawn from her real profile instead of a rotated front view.
FACE_SHEET = [
    ("profile-left", "head turned to her left in full side profile, neutral relaxed expression"),
    ("profile-right", "head turned to her right in full side profile, lips slightly parted"),
    ("three-quarter-left", "three-quarter view turned to her left, soft smile"),
    ("three-quarter-right", "three-quarter view turned to her right, dreamy expression, eyes looking past the camera"),
    ("over-shoulder", "body turned away, looking back over her shoulder at the camera, playful half-smile"),
    ("looking-down", "head tilted down, eyes looking down, soft smile"),
    ("looking-up", "chin raised, looking up and to the side, biting her lower lip"),
    ("laughing", "laughing with her eyes closed and head tilted back, mouth open"),
    ("big-smile", "big genuine smile at the camera, teeth showing, eyes crinkled"),
    ("sultry", "looking straight into the camera, sultry and confident, no smile, lips slightly parted"),
    ("kiss", "blowing a kiss toward the camera, lips puckered"),
    ("shy", "eyes down, shy smile, a strand of hair across her face"),
]

# Head angle and expression, rotated across a batch so at most a few look straight
# into the lens. Without this every image copied the refs: front-facing, same
# closed-mouth smile. Each maps to the FACE_SHEET portrait used as its reference.
GAZES = [
    ("head turned in profile, looking off to the side, lips slightly parted", "profile-left"),
    ("looking back over her shoulder at the camera, playful half-smile", "over-shoulder"),
    ("looking down at her phone, absorbed, soft smile", "looking-down"),
    ("laughing with her eyes closed, head tilted back", "laughing"),
    ("three-quarter view, gazing out of frame, dreamy expression", "three-quarter-right"),
    ("looking up and to the side, biting her lower lip", "looking-up"),
    ("eyes down, shy smile, hair falling across her face", "shy"),
    ("looking straight into the camera, sultry and confident, no smile", "sultry"),
    ("big genuine smile at the camera, teeth showing", "big-smile"),
    ("blowing a kiss toward the camera", "kiss"),
    ("head turned in profile to the other side, calm expression", "profile-right"),
    ("three-quarter view toward the camera, soft smile", "three-quarter-left"),
]


def assign_gazes(scenes):
    start = random.randrange(len(GAZES))
    for i, sc in enumerate(scenes):
        sc["gaze"], sc["sheet"] = GAZES[(start + i * 5) % len(GAZES)]
    return scenes


def sheet_prompt(char, text):
    b = char.bible
    return (f"The same woman as in the reference photos, {b['safety']['adult_anchor']}. Keep her exact face, "
            f"facial features, hair and skin tone. {b.get('identity', '')}. Head-and-shoulders portrait, "
            f"{text}. She wears a plain white tank top. Plain light-grey studio background, soft even light. "
            f"{REALISM}.")


def _parse(text, n):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        raise ValueError("no JSON array in the answer")
    raw = re.sub(r",\s*([\]}])", r"\1", m.group(0))
    scenes = json.loads(raw)
    keys = ("setting", "outfit", "action", "shot", "light", "mood", "motion", "caption", "beat", "tags")
    good = []
    for s in scenes:
        if isinstance(s, dict) and all(k in s for k in keys[:3]):
            s.setdefault("tags", [])
            s["tags"] = [str(t).lower().strip() for t in s["tags"] if str(t).strip()][:6]
            for k in keys:
                s.setdefault(k, "")
            good.append(s)
    if not good:
        raise ValueError("no usable scene objects")
    return good[:n]


POSES = ["looking back over her shoulder", "hip popped, one hand on her waist", "lying on her side, propped on an elbow",
         "sitting on the edge, legs crossed", "walking toward the camera", "arching her back, hands in her hair",
         "leaning against the wall, one knee bent", "mirror selfie, phone in hand, hip popped"]
SEXY_SHOTS = ["full-body shot", "three-quarter body shot from a low angle", "mirror selfie", "half-body close shot",
              "over-the-shoulder shot"]
SEXY_MOODS = ["seductive", "playful", "confident", "sultry", "teasing smile"]


def fallback(char, n, hint=""):
    """No LLM: still a model shoot, never a tourist photo."""
    b = char.bible
    out = []
    for _ in range(n):
        place = random.choice(b["recurring_places"])
        style = random.choice(b["style"])
        pose = random.choice(POSES)
        light = random.choice(TIMES)
        out.append({
            "setting": f"{place}" + (f", {hint}" if hint else ""),
            "outfit": style, "action": pose, "shot": random.choice(SEXY_SHOTS),
            "light": light, "mood": random.choice(SEXY_MOODS),
            "motion": "she slowly shifts her pose and looks into the camera, hair moving softly",
            "caption": "", "beat": f"Spent some time at {place}.",
            "tags": [place.split()[-1], style.split()[-1], light.split(",")[0]],
        })
    return out


def write(char, lane="social", hint="", n=1, chunk=4):
    """In batches of `chunk`: an 8B model asked for eight JSON objects at once broke
    the JSON on the first cloud run. Each batch retries once before falling back."""
    scenes = []
    while len(scenes) < n:
        k = min(chunk, n - len(scenes))
        got = None
        for attempt in range(2):
            try:
                got = _parse(_ask(_prompt(char, lane, hint, k), lane), k)
                break
            except Exception as e:
                log(f"LLM scene writing failed ({type(e).__name__}: {str(e)[:120]}), attempt {attempt + 1}")
        scenes += got or fallback(char, k, hint)
    scenes = assign_gazes(scenes[:n])
    char.add_recent(scenes)
    return scenes


# ------------------------------------------------------------------------ prompts
REALISM = ("photorealistic Instagram model photo, professional photoshoot quality, flattering light, "
           "natural skin texture, shallow depth of field, sharp focus, no text, no watermark")


def scene_prompt(char, scene, lane, n_refs):
    """The prompt for one scene, rendered from her refs. Refs are numbered <imageN>
    in the order engine.image() passes them, which is character.refs() order."""
    b = char.bible
    lane_cfg = b["lanes"].get(lane, {})
    refs = ", ".join(f"<image{i + 1}>" for i in range(n_refs))
    if n_refs > 1:
        who = (f"{refs} are reference photos of the same single woman. Show only her, once -- one "
               f"woman in the photo, no duplicates or twins -- with her exact face, facial features, "
               f"hair and skin tone")
    elif n_refs == 1:
        who = f"The woman from {refs}, alone in the photo, with her exact face, facial features, hair and skin tone"
    else:
        who = "A single woman"
    parts = [
        f"{scene['shot']} of {b['safety']['adult_anchor']}. {who}.",
        f"{b.get('identity', '')}." if b.get("identity") else "",
        f"Setting: {scene['setting']}.",
        f"She wears {scene['outfit']}.",
        f"She is {scene['action']}, {scene['mood']}.",
        f"Her head and face: {scene['gaze']}." if scene.get("gaze") else "",
        ("The reference photos only show who she is; her head angle, expression and pose follow this scene, "
         "not the references." if n_refs else ""),
        "She is the only person in the photo, nobody else in the frame or background.",
        f"Lighting: {scene['light']}.",
        lane_cfg.get("cue", "") + ".",
        REALISM + ".",
    ]
    return " ".join(p for p in parts if p and p != ".")


def negative(char, lane):
    b = char.bible
    neg = [b["safety"]["negative"], b["lanes"].get(lane, {}).get("negative", ""),
           "deformed hands, extra fingers, blurry, cartoon, 3d render"]
    return ", ".join(x for x in neg if x)


def casting_look(char):
    pools = char.bible["casting"]
    return {k: random.choice(v) for k, v in pools.items() if v}


def casting_prompt(char, look, hint=""):
    b = char.bible
    desc = ", ".join(look.values())
    anchor = b["safety"]["adult_anchor"]
    if "years old" not in anchor:
        anchor += f", {b['age']} years old"
    return (f"Candid smartphone portrait photo of {anchor}, "
            f"{desc}. Head and shoulders, looking at the camera with a natural relaxed smile, "
            f"plain light background, soft daylight. {hint + '. ' if hint else ''}{REALISM}.")


# Master refs, generated from the picked candidate. 01 is <image1> forever after, so
# the front close-up goes first.
REF_SHOTS = [
    ("01-front", "Front-facing close-up portrait, neutral relaxed expression, looking straight at the camera, "
                 "plain light-grey studio background, soft even light"),
    ("02-three-quarter", "Three-quarter view head-and-shoulders portrait, slight smile, plain light-grey "
                         "studio background, soft even light"),
    ("03-profile", "Side profile portrait facing left, plain light-grey studio background, soft even light"),
    ("04-full-body", "Full-body standing photo, fitted plain t-shirt and jeans, white sneakers, relaxed pose, "
                     "plain light-grey studio background, soft even light"),
]


def ref_prompt(char, shot_text):
    b = char.bible
    return (f"The same woman as in <image1>, {b['safety']['adult_anchor']}. Keep her exact face, facial "
            f"features, hair, skin tone and body proportions. {b.get('identity', '')}. {shot_text}. {REALISM}.")
