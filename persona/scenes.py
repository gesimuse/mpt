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


def _ask(prompt, lane):
    """llm.ask goes to HF's router first, then Ollama. In the Fanvue lane only
    Ollama is used: its scenes are still SFW, but which posts are destined for
    Fanvue is nobody else's business."""
    import llm
    if lane == "fanvue" or config.flag("PERSONA_LOCAL_LLM_ONLY"):
        return llm._ask_ollama(prompt, max_tokens=900, temperature=0.9)
    return llm.ask(prompt, max_tokens=900, temperature=0.9)


def _prompt(char, lane, hint, n):
    b = char.bible
    liked, disliked = char.leaning()
    story = "\n".join(f"- {s['beat']}" for s in char.story(8)) or "- (nothing yet, this is her first post)"
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

Write {n} NEW scenes that continue her story (new day, small events, recurring places,
sometimes somewhere new). Lean toward what people liked. Do not describe her face or
hair; that is fixed.

Return ONLY a JSON array of {n} objects with these keys:
"setting" (where, specific), "outfit", "action" (what she is doing, pose),
"shot" (camera framing), "light", "mood", "motion" (one sentence: how she moves in a
5-second video of this photo), "caption" (first-person social caption, max 20 words,
1-2 emojis, no hashtags), "beat" (one past-tense sentence for her storyline),
"tags" (3-5 short lowercase tags like "beach", "golden hour", "summer dress").
"""


def _parse(text, n):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        raise ValueError("no JSON array in the answer")
    scenes = json.loads(m.group(0))
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


def fallback(char, n, hint=""):
    b = char.bible
    out = []
    for _ in range(n):
        place = random.choice(b["recurring_places"])
        style = random.choice(b["style"])
        mood = random.choice(MOODS)
        light = random.choice(TIMES)
        out.append({
            "setting": f"{place} in {b['home_city']}" + (f", {hint}" if hint else ""),
            "outfit": style, "action": "posing naturally, candid moment", "shot": random.choice(SHOTS),
            "light": light, "mood": mood, "motion": "she turns toward the camera and smiles, hair moving slightly",
            "caption": "", "beat": f"Spent some time at {place}.",
            "tags": [place.split()[-1], style.split()[0], light.split(",")[0]],
        })
    return out


def write(char, lane="social", hint="", n=1):
    try:
        scenes = _parse(_ask(_prompt(char, lane, hint, n), lane), n)
        if len(scenes) < n:
            scenes += fallback(char, n - len(scenes), hint)
        return scenes
    except Exception as e:
        log(f"LLM scene writing failed ({type(e).__name__}: {str(e)[:150]}); using bible fallback")
        return fallback(char, n, hint)


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
