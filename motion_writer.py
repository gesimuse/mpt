"""Turns a still image's own SD generation prompt into a short motion instruction
for image-to-video -- an actual action for the person to do, not a re-description of
the photo's outfit/setting/lighting that the SD prompt already is. Feeding the SD
prompt straight into I2V as the motion prompt asks the video model to reproduce the
STILL, not animate it.

The instruction has to be SEDUCTIVE, matching imageslides.SEXY_CUE's own "every image
must be sexy" baseline on the photo side. The first version of the rubric here asked
only for "a smile, a glance, a head tilt, a hand movement", and got exactly that back:
a live run produced "She raises an eyebrow and gently touches the glass with her index
finger" for a rooftop-bar photo -- a perfectly good instruction for a different
account. The still was doing all the work and the motion was undoing it.

Same policy line as the image side, restated in the rubric because the video model
never sees NEGATIVE_HARD: fully clothed, no undressing, nothing explicit. TikTok
removes the post otherwise, and this app is unaudited. videogen.VIDEO_NEGATIVE is the
second layer -- a negative prompt on every generation, wherever it runs.

Goes through llm.ask() -- HF's router first, a local Ollama instance second. Same
backend ladder caption_writer.py uses, and for the same reason: the Ollama-only path
this used to have timed out on essentially every CI run, so motion prompts were almost
always None and the picker fell back to showing each photo's raw SD prompt."""
import random
import re

import llm

# Shown to the model a few at a time, not all at once -- see _movement_menu().
MOVEMENTS = [
    "a slow arch of the back, hips shifting, weight moving onto one leg",
    "a hand tracing her collarbone, sliding down her waist, resting on her hip",
    "fingers running through her hair, pushing it back off her shoulder",
    "a slow turn toward camera that brings her curves into the frame",
    "a look over the shoulder, chin dropping, eyes coming up to the lens",
    "lips parting, a slow breath, a knowing half-smile",
    "leaning in toward the camera, or slowly straightening up out of a lean",
    "fabric or wet skin catching the light as she moves",
    "a slow stretch, arms lifting, spine lengthening",
    "crossing one leg over the other, settling back into the pose",
    "biting her lip, then letting a smile break through",
    "head tipping back, throat exposed, eyes closing for a beat",
]
# How many of MOVEMENTS the model is shown per call.
MENU_SIZE = 4

RUBRIC = """A still photo was generated from this prompt:
{image_prompt}

It's about to be animated into a short clip with an image-to-video model, starting
from that exact photo. Write ONE motion instruction for the video.

The clip has to read as seductive -- that is the whole point of the account, not a
flourish. The still is already sexy; the motion must carry that, not neutralise it.
Movement with weight and intention behind it, slow rather than busy. For instance:

{menu}

Choose whichever of those actually suits THIS photo's pose and setting, or write
something in the same spirit that fits it better. Plus at most ONE ambient motion
(hair, fabric, water, light) on top of it.

Hard limits -- the clip is posted to TikTok, so breaking these gets it removed:
she stays fully clothed in whatever she is already wearing, no undressing, no
removing or pulling aside clothing, no touching between the legs or on the chest,
nothing explicit. Suggestive, not pornographic.

Do NOT restate her appearance, outfit, or the setting -- the video model already has
the photo, it only needs to know what moves. Under 25 words. No camera directions
unless the prompt above already implies one. No quotes, no preamble, just the
instruction itself."""


def _movement_menu(k=None):
    """A few of MOVEMENTS, freshly sampled per call.

    Showing the model the whole list makes it pick the same entry every time: a live
    run with all twelve inline returned "her hand slides from her collarbone down her
    waist and rests on her hip" for FOUR different photos in a row -- bullet two,
    near-verbatim, every time. A carousel whose images each pre-fill the picker with
    the identical motion prompt is the exact failure this module was written to avoid
    (see autopilot._motion_prompts_for on why a fixed fallback string is worse than
    none). Sampling keeps the menu short enough to read as examples rather than a
    menu to pick from, and different enough per image to actually diverge."""
    return "\n".join(f"  - {m}" for m in random.sample(MOVEMENTS, k or MENU_SIZE))


def log(msg): print(f"[motion_writer] {msg}", flush=True)


def _is_menu_echo(motion):
    """True when the model handed back one of the example movements near-verbatim
    instead of adapting it to this photo. Seen live: for a backstage/mirror still it
    returned "head tipping back, throat exposed, eyes closing for a beat", which is
    MOVEMENTS[11] word for word. Not harmful on its own -- the menu is resampled per
    image, so echoes still differ between images -- but it means the photo's own pose
    and setting were ignored, which is the entire reason this runs per image rather
    than once per batch."""
    norm = re.sub(r"[^a-z ]", "", motion.lower()).strip()
    return any(re.sub(r"[^a-z ]", "", m.lower()).strip() == norm for m in MOVEMENTS)


def write(image_prompt, attempts=2):
    """A short, seductive motion instruction for animating the photo that
    image_prompt generated. Raises on failure -- callers fall back to niches.json's
    own static motionforge_prompt rather than this module retrying forever; a hiccup
    here must never block recording an otherwise-good photo batch."""
    motion = None
    for i in range(attempts):
        raw = llm.ask(RUBRIC.format(image_prompt=image_prompt, menu=_movement_menu()),
                      max_tokens=80, temperature=0.95)
        motion = raw.strip().strip('"').strip()
        motion = re.sub(r"^(?:motion instruction:|instruction:)\s*", "", motion,
                        flags=re.I)
        if not motion:
            raise RuntimeError(f"empty motion prompt from: {raw[:150]}")
        if not _is_menu_echo(motion):
            break
        if i < attempts - 1:
            # A fresh menu, not the same one again -- retrying with identical
            # examples would most likely echo the same line back.
            log(f"model echoed an example verbatim ({motion!r}); retrying")
    log(f"wrote motion prompt for {image_prompt[:60]!r}: {motion!r}")
    return motion
