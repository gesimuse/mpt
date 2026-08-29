"""Turns a still image's own SD generation prompt into a short motion instruction
for image-to-video (Wan 2.2 I2V via motionforge) -- an actual action for the person
to do (a smile, a head tilt, hair moving), not a re-description of the photo's
outfit/setting/lighting that the SD prompt already is. Feeding the SD prompt straight
into I2V as the motion prompt asks the video model to reproduce the STILL, not animate
it.

Goes through llm.ask() -- HF's router first, a local Ollama instance second. Same
backend ladder caption_writer.py uses, and for the same reason: the Ollama-only path
this used to have timed out on essentially every CI run, so motion prompts were almost
always None and the picker fell back to showing each photo's raw SD prompt."""
import re

import llm

RUBRIC = """A still photo was generated from this prompt:
{image_prompt}

It's about to be animated into a short clip with an image-to-video model, starting
from that exact photo. Write ONE short motion instruction for the video -- describe
an action the person in the photo does (a smile, a glance, a head tilt, a hand
movement) plus at most one ambient motion (hair, fabric, light). Do NOT restate her
appearance, outfit, or the setting -- the video model already has the photo, it only
needs to know what moves. Under 20 words. No camera directions unless the prompt above
already implies one. No quotes, no preamble, just the instruction itself."""


def log(msg): print(f"[motion_writer] {msg}", flush=True)


def write(image_prompt):
    """A short motion instruction for animating the photo that image_prompt
    generated. Raises on failure -- callers fall back to niches.json's own static
    motionforge_prompt rather than this module retrying forever; a hiccup here
    must never block recording an otherwise-good photo batch."""
    raw = llm.ask(RUBRIC.format(image_prompt=image_prompt),
                  max_tokens=80, temperature=0.8)
    motion = raw.strip().strip('"').strip()
    motion = re.sub(r"^(?:motion instruction:|instruction:)\s*", "", motion, flags=re.I)
    if not motion:
        raise RuntimeError(f"empty motion prompt from: {raw[:150]}")
    log(f"wrote motion prompt for {image_prompt[:60]!r}: {motion!r}")
    return motion
