"""Turns a still image's own SD generation prompt into a short motion instruction
for image-to-video (Wan 2.2 I2V via motionforge) -- an actual action for the person
to do (a smile, a head tilt, hair moving), not a re-description of the photo's
outfit/setting/lighting that the SD prompt already is. Feeding the SD prompt straight
into I2V as the motion prompt asks the video model to reproduce the STILL, not animate
it.

Runs against the same local Ollama instance as caption_writer.py/supervisor.py --
same OLLAMA_URL, a text model here rather than vision."""
import os, re, time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
TEXT_MODEL = os.environ.get("CAPTION_MODEL", "llama3.2:3b")

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


def _ask(prompt, attempts=3):
    last = None
    for i in range(attempts):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model": TEXT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 80,
                    "temperature": 0.8,
                },
                timeout=30,
            )
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise RuntimeError(f"{r.status_code} {r.text[:150]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.Timeout, requests.ConnectionError, RuntimeError) as e:
            last = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    raise last


def write(image_prompt):
    """A short motion instruction for animating the photo that image_prompt
    generated. Raises on failure -- callers fall back to niches.json's own static
    motionforge_prompt rather than this module retrying forever; a hiccup here
    must never block recording an otherwise-good photo batch."""
    raw = _ask(RUBRIC.format(image_prompt=image_prompt))
    motion = raw.strip().strip('"').strip()
    motion = re.sub(r"^(?:motion instruction:|instruction:)\s*", "", motion, flags=re.I)
    if not motion:
        raise RuntimeError(f"empty motion prompt from: {raw[:150]}")
    log(f"wrote motion prompt for {image_prompt[:60]!r}: {motion!r}")
    return motion
