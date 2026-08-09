"""A fresh caption + hashtag set per post, written by an LLM from the actual theme
that batch used -- not picked from a fixed rotating pool.

Root problem this replaces: niches.json's old "hashtags" was a single fixed string
used on literally every post, forever, and "captions" was a small static pool (14
phrases) that cycles back to the same lines regardless of how different the images
actually are. The images vary (checkpoint, outfit, pose, camera angle); the post TEXT
never did.

Same NIM API supervisor.py already uses for vision QA -- a plain text chat model
here, not a vision one, reusing the account's existing free NIM credits."""
import os, re, time

import requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TEXT_MODEL = os.environ.get("CAPTION_MODEL", "meta/llama-3.1-8b-instruct")

RUBRIC = """Write a short TikTok caption for an AI-generated photo carousel. The
photos show: {vibe}

Style: short, punchy, confident, a little flirty -- one line, under 100 characters.
No hashtags in this line. No emoji unless it genuinely fits. Sound like a real person
captioning their own photos, not an ad or an AI describing an image. Do not describe
what is literally in the photo (no "wearing a...", no "posing in..." ) -- write the
kind of line a person adds ON TOP of a photo, a mood or a one-liner, not a caption of
the caption.

Then on a new line, 4-6 relevant lowercase hashtags, space-separated, each starting
with #: a couple of broad-reach ones (aiart-adjacent) and a couple specific to the
actual vibe above, not the same generic set every time.

Respond in exactly this format, nothing else, no markdown:
CAPTION: <the caption line>
HASHTAGS: <hashtag1 hashtag2 hashtag3 hashtag4>"""


def log(msg): print(f"[caption_writer] {msg}", flush=True)


def _ask(prompt, attempts=3):
    key = os.environ.get("NIM_API_KEY")
    if not key:
        raise RuntimeError("NIM_API_KEY not set")
    last = None
    for i in range(attempts):
        try:
            r = requests.post(
                NIM_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": TEXT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.9,
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


_CAPTION_RE = re.compile(r"CAPTION:\s*(.+)", re.I)
_HASHTAGS_RE = re.compile(r"HASHTAGS:\s*(.+)", re.I)


def _parse(text):
    cap_match = _CAPTION_RE.search(text)
    tag_match = _HASHTAGS_RE.search(text)
    if not cap_match or not tag_match:
        raise RuntimeError(f"unparseable response: {text[:150]}")
    caption = cap_match.group(1).strip().strip('"')
    tags = " ".join(t for t in tag_match.group(1).split() if t.startswith("#"))
    if not caption or not tags:
        raise RuntimeError(f"empty caption or hashtags in: {text[:150]}")
    return caption, tags


def write(vibe):
    """One (caption_line, hashtags) pair for this vibe. Raises on failure --
    callers fall back to the static niches.json pool rather than this module
    retrying forever; a caption-writing hiccup must never block an otherwise-good
    batch of images from getting posted."""
    raw = _ask(RUBRIC.format(vibe=vibe))
    caption, tags = _parse(raw)
    log(f"wrote caption for vibe={vibe!r}: {caption!r} {tags!r}")
    return caption, tags
