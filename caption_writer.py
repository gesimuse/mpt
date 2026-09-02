"""A fresh caption + hashtag set per post, written by an LLM from the actual theme
that batch used -- not picked from a fixed rotating pool.

Root problem this replaces: niches.json's old "hashtags" was a single fixed string
used on literally every post, forever, and "captions" was a small static pool (14
phrases) that cycles back to the same lines regardless of how different the images
actually are. The images vary (checkpoint, outfit, pose, camera angle); the post TEXT
never did.

The LLM call itself goes through llm.ask() -- HF's router first, a local Ollama
instance second (see llm.py for why: the old Ollama-only path with a 30s timeout was
what made this module fail on almost every real run, so the "fresh hashtags per post"
promise above was true in the code and false in posted.json)."""
import os
import re

import llm
import trends

RUBRIC = """Write a short TikTok caption for an AI-generated photo carousel. The
photos show: {vibe}

Style: short, punchy, confident, a little flirty -- one line, under 100 characters.
No hashtags in this line. No emoji unless it genuinely fits. Sound like a real person
captioning their own photos, not an ad or an AI describing an image. Do not describe
what is literally in the photo (no "wearing a...", no "posing in..." ) -- write the
kind of line a person adds ON TOP of a photo, a mood or a one-liner, not a caption of
the caption.

Then on a new line, 4-5 relevant lowercase hashtags, space-separated, each starting
with #: a couple of broad-reach ones (aiart-adjacent) and a couple specific to the
actual vibe above, not the same generic set every time.
{trending}

Respond in exactly this format, nothing else, no markdown:
CAPTION: <the caption line>
HASHTAGS: <hashtag1 hashtag2 hashtag3 hashtag4>"""


def log(msg): print(f"[caption_writer] {msg}", flush=True)


_CAPTION_RE = re.compile(r"CAPTION:\s*(.+)", re.I)
_HASHTAGS_RE = re.compile(r"HASHTAGS:\s*(.*)", re.I | re.S)
_TAG_RE = re.compile(r"#\w+")
# The rubric asks for a handful; a model that ignores that and emits fifteen would
# make every post read as tag spam, so cap it here rather than trusting the
# instruction. Five, by operator preference, and applied to photo and video posts
# alike -- there is no reason for the two to differ.
MAX_TAGS = int(os.environ.get("MAX_HASHTAGS", "5"))


def _parse(text):
    """(caption, hashtags) out of the model's reply.

    Tolerant on purpose. The labelled format the rubric asks for is the happy path,
    but a live run against a real 8B model produced a bare caption line followed by a
    row of hashtags with no labels at all -- which the strict version of this raised
    on, sending the post straight back to the fixed hashtag string this module exists
    to replace. Hashtags are also collected from EVERYTHING after the label, not just
    its first line: a model that wraps them onto a second line used to lose all but
    the first (seen live -- one post came out with a single hashtag)."""
    tag_match = _HASHTAGS_RE.search(text)
    cap_match = _CAPTION_RE.search(text)
    if cap_match and tag_match:
        caption = cap_match.group(1).strip().strip('"')
        tags = _TAG_RE.findall(tag_match.group(1))
    else:
        # Unlabelled: the caption is the first line that isn't just hashtags, and the
        # hashtags are every #tag anywhere in the reply.
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        caption = next((ln.strip('"') for ln in lines
                        if not ln.lstrip().startswith("#")), "")
        # Don't let a hashtag that sits inside the caption line count twice.
        tags = _TAG_RE.findall(text.replace(caption, "", 1))
    if not caption or not tags:
        raise RuntimeError(f"unparseable response: {text[:150]}")
    # dict.fromkeys: de-duplicate while keeping the model's own ordering, since the
    # rubric asks for broad-reach tags first and specific ones after.
    tags = list(dict.fromkeys(tags))[:MAX_TAGS]
    return caption, " ".join(tags)


def _trending_clause(niche=None, state=None):
    """A line naming what is actually trending on TikTok right now, or "" when we
    can't find out. Deliberately advisory, not prescriptive: the ask was for hashtags
    that stop being identical every post, and pasting today's top-10 verbatim onto
    every post would just swap one fixed set for another."""
    tags = trends.tiktok_hashtags(state=state, niche=niche)
    if not tags:
        return ""
    return ("Currently trending on TikTok: " + " ".join(tags) + ". Use at most 2 of "
            "these, and only if they genuinely fit the vibe above -- write the rest "
            "yourself.")


def write(vibe, niche=None, state=None):
    """One (caption_line, hashtags) pair for this vibe. Raises on failure --
    callers fall back to the static niches.json pool rather than this module
    retrying forever; a caption-writing hiccup must never block an otherwise-good
    batch of images from getting posted."""
    raw = llm.ask(RUBRIC.format(vibe=vibe,
                                trending=_trending_clause(niche=niche, state=state)),
                  max_tokens=200, temperature=0.9)
    caption, tags = _parse(raw)
    log(f"wrote caption for vibe={vibe!r}: {caption!r} {tags!r}")
    return caption, tags
