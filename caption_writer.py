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
Suggestive is fine; explicit is not. Never use sexually explicit words, never
reference nudity or sex acts, never use adult-content or OnlyFans-style language --
that is what gets a post removed rather than just unflattering. No hashtags in this
line. No emoji unless it genuinely fits. Sound like a real person captioning their own
photos, not an ad or an AI describing an image. Do not describe what is literally in
the photo (no "wearing a...", no "posing in..." ) -- write the kind of line a person
adds ON TOP of a photo, a mood or a one-liner, not a caption of the caption.

Then on a new line, 4-5 relevant lowercase hashtags, space-separated, each starting
with #: a couple of broad-reach ones (aiart-adjacent) and a couple specific to the
actual vibe above, not the same generic set every time. Hashtags only, no
sexually-explicit or adult-content tags -- those get a post removed rather than
just under-performing.
{trending}

Respond in exactly this format, nothing else, no markdown:
CAPTION: <the caption line>
HASHTAGS: <hashtag1 hashtag2 hashtag3 hashtag4>"""


def log(msg): print(f"[caption_writer] {msg}", flush=True)


_CAPTION_RE = re.compile(r"CAPTION:\s*(.+)", re.I)
_HASHTAGS_RE = re.compile(r"HASHTAGS:\s*(.*)", re.I | re.S)
_TAG_RE = re.compile(r"#\w+")

# A TikTok-valid hashtag: letters/digits/underscore only. TikTok silently fails to
# register anything else as a clickable tag (spaces split it into plain text,
# punctuation breaks it), so an invalid one is worse than useless -- it eats one of
# the five slots for nothing. Length capped at a generous 30; nothing legitimate this
# rubric produces is longer, and if the model runs on, a wall of text is not a tag.
_VALID_TAG_RE = re.compile(r"^#[A-Za-z0-9_]{2,30}$")

# The rubric says "suggestive is fine, explicit is not", but that is an instruction, not
# a guarantee -- this project has already seen a model ignore its own format
# instructions (unlabelled replies, wrapped hashtags) more than once. This is the
# enforced backstop: a real check, not another sentence in the prompt. It is
# deliberately conservative (word-boundary substring match, not an exhaustive list) --
# a false positive costs one caption regenerated from the static pool; a false negative
# costs the account a strike.
# "sex" gets its own explicit list rather than a "sex\w*" wildcard: this project's
# own vocabulary already uses "sexy"/"sexier" (motion_writer.py, and the rubric above
# asks for "a little flirty"), and a wildcard blocked "Feeling sexy and confident
# today" -- an entirely ordinary caption. Narrowing the wildcard with a negative
# lookahead still caught "sexist" and "sextant". Everything else below keeps its
# wildcard: "nude", "porn", "camgirl" etc have no legitimate near-miss the way "sex"
# does, so nude\w* correctly catches "nudes" with no cost.
_UNSAFE_RE = re.compile(
    r"(?<![a-z])(nude\w*|naked|nsfw|porn\w*|xxx|sex|sexual|sexually|sexting|"
    r"explicit|onlyfans|18\s*\+|camgirl\w*|escort\w*|hookup\w*|stripper\w*|"
    r"strip club|thirst\s*trap|hentai\w*|fetish\w*|nipple\w*|orgasm\w*|erotic\w*|"
    r"lewd|horny|cum\b)(?![a-z])", re.I)


def _unsafe_term(text):
    """The matched term, or None. Checked against the caption line and the hashtags
    together -- a clean caption with one bad tag is exactly as postable as one bad
    caption, so both must pass."""
    m = _UNSAFE_RE.search(text or "")
    return m.group(0) if m else None


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
    tags = list(dict.fromkeys(tags))
    # Drop hashtags TikTok would not register as tags BEFORE capping to MAX_TAGS, so
    # one malformed tag does not cost a real one its slot.
    invalid = [t for t in tags if not _VALID_TAG_RE.match(t)]
    if invalid:
        log(f"dropping {len(invalid)} invalid hashtag(s): {invalid}")
        tags = [t for t in tags if t not in invalid]
    tags = tags[:MAX_TAGS]
    if not tags:
        raise RuntimeError(f"no valid hashtags survived: {text[:150]}")
    # The enforced backstop -- see _UNSAFE_RE. Checked on caption and hashtags
    # together, and raised the same way an unparseable reply is: image_caption()'s
    # existing fallback to the static pool is exactly the right behaviour here too.
    bad = _unsafe_term(caption) or _unsafe_term(" ".join(tags))
    if bad:
        raise RuntimeError(f"blocked unsafe term {bad!r} in generated caption/tags")
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
