"""Turns a still image's own SD generation prompt into a short motion instruction
for image-to-video -- an actual action for the person to do, not a re-description of
the photo's outfit/setting/lighting that the SD prompt already is. Feeding the SD
prompt straight into I2V as the motion prompt asks the video model to reproduce the
STILL, not animate it.

The instruction has to be SEDUCTIVE, matching imageslides.SEXY_CUE's own "every image
must be sexy" baseline on the photo side, and it should show her BODY -- a still
already shows the face; turning, walking and arching are what a photo cannot do.

Three rounds of live testing shaped the rubric, each fixing what the previous one
actually produced rather than what it was supposed to:

  1. Asking for "a smile, a glance, a head tilt, a hand movement" returned exactly
     that -- "She raises an eyebrow and gently touches the glass with her index
     finger" for a rooftop-bar photo. A fine instruction for a different account; the
     still was doing all the work and the motion was undoing it.
  2. Listing every example movement inline made the model copy one: the same
     collarbone-to-hip line came back for four different photos, near-verbatim. Hence
     _movement_menu()'s per-call sample.
  3. Sampling that menu uniformly produced ZERO turns across six photos -- the model
     reliably prefers hands and faces. Turns only started appearing once REVEALS got
     guaranteed slots AND the rubric made a turn a RULE for wide framings rather than
     a preference. That took it from 1/8 to 3/4 of wide shots.

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
#
# Split into two groups because an unweighted list does not produce the mix this
# account wants. With all of these pooled together and five sampled at random, a live
# spread of six photos came back with ZERO turns: the model reliably prefers hands and
# posture, so "turning around to show her body" -- the single most-wanted movement
# here -- simply never got written. REVEALS is therefore guaranteed slots in every
# menu rather than left to chance.
#
# Every entry is clothed, non-explicit, and squarely inside what TikTok's own feed is
# full of. The policy line lives in the rubric, not here: each of these gets handed to
# a video model that interprets loosely, so none of them so much as mentions clothing
# coming off or being moved.

# Whole-body movement -- the part a still photo cannot do.
REVEALS = [
    "she turns away from the camera, hips leading, and looks back over her shoulder",
    "a slow turn in place, letting the camera read her whole silhouette",
    "she starts to walk away, hips swaying, and glances back",
    "turning her back to the camera, arching, then looking round at the lens",
    "a slow pivot from profile to facing the camera, weight rolling through her hips",
    "she walks toward the camera, hips swaying, holding its gaze",
    "she rises up out of the pose, body lengthening as she straightens",
    "hands smoothing down her sides, from her ribs to her hips",
    "her hands run up her thighs and settle on her hips",
    "a slow arch of the back, hips shifting, weight moving onto one leg",
]

# Everything else: posture, breath, face. Carries the tighter framings, where a full
# turn has nowhere to go.
ACCENTS = [
    "a hand tracing her collarbone, sliding down her waist, resting on her hip",
    "fingers running through her hair, pushing it back off her shoulder",
    "shoulders rolling back, chest lifting, waist drawing in",
    "she leans forward toward the camera, then rolls slowly back up",
    "sinking lower into the pose, then rising back through it",
    "crossing one leg over the other, settling back into the pose",
    "a slow stretch, arms lifting, spine lengthening",
    "a look over the shoulder, chin dropping, eyes coming up to the lens",
    "lips parting, a slow breath, a knowing half-smile",
    "biting her lip, then letting a smile break through",
    "head tipping back, throat exposed, eyes closing for a beat",
    "fabric or wet skin catching the light as she moves",
]
MOVEMENTS = REVEALS + ACCENTS
# Per menu: this many whole-body reveals, plus this many accents.
MENU_REVEALS = 3
MENU_ACCENTS = 2
MENU_SIZE = MENU_REVEALS + MENU_ACCENTS

RUBRIC = """A still photo was generated from this prompt:
{image_prompt}

It's about to be animated into a short clip with an image-to-video model, starting
from that exact photo. Write ONE motion instruction for the video.

The clip has to read as seductive -- that is the whole point of the account, not a
flourish. The still is already sexy; the motion must carry that, not neutralise it.
Prefer movement that shows her BODY, not just her face: a still already shows the
face, and turning, walking, arching, or running her own hands over her shape is the
part a photo cannot do. Slow and deliberate, never busy. For instance:

{menu}

Match it to the FRAMING, which the prompt above names:

  - Wide, full-body, or full-length shot: the instruction MUST change her
    orientation -- turning, pivoting, walking, or turning away and looking back.
    Hands moving on a body that stays put wastes the one thing a wide shot is for.
  - Medium or three-quarter shot: a turn of the upper body, an arch, a shift of
    weight through the hips, or her hands following her own shape.
  - Close-up: her face and breath -- a turn has nowhere to go in that frame.

Otherwise pick whichever example above suits this photo, or write something in the
same spirit that fits it better. Plus at most ONE ambient motion (hair, fabric,
water, light) on top of it.

Keep it to what {length_s} seconds can actually hold. The model has that long and no
more, so a movement that BEGINS and reads clearly beats one that has to complete: she
turns partway and looks back, rather than spinning all the way round.

Write it in plain words -- no numbers of any kind. No degrees ("pivots 180 degrees"),
no timings ("holds for 0.5 seconds"), no counts. A video model reads those as noise
and a real live run produced both.

Hard limits -- the clip is posted to TikTok, so breaking these gets it removed:
she stays fully clothed in exactly what she is already wearing, no undressing, no
removing, lifting or pulling aside clothing, no hands between her legs or on her
chest, no simulated sex acts, nothing explicit. Suggestive, not pornographic --
the line is what a real creator posts to a public feed.

Do NOT restate her appearance, outfit, or the setting -- the video model already has
the photo, it only needs to know what moves. Under 25 words. No camera directions
unless the prompt above already implies one. No quotes, no preamble, just the
instruction itself."""


def log(msg): print(f"[motion_writer] {msg}", flush=True)


def _movement_menu(reveals=None, accents=None):
    """A short, freshly sampled menu, weighted toward whole-body movement.

    Two separate findings drove this shape. Showing the model the ENTIRE list made it
    pick the same entry every time: a live run returned "her hand slides from her
    collarbone down her waist and rests on her hip" for four different photos in a
    row, near-verbatim. And pooling every movement into one uniform sample produced no
    turns at all across six photos -- the model's own preference is for hands and
    faces, so the reveal vocabulary has to be given slots, not offered them.

    Sampling per call also keeps the menu short enough to read as examples rather than
    a list to choose from, and different enough per image that a carousel's prompts
    actually diverge (see autopilot._motion_prompts_for on why one repeated prompt
    across a batch is worse than none at all)."""
    picked = (random.sample(REVEALS, reveals or MENU_REVEALS)
              + random.sample(ACCENTS, accents or MENU_ACCENTS))
    # Shuffled so the reveals aren't always the first thing it reads -- an ordering
    # the model would otherwise learn to treat as a ranking.
    random.shuffle(picked)
    return "\n".join(f"  - {m}" for m in picked)


# A trailing parenthetical the model tacks on to editorialise about its own answer --
# "(slow, deliberate, seductive motion)", seen live. Harmless to a reader, but it goes
# straight into an I2V prompt where it is just tokens describing nothing that moves.
# Only stripped at the very END, so a legitimate mid-sentence aside survives.
_TRAILING_ASIDE_RE = re.compile(r"\s*\([^()]*\)\s*$")
_LABEL_RE = re.compile(r"^(?:motion instruction:|instruction:)\s*", re.I)


def _clean(raw):
    motion = raw.strip().strip('"').strip()
    motion = _LABEL_RE.sub("", motion)
    motion = _TRAILING_ASIDE_RE.sub("", motion)
    return motion.strip().strip('"').strip()


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


# Matches niches.json's motionforge_length_s and autopilot_video.yml's own length_s
# input default. Only used to tell the model how much time the movement has to fit
# into, so a per-niche override isn't worth threading through _motion_prompts_for --
# every clip this account posts is five seconds.
DEFAULT_LENGTH_S = 5.0


def write(image_prompt, length_s=DEFAULT_LENGTH_S, attempts=2):
    """A short, seductive motion instruction for animating the photo that
    image_prompt generated. Raises on failure -- callers fall back to niches.json's
    own static motionforge_prompt rather than this module retrying forever; a hiccup
    here must never block recording an otherwise-good photo batch."""
    motion = None
    for i in range(attempts):
        raw = llm.ask(RUBRIC.format(image_prompt=image_prompt,
                                    menu=_movement_menu(),
                                    length_s=length_s),
                      max_tokens=80, temperature=0.95)
        motion = _clean(raw)
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
