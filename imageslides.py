"""Image slideshows: generate, QA, and hand off for a native TikTok photo carousel.

Generation is local (sdgen.py): free, no API key. Each run searches CivitAI fresh
(decide_reference) and commits to one checkpoint + one real showcase prompt for the
whole batch; every image in the batch is a camera/lighting variation of that one
prompt (build_variations), not a grab bag of unrelated faces. Every image is then
reviewed by supervisor.py before anything reaches a channel -- a 4-step LCM generation
is fast specifically because it skips the refinement pass that would normally fix
anatomy mistakes, so those are more common here than in a typical 25-50 step render,
not less. A round that leaves the batch short of min_images tries a fresh round of
variations rather than falling back to reused/failed images.

No static prompt fallback: if CivitAI can't be reached, generate() raises instead of
falling back to a hand-written scene/style list -- every batch's checkpoint and prompt
must come from a real, live CivitAI search.

Content policy: adult subjects, nothing exposing genitals/nipples/full nudity. Swimwear
and lingerie are within policy -- the earlier draft of this file blocked them entirely,
which is stricter than necessary and not what was asked for. The line is nudity, not
how much skin an outfit shows.

Delivery: TikTok's Photo Mode carousel, queued as a native inbox draft (tiktok.py),
caption pre-filled on the draft (also saved to CAPTIONS.md as a fallback the account
owner can paste by hand if the pre-fill doesn't take). The caption itself is written
per post by caption_writer.write(), from the actual theme (see DEFAULT_THEMES) that
batch used, not picked from a fixed pool -- the account owner still opens the app to
add a trending sound and post by hand either way.
"""
import os, random, re

import caption_writer
import civitai

try:
    import sdgen
    import supervisor
except ImportError:
    # The lean video-only workflow (autopilot_video.yml, `pip install kaggle
    # requests` only) still does a plain `import imageslides` at module level for
    # image_caption()/decide_reference() -- a live run crashed there with
    # "ModuleNotFoundError: No module named 'PIL'" pulled in via sdgen -> refine ->
    # PIL/torch/diffusers/ultralytics/mediapipe, none of which that workflow
    # installs (or needs -- it only animates an existing image, never generates
    # one). None here, not a lazy import inside generate(): test_pipeline.py's
    # mock.patch.object(imageslides.sdgen, ...) pattern needs these to already be
    # real module attributes before generate() ever runs. generate() itself raises
    # a clear error below if it's actually called without them.
    sdgen = None
    supervisor = None

# Bundled outfit+location+mood+vibe, one coherent "moment" per entry, instead of
# picking each independently. Independent random picks could land a bikini with
# "candlelit bathtub" and "walking toward camera, confident stride" in the same
# image -- individually fine, but reads as a random recombination, not a scene. Each
# theme here is drawn from the same outfit/location/mood vocabulary the old separate
# lists used (nothing lost), just bundled so a whole batch reads as one moment.
# `vibe` is a short, human description of that moment, fed to caption_writer.write()
# so the post's caption/hashtags are actually about what the batch looks like,
# instead of picked from a fixed pool disconnected from the images.
DEFAULT_THEMES = [
    {"vibe": "a lazy morning at home",
     "outfit": "wearing a lace-trim camisole and matching shorts, loungewear at home",
     "location": "in a dimly lit bedroom, silk sheets",
     "mood": "relaxed sultry pose, soft lighting"},
    {"vibe": "a sunny beach day",
     "outfit": "wearing a two-piece bikini, beach setting",
     "location": "on a beach at sunset, waves in the background",
     "mood": "playful confident energy, glamour photography lighting"},
    {"vibe": "poolside at a luxury villa",
     "outfit": "wearing a string bikini, beach setting",
     "location": "poolside at a luxury villa, golden hour",
     "mood": "confident sultry gaze, alluring pose"},
    {"vibe": "getting ready for a night out",
     "outfit": "wearing a fitted bodycon mini dress and heels",
     "location": "backstage in a dressing room, mirror lights",
     "mood": "flirty smile, relaxed confident posture"},
    {"vibe": "a rooftop bar at night",
     "outfit": "wearing a cut-out bodycon dress with side cutouts",
     "location": "on a rooftop bar at night, neon lighting",
     "mood": "sultry pout, confident direct eye contact"},
    {"vibe": "a quiet night in a hotel room",
     "outfit": "wearing a satin slip dress with thin straps",
     "location": "in a hotel room, city lights through the window",
     "mood": "lying back, relaxed sultry pose, soft lighting"},
    {"vibe": "candlelight and quiet",
     "outfit": "wearing a lace lingerie set with a silk robe draped open",
     "location": "in a candlelit bathtub, warm ambient light",
     "mood": "sultry expression, soft dramatic shadows"},
    {"vibe": "steam and soft light",
     "outfit": "wearing a soaking wet, transparent tank top and denim shorts",
     "location": "in a steamy shower, glass fogged with steam",
     "mood": "wet skin, dripping water, glistening body"},
    {"vibe": "a day on the water",
     "outfit": "wearing a plunging halter mini dress, backless",
     "location": "on a private yacht deck, ocean backdrop",
     "mood": "walking toward camera, confident sultry stride"},
    {"vibe": "city lights from the balcony",
     "outfit": "wearing a strapless satin mini dress",
     "location": "on a balcony at night, city skyline behind her",
     "mood": "leaning forward, playful confident energy"},
    {"vibe": "a rainy moody afternoon",
     "outfit": "wearing a sheer mesh top over a bralette",
     "location": "in a rain-soaked room, window backlighting",
     "mood": "sultry expression, soft dramatic shadows"},
    {"vibe": "backstage before a shoot",
     "outfit": "wearing a corset top and a denim mini skirt",
     "location": "backstage in a dressing room, mirror lights",
     "mood": "hands running through hair, sultry confident gaze"},
    {"vibe": "an evening in, curves and candlelight",
     "outfit": "wearing a tight, clingy bodycon dress that hugs every curve",
     "location": "in a red velvet lounge, moody dramatic lighting",
     "mood": "arched back pose, confident sultry expression"},
    {"vibe": "just out of the pool",
     "outfit": "wearing a wet white t-shirt clinging to her figure, poolside",
     "location": "poolside at a luxury villa, golden hour",
     "mood": "tight clingy fabric, curves accentuated, sultry pose"},
]
# "studio" deliberately excluded: it's ubiquitous photography jargon ("studio light",
# "studio backdrop"), not a narrative setting -- including it meant almost every real
# harvested prompt looked like it already had a location and location injection barely
# ever fired, caught by this file's own test suite.
_LOCATION_RE = re.compile(
    r"\b(?:kitchen|cockpit|office|classroom|hospital|courtroom|gym|street|"
    r"park|library|cafe|bedroom|shower|pool|yacht|hotel|rooftop|bathtub|beach|"
    r"lounge|balcony|backstage)\b", re.I)

# Hard line: no exposed nipples/genitals, no real nudity, no minors. Everything else
# (swimwear, lingerie, loungewear) is within policy and is not filtered here.
#
# "caucasian" added after supervisor.py's ethnicity_excluded gate (Chinese/East Asian
# appearance -- an explicit operator preference, see that module) was rejecting far
# too large a share of a live batch. sdgen.py's NEGATIVE_HARD already listed
# "chinese, chinese woman" and that alone wasn't enough -- negative-prompt terms are
# a weak lever against a holistic attribute like perceived ethnicity, especially
# against a checkpoint whose own training skews that way regardless of prompt. A
# positive descriptor in the prefix that's glued onto EVERY image, across every
# checkpoint, is a much stronger lever for SD/SDXL sampling than suppressing it in
# the negative prompt. supervisor.py's gate stays in place either way -- this only
# reduces how often it has to fire, not what it enforces.
SAFETY_PREFIX = "beautiful adult latina woman in her late twenties"
# Every image must read as sexy, not just "sometimes, when the random mood pick lands
# right" -- this is the guaranteed baseline, always present; DEFAULT_MOOD on top of it
# is what varies the specific pose/expression between images in the same batch.
SEXY_CUE = "seductive, sexy, alluring, round breasts, round ass, beautiful"
NEGATIVE_HARD = ("child, teen, minor, young girl, schoolgirl, nude, topless, "
                 "exposed nipples, exposed genitals, explicit sexual content")
# Hand/finger terms expanded per community-standard SD negative-prompt practice (the
# generic "extra fingers, mutated hands" alone measured weaker live than this fuller
# list, which explicitly names each specific finger-count failure mode rather than
# relying on the model to generalize from "extra/mutated").
NEGATIVE_QUALITY = ("cartoon, illustration, painting, anime, 3d render, deformed, "
                    "extra fingers, missing fingers, fused fingers, extra limbs, "
                    "mutated hands, deformed hands, bad hands, malformed hands, "
                    "extra hands, missing hands, bad anatomy, blurry, watermark, "
                    "text, logo, malformed")


def log(msg): print(f"[imageslides] {msg}", flush=True)


# SAFETY_PREFIX always asserts a female adult subject; a harvested prompt describing a
# man (a real, live sample skewed heavily male -- Realistic Vision's own showcase set
# was 5 of 7 male-subject after the age filter) would contradict it in the same prompt,
# which confuses the model rather than overriding cleanly. Filtered per niche, not in
# civitai.py, since that module is subject-agnostic by design.
_MALE_ONLY_RE = re.compile(r"\b(?:man|men|male|guy|boy|father|husband|boyfriend)\b", re.I)
_FEMALE_RE = re.compile(r"\b(?:woman|women|female|girl|lady|mother|wife|girlfriend|she|her)\b",
                        re.I)
# A live sample harvested "photo of autumn landscape, dramatic lighting, gloomy, cloudy
# weather" -- no person in it at all. SAFETY_PREFIX still glued "adult woman" onto that,
# so the model rendered a tiny incidental figure inside a landscape shot instead of a
# portrait. A prompt with no person/portrait signal is not raw material for this niche,
# regardless of how it scores on gender.
_PORTRAIT_RE = re.compile(
    r"\b(?:portrait|closeup|close-up|face|wearing|outfit|dress|fashion|model|headshot|"
    r"selfie|photo of (?:a |an )?(?:woman|girl|lady|person)|editorial|full body|"
    r"full length|standing)\b", re.I)
# Once search can land on any qualifying checkpoint rather than a few pre-vetted
# presets, it can pick a "merge"-style model whose own showcase gallery is mostly
# creature/character blends, not photography of people -- a live search's decided
# prompt, after the celebrity filter removed the worse offenders, was still "a
# humanoid boar Electrode hybrid creature ... portrait photo": it has "portrait" and no
# male-only term, so the checks above alone let it through. This is a distinct failure
# from gender mismatch (nothing unsafe about it, just the wrong subject for this niche
# entirely) and needs its own check.
_NONHUMAN_RE = re.compile(
    r"\b(?:creature|hybrid|monster|beast|humanoid|anthro|furry|dragon|demon|robot|"
    r"android|alien|chimera)\b", re.I)
# A live search's decided prompt (from 'NLIGHT Realistic', an otherwise qualifying
# checkpoint) read "high quality,8K,a girl,blender,3d model,...FASHION SHOOT..." --
# the checkpoint's OWN showcase prompt explicitly asked for a 3D render, which directly
# contradicts NEGATIVE_QUALITY's "3d render" below: positive and negative prompt
# fighting each other in the same generation call, guaranteed confused output. This
# niche wants photorealism, not CGI, regardless of which checkpoint gets picked.
_NONPHOTO_RE = re.compile(
    r"\b(?:3d model|3d render|blender|cgi|render(?:ed)?|cartoon|anime|illustration|"
    r"painting|drawing|sketch)\b", re.I)


def _matches_subject(prompt, niche):
    if not _PORTRAIT_RE.search(prompt) or _NONHUMAN_RE.search(prompt) or _NONPHOTO_RE.search(prompt):
        return False
    if niche.get("subject_gender", "woman") != "woman":
        return True
    return not (_MALE_ONLY_RE.search(prompt) and not _FEMALE_RE.search(prompt))


# "uniform" added after a live search decided on a showcase prompt describing a pilot
# "dressed in a crisp ... uniform" -- without it, an injected outfit (e.g. a bikini)
# would have landed on top of an already-clothed subject, the same two-outfit
# contradiction _CLOTHING_RE exists to prevent for "black dress" and the rest.
_CLOTHING_RE = re.compile(
    r"\b(?:wearing|dress|shirt|coat|jacket|outfit|skirt|jeans|sweater|jumper|top|"
    r"blouse|trousers|swimsuit|bikini|lingerie|suit|blazer|cardigan|gown|uniform)\b", re.I)


DEFAULT_CIVITAI_QUERIES = [
    "portrait woman fashion photography editorial",
    "beauty portrait woman fashion photography glamour",
    "glamour photography woman",
    "realistic portrait woman photography",
]


# A checkpoint needs at least this many recorded batches before its track record is
# trusted enough to influence odds -- one lucky or unlucky early batch (small sample,
# especially a round of only 3-6 images) must not permanently tilt selection. Below
# this, or for a checkpoint never seen before, weight is the neutral 1.0 baseline
# civitai.py already applies to any candidate missing from this dict.
MIN_SAMPLES_TO_TRUST = 3
# How far a track record can move odds off the neutral 1.0 baseline in EITHER
# direction, at the extremes (100%/0% pass rate). Originally 1.0-3.0 (i.e. only ever
# boosted a good performer, floor was the same 1.0 an untested checkpoint already
# gets) -- a checkpoint that consistently produced bad batches (anatomy, or a
# checkpoint-level bias like ethnicity_excluded -- see supervisor.py's rubric) was
# NEVER actually suppressed below the odds a brand-new, never-tried checkpoint gets,
# so a known-bad one kept getting picked at the same rate forever. Now spans below
# 1.0 too, so a proven-bad checkpoint actually becomes less likely, not just
# "no more likely than an unknown".
MIN_WEIGHT_MULTIPLIER = 0.15
MAX_WEIGHT_MULTIPLIER = 3.0


def _model_weights(state):
    """{model_id: float} from state["model_stats"] (used/passed counts per "model_id:
    version_id" spec, recorded by generate() after each round's QA) -- fed to
    civitai.decide_reference() to nudge future runs toward checkpoints with a good
    pass rate and away from ones with a bad one. None/missing state -> {}, which
    civitai.py already treats as "no preference, plain shuffle"."""
    stats = (state or {}).get("model_stats") or {}
    weights = {}
    for spec, s in stats.items():
        used = s.get("used", 0)
        if used < MIN_SAMPLES_TO_TRUST:
            continue
        model_id = spec.split(":", 1)[0]
        try:
            model_id = int(model_id)
        except ValueError:
            continue
        pass_rate = s.get("passed", 0) / used
        weights[model_id] = (MIN_WEIGHT_MULTIPLIER
                             + pass_rate * (MAX_WEIGHT_MULTIPLIER - MIN_WEIGHT_MULTIPLIER))
    return weights


def _record_model_result(state, spec, name, generated, approved):
    """After a round's QA, log this checkpoint's outcome into state["model_stats"] --
    the raw material _model_weights() turns into future selection odds. Keyed by the
    canonical "model_id:version_id" spec, same as everywhere else in this file."""
    if state is None:
        return
    stats = state.setdefault("model_stats", {})
    entry = stats.setdefault(spec, {"name": name, "used": 0, "passed": 0})
    entry["name"] = name  # keep the latest name, checkpoints occasionally get renamed
    entry["used"] += generated
    entry["passed"] += approved


def _theme_weights(state):
    """{vibe: float} from state["theme_stats"], the same shape and the same
    thresholds _model_weights() applies to checkpoints -- a theme whose batches keep
    getting rejected (a pose/setting this checkpoint family renders badly, an outfit
    the vision QA keeps failing) should get picked less often, and one that keeps
    passing more often. Untested themes stay at the neutral 1.0 so the list keeps
    exploring instead of collapsing onto whichever theme happened to go first."""
    stats = (state or {}).get("theme_stats") or {}
    weights = {}
    for vibe, s in stats.items():
        used = s.get("used", 0)
        if used < MIN_SAMPLES_TO_TRUST:
            continue
        pass_rate = s.get("passed", 0) / used
        weights[vibe] = (MIN_WEIGHT_MULTIPLIER
                         + pass_rate * (MAX_WEIGHT_MULTIPLIER - MIN_WEIGHT_MULTIPLIER))
    # The owner's own verdicts multiply on top, and can apply to a theme with no QA
    # history yet (it starts from the neutral 1.0 in that case) -- a theme the owner
    # keeps refusing to post should get picked less often even if every image in it
    # passed QA cleanly, which is exactly the case QA alone can't see.
    for vibe, posted_rate in _owner_theme_rates(state).items():
        factor = OWNER_MIN_FACTOR + posted_rate * (OWNER_MAX_FACTOR - OWNER_MIN_FACTOR)
        combined = weights.get(vibe, 1.0) * factor
        weights[vibe] = min(MAX_WEIGHT_MULTIPLIER,
                            max(MIN_WEIGHT_MULTIPLIER, combined))
    return weights


# How far the account owner's own posted/skipped verdicts may move a theme's odds,
# multiplied on top of the QA-derived weight. Narrower than the QA range on purpose:
# vision QA sees every image in a batch, while a verdict is one judgement about one
# draft, so it accumulates evidence far more slowly.
OWNER_MIN_SAMPLES = 3
OWNER_MIN_FACTOR = 0.5
OWNER_MAX_FACTOR = 1.5


def _owner_theme_rates(state):
    """{vibe: posted_rate} from the account owner's own verdicts in
    state["uploads"] -- picker.html writes owner_verdict ("posted"/"skipped") onto an
    upload entry, and autopilot records that entry's `vibe`.

    This is the only real engagement signal available without TikTok's video.list
    scope (which needs an app audit this unaudited app can't get). QA pass rate says
    an image was well-formed; a verdict says a human actually wanted to post it, which
    is a different and more interesting question."""
    counts = {}
    for u in (state or {}).get("uploads", []):
        verdict, vibe = u.get("owner_verdict"), u.get("vibe")
        if not vibe or verdict not in ("posted", "skipped"):
            continue
        seen, posted = counts.get(vibe, (0, 0))
        counts[vibe] = (seen + 1, posted + (verdict == "posted"))
    return {vibe: posted / seen for vibe, (seen, posted) in counts.items()
            if seen >= OWNER_MIN_SAMPLES}


def _record_theme_result(state, vibe, generated, approved):
    """Sibling of _record_model_result, keyed by the theme's own vibe string."""
    if state is None or not vibe:
        return
    stats = state.setdefault("theme_stats", {})
    entry = stats.setdefault(vibe, {"used": 0, "passed": 0})
    entry["used"] += generated
    entry["passed"] += approved


def decide_reference(niche, state=None):
    """Search CivitAI for a checkpoint with a real, on-subject showcase prompt, and
    commit to it for this whole run: one model, one reference photo, every slide in
    the batch is a variation of it, not a grab bag of unrelated faces.

    The search query itself is randomised per run too (civitai_queries, a list; a
    single civitai_query string still works as a one-item list) -- civitai.decide_
    reference() already randomises which qualifying model+prompt a fixed query
    resolves to, but a fixed query alone still biases toward whatever CivitAI ranks
    highest for it. Rotating the query is what actually varies the THEME run to run,
    not just the exact photo within one theme.

    state, when given, supplies model_stats -- past QA pass rates -- as soft odds
    (see _model_weights()); without it, selection is a plain shuffle, same as before
    this existed.

    civitai.py's search_candidates()/decide_reference() stay subject-agnostic; the
    niche's gender/portrait rules and preference weights are passed in rather than
    living there."""
    queries = niche.get("civitai_queries") or (
        [niche["civitai_query"]] if niche.get("civitai_query") else DEFAULT_CIVITAI_QUERIES)
    query = random.choice(queries)
    resolved, reference = civitai.decide_reference(
        query, prompt_filter=lambda p: _matches_subject(p, niche),
        weights=_model_weights(state))
    log(f"decided on {resolved['name']!r} v{resolved['version_id']} (query: {query!r}), "
        f"prompt: {reference['prompt'][:100]}")
    return resolved, reference


# Ten variations of one decided prompt, not ten unrelated prompts -- the subject,
# outfit and checkpoint stay fixed for the whole batch, only framing/light/angle
# differ, matching how one real photoset of one person actually looks across a
# carousel rather than reading as ten different people.
CAMERA_MODIFIERS = [
    "close-up shot, shallow depth of field",
    "three-quarter view, soft studio lighting",
    "wide shot, natural window light",
    "slightly low angle, golden hour light",
    "profile view, moody side lighting",
    "over-the-shoulder framing, warm ambient light",
    "eye-level shot, diffused overcast light",
    "slight high angle, soft backlighting",
    "medium shot, gentle rim lighting",
    "candid framing, natural daylight",
    # None of the above ever actually asked for full-body framing -- "medium shot",
    # "three-quarter view" etc. read as waist-up in practice, so the whole mix skewed
    # portrait/closeup regardless of which reference prompt got picked.
    "full body shot, standing pose, wide framing",
    "full length shot, head to toe, natural stance",
    "full body portrait, standing, wide angle",
]


def build_variations(prefix, reference_text, base_negative, n, niche):
    """prefix (identity, clothing, mood) and the camera modifier go BEFORE the
    harvested reference text, not after. CLIP hard-truncates at 77 tokens, and a live
    run hit a showcase prompt long enough on its own to push everything appended after
    it clean off the end -- mood and camera framing included, confirmed by diffusers'
    own truncation warning. Putting the short, controlled part first means a long
    reference prompt loses its own tail to truncation instead of silently eating ours."""
    mods = niche.get("camera_modifiers") or CAMERA_MODIFIERS
    picked = random.sample(mods, min(n, len(mods)))
    while len(picked) < n:
        picked.append(random.choice(mods))
    prompts = [f"{prefix}, {m}, {reference_text}" for m in picked]
    return prompts, [base_negative] * n


def _static_caption(niche):
    """The old fixed-pool behaviour, kept as a fallback for when the LLM path is
    unavailable or fails -- niches.json's own "hashtags" used to be a single string
    posted on literally every image, forever, and "captions" a small pool (14
    phrases) that cycles back to the same lines regardless of how different the
    images actually are. Never the primary path anymore; see image_caption()."""
    lines = niche.get("captions") or ["Slow mornings and soft light."]
    tags = niche.get("hashtags", "")
    disclosure = niche.get("ai_disclosure", "AI-generated imagery")
    return f"{random.choice(lines)}\n\n{disclosure}\n\n{tags}".strip()


def image_caption(niche, vibe=None, state=None):
    """A fresh caption + hashtags per post, written by caption_writer.write() from
    this batch's actual theme (vibe) -- falls back to the old static pool
    (_static_caption) when no vibe is available (a niche overriding "themes" with
    something that has no vibe field, unlikely) or the LLM call fails for any
    reason. A caption-writing hiccup must never block an otherwise-good batch of
    images from getting posted, so this never raises.

    state is passed through so caption_writer can cache the day's trend lookup on it
    (trends.py) instead of refetching per run; niche carries that niche's own manual
    trend_hashtags list. Both optional -- without them the rubric just carries no
    trend hint."""
    disclosure = niche.get("ai_disclosure", "AI-generated imagery")
    if vibe:
        try:
            caption, tags = caption_writer.write(vibe, niche=niche, state=state)
            return f"{caption}\n\n{disclosure}\n\n{tags}".strip()
        except Exception as e:
            log(f"caption_writer failed, falling back to the static pool "
                f"({type(e).__name__}: {str(e)[:100]})")
    return _static_caption(niche)


def _build_prefix(niche, reference, state=None):
    """Returns (prefix, vibe) -- vibe is the theme's short human description, threaded
    through to caption_writer.write() so the post's caption/hashtags are actually
    about this batch's moment, not picked from a fixed pool disconnected from it.

    state, when given, biases WHICH theme gets picked by that theme's own past QA
    pass rate (_theme_weights) -- the same learn-from-outcomes loop decide_reference
    already runs for checkpoints, extended to the other half of what determines how a
    batch looks. Without state it is a plain uniform pick, exactly as before."""
    # An explicit outfit/location is only injected when the reference prompt does not
    # already name one -- appending "wearing jeans and a coat" onto a prompt that
    # already says "wearing a black dress" (or "poolside cabana" onto "in a bustling
    # gourmet kitchen") gives the model two contradictory settings in one prompt.
    # SEXY_CUE and mood are both added unconditionally -- unlike outfit/location, a
    # pose/expression cue does not conflict with whatever the reference prompt already
    # says. SEXY_CUE is the guaranteed baseline ("every image must be sexy" is a hard
    # requirement, not a random pick); mood adds per-image variety on top of it.
    themes = niche.get("themes") or DEFAULT_THEMES
    # Weighted by past QA outcome per theme (state["theme_stats"]), the same way
    # decide_reference already weights checkpoints. Falls back to a plain uniform
    # pick when there is no state or nothing has enough samples yet -- identical
    # behaviour to the random.choice this replaces.
    weights = [_theme_weights(state).get(t.get("vibe"), 1.0) for t in themes]
    theme = random.choices(themes, weights=weights, k=1)[0]
    clothing = "" if _CLOTHING_RE.search(reference["prompt"]) else f"{theme['outfit']}, "
    setting = "" if _LOCATION_RE.search(reference["prompt"]) else f"{theme['location']}, "
    prefix = f"{SAFETY_PREFIX}, {clothing}{setting}{SEXY_CUE}, {theme['mood']}"
    return prefix, theme["vibe"]


# Bounds for adopting the checkpoint creator's own posted generation settings. Their
# resolution is trusted more broadly (SD1.5 checkpoints commonly showcase anywhere in
# this band); steps/cfg_scale are trusted only within LCM's own realistic range --
# even when their sampler was LCM-family, a wildly out-of-range posted value (bad data,
# a typo) must not be allowed to make the whole batch of `count` images (all sharing
# this same reference) blow past the CI time budget.
_SIZE_MIN, _SIZE_MAX = 384, 896
_STEPS_MIN, _STEPS_MAX = 4, 10
_CFG_MIN, _CFG_MAX = 1.0, 2.5


def _adopted_settings(reference):
    """What to actually reuse from the checkpoint creator's own posted example, and
    what to leave to our own defaults. The creator's resolution and (when their own
    sampler was already LCM-family) their steps/cfg_scale are exactly the settings
    that earned that image its engagement -- they know their own checkpoint better
    than a fixed guess does. A normal 25-40-step DPM++/Euler posted example is NOT
    portable to our fused-LCM pipeline, though: copying its step count without also
    switching schedulers would not reproduce their result, just run needlessly slow."""
    out = {}
    w, h = reference.get("width"), reference.get("height")
    if w and h and _SIZE_MIN <= w <= _SIZE_MAX and _SIZE_MIN <= h <= _SIZE_MAX:
        # SD's VAE downsamples by 8, so width/height must be multiples of 8 -- a real
        # run crashed every single image in a round on this exact gap: a checkpoint's
        # own posted Size (1025x768 style values do happen) was 513, which passed the
        # min/max range check above but isn't divisible by 8, and diffusers raises
        # rather than silently rounding. Round to the nearest multiple of 8 instead of
        # rejecting the setting outright -- still much closer to what the creator
        # actually posted than falling back to our own fixed default.
        out["width"] = max(_SIZE_MIN, min(_SIZE_MAX, round(w / 8) * 8))
        out["height"] = max(_SIZE_MIN, min(_SIZE_MAX, round(h / 8) * 8))

    sampler = (reference.get("sampler") or "").lower()
    if "lcm" in sampler:
        steps = reference.get("steps")
        if isinstance(steps, (int, float)) and _STEPS_MIN <= steps <= _STEPS_MAX:
            out["steps"] = int(steps)
        cfg = reference.get("cfg_scale")
        if isinstance(cfg, (int, float)) and _CFG_MIN <= cfg <= _CFG_MAX:
            out["guidance"] = float(cfg)
    return out


def generate(niche, count=None, workdir=None, max_rounds=2, state=None):
    """Decide on a CivitAI checkpoint + reference prompt, generate `count` camera
    variations of it, keep what passes supervisor.py review, and repeat with a fresh
    round if the batch still falls short of min_images.

    A round that ends with ZERO approved images -- whether nothing generated at all
    (a gated CivitAI model 401'd on every download attempt, a live run hit this
    exactly) or a full batch generated but every one rejected by supervisor.py (a
    checkpoint whose trained style just isn't photorealistic, however the prompt is
    worded, scores low on every image) -- means the checkpoint itself is the wrong
    pick for this run, not that the prompts were bad. Either way the NEXT round
    re-decides (fresh search, likely a different checkpoint) instead of paying for a
    second expensive round (refine + upscale run on every image before supervisor
    ever sees it) on the same checkpoint.

    state, when given (autopilot.py's posted.json-backed dict), gets each round's
    checkpoint outcome recorded into state["model_stats"] (_record_model_result) and
    feeds past outcomes back into decide_reference() as soft selection odds -- the
    "learn which checkpoints/prompts tend to work" loop. Without it, generate()
    behaves exactly as before: plain shuffle, no memory across runs.

    Raises once max_rounds is exhausted without reaching min_images -- an image-only
    post with too few photos is not worth publishing, and a silent quality drop should
    never ship. Not an infinite retry: max_rounds caps the worst case at
    max_rounds * count generations, each ~130s on a GH Actions runner.

    Returns (image_paths, vibe, image_prompts) -- vibe is the theme's short
    description (see DEFAULT_THEMES), threaded through so the caller can write a
    caption that's actually about this batch's moment, not a generic one.
    image_prompts is the actual per-image SD prompt behind each returned path
    (same order), so a picker UI can show a motion-forge suggestion grounded in
    what that specific photo's framing/pose/lighting actually is, instead of one
    generic prompt shared across the whole batch. Entries can be None for a path
    whose originating prompt couldn't be recovered (defensive; shouldn't happen
    in practice since sdgen.generate_batch names files sd_<index>.jpg)."""
    if sdgen is None or supervisor is None:
        raise RuntimeError(
            "sdgen/supervisor unavailable (PIL/torch/diffusers not installed) -- "
            "generate() needs the full local image-gen stack; the video-only "
            "workflow should never call this, only image_caption()/decide_reference()")
    import tempfile
    from pathlib import Path

    count = count or int(niche.get("images_per_video", 10))
    min_images = int(niche.get("min_images", 3))
    max_images = int(niche.get("max_images", niche.get("images_per_video", 5)))
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="imageslides_"))
    workdir.mkdir(parents=True, exist_ok=True)

    supervisor_on = os.environ.get("SUPERVISOR_ENABLED", "1").strip().lower() not in (
        "0", "false", "no")
    approved, generated_count = [], 0
    # str(path) -> the SD prompt that actually produced it, so the caller can hand a
    # picker UI a per-image motion-forge suggestion instead of one prompt for the
    # whole batch. sdgen.generate_batch names files sd_<index>.jpg (index into that
    # round's `prompts`), which survives supervisor.py's filtering unchanged.
    prompt_by_path = {}
    # Broken-supervisor fallback book-keeping. Populated by rounds where the
    # supervisor could not reach a verdict on ANY image (mllama arch error, model
    # server down); consumed after the last round to decide whether to push raw
    # generations instead of failing the whole run.
    supervisor_broken_batches = 0
    broken_generations = []
    vibe = None  # the last round's theme vibe -- returned alongside the images so the
    # caller can write a caption that's actually about this batch's moment.
    # No static-formula fallback: every batch's model and prompt must come from a real
    # CivitAI search, not a hand-written scene/style list. If CivitAI can't be reached
    # at all, decide_reference() raises and the run fails loudly here, rather than
    # silently shipping generic images.
    resolved, reference = decide_reference(niche, state=state)

    for round_num in range(1, max_rounds + 1):
        civitai_spec = f"{resolved['model_id']}:{resolved['version_id']}"
        prefix, vibe = _build_prefix(niche, reference, state=state)
        base_negative = ", ".join(
            x for x in (NEGATIVE_HARD, reference["negative_prompt"], NEGATIVE_QUALITY) if x)
        log(f"round {round_num}: {resolved['name']!r} | prefix: {prefix} | "
            f"reference: {reference['prompt'][:120]}")

        prompts, negatives = build_variations(
            prefix, reference["prompt"], base_negative, count, niche)
        adopted = _adopted_settings(reference)
        if adopted:
            log(f"round {round_num}: using the checkpoint creator's own posted "
                f"settings where safe: {adopted}")
        try:
            # civitai_model always wins over model_key in sdgen (see its docstring),
            # and every batch now always has one -- no model_key to pass here at all.
            generated = sdgen.generate_batch(
                prompts, workdir / f"round{round_num}",
                negative_prompts=negatives, civitai_model=civitai_spec, **adopted)
        except RuntimeError as e:
            generated = []
            log(f"round {round_num}: checkpoint unusable ({str(e)[:150]})")

        for path in generated:
            m = re.search(r"sd_(\d+)\.[^.]+$", str(path))
            if m and int(m.group(1)) < len(prompts):
                prompt_by_path[str(path)] = prompts[int(m.group(1))]

        generated_count += len(generated)
        if supervisor_on:
            filtered = supervisor.filter_images(generated)
            newly_approved = list(filtered)
            if getattr(filtered, "supervisor_broken", False) and generated:
                # Supervisor could not reach a verdict on ANY image this round
                # (mllama arch missing, model server down, network error) --
                # not a content signal. Record it so the outer fallback below
                # can decide to push raw generations rather than throwing away
                # a full round of successful image gen over an unavailable QA.
                supervisor_broken_batches += 1
                broken_generations.extend(generated)
                log(f"round {round_num}: supervisor could not reach a verdict "
                    f"on any of {len(generated)} images -- treating supervisor "
                    f"as unavailable for this round")
        else:
            newly_approved = generated
            log("SUPERVISOR_ENABLED=0: skipping vision QA")
        approved.extend(newly_approved)
        _record_model_result(state, civitai_spec, resolved['name'],
                             len(generated), len(newly_approved))
        _record_theme_result(state, vibe, len(generated), len(newly_approved))
        log(f"round {round_num}/{max_rounds}: {len(newly_approved)}/{len(generated)} passed, "
            f"{len(approved)}/{min_images} needed")
        if len(approved) >= min_images:
            break
        if not newly_approved and round_num < max_rounds:
            log("nothing approved this round; re-deciding a fresh checkpoint+prompt")
            sdgen.unload_all()
            resolved, reference = decide_reference(niche, state=state)

    if len(approved) < min_images:
        # Every image we DID successfully generate was rejected because the
        # supervisor couldn't render a verdict (never because it judged content
        # bad). Push raw generations rather than lose a real batch of image gen
        # to broken QA infra -- the account owner still reviews every draft in
        # the TikTok app before publishing, so the human gate downstream isn't
        # bypassed. If supervisor was ONLY partially broken (some real verdicts,
        # some errors), we don't fall back -- that would be a real safety hole.
        if supervisor_broken_batches and broken_generations:
            log(f"supervisor unavailable across all rounds -- pushing "
                f"{len(broken_generations)} raw generations without QA "
                f"(account owner still reviews the draft in TikTok before posting)")
            kept = broken_generations[:max_images]
            return kept, vibe, [prompt_by_path.get(str(p)) for p in kept]
        raise RuntimeError(
            f"only {len(approved)} of {generated_count} images passed review across "
            f"{max_rounds} round(s) (need at least {min_images}); not posting")
    log(f"{len(approved)}/{generated_count} images approved")
    kept = approved[:max_images]
    return kept, vibe, [prompt_by_path.get(str(p)) for p in kept]
