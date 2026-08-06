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

Content policy: adult subjects, nothing exposing genitals/nipples/full nudity. Swimwear
and lingerie are within policy -- the earlier draft of this file blocked them entirely,
which is stricter than necessary and not what was asked for. The line is nudity, not
how much skin an outfit shows.

Delivery: TikTok's Photo Mode carousel, posted directly through Buffer
(buffer.publish_photos), which was verified live against this account -- a multi-image
post was accepted and returned PostActionSuccess. That gets the caption, hashtags and
AI-disclosure attached automatically, the same as the video pipeline, with no manual
step. It replaces an earlier design that shipped a silent video and relied on the
account owner adding a TikTok sound by hand; carousels do not need that workaround.
"""
import os, random, re

import civitai
import sdgen
import supervisor

# Explicit outfits rather than a general "modest clothing" instruction: an earlier
# attempt at the latter still produced a subject in underwear on a public-street prompt.
# Naming a complete, specific outfit per image leaves nothing for the model to infer.
DEFAULT_OUTFITS = [
    "wearing straight-leg jeans, a white shirt and a long wool coat",
    "wearing a midi skirt below the knee, a knitted jumper and ankle boots",
    "wearing tailored trousers, a turtleneck and a trench coat",
    "wearing wide-leg trousers, a blouse buttoned to the collar and a blazer",
    "wearing a long-sleeved midi dress with opaque tights and flat shoes",
    "wearing corduroy trousers, a cardigan over a t-shirt and a scarf",
    "wearing a one-piece swimsuit and a light cover-up, beach setting",
    "wearing a lace-trim camisole and matching shorts, loungewear at home",
    "wearing a fitted bodycon mini dress and heels",
    "wearing a two-piece bikini, beach setting",
    "wearing a cropped tank top and high-waisted denim shorts",
    "wearing a satin slip dress with thin straps",
    "wearing a high-cut one-piece swimsuit, poolside",
]
# Pose/expression, appended unconditionally regardless of where the reference prompt
# came from -- this is the actual tone lever, not the outfit list above (which most
# harvested prompts already satisfy on their own and skip entirely).
DEFAULT_MOOD = [
    "confident sultry gaze, alluring pose",
    "sensual over-the-shoulder glance, soft smile",
    "playful confident energy, glamour photography lighting",
    "sultry expression, soft dramatic shadows",
    "flirty smile, relaxed confident posture",
    "sultry pout, confident direct eye contact",
]
DEFAULT_SCENES = [
    "walking through a sunlit city street in autumn",
    "sitting by a cafe window on a rainy afternoon",
    "standing on a balcony at golden hour overlooking rooftops",
    "browsing a flower market in the morning",
    "reading in a quiet library with tall windows",
    "waiting at a train platform in soft winter light",
    "relaxing on a sunlounger by a pool",
    "sitting on a bed in a sunlit bedroom in the morning",
]
DEFAULT_STYLES = [
    "35mm film photography, shallow depth of field",
    "editorial fashion photography, soft natural light",
    "cinematic still, warm colour grade",
]

# Hard line: no exposed nipples/genitals, no real nudity, no minors. Everything else
# (swimwear, lingerie, loungewear) is within policy and is not filtered here.
SAFETY_PREFIX = "adult woman in her late twenties"
NEGATIVE_HARD = ("child, teen, minor, young girl, schoolgirl, nude, topless, "
                 "exposed nipples, exposed genitals, explicit sexual content")
NEGATIVE_QUALITY = ("cartoon, illustration, painting, anime, 3d render, deformed, "
                    "extra fingers, extra limbs, mutated hands, bad anatomy, blurry, "
                    "watermark, text, logo")


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
    r"selfie|photo of (?:a |an )?(?:woman|girl|lady|person)|editorial)\b", re.I)
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


def _matches_subject(prompt, niche):
    if not _PORTRAIT_RE.search(prompt) or _NONHUMAN_RE.search(prompt):
        return False
    if niche.get("subject_gender", "woman") != "woman":
        return True
    return not (_MALE_ONLY_RE.search(prompt) and not _FEMALE_RE.search(prompt))


_CLOTHING_RE = re.compile(
    r"\b(?:wearing|dress|shirt|coat|jacket|outfit|skirt|jeans|sweater|jumper|top|"
    r"blouse|trousers|swimsuit|bikini|lingerie|suit|blazer|cardigan|gown)\b", re.I)


def decide_reference(niche):
    """Search CivitAI once for a checkpoint with a real, on-subject showcase prompt,
    and commit to it for this whole run: one model, one reference photo, every slide
    in the batch is a variation of it, not a grab bag of unrelated faces.

    civitai.py's search_candidates()/decide_reference() stay subject-agnostic; the
    niche's gender/portrait rules are passed in as a filter rather than living there."""
    query = niche.get("civitai_query", "portrait woman fashion photography editorial")
    resolved, reference = civitai.decide_reference(
        query, prompt_filter=lambda p: _matches_subject(p, niche))
    log(f"decided on {resolved['name']!r} v{resolved['version_id']}, "
        f"prompt: {reference['prompt'][:100]}")
    return resolved, reference


def _formula_reference(niche):
    """Fallback only, used when CivitAI cannot be reached at all: a reference prompt
    built from the niche's own scene/style/outfit lists instead of a harvested one."""
    scenes = niche.get("scenes") or DEFAULT_SCENES
    styles = niche.get("styles") or DEFAULT_STYLES
    prompt = (f"RAW photo, {random.choice(scenes)}, "
             f"{random.choice(styles)}, detailed skin texture")
    return None, {"prompt": prompt, "negative_prompt": ""}


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


def image_caption(niche):
    lines = niche.get("captions") or ["Slow mornings and soft light."]
    tags = niche.get("hashtags", "")
    disclosure = niche.get("ai_disclosure", "AI-generated imagery")
    return f"{random.choice(lines)}\n\n{disclosure}\n\n{tags}".strip()


def generate(niche, count=None, workdir=None, max_rounds=2):
    """Decide on one CivitAI checkpoint + reference prompt, generate `count` camera
    variations of it, keep what passes supervisor.py review, and repeat with a fresh
    batch of variations if a round still leaves the set short of min_images.

    Raises once max_rounds is exhausted without reaching min_images -- an image-only
    post with too few photos is not worth publishing, and a silent quality drop should
    never ship. Not an infinite retry: max_rounds caps the worst case at
    max_rounds * count generations, each ~130s on a GH Actions runner."""
    import tempfile
    from pathlib import Path

    count = count or int(niche.get("images_per_video", 10))
    min_images = int(niche.get("min_images", 3))
    max_images = int(niche.get("max_images", niche.get("images_per_video", 5)))
    workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="imageslides_"))
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        resolved, reference = decide_reference(niche)
    except Exception as e:
        log(f"CivitAI discovery unavailable ({type(e).__name__}: {str(e)[:150]}); "
            "falling back to the built-in prompt formula")
        resolved, reference = _formula_reference(niche)

    civitai_spec = f"{resolved['model_id']}:{resolved['version_id']}" if resolved else None
    # An explicit outfit is only injected when the reference prompt does not already
    # name one -- appending "wearing jeans and a coat" onto a prompt that already says
    # "wearing a black dress" gives the model two contradictory outfits at once. Mood is
    # added unconditionally instead -- unlike an outfit, a pose/expression cue does not
    # conflict with whatever the reference prompt already says, and this is what keeps
    # the tone consistent even when the decided reference is mundane (a live search
    # once decided on a showcase prompt describing a chef in a kitchen). prefix (not
    # reference["prompt"]) is what build_variations puts before the camera modifier --
    # see its docstring for why the harvested text goes last, not this.
    outfit = random.choice(niche.get("outfits") or DEFAULT_OUTFITS)
    clothing = "" if _CLOTHING_RE.search(reference["prompt"]) else f"{outfit}, "
    mood = random.choice(niche.get("mood") or DEFAULT_MOOD)
    prefix = f"{SAFETY_PREFIX}, {clothing}{mood}"
    base_negative = ", ".join(
        x for x in (NEGATIVE_HARD, reference["negative_prompt"], NEGATIVE_QUALITY) if x)
    log(f"prefix: {prefix} | reference: {reference['prompt'][:120]}")

    supervisor_on = os.environ.get("SUPERVISOR_ENABLED", "1").strip().lower() not in (
        "0", "false", "no")
    approved, generated_count = [], 0
    for round_num in range(1, max_rounds + 1):
        prompts, negatives = build_variations(
            prefix, reference["prompt"], base_negative, count, niche)
        generated = sdgen.generate_batch(
            prompts, workdir / f"round{round_num}", model_key=niche.get("sd_model"),
            negative_prompts=negatives, civitai_model=civitai_spec)
        generated_count += len(generated)
        newly_approved = supervisor.filter_images(generated) if supervisor_on else generated
        if not supervisor_on:
            log("SUPERVISOR_ENABLED=0: skipping vision QA")
        approved.extend(newly_approved)
        log(f"round {round_num}/{max_rounds}: {len(newly_approved)}/{len(generated)} passed, "
            f"{len(approved)}/{min_images} needed")
        if len(approved) >= min_images:
            break

    if len(approved) < min_images:
        raise RuntimeError(
            f"only {len(approved)} of {generated_count} images passed review across "
            f"{max_rounds} round(s) (need at least {min_images}); not posting")
    log(f"{len(approved)}/{generated_count} images approved")
    return approved[:max_images]
