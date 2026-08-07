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

Delivery: TikTok's Photo Mode carousel, queued as a native inbox draft (tiktok.py) --
no caption travels with it, by design: the account owner opens the app, adds trending
sound, pastes the caption from CAPTIONS.md, and posts by hand. Not yet verified against
a real TikTok account; see tiktok.py's module docstring.
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
    "wearing a wet white t-shirt clinging to her figure, poolside",
    "wearing a tight, clingy bodycon dress that hugs every curve",
    "wearing a sheer, see-through lace dress over a bikini",
    "wearing a soaking wet, transparent tank top and denim shorts",
    "wearing a skin-tight, wet swimsuit fresh out of the pool",
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
    "wet skin, dripping water, glistening body",
    "tight clingy fabric, curves accentuated, sultry pose",
]

# Hard line: no exposed nipples/genitals, no real nudity, no minors. Everything else
# (swimwear, lingerie, loungewear) is within policy and is not filtered here.
SAFETY_PREFIX = "adult woman in her late twenties"
# Every image must read as sexy, not just "sometimes, when the random mood pick lands
# right" -- this is the guaranteed baseline, always present; DEFAULT_MOOD on top of it
# is what varies the specific pose/expression between images in the same batch.
SEXY_CUE = "seductive, sexy, alluring"
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


def decide_reference(niche):
    """Search CivitAI for a checkpoint with a real, on-subject showcase prompt, and
    commit to it for this whole run: one model, one reference photo, every slide in
    the batch is a variation of it, not a grab bag of unrelated faces.

    The search query itself is randomised per run too (civitai_queries, a list; a
    single civitai_query string still works as a one-item list) -- civitai.decide_
    reference() already randomises which qualifying model+prompt a fixed query
    resolves to, but a fixed query alone still biases toward whatever CivitAI ranks
    highest for it. Rotating the query is what actually varies the THEME run to run,
    not just the exact photo within one theme.

    civitai.py's search_candidates()/decide_reference() stay subject-agnostic; the
    niche's gender/portrait rules are passed in as a filter rather than living there."""
    queries = niche.get("civitai_queries") or (
        [niche["civitai_query"]] if niche.get("civitai_query") else DEFAULT_CIVITAI_QUERIES)
    query = random.choice(queries)
    resolved, reference = civitai.decide_reference(
        query, prompt_filter=lambda p: _matches_subject(p, niche))
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


def _build_prefix(niche, reference):
    # An explicit outfit is only injected when the reference prompt does not already
    # name one -- appending "wearing jeans and a coat" onto a prompt that already says
    # "wearing a black dress" gives the model two contradictory outfits at once.
    # SEXY_CUE and mood are both added unconditionally -- unlike an outfit, a pose/
    # expression cue does not conflict with whatever the reference prompt already
    # says. SEXY_CUE is the guaranteed baseline ("every image must be sexy" is a hard
    # requirement, not a random pick); mood adds per-image variety on top of it.
    outfit = random.choice(niche.get("outfits") or DEFAULT_OUTFITS)
    clothing = "" if _CLOTHING_RE.search(reference["prompt"]) else f"{outfit}, "
    mood = random.choice(niche.get("mood") or DEFAULT_MOOD)
    return f"{SAFETY_PREFIX}, {clothing}{SEXY_CUE}, {mood}"


def generate(niche, count=None, workdir=None, max_rounds=2):
    """Decide on a CivitAI checkpoint + reference prompt, generate `count` camera
    variations of it, keep what passes supervisor.py review, and repeat with a fresh
    round if the batch still falls short of min_images.

    A round that produces ZERO images (not "some generated but failed QA" -- none
    generated at all) means the checkpoint itself is broken for this run, not that the
    prompts were bad -- a live run hit this exactly: a gated CivitAI model 401'd on
    every single download attempt, and since the old code decided on one checkpoint
    once and reused it for every round, all rounds failed identically. A round like
    that makes the NEXT round re-decide (fresh search, likely a different checkpoint)
    instead of retrying the same broken one.

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

    supervisor_on = os.environ.get("SUPERVISOR_ENABLED", "1").strip().lower() not in (
        "0", "false", "no")
    approved, generated_count = [], 0
    # No static-formula fallback: every batch's model and prompt must come from a real
    # CivitAI search, not a hand-written scene/style list. If CivitAI can't be reached
    # at all, decide_reference() raises and the run fails loudly here, rather than
    # silently shipping generic images.
    resolved, reference = decide_reference(niche)

    for round_num in range(1, max_rounds + 1):
        civitai_spec = f"{resolved['model_id']}:{resolved['version_id']}"
        prefix = _build_prefix(niche, reference)
        base_negative = ", ".join(
            x for x in (NEGATIVE_HARD, reference["negative_prompt"], NEGATIVE_QUALITY) if x)
        log(f"round {round_num}: {resolved['name']!r} | prefix: {prefix} | "
            f"reference: {reference['prompt'][:120]}")

        prompts, negatives = build_variations(
            prefix, reference["prompt"], base_negative, count, niche)
        try:
            # civitai_model always wins over model_key in sdgen (see its docstring),
            # and every batch now always has one -- no model_key to pass here at all.
            generated = sdgen.generate_batch(
                prompts, workdir / f"round{round_num}",
                negative_prompts=negatives, civitai_model=civitai_spec)
        except RuntimeError as e:
            generated = []
            log(f"round {round_num}: checkpoint unusable ({str(e)[:150]})")

        generated_count += len(generated)
        newly_approved = supervisor.filter_images(generated) if supervisor_on else generated
        if not supervisor_on:
            log("SUPERVISOR_ENABLED=0: skipping vision QA")
        approved.extend(newly_approved)
        log(f"round {round_num}/{max_rounds}: {len(newly_approved)}/{len(generated)} passed, "
            f"{len(approved)}/{min_images} needed")
        if len(approved) >= min_images:
            break
        if not generated and round_num < max_rounds:
            log("nothing generated this round; re-deciding a fresh checkpoint+prompt")
            resolved, reference = decide_reference(niche)

    if len(approved) < min_images:
        raise RuntimeError(
            f"only {len(approved)} of {generated_count} images passed review across "
            f"{max_rounds} round(s) (need at least {min_images}); not posting")
    log(f"{len(approved)}/{generated_count} images approved")
    return approved[:max_images]
