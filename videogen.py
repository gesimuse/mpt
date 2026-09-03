"""Image-to-video on Hugging Face ZeroGPU Spaces, called directly.

Used to shell out to a Kaggle kernel that just proxied to the same Space -- Kaggle
never did any compute itself, it was pure push/poll/download plumbing kept "for
continuity with the existing pipeline" (motionforge's own words). Calling gradio_client
directly removes all of that for the same result.

Kaggle was later tried again for video, this time doing real compute (LTX-Video 0.9.5
on a T4) as a fallback for when ZeroGPU quota runs out. It worked end to end -- the
output quality did not, and the account owner rejected it outright. Removed. If anyone
revisits this: the engineering is not the hard part (a working kernel took two
attempts; the sizing notes are in git history around a1041a4), the model is. Nothing
that fits a 15 GB T4 came close to Wan 2.2 here.

Two independent axes of failure, and this module walks both:

  * TOKENS. ZeroGPU's free quota is ~5 min/day per HF ACCOUNT. HF_TOKENS
    (comma-separated) rotates to the next account when one is spent. Tokens from the
    SAME account share one pool, so rotating those does nothing.
  * SPACES. Any single Space can be down, restarting, rate-limited, or have changed its
    signature. SPACES below is an ordered ladder; a Space-side failure moves to the next
    one rather than failing the run.

Quota detection is by MESSAGE, not exception class: HF's real text is
"You have exceeded your GPU quota (60s requested vs. 42s left)". The previous check
looked for "zerogpu quota" / "exceeded your free", neither of which appears in that
string -- so the first token's quota error was re-raised immediately and every extra
token in HF_TOKENS was never tried. That is the bug behind "it uses one token then
stops".

Needs env: HF_TOKEN, or HF_TOKENS (comma-separated) for more than one account.
Optional: VIDEO_SPACES, a comma-separated list of space ids to override the ladder.
"""
import os, subprocess, sys, tempfile, urllib.request
from pathlib import Path


def log(msg): print(f"[videogen] {msg}", flush=True)


# Portrait, and a multiple of 32 -- both Wan and LTX round to 32, and TikTok wants
# vertical. Only the Spaces that expose explicit height/width use these.
_H, _W = 832, 480

# Applied to every generation on every Space unless a caller passes its own.
#
# Two jobs. The first mirrors imageslides.NEGATIVE_HARD: the video model never sees
# that, and motion_writer's rubric now deliberately asks for seductive movement, so
# the one thing standing between "suggestive" and a post TikTok removes is this
# string. A prompt about arching backs and hands on hips is exactly the input that
# makes an I2V model drift toward undressing on its own, without ever being asked to.
#
# The second is ordinary I2V quality: these models are far more prone to warping a
# face or melting a hand across frames than a still generator is, because nothing
# enforces temporal consistency on the parts of the frame that move most.
VIDEO_NEGATIVE = (
    "nude, topless, undressing, removing clothing, exposed nipples, exposed genitals, "
    "explicit sexual content, child, teen, minor, "
    "distorted face, warped face, deformed hands, extra fingers, extra limbs, "
    "morphing anatomy, flickering, jittery motion, blurry, watermark, text, logo")


def _clamp(value, lo, hi):
    """Pin a slider argument inside the range the Space actually accepts.

    Out-of-range values do not come back as a useful error. A Slider outside its
    bounds surfaces as a bare `AppError: RuntimeError` with no detail at all, which
    reads exactly like the Space being broken -- and since a non-quota failure moves
    the ladder on, one stale bound silently retires a whole rung. Clamping per Space
    means the caller keeps asking for what it wants and each Space takes what it can."""
    return max(lo, min(hi, value))


def _wan22_rcm_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    from gradio_client import handle_file
    return dict(
        input_image=handle_file(str(image_path)), prompt=prompt,
        steps=_clamp(steps, 1, 30),
        negative_prompt=negative_prompt or "",
        duration_seconds=_clamp(length_s, 0.5, 5.0),
        guidance_scale=1.0, guidance_scale_2=1.0,
        seed=seed if seed is not None else 0, randomize_seed=seed is None)


def _upsampler_wan22_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    """Upsampler/wan-2-2-14b-image-to-video -- same Wan 2.2 14B weights as the rCM
    Space above, so its output is interchangeable here, and it asks ZeroGPU for less
    time for the same request (150s where rCM asks 195s at 5s/8 steps). That gap is
    the whole reason it is on the ladder: a token with 160s left can serve this and
    not rCM."""
    from gradio_client import handle_file
    return dict(
        input_image=handle_file(str(image_path)), prompt=prompt,
        steps=_clamp(steps, 1, 12),
        negative_prompt=negative_prompt or "",
        duration_seconds=_clamp(length_s, 0.5, 5.0),
        guidance_scale=1.0, guidance_scale_2=1.0,
        seed=seed if seed is not None else 0, randomize_seed=seed is None,
        end_image=None)


def _wan22_fast_b64_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    """prithivMLmods/Wan2.2-Fast. Takes the image as a base64 STRING, not a file
    handle -- the only Space here that does. Its parameters carry no declared ranges
    (they are plain API inputs, not Sliders), so nothing is clamped."""
    import base64
    return dict(
        image_b64=base64.b64encode(Path(image_path).read_bytes()).decode(),
        prompt=prompt, steps=steps, negative_prompt=negative_prompt or "",
        duration_seconds=length_s, guidance_scale=1.0, guidance_scale_2=1.0,
        seed=seed if seed is not None else 0, randomize_seed=seed is None)


def _wan21_fast_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    from gradio_client import handle_file
    return dict(
        input_image=handle_file(str(image_path)), prompt=prompt,
        height=_clamp(_H, 128, 896), width=_clamp(_W, 128, 896),
        negative_prompt=negative_prompt or "",
        duration_seconds=_clamp(length_s, 0.3, 3.4),
        guidance_scale=1.0, steps=_clamp(steps, 1, 30),
        seed=seed if seed is not None else 0, randomize_seed=seed is None)


def _ltx_distilled_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    from gradio_client import handle_file
    # LTX's own endpoint takes no step count -- it's a distilled model with a fixed
    # schedule, so `steps` is deliberately dropped here rather than mapped onto
    # ui_guidance_scale or anything else it doesn't mean.
    return dict(
        prompt=prompt, negative_prompt=negative_prompt or "",
        input_image_filepath=handle_file(str(image_path)), input_video_filepath="",
        height_ui=768, width_ui=512, mode="image-to-video", duration_ui=length_s,
        ui_frames_to_use=9, seed_ui=seed if seed is not None else 0,
        randomize_seed=seed is None, ui_guidance_scale=1.0, improve_texture_flag=True)


# Ordered ladder. Signatures below were read off each Space's own view_api(), not
# guessed -- three separate live incidents (commits 3228192, 2ceda0a, ed788ec) came
# from assuming an argument name or type here.
# LTX is deliberately absent. It was tried twice -- self-hosted 2B on a Kaggle T4 and
# the hosted 13B Space -- and rejected both times by the account owner: its motion does
# not read as a real person moving, which is the entire product here. Keeping it as a
# fallback would mean a quota-exhausted run silently shipping output that would never
# be posted, which is worse than the run failing. Its adapter is kept below so
# VIDEO_SPACES can still name it for a one-off experiment.
#
# Ordered cheapest-quality-loss first, and the ZeroGPU seconds each one asks for at
# 5s/8 steps are noted because that number, not the daily allowance, is what decides
# whether a run happens: the allowance is ~300s per account per day, so a 195s Space
# fits once and a 150s Space fits twice.
#
# multimodalart/wan2-1-fast used to sit on this ladder as the fallback. It is dead:
# on 2026-09-03 it returned a bare `AppError: RuntimeError` for every token AND for
# an anonymous call using the Space's own default arguments, which rules out both
# quota and our arguments. It is kept as an adapter below so VIDEO_SPACES can pick it
# up again if its owner fixes it, but a rung that always fails only wastes a round
# trip on every quota-exhausted run.
SPACES = [
    ("linoyts/wan2-2-i2v-rCM", "/generate_video", _wan22_rcm_args),            # 195s
    ("Upsampler/wan-2-2-14b-image-to-video", "/generate_video",
     _upsampler_wan22_args),                                                   # 150s
    ("prithivMLmods/Wan2.2-Fast", "/generate_video", _wan22_fast_b64_args),    # 120s
]
# Reachable only by naming it explicitly in VIDEO_SPACES.
OPTIONAL_SPACES = [
    ("Lightricks/ltx-video-distilled", "/image_to_video", _ltx_distilled_args),
    ("multimodalart/wan2-1-fast", "/generate_video", _wan21_fast_args),
]


def _spaces():
    """The ladder, or just the space ids named in VIDEO_SPACES (in that order). An id
    with no adapter here is skipped loudly rather than called with the wrong argument
    names -- there is no generic I2V signature to fall back on."""
    raw = os.environ.get("VIDEO_SPACES", "").strip()
    if not raw:
        return SPACES
    wanted = [s.strip() for s in raw.split(",") if s.strip()]
    by_id = {entry[0]: entry for entry in SPACES + OPTIONAL_SPACES}
    out = []
    for sid in wanted:
        if sid in by_id:
            out.append(by_id[sid])
        else:
            log(f"VIDEO_SPACES names {sid!r}, which has no argument adapter here; "
                "skipping it")
    return out


def _ensure_gradio_client():
    """Install gradio_client only if it is actually missing.

    It used to install unconditionally, which is harmless on a fresh CI runner and
    fatal anywhere else: on a PEP 668 "externally managed" system python, pip refuses
    outright and check=True turned that into a crash before a single Space was called.
    The import check makes running this locally -- to test a Space by hand, which is
    the only way to judge one -- possible at all, and skips a redundant install in CI."""
    try:
        import gradio_client  # noqa: F401
        return
    except ImportError:
        pass
    log("gradio_client missing, installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio_client"],
                   check=True)


def _fetch_image(image_url, tmp_dir):
    """Download image_url to a local file and return its path. handle_file()
    claims to accept a bare URL directly, but a live run passing the URL
    straight through got an opaque AppError back from the Space (its own
    exception, no detail exposed) failing almost instantly -- consistent
    with the image input never actually loading. Downloading it ourselves
    first (what the original Kaggle-kernel design always did) sidesteps
    whatever that was."""
    dest = Path(tmp_dir) / "input.jpg"
    urllib.request.urlretrieve(image_url, dest)
    log(f"image: {image_url[:80]} -> {dest} ({dest.stat().st_size // 1024}KB)")
    return dest


def _tokens():
    """Every distinct token we have, HF_TOKENS and HF_TOKEN combined.

    This used to return HF_TOKENS *instead of* HF_TOKEN, so a setup with a token in
    each -- three accounts' worth -- only ever tried two of them, and the third
    account's untouched daily quota sat there while the run failed. Caught on a real
    run whose log said "token 1/2" against a .env holding three distinct tokens.

    Order matters: HF_TOKENS first, since that's the list someone curates for exactly
    this, with HF_TOKEN appended as one more account rather than a replacement. De-
    duplicated because the same token appearing in both is the obvious way to write
    it, and retrying an already-spent quota is a wasted round trip."""
    raw = os.environ.get("HF_TOKENS", "").strip()
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    single = os.environ.get("HF_TOKEN", "").strip()
    if single:
        toks.append(single)
    # dict.fromkeys de-duplicates while preserving order.
    toks = list(dict.fromkeys(toks))
    return toks or [""]  # "" = anonymous call


def _is_quota_error(e: Exception) -> bool:
    """True for HF's ZeroGPU quota-exceeded error -- the one failure mode where trying
    the NEXT TOKEN can help (quota is per account). Any other error would fail the same
    way on every token, so it advances the SPACE instead.

    Matched on the message, deliberately: HF's real wording is "You have exceeded your
    GPU quota (60s requested vs. 42s left)", sometimes wrapped as "The upstream Gradio
    app has raised an exception: ...". The old check looked for "zerogpu quota" and
    "exceeded your free", neither of which occurs in that string, so rotation never
    fired even once."""
    msg = str(e).lower()
    if "gpu quota" in msg:
        return True
    return "exceeded" in msg and "quota" in msg


def _call_space(space_id, api_name, arg_fn, image_path, prompt, negative_prompt,
                length_s, steps, seed, token):
    from gradio_client import Client
    client = Client(space_id, token=token or None)
    log(f"calling {space_id}{api_name} (seconds={length_s}, steps={steps}, "
        f"seed={seed}, token={'set' if token else 'anonymous'})")
    kwargs = arg_fn(image_path, prompt, negative_prompt, length_s, steps, seed)
    return client.predict(api_name=api_name, **kwargs)


def _video_path_from(result):
    """Every Space here returns (video, seed) in some shape -- a path string, a dict
    with a "video"/"path" key, or a tuple of either."""
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        return first if isinstance(first, str) else (first or {}).get("video")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return result.get("video") or result.get("path")
    return None


# How far a quota-blocked run is allowed to shrink, as (seconds, steps) multipliers
# of what the caller asked for. A ZeroGPU Space reserves GPU time in proportion to
# the work requested -- measured live on linoyts/wan2-2-i2v-rCM, which asks for 195s
# at 5.0s/8 steps and ran on an account with 78s left once asked for 3.0s/4 steps.
# The free allowance is ~300s per account per day, so this is the difference between
# one video a day per account and three or four.
#
# Full size is always tried first and the ladder only shrinks after every Space x
# token combination has come back quota-blocked, so a run with quota to spare is
# unaffected. The floor stops well short of unusable: below ~3s there is not enough
# motion for a post, and a clip too short to publish is no better than no clip.
_QUOTA_RUNGS = [(1.0, 1.0), (0.8, 0.625), (0.6, 0.5)]
_MIN_SECONDS, _MIN_STEPS = 3.0, 4


def _quota_rungs(length_s, steps):
    """Successively cheaper (seconds, steps) pairs, de-duplicated and floored."""
    out = []
    for s_mul, st_mul in _QUOTA_RUNGS:
        rung = (max(_MIN_SECONDS, round(length_s * s_mul, 1)),
                max(_MIN_STEPS, int(steps * st_mul)))
        if rung not in out:
            out.append(rung)
    return out


def _walk_ladder(spaces, tokens, image_path, prompt, negative_prompt, length_s,
                 steps, seed):
    """One full pass over every Space x token at one size.

    Returns (video_path, last_error, every_failure_was_quota). That last flag is what
    tells the caller whether retrying at a smaller size could possibly help."""
    video_path, last_err, all_quota = None, None, True
    for s_i, (space_id, api_name, arg_fn) in enumerate(spaces):
        for t_i, token in enumerate(tokens):
            try:
                result = _call_space(space_id, api_name, arg_fn, image_path, prompt,
                                     negative_prompt, length_s, steps, seed, token)
                video_path = _video_path_from(result)
                if not video_path:
                    # A 200 that carries no video is a Space failure like any other --
                    # raised here so it lands in the same handler and the ladder can
                    # move on, instead of being mistaken for a successful call.
                    raise RuntimeError(f"unexpected Space return: {result!r}")
                break
            except Exception as e:
                last_err = e
                video_path = None
                if _is_quota_error(e):
                    if t_i < len(tokens) - 1:
                        log(f"token {t_i + 1}/{len(tokens)} is out of ZeroGPU quota, "
                            f"trying the next account ({str(e)[:150]})")
                        continue
                    log(f"every one of the {len(tokens)} token(s) is out of ZeroGPU "
                        f"quota on {space_id}")
                    break
                # Not a quota problem: the same call fails identically on every token,
                # so move to the next Space rather than burning the token list.
                all_quota = False
                log(f"{space_id} failed ({type(e).__name__}: {str(e)[:150]})")
                break
        if video_path:
            return video_path, last_err, all_quota
        if s_i < len(spaces) - 1:
            log(f"falling back to {spaces[s_i + 1][0]}")
    return None, last_err, all_quota


def generate(image_url, prompt, length_s=5.0, steps=4, seed=None,
             negative_prompt=None, out_dir=None):
    """Walk the Space ladder, rotating tokens within each Space on a ZeroGPU quota
    error, and return the local mp4 path (re-encoded for TikTok). Raises only once
    every Space x token combination is exhausted -- the caller treats that as a
    skipped run."""
    # autopilot.py's _run_video_niche passes these through straight from env
    # vars (VIDEO_STEPS/VIDEO_LENGTH_S), so they arrive as strings -- the
    # Space's steps/duration_seconds are both Slider (float) components, and
    # a live run crashed there with an opaque, unhelpful "the upstream Gradio
    # app has raised an exception" every time. The old Kaggle-kernel design
    # never hit this because its template substitution did int("__STEPS__")/
    # float("__LENGTH_S__") before ever reaching predict().
    length_s, steps = float(length_s), int(steps)
    negative_prompt = VIDEO_NEGATIVE if negative_prompt is None else negative_prompt
    _ensure_gradio_client()

    work_dir = tempfile.mkdtemp(prefix="videogen_in_")
    image_path = _fetch_image(image_url, work_dir)

    spaces, tokens = _spaces(), _tokens()
    if not spaces:
        raise RuntimeError("no usable video Space configured (check VIDEO_SPACES)")

    video_path, last_err, all_quota = None, None, False
    for r_i, (rung_s, rung_steps) in enumerate(_quota_rungs(length_s, steps)):
        if r_i:
            log(f"whole ladder is quota-blocked at {length_s}s/{steps} steps; "
                f"retrying smaller ({rung_s}s, {rung_steps} steps) -- a Space asks "
                "ZeroGPU for less time when asked for less video")
        video_path, last_err, all_quota = _walk_ladder(
            spaces, tokens, image_path, prompt, negative_prompt, rung_s, rung_steps,
            seed)
        # Only a pure quota wall is worth retrying smaller. If any Space failed for
        # its own reasons, a shorter clip would fail there the same way.
        if video_path or not all_quota:
            break
    if not video_path:
        raise RuntimeError(
            f"every video Space failed ({len(spaces)} space(s) x {len(tokens)} "
            f"token(s)); last error: {type(last_err).__name__}: {str(last_err)[:250]}"
        ) from last_err

    out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="videogen_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "final.mp4"
    _normalize_for_tiktok(Path(video_path), dest)
    log(f"wrote {dest} ({dest.stat().st_size // 1024}KB)")
    return str(dest)


# TikTok's Content Posting API rejected a real draft with fail_reason=
# frame_rate_check_failed -- confirmed live, publish_id came back accepted but
# check_publish_status later showed FAILED. Wan 2.2 I2V rCM's raw output frame rate
# isn't in TikTok's accepted range; the Space's own output was never meant to be
# TikTok-ready as-is, it's a generic video export. Re-encode to a safe, common
# frame rate here rather than pushing the raw output straight through. Applies to
# every Space on the ladder, not just the first -- none of them target TikTok.
_TIKTOK_FPS = 30


def _normalize_for_tiktok(src, dest):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-r", str(_TIKTOK_FPS),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(dest)],
            check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found -- can't normalize frame rate for TikTok")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg frame-rate normalize failed: {e.stderr[-500:]}")
