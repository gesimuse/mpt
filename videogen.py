"""Image-to-video on Hugging Face ZeroGPU Spaces, called directly.

Used to shell out to a Kaggle kernel that just proxied to the same Space -- Kaggle
never did any compute itself, it was pure push/poll/download plumbing kept "for
continuity with the existing pipeline" (motionforge's own words). Calling gradio_client
directly removes all of that for the same result. (Kaggle IS back in the picture for
video, but as a real self-hosted GPU fallback in kaggle_videogen.py, not as a proxy --
and Kaggle's API can in fact request a specific accelerator, `kaggle kernels push
--accelerator NvidiaTeslaT4`, contrary to what this docstring used to claim.)

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
import os, subprocess, tempfile, urllib.request
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


def _wan22_rcm_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    from gradio_client import handle_file
    return dict(
        input_image=handle_file(str(image_path)), prompt=prompt, steps=steps,
        negative_prompt=negative_prompt or "", duration_seconds=length_s,
        guidance_scale=1.0, guidance_scale_2=1.0,
        seed=seed if seed is not None else 0, randomize_seed=seed is None)


def _wan21_fast_args(image_path, prompt, negative_prompt, length_s, steps, seed):
    from gradio_client import handle_file
    return dict(
        input_image=handle_file(str(image_path)), prompt=prompt, height=_H, width=_W,
        negative_prompt=negative_prompt or "", duration_seconds=length_s,
        guidance_scale=1.0, steps=steps,
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
SPACES = [
    ("linoyts/wan2-2-i2v-rCM", "/generate_video", _wan22_rcm_args),
    ("multimodalart/wan2-1-fast", "/generate_video", _wan21_fast_args),
    ("Lightricks/ltx-video-distilled", "/image_to_video", _ltx_distilled_args),
]


def _spaces():
    """The ladder, or just the space ids named in VIDEO_SPACES (in that order). An id
    with no adapter here is skipped loudly rather than called with the wrong argument
    names -- there is no generic I2V signature to fall back on."""
    raw = os.environ.get("VIDEO_SPACES", "").strip()
    if not raw:
        return SPACES
    wanted = [s.strip() for s in raw.split(",") if s.strip()]
    by_id = {sid: entry for entry in SPACES for sid in (entry[0],)}
    out = []
    for sid in wanted:
        if sid in by_id:
            out.append(by_id[sid])
        else:
            log(f"VIDEO_SPACES names {sid!r}, which has no argument adapter here; "
                "skipping it")
    return out


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


def generate(image_url, prompt, length_s=5.0, steps=4, seed=None,
             negative_prompt=None, out_dir=None):
    """Walk the Space ladder, rotating tokens within each Space on a ZeroGPU quota
    error, and return the local mp4 path (re-encoded for TikTok). Raises only once
    every Space x token combination is exhausted -- the caller treats that as a
    skipped run (autopilot.py then tries kaggle_videogen, if configured)."""
    # autopilot.py's _run_video_niche passes these through straight from env
    # vars (VIDEO_STEPS/VIDEO_LENGTH_S), so they arrive as strings -- the
    # Space's steps/duration_seconds are both Slider (float) components, and
    # a live run crashed there with an opaque, unhelpful "the upstream Gradio
    # app has raised an exception" every time. The old Kaggle-kernel design
    # never hit this because its template substitution did int("__STEPS__")/
    # float("__LENGTH_S__") before ever reaching predict().
    length_s, steps = float(length_s), int(steps)
    negative_prompt = VIDEO_NEGATIVE if negative_prompt is None else negative_prompt
    subprocess.run(["pip", "install", "-q", "gradio_client"], check=True)

    work_dir = tempfile.mkdtemp(prefix="videogen_in_")
    image_path = _fetch_image(image_url, work_dir)

    spaces, tokens = _spaces(), _tokens()
    if not spaces:
        raise RuntimeError("no usable video Space configured (check VIDEO_SPACES)")
    video_path, last_err = None, None
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
                log(f"{space_id} failed ({type(e).__name__}: {str(e)[:150]})")
                break
        if video_path:
            break
        if s_i < len(spaces) - 1:
            log(f"falling back to {spaces[s_i + 1][0]}")
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


def normalize_for_tiktok(src, dest):
    """Public alias -- kaggle_videogen.py's self-hosted path needs the exact same
    re-encode, and duplicating the ffmpeg invocation is how the two would drift."""
    return _normalize_for_tiktok(Path(src), Path(dest))


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
