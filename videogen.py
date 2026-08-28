"""Wan 2.2 I2V rCM via the linoyts/wan2-2-i2v-rCM HF ZeroGPU Space, called directly.

Used to shell out to a Kaggle kernel that just proxied to this same Space --
Kaggle never did any compute itself, it was pure push/poll/download plumbing
kept "for continuity with the existing pipeline" (motionforge's own words).
That plumbing turned out to be a real liability with no offsetting benefit:
Kaggle's kernel API has no way to request a specific accelerator, kernel
push/poll adds real latency, and it needed its own account, secrets, and a
sibling repo checkout. Calling gradio_client directly here removes all of
that for the exact same result -- motionforge's own README already said as
much: "You could rip Kaggle out and call the Space directly from GH Actions
in ~30 lines; the kernel is a thin proxy."

Needs env: HF_TOKEN, or HF_TOKENS (comma-separated) to rotate across more
than one HF account's free ZeroGPU quota (5min/day per account -- tokens
from the SAME account share one pool, rotating those does nothing)."""
import os, subprocess
from pathlib import Path


def log(msg): print(f"[videogen] {msg}", flush=True)

SPACE_ID = "linoyts/wan2-2-i2v-rCM"


def _tokens():
    raw = os.environ.get("HF_TOKENS", "").strip()
    if raw:
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if toks:
            return toks
    single = os.environ.get("HF_TOKEN", "").strip()
    return [single] if single else [""]  # "" = anonymous call


def _is_quota_error(e: Exception) -> bool:
    """True for HF's "exceeded your free ZeroGPU quota" AppError specifically --
    the one failure mode where trying the next token can actually help. Any
    other error (bad prompt, Space down, network blip) would fail identically
    on every token, so it's raised immediately instead of burning the rest of
    the list."""
    msg = str(e).lower()
    return "zerogpu quota" in msg or "exceeded your free" in msg


def _call_space(image_url, prompt, negative_prompt, length_s, steps, seed, token):
    from gradio_client import Client, handle_file
    client = Client(SPACE_ID, token=token or None)
    log(f"calling Space (seconds={length_s}, steps={steps}, seed={seed}, "
        f"token={'set' if token else 'anonymous'})")
    return client.predict(
        input_image=handle_file(image_url),  # gradio_client fetches URLs directly
        prompt=prompt,
        steps=steps,
        negative_prompt=negative_prompt or "",
        duration_seconds=length_s,
        guidance_scale=1.0,
        guidance_scale_2=1.0,
        seed=seed if seed is not None else 0,
        randomize_seed=seed is None,
        api_name="/generate_video",
    )


def generate(image_url, prompt, length_s=5.0, steps=4, seed=None,
             negative_prompt=None, out_dir=None):
    """Call the HF Space directly, rotating across HF_TOKENS on a ZeroGPU
    quota-exceeded error, and return the local mp4 path (re-encoded for
    TikTok). Raises if every token is exhausted or the Space itself fails --
    caller treats it as a skipped run."""
    from gradio_client.exceptions import AppError
    subprocess.run(["pip", "install", "-q", "gradio_client"], check=True)

    tokens = _tokens()
    result, last_err, succeeded = None, None, False
    for i, token in enumerate(tokens):
        try:
            result = _call_space(image_url, prompt, negative_prompt, length_s,
                                 steps, seed, token)
            succeeded = True
            break
        except AppError as e:
            last_err = e
            if i < len(tokens) - 1 and _is_quota_error(e):
                log(f"token {i + 1}/{len(tokens)} hit its ZeroGPU quota, "
                    f"trying the next one ({str(e)[:150]})")
                continue
            raise
    if not succeeded:
        raise last_err

    # Space returns (video_path, seed).
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        video_path = first if isinstance(first, str) else first.get("video")
    elif isinstance(result, str):
        video_path = result
    elif isinstance(result, dict):
        video_path = result.get("video") or result.get("path")
    else:
        video_path = None
    if not video_path:
        raise RuntimeError(f"unexpected Space return: {result!r}")

    import tempfile
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
# frame rate here rather than pushing the raw output straight through.
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
