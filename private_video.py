"""Animate one hand-sent Telegram photo on Kaggle and hand the mp4 back to Telegram,
without any copy of either living anywhere else.

The ordinary path (autopilot._run_video_niche) is built around publishing: the still
is on the public gh-pages branch because TikTok's PULL_FROM_URL has to fetch it, the
mp4 goes there too, both are recorded in posted.json, and the clip is queued to
TikTok as an inbox draft. That is correct for the account's own content and wrong for
a photo the owner drops into the channel by hand, which is the case this module
exists for. So this is a SEPARATE entry point rather than a flag on that one: the
publishing code is not reached at all, instead of being reached and asked not to act.

The flow, end to end:

    Telegram photo (+ prompt as its caption, or as a reply to it)
      -> the Worker dispatches private_video.yml with the photo's file_id
      -> this script fetches the bytes with the bot token, in the runner's own memory
      -> the still travels INSIDE a private Kaggle kernel's source (image_b64)
      -> the mp4 comes back to the runner
      -> uploaded straight into the videos channel as raw bytes
      -> local files deleted, the Kaggle kernel overwritten with a no-op

What is guaranteed: nothing is committed to any branch, nothing is hosted on
gh-pages, posted.json and CAPTIONS.md are untouched, and TikTok is never called.

What is NOT, stated plainly rather than left to be discovered:

  * Kaggle receives the image and the video. It is a private notebook, and
    kaggle_videogen.scrub_kernel() overwrites the source and output afterwards, but
    Kaggle keeps the kernel's version history and has no delete in its API -- that
    is a manual action on Kaggle's own site. This is inherent to "generate it on
    someone else's GPU"; only a local run avoids it.
  * The GitHub Actions run itself is public on this repo, so the workflow INPUTS are
    visible in the run's UI: the prompt text, and the file_id. A file_id is useless
    without the bot token (a secret), but it is a pointer to the image. Nothing else
    about the run is public: this script prints no prompt text, no image, no video,
    and no kernel log, and the workflow uploads no artifacts.

Env:
  TELEGRAM_FILE_ID        the photo, as Telegram's own file id
  MOTION_PROMPT           what it should do
  TELEGRAM_BOT_TOKEN      fetches the photo, sends the video
  TELEGRAM_VIDEO_CHAT_ID  where the video lands (falls back to TELEGRAM_CHAT_ID)
  TELEGRAM_CHAT_ID        where failures are reported (the channel it came from)
  KAGGLE_USERNAME, KAGGLE_API_TOKEN
  VIDEO_LENGTH_FRAMES     default 81 (~5s at Wan2GP's 16fps)
  VIDEO_STEPS             default 8
  VIDEO_RESOLUTION        default 512x896
"""
import os
import sys
import tempfile
from pathlib import Path

import requests

import kaggle_videogen
import motion_writer
import telegram

API = "https://api.telegram.org"


def log(msg):
    # Through telegram.redact: this lands in a PUBLIC Actions log, and the bot token
    # is a path segment of every URL used here, so any error string can carry it.
    print(f"[private_video] {telegram.redact(msg)}", flush=True)


def _fetch_photo(file_id, dest_dir):
    """Telegram's copy of the photo, downloaded into the runner. The URL embeds the
    bot token, so it is never logged or passed onward -- see rehostFromTelegram in
    worker/src/index.js for the same rule on the Worker side."""
    token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    r = requests.get(f"{API}/bot{token}/getFile", params={"file_id": file_id},
                     timeout=30)
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"getFile rejected: {telegram.redact(body)[:200]}")
    path = body["result"]["file_path"]
    dest = Path(dest_dir) / "input.jpg"
    with requests.get(f"{API}/file/bot{token}/{path}", timeout=120, stream=True) as f:
        f.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in f.iter_content(1 << 16):
                out.write(chunk)
    log(f"fetched the photo from Telegram ({dest.stat().st_size // 1024}KB)")
    return dest


# Wan2GP renders at 512x896 here, and Telegram hands back up to ~1280px. The still
# travels inside the kernel's own SOURCE file as base64 (~1.33x the bytes), and Kaggle
# caps how large a pushed script may be -- so a photo straight off a phone can be the
# difference between a run and a rejected push. Downscaling to a long side comfortably
# above the render size costs nothing visible and keeps the payload small.
MAX_EDGE = 1280


def _shrink(path):
    """Downscale in place when the still is larger than MAX_EDGE. Best effort: if
    Pillow is not installed the original is used, which is correct for every photo
    that was already small enough."""
    try:
        from PIL import Image
    except ImportError:
        log("Pillow not installed; sending the photo at its original size")
        return path
    with Image.open(path) as im:
        if max(im.size) <= MAX_EDGE:
            return path
        im = im.convert("RGB")
        im.thumbnail((MAX_EDGE, MAX_EDGE))
        im.save(path, "JPEG", quality=92)
    log(f"downscaled to {path.stat().st_size // 1024}KB for the kernel payload")
    return path


def main():
    file_id = os.environ.get("TELEGRAM_FILE_ID", "").strip()
    prompt = os.environ.get("MOTION_PROMPT", "").strip()
    if not file_id or not prompt:
        sys.exit("TELEGRAM_FILE_ID and MOTION_PROMPT are both required")

    frames = int(os.environ.get("VIDEO_LENGTH_FRAMES", "").strip() or 81)
    steps = int(os.environ.get("VIDEO_STEPS", "").strip() or 8)
    resolution = os.environ.get("VIDEO_RESOLUTION", "").strip() or "512x896"
    source_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or None

    # Everything the run touches lives in one temp dir, removed in the finally below
    # whatever happens -- a runner is torn down anyway, but "the video only ever
    # existed in one place" should not depend on that.
    work = tempfile.mkdtemp(prefix="private_video_")
    try:
        image_path = _shrink(_fetch_photo(file_id, work))
        # Same rewrite the Kaggle path applies to its own prompts: Wan2GP's docs
        # recommend the second-by-second format for this checkpoint. A prompt already
        # written that way passes through untouched.
        prompt = motion_writer.timeline(prompt, seconds=round(frames / 16))
        log(f"generating: {frames} frames at {resolution}, {steps} steps, "
            f"prompt {len(prompt)} chars")
        video = kaggle_videogen.generate_from_image(
            image_path, prompt, video_length=frames, resolution=resolution,
            steps=steps, dest=str(Path(work) / "out.mp4"))
        telegram.send_video_file(
            str(video),
            caption=f"🔒 private run\n\n{prompt}"[:1024])
        log("delivered to the videos channel")
    except Exception as e:
        # The message itself goes to Telegram, not to the public log: a Kaggle
        # failure quotes the kernel's own log tail back, which can carry the prompt
        # and the task settings. The public log gets the type only, which is enough
        # to see THAT a run failed when reading the Actions list.
        log(f"failed: {type(e).__name__}")
        try:
            telegram.send_text(
                f"🔒 private run failed:\n{type(e).__name__}: "
                f"{telegram.redact(e)[:1200]}", chat_id=source_chat)
        except Exception as send_err:
            log(f"could not report the failure to Telegram either: {send_err}")
        # sys.exit with a bare label, NOT `raise`: re-raising prints the whole
        # traceback to stderr, and a Kaggle failure's message carries the kernel's
        # own log tail -- which quotes the task and can quote the prompt. That
        # traceback would land in the public Actions log, undoing the reason the
        # detail was routed to Telegram in the first place. The run still fails.
        sys.exit(f"private run failed ({type(e).__name__}) -- "
                 "the reason was sent to the Telegram channel")
    finally:
        for p in Path(work).rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        try:
            Path(work).rmdir()
        except OSError:
            pass
        # Best effort, always: leaving the kernel holding the still and the clip is
        # the one leftover this flow can actually do something about, and a scrub
        # that fails must not turn a delivered video into a failed run.
        try:
            kaggle_videogen.scrub_kernel()
        except Exception as e:
            log(f"could not scrub the Kaggle kernel ({type(e).__name__}) -- its "
                "stored source still holds this run's image until the next push")


if __name__ == "__main__":
    main()
