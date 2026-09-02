#!/usr/bin/env python3
"""
MPT Autopilot (aibeauty): search CivitAI for a real reference photo -> generate and QA
camera variations locally -> queue the survivors as a native TikTok inbox draft, ready
for the account owner to add sound and a caption by hand before posting.

Env vars:
  OLLAMA_URL                     used by supervisor.py
  CIVITAI_API_KEY                read-only: CivitAI search/gallery only, never near
                                  CivitAI's separate paid generation API
  TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET / TIKTOK_REFRESH_TOKEN_<NICHEID>
  PAGES_BASE_URL                  e.g. https://gesimuse.github.io/mpt -- images are
                                   committed to the gh-pages branch (tiktok.py) and
                                   served directly from there for TikTok to fetch
  HF_TOKEN                        attributes HF ZeroGPU calls to your account (avoids
                                   anonymous throttling). Only needed by the video
                                   niche (content_type: video_via_motionforge).
Optional:
  NICHES      comma-separated niche ids to run (default: all)
  DRY_RUN     generate + QA, write to ./out, never queue a draft -- the natural way to
              run this on a local GPU box: generate, look at ./out yourself, and use
              push_draft.py on whichever batch turned out well.
"""
import json, os, shutil, sys, time
from datetime import datetime
from pathlib import Path

import imageslides
import kaggle_imagegen
import motion_writer
import telegram
import tiktok
import videogen

ROOT = Path(__file__).resolve().parent

# load .env if present; skipped for real deploys (CI sets env directly)
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

STATE_FILE = ROOT / "posted.json"
# DRY_RUN generates and QAs everything but queues nothing, leaving the images in
# ./out for review.
DRY_RUN = os.environ.get("DRY_RUN", "").strip().lower() in ("1", "true", "yes")
OUT_DIR = ROOT / "out"
RUN_ATTEMPTS = int(os.environ.get("RUN_ATTEMPTS", "3"))
# TikTok's Content Posting API caps at 5 pending (unposted) drafts within any rolling
# 24h period -- exceeding it fails new pushes with spam_risk_too_many_pending_share
# (confirmed against TikTok's own docs). There is no API to ask TikTok how many drafts
# are still sitting untouched in the inbox right now, so this counts our own successful
# pushes to this niche from posted.json instead.
#
# Deliberately a CALENDAR DAY, not the rolling 24h window TikTok itself uses: with the
# five crons at 05-13 UTC, a rolling window let yesterday's late runs still occupy this
# morning's budget, so the first run(s) of a day got skipped for drafts the account
# owner had usually already dealt with. Commit 06709de ("reset aibeauty pending-drafts
# count for today") was a manual workaround for exactly that. A midnight reset is
# slightly more permissive than TikTok's real limit at the boundary -- the guard against
# actually tripping spam_risk_too_many_pending_share is that all five crons sit inside a
# single UTC day, so at most MAX_PENDING_DRAFTS get pushed per day regardless.
MAX_PENDING_DRAFTS = int(os.environ.get("MAX_PENDING_DRAFTS", "5"))


def log(msg): print(f"[autopilot] {msg}", flush=True)


def _motion_prompts_for(image_prompts):
    """One motion instruction per image_prompt, via motion_writer (LLM) -- None on
    any failure (Ollama unreachable, as it always is in CI today), NOT a baked-in
    generic string. A fixed fallback string here would make every image in a batch
    (and every batch, forever, while Ollama stays unreachable) show the identical
    prompt in the picker -- worse than before this feature existed, when the
    picker at least showed each image's own (if imperfect) SD prompt. Leaving it
    None lets the picker fall back to that per-image SD prompt instead, so
    photos stay visually distinguishable even without a working LLM."""
    out = []
    for p in image_prompts:
        if not p:
            out.append(None)
            continue
        try:
            out.append(motion_writer.write(p))
        except Exception as e:
            log(f"motion_writer failed, no motion prompt recorded for this image "
                f"({type(e).__name__}: {str(e)[:100]})")
            out.append(None)
    return out


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"topics": {}, "uploads": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def write_pending_captions(state, keep=10):
    """A phone-readable list of captions for whatever is still sitting as an inbox
    draft -- every upload here is a draft, there is no auto-published path to skip."""
    lines = ["# TikTok captions to paste when finishing a draft", ""]
    for u in reversed(state["uploads"][-keep:]):
        if not u.get("tiktok") or not u.get("tiktok_caption"):
            continue
        lines += [f"## {u['ts']} — {u['niche']}", "", "```", u["tiktok_caption"], "```", ""]
    (ROOT / "CAPTIONS.md").write_text("\n".join(lines))


def _drafts_today(state, niche_id, today=None):
    """How many drafts we've pushed for this niche so far TODAY, per posted.json -- our
    proxy for TikTok's own 5-pending cap (see MAX_PENDING_DRAFTS for why this is a
    calendar day rather than TikTok's own rolling 24h window).

    `ts` is written by time.strftime, i.e. in the runner's local time, which is UTC on
    GitHub Actions -- comparing it against datetime.now().date() is self-consistent
    because both sides come from the same clock."""
    today = today or datetime.now().date()
    count = 0
    for u in state.get("uploads", []):
        if u.get("niche") != niche_id or not u.get("tiktok"):
            continue
        try:
            ts = datetime.strptime(u["ts"], "%Y-%m-%dT%H:%M:%S")
        except (KeyError, ValueError):
            continue
        if ts.date() == today:
            count += 1
    return count


def run_niche(niche, state):
    if niche.get("content_type") == "video_via_motionforge":
        return _run_video_niche(niche, state)
    if not DRY_RUN and not tiktok.enabled(niche["id"]):
        log(f"[{niche['id']}] skipped: set TIKTOK_CLIENT_KEY/TIKTOK_CLIENT_SECRET/"
            f"TIKTOK_REFRESH_TOKEN_{niche['id'].upper()}")
        return
    videos_this_run = niche.get("videos_per_run", 1)
    if not DRY_RUN:
        pending = _drafts_today(state, niche["id"])
        if pending >= MAX_PENDING_DRAFTS:
            log(f"[{niche['id']}] skipped: {pending} drafts already pushed today "
                f"(cap {MAX_PENDING_DRAFTS}); clear or post them in the TikTok app "
                "before more get queued")
            return
        # Also clamp a partial run: pushing all videos_per_run when some of the cap is
        # already used would hit spam_risk_too_many_pending_share mid-run instead of
        # stopping cleanly at the boundary.
        remaining = MAX_PENDING_DRAFTS - pending
        if videos_this_run > remaining:
            log(f"[{niche['id']}] {pending} drafts already pushed today; "
                f"generating {remaining} instead of {videos_this_run} to stay under "
                f"the cap of {MAX_PENDING_DRAFTS}")
            videos_this_run = remaining
    used = state["topics"].setdefault(niche["id"], [])
    for _ in range(videos_this_run):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        # Kaggle's own GPU generates the same images much faster than the local
        # GH Actions CPU runner -- tried first when configured, but NEVER the
        # only path: any failure here (missing creds, push/poll error, kernel
        # crash) falls straight back to the exact same local imageslides.generate()
        # call this always used, unchanged. A speed win is not worth risking the
        # pipeline that already works.
        images = vibe = look = image_prompts = None
        if kaggle_imagegen.available():
            try:
                images, vibe, look, image_prompts = kaggle_imagegen.generate(
                    niche, state=state)
                log(f"[{niche['id']}] generated on Kaggle's GPU ({len(images)} images)")
            except Exception as e:
                log(f"[{niche['id']}] Kaggle image gen failed "
                    f"({type(e).__name__}: {str(e)[:200]}); falling back to local")
        if images is None:
            images, vibe, look, image_prompts = imageslides.generate(niche, state=state)
        caption = imageslides.image_caption(niche, vibe=vibe, state=state)
        log(f"[{niche['id']}] caption (pre-filled on the draft; also saved to "
            f"CAPTIONS.md as a fallback):\n{caption}")

        if DRY_RUN:
            dest_dir = OUT_DIR / f"{niche['id']}-{stamp}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy(img, dest_dir / Path(img).name)
            (dest_dir / "caption.txt").write_text(caption + "\n")
            log(f"[{niche['id']}] DRY_RUN: no upload, {len(images)} images at {dest_dir}")
            used.append(f"{niche['id']}-{stamp}")
            continue

        # Pre-host so we can record the URLs on state -- the video niche
        # (aibeautyvideo) reads state["uploads"][-1]["image_urls"] to pick a source
        # image for motionforge without re-generating. publish_photos_draft accepts
        # image_urls to skip its own hosting when we've done it here.
        image_urls = [tiktok.host_file(p) for p in images]
        motion_prompts = _motion_prompts_for(image_prompts)
        publish_id = tiktok.publish_photos_draft(
            images, niche["id"], image_urls=image_urls, caption=caption)
        used.append(f"{niche['id']}-{stamp}")
        # A publish_id back from init means TikTok ACCEPTED the job, not that it
        # actually reached the inbox -- confirmed live, 4 of 8 recorded "successes"
        # had actually failed downstream (photo_pull_failed/file_format_check_failed)
        # and were never polled for their real outcome, so they kept counting toward
        # the pending-drafts cap and blocking real runs while nothing was in the
        # inbox to show for it. Poll for the real status before recording success.
        status, fail_reason = (
            tiktok.check_publish_status(publish_id, niche["id"]) if publish_id
            else (None, None))
        if publish_id and status != "SEND_TO_USER_INBOX":
            log(f"[{niche['id']}] draft did not actually reach the inbox: "
                f"status={status} fail_reason={fail_reason}")
        state["uploads"].append({
            "niche": niche["id"], "topic": f"image slideshow {stamp}",
            "title": caption.splitlines()[0][:95],
            "tiktok": status == "SEND_TO_USER_INBOX", "tiktok_via": "inbox",
            "tiktok_post_id": publish_id, "tiktok_status": status,
            "tiktok_caption": caption,
            "image_urls": image_urls,
            # Per-image SD prompt (same order as image_urls) -- the actual framing/
            # pose/lighting that generated each photo. Kept for reference/debugging;
            # NOT what the video picker pre-fills (that's motion_prompts below) --
            # this describes the still, not a motion.
            "image_prompts": image_prompts,
            # The theme this batch was built from (imageslides.DEFAULT_THEMES). Two
            # jobs: it's the key picker.html's posted/skipped verdict gets attributed
            # to (imageslides._owner_theme_rates), and it makes a batch's own look
            # legible in posted.json without reverse-engineering it from a prompt.
            "vibe": vibe,
            # Which SUBJECTS entry this batch's woman came from. Recorded for the same
            # reason as vibe: imageslides._recently_used reads both back so the next
            # few batches don't repeat either.
            "look": look,
            # Per-image motion instruction (motion_writer.write, an LLM rewrite of
            # image_prompts[i] into an actual action for the person to do) -- what
            # the video picker actually pre-fills its motion-prompt field with.
            "motion_prompts": motion_prompts,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save_state(state)
        write_pending_captions(state)
        # Review surface. Posted AFTER state is saved, so a Telegram outage can never
        # cost us the record of a batch that already reached TikTok -- and the button
        # callbacks look this batch up in posted.json by ts, so the entry has to exist
        # before the buttons can work at all.
        if telegram.enabled():
            # Message ids recorded alongside the urls: without them a message can
            # never be deleted or edited later, which is what made moving the videos
            # to their own channel a manual cleanup job.
            entry = state["uploads"][-1]
            ids = telegram.post_batch(image_urls, entry["ts"],
                                      caption=caption, motion_prompts=motion_prompts)
            if ids:
                entry["telegram_message_ids"] = {
                    url: {"chat_id": os.environ.get("TELEGRAM_CHAT_ID"),
                          "message_id": mid}
                    for url, mid in zip(image_urls, ids)}
                save_state(state)
        for img in images:
            try:
                os.remove(img)
            except OSError:
                pass


def _pick_source_image_url(niche, state):
    """The most recent successfully-posted image from the photo niche this video
    niche is chained to (default: strip 'video' suffix off the id, so 'aibeautyvideo'
    reuses 'aibeauty'). Skips images we've already SUCCESSFULLY animated -- a video
    attempt that failed (Kaggle error, TikTok rejected the draft) must not burn its
    source image forever; the whole point of a retriable failure is that the image
    is still fair game for another attempt. state["uploads"] entries this function
    created carry motionforge_source_url, and we compare against those where
    tiktok=true. Returns None when there's nothing fresh to work with (photo cron
    hasn't run yet, or every recent image is already successfully video'd)."""
    source_niche = niche.get("source_niche") or niche["id"].removesuffix("video")
    used = {u.get("motionforge_source_url") for u in state.get("uploads", [])
            if u.get("niche") == niche["id"] and u.get("tiktok")}
    for u in reversed(state.get("uploads", [])):
        if u.get("niche") != source_niche or not u.get("tiktok"):
            continue
        urls = u.get("image_urls") or []
        for url in urls:
            if url and url not in used:
                return url
    return None


def _publish_video(niche, state, token_niche, video_url, caption, topic, extra_fields):
    """Publish an already-hosted mp4 URL to TikTok and record the outcome in state --
    shared by both a fresh generation and a retry of one already generated, so a
    TikTok-side failure always leaves the SAME shape behind (video_url,
    tiktok_fail_reason included) for the picker to list and let the account owner
    download or retry regardless of whether TikTok ever accepted it."""
    publish_id = tiktok.publish_video_draft(
        None, niche["id"], video_url=video_url, caption=caption, token_niche=token_niche)
    status, fail_reason = (
        tiktok.check_publish_status(publish_id, niche["id"], token_niche=token_niche)
        if publish_id else (None, None))
    if publish_id and status != "SEND_TO_USER_INBOX":
        log(f"[{niche['id']}] draft did not actually reach the inbox: "
            f"status={status} fail_reason={fail_reason}")
    state["uploads"].append({
        "niche": niche["id"], "topic": topic,
        "title": caption.splitlines()[0][:95],
        "tiktok": status == "SEND_TO_USER_INBOX", "tiktok_via": "inbox_video",
        "tiktok_post_id": publish_id, "tiktok_status": status,
        "tiktok_fail_reason": fail_reason,
        "tiktok_caption": caption,
        "video_url": video_url,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra_fields,
    })
    save_state(state)
    write_pending_captions(state)
    if telegram.enabled():
        entry = state["uploads"][-1]
        try:
            mid = telegram.send_video(
                video_url, entry["ts"],
                caption=("✅ queued to TikTok" if entry.get("tiktok")
                         else f"❌ TikTok rejected it: {fail_reason or status}"),
                failed=not entry.get("tiktok"))
            # Recorded for the same reason image ids are: without it the message can
            # never be moved or removed later, which is what left a video stranded in
            # the photos channel with no way to clean it up programmatically.
            entry.setdefault("telegram_message_ids", {})[video_url] = {
                "chat_id": telegram.video_chat_id(), "message_id": mid}
            save_state(state)
        except Exception as e:
            log(f"could not post the video to Telegram "
                f"({type(e).__name__}: {str(e)[:150]})")


def _run_video_niche(niche, state):
    """Reuse an image the photo niche already generated + QA'd, animate it with
    videogen.py (HF ZeroGPU / Wan 2.2 I2V rCM), publish the mp4 as a TikTok inbox
    draft via MPT's own tiktok.publish_video_draft. No cross-repo workflow_dispatch,
    no artifact plumbing.

    Uses the source niche's TikTok token (tiktok_token_niche in niches.json; defaults
    to source_niche) so the video draft lands in the same channel as the photos --
    no second OAuth setup needed.

    No pending-drafts cap check here: TikTok's cap applies to the ACCOUNT, so the photo
    cron and video pushes collectively must stay under 5, but we can't poll TikTok for
    that number. Kept simple: this niche has no cron at all -- every video run is fired
    by hand from picker.html, so the account owner is already looking at the inbox when
    one goes out."""
    token_niche = niche.get("tiktok_token_niche") or (
        niche.get("source_niche") or niche["id"].removesuffix("video"))
    if not DRY_RUN and not tiktok.enabled(token_niche):
        log(f"[{niche['id']}] skipped: TIKTOK_REFRESH_TOKEN_{token_niche.upper()} not set")
        return

    # VIDEO_RETRY_URL (the picker's Retry button on an already-generated video that
    # TikTok rejected) skips the HF Space call entirely -- no reason to pay another
    # multi-minute generation to re-attempt publishing the SAME mp4 that already
    # exists and is already hosted.
    retry_url = os.environ.get("VIDEO_RETRY_URL", "").strip()
    if retry_url:
        prompt = (os.environ.get("VIDEO_PROMPT", "").strip()
                  or niche.get("motionforge_prompt", "").strip())
        caption = imageslides.image_caption(niche, vibe=prompt or None, state=state)
        log(f"[{niche['id']}] retrying publish of {retry_url} (no regeneration)")
        if DRY_RUN:
            log(f"[{niche['id']}] DRY_RUN: would retry-publish {retry_url}")
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        _publish_video(niche, state, token_niche, retry_url, caption,
                       f"video retry {stamp}", {"tiktok_via": "inbox_video_retry"})
        return

    # Env overrides win over auto-pick -- push_video.py's local UI fires
    # workflow_dispatch with the user's chosen image + edited prompt as inputs,
    # which the workflow surfaces as VIDEO_IMAGE_URL / VIDEO_PROMPT here.
    image_url = os.environ.get("VIDEO_IMAGE_URL", "").strip()
    if not image_url:
        image_url = _pick_source_image_url(niche, state)
    if not image_url:
        log(f"[{niche['id']}] no source image (VIDEO_IMAGE_URL unset and no fresh "
            f"aibeauty upload in state); nothing to animate")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    prompt = (os.environ.get("VIDEO_PROMPT", "").strip()
              or niche.get("motionforge_prompt", "").strip())
    length_s = os.environ.get("VIDEO_LENGTH_S", "").strip() or niche.get("motionforge_length_s", "5.0")
    steps = os.environ.get("VIDEO_STEPS", "").strip() or niche.get("motionforge_steps", "4")
    video_path = videogen.generate(image_url, prompt, length_s=length_s, steps=steps)
    caption = imageslides.image_caption(niche, vibe=prompt or None, state=state)
    log(f"[{niche['id']}] caption (pre-filled on the draft):\n{caption}")

    if DRY_RUN:
        dest_dir = OUT_DIR / f"{niche['id']}-{stamp}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(video_path, dest_dir / Path(video_path).name)
        (dest_dir / "caption.txt").write_text(caption + "\n")
        log(f"[{niche['id']}] DRY_RUN: no upload, video at {dest_dir}")
        return

    # Host regardless of what happens next -- a generated video must never become
    # invisible just because TikTok's own downstream check rejects it (a live run
    # hit exactly this: frame_rate_check_failed, mp4 was fine, just nowhere to see
    # or retry it from since nothing recorded the hosted URL).
    video_url = tiktok.host_file(video_path)
    _publish_video(niche, state, token_niche, video_url, caption,
                   f"video {stamp}", {"motionforge_source_url": image_url})
    try:
        os.remove(video_path)
    except OSError:
        pass


def run_niche_with_retries(niche, state, attempts=None):
    """Most failures here are other people's outages -- CivitAI rate limiting, NIM
    refusing a vision-QA request. Waiting for the next cron to retry wastes a slot, so
    retry inside the run with a widening gap."""
    attempts = attempts or RUN_ATTEMPTS
    last = None
    for i in range(attempts):
        try:
            run_niche(niche, state)
            return
        except Exception as e:
            last = e
            log(f"[{niche['id']}] attempt {i + 1}/{attempts} failed: {type(e).__name__}: {str(e)[:200]}")
            if i < attempts - 1:
                wait = 30 * (i + 1)
                log(f"[{niche['id']}] retrying in {wait}s")
                time.sleep(wait)
    raise last


def main():
    niches = json.loads((ROOT / "niches.json").read_text())["niches"]
    only = [s.strip() for s in os.environ.get("NICHES", "").split(",") if s.strip()]
    if only:
        niches = [n for n in niches if n["id"] in only]
    state = load_state()
    failures = []
    for niche in niches:
        try:
            run_niche_with_retries(niche, state)
        except Exception as e:
            log(f"[{niche['id']}] FAILED: {e}")
            failures.append(niche["id"])
    if failures:
        sys.exit(f"Failed niches: {failures}")
    log("All niches done.")


if __name__ == "__main__":
    main()
