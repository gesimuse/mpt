#!/usr/bin/env python3
"""
MPT Autopilot (aibeauty): search CivitAI for a real reference photo -> generate and QA
camera variations locally -> queue the survivors as a native TikTok inbox draft, ready
for the account owner to add sound and a caption by hand before posting.

Env vars:
  NIM_API_KEY                    NVIDIA NIM key -- used by supervisor.py's vision QA
  CIVITAI_API_KEY                read-only: CivitAI search/gallery only, never near
                                  CivitAI's separate paid generation API
  TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET / TIKTOK_REFRESH_TOKEN_<NICHEID>
  GITHUB_TOKEN / GITHUB_REPOSITORY   hosts generated images so TikTok can fetch them
Optional:
  NICHES      comma-separated niche ids to run (default: all)
  DRY_RUN     generate + QA, write to ./out, never queue a draft -- the natural way to
              run this on a local GPU box: generate, look at ./out yourself, and use
              push_draft.py on whichever batch turned out well.
"""
import json, os, shutil, sys, time
from pathlib import Path

import imageslides
import tiktok

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


def log(msg): print(f"[autopilot] {msg}", flush=True)


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


def run_niche(niche, state):
    if not DRY_RUN and not tiktok.enabled(niche["id"]):
        log(f"[{niche['id']}] skipped: set TIKTOK_CLIENT_KEY/TIKTOK_CLIENT_SECRET/"
            f"TIKTOK_REFRESH_TOKEN_{niche['id'].upper()}")
        return
    used = state["topics"].setdefault(niche["id"], [])
    for _ in range(niche.get("videos_per_run", 1)):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        images = imageslides.generate(niche)
        caption = imageslides.image_caption(niche)
        log(f"[{niche['id']}] caption (paste by hand when finishing the draft):\n{caption}")

        if DRY_RUN:
            dest_dir = OUT_DIR / f"{niche['id']}-{stamp}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy(img, dest_dir / Path(img).name)
            (dest_dir / "caption.txt").write_text(caption + "\n")
            log(f"[{niche['id']}] DRY_RUN: no upload, {len(images)} images at {dest_dir}")
            used.append(f"{niche['id']}-{stamp}")
            continue

        publish_id = tiktok.publish_photos_draft(images, niche["id"])
        used.append(f"{niche['id']}-{stamp}")
        state["uploads"].append({
            "niche": niche["id"], "topic": f"image slideshow {stamp}",
            "title": caption.splitlines()[0][:95],
            "tiktok": bool(publish_id), "tiktok_via": "inbox",
            "tiktok_post_id": publish_id, "tiktok_caption": caption,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save_state(state)
        write_pending_captions(state)
        for img in images:
            try:
                os.remove(img)
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
