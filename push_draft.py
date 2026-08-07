#!/usr/bin/env python3
"""Push an already-generated, already-reviewed batch to TikTok as an inbox draft.

The other half of the local-GPU workflow: `DRY_RUN=1 python3 autopilot.py` generates
and QAs a batch on whatever hardware is available (sdgen.py auto-detects a local GPU
if present) and writes it to ./out/<niche>-<stamp>/ instead of posting anything. Look
at the images yourself, and if a batch is worth posting, hand its folder to this
script -- it hosts the images, queues the TikTok inbox draft, and records the upload
in posted.json/CAPTIONS.md exactly like a live (non-DRY_RUN) autopilot.py run would.

Usage:
  python3 push_draft.py out/aibeauty-20260101-120000
  python3 push_draft.py out/aibeauty-20260101-120000 --niche aibeauty  # if the folder
                                                                        # name doesn't
                                                                        # start with the
                                                                        # niche id
"""
import argparse, sys, time
from pathlib import Path

import autopilot
import tiktok


def push_draft(folder, niche_id=None):
    """Host the folder's images, queue the TikTok inbox draft, and record the upload.
    Returns the publish_id. Raises ValueError/RuntimeError on anything that stops the
    push -- kept separate from main() so it's callable (and testable) without argparse
    or a process exit."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"not a directory: {folder}")
    niche_id = niche_id or folder.name.rsplit("-", 2)[0]

    images = sorted(folder.glob("*.jpg"))
    if len(images) < 2:
        raise ValueError(f"{folder} has {len(images)} .jpg file(s); "
                         "a TikTok carousel needs at least 2")
    caption_file = folder / "caption.txt"
    caption = caption_file.read_text().strip() if caption_file.exists() else ""

    print(f"Pushing {len(images)} images from {folder} as an inbox draft for {niche_id!r}...")
    publish_id = tiktok.publish_photos_draft([str(p) for p in images], niche_id, caption=caption)
    if publish_id is None:
        raise RuntimeError(
            f"no TikTok credentials configured for {niche_id!r} "
            f"(TIKTOK_CLIENT_KEY/TIKTOK_CLIENT_SECRET/TIKTOK_REFRESH_TOKEN_{niche_id.upper()})")

    # init's 200 OK only means the job was accepted, not that it reached the inbox --
    # live-confirmed a real batch of "successful" pushes included several that had
    # actually failed downstream (photo_pull_failed/file_format_check_failed) and
    # never showed up in the app. Poll for the real outcome before recording it.
    print("Waiting for TikTok to confirm the draft actually reached the inbox...")
    status, fail_reason = tiktok.check_publish_status(publish_id, niche_id)
    if status != "SEND_TO_USER_INBOX":
        print(f"WARNING: draft did not reach the inbox -- status={status} "
              f"fail_reason={fail_reason}. Recording it as failed, not queued.")

    state = autopilot.load_state()
    state["uploads"].append({
        "niche": niche_id, "topic": f"local batch {folder.name}",
        "title": caption.splitlines()[0][:95] if caption else folder.name,
        "tiktok": status == "SEND_TO_USER_INBOX", "tiktok_via": "inbox",
        "tiktok_post_id": publish_id, "tiktok_status": status,
        "tiktok_caption": caption,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    autopilot.save_state(state)
    autopilot.write_pending_captions(state)
    if status == "SEND_TO_USER_INBOX":
        print(f"Queued as draft, publish_id={publish_id}. Caption pre-filled on the draft "
              "(also saved to CAPTIONS.md as a fallback) -- open the TikTok app, add sound, "
              "and post.")
    return publish_id


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="a DRY_RUN output folder, e.g. out/aibeauty-20260101-120000")
    p.add_argument("--niche", help="niche id, if not inferable from the folder name")
    args = p.parse_args()
    try:
        push_draft(args.folder, args.niche)
    except (ValueError, RuntimeError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
