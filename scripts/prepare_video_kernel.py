"""Prepare the Kaggle kernel bundle for mpt's image-to-video fallback.

Same shape as scripts/prepare_image_kernel.py (read that one first -- the base64
payload substitution and the "Kaggle slugifies the TITLE, not your id" trap are
identical here), but for kaggle/video_pipeline.py.

Inputs (env):
  KAGGLE_USERNAME       required -- owner of the kernel
  VIDEOGEN_PAYLOAD_JSON required -- JSON: {"image_url": ..., "prompt": ...,
                        "length_s": ..., "steps": ..., "negative_prompt": ...}
"""
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kernel_build_videogen"
# Must match what Kaggle slugifies the title below into -- a live push of the IMAGE
# kernel confirmed Kaggle silently creates the kernel under its OWN title-derived slug
# when that disagrees with kernel-metadata.json's `id`, 403ing every later status/
# output call that polled the slug we thought we'd used.
KERNEL_SLUG = "mpt-video-gen-worker"
KERNEL_TITLE = "MPT Video Gen Worker"


def require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def main() -> None:
    username = require("KAGGLE_USERNAME")
    payload_json = require("VIDEOGEN_PAYLOAD_JSON")
    json.loads(payload_json)  # fail fast here, not inside the kernel, if malformed

    src = (ROOT / "kaggle" / "video_pipeline.py").read_text()
    placeholder = "__PAYLOAD_B64__"
    if placeholder not in src:
        sys.exit(f"Template placeholder {placeholder} not found in video_pipeline.py")
    src = src.replace(placeholder, base64.b64encode(payload_json.encode()).decode())

    BUILD.mkdir(exist_ok=True)
    (BUILD / "video_pipeline.py").write_text(src)
    (BUILD / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "video_pipeline.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))

    print(f"[prepare_video_kernel] Bundle ready at {BUILD} "
          f"(kernel {username}/{KERNEL_SLUG})")


if __name__ == "__main__":
    main()
