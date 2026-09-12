"""Prepare the Kaggle kernel bundle for mpt's Wan2GP-based image-to-video generation
-- a second Kaggle-T4 video attempt (see kernel_build_videogen/video_pipeline.py's
docstring for why this is a different bet than the LTX one that got removed in
31a738f) run on Kaggle's own GPU as a fallback under videogen.py's ZeroGPU ladder.

Inputs (env):
  KAGGLE_USERNAME        required -- owner of the kernel
  VIDEOGEN_PAYLOAD_JSON  required -- JSON: {"image_url": str, "prompt": str,
                         "video_length": int, "resolution": str, "seed": int} (see
                         kaggle_videogen.py's _generate_on_kaggle)
"""
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kernel_build_videogen"
# Same reasoning as prepare_image_kernel.py's own KERNEL_SLUG comment: Kaggle
# slugifies the title, not kernel-metadata.json's `id`, when the two disagree.
KERNEL_SLUG = "mpt-video-gen-worker"


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
    # base64, not a direct string substitution: the payload (prompt text) can contain
    # quotes/backslashes/newlines that would otherwise need careful escaping to stay
    # valid once dropped into a Python source literal -- same reasoning as
    # prepare_image_kernel.py.
    payload_b64 = base64.b64encode(payload_json.encode()).decode()
    placeholder = "__PAYLOAD_B64__"
    if placeholder not in src:
        sys.exit(f"Template placeholder {placeholder} not found in video_pipeline.py")
    src = src.replace(placeholder, payload_b64)

    BUILD.mkdir(exist_ok=True)
    (BUILD / "video_pipeline.py").write_text(src)
    (BUILD / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "MPT Video Gen Worker",
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

    print(f"[prepare_video_kernel] Bundle ready at {BUILD} (kernel {username}/{KERNEL_SLUG})")


if __name__ == "__main__":
    main()
