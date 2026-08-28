"""Prepare the Kaggle kernel bundle for mpt's own aibeauty image generation --
a faster alternative to the local (GH Actions CPU) path in imageslides.py, run
on Kaggle's own GPU. See kaggle/image_pipeline.py's docstring for why this is
lower-risk than the Wan2.2-on-Kaggle attempt that got reverted.

Inputs (env):
  KAGGLE_USERNAME    required -- owner of the kernel
  NICHE_ID           required -- e.g. "aibeauty"
  CIVITAI_API_KEY    optional
  MPT_REPO           default https://github.com/gesimuse/mpt.git
  MPT_REF            default main
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kernel_build_imagegen"
KERNEL_SLUG = "mpt-imagegen-worker"


def require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def main() -> None:
    username = require("KAGGLE_USERNAME")
    niche_id = require("NICHE_ID")
    civitai_key = os.environ.get("CIVITAI_API_KEY", "").strip()
    mpt_repo = os.environ.get("MPT_REPO", "https://github.com/gesimuse/mpt.git").strip()
    mpt_ref = os.environ.get("MPT_REF", "main").strip()

    src = (ROOT / "kaggle" / "image_pipeline.py").read_text()
    subs = {
        "__NICHE_ID__": niche_id,
        "__CIVITAI_API_KEY__": civitai_key,
        "__MPT_REPO__": mpt_repo,
        "__MPT_REF__": mpt_ref,
    }
    for k, v in subs.items():
        if k not in src:
            sys.exit(f"Template placeholder {k} not found in image_pipeline.py")
        src = src.replace(k, v)

    BUILD.mkdir(exist_ok=True)
    (BUILD / "image_pipeline.py").write_text(src)
    (BUILD / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "MPT Image Gen Worker",
        "code_file": "image_pipeline.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=2))

    print(f"[prepare_image_kernel] Bundle ready at {BUILD} "
          f"(kernel {username}/{KERNEL_SLUG}, niche={niche_id})")


if __name__ == "__main__":
    main()
