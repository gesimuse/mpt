"""Prepare the Kaggle kernel bundle for mpt's own aibeauty image generation --
a faster alternative to the local (GH Actions CPU) path in imageslides.py, run
on Kaggle's own GPU. See kaggle/image_pipeline.py's docstring for why this is
lower-risk than the Wan2.2-on-Kaggle attempt that got reverted, and
kaggle_imagegen.py's docstring for why the kernel receives an already-resolved
checkpoint link rather than a CivitAI spec to resolve itself.

Inputs (env):
  KAGGLE_USERNAME         required -- owner of the kernel
  IMAGEGEN_PAYLOAD_JSON   required -- JSON: {"resolved": {...}, "prompts": [...],
                          "negatives": [...], "adopted": {...}} (see
                          kaggle_imagegen.py's _generate_batch_on_kaggle)
  CIVITAI_API_KEY         optional -- unused by the kernel itself (it never calls
                          civitai.com), kept only in case a future kernel step needs it
  MPT_REPO                default https://github.com/gesimuse/mpt.git
  MPT_REF                 default main
"""
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kernel_build_imagegen"
# Must match what Kaggle actually slugifies "MPT Image Gen Worker" (the title
# below) into -- a live push confirmed Kaggle silently creates the kernel
# under its OWN title-derived slug when it disagrees with kernel-metadata.json's
# `id`, not the id we asked for ("mpt-imagegen-worker" got created as
# "mpt-image-gen-worker" instead, 403ing every subsequent status/output call
# that polled the slug we thought we'd used).
KERNEL_SLUG = "mpt-image-gen-worker"


def require(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def main() -> None:
    username = require("KAGGLE_USERNAME")
    payload_json = require("IMAGEGEN_PAYLOAD_JSON")
    json.loads(payload_json)  # fail fast here, not inside the kernel, if malformed
    mpt_repo = os.environ.get("MPT_REPO", "https://github.com/gesimuse/mpt.git").strip()
    mpt_ref = os.environ.get("MPT_REF", "main").strip()

    src = (ROOT / "kaggle" / "image_pipeline.py").read_text()
    # base64, not a direct string substitution: the payload is real JSON (prompts
    # included) which can contain quotes/backslashes/newlines that would otherwise
    # need careful escaping to stay valid once dropped into a Python source literal.
    payload_b64 = base64.b64encode(payload_json.encode()).decode()
    subs = {
        "__PAYLOAD_B64__": payload_b64,
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

    print(f"[prepare_image_kernel] Bundle ready at {BUILD} (kernel {username}/{KERNEL_SLUG})")


if __name__ == "__main__":
    main()
