"""mpt image generation -- Kaggle worker. Clones mpt itself, runs
imageslides.generate() for one niche on Kaggle's own GPU (instead of the
default GH Actions CPU runner), copies the resulting images back.

Kaggle's inability to request a specific accelerator via its API turned out
to matter here too, not just for the Wan2.2 video attempt (reverted -- see
videogen.py's docstring): a live P100 kernel died on EVERY image with "CUDA
error: no kernel image is available for execution on the device". Confirmed
live and via Kaggle's own docker-python#1546 -- Kaggle's current base image
ships a PyTorch build that dropped Pascal (sm_60, what the P100 is) entirely,
so ANY GPU op fails there now, not just exotic quantized kernels. Unlike the
Wan2.2 case though, this pipeline has no exotic dependency on a specific
PyTorch/quantization build -- plain SD1.5/SDXL fp16 -- so explicitly
reinstalling torch 2.7.0 (the last release with sm_60 support; dropped in
2.8's cu128 builds) overriding Kaggle's own default is a real fix here, not
a dead end the way it was for Wan2.2's torchao/AOT-compiled requirements.

Placeholders below are substituted by scripts/prepare_image_kernel.py at
push time.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

NICHE_ID = "__NICHE_ID__"
CIVITAI_API_KEY = "__CIVITAI_API_KEY__"
MPT_REPO = "__MPT_REPO__"
MPT_REF = "__MPT_REF__"

WORK = Path("/kaggle/working")
MPT_DIR = Path("/kaggle/tmp/mpt"); MPT_DIR.parent.mkdir(parents=True, exist_ok=True)
OUT_IMAGES = WORK / "images"
STATUS = WORK / "status.json"


def log(m: str) -> None:
    print(f"[kaggle_imagegen] {m}", flush=True)


def write_status(stage: str, ok: bool, extra: dict | None = None) -> None:
    payload = {"stage": stage, "ok": ok, "ts": time.time()}
    if extra:
        payload.update(extra)
    STATUS.write_text(json.dumps(payload, indent=2))


def main() -> None:
    write_status("start", True)
    try:
        log(f"cloning {MPT_REPO}@{MPT_REF}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", MPT_REF, MPT_REPO, str(MPT_DIR)],
            check=True)

        # Kaggle's own preinstalled torch dropped Pascal (sm_60, the P100) support
        # entirely -- confirmed live: every image died with "CUDA error: no kernel
        # image is available for execution on the device" when relying on the
        # default install. 2.7.0 is the last release that still ships sm_60 in its
        # CUDA wheels (dropped in 2.8's cu128 builds) -- explicitly overriding
        # Kaggle's default with it, matched with the paired torchvision release.
        log("installing torch 2.7.0 (last release with Pascal/P100 support, "
            "overriding Kaggle's own default)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.7.0", "torchvision==0.22.0",
             "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True)

        log("installing remaining deps (matches autopilot.yml's list, minus ollama -- "
            "supervisor is disabled below, so no vision model needed here)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "diffusers", "transformers", "accelerate", "safetensors", "peft", "compel",
             # Kaggle's base image ships torchao 0.10.0 -- a live run crashed every
             # single image with "Found an incompatible version of torchao ...
             # only versions above 0.16.0 are supported" (compel/diffusers check
             # this at import/load time even though this pipeline never uses
             # torchao's quantization APIs directly).
             "torchao>=0.16.0",
             "ultralytics", "super-image", "mediapipe==0.10.21", "controlnet_aux",
             "requests"],
            check=True)

        import os
        os.environ["SUPERVISOR_ENABLED"] = "0"
        if CIVITAI_API_KEY:
            os.environ["CIVITAI_API_KEY"] = CIVITAI_API_KEY

        sys.path.insert(0, str(MPT_DIR))
        import imageslides

        niches = json.loads((MPT_DIR / "niches.json").read_text())
        niche = next((n for n in niches["niches"] if n["id"] == NICHE_ID), None)
        if niche is None:
            sys.exit(f"niche {NICHE_ID!r} not found in niches.json")

        log(f"generating for niche {NICHE_ID!r}...")
        # image_prompts is a list, same order/length as images (imageslides.py's
        # own contract) -- not a dict, no re-keying needed, just carry the list
        # through alongside the copied-back filenames in that same order.
        images, vibe, image_prompts = imageslides.generate(niche, state=None)

        OUT_IMAGES.mkdir(parents=True, exist_ok=True)
        names = []
        for img in images:
            dest = OUT_IMAGES / Path(img).name
            shutil.copy(img, dest)
            names.append(dest.name)

        write_status("done", True, {
            "images": names,
            "vibe": vibe,
            "image_prompts": image_prompts,
        })
        log(f"wrote {len(names)} images")
    except Exception as e:
        write_status("failed", False, {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
