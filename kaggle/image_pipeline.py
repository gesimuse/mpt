"""mpt image generation -- Kaggle worker. Clones mpt itself, runs
imageslides.generate() for one niche on Kaggle's own GPU (instead of the
default GH Actions CPU runner), copies the resulting images back.

Unlike the Wan2.2 video-generation attempt on Kaggle (reverted -- see
videogen.py's docstring), this has none of the exotic-quantized-kernel /
accelerator-selection landmines: sdgen.py's standard SD1.5/SDXL fp16
inference has run fine on P100-class GPUs for years, so Kaggle's inability
to request a specific accelerator via its API doesn't block this the way it
did there.

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

        # torch is NOT reinstalled -- Kaggle's own preinstalled build is already
        # matched to its GPU driver. A real run tried reinstalling torch for a
        # different (Wan2.2) Kaggle experiment and that alone broke GPU support
        # entirely on a P100 (current PyTorch dropped Pascal/sm_60 support).
        # Standard fp16 SD1.5/SDXL inference has no such requirement.
        log("installing deps (matches autopilot.yml's list, minus torch/ollama -- "
            "supervisor is disabled below, so no vision model needed here)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "diffusers", "transformers", "accelerate", "safetensors", "peft", "compel",
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
