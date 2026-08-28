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
             # cu121 (only through 2.5.1) and cu124 (only through 2.6.0) have both
             # been pruned of 2.7.0 -- confirmed live against the real index,
             # despite PyTorch's own "previous versions" docs page still listing
             # both as valid for 2.7.0 (a stale snapshot; older CUDA-version
             # indices get pruned of old releases over time as newer ones become
             # current). cu128 is the newest index and hasn't been pruned yet.
             "--index-url", "https://download.pytorch.org/whl/cu128"],
            check=True)

        log("installing remaining deps (matches autopilot.yml's list, minus ollama -- "
            "supervisor is disabled below, so no vision model needed here)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "diffusers",
             # Confirmed root cause via the diagnostic below (a live run's FULL
             # traceback, not the 100-char-truncated one-liner from sdgen.py's
             # own per-image handler): latest transformers unconditionally
             # imports its torchao quantizer support just to load CLIPTextModel
             # -- even though this pipeline never uses quantization -- and that
             # chain does `from torch.nn.functional import ScalingType,
             # scaled_grouped_mm`, symbols that only exist in torch 2.8+. We're
             # pinned to torch 2.7.0 specifically because that's the last
             # release supporting the P100's sm_60 -- an unresolvable conflict
             # with any transformers release new enough to require torch 2.8+
             # internals. 4.54.1 (2025-07-29) predates torch 2.8's release
             # (2025-08-06) entirely, so it cannot contain code assuming
             # 2.8-only APIs -- no torchao version pin needed on top of this,
             # letting pip resolve whatever (if anything) this older
             # transformers actually declares as a real dependency.
             "transformers==4.54.1",
             "accelerate", "safetensors", "peft", "compel",
             # Re-pinned here too (already installed above via the cu128 index)
             # so ultralytics/super-image see it already satisfied and don't
             # pull in a mismatched plain-PyPI build -- the real fix for THIS
             # incident turned out to be the transformers pin above (confirmed
             # via the diagnostic's full traceback), but this guards against
             # the torch/torchvision ABI mismatch autopilot.yml's own comments
             # already document happening for the CPU path, which is a real
             # risk here too even though it wasn't the actual cause this time.
             "torchvision==0.22.0",
             "ultralytics", "super-image", "mediapipe==0.10.21", "controlnet_aux",
             "requests"],
            check=True)

        # Diagnostic: sdgen.py's own per-image exception handler truncates
        # errors to 100 chars, which is exactly what hid the real cause of a
        # prior live failure here. Import the actual failing module directly,
        # first, so if it breaks again the FULL traceback lands in status.json
        # instead of another one-line guess.
        try:
            import diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion  # noqa: F401
        except Exception:
            import traceback
            # sys.exit (SystemExit), not raise -- the outer `except Exception`
            # below would otherwise overwrite this detailed status.json with
            # its own truncated str(e) version.
            write_status("failed", False, {"error": "diffusers import failed",
                                           "traceback": traceback.format_exc()})
            sys.exit(1)

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
