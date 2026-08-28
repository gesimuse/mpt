"""mpt image generation -- Kaggle worker. Downloads one already-resolved
CivitAI checkpoint (a direct Cloudflare R2 link, not civitai.com's own
endpoint) and runs sdgen.generate_batch() for one round's prompts on Kaggle's
own GPU (instead of the default GH Actions CPU runner), then writes the
resulting images plus a status.json back for kaggle_imagegen.py to collect.

Deliberately narrow: this kernel never talks to civitai.com (search, decide,
resolve) at all -- confirmed live that civitai.com's own domain 451s requests
from Kaggle's network, while the R2 storage layer underneath is independently
reachable. kaggle_imagegen.py does all CivitAI decision-making in GH Actions
(unblocked) and hands this kernel an already-resolved checkpoint dict whose
"url" is the final R2 link, plus this round's prompts/negatives/adopted
settings. civitai.resolve() is monkeypatched below to just return that dict --
civitai.download() itself needs no changes, it already only ever GETs
resolved["url"] directly.

Kaggle's inability to request a specific accelerator via its API turned out
to matter here too, not just for the Wan2.2 video attempt (reverted -- see
videogen.py's docstring): a live P100 kernel died on EVERY image with "CUDA
error: no kernel image is available for execution on the device". Confirmed
live and via Kaggle's own docker-python#1546 -- Kaggle's current base image
ships a PyTorch build that dropped Pascal (sm_60, what the P100 is) entirely.

The drop is tied to the CUDA TOOLKIT version a wheel is compiled against, not
to the torch release number -- confirmed via PyTorch's own dev-discuss
mailing list ("Maxwell and Pascal architecture support removed in CUDA 12.8
and 12.9 builds"). An earlier attempt here pinned torch==2.7.0 via the cu128
index specifically because cu121/cu124 have both been pruned of a 2.7.0
wheel -- but a live run showed that still hits the exact same CUDA error:
cu128 itself is what dropped Pascal, regardless of which torch release
number happens to be built against it. The actual fix is a torch version old
enough that its matching index predates CUDA 12.8: torch==2.6.0 via cu124
(compiled against CUDA 12.4, well before the sm_60 drop).

Placeholders below are substituted by scripts/prepare_image_kernel.py at
push time.
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"
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
        payload = json.loads(base64.b64decode(PAYLOAD_B64).decode())
        resolved, prompts = payload["resolved"], payload["prompts"]
        negatives, adopted = payload["negatives"], payload["adopted"]

        # Kaggle's base image ships TensorFlow preinstalled (general data-science
        # environment), and transformers unconditionally tries to import it just
        # to load CLIPImageProcessor -- we never use TF at all. A live run's full
        # traceback (only visible thanks to the diagnostic below) showed pinning
        # transformers==4.54.1 (to fix a DIFFERENT torch-2.8-only-API conflict)
        # pulled in a protobuf version incompatible with Kaggle's own preinstalled
        # tensorflow, breaking that unwanted import outright. USE_TF=0 is
        # transformers' own documented flag for exactly this: skip every
        # TensorFlow-dependent code path regardless of whether TF is installed,
        # instead of fighting a protobuf version neither of us actually needs.
        import os
        os.environ["USE_TF"] = "0"

        log(f"cloning {MPT_REPO}@{MPT_REF}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", MPT_REF, MPT_REPO, str(MPT_DIR)],
            check=True)

        # Kaggle's own preinstalled torch dropped Pascal (sm_60, the P100) support
        # entirely -- confirmed live: every image died with "CUDA error: no kernel
        # image is available for execution on the device" when relying on the
        # default install. An earlier attempt here pinned torch==2.7.0 via the
        # cu128 index (the only one still carrying a 2.7.0 wheel -- cu121/cu124
        # have both been pruned of it), which hit the EXACT same CUDA error live:
        # the Pascal drop is tied to the CUDA TOOLKIT a wheel is built against
        # (12.8+, confirmed via PyTorch's own dev-discuss mailing list), not to
        # the torch release number -- cu128 lacks sm_60 no matter which torch
        # version is pinned against it. cu124 (compiled against CUDA 12.4,
        # predating the drop) still has it, at the cost of an older torch/
        # torchvision pairing than we'd otherwise want.
        log("installing torch 2.6.0 via cu124 (last CUDA-toolkit index before "
            "12.8 dropped Pascal/P100 support, overriding Kaggle's own default)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "torch==2.6.0", "torchvision==0.21.0",
             "--index-url", "https://download.pytorch.org/whl/cu124"],
            check=True)

        log("installing remaining deps (matches autopilot.yml's list, minus ollama -- "
            "supervisor QA runs back in GH Actions after images are copied back, "
            "not here)...")
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
             # pinned to torch 2.6.0 specifically because cu124 is the newest
             # CUDA-toolkit index that still supports the P100's sm_60 (see the
             # torch install above) -- an unresolvable conflict with any
             # transformers release new enough to require torch 2.8+ internals.
             # 4.54.1 (2025-07-29) predates torch 2.8's release (2025-08-06)
             # entirely, so it cannot contain code assuming 2.8-only APIs -- no
             # torchao version pin needed on top of this, letting pip resolve
             # whatever (if anything) this older transformers actually declares
             # as a real dependency.
             "transformers==4.54.1",
             "accelerate", "safetensors", "peft", "compel",
             # Re-pinned here too (already installed above via the cu124 index)
             # so ultralytics/super-image see it already satisfied and don't
             # pull in a mismatched plain-PyPI build -- the real fix for THIS
             # incident turned out to be the transformers pin above (confirmed
             # via the diagnostic's full traceback), but this guards against
             # the torch/torchvision ABI mismatch autopilot.yml's own comments
             # already document happening for the CPU path, which is a real
             # risk here too even though it wasn't the actual cause this time.
             "torchvision==0.21.0",
             "ultralytics", "super-image", "mediapipe==0.10.21", "controlnet_aux",
             "requests"],
            check=True)

        # Kaggle's base image ships torchao preinstalled (0.10.0, confirmed live) --
        # same shape of problem as the preinstalled TensorFlow USE_TF=0 works around
        # above, just for a different library. transformers AND peft both probe
        # is_torchao_available() before touching torchao at all, and that probe
        # returns a clean False when torchao isn't importable -- but if it IS
        # installed, peft's own copy of that probe raises outright on anything below
        # 0.16.0 ("Found an incompatible version of torchao..."), and 0.16.0+ itself
        # transitively imports torch.nn.functional.ScalingType/scaled_grouped_mm,
        # torch 2.8-only symbols that don't exist on torch 2.6.0 (pinned above for
        # Pascal/P100 support) -- both confirmed live, one right after "fixing" the
        # other. No version of torchao satisfies both peft's floor and torch 2.6.0,
        # so removing it outright is the only version that works with both: plain,
        # unquantized LoRA fusing never needed it in the first place.
        log("removing Kaggle's preinstalled torchao -- transformers/peft both skip "
            "it cleanly when it's simply not there, avoiding a version nothing "
            "satisfies (see comment above)...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
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

        sys.path.insert(0, str(MPT_DIR))
        import civitai
        import sdgen

        # civitai.resolve() is the only function that ever talks to civitai.com;
        # civitai.download() (called from sdgen's _load_civitai, unchanged) only
        # ever GETs resolved["url"] directly, which is already the final R2 link
        # kaggle_imagegen.py resolved before baking this payload in -- so this
        # kernel makes zero requests to civitai.com's own domain.
        civitai.resolve = lambda spec: resolved

        log(f"generating {len(prompts)} images with {resolved['name']!r}...")
        images = sdgen.generate_batch(
            prompts, OUT_IMAGES, negative_prompts=negatives,
            civitai_model=f"{resolved['model_id']}:{resolved['version_id']}", **adopted)

        write_status("done", True, {"images": [Path(p).name for p in images]})
        log(f"wrote {len(images)} images")
    except Exception as e:
        write_status("failed", False, {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
