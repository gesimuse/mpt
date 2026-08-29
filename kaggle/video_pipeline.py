"""mpt image-to-video -- Kaggle worker. Animates one already-hosted still into a
short mp4 on Kaggle's own GPU, then writes it plus a status.json back for
kaggle_videogen.py to collect.

Why this exists: HF ZeroGPU (videogen.py, the primary path) gives roughly 5 minutes
of GPU per HF account per DAY. That is a hard ceiling no amount of retrying moves.
Kaggle gives ~30 GPU-hours a week on a real T4. So ZeroGPU stays first -- it is far
faster end to end (~2 min vs ~10-20 here, most of it model download) and produces the
better model's output -- and this is what runs when every Space x token combination
is spent.

Model is LTX-Video 0.9.x distilled, NOT Wan 2.2 TI2V-5B, deliberately:
  * The T4 has no bf16. Wan's reference dtype is bf16; forcing it to fp16 is a real
    NaN risk, and 5B on 16GB needs aggressive offload on top.
  * LTX distilled is ~2B, fp16-native, and fits a T4 comfortably.
Wan remains the quality path on ZeroGPU, where the hardware suits it.

The kernel is pushed with --accelerator NvidiaTeslaT4 (sm_75). If Kaggle hands us a
P100 anyway, sm_60 is detected below and the run is abandoned with a clear reason
rather than dying inside CUDA -- the cu124-torch workaround the image kernel carries
is not worth replicating for a fallback path.

The source image arrives as a gh-pages URL. That is reachable from Kaggle; civitai.com
is NOT (confirmed -- it 451s every request from Kaggle's network), which is why the
image kernel is handed a pre-resolved R2 link and never talks to CivitAI. Nothing here
touches CivitAI at all.

Placeholders below are substituted by scripts/prepare_video_kernel.py at push time.
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

PAYLOAD_B64 = "__PAYLOAD_B64__"

WORK = Path("/kaggle/working")
OUT_VIDEO = WORK / "video.mp4"
STATUS = WORK / "status.json"

MODEL_ID = "Lightricks/LTX-Video-0.9.7-distilled"
# LTX works in latent tiles of 32 px and 8 frames; anything else gets rounded by the
# pipeline anyway, so round here where it's visible. Portrait, for TikTok.
HEIGHT, WIDTH = 768, 512
FPS = 24


def log(m: str) -> None:
    print(f"[kaggle_videogen] {m}", flush=True)


def write_status(stage: str, ok: bool, extra: dict | None = None) -> None:
    payload = {"stage": stage, "ok": ok, "ts": time.time()}
    if extra:
        payload.update(extra)
    STATUS.write_text(json.dumps(payload, indent=2))


def main() -> None:
    write_status("start", True)
    try:
        payload = json.loads(base64.b64decode(PAYLOAD_B64).decode())
        image_url = payload["image_url"]
        prompt = payload["prompt"]
        negative_prompt = payload.get("negative_prompt") or (
            "worst quality, inconsistent motion, blurry, jittery, distorted")
        length_s = float(payload.get("length_s") or 5.0)

        # Same reason the image kernel sets it: Kaggle's base image ships TensorFlow,
        # and transformers imports it unconditionally for tokenizer/processor loading
        # even though nothing here uses it.
        import os
        os.environ["USE_TF"] = "0"

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device on this kernel")
        capability = torch.cuda.get_device_capability()
        log(f"accelerator: {torch.cuda.get_device_name()} "
            f"(sm_{capability[0]}{capability[1]}), torch {torch.__version__}")
        if capability == (6, 0):
            # See the module docstring: the image kernel works around this with a
            # cu124 torch reinstall, which costs ~5 minutes. Not worth it on a
            # fallback path that only runs when ZeroGPU is already exhausted -- fail
            # fast and let the caller report it.
            raise RuntimeError(
                "Kaggle gave this kernel a P100 (sm_60), which its preinstalled "
                "torch has no kernels for; asked for NvidiaTeslaT4. Retry later.")

        log("installing diffusers/transformers...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "diffusers", "transformers", "accelerate", "safetensors", "imageio",
             "imageio-ffmpeg", "sentencepiece", "requests"],
            check=True)

        import requests
        from PIL import Image
        from diffusers import LTXConditionPipeline
        from diffusers.utils import export_to_video

        log(f"fetching source image {image_url[:90]}")
        r = requests.get(image_url, timeout=60)
        r.raise_for_status()
        src = WORK / "input.jpg"
        src.write_bytes(r.content)
        image = Image.open(src).convert("RGB").resize((WIDTH, HEIGHT))

        log(f"loading {MODEL_ID} (fp16)...")
        pipe = LTXConditionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        # Offload rather than .to("cuda"): 16GB is enough without it for a 2B model,
        # but offload costs little here (the run is download-bound, not compute-bound)
        # and removes OOM as a failure mode if Kaggle ever hands us a smaller card.
        pipe.enable_model_cpu_offload()

        num_frames = int(length_s * FPS)
        num_frames = num_frames - (num_frames % 8) + 1  # LTX wants 8n+1 frames
        log(f"generating {num_frames} frames at {WIDTH}x{HEIGHT}...")
        frames = pipe(
            image=image, prompt=prompt, negative_prompt=negative_prompt,
            width=WIDTH, height=HEIGHT, num_frames=num_frames,
            num_inference_steps=int(payload.get("steps") or 8),
        ).frames[0]

        export_to_video(frames, str(OUT_VIDEO), fps=FPS)
        write_status("done", True, {"video": OUT_VIDEO.name,
                                    "frames": num_frames, "fps": FPS})
        log(f"wrote {OUT_VIDEO} ({OUT_VIDEO.stat().st_size // 1024}KB)")
    except Exception as e:
        import traceback
        # Full traceback into status.json, not just str(e) -- the image kernel's own
        # history is a run of failures whose real cause was only ever visible here,
        # buried under pip warnings in the kernel's stdout log.
        write_status("failed", False, {"error": str(e),
                                       "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
