"""Local, free, CPU text-to-image generation for the image-slideshow niches.

Runs entirely on the machine executing it: no API key, no per-image cost, no rate
limit beyond wall-clock time. The tradeoff for that is speed, so the model and settings
here are chosen specifically for a 2-vCPU GitHub Actions runner, not for quality ceiling.

Model choice: SD1.5-class checkpoints only, run through LCM (4 inference steps) rather
than the ~25-50 a checkpoint normally wants. An SDXL-class checkpoint (Segmind-Vega, an
SDXL distillation) measured at ~118s/image even on 16 desktop threads; the same LCM
setup on an SD1.5 checkpoint, throttled to 2 threads to match a GH Actions runner,
measured 61-125s/image depending on resolution and negative-prompt use. Multiple
checkpoints can be configured -- this is not tied to one model -- as long as each is
SD1.5-architecture so the same LCM-LoRA applies.

MODELS lists checkpoints known to work with this setup: popular CivitAI-lineage
realistic-portrait checkpoints mirrored to Hugging Face, each Apache/OpenRAIL licensed
for commercial use. civitai.py can pull real prompts that were paired with the same
checkpoint on CivitAI; when that is unreachable (CivitAI blocks some countries outright,
unrelated to any key) PROMPT_BANK below is the fallback so a network block costs
variety, never a run.
"""
import os, random, time
from pathlib import Path

import civitai
import clip_encode
import refine
import upscale

# Built-in presets, kept as a no-setup-required fallback. The primary path is
# CIVITAI_MODEL / a niche's "civitai_model" -- see load_civitai below -- which downloads
# and runs whatever checkpoint the operator names, not one of these three.
MODELS = {
    # key -> (HF repo id, needs a separate LCM-LoRA fused on top, CivitAI search term)
    "dreamshaper": ("SimianLuo/LCM_Dreamshaper_v7", None, "Dreamshaper"),
    "realistic-vision": ("SG161222/Realistic_Vision_V6.0_B1_noVAE",
                         "latent-consistency/lcm-lora-sdv1-5", "Realistic Vision"),
    "absolute-reality": ("digiplay/AbsoluteReality_v1.8.1",
                         "latent-consistency/lcm-lora-sdv1-5", "AbsoluteReality"),
}
DEFAULT_MODEL = os.environ.get("SD_MODEL", "dreamshaper")
# A CivitAI model spec (URL, bare id, or "modelId:versionId") takes over from the
# presets above when set -- this is the "use THIS model" path.
CIVITAI_MODEL = os.environ.get("CIVITAI_MODEL", "").strip()

LCM_LORA = {"sd15": "latent-consistency/lcm-lora-sdv1-5",
           "sdxl": "latent-consistency/lcm-lora-sdxl"}

WIDTH = int(os.environ.get("SD_WIDTH", "512"))
HEIGHT = int(os.environ.get("SD_HEIGHT", "896"))
# 4 was the original speed-first floor; LCM's own designed range is roughly 4-8 steps,
# so 6 stays inside that range while giving a bit more of the refinement pass that
# fixes hand/finger anatomy -- a live batch showed hand errors slipping past even a
# stricter negative prompt, and step count is the other documented lever for it.
STEPS = int(os.environ.get("SD_STEPS", "6"))
# LCM effectively ignores negative_prompt at guidance_scale=1.0 (no unconditional branch
# to steer against). 1.8 measured within ~5% of the 1.0 timing and makes negatives work.
GUIDANCE = float(os.environ.get("SD_GUIDANCE", "1.8"))

NEGATIVE_HARD = ("child, teen, minor, young girl, schoolgirl, loli, nude, topless, "
                 "exposed nipples, exposed genitals, explicit, nsfw, "
                 "celebrity likeness, real person, chinese, chinese woman, asian, "
                 "east asian, korean, japanese, asian woman")
NEGATIVE_QUALITY = ("cartoon, illustration, painting, anime, 3d render, deformed, "
                    "extra fingers, extra limbs, mutated hands, bad anatomy, blurry, "
                    "watermark, text, logo")
# Suppressing the specific way LCM output fails to look real, which is different from
# the generic "not a cartoon" list above. At 6 steps the model has no room to resolve
# skin into texture, so it settles on a smooth, evenly-lit, retouched-looking surface
# -- plastic, waxy, doll-like. None of those words appeared in any negative prompt
# this pipeline used, so nothing was pushing against the failure mode it actually has.
# Paired with imageslides.REALISM_CUE, which asks for the same thing from the positive
# side; a negative alone is the weaker of the two levers.
NEGATIVE_REALISM = ("plastic skin, waxy skin, airbrushed, smooth featureless skin, "
                    "poreless, doll, mannequin, uncanny, cgi, video game render, "
                    "overprocessed, oversaturated, overexposed, blown highlights, "
                    "heavy retouching, beauty filter")

_pipes = {}
_pipe_arch = {}  # cache_key -> "sd15"/"sdxl", needed to pick compel's SD1.5 vs SDXL API


def log(msg): print(f"[sdgen] {msg}", flush=True)


def unload_all():
    """Evict every cached base pipeline AND the inpaint/ControlNet pipes refine.py
    and hand_pose.py built from them (from_pipe() wraps the same weights in a new
    object, so dropping only sdgen's own cache still leaves those modules holding a
    live reference, keeping the GPU memory allocated).

    Call this before loading a genuinely DIFFERENT checkpoint. A real local GPU run
    OOM'd on round 2's very first image with a tiny 20MiB allocation -- round 1's
    checkpoint (base + face inpaint + hand ControlNet inpaint pipelines) was still
    fully resident when round 2's re-decide (imageslides.py, triggered on an
    all-rejected round) loaded a second, different checkpoint on top of it. CPU CI
    never hits this: system RAM has far more headroom than a GPU's VRAM."""
    import gc, torch
    _pipes.clear()
    _pipe_arch.clear()
    refine._inpaint_pipes.clear()
    refine.hand_pose._cn_inpaint_pipes.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _device():
    """CUDA when available (a local GPU box), CPU otherwise (every GH Actions runner --
    none of them have a GPU, so this always resolves to CPU there, unchanged from
    before). fp16 on GPU: roughly half the memory and time of fp32 for the same
    output, standard practice for SD1.5/SDXL inference; fp32 stays on CPU since fp16
    matmul is not accelerated there and can be numerically flaky on some CPU builds."""
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load(model_key):
    """Load and cache a preset pipeline for the process lifetime. Loading is a few
    seconds once the checkpoint is on disk; the download itself (a few GB) happens once
    per model and is left to Hugging Face's own on-disk cache."""
    if model_key in _pipes:
        return _pipes[model_key]
    if model_key not in MODELS:
        raise RuntimeError(f"unknown SD model {model_key!r}; choices: {list(MODELS)}")
    repo, lora, _ = MODELS[model_key]

    from diffusers import DiffusionPipeline, LCMScheduler
    device, dtype = _device()
    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if lora:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    log(f"{model_key} loaded in {time.time() - t0:.0f}s ({device})")
    _pipes[model_key] = pipe
    _pipe_arch[model_key] = "sd15"  # every built-in preset is SD1.5-class
    return pipe


#  from_single_file() infers a checkpoint's pipeline config (which components it has,
# their shapes) from the checkpoint's own weight structure when not told otherwise --
# a real run crashed on this exact gap: a genuine SD1.5 checkpoint (Vendo
# Photorealistic 2, confirmed live, no config= override) got its structure
# misidentified as lllyasviel/control_v11p_sd15_canny (a ControlNet, not a full
# pipeline), and diffusers then failed looking for a model_index.json that ControlNet
# repo was never going to have. Passing an explicit, known-good base config repo
# per architecture sidesteps the guess entirely -- these are the standard base
# repos non-first-party SD1.5/SDXL checkpoints are conventionally fine-tuned from.
_BASE_CONFIG = {"sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
               "sdxl": "stabilityai/stable-diffusion-xl-base-1.0"}


def _load_civitai(spec):
    """Download (once, cached by version id) and load an operator-chosen checkpoint.
    SD 1.5 and SDXL are both supported; the matching LCM-LoRA is picked automatically
    since a mismatched one silently produces garbage rather than erroring."""
    if spec in _pipes:
        return _pipes[spec]

    from diffusers import LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
    resolved = civitai.resolve_and_download(spec)
    pipeline_cls = StableDiffusionXLPipeline if resolved["arch"] == "sdxl" else StableDiffusionPipeline
    device, dtype = _device()

    t0 = time.time()
    pipe = pipeline_cls.from_single_file(
        str(resolved["path"]), torch_dtype=dtype, safety_checker=None,
        config=_BASE_CONFIG[resolved["arch"]])
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA[resolved["arch"]])
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    log(f"{resolved['name']!r} loaded in {time.time() - t0:.0f}s ({device})")
    _pipes[spec] = pipe
    _pipe_arch[spec] = resolved["arch"]
    return pipe


def _encode(pipe, arch, prompt, negative):
    """Thin re-export of clip_encode.encode() -- kept as a module-level name here
    since it's the one existing tests patch, and sdgen.py's own call site already
    reads naturally as "this module's own encode step". The actual implementation
    lives in clip_encode.py now, shared with refine.py's inpaint calls, which had
    the exact same 77-token truncation problem this solves (see that module)."""
    return clip_encode.encode(pipe, arch, prompt, negative)


def generate_image(prompt, dest, model_key=None, negative_prompt="", seed=None,
                   civitai_model=None, width=None, height=None, steps=None, guidance=None):
    """One image, saved to `dest`. Returns the elapsed seconds.

    civitai_model (or the CIVITAI_MODEL env var) takes precedence over model_key: an
    operator-named checkpoint always wins over the built-in presets.

    width/height/steps/guidance override the module defaults for this call only --
    imageslides.py uses this to adopt a harvested checkpoint's own posted generation
    settings (the ones that earned that showcase image its engagement) when they're
    within a sane range, instead of always using the same fixed values regardless of
    what the checkpoint's own creator demonstrated worked."""
    civitai_model = civitai_model or CIVITAI_MODEL
    if civitai_model:
        pipe = _load_civitai(civitai_model)
        cache_key = civitai_model
    else:
        cache_key = model_key or DEFAULT_MODEL
        pipe = _load(cache_key)
    import torch
    device, _ = _device()
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    neg = ", ".join(p for p in (NEGATIVE_HARD, NEGATIVE_QUALITY, NEGATIVE_REALISM,
                                negative_prompt) if p)
    encode_kwargs = _encode(pipe, _pipe_arch[cache_key], prompt, neg)
    t0 = time.time()
    image = pipe(
        num_inference_steps=steps or STEPS,
        guidance_scale=guidance if guidance is not None else GUIDANCE,
        width=width or WIDTH, height=height or HEIGHT, generator=generator,
        **encode_kwargs,
    ).images[0]
    # ADetailer-style pass: LCM's 4-8 steps leave no room for the model to fix its
    # own face/hand mistakes the way a normal 25-50 step run would, so re-render just
    # those regions at higher fidelity via the same already-loaded pipe. Cheap to
    # skip on failure -- refine() never raises, falls back to the un-refined image.
    image = refine.refine(image, pipe, prompt, neg,
                          steps=steps or STEPS, guidance=guidance if guidance is not None else GUIDANCE,
                          kinds=("face", "hand"), arch=_pipe_arch[cache_key])
    # Sharper/higher-res final image than the base render alone -- TikTok favors
    # higher resolution photo posts, up to a point it does NOT document: a live batch
    # at 1152x1728 (a checkpoint's own adopted resolution, see imageslides.
    # _adopted_settings, upscaled 2x) got every image rejected with
    # fail_reason=picture_size_check_failed, and re-hosting the SAME images resized to
    # 1024x1536 with nothing else changed reached the inbox clean -- confirmed live,
    # isolating the dimensions themselves, not content. TikTok's own marketing copy
    # for photo posts names 1080x1920 as the canvas; clamp to that, matching a
    # confirmed-good size instead of guessing where the real undocumented ceiling is.
    image = upscale.upscale(image)
    from PIL import Image
    image.thumbnail((1080, 1920), Image.LANCZOS)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # JPEG, not PNG -- a live check found TikTok's Content Posting API rejecting
    # PNG-sourced photo posts with file_format_check_failed; JPEG is what it actually
    # expects. SD output has no alpha channel, but converting explicitly guards
    # against a checkpoint/pipeline that ever returns RGBA (JPEG has no alpha, and
    # PIL raises rather than silently dropping it).
    image.convert("RGB").save(dest, "JPEG", quality=95)
    dt = time.time() - t0
    log(f"{cache_key}: {dt:.0f}s -> {dest.name}")
    return dt


def generate_batch(prompts, workdir, model_key=None, negative_prompts=None, civitai_model=None,
                   width=None, height=None, steps=None, guidance=None):
    """Sequential generation for a set of prompts. Sequential, not parallel, on purpose:
    a GH Actions job already has just 2 vCPUs, so parallel workers would contend for the
    same cores rather than add throughput.

    Never ran on a GPU box before a real local run OOM'd on image 3 of 10, every
    image after that failing too, even at 2MiB allocations -- PyTorch's caching
    allocator holds onto freed blocks (crop tensors from refine.py's per-region
    inpaint calls, mainly) instead of returning them, and with three separate
    from_pipe() pipelines sharing the same weights (base, face inpaint, hand
    ControlNet inpaint) plus YOLO/mediapipe running per region, that adds up across
    a batch. CPU CI never surfaced this since CPU has no equivalent allocator
    pressure. Clearing the cache between images keeps steady-state VRAM flat instead
    of growing every image; a no-op on CPU."""
    negative_prompts = negative_prompts or [""] * len(prompts)
    paths = []
    for i, (prompt, neg) in enumerate(zip(prompts, negative_prompts)):
        dest = Path(workdir) / f"sd_{i}.jpg"
        try:
            generate_image(prompt, dest, model_key=model_key, negative_prompt=neg,
                           seed=random.randint(1, 10**9), civitai_model=civitai_model,
                           width=width, height=height, steps=steps, guidance=guidance)
            paths.append(dest)
        except Exception as e:
            log(f"image {i + 1} failed ({type(e).__name__}: {str(e)[:100]}); skipping it")
        finally:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not paths:
        raise RuntimeError("no images were generated")
    return paths
