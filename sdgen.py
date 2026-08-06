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
STEPS = int(os.environ.get("SD_STEPS", "4"))
# LCM effectively ignores negative_prompt at guidance_scale=1.0 (no unconditional branch
# to steer against). 1.8 measured within ~5% of the 1.0 timing and makes negatives work.
GUIDANCE = float(os.environ.get("SD_GUIDANCE", "1.8"))

NEGATIVE_HARD = ("child, teen, minor, young girl, schoolgirl, loli, nude, topless, "
                 "exposed nipples, exposed genitals, explicit, nsfw, "
                 "celebrity likeness, real person")
NEGATIVE_QUALITY = ("cartoon, illustration, painting, anime, 3d render, deformed, "
                    "extra fingers, extra limbs, mutated hands, bad anatomy, blurry, "
                    "watermark, text, logo")

_pipes = {}
_pipe_arch = {}  # cache_key -> "sd15"/"sdxl", needed to pick compel's SD1.5 vs SDXL API


def log(msg): print(f"[sdgen] {msg}", flush=True)


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
        str(resolved["path"]), torch_dtype=dtype, safety_checker=None)
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
    """The full prompt, not truncated at 77 tokens. compel chunks a long prompt into
    multiple 77-token windows, encodes each through CLIP separately, and concatenates
    the embeddings -- the same technique Automatic1111 uses for "long prompts". The
    flavour/detail text is the actual reason to harvest a real prompt from CivitAI
    instead of writing one by hand, so silently dropping its tail at the CLIP limit
    defeats the point. Falls back to a plain (truncated) call only if compel is not
    installed, so a missing optional dependency degrades instead of breaking generation."""
    try:
        from compel import Compel, ReturnedEmbeddingsType
    except ImportError:
        log("compel not installed; falling back to a 77-token-truncated prompt")
        return {"prompt": prompt, "negative_prompt": negative}

    if arch == "sdxl":
        compel_proc = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        cond, pooled = compel_proc(prompt)
        neg_cond, neg_pooled = compel_proc(negative)
        cond, neg_cond = compel_proc.pad_conditioning_tensors_to_same_length([cond, neg_cond])
        return {"prompt_embeds": cond, "pooled_prompt_embeds": pooled,
               "negative_prompt_embeds": neg_cond, "negative_pooled_prompt_embeds": neg_pooled}

    compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder,
                         truncate_long_prompts=False)
    cond = compel_proc(prompt)
    neg_cond = compel_proc(negative)
    cond, neg_cond = compel_proc.pad_conditioning_tensors_to_same_length([cond, neg_cond])
    return {"prompt_embeds": cond, "negative_prompt_embeds": neg_cond}


def generate_image(prompt, dest, model_key=None, negative_prompt="", seed=None,
                   civitai_model=None):
    """One image, saved to `dest`. Returns the elapsed seconds.

    civitai_model (or the CIVITAI_MODEL env var) takes precedence over model_key: an
    operator-named checkpoint always wins over the built-in presets."""
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
    neg = ", ".join(p for p in (NEGATIVE_HARD, NEGATIVE_QUALITY, negative_prompt) if p)
    encode_kwargs = _encode(pipe, _pipe_arch[cache_key], prompt, neg)
    t0 = time.time()
    image = pipe(
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        width=WIDTH, height=HEIGHT, generator=generator,
        **encode_kwargs,
    ).images[0]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    dt = time.time() - t0
    log(f"{model_key}: {dt:.0f}s -> {dest.name}")
    return dt


def generate_batch(prompts, workdir, model_key=None, negative_prompts=None, civitai_model=None):
    """Sequential generation for a set of prompts. Sequential, not parallel, on purpose:
    a GH Actions job already has just 2 vCPUs, so parallel workers would contend for the
    same cores rather than add throughput."""
    negative_prompts = negative_prompts or [""] * len(prompts)
    paths = []
    for i, (prompt, neg) in enumerate(zip(prompts, negative_prompts)):
        dest = Path(workdir) / f"sd_{i}.png"
        try:
            generate_image(prompt, dest, model_key=model_key, negative_prompt=neg,
                           seed=random.randint(1, 10**9), civitai_model=civitai_model)
            paths.append(dest)
        except Exception as e:
            log(f"image {i + 1} failed ({type(e).__name__}: {str(e)[:100]}); skipping it")
    if not paths:
        raise RuntimeError("no images were generated")
    return paths
