"""ADetailer-style post-process: detect faces/hands in a generated image and
re-render just those regions at higher fidelity, blending the result back in.

Fast 4-8 step LCM generation is what makes this niche's throughput possible on a
free CPU/GPU runner, but it's also exactly why hands and faces come out wrong more
often than a normal 25-50 step render would -- there's no room for the model to
self-correct at that step count. Real CivitAI checkpoint creators use this exact
technique themselves -- seen directly in harvested showcase metadata this session:
"ADetailer model": "face_yolov8n.pt", "Face restoration": "CodeFormer". This is that
same idea, built with what's already in this pipeline: no second generation backend,
no new checkpoint load -- AutoPipelineForInpainting.from_pipe() reuses the already-
loaded pipeline in memory, same weights, same LCM scheduler, same fused LoRA
(verified live: both carry over automatically).

Detector: Bingsu/adetailer's own YOLOv8 face/hand models (Apache-2.0, free, a few MB
each) -- the same weights ADetailer itself ships, downloaded once via huggingface_hub
and cached like everything else HF-hosted in this pipeline. That YOLO detector only
finds WHERE a face/hand is; what actually happens inside that region differs by
kind. Faces get a plain text-prompted inpaint (below). Hands are handled by
hand_pose.py instead -- text prompts like "five fingers" cannot reliably enforce
finger topology (live-tested: failed on roughly half of real detected hands), so
hands get real structural guidance from a MediaPipe-landmarks-driven ControlNet pose
skeleton. See that module's docstring for the live evidence.

Inpainting runs on a small CROPPED canvas, not the full frame, and is resized back
into place afterward -- the same crop/inpaint/paste-back approach ADetailer itself
uses. sdgen.py's own timing notes put a full 512x896 frame at 61-125s/image on a GH
Actions CPU runner; inpainting the full frame again for every region would come
close to doubling or tripling that per image, which the workflow's timeout budget
has no room for. A small square canvas (REFINE_INPAINT_SIZE, default 384) is a
fraction of those pixels regardless of how big the original frame is.
"""
import os

from PIL import Image, ImageDraw, ImageFilter

import clip_encode
import hand_pose

REFINE_ENABLED = os.environ.get("REFINE_FACES", "1").strip().lower() not in ("0", "false", "no")
# Hand detection is the less mature of the two models (0.86 confidence for a clean
# face hit vs 0.4-0.5 for hands even when a hand IS in frame) -- a live check at 0.4
# caught a false positive (shoulder/hair mistaken for a hand at 0.44) on a real
# generated image. 0.5 is a deliberate compromise, not a measured optimum; there's no
# larger labelled sample to tune against yet.
MIN_CONFIDENCE = float(os.environ.get("REFINE_MIN_CONFIDENCE", "0.5"))
# How far each detected box is expanded on every side before cropping -- a live check
# found the raw YOLO box alone cropped too tight, right at the hairline/jaw, leaving
# a visible seam; padding gives the inpaint room to blend naturally.
PAD_FRACTION = 0.3
# Hands are handled entirely separately now -- see hand_pose.py's module docstring.
# Text-prompt-only inpainting (what this dict still governs, for faces) cannot
# reliably enforce finger topology; hand_pose.py uses MediaPipe landmarks + a
# ControlNet pose skeleton for real structural guidance instead, with its own
# live-tuned strength/steps (hand_pose.STRENGTH/STEPS), not these.
STRENGTH_BY_KIND = {"face": 0.4}
DEFAULT_STRENGTH = float(os.environ.get("REFINE_STRENGTH", "0.4"))
# Extra wording appended to the base prompt/negative for this region only -- the base
# prompt is about pose/outfit/location, not anatomy, so it does nothing to steer the
# inpaint toward a correct hand specifically. Verified live: an inpaint pass using
# this exact hand wording visibly fixed indistinct/fused-looking fingers on a real
# generated image, cleanly, with no seam.
PROMPT_CUE_BY_KIND = {
    "face": "detailed face, sharp focus, symmetrical natural features",
    "hand": "detailed hand, five fingers, natural hand anatomy",
}
NEGATIVE_CUE_BY_KIND = {
    "hand": "extra fingers, fused fingers, missing fingers, malformed hand, "
           "extra hand, mutated hand, bad hand anatomy",
}
# Kinds where the base prompt is REPLACED rather than appended to. Live-caught bug:
# a "holding phone" base prompt (true of one hand in the shot) bled into the OTHER,
# unrelated hand's inpaint and hallucinated a phone-shaped object into a hand that
# was just resting -- the base prompt's action/scene wording is specific to the
# whole image, not to any one region, and a local crop has no way to know it only
# applies elsewhere. Faces don't share this failure mode (no live case of a face
# inpaint hallucinating an unrelated object from scene text), so they keep the
# base-prompt context; hands get a minimal, anatomy-only prompt instead.
PROMPT_REPLACE_KINDS = {"hand"}
# At most this many regions per kind get refined, highest confidence first. Was
# dropped to 1 (just the best hand, not both) after a real run got killed by the
# workflow's old 90min timeout -- but that timeout is now 150min (see autopilot.yml),
# and a real post-timeout-fix run finished a full round (10 images, face+hand) in
# ~38min, comfortably inside the new budget even doubled. Restored to 2: a real
# posted image was reported with a hideous hand shortly after this was dropped to 1
# -- refining only the single best-confidence hand left the other one (commonly
# also in frame -- a hand on a hip, the other holding something) untouched.
MAX_REGIONS_PER_KIND = int(os.environ.get("REFINE_MAX_REGIONS_PER_KIND", "2"))
# Cap on the LONGER side of the canvas the crop is resized onto before inpainting,
# then resized back from afterward -- small on purpose, see module docstring for the
# CPU-time reasoning. NOT a forced square: a hand crop is usually tall/narrow (e.g.
# 102x157 seen in a real run), and force-resizing that into a square canvas squishes
# it horizontally before inpainting and stretches it back after, distorting finger
# proportions on every single hand refine -- live-suspected as a real contributor to
# hands looking worse after "refinement", not just failing to improve them. See
# _canvas_size() below, which preserves the crop's own aspect ratio instead.
INPAINT_SIZE = int(os.environ.get("REFINE_INPAINT_SIZE", "384"))
# Pixels of the crop's own edge left outside the feathered mask, so the paste-back
# blends into the surrounding, un-refined image instead of showing a hard rectangle.
FEATHER_MARGIN = 12
FEATHER_BLUR = 15

_detectors = {}
_inpaint_pipes = {}


def log(msg): print(f"[refine] {msg}", flush=True)


def _detector(kind):
    """kind: "face" or "hand". Cached per process -- the weight file is tiny but
    there's no reason to reload it for every image in a batch."""
    if kind in _detectors:
        return _detectors[kind]
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    path = hf_hub_download("Bingsu/adetailer", f"{kind}_yolov8n.pt")
    _detectors[kind] = YOLO(path)
    return _detectors[kind]


def _detect_boxes(image, kind):
    model = _detector(kind)
    results = model(image, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if conf >= MIN_CONFIDENCE:
                boxes.append((box, conf))
    return boxes


def _canvas_size(cw, ch):
    """Aspect-ratio-preserving inpaint canvas dims for a `cw`x`ch` crop: scaled so
    the LONGER side hits INPAINT_SIZE, both dims rounded to a multiple of 8 (SD's VAE
    requirement). NOT a forced square -- see INPAINT_SIZE's comment for why that
    distorted proportions on every hand refine."""
    scale = INPAINT_SIZE / max(cw, ch)
    w = max(8, round(cw * scale / 8) * 8)
    h = max(8, round(ch * scale / 8) * 8)
    return w, h


def _crop_box(size, box):
    """Detected box, padded, clamped to the image, and rounded to whole pixels."""
    W, H = size
    x0, y0, x1, y1 = box
    pad_x, pad_y = (x1 - x0) * PAD_FRACTION, (y1 - y0) * PAD_FRACTION
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(W, x1 + pad_x), min(H, y1 + pad_y)
    x0, y0, x1, y1 = int(x0), int(y0), int(round(x1)), int(round(y1))
    x1, y1 = max(x1, x0 + 8), max(y1, y0 + 8)
    return x0, y0, x1, y1


def _feathered_mask(size, margin=FEATHER_MARGIN, blur=FEATHER_BLUR):
    """A near-full white mask inset by `margin` and blurred, so a paste using it
    blends into the surrounding image instead of leaving a visible seam.

    margin/blur are capped to a fraction of the mask's own smaller side (not just
    passed through) -- the fixed FEATHER_MARGIN/FEATHER_BLUR defaults were tuned
    against the INPAINT_SIZE=384 canvas, but the final paste-back mask is sized to
    the ACTUAL detected crop, which can be much smaller (a face far from camera in a
    full-body shot). Applying a 15px blur to e.g. a 64px mask blurs past the flat
    center entirely, leaving the whole region under-opacity everywhere including the
    middle -- caught by a unit test asserting the center pixel comes out fully
    refined, not partially blended with the original."""
    w, h = size
    cap = max(2, min(w, h) // 8)
    margin, blur = min(margin, cap), min(blur, cap)
    mask = Image.new("L", size, 0)
    m = min(margin, w // 2 - 1, h // 2 - 1)
    ImageDraw.Draw(mask).rectangle([m, m, w - m, h - m], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(blur))


def _inpaint_pipe_for(pipe):
    """AutoPipelineForInpainting.from_pipe() reuses the SAME loaded weights (no
    second checkpoint load, no second download) and preserves whatever scheduler and
    fused LoRA the base pipe already has."""
    key = id(pipe)
    if key in _inpaint_pipes:
        return _inpaint_pipes[key]
    from diffusers import AutoPipelineForInpainting
    inpaint = AutoPipelineForInpainting.from_pipe(pipe)
    _inpaint_pipes[key] = inpaint
    return inpaint


def refine(image, pipe, prompt, negative_prompt, steps, guidance, kinds=("face",), arch="sd15"):
    """Detect up to MAX_REGIONS_PER_KIND regions of each kind in `image` (highest
    confidence first), inpaint a small cropped canvas over each, and blend the
    result back at full resolution. Returns the original image unchanged if nothing
    was detected, detection/inpaint failed, or REFINE_FACES=0. Never raises -- a
    failed refinement pass falls back to the un-refined image rather than losing an
    otherwise-good generation over a post-process step.

    arch ("sd15"/"sdxl") is needed by clip_encode.encode() below -- a harvested
    reference prompt is often already near CLIP's 77-token limit on its own, and the
    face cue this appends (PROMPT_CUE_BY_KIND) landed right past it in a live check,
    silently dropped by a plain (un-chunked) prompt= string. Defaults to "sd15" since
    every built-in preset and most harvested checkpoints are; callers with an SDXL
    pipe must pass arch="sdxl" explicitly or the cue keeps getting truncated away."""
    if not REFINE_ENABLED:
        return image
    result = image
    for kind in kinds:
        try:
            boxes = _detect_boxes(result, kind)
        except Exception as e:
            log(f"{kind} detection failed, skipping ({type(e).__name__}: {str(e)[:100]})")
            continue
        if not boxes:
            continue
        is_hand = kind == "hand"
        # Built lazily, only once a region actually needs it -- skips the pipe
        # conversion entirely on an image with nothing to refine, and is still
        # cached (by pipe identity) across kinds/images that DO need it. Hands use a
        # separate ControlNet-attached pipe (hand_pose.py) for real structural
        # guidance -- see that module's docstring for why plain text-prompted
        # inpainting (what faces still use) isn't reliable enough for hands.
        inpaint_pipe = hand_pose.inpaint_pipe_for(pipe) if is_hand else _inpaint_pipe_for(pipe)
        strength = hand_pose.STRENGTH if is_hand else STRENGTH_BY_KIND.get(kind, DEFAULT_STRENGTH)
        region_steps = hand_pose.STEPS if is_hand else steps
        if kind in PROMPT_REPLACE_KINDS:
            region_prompt = PROMPT_CUE_BY_KIND.get(kind, prompt)
        else:
            region_prompt = ", ".join(
                p for p in (prompt, PROMPT_CUE_BY_KIND.get(kind)) if p)
        region_negative = ", ".join(
            p for p in (negative_prompt, NEGATIVE_CUE_BY_KIND.get(kind)) if p)
        encode_kwargs = clip_encode.encode(inpaint_pipe, arch, region_prompt, region_negative)
        top_boxes = sorted(boxes, key=lambda b: b[1], reverse=True)[:MAX_REGIONS_PER_KIND]
        for box, conf in top_boxes:
            x0, y0, x1, y1 = _crop_box(result.size, box)
            crop = result.crop((x0, y0, x1, y1))
            cw, ch = crop.size
            canvas_w, canvas_h = _canvas_size(cw, ch)
            canvas = crop.resize((canvas_w, canvas_h), Image.LANCZOS)
            canvas_mask = _feathered_mask((canvas_w, canvas_h))
            extra_kwargs = {}
            if is_hand:
                try:
                    control_image = hand_pose.skeleton_for(canvas)
                except Exception as e:
                    log(f"hand landmark detection failed, skipping region "
                       f"({type(e).__name__}: {str(e)[:100]})")
                    continue
                if control_image is None:
                    # No recognizable hand structure to guide toward -- live-
                    # confirmed this correlates with a hand malformed enough that
                    # blind inpainting made it WORSE, not better. Leave it for
                    # supervisor.py's anatomy_ok QA gate instead of guessing.
                    log(f"no hand landmarks found (confidence {conf:.2f}, crop "
                       f"{cw}x{ch}), skipping region")
                    continue
                extra_kwargs = {"control_image": control_image,
                               "controlnet_conditioning_scale": hand_pose.CONTROLNET_CONDITIONING_SCALE}
            try:
                refined_canvas = inpaint_pipe(
                    image=canvas, mask_image=canvas_mask,
                    num_inference_steps=region_steps, guidance_scale=guidance, strength=strength,
                    width=canvas_w, height=canvas_h,
                    **encode_kwargs, **extra_kwargs,
                ).images[0]
            except Exception as e:
                log(f"{kind} inpaint failed, keeping original region "
                   f"({type(e).__name__}: {str(e)[:100]})")
                continue
            refined_crop = refined_canvas.resize((cw, ch), Image.LANCZOS)
            paste_mask = _feathered_mask((cw, ch))
            result = result.copy()
            result.paste(refined_crop, (x0, y0), paste_mask)
            log(f"refined {kind} (confidence {conf:.2f}, crop {cw}x{ch}, "
               f"canvas {canvas_w}x{canvas_h}, steps {region_steps}, strength {strength})")
    return result
