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
and cached like everything else HF-hosted in this pipeline.

Inpainting runs on a small CROPPED canvas, not the full frame, and is resized back
into place afterward -- the same crop/inpaint/paste-back approach ADetailer itself
uses. sdgen.py's own timing notes put a full 512x896 frame at 61-125s/image on a GH
Actions CPU runner; inpainting the full frame again for every region would come
close to doubling or tripling that per image, which the workflow's 90min timeout
budget has no room for. A small square canvas (REFINE_INPAINT_SIZE, default 384) is
a fraction of those pixels regardless of how big the original frame is.
"""
import os

from PIL import Image, ImageDraw, ImageFilter

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
STRENGTH = float(os.environ.get("REFINE_STRENGTH", "0.4"))
# Square canvas the crop is resized onto before inpainting, then resized back from
# afterward. Small on purpose -- see module docstring for the CPU-time reasoning.
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


def refine(image, pipe, prompt, negative_prompt, steps, guidance, kinds=("face",)):
    """Detect the highest-confidence region of each kind in `image`, inpaint a small
    cropped canvas over it, and blend the result back at full resolution. Returns the
    original image unchanged if nothing was detected, detection/inpaint failed, or
    REFINE_FACES=0. Never raises -- a failed refinement pass falls back to the
    un-refined image rather than losing an otherwise-good generation over a
    post-process step."""
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
        # Built lazily, only once a region actually needs it -- skips the pipe
        # conversion entirely on an image with nothing to refine, and is still
        # cached (by pipe identity) across kinds/images that DO need it.
        inpaint_pipe = _inpaint_pipe_for(pipe)
        box, conf = max(boxes, key=lambda b: b[1])
        x0, y0, x1, y1 = _crop_box(result.size, box)
        crop = result.crop((x0, y0, x1, y1))
        cw, ch = crop.size
        canvas = crop.resize((INPAINT_SIZE, INPAINT_SIZE), Image.LANCZOS)
        canvas_mask = _feathered_mask((INPAINT_SIZE, INPAINT_SIZE))
        try:
            refined_canvas = inpaint_pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                image=canvas, mask_image=canvas_mask,
                num_inference_steps=steps, guidance_scale=guidance, strength=STRENGTH,
                width=INPAINT_SIZE, height=INPAINT_SIZE,
            ).images[0]
        except Exception as e:
            log(f"{kind} inpaint failed, keeping original region "
               f"({type(e).__name__}: {str(e)[:100]})")
            continue
        refined_crop = refined_canvas.resize((cw, ch), Image.LANCZOS)
        paste_mask = _feathered_mask((cw, ch))
        result = result.copy()
        result.paste(refined_crop, (x0, y0), paste_mask)
        log(f"refined {kind} (confidence {conf:.2f}, crop {cw}x{ch})")
    return result
