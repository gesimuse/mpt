"""Super-resolution pass, run after generation (and refine.py's face/hand touch-up)
and before saving. TikTok favors higher-resolution photo posts for reach, and this
pipeline's base render tops out at 512x896 (sdgen.py's WIDTH/HEIGHT) -- CPU time on a
GH Actions runner is the limiting factor there, not model quality ceiling.

Model: eugenesiow/pan-bam (PAN, "Pixel Attention Network"), a small pure-PyTorch
super-resolution model via the `super-image` package. The obvious first choice,
Real-ESRGAN, was tried and rejected: its `realesrgan` package depends on `basicsr`,
whose own setup.py fails to build on this Python version (`KeyError: '__version__'`
in basicsr's version-detection code, unrelated to anything in this repo) -- confirmed
live, not a hypothetical. `super-image` has no such dependency and installs clean.

Live-verified: 2x upscale of a real 512x896 generated image (out/aibeauty-.../sd_2.jpg)
took 3.3s on a desktop CPU core and produced a visibly sharper 1024x1792 result with
no artifacts, colors intact.
"""
import os

import numpy as np
from PIL import Image

UPSCALE_ENABLED = os.environ.get("UPSCALE_ENABLED", "1").strip().lower() not in ("0", "false", "no")
# eugenesiow/pan-bam ships 2x/3x/4x pretrained weights; 2x is the one actually
# verified live above.
UPSCALE_SCALE = int(os.environ.get("UPSCALE_SCALE", "2"))

_models = {}


def log(msg): print(f"[upscale] {msg}", flush=True)


def _model(scale):
    if scale in _models:
        return _models[scale]
    from super_image import PanModel
    _models[scale] = PanModel.from_pretrained("eugenesiow/pan-bam", scale=scale)
    return _models[scale]


def upscale(image, scale=None):
    """Returns a `scale`x sharper version of `image`. Falls back to the original,
    unresized image on any failure or when UPSCALE_ENABLED=0 -- an upscale pass is a
    quality bump, not something worth losing an otherwise-good generation over."""
    if not UPSCALE_ENABLED:
        return image
    scale = scale or UPSCALE_SCALE
    try:
        from super_image import ImageLoader
        model = _model(scale)
        inputs = ImageLoader.load_image(image.convert("RGB"))
        preds = model(inputs)
        # super_image.ImageLoader only offers a cv2-based save-to-file helper; this
        # mirrors its own tensor -> array math (see _process_image_to_save) but
        # returns a PIL Image directly, without an opencv round-trip.
        arr = preds.data.cpu().numpy()[0].transpose((1, 2, 0)) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr, mode="RGB")
        log(f"{image.size[0]}x{image.size[1]} -> {result.size[0]}x{result.size[1]}")
        return result
    except Exception as e:
        log(f"upscale failed, keeping original resolution ({type(e).__name__}: {str(e)[:100]})")
        return image
