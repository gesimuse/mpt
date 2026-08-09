"""Pose-guided hand inpainting: MediaPipe hand landmarks -> OpenPose-format hand
skeleton -> ControlNet-guided inpaint, instead of hoping text prompts alone produce
correct finger topology.

Why this exists: refine.py's plain text-prompted inpaint (no structural guidance)
was live-tested across 8 real detected hands from real generated images and failed
on 4 of them -- a patch of visibly mismatched skin tone, a hand replaced by a
hallucinated psychedelic pattern, a phone turned into an unrecognizable purple
shape, a hallucinated eye where a wrist met an arm. A text prompt like "five
fingers" cannot reliably enforce topology; nothing about it tells the model WHERE
each finger goes, and a diffusion model has no real counting mechanism at 4-10
steps. MediaPipe's 21-point hand landmarks use the exact same topology/indexing as
OpenPose's hand skeleton (wrist=0, thumb=1-4, index=5-8, middle=9-12, ring=13-16,
pinky=17-20), so a real detected hand shape can be rendered as a skeleton and fed to
lllyasviel/control_v11p_sd15_openpose as an actual structural constraint, not a
hope.

Fixable range: only hands MediaPipe can find landmarks on. A hand malformed enough
to have no recognizable structure (fused into a formless blob) won't have any
landmarks to extract -- live-confirmed, 0 hands found on a real severely-malformed
case that even the YOLO hand detector couldn't detect either. That is the CORRECT
outcome here, not a gap to work around: the plain-inpaint approach applied to that
same class of image made it worse (turned it into a smudge, live-reproduced
separately). A hand with no findable structure is what supervisor.py's anatomy_ok
QA gate is for, not something to guess a fix for.

Live-tuned strength: controlnet_conditioning_scale governs the hand's own shape;
`strength` governs how much of everything ELSE in the crop (a phone, fabric, skin
around the hand) gets regenerated alongside it, since the pose skeleton only
constrains the hand pixels. 0.6 (refine.py's plain-inpaint value, tried here first)
reliably broke non-hand content in the same crop -- all 4 real failures above came
from that setting. 0.35 fixed every one of those specific failures on re-test, then
was confirmed clean again on a separate 3-image batch that had never been used
while tuning (11/11 clean total across both rounds, plus correct skip-on-no-
landmarks behavior on two separate unrelated crops).

mediapipe's `solutions` API (what controlnet_aux's own hand-skeleton-drawing code
expects) was removed in current mediapipe releases in favor of a new Tasks API --
stubbed with a MagicMock before importing controlnet_aux so its unrelated, unused
mediapipe_face submodule doesn't crash the whole package import at import time. The
actual hand landmark detection here uses the new Tasks API directly, untouched by
that stub.

Second real failure mode, found the same way (live testing, not guessing): even at
the tuned strength=0.35, a real production run still produced a hand replaced by
chaotic rainbow/static noise. Root-caused by elimination, not assumption -- tried
each candidate cause in turn against the exact failing crop: a garbled/tangled
MediaPipe skeleton was the first suspect (visually confirmed on this specific case,
compared side by side against a clean skeleton from the same image), but neither a
stricter landmark-confidence threshold (the bad detection scored 0.96, higher than
many good ones) nor several proposed geometric plausibility checks (fingertip
angular spread, edge self-intersection count, per-finger path straightness)
separated good skeletons from bad ones cleanly across the available samples --
false positives on legitimate closed/curled-finger poses, false negatives on
actually-tangled ones. Lowering controlnet_conditioning_scale (0.5, 0.3) to make the
model less rigidly bound to the skeleton did NOT fix it. Running the exact same crop
through plain inpainting with NO ControlNet at all still produced the same category
of noise, which ruled out the skeleton/ControlNet as the cause entirely. Brightness/
contrast of the input crop was also checked and ruled out (the clean examples span a
wider brightness range than the one confirmed-bad case).

What actually separates them, measured directly on the OUTPUT rather than predicted
from the input: this failure mode has a distinct visual signature (chaotic
per-pixel color-channel divergence -- literally the "rainbow static" look) that
normal skin/fabric output doesn't have, however it was caused. _looks_glitched()
below measures that directly and rejects the result if it's present, falling back
to the un-refined region -- the same "leave it alone rather than guess" principle
already used for the no-landmarks case, just applied after generation instead of
before. Threshold tuned against a small live sample (4 confirmed-clean canvas-sized
outputs scoring 3.1-4.6; the confirmed-bad production-default case scoring 6.05) --
real margin, but a small sample; this is a safety net for the failure pattern
actually observed, not a proof no other failure mode exists.
"""
import os
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

CONTROLNET_ID = "lllyasviel/control_v11p_sd15_openpose"
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
LANDMARKER_PATH = Path.home() / ".cache" / "mediapipe-models" / "hand_landmarker.task"
MIN_LANDMARK_CONFIDENCE = float(os.environ.get("REFINE_HAND_LANDMARK_CONFIDENCE", "0.15"))
# Live-tuned, see module docstring for the specific failures this fixed.
STRENGTH = float(os.environ.get("REFINE_HAND_STRENGTH", "0.35"))
CONTROLNET_CONDITIONING_SCALE = float(os.environ.get("REFINE_HAND_CONTROLNET_SCALE", "1.0"))
STEPS = int(os.environ.get("REFINE_HAND_STEPS", "10"))
# Live-tuned, see module docstring's "Second real failure mode" section -- margin
# between 4 confirmed-clean outputs (3.1-4.6) and the confirmed-bad one (6.05).
GLITCH_THRESHOLD = float(os.environ.get("REFINE_HAND_GLITCH_THRESHOLD", "5.5"))

_controlnet = None
_landmark_detector = None
_cn_inpaint_pipes = {}


def log(msg): print(f"[hand_pose] {msg}", flush=True)


def _looks_glitched(image):
    """True if `image` has the chaotic per-pixel color-channel divergence ("rainbow
    static") signature of the failure mode this module's docstring describes --
    measured directly on the output, not predicted from the input (nothing tried on
    the input side separated good from bad reliably). Mean absolute pixel-to-pixel
    change in the R-G and G-B difference channels: real skin/fabric has smoothly
    correlated color channels (low value here); this failure mode doesn't."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    rg, gb = arr[..., 0] - arr[..., 1], arr[..., 1] - arr[..., 2]
    noise = (np.abs(np.diff(rg, axis=0)).mean() + np.abs(np.diff(rg, axis=1)).mean()
            + np.abs(np.diff(gb, axis=0)).mean() + np.abs(np.diff(gb, axis=1)).mean())
    return noise > GLITCH_THRESHOLD


def _ensure_landmarker_downloaded():
    if LANDMARKER_PATH.exists():
        return
    import requests
    LANDMARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(LANDMARKER_URL, timeout=60)
    r.raise_for_status()
    LANDMARKER_PATH.write_bytes(r.content)


def _landmarker():
    global _landmark_detector
    if _landmark_detector is not None:
        return _landmark_detector
    _ensure_landmarker_downloaded()
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    base_options = mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2,
        min_hand_detection_confidence=MIN_LANDMARK_CONFIDENCE)
    _landmark_detector = mp_vision.HandLandmarker.create_from_options(options)
    return _landmark_detector


def _op_util():
    """controlnet_aux's own OpenPose hand-skeleton renderer -- see module docstring
    for why mediapipe.solutions needs stubbing first."""
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        mp.solutions = mock.MagicMock()
    from controlnet_aux.open_pose import body, util
    return util, body


def skeleton_for(canvas):
    """A hand-pose control image for `canvas` (a PIL Image), or None if no hand
    landmarks were found -- meaning this region has no recognizable hand structure
    to guide toward (see module docstring: that's a real QA-gate case, not
    something to inpaint a guess for)."""
    import mediapipe as mp
    detector = _landmarker()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(canvas.convert("RGB")))
    result = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None
    util, body = _op_util()
    lm = result.hand_landmarks[0]
    keypoints = [body.Keypoint(x=p.x, y=p.y, score=1.0, id=i) for i, p in enumerate(lm)]
    w, h = canvas.size
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    return Image.fromarray(util.draw_handpose(blank, keypoints))


def _controlnet_model():
    global _controlnet
    if _controlnet is not None:
        return _controlnet
    import torch
    from diffusers import ControlNetModel
    _controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float32)
    return _controlnet


def inpaint_pipe_for(pipe):
    """StableDiffusionControlNetInpaintPipeline.from_pipe() reuses the same loaded
    weights (no second checkpoint load) the same way refine.py's plain
    AutoPipelineForInpainting.from_pipe() already does, just with a ControlNet
    attached for pose guidance. Cached by pipe identity, same pattern as refine.py.

    Live-caught bug: ControlNetModel.from_pretrained() loads onto CPU by default,
    while `pipe` itself may already be on GPU (sdgen.py moves it there when one's
    available) -- constructing the combined pipeline without matching devices raised
    "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should
    be the same" on the very first real end-to-end run. Moved to `pipe`'s own
    device/dtype explicitly; .to() is a no-op if it's already there."""
    key = id(pipe)
    if key in _cn_inpaint_pipes:
        return _cn_inpaint_pipes[key]
    from diffusers import StableDiffusionControlNetInpaintPipeline
    controlnet = _controlnet_model().to(pipe.device, dtype=pipe.unet.dtype)
    cn_pipe = StableDiffusionControlNetInpaintPipeline.from_pipe(pipe, controlnet=controlnet)
    _cn_inpaint_pipes[key] = cn_pipe
    return cn_pipe
