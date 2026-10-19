"""Is her head in proportion with her body? (2026-10-17: Lena photos came out with
the head too big -- a camera above her, a wide lens -- and some too small.)

DWPose (Wan2GP's own copy, ckpts/pose/, on the CPU so it never competes with the
image model for the GPU) finds her shoulders, hips, ankles and face points, and
two ratios are judged against the face's chin-to-brow length:

  torso / face   shoulders-to-hips. Her full-body refs: 2.9-3.2.
  height / face  brow-to-ankles, when her feet are in the photo. Refs: 9.2-10.8.

Measured on 80 of Lena's photos: below 2.5 / 8.5 the head was clearly too big
(a top-down shot, a seated wide-angle shot), above 3.45 / 11.5 too small. Fails
open (no verdict) when the model is missing or her shoulders/hips are not seen.
"""
import os
import sys
import threading

TORSO = (2.5, 3.45)
HEIGHT = (8.5, 11.5)

_WB = None
_LOCK = threading.Lock()


def _model():
    global _WB
    with _LOCK:
        if _WB is None:
            root = os.path.expanduser(os.environ.get("PERSONA_WAN2GP_DIR") or os.environ.get("WAN2GP_DIR")
                                      or "~/apps/Wan2GP")
            pose = os.path.join(root, "ckpts", "pose")
            if not os.path.exists(os.path.join(pose, "dw-ll_ucoco_384.onnx")):
                # Kaggle's fresh Wan2GP has none: fetched from where Wan2GP's own
                # preprocessing gets them (2026-10-19: Kaggle renders skipped this check).
                try:
                    from huggingface_hub import hf_hub_download
                    for f in ("dw-ll_ucoco_384.onnx", "yolox_l.onnx"):
                        hf_hub_download("DeepBeepMeep/Wan2.1", f"pose/{f}", local_dir=os.path.join(root, "ckpts"))
                except Exception as e:
                    print(f"[persona.proportions] pose model unavailable: {type(e).__name__}: {str(e)[:150]}", flush=True)
                    _WB = False
                    return _WB
            if root not in sys.path:
                sys.path.insert(0, root)
            from preprocessing.dwpose.wholebody import Wholebody
            _WB = Wholebody(f"{pose}/yolox_l.onnx", f"{pose}/dw-ll_ucoco_384.onnx", device="cpu")
        return _WB


def measure(path):
    """{"torso": ratio, "height": ratio or None}, or None when it can't tell."""
    import cv2
    import numpy as np
    wb = _model()
    if not wb:
        return None
    img = cv2.imread(str(path))
    K, S, det = wb(img)
    if not len(det):
        return None
    # The biggest person in the photo is her.
    i = int(np.argmax([(d[2] - d[0]) * (d[3] - d[1]) for d in det]))
    k, s = K[i], S[i]
    # OpenPose order: 2/5 shoulders, 8/11 hips, 10/13 ankles; face points from 24
    # (chin 24+8, brows 24+19 and 24+24).
    if min(s[j] for j in (2, 5, 32, 43, 48)) < 0.3:
        return None
    face = float(np.linalg.norm(k[32] - (k[43] + k[48]) / 2))
    if face < 4:
        return None
    # Hips clearly seen and well inside the frame, else the torso is not judged: in
    # close-ups cut at the waist the guessed hips flagged good photos (2026-10-17).
    torso = None
    if min(s[8], s[11]) > 0.6 and max(k[8][1], k[11][1]) < 0.9 * img.shape[0]:
        torso = float(np.linalg.norm((k[2] + k[5]) / 2 - (k[8] + k[11]) / 2)) / face
    height = None
    # Ankles only when clearly seen inside the frame: a photo cut at the shins puts
    # guessed ankles on the bottom edge.
    if min(s[10], s[13]) > 0.5 and max(k[10][1], k[13][1]) < 0.97 * img.shape[0]:
        height = float((k[10][1] + k[13][1]) / 2 - (k[43][1] + k[48][1]) / 2) / face
    # Shoulder width against face width (jaw points 24+0 / 24+16), only when she faces
    # the camera: a turned head or body narrows one of them. Slim bodies came out with
    # narrow shoulders under a full-size head (2026-10-18).
    width = None
    if min(s[24], s[40]) > 0.4 and abs(float(k[2][0] - k[5][0])) > 0.8 * float(np.linalg.norm(k[2] - k[5])):
        jaw = float(np.linalg.norm(k[24] - k[40]))
        nose_mid = abs(float(k[54][0] - (k[24][0] + k[40][0]) / 2)) < 0.15 * jaw
        if nose_mid and jaw > 4:
            width = float(np.linalg.norm(k[2] - k[5])) / jaw
    if torso is None and height is None and width is None:
        return None
    return {"torso": round(torso, 2) if torso else None, "height": round(height, 2) if height else None,
            "width": round(width, 2) if width else None}


_REF = {}
# Her own full-body ref's ratios scale the limits (2026-10-18: Lola's thick fringe
# hides her brows, so her face measures short and good photos read "head too small";
# her ref is 3.40 torso / 11.06 height against Lena's 2.96 / 9.82).
_TYPICAL = (2.96, 9.82)
_TYPICAL_WIDTH = 1.96
WIDTH = 1.72  # refs 1.9-2.1; 1.43-1.69 looked big-headed


def check(path, ref=None):
    """(ok, reason, ratios). ok when in proportion or when it can't tell. `ref`: her
    full-body ref photo, to judge against her own proportions."""
    try:
        m = measure(path)
        if ref and str(ref) not in _REF:
            _REF[str(ref)] = measure(ref) or {}
        r = _REF.get(str(ref), {}) if ref else {}
    except Exception as e:
        return True, f"proportions unavailable ({type(e).__name__})", None
    if not m:
        return True, "", None
    if m.get("width") and m["width"] < WIDTH * (r.get("width") or _TYPICAL_WIDTH) / _TYPICAL_WIDTH:
        return False, "head too big for her shoulders", m
    kt = (r.get("torso") or _TYPICAL[0]) / _TYPICAL[0]
    kh = (r.get("height") or _TYPICAL[1]) / _TYPICAL[1]
    if (m["torso"] and m["torso"] < TORSO[0] * kt) or (m["height"] and m["height"] < HEIGHT[0] * kh):
        return False, "head too big for her body", m
    if (m["torso"] and m["torso"] > TORSO[1] * kt) or (m["height"] and m["height"] > HEIGHT[1] * kh):
        return False, "head too small for her body", m
    return True, "", m


def twisted(path):
    """True when her back faces the camera (shoulders swapped: her right shoulder on
    the image's right) while her face looks straight at it -- the head turned round
    over her back. False when not, None when it can't tell. Replaces the vision
    model's twist question, which called 15 of 18 hip-pop glam poses twisted
    (2026-10-17)."""
    import cv2
    import numpy as np
    wb = _model()
    if not wb:
        return None
    K, S, det = wb(cv2.imread(str(path)))
    if not len(det):
        return None
    i = int(np.argmax([(d[2] - d[0]) * (d[3] - d[1]) for d in det]))
    k, s = K[i], S[i]
    # 2 right shoulder, 5 left; face points 24+36 / 24+45 eye corners, 24+30 nose tip.
    if min(s[2], s[5], s[60], s[69], s[54]) < 0.4:
        return None
    width = float(np.linalg.norm(k[2] - k[5]))
    back_to_camera = (k[2][0] - k[5][0]) > 0.3 * width
    eyes = sorted([k[60][0], k[69][0]])
    frontal = eyes[0] < k[54][0] < eyes[1] and (eyes[1] - eyes[0]) > 0.25 * width
    return bool(back_to_camera and frontal)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, check(f), "twisted:", twisted(f))
