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
            root = os.path.expanduser(os.environ.get("PERSONA_WAN2GP_DIR", "~/apps/Wan2GP"))
            pose = os.path.join(root, "ckpts", "pose")
            if not os.path.exists(os.path.join(pose, "dw-ll_ucoco_384.onnx")):
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
    if torso is None and height is None:
        return None
    return {"torso": round(torso, 2) if torso else None, "height": round(height, 2) if height else None}


def check(path):
    """(ok, reason, ratios). ok when in proportion or when it can't tell."""
    try:
        m = measure(path)
    except Exception as e:
        return True, f"proportions unavailable ({type(e).__name__})", None
    if not m:
        return True, "", None
    if (m["torso"] and m["torso"] < TORSO[0]) or (m["height"] and m["height"] < HEIGHT[0]):
        return False, "head too big for her body", m
    if (m["torso"] and m["torso"] > TORSO[1]) or (m["height"] and m["height"] > HEIGHT[1]):
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
