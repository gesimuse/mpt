"""Which way is her head actually turned? (2026-10-18: the face sheet's "three-quarter",
"profile-right", "looking up/down" portraits measured within ~10 degrees of her front
ref for most girls -- the edit model kept the input pose -- so every photo drawn from
them faced the same way.)

InsightFace's 3D landmarks give pitch/yaw in degrees (CPU, like faceid). Yaw is
negative when she turns to HER left (Lena's left profile ref: -55). Pitch is judged
against her own front ref, since the estimator's zero differs per face.
"""
import threading

_APP = None
_LOCK = threading.Lock()

# (yaw range, pitch change from her front ref) each sheet stem must show.
TARGETS = {
    "": ((-12, 12), None),
    "three-quarter-left": ((-45, -15), None),
    "three-quarter-right": ((15, 45), None),
    "profile-left": ((-90, -35), None),
    "profile-right": ((35, 90), None),
    "looking-up": ((-30, 30), (8, 60)),
    "looking-down": ((-30, 30), (-60, -8)),
}


def _app():
    global _APP
    with _LOCK:
        if _APP is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                               allowed_modules=["detection", "landmark_3d_68"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _APP = app
        return _APP


def pose(path):
    """(pitch, yaw) of the biggest face, or None."""
    import cv2
    faces = _app().get(cv2.imread(str(path)))
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return float(f.pose[0]), float(f.pose[1])


def matches(stem, path, front_pitch=0.0):
    """(ok, (pitch, yaw)). ok when the head shows the stem's angle, or when the stem
    has no target or no face is found (a profile can hide from the detector)."""
    if stem is None:
        return True, None
    target = TARGETS.get(stem)
    if target is None:
        return True, None
    try:
        p = pose(path)
    except Exception:
        return True, None
    if p is None:
        return True, None
    (lo, hi), dp = target
    ok = lo <= p[1] <= hi
    if dp:
        ok = ok and dp[0] <= p[0] - front_pitch <= dp[1]
    return ok, (round(p[0]), round(p[1]))
