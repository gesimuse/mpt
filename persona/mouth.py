"""How calm her mouth is in a talk clip (2026-11-24: Leila's kiss-shaped pouts and a wide-open
yell, Skye's shocked "O"). Lower is calmer. insightface's 68 face points on every 3rd frame:
the widest the lips open (inner lip gap / face width) plus how far the mouth narrows into a
pout (its narrowest width against its usual width). Measured on the two bad clips: the
yell and the pout were each clip's worst frame."""
import numpy as np

_APP = None


def _app():
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis
        _APP = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"],
                            providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=-1, det_size=(640, 640))
    return _APP


def score(clip, end=None):
    """(score, detail) -- None score when no face was found."""
    import cv2
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    n, rows = 0, []
    while True:
        ok, im = cap.read()
        if not ok or (end and n / fps > end):
            break
        n += 1
        if n % 3:
            continue
        fs = _app().get(im)
        if not fs:
            continue
        L = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0])).landmark_3d_68[:, :2]
        fw = np.linalg.norm(L[16] - L[0]) or 1
        rows.append((np.linalg.norm(L[66] - L[62]) / fw, np.linalg.norm(L[54] - L[48]) / fw))
    cap.release()
    if len(rows) < 5:
        return None, "no face"
    a = np.array(rows)
    wide, pout = a[:, 0].max(), 1 - a[:, 1].min() / np.median(a[:, 1])
    return float(wide + pout), f"widest {wide:.3f}, pout {pout:.3f}"
