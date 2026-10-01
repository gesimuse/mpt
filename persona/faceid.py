"""Is this still her, and does she read as an adult?

InsightFace's buffalo_l pack: an ArcFace embedding for identity (cosine similarity
against the mean of her master refs) and its age estimator for the adult floor.
Runs on CPU so it never competes with Wan2GP for VRAM.

Two gates, with different failure behaviour:
  * identity  -- soft. Below PERSONA_FACE_MIN (default 0.45) the image is dropped as
                 off-model. If the checker cannot load, images pass with a warning.
  * age       -- hard. Any detected face estimated under PERSONA_MIN_AGE (default
                 21) is dropped, and so is an image where no face can be checked.
                 If the checker cannot load, nothing passes. A fictional adult
                 persona must never come out looking underage, whatever the prompt
                 or model did.

Licence note: InsightFace's pretrained packs are released for non-commercial
research. Here it only filters, it never generates, but if that matters for your
use, set PERSONA_FACE_CHECK=0 and judge identity yourself. The age gate
is then off too, and bot.py says so in every caption.
"""
import threading

import numpy as np

from . import config

_APP = None
_LOCK = threading.Lock()
_LOAD_ERROR = None


def log(msg):
    print(f"[persona.faceid] {msg}", flush=True)


def enabled():
    return config.flag("PERSONA_FACE_CHECK", "1")


def _app():
    global _APP, _LOAD_ERROR
    with _LOCK:
        if _APP is None and _LOAD_ERROR is None:
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                                   allowed_modules=["detection", "recognition", "genderage"])
                app.prepare(ctx_id=-1, det_size=(640, 640))
                _APP = app
            except Exception as e:
                _LOAD_ERROR = f"{type(e).__name__}: {e}"
                log(f"face checker unavailable ({_LOAD_ERROR[:200]})")
    return _APP


def _faces(path):
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return []
    return _app().get(img)


def _main_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None


def reference_embedding(ref_paths):
    if not enabled() or _app() is None:
        return None
    embs = []
    for p in ref_paths:
        f = _main_face(_faces(p))
        if f is not None:
            embs.append(f.normed_embedding)
    if not embs:
        return None
    mean = np.mean(embs, axis=0)
    return mean / np.linalg.norm(mean)


def check(path, ref_emb):
    """Returns (ok, info). info carries similarity/age/reason for the caption."""
    min_age = float(config.env("PERSONA_MIN_AGE", "21"))
    min_sim = float(config.env("PERSONA_FACE_MIN", "0.45"))
    if not enabled():
        return True, {"note": "face check off"}
    if _app() is None:
        return False, {"reason": f"face checker failed to load ({(_LOAD_ERROR or '')[:80]}); "
                                 "age cannot be verified, so nothing passes"}
    faces = _faces(path)
    if not faces:
        return False, {"reason": "no face found, age cannot be verified"}
    ages = [float(getattr(f, "age", 0) or 0) for f in faces]
    if min(ages) < min_age:
        return False, {"reason": f"estimated age {min(ages):.0f} < {min_age:.0f}", "age": min(ages)}
    info = {"age": round(min(ages))}
    if ref_emb is not None:
        # Reference-guided models sometimes draw her once per reference image.
        # Bystanders score far below the floor; a second face above it is her again.
        sims = sorted((float(np.dot(f.normed_embedding, ref_emb)) for f in faces), reverse=True)
        clones = sum(1 for x in sims if x >= min_sim)
        if clones > 1:
            return False, {**info, "reason": f"she appears {clones} times in one image"}
        sim = float(np.dot(_main_face(faces).normed_embedding, ref_emb))
        info["sim"] = round(sim, 2)
        if sim < min_sim:
            return False, {**info, "reason": f"off-model (similarity {sim:.2f} < {min_sim})"}
    return True, info
