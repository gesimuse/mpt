"""Put her face back on every frame of a 🎭 recreate.

Viggle takes hair, skin and clothes from the edited frame but rebuilds the face
from the original footage: on 2026-10-10 the edited frames matched her at
0.72-0.77 (InsightFace) and the video's frames fell to 0.06-0.2 within seconds --
"the hair changed but the face is still the original". So after Viggle each
frame's face is swapped to her identity (inswapper_128, her embedding averaged
over her refs) and sharpened (GFPGAN 1.4, inswapper works at 128px).

    python -m persona.faceswap <video> <out>

Models: ~/.insightface/swap/{inswapper_128.onnx,gfpgan_1.4.onnx}, fetched from
the FaceFusion assets release on first use. Note: inswapper_128 is released for
non-commercial research use -- fine for the review channel, but look at its
license before anything made with it goes to Fanvue.
"""
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from . import config

MODELS = Path.home() / ".insightface" / "swap"
ASSETS = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"
# FFHQ 512 alignment (what GFPGAN was trained on), as used by FaceFusion.
# Her face against her averaged identity, front-on frames of a good recreate:
# 0.6-0.8; Viggle's own faces (the original girl's): 0.05-0.3.
LOOKS_LIKE_HER = 0.45
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


def log(msg):
    print(f"[persona.faceswap] {msg}", flush=True)


def _model(name):
    path = MODELS / name
    if not path.exists() or path.stat().st_size < 1_000_000:
        MODELS.mkdir(parents=True, exist_ok=True)
        log(f"downloading {name}")
        urllib.request.urlretrieve(ASSETS + name, path)
    return path


def _providers():
    import onnxruntime as ort
    have = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in have]


class Swapper:
    def __init__(self, char):
        import cv2
        import insightface
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        self.cv2 = cv2
        # Its own detector on the GPU: faceid's CPU one took 0.9s a call, twice a
        # frame, most of a recreate's swap time.
        self.app = FaceAnalysis(name="buffalo_l", providers=_providers(), allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in _providers() else -1, det_size=(640, 640))
        self.swap = insightface.model_zoo.get_model(str(_model("inswapper_128.onnx")), providers=_providers())
        self.restore = ort.InferenceSession(str(_model("gfpgan_1.4.onnx")), providers=_providers())
        embs = []
        refs = sorted(char.refs_dir.glob("*.jpg")) + sorted((char.refs_dir / "sheet").glob("*.jpg"))
        for p in refs:
            img = cv2.imread(str(p))
            faces = self.app.get(img) if img is not None else []
            if faces:
                embs.append(max(faces, key=_area).normed_embedding)
        if not embs:
            raise RuntimeError(f"no face found in her refs ({char.refs_dir})")
        mean = np.mean(embs, axis=0)

        class Source:  # what INSwapper.get reads from the source face
            normed_embedding = mean / np.linalg.norm(mean)
        self.source = Source()
        log(f"her identity from {len(embs)} refs")

    def _sharpen(self, frame, face, blend=0.8):
        cv2 = self.cv2
        m = cv2.estimateAffinePartial2D(face.kps.astype(np.float32), FFHQ_512, method=cv2.LMEDS)[0]
        crop = cv2.warpAffine(frame, m, (512, 512), borderMode=cv2.BORDER_REPLICATE)
        x = (crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5
        out = self.restore.run(None, {self.restore.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
        out = ((out.transpose(1, 2, 0).clip(-1, 1) + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1]
        out = cv2.addWeighted(out, blend, crop, 1 - blend, 0)
        mask = np.zeros((512, 512), np.float32)
        cv2.ellipse(mask, (256, 280), (170, 215), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 18)
        inv = cv2.invertAffineTransform(m)
        h, w = frame.shape[:2]
        back = cv2.warpAffine(out, inv, (w, h), borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(mask, inv, (w, h))[:, :, None]
        return (back * alpha + frame * (1 - alpha)).astype(np.uint8)

    def _text_mask(self, img):
        """On-screen captions over her face (white text, dark outline): the swap and
        GFPGAN turned "POV: you wear a tube top" into noise (2026-10-10), so those
        pixels are kept from the frame as it was."""
        cv2 = self.cv2
        white = (img.min(axis=2) > 225) & (img.max(axis=2).astype(int) - img.min(axis=2) < 25)
        mask = cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8))
        return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[:, :, None]

    def frame(self, img):
        """(swapped frame, how front-on her face is 0..1) -- (img, 0.0) without a face."""
        faces = self.app.get(img)
        if not faces:
            return img, 0.0
        face = max(faces, key=_area)
        orig = img
        img = self.swap.get(img, face, self.source, paste_back=True)
        img = self._sharpen(img, face)  # the swap keeps the face where it was
        text = self._text_mask(orig)
        img = (orig * text + img * (1 - text)).astype(np.uint8)
        return img, _frontal(face)


    def check(self, frames, frontal, fps, step=4, low=LOOKS_LIKE_HER):
        """How much the finished video looks like her: every `step`-th frame where
        she faces the camera, her face against her identity. Turned-away frames
        are not judged (they are left as Viggle made them on purpose)."""
        sims = []
        for i in range(0, len(frames), step):
            if frontal[i] < 0.5:
                continue
            faces = self.app.get(self.cv2.imread(str(frames[i])))
            sims.append((i, float(max(faces, key=_area).normed_embedding @ self.source.normed_embedding) if faces else 0.0))
        if not sims:
            return {"judged": 0}
        run = longest = 0
        for _, sim in sims:
            run = run + 1 if sim < low else 0
            longest = max(longest, run)
        vals = [sim for _, sim in sims]
        return {"judged": len(sims), "median": round(float(np.median(vals)), 2), "min": round(min(vals), 2),
                "low_share": round(sum(v < low for v in vals) / len(vals), 2),
                "low_seconds": round(longest * step / fps, 1)}


def _frontal(face):
    """1 facing the camera, 0 in profile: where the nose sits between the eyes
    (InsightFace keypoints: eyes, nose, mouth corners), and how wide the eyes
    are next to the face box."""
    le, re_, nose = face.kps[0], face.kps[1], face.kps[2]
    eye_w = abs(re_[0] - le[0]) or 1.0
    centred = 1 - min(1.0, abs(nose[0] - (le[0] + re_[0]) / 2) / (eye_w / 2))
    spread = min(1.0, eye_w / max(1.0, face.bbox[2] - face.bbox[0]) / 0.4)
    return float(centred * spread)


def weights(frontal, lo=0.15, hi=0.45, smooth=5):
    """How much of the swap each frame gets. inswapper only knows front-on faces:
    on a head turned into profile it pasted a frontal face onto the side of her
    head (2026-10-10, "when she turns her head her face does something weird").
    So it fades out between `hi` and `lo`, smoothed over `smooth` frames so the
    face never pops from one identity to the other. Measured: profile frames
    0-0.1, front-on and three-quarter 0.45+; at 0.35-0.65 most frames of a
    slightly angled clip were only half swapped."""
    w = np.clip((np.array(frontal, dtype=np.float32) - lo) / (hi - lo), 0, 1)
    if len(w) > smooth:
        k = np.ones(smooth, np.float32) / smooth
        w = np.minimum(w, np.convolve(np.pad(w, smooth // 2, mode="edge"), k, mode="valid"))
    return w


def _area(f):
    return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])


def video(src, out, char=None):
    from . import character
    slug = config.env("PERSONA_RECREATE_SLUG")
    char = char or (character.Character(slug) if slug else character.active())
    sw = Swapper(char)
    cv2 = sw.cv2
    fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout.strip() or "24"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-q:v", "1", str(tmp / "%05d.png")], check=True)
        frames = sorted(tmp.glob("*.png"))
        swapped_dir = tmp / "swapped"
        swapped_dir.mkdir()
        frontal = []
        for f in frames:
            img, front = sw.frame(cv2.imread(str(f)))
            frontal.append(front)
            cv2.imwrite(str(swapped_dir / f.name), img)
        w = weights(frontal)
        for f, a in zip(frames, w):
            if a > 0:
                orig, img = cv2.imread(str(f)), cv2.imread(str(swapped_dir / f.name))
                cv2.imwrite(str(f), (img * a + orig * (1 - a)).astype(np.uint8))
        log(f"{len(frames)} frames: swapped {int((w >= 0.99).sum())}, faded {int(((w > 0) & (w < 0.99)).sum())}, "
            f"left as is {int((w == 0).sum())} (no face or turned away)")
        report = sw.check(frames, frontal, float(fps.split("/")[0]) / float((fps.split("/") + ["1"])[1]))
        Path(out).with_suffix(".json").write_text(json.dumps(report))
        log(f"face check: {report}")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", fps, "-i", str(tmp / "%05d.png"), "-i", str(src),
                        "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "copy", "-shortest", str(out)], check=True)
    return Path(out)


if __name__ == "__main__":
    config.load_env()
    print(video(sys.argv[1], sys.argv[2]))
