"""Her head back in proportion in a finished video (2026-10-22: a recreate kept the
dancer's long body, so Lena's face read as too small; a photo animation kept the
photo's narrow shoulders under a full-size head, too big).

    python -m persona.headscale <video> <out> [--ref her 04-full-body.jpg] [--scale 1.1]

How far off: a few frames measured with proportions.measure() against her own
full-body ref -- torso/face (face too small when long) and shoulders/face width
(head too big when narrow). The ratio furthest from hers wins, corrected part of
the way (SOFTEN) and at most MAX_CHANGE, since a full correction distorts her hair
and neck more than the error it fixes.

How: every frame, her head (insightface's face box, smoothed over time so it does
not wobble) is magnified around its centre by the factor, fading back to 1 over a
ring around it -- like a phone app's reshape tool. Audio is kept.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOFTEN = 0.6
MAX_CHANGE = 0.12
MIN_CHANGE = 0.04


def log(msg):
    print(f"[persona.headscale] {msg}", flush=True)


def factor(video, ref, samples=10):
    """How much bigger (>1) or smaller (<1) her head should be; 1.0 when it can't tell."""
    import statistics
    from . import proportions
    want = proportions.measure(ref) if ref else None
    want = want or {"torso": proportions._TYPICAL[0], "width": proportions._TYPICAL_WIDTH}
    with tempfile.TemporaryDirectory() as tmp:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(video)], capture_output=True, text=True).stdout.strip() or 0)
        if dur <= 0:
            return 1.0
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={samples / dur}",
                        f"{tmp}/%03d.jpg"], check=True)
        ms = [m for m in (proportions.measure(f) for f in sorted(Path(tmp).glob("*.jpg"))) if m]
    # torso/face too long = head too small (scale up); shoulders/face too narrow =
    # head too big (scale down). The bigger error is the one you see.
    errs = []
    for key in ("torso", "width"):
        got = [m[key] for m in ms if m.get(key)]
        if len(got) >= 3 and want.get(key):
            errs.append(statistics.median(got) / want[key])
    if not errs:
        return 1.0
    err = max(errs, key=lambda e: abs(e - 1))
    s = min(1 + MAX_CHANGE, max(1 - MAX_CHANGE, err ** SOFTEN))
    log(f"{Path(video).name}: ratios vs hers {[round(e, 2) for e in errs]} -> head x{s:.2f}")
    return s if abs(s - 1) >= MIN_CHANGE else 1.0


def _warp(img, cx, cy, fh, s, maps):
    """Magnify by `s` inside 1.3 face-heights of (cx, cy), back to 1 at 2.4."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if "grid" not in maps:
        maps["grid"] = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = maps["grid"]
    dx, dy = xx - cx, yy - cy
    d = np.sqrt(dx * dx + dy * dy)
    r0, r1 = 1.3 * fh, 2.4 * fh
    t = np.clip((d - r0) / (r1 - r0), 0, 1)
    t = t * t * (3 - 2 * t)
    m = s + (1 - s) * t  # magnification at this distance
    return cv2.remap(img, cx + dx / m, cy + dy / m, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply(src, dest, s):
    import cv2
    from . import headpose
    app = headpose._app()
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        out, box, maps, n, seen = None, None, {}, 0, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if out is None:
                out = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            faces = app.get(img)
            if faces:
                x1, y1, x2, y2 = max(faces, key=lambda f: f.det_score).bbox
                # Centre a little above the face's: her head is face plus hair.
                new = ((x1 + x2) / 2, (y1 + y2) / 2 - 0.15 * (y2 - y1), y2 - y1)
                box = new if box is None else tuple(0.6 * a + 0.4 * b for a, b in zip(box, new))
                seen += 1
            out.write(_warp(img, *box, s, maps) if box else img)
            n += 1
        cap.release()
        if out is None:
            raise RuntimeError("no frames")
        out.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(src), "-map", "0:v", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                        str(dest)], check=True)
    log(f"{Path(src).name}: head x{s:.2f} on {seen}/{n} frames")
    return dest


def fix(src, dest, ref=None):
    """Writes `dest` (scaled, or a plain copy when she is in proportion); returns the factor."""
    try:
        s = factor(src, ref)
        if s != 1.0:
            apply(src, dest, s)
            return s
    except Exception as e:
        log(f"skipped ({type(e).__name__}: {str(e)[:200]})")
    if Path(src) != Path(dest):
        shutil.copyfile(src, dest)
    return 1.0


if __name__ == "__main__":
    a = sys.argv[1:]
    ref = a[a.index("--ref") + 1] if "--ref" in a else None
    if "--scale" in a:
        apply(a[0], a[1], float(a[a.index("--scale") + 1]))
    else:
        fix(a[0], a[1], ref)
