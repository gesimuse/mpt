"""Every finished video, made fit for every platform it goes to (2026-11-05):

  * 1080x1920 (9:16). Facebook refuses a Reel under 960px tall or narrower than
    9:16 ("Video height must be at least 960px", "aspect ratio is too narrow"),
    and the talks and 🎬 videos come out at ~512x896. A frame within 12% of 9:16
    is scaled and cropped to fill; anything else is scaled to fit and padded.
  * at least 30 fps: Instagram refuses a Reel under 23 fps (Wan makes 16).
  * at most 19MB: Telegram refuses a bot's download past 20MB, and Buffer fetches
    every video through it. The bitrate is set from the video's length.

Used by the laptop agent (agent.deliver) and the Kaggle kernel, so a video is
the same wherever it was made. Never raises: on any failure the original stays.
"""

import json
import subprocess
from pathlib import Path

W, H = 1080, 1920
LIMIT = 19_000_000


def _probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                   "stream=codec_type,width,height,r_frame_rate:format=duration",
                                   "-of", "json", str(path)]).decode()
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, _, den = v.get("r_frame_rate", "30/1").partition("/")
    return {"w": int(v["width"]), "h": int(v["height"]), "fps": float(num) / float(den or 1),
            "dur": float(d.get("format", {}).get("duration") or 0),
            "audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def needed(path):
    """True when the video is not already 1080x1920, >= 23 fps and under the size limit."""
    try:
        i = _probe(path)
    except Exception:
        return False
    return (i["w"], i["h"]) != (W, H) or i["fps"] < 23 or Path(path).stat().st_size > LIMIT


def social(path, log=print):
    """Re-encode `path` in place for Reels / TikTok / Facebook. Returns True if changed."""
    path = Path(path)
    if path.suffix != ".mp4" or not path.exists() or not needed(path):
        return False
    try:
        i = _probe(path)
        ratio = (i["w"] / i["h"]) / (W / H)
        if 0.88 <= ratio <= 1.12:
            frame = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}"
        else:
            frame = (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                     f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")
        vf = frame + ",setsar=1" + (",fps=30" if i["fps"] < 23 else "")
        # Video bitrate that keeps the file under the limit (audio 160k), capped at 8 Mbit/s.
        kbps = 8000 if i["dur"] <= 0 else max(800, min(8000, int(LIMIT * 8 / i["dur"] / 1000 * 0.92) - 160))
        tmp = path.with_name(path.stem + "-vfit.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vf", vf,
                        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                        "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k", "-pix_fmt", "yuv420p",
                        *(["-c:a", "aac", "-b:a", "160k"] if i["audio"] else ["-an"]),
                        "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(path)
        log(f"{path.name}: {i['w']}x{i['h']} {i['fps']:.0f}fps -> {W}x{H}"
            f"{' 30fps' if i['fps'] < 23 else ''}, {path.stat().st_size // 1_000_000}MB")
        return True
    except Exception as e:
        log(f"{path.name}: not re-encoded ({type(e).__name__}: {e})")
        return False


def fit_dir(folder, log=print):
    """Every .mp4 in `folder` (a Kaggle run's out/items) made fit."""
    for p in sorted(Path(folder).glob("*.mp4")):
        social(p, log)
