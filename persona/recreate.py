"""Recreate a short video (a TikTok trend, a dance) with the persona in it.

    python -m persona.recreate <video> <out_dir> [--seconds 10]

1. trim the clip to its first --seconds (PERSONA_RECREATE_MAX, default 10), audio kept
2. edit her into its first frame with the image model and her refs (Qwen 2.1:
   "replace the woman in image 1 with the woman from image 2, keep pose, clothes,
   framing")
3. Viggle-Animate carries that first frame through the whole clip, following the
   original motion; the original audio stays

Chosen 2026-10-08 after a laptop test on a real TikTok: Viggle kept the room,
chair, outfit, gestures and timing with her face and hair. Wan 2.2 Animate's
replacement mode needs a hand-made person mask per frame, so it was not used.

Steps 2 and 3 run in separate processes: Qwen-Image and the 20B Viggle model do
not fit in 30GB of RAM together (the first attempt was OOM-killed), and neither
do they on Kaggle.

Use it for trends and dances with one person in the frame. It replaces the
person entirely (no real person's likeness is kept), but the background, moves
and audio are the original creator's.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from . import config


def log(msg):
    print(f"[persona.recreate] {msg}", flush=True)


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def trim(src, dest, seconds):
    """First `seconds` of the clip, resampled to 24fps. Viggle reads the control
    video frame by frame at its own 24fps: fed a 30fps TikTok it played 30 frames
    of motion per 24 -- everything 25% slow, hands drifting out of sync with the
    original and the audio (2026-10-09). At 24fps, frames and time agree."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(seconds), "-vf", "fps=24",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)], check=True)
    return dest


def first_frame(src, dest):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dest)], check=True)
    return dest


def shots(clip, seconds, threshold=0.3, min_len=0.6):
    """[(start, end)] of each continuous shot. Viggle carries her look from one
    edited frame only through continuous footage: on a montage TikTok (kitchen,
    sofa, bed) she was lost after the first cut and the original girl came back
    (2026-10-09). Each shot now gets its own edited frame."""
    r = subprocess.run(["ffmpeg", "-i", str(clip), "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True)
    import re
    cuts = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
    bounds = [0.0]
    for c in cuts:
        if c - bounds[-1] >= min_len and seconds - c >= min_len:
            bounds.append(c)
    bounds.append(seconds)
    return list(zip(bounds[:-1], bounds[1:]))


def best_frame(shot, dest, step=0.25):
    """The shot's clearest front-facing frame, as the reference Viggle copies her
    look from (any frame of the shot can be the reference). The first frame was
    used before; in a clip that opens mid eye-roll her eyes swirled while Viggle
    morphed from the edited straight gaze into the roll (2026-10-09). Falls back
    to the first frame when no face is found or the detector cannot load."""
    import tempfile
    from . import faceid
    try:
        app = faceid._app()
    except Exception:
        app = None
    if app is None:
        return first_frame(shot, dest)
    import cv2
    best, best_score = None, -1.0
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(shot), "-vf", f"fps={1 / step}", f"{tmp}/%03d.jpg"], check=True)
        for f in sorted(Path(tmp).glob("*.jpg")):
            img = cv2.imread(str(f))
            faces = app.get(img) if img is not None else []
            if not faces:
                continue
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            size = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            # Front-on: the nose sits midway between the eyes (insightface kps:
            # left eye, right eye, nose, mouth corners).
            le, re_, nose = face.kps[0], face.kps[1], face.kps[2]
            eye_w = abs(re_[0] - le[0]) or 1
            frontal = 1 - min(1, abs(nose[0] - (le[0] + re_[0]) / 2) / (eye_w / 2))
            score = float(face.det_score) * frontal * (size ** 0.5)
            if score > best_score:
                best, best_score = f, score
        if best is None:
            return first_frame(shot, dest)
        import shutil
        shutil.copyfile(best, dest)
    log(f"{Path(shot).name}: reference frame {best.name} (score {best_score:.0f})")
    return dest


def _stage_edit(work):
    from . import character, registry
    from .engine import Engine
    char = character.active()
    front = char.refs_dir / "01-front.jpg"
    eng = Engine(profile=config.env("PERSONA_RECREATE_PROFILE", "5"))
    model = registry.get(config.load_settings()["roles"].get("edit", "qwen21"))
    prompt = ("Replace the woman in image 1 with the woman from image 2. Keep image 1's exact pose, hand positions, "
              "facial expression, eye direction, eyelids and mouth shape, clothing, camera framing, background and "
              "lighting; only her identity -- face, hair and body -- becomes the woman from image 2 "
              f"({char.bible.get('identity', '')}).")
    for first in sorted(work.glob("shot*-first.jpg")):
        stem = first.name.replace("-first.jpg", "-persona")
        s = eng.build_settings(model, prompt=prompt, image_refs=[str(first), str(front)],
                               video_prompt_type="KI", resolution=config.env("PERSONA_RECREATE_RES", "608x1072"),
                               seed=-1)
        out = eng.run(s, work, stem)[0]
        log(f"{first.name}: {out.name}")


def _stage_viggle(work):
    """One Viggle pass per shot with Wan2GP's sliding window, so each shot is one
    continuous video. Two gotchas from 2026-10-08: the length has to be given in
    seconds ("8.5s"), and Wan2GP returns the first window as an intermediate file
    before the finished one -- the LONGEST file is the video."""
    from .engine import Engine
    eng = Engine(profile=config.env("PERSONA_RECREATE_PROFILE", "5"))
    model = {"id": "viggle", "wan2gp_model_type": "viggle_animate", "settings": {}}
    for shot in sorted(work.glob("shot[0-9][0-9].mp4")):
        edited = next(work.glob(f"{shot.stem}-persona.*"))
        seconds = _probe(shot)
        s = eng.build_settings(model, prompt="a woman, natural movement", video_guide=str(shot),
                               image_refs=[str(edited)], resolution=config.env("PERSONA_RECREATE_RES", "608x1072"),
                               video_length=f"{seconds:.2f}s", sliding_window_size=124, sliding_window_overlap=18,
                               seed=-1)
        files = eng.run(s, work, f"{shot.stem}-viggle")
        final = max(files, key=_probe)
        # Viggle renders at least 107 frames (~4.5s): a shorter shot comes back
        # padded, which would push every later shot out of sync with the audio.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(final), "-t", f"{seconds:.3f}", "-an",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(work / f"{shot.stem}-out.mp4")],
                       check=True)
        log(f"{shot.stem}: {_probe(work / f'{shot.stem}-out.mp4'):.2f}s (shot {seconds:.2f}s)")


def recreate(src, out_dir, seconds=None):
    """Returns the path of her version of the clip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = float(seconds or config.env("PERSONA_RECREATE_MAX", "10"))
    seconds = min(seconds, _probe(src) or seconds)
    clip = trim(src, out_dir / "clip.mp4", seconds)
    parts = shots(clip, seconds)
    log(f"{len(parts)} shot(s): {[(round(a, 2), round(b, 2)) for a, b in parts]}")
    for i, (a, b) in enumerate(parts):
        shot = out_dir / f"shot{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-i", str(clip), "-t", f"{b - a:.3f}",
                        "-vf", "fps=24", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(shot)],
                       check=True)
        # The shot's first frame, edited with its expression and eye direction kept:
        # a mismatch there (straight gaze vs. an eye-roll) made her pupils swirl.
        # best_frame() picks the most front-on frame instead, but head pose does
        # not see eyes rolled up or closed, so it is not used by default.
        first_frame(shot, out_dir / f"shot{i:02d}-first.jpg")
    t = time.time()
    for stage in ("edit", "viggle"):
        r = subprocess.run([sys.executable, "-m", "persona.recreate", "--stage", stage, str(out_dir)],
                           cwd=str(config.REPO))
        if r.returncode != 0:
            raise RuntimeError(f"recreate stage {stage} failed ({r.returncode})")
    outs = sorted(out_dir.glob("shot[0-9][0-9]-out.mp4"))
    (out_dir / "shots.txt").write_text("".join(f"file '{p.name}'\n" for p in outs))
    joined = out_dir / "joined.mp4"
    # Hard cuts between shots, exactly where the original cuts.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(out_dir / "shots.txt"),
                    "-map", "0:v", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(joined)], check=True)
    out = out_dir / "recreated.mp4"
    # The original clip's audio, cut to the video's length.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-i", str(clip), "-map", "0:v", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    log(f"done in {int(time.time() - t)}s: {out}")
    return out


def main():
    config.load_env()
    args = sys.argv[1:]
    if args[:1] == ["--stage"]:
        work = Path(args[2])
        if args[1] == "edit":
            _stage_edit(work)
        else:
            _stage_viggle(work)
        return
    seconds = None
    if "--seconds" in args:
        i = args.index("--seconds")
        seconds = args[i + 1]
        args = args[:i] + args[i + 2:]
    print(json.dumps({"video": str(recreate(args[0], args[1], seconds))}))


if __name__ == "__main__":
    main()
