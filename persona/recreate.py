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
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(seconds), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(dest)], check=True)
    return dest


def first_frame(src, dest):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", "1", "-q:v", "2", str(dest)], check=True)
    return dest


def _stage_edit(work):
    from . import character, registry
    from .engine import Engine
    char = character.active()
    front = char.refs_dir / "01-front.jpg"
    eng = Engine(profile=config.env("PERSONA_RECREATE_PROFILE", "5"))
    model = registry.get(config.load_settings()["roles"].get("edit", "qwen21"))
    prompt = ("Replace the woman in image 1 with the woman from image 2. Keep image 1's exact pose, hand positions, "
              "clothing, camera framing, background and lighting; only her face, hair and body become the woman "
              f"from image 2 ({char.bible.get('identity', '')}).")
    s = eng.build_settings(model, prompt=prompt, image_refs=[str(work / "first.jpg"), str(front)],
                           video_prompt_type="KI", resolution=config.env("PERSONA_RECREATE_RES", "608x1072"), seed=-1)
    out = eng.run(s, work, "first-persona")[0]
    log(f"first frame: {out}")


SEGMENT = 5.0   # seconds; Viggle renders one 124-frame window (5.17s at 24fps)
OVERLAP = 1.0   # seconds each segment repeats from the previous one, to crossfade over
FADE = 0.4      # crossfade length inside the overlap; 0.75 left visible double hands mid-gesture


def segment_starts(seconds):
    """0, 4, 8, ...: every segment after the first starts OVERLAP seconds early, so
    joins can be crossfaded instead of cut (a hard cut was clearly visible)."""
    starts, t = [0.0], 0.0
    while t + SEGMENT < seconds - 0.05:
        t += SEGMENT - OVERLAP
        starts.append(t)
    return starts


def _stage_viggle(work, n_segments):
    """One Viggle window per segment of the clip, all with the same edited frame.
    Wan2GP's sliding window for Viggle rendered only the first window when asked
    for 192 frames (2026-10-08), so longer clips are split here and joined after."""
    from .engine import Engine
    eng = Engine(profile=config.env("PERSONA_RECREATE_PROFILE", "5"))
    model = {"id": "viggle", "wan2gp_model_type": "viggle_animate", "settings": {}}
    edited = next(work.glob("first-persona.*"))
    import random
    seed = random.randint(0, 2**31 - 1)   # the same for every segment: closer details at the joins
    for i in range(n_segments):
        seg = work / f"seg{i:02d}.mp4"
        s = eng.build_settings(model, prompt="a woman, natural movement", video_guide=str(seg),
                               image_refs=[str(edited)], resolution=config.env("PERSONA_RECREATE_RES", "608x1072"),
                               seed=seed)
        out = eng.run(s, work, f"out{i:02d}")[0]
        log(f"segment {i + 1}/{n_segments}: {out}")


def recreate(src, out_dir, seconds=None):
    """Returns the path of her version of the clip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = float(seconds or config.env("PERSONA_RECREATE_MAX", "10"))
    seconds = min(seconds, _probe(src) or seconds)
    clip = trim(src, out_dir / "clip.mp4", seconds)
    first_frame(clip, out_dir / "first.jpg")
    starts = segment_starts(seconds)
    n = len(starts)
    for i, start in enumerate(starts):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(clip), "-t", str(SEGMENT),
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                        str(out_dir / f"seg{i:02d}.mp4")], check=True)
    t = time.time()
    for stage, extra in (("edit", []), ("viggle", [str(n)])):
        r = subprocess.run([sys.executable, "-m", "persona.recreate", "--stage", stage, str(out_dir), *extra],
                           cwd=str(config.REPO))
        if r.returncode != 0:
            raise RuntimeError(f"recreate stage {stage} failed ({r.returncode})")
    parts = sorted(out_dir.glob("out[0-9][0-9].mp4"))
    joined = out_dir / "joined.mp4"
    if len(parts) == 1:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(parts[0]), "-map", "0:v", "-c:v", "libx264",
                        "-crf", "18", "-pix_fmt", "yuv420p", str(joined)], check=True)
    else:
        # Each next part starts OVERLAP seconds before the previous one ends: fade
        # across the last FADE seconds of that overlap, timed to the original clip.
        inputs, chain, prev = [], [], "[0:v]"
        for p in parts:
            inputs += ["-i", str(p)]
        for i in range(1, len(parts)):
            offset = starts[i]   # part i begins at its own time in the original clip
            label = f"[x{i}]"
            chain.append(f"{prev}[{i}:v]xfade=transition=fade:duration={FADE}:offset={offset:.3f}{label}")
            prev = label
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chain), "-map", prev,
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(joined)], check=True)
    out = out_dir / "recreated.mp4"
    # The original clip's audio, cut to the video's length.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(joined), "-i", str(clip), "-map", "0:v", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    log(f"done in {int(time.time() - t)}s ({len(parts)} segments): {out}")
    return out


def main():
    config.load_env()
    args = sys.argv[1:]
    if args[:1] == ["--stage"]:
        work = Path(args[2])
        if args[1] == "edit":
            _stage_edit(work)
        else:
            _stage_viggle(work, int(args[3]))  # number of segments
        return
    seconds = None
    if "--seconds" in args:
        i = args.index("--seconds")
        seconds = args[i + 1]
        args = args[:i] + args[i + 2:]
    print(json.dumps({"video": str(recreate(args[0], args[1], seconds))}))


if __name__ == "__main__":
    main()
