"""TaoMate-FVSR face refiner over a finished talk clip (2026-11-11: the user picked it over
GFPGAN -- sharp lips, teeth and skin, the mouth pushed to follow the voice).

It lives in the newer Wan2GP at ~/apps/Wan2GP-new (its own venv), so it runs as a
separate process:

    ~/apps/Wan2GP-new/.venv/bin/python persona/taomate.py <src.mp4> <out.mp4> [<face-ref.jpg>]

~40-70 min per 15-20s clip on the A4500 (10 steps per window of ~50 frames).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("TAOMATE_WAN2GP", Path.home() / "apps/Wan2GP-new"))


def refine(src, out, ref=None, timeout=3 * 3600, opts=None):
    """Run the refiner in Wan2GP-new's venv; returns `out`. Raises on failure.
    `opts`: {"method": "taomatefvsr" | "h3facerefine", "face_strength": .., "follow_audio": ..}."""
    cmd = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), str(src), str(out), str(ref or ""),
           json.dumps(opts or {})]
    log = Path(out).with_suffix(".log")
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    if not Path(out).exists():
        raise RuntimeError(f"TaoMate gave no file (see {log})")
    return Path(out)


def _main(src, out, ref=None, opts="{}"):
    opts = json.loads(opts or "{}")
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from shared.api import init
    session = init(root=ROOT, output_dir=Path(out).parent / "taomate-raw", cli_args=["--attention", "sdpa"])
    method = opts.pop("method", "taomatefvsr")
    kw = {"spatial_postprocessing_parameters": {"face_count": 1, "face_strength": 1.0, **opts}, "return_media": False}
    if ref:
        kw["postprocessing_reference_images"] = [str(ref)]
    t = time.time()
    r = session.submit_media_postprocessing(str(src), spatial_upsampling=method, **kw).result()
    shutil.copyfile(r.generated_files[-1], out)
    print(f"taomate done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    _main(*sys.argv[1:5])
