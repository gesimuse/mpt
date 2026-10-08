"""Every model in the registry, run in-process through one Wan2GP runtime.

Wan2GP runs Qwen-Image 2.1, Qwen-Image 2512 and Edit 2511, and Wan 2.2 video behind
one Python API (shared/api.py). local/wan2gp_bot.py and the Kaggle video kernel
already use it, so one runtime covers every model, and only the model that is
needed sits in RAM. Switching from an image model to the video model is Wan2GP's
own unload/load, not a second process.

Must run under Wan2GP's own venv (torch, mmgp and the rest live there):
    ~/apps/Wan2GP/.venv/bin/python -m persona.bot

Same RAM rule as wan2gp_bot.py: this, that bot and the Wan2GP web UI are the same
runtime on 30GB of RAM. Run one at a time.
"""
import json
import shutil
import sys
import threading
import time
from pathlib import Path

from . import config

PROGRESS_EVERY = 30


def log(msg):
    print(f"[persona.engine] {msg}", flush=True)


class GenerationError(RuntimeError):
    pass


class Engine:
    def __init__(self, wan_dir=None, profile=None, attention=None, verbose=None):
        self.wan_dir = Path(wan_dir or config.env("WAN2GP_DIR") or Path.home() / "apps/Wan2GP").expanduser()
        self.profile = str(profile or config.env("WAN2GP_PROFILE", "4"))
        self.attention = attention or config.env("WAN2GP_ATTENTION", "sdpa")
        self.verbose = config.flag("PERSONA_VERBOSE") if verbose is None else verbose
        self._session = None
        self._lock = threading.Lock()
        self._scratch = config.home() / ".engine-out"

    # ------------------------------------------------------------------ runtime
    def install_finetunes(self):
        """Copy persona/wan2gp_finetunes/*.json into Wan2GP's finetunes/ folder. This
        is how a model Wan2GP does not ship, such as the uncensored Qwen re-upload,
        becomes a model_type it can load. Overwrites, so an edit here wins."""
        dest = self.wan_dir / "finetunes"
        dest.mkdir(exist_ok=True)
        for src in (config.PACKAGE / "wan2gp_finetunes").glob("*.json"):
            target = dest / src.name
            if not target.exists() or target.read_text() != src.read_text():
                shutil.copyfile(src, target)
                log(f"installed Wan2GP finetune {src.name}")

    def session(self):
        with self._lock:
            if self._session is None:
                self.install_finetunes()
                sys.path.insert(0, str(self.wan_dir))
                from shared.api import init
                log(f"loading Wan2GP from {self.wan_dir} (profile {self.profile}, {self.attention})")
                started = time.time()
                self._scratch.mkdir(parents=True, exist_ok=True)
                self._session = init(
                    root=self.wan_dir,
                    output_dir=self._scratch,
                    cli_args=["--profile", self.profile, "--attention", self.attention],
                    console_output=self.verbose,
                )
                log(f"runtime ready in {int(time.time() - started)}s")
        return self._session

    # ----------------------------------------------------------------- settings
    def _profile(self, model):
        rel = model.get("profile")
        if not rel:
            return {}
        path = self.wan_dir / rel
        if not path.exists():
            raise GenerationError(f"{model['id']}: profile {rel} not found in {self.wan_dir}")
        data = json.loads(path.read_text())
        data.pop("profile_priority", None)
        return data

    def build_settings(self, model, **job):
        """Wan2GP's own defaults for the model, then its accelerator profile, then the
        registry's overrides, then this job. Later wins."""
        mt = model["wan2gp_model_type"]
        s = self.session()
        try:
            settings = s.get_default_settings(mt)
        except Exception as e:
            raise GenerationError(f"Wan2GP does not know model_type {mt!r} "
                                  f"({type(e).__name__}: {e})") from None
        settings.update(self._profile(model))
        settings.update(model.get("settings", {}))
        settings.update({k: v for k, v in job.items() if v is not None})
        settings["model_type"] = mt
        return settings

    # ---------------------------------------------------------------------- run
    def run(self, settings, dest_dir, stem):
        """One Wan2GP task. Moves what it produced into dest_dir as <stem>[-i].<ext>
        and returns those paths."""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        job = self.session().submit_task(settings)
        last = 0.0
        try:
            for event in job.events.iter(timeout=0.5):
                if event.kind != "progress" or time.time() - last < PROGRESS_EVERY:
                    continue
                last = time.time()
                p = event.data
                step, total = getattr(p, "current_step", None), getattr(p, "total_steps", None)
                log(f"  {getattr(p, 'phase', '')} {step}/{total}" if step else f"  {getattr(p, 'phase', '')}")
        except Exception as e:
            log(f"progress stream stopped early ({type(e).__name__})")
        result = job.result(timeout=int(config.env("WAN2GP_TIMEOUT", "5400")))
        took = int(time.time() - started)
        if not result.success or not result.generated_files:
            reasons = "; ".join(getattr(e, "message", str(e)) for e in result.errors) \
                or "no file produced and no error reported"
            raise GenerationError(f"{settings['model_type']} failed after {took}s: {reasons[:600]}")
        out = []
        for i, f in enumerate(result.generated_files):
            src = Path(f)
            name = f"{stem}{'' if i == 0 else f'-{i}'}{src.suffix}"
            target = dest_dir / name
            shutil.move(str(src), target)
            out.append(target)
        log(f"{settings['model_type']}: {len(out)} file(s) in {took}s")
        return out

    def image(self, model, prompt, dest_dir, stem, refs=(), seed=-1, negative=None,
              resolution=None):
        """One image. `refs` are reference photos (her master refs), cut to what the
        model accepts; with none, it is plain text-to-image."""
        refs = [str(r) for r in refs][: model.get("max_refs", 0)]
        job = {"prompt": prompt, "seed": seed, "batch_size": 1, "resolution": resolution}
        if negative and float(self._effective_guidance(model)) > 1:
            job["negative_prompt"] = negative
        if refs:
            job["image_refs"] = refs
            job["video_prompt_type"] = "I"   # refs are people, not a main/background image
        settings = self.build_settings(model, **job)
        return self.run(settings, dest_dir, stem)[0]

    def video(self, model, image, prompt, dest_dir, stem, seed=-1, frames=None):
        job = {"prompt": prompt, "image_start": str(image), "seed": seed, "batch_size": 1,
               "video_length": frames}
        settings = self.build_settings(model, **job)
        # With sliding windows Wan2GP returns an intermediate file per window before
        # the finished video; the last one is the video.
        return self.run(settings, dest_dir, stem)[-1]

    def _effective_guidance(self, model):
        """Negative prompts only do anything when guidance > 1. The 8-step
        accelerator profiles run at guidance 1, where Wan2GP ignores them."""
        merged = {**self._profile(model), **model.get("settings", {})}
        return merged.get("guidance_scale", 4)
