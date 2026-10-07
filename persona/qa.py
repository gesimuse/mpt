"""Free, local photo check (a partial safety net -- see the 2026-10-07 test below): a small vision model counts her hands and arms, what
each hand holds and how far her head is turned, and we judge the counts.

Why counts: on 2026-10-07 a yes/no "any defects?" question flipped between runs
on the same photo, while counts were stable and caught a hand holding a phone
and a book at once. Runs on the GPU after the image model (Kaggle T4 or the
laptop), so there is no quota or credit to run out of.

Tested 2026-10-07 on 3 known-bad and 5 known-good photos: catches a head turned
almost backwards and (borderline) a hand holding a phone and a book; misses a
subtle extra forearm; no false alarms. 🔁 Redo covers what it misses.

PERSONA_VISION_QA_MODEL picks the model (default Qwen/Qwen2.5-VL-7B-Instruct,
loaded in 4-bit when bitsandbytes is available, which keeps it near 6GB VRAM).
"""
import json
import re
import threading

from . import config

QA_PROMPT = ('Look carefully at the woman in this photo and answer with ONLY JSON:\n'
             '{"hands_visible": <number of her hands you can see>, "arms_visible": <number of her arms>, '
             '"objects_in_one_hand": <max number of separate objects held in a single hand>, '
             '"head_turned_unnaturally": <true if her face points more than ~100 degrees away from where '
             'her chest faces>, "limb_defect": <true if any arm, leg, hand or finger is fused, melted, '
             'duplicated or impossible>}')

# Asked on its own: inside a bigger question the model also flagged good photos
# for "holding several things". Alone, it caught the head turned almost backwards.
TWIST_PROMPT = ('Is this woman\'s head or torso rotated further than a real person physically can (for '
                'example her face pointing backwards over her back)? Answer ONLY JSON: '
                '{"body_twisted_impossibly": true or false}')

# Asked on its own: inside a bigger question the model also flagged good photos
# for "holding several things". Alone, it caught the head turned almost backwards.
TWIST_PROMPT = ('Is this woman\'s head or torso rotated further than a real person physically can (for '
                'example her face pointing backwards over her back)? Answer ONLY JSON: '
                '{"body_twisted_impossibly": true or false}')

_MODEL = None
_LOCK = threading.Lock()


def log(msg):
    print(f"[persona.qa] {msg}", flush=True)


def verdict(v):
    problems = []
    if int(v.get("hands_visible") or 0) > 2:
        problems.append(f"{v['hands_visible']} hands")
    if int(v.get("arms_visible") or 0) > 2:
        problems.append(f"{v['arms_visible']} arms")
    if int(v.get("objects_in_one_hand") or 0) > 1:
        problems.append("one hand holding several things")
    if v.get("head_turned_unnaturally") is True:
        problems.append("head turned unnaturally")
    if v.get("limb_defect") is True:
        problems.append("limb defect")
    return not problems, ", ".join(problems)


def enabled():
    return config.flag("PERSONA_VISION_QA", "1")


def _load():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: I001
            name = config.env("PERSONA_VISION_QA_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
            import transformers
            dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            kw = {dkey: torch.float16, "device_map": "cuda"}
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                                bnb_4bit_quant_type="nf4")
            except ImportError:
                pass
            log(f"loading {name} ({'4-bit' if 'quantization_config' in kw else 'fp16'})")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kw)
            # Small images are enough to count hands and keep it fast on a T4.
            # Two processors, one per question: the processor's own resize decides what
            # the model sees, and the count only came out right at 1024 tiles.
            procs = {mp: AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=mp * 28 * 28)
                     for mp in (1024, 2048)}
            _MODEL = (model, procs)
    return _MODEL


def unload():
    global _MODEL
    with _LOCK:
        if _MODEL is not None:
            import gc
            import torch
            _MODEL = None
            gc.collect()
            torch.cuda.empty_cache()


def _ask(path, prompt, tiles):
    """`tiles` picks the processor: tested 2026-10-07, the counting question was only
    right at 1024 (it caught a phone and a book in one hand), the twist question
    needed 2048 (it caught a head turned almost backwards)."""
    import torch
    from PIL import Image
    model, procs = _load()
    proc = procs[tiles]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[Image.open(path).convert("RGB")], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    answer = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return json.loads(re.search(r"\{.*\}", answer, re.S).group(0))


_WORKER = None


def _worker_check(path):
    """Ask the checker running in its own process (PERSONA_QA_SUBPROCESS=1). On the
    Kaggle T4 the model failed inside the Wan2GP process with a CUDA JIT error
    (ERROR_UNSUPPORTED_CAST) yet ran fine in a clean process, so there it lives in
    a separate interpreter, PERSONA_QA_PYTHON (default: this one)."""
    import subprocess
    import sys
    global _WORKER
    if _WORKER is None or _WORKER.poll() is not None:
        env = dict(__import__("os").environ, PERSONA_QA_SUBPROCESS="0")
        _WORKER = subprocess.Popen([config.env("PERSONA_QA_PYTHON") or sys.executable, "-m", "persona.qa", "--worker"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
                                   cwd=str(config.REPO))
    _WORKER.stdin.write(json.dumps({"path": str(path)}) + "\n")
    _WORKER.stdin.flush()
    while True:
        line = _WORKER.stdout.readline()
        if not line:
            raise RuntimeError("QA worker exited")
        if line.startswith("QA_RESULT "):
            r = json.loads(line[len("QA_RESULT "):])
            return r["ok"], r["reason"], r["counts"]
        print(f"[qa-worker] {line.rstrip()}", flush=True)


def check(path):
    """(ok, reason, counts). Fails open (ok, 'qa unavailable') if the model cannot run."""
    if not enabled():
        return True, "qa off", {}
    if config.flag("PERSONA_QA_SUBPROCESS", "0"):
        try:
            return _worker_check(path)
        except Exception as e:
            log(f"worker failed ({type(e).__name__}: {str(e)[:200]})")
            return True, "qa unavailable", {}
    try:
        counts = _ask(path, QA_PROMPT, 1024)
        counts["head_turned_unnaturally"] = bool(_ask(path, TWIST_PROMPT, 2048).get("body_twisted_impossibly"))
        ok, reason = verdict(counts)
        return ok, reason, counts
    except Exception as e:
        import traceback
        log(f"check could not run ({type(e).__name__}: {str(e)[-300:]})\n{traceback.format_exc()[-1500:]}")
        return True, "qa unavailable", {}


def _worker_main():
    """`python -m persona.qa --worker`: one JSON request per stdin line, one
    QA_RESULT line back. Everything else it prints is ordinary log output."""
    import sys
    for line in sys.stdin:
        if not line.strip():
            continue
        ok, reason, counts = check(json.loads(line)["path"])
        print("QA_RESULT " + json.dumps({"ok": ok, "reason": reason, "counts": counts}), flush=True)


if __name__ == "__main__":
    import sys
    if "--worker" in sys.argv:
        config.load_env()
        _worker_main()
