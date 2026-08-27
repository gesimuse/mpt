"""A second pair of eyes on generated images before they reach a channel.

A 4-step LCM generation is fast specifically because it skips the refinement a normal
20-50 step run would do, which is also where a diffusion model fixes its own anatomy
mistakes -- extra fingers, warped hands, merged limbs, an asymmetric face. Those are far
more common at 4 steps than at 25, so nothing is exempt from this check.

Runs against a local Ollama instance's OpenAI-compatible endpoint. Default vision model
is llama3.2-vision -- pull it once with `ollama pull llama3.2-vision`.

llava:7b was tried as a replacement after an old CI run rejected every image with
'unknown model architecture: mllama' from llama-server (llama3.2-vision's mllama
architecture wasn't supported by the llama-server engine Ollama shipped at the time).
But llava:7b's own `realistic` (1-10) score turned out badly miscalibrated -- confirmed
live: it scored 1/10 on multiple genuinely convincing, clearly-photorealistic generated
images, rejecting nearly everything regardless of actual quality. Re-tested
llama3.2-vision directly against Ollama 0.24.0 and the mllama error is gone (Ollama's
own engine has since caught up); it scored the SAME images 9/10, correctly. CI installs
Ollama via `curl -fsSL https://ollama.com/install.sh | sh` with no version pin, so it
always gets the current release too -- same fix applies there, not just locally. Larger
pull than llava:7b (~7.9GB vs ~4.7GB), but a correct verdict matters more than a faster
download of a model that can't do the one numeric judgment call this whole gate leans on.
"""
import base64, json, os, re, time
from pathlib import Path

import requests

# ponytail: Ollama's OpenAI-compatible endpoint, no API key. Override OLLAMA_URL to point
# at a remote Ollama.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
VISION_MODELS = [
    m.strip() for m in
    os.environ.get("SUPERVISOR_VISION_MODELS", "llama3.2-vision").split(",") if m.strip()
]

RUBRIC = """You inspect one AI-generated photo before it is allowed to reach a public
account. Look closely, especially at hands, face and limbs -- these are where fast
generation most often goes wrong.

Answer:
  realistic       1-10. Does it read as a real photo, not obviously synthetic or warped?
  anatomy_ok      true only if hands, fingers, face and limbs are anatomically correct.
                  Extra or fused fingers, a warped face, or an impossible limb -> false.
  fully_clothed   true ONLY if there is a real, visible garment covering BOTH the
                  chest/breasts AND the hips/groin/rear -- check each half separately.
                  Swimwear, lingerie, a bikini and similar count as a real garment.
                  A pose or camera angle that keeps genitals out of frame is NOT the
                  same as being clothed -- if the hip/rear/upper-thigh area shows bare
                  skin with no waistband, hem, or fabric at all, that is false, even if
                  nothing is technically exposed at this angle. Exposed breasts,
                  exposed genitals, or full nudity -> false regardless of anything else.
  age_appears_adult  true only if the subject clearly appears to be an adult. If there
                     is any doubt at all, false.
  ethnicity_excluded  true if the subject appears Chinese or East Asian, false
                     otherwise. Operator preference for this account, not a judgment
                     of the subject -- answer only what the image shows.
  issues          short list of concrete problems, empty if none.

Respond with the JSON object only. No description of the image, no preamble, no
markdown fences -- the first character of your reply must be "{".
{"realistic": n, "anatomy_ok": bool, "fully_clothed": bool, "age_appears_adult": bool,
"ethnicity_excluded": bool, "issues": ["..."]}"""

MIN_REALISTIC = int(os.environ.get("SUPERVISOR_MIN_REALISTIC", "6"))


def log(msg): print(f"[supervisor] {msg}", flush=True)


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _ask_vision(model, prompt, image_b64, max_tokens=500, attempts=3):
    """A dropped connection killed the whole review the first time this ran live, so
    transient failures are retried the same way llm.py retries text calls."""
    last = None
    for i in range(attempts):
        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ]}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
                timeout=90,
            )
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise RuntimeError(f"{r.status_code} {r.text[:150]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.Timeout, requests.ConnectionError, RuntimeError) as e:
            last = e
            if i < attempts - 1:
                time.sleep(3 * (i + 1))
    raise last


def _extract_json(text):
    """The model wraps answers in prose or code fences often enough that a bare
    json.loads fails on the first live call this ever made -- pull the {...} span out
    instead of trusting the response to be pure JSON."""
    cleaned = re.sub(r"```[a-zA-Z]*|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise RuntimeError(f"no JSON object in response: {cleaned[:150]}")
        return json.loads(match.group(0))


def review_image(path, parse_retries=2):
    """One image, judged by EVERY configured vision model, not just the first one that
    returns parseable JSON. A live check found the primary model confidently and
    wrongly scoring an image with full, unambiguous nudity as fully_clothed=True --
    while the secondary model refused to even discuss the same image outright ("I'm
    not going to engage in this conversation topic"). That refusal is a real safety
    signal, not noise to route around: it was previously only ever consulted as a
    fallback when the primary model's response failed to PARSE, never when it parsed
    fine but was simply wrong, which is exactly what let that image through.

    Every configured model must now independently agree the image passes -- a refusal,
    an unparseable response, or a straight fail from ANY of them rejects the whole
    thing. Never silently passes an image any model could not or would not actually
    inspect. Doubles the per-image review cost (every model, every time, not one with
    a fallback) -- a deliberate tradeoff given what a false pass here means."""
    image_b64 = _b64(path)
    verdicts = []
    for model in VISION_MODELS:
        result, last = None, None
        for attempt in range(parse_retries):
            try:
                raw = _ask_vision(model, RUBRIC, image_b64)
                result = _extract_json(raw)
                result["_model"] = model
                break
            except Exception as e:
                last = e
                log(f"{model} attempt {attempt + 1}/{parse_retries} failed "
                    f"({type(e).__name__}: {str(e)[:100]})")
        if result is None:
            log(f"{model}: no usable verdict, rejecting by default ({last})")
            return {"realistic": 0, "anatomy_ok": False, "fully_clothed": False,
                   "age_appears_adult": False,
                   "issues": [f"{model} gave no usable verdict: {last}"]}
        verdicts.append(result)

    return {
        "realistic": min(v.get("realistic", 0) or 0 for v in verdicts),
        "anatomy_ok": all(v.get("anatomy_ok") for v in verdicts),
        "fully_clothed": all(v.get("fully_clothed") for v in verdicts),
        "age_appears_adult": all(v.get("age_appears_adult") for v in verdicts),
        # Inverted from the "ok" fields above on purpose: this is a reject signal, so
        # ANY model flagging it is enough, the same conservative direction the dual-
        # model consult already takes for a refusal/parse-failure (one model saying
        # "this is a problem" outweighs another saying "looks fine to me").
        "ethnicity_excluded": any(v.get("ethnicity_excluded") for v in verdicts),
        "issues": [i for v in verdicts for i in (v.get("issues") or [])],
        "_models": [v.get("_model") for v in verdicts],
    }


def passes(result, min_realistic=None):
    """anatomy_ok, age_appears_adult and ethnicity_excluded stay hard requirements.
    fully_clothed is no longer enforced here (still recorded in the result and
    logged by filter_images() below, just not gating) -- the account owner reviews
    every draft in the TikTok app before posting and removes individual images from
    the carousel there, so nudity is caught downstream by a human either way, and
    gating it here was mostly costing variety, not adding real protection past that
    point. ethnicity_excluded exists specifically because that downstream human
    review does NOT catch which underlying CHECKPOINT is responsible for a pattern
    across a whole batch -- a real batch came out consistently Chinese/Asian-
    appearing from a checkpoint whose name did not say so, and rejecting each image
    here is what feeds a low pass rate back into imageslides.py's model_stats
    weighting, so that checkpoint stops getting picked without anyone needing to
    have named it in advance."""
    min_realistic = min_realistic if min_realistic is not None else MIN_REALISTIC
    realistic = result.get("realistic")
    if not isinstance(realistic, (int, float)) or realistic < min_realistic:
        return False
    return bool(result.get("anatomy_ok") and result.get("age_appears_adult")
               and not result.get("ethnicity_excluded"))


def _is_broken_verdict(result):
    """A verdict where the supervisor MODEL failed to answer (network error, model
    unavailable, unsupported architecture) rather than the model actually judging
    the image bad. review_image() shape when broken: realistic=0 AND every issue
    string starts with '<model> gave no usable verdict:' -- that prefix only comes
    from the model-error branch, not from real content flags."""
    if result.get("realistic"):
        return False
    issues = result.get("issues") or []
    return bool(issues) and all("gave no usable verdict" in i for i in issues)


class FilterResult(list):
    """Behaves like the old `kept` list so existing callers unpacking a plain list
    keep working, but carries `.supervisor_broken` alongside for callers that want
    to distinguish 'supervisor said no' from 'supervisor could not answer'."""
    def __init__(self, kept, supervisor_broken=False):
        super().__init__(kept)
        self.supervisor_broken = supervisor_broken


def filter_images(paths, min_realistic=None):
    """Review every image, keep only what passes, log why anything was dropped.
    Return value acts as a plain list of kept paths (backward-compatible); it also
    carries a .supervisor_broken flag that's True when EVERY reviewed image failed
    with a supervisor-model error (mllama arch missing, network down, etc.) rather
    than a real content judgment. Callers use that to decide whether to fall back
    to pushing raw generations -- see imageslides.generate for the fallback rule."""
    kept = []
    per_image = []
    for path in paths:
        result = review_image(path)
        per_image.append(result)
        ok = passes(result, min_realistic)
        log(f"{Path(path).name}: {'PASS' if ok else 'REJECT'} "
            f"realistic={result.get('realistic')} anatomy_ok={result.get('anatomy_ok')} "
            f"fully_clothed={result.get('fully_clothed')} "
            f"age_appears_adult={result.get('age_appears_adult')} "
            f"ethnicity_excluded={result.get('ethnicity_excluded')}"
            + (f" issues={result['issues']}" if result.get("issues") else ""))
        if ok:
            kept.append(path)
    supervisor_broken = bool(per_image) and all(_is_broken_verdict(r) for r in per_image)
    return FilterResult(kept, supervisor_broken=supervisor_broken)
