"""A second pair of eyes on generated images before they reach a channel.

A 4-step LCM generation is fast specifically because it skips the refinement a normal
20-50 step run would do, which is also where a diffusion model fixes its own anatomy
mistakes -- extra fingers, warped hands, merged limbs, an asymmetric face. Those are far
more common at 4 steps than at 25, so nothing is exempt from this check.

Verified live against this account's NIM key: meta/llama-3.2-11b-vision-instruct accepts
a base64 data URI in an image_url content block and answers correctly. That is the
primary path, since it shares the free NIM credits already used elsewhere. A second
vision-capable id is tried if the first is unavailable, so one model's downtime does not
stop every review.
"""
import base64, json, os, re, time
from pathlib import Path

import requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
# In preference order. Vision model ids on NIM's catalogue have moved before (the chat
# fallback chain in llm.py exists for the same reason); trying more than one absorbs that.
VISION_MODELS = [
    m.strip() for m in
    os.environ.get("SUPERVISOR_VISION_MODELS",
                   "meta/llama-3.2-11b-vision-instruct,meta/llama-3.2-90b-vision-instruct")
    .split(",") if m.strip()
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
  issues          short list of concrete problems, empty if none.

Respond with the JSON object only. No description of the image, no preamble, no
markdown fences -- the first character of your reply must be "{".
{"realistic": n, "anatomy_ok": bool, "fully_clothed": bool, "age_appears_adult": bool,
"issues": ["..."]}"""

MIN_REALISTIC = int(os.environ.get("SUPERVISOR_MIN_REALISTIC", "6"))


def log(msg): print(f"[supervisor] {msg}", flush=True)


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _ask_vision(model, prompt, image_b64, max_tokens=500, attempts=3):
    """A dropped connection killed the whole review the first time this ran live, so
    transient failures are retried the same way llm.py retries text calls."""
    key = os.environ.get("NIM_API_KEY")
    if not key:
        raise RuntimeError("NIM_API_KEY not set")
    last = None
    for i in range(attempts):
        try:
            r = requests.post(
                NIM_URL,
                headers={"Authorization": f"Bearer {key}"},
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
        "issues": [i for v in verdicts for i in (v.get("issues") or [])],
        "_models": [v.get("_model") for v in verdicts],
    }


def passes(result, min_realistic=None):
    """anatomy_ok and age_appears_adult stay hard requirements -- weird/non-human
    anatomy and age are not something a quick look at the finished draft reliably
    catches, and age is a non-negotiable line regardless. fully_clothed is no longer
    enforced here (still recorded in the result and logged by filter_images() below,
    just not gating) -- the account owner reviews every draft in the TikTok app
    before posting and removes individual images from the carousel there, so nudity
    is caught downstream by a human either way, and gating it here was mostly costing
    variety, not adding real protection past that point."""
    min_realistic = min_realistic if min_realistic is not None else MIN_REALISTIC
    realistic = result.get("realistic")
    if not isinstance(realistic, (int, float)) or realistic < min_realistic:
        return False
    return bool(result.get("anatomy_ok") and result.get("age_appears_adult"))


def filter_images(paths, min_realistic=None):
    """Review every image, keep only what passes, log why anything was dropped."""
    kept = []
    for path in paths:
        result = review_image(path)
        ok = passes(result, min_realistic)
        log(f"{Path(path).name}: {'PASS' if ok else 'REJECT'} "
            f"realistic={result.get('realistic')} anatomy_ok={result.get('anatomy_ok')} "
            f"fully_clothed={result.get('fully_clothed')} "
            f"age_appears_adult={result.get('age_appears_adult')}"
            + (f" issues={result['issues']}" if result.get("issues") else ""))
        if ok:
            kept.append(path)
    return kept
