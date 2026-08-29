"""One OpenAI-compatible chat call, tried against every backend we have, in order.

Why this exists: caption_writer.py and motion_writer.py each had their own copy of the
same "POST to Ollama, retry 3x, raise" helper, and both were failing in production for
the same reason. Of 11 recorded uploads in posted.json, exactly ONE ever got an
LLM-written caption -- every other post shipped niches.json's fixed hashtag string
(#aiart #aigenerated #aiphotography #confident, on literally every post) because
llama3.2:3b generating ~200 tokens on a GH Actions runner's 2 vCPUs takes far longer
than the 30s timeout those helpers used, so all three attempts timed out and the caller
fell back to the static pool. The video workflow was worse still: it never installed
Ollama at all, so its captions could never be anything but the static pool.

Backends, in order:
  1. HF router (https://router.huggingface.co/v1/chat/completions) -- OpenAI-compatible,
     hosted, fast, and it needs no new secret: the same HF_TOKEN/HF_TOKENS this repo
     already uses for ZeroGPU video generation. Rotates across HF_TOKENS on an
     auth/credit error, since Inference Providers' free credits are per account.
     NOTE: the token must be a fine-grained one with "Make calls to Inference
     Providers" -- a ZeroGPU-only read token 401s here (and rotation moves past it).
  2. Ollama (OLLAMA_URL) -- the original local path, kept as a real fallback so a run
     still writes its own captions if HF is unreachable or out of credit. Timeout
     raised to 240s: the 30s one was the actual bug above, not a tuning preference.

Callers still get an exception when every backend fails; they fall back to their own
static pool rather than this module retrying forever."""
import os, time

import requests

HF_CHAT_URL = os.environ.get(
    "HF_CHAT_URL", "https://router.huggingface.co/v1/chat/completions")
# ":fastest" is the router's own provider-selection suffix (pick the highest-throughput
# provider serving this model). Override with CAPTION_HF_MODEL.
#
# Deliberately NOT a reasoning model. gpt-oss-20b was tried first and returned
# {"content": null, "reasoning": "..."} against the real caption rubric -- it spent the
# whole max_tokens budget thinking and emitted no answer, which arrives here as an
# empty completion. Qwen3-8B does the same (reasoning_content set, content null).
# Llama-3.1-8B-Instruct has no reasoning channel at all: every token it is given goes
# into the answer, which is what a one-line caption actually needs.
HF_MODEL = os.environ.get("CAPTION_HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:fastest")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("CAPTION_MODEL", "llama3.2:3b")
# 30s (the old value) is well under what llama3.2:3b needs on 2 vCPUs -- see the module
# docstring. This is a ceiling for a stuck request, not an expected duration.
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "240"))
HF_TIMEOUT = int(os.environ.get("HF_TIMEOUT", "60"))


def log(msg): print(f"[llm] {msg}", flush=True)


def hf_tokens():
    """HF_TOKENS (comma-separated, one per account) if set, else the single HF_TOKEN,
    else nothing. Same parsing as videogen._tokens, minus its anonymous fallback --
    the router has no anonymous tier, an unauthenticated call is just a 401."""
    raw = os.environ.get("HF_TOKENS", "").strip()
    if raw:
        toks = [t.strip() for t in raw.split(",") if t.strip()]
        if toks:
            return toks
    single = os.environ.get("HF_TOKEN", "").strip()
    return [single] if single else []


# Status codes where trying the NEXT HF token can actually help: this account is out of
# free credits (402), rate limited (429), or its token lacks the Inference Providers
# permission (401/403). Anything else fails identically on every token.
_ROTATE_STATUSES = {401, 402, 403, 429}
# Retry the SAME backend on these -- transient, not account-specific.
_RETRY_STATUSES = {408, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def _post(url, headers, body, timeout):
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    return r


def _extract(response_json):
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"unexpected response shape: {str(response_json)[:200]}") from e
    if not content or not content.strip():
        raise LLMError("empty completion")
    return content


def _ask_hf(prompt, max_tokens, temperature):
    tokens = hf_tokens()
    if not tokens:
        raise LLMError("no HF_TOKEN/HF_TOKENS set")
    body = {"model": HF_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    last = None
    for i, token in enumerate(tokens):
        headers = {"Authorization": f"Bearer {token}",
                   "Content-Type": "application/json"}
        for attempt in range(2):
            try:
                r = _post(HF_CHAT_URL, headers, body, HF_TIMEOUT)
            except (requests.Timeout, requests.ConnectionError) as e:
                last = LLMError(f"{type(e).__name__}: {str(e)[:120]}")
                break
            if r.status_code in _ROTATE_STATUSES:
                last = LLMError(f"HF token {i + 1}/{len(tokens)}: "
                                f"{r.status_code} {r.text[:150]}")
                log(f"HF token {i + 1}/{len(tokens)} rejected ({r.status_code}), "
                    "trying the next one" if i < len(tokens) - 1
                    else f"HF token {i + 1}/{len(tokens)} rejected ({r.status_code})")
                break
            if r.status_code in _RETRY_STATUSES:
                last = LLMError(f"{r.status_code} {r.text[:150]}")
                if attempt == 0:
                    time.sleep(2)
                    continue
                break
            if not r.ok:
                raise LLMError(f"HF router {r.status_code}: {r.text[:200]}")
            return _extract(r.json())
    raise last or LLMError("HF router: every token failed")


def _ask_ollama(prompt, max_tokens, temperature, attempts=2):
    body = {"model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    last = None
    for i in range(attempts):
        try:
            r = _post(OLLAMA_URL, {"Content-Type": "application/json"}, body,
                      OLLAMA_TIMEOUT)
            if r.status_code in _RETRY_STATUSES or r.status_code == 429:
                raise LLMError(f"{r.status_code} {r.text[:150]}")
            r.raise_for_status()
            return _extract(r.json())
        except (requests.Timeout, requests.ConnectionError, LLMError,
                requests.HTTPError) as e:
            last = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    raise last


def ask(prompt, max_tokens=200, temperature=0.9):
    """The completion text from the first backend that answers. Raises LLMError with
    every backend's own failure attached when none does -- callers treat that as
    "use the static fallback", never as a reason to block a batch from shipping."""
    errors = []
    for name, fn in (("hf", _ask_hf), ("ollama", _ask_ollama)):
        try:
            return fn(prompt, max_tokens, temperature)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:150]}")
            log(f"{name} backend unavailable ({type(e).__name__}: {str(e)[:120]})")
    raise LLMError("every LLM backend failed -- " + " | ".join(errors))
