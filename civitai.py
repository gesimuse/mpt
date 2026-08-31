"""Real, popular CivitAI prompts, and the checkpoint chosen to run them on.

The flow this exists for: search CivitAI's public gallery for well-liked portrait
images, take the prompts that actually produced them, and generate from those --
never a static, hand-written prompt list. harvest_prompts() is the primary path;
imageslides.py only falls back to its own prompt formula when CivitAI cannot be
reached at all.

Point CIVITAI_MODEL (env var, or a niche's "civitai_model" field) at a model and it is
resolved, downloaded once, and used for every image that niche generates. Accepted
forms: a civitai.com URL, a bare model id, "<model id>:<version id>" to pin an exact
version, or free text -- "realistic portrait woman" -- which searches the catalogue and
picks the most-downloaded qualifying result, the same way civitai_query already works
for prompts.

CivitAI's /api/v1/models and /api/v1/images returned 451 for this account until a
CIVITAI_API_KEY was set (see _headers()); with a key both work normally. Model-version
lookups and downloads work either way.

Only SafeTensor files are ever downloaded. CivitAI also offers PickleTensor (.ckpt)
files for backward compatibility, but pickle deserialisation can execute arbitrary code
embedded in the file -- there is no reason to accept that format when the same weights
are available as SafeTensor, which cannot. A failed or missing virus/pickle scan on the
file CivitAI itself reports is also refused.
"""
import os, random, re, time
from pathlib import Path
from urllib.parse import urlparse

import requests

API = "https://civitai.com/api/v1"
CACHE_DIR = Path.home() / ".cache" / "civitai-models"


def log(msg): print(f"[civitai] {msg}", flush=True)


def _headers():
    """CIVITAI_API_KEY is read-only here: it is sent on GET requests to CivitAI's own
    search/gallery endpoints, and nowhere near their (separate, paid) generation API.
    This module never triggers generation on CivitAI's infrastructure, on this key or
    any other -- every image is produced locally, by sdgen.py, on this machine.

    The key also fixes a real problem: /api/v1/models and /api/v1/images returned 451
    to this account without it, for reasons CivitAI does not document on the response.

    Downloads used to work unauthenticated; they no longer do for every model. See
    _download_headers below -- this function is for the JSON API, that one for the
    file endpoint, and they differ on redirects."""
    headers = {"User-Agent": "mpt-autopilot/1.0"}
    key = (os.environ.get("CIVITAI_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


# The signed storage host CivitAI's download endpoint redirects to. An Authorization
# header on a pre-signed URL can fail signature validation on S3-compatible backends,
# so the key must reach civitai.com and stop there. requests already drops Authorization
# on a cross-host redirect, which handles the hop; this is the belt to that braces.
_CIVITAI_API_HOSTS = {"civitai.com", "www.civitai.com"}


def _download_headers(url):
    """Headers for CivitAI's FILE endpoint (/api/download/...), which is a different
    problem from _headers()' JSON API.

    Both this and download() used to send a bare User-Agent, on a comment asserting
    that resolved["url"] is "a pre-signed R2 storage link, not the civitai.com API".
    That was wrong on both counts: _resolved() sets url to chosen["downloadUrl"], which
    IS civitai.com/api/download/models/<id>, and CivitAI now gates at least some models
    behind auth there. Confirmed live on model 352289: no auth -> 401, API key -> 200
    then a 307 to b2.civitai.com. It broke every Kaggle image run (the 401 surfaced as
    "Kaggle image gen failed", silently falling back to ~40min of local CPU generation)
    and would have broken local downloads of the same gated models too.

    The key is attached only for civitai.com itself, never for the storage host."""
    headers = {"User-Agent": "mpt-autopilot/1.0"}
    key = (os.environ.get("CIVITAI_API_KEY") or "").strip()
    if key and urlparse(url).hostname in _CIVITAI_API_HOSTS:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _get(url, params=None, attempts=3, timeout=30):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if r.status_code in (403, 451):
                raise RuntimeError(
                    f"CivitAI returned {r.status_code} for this network. If this is a "
                    "region block it will not clear on retry; try a different runner "
                    "or host the file elsewhere.")
            r.raise_for_status()
            return r
        except RuntimeError:
            raise
        except requests.RequestException as e:
            last = e
            if i < attempts - 1:
                time.sleep(2 * (i + 1))
    raise last


_URL_RE = re.compile(r"civitai\.com/models/(\d+)(?:.*?modelVersionId=(\d+))?", re.I)


def _parse_spec(spec):
    """(model_id, version_id) from a URL, 'model:version', or a bare model id.
    version_id is None when the caller wants whatever CivitAI marks current."""
    spec = spec.strip()
    m = _URL_RE.search(spec)
    if m:
        return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)
    if ":" in spec:
        model_id, version_id = spec.split(":", 1)
        return int(model_id), int(version_id)
    return int(spec), None


def _version_info(model_id, version_id):
    if version_id:
        return _get(f"{API}/model-versions/{version_id}").json()
    data = _get(f"{API}/models/{model_id}").json()
    versions = data.get("modelVersions") or []
    if not versions:
        raise RuntimeError(f"model {model_id} has no versions")
    return versions[0]  # CivitAI lists newest first


def _arch_for(base_model):
    base_model = (base_model or "").upper()
    if "XL" in base_model:
        return "sdxl"
    if "SD 1" in base_model or "1.5" in base_model:
        return "sd15"
    return None


def _pick_safetensor(files):
    """The scan-clean SafeTensor file to use, primary first, or None. Never a
    PickleTensor/.ckpt: pickle deserialisation can execute arbitrary code embedded in
    the file, and the same weights are available as SafeTensor, which cannot."""
    candidates = [f for f in files
                 if (f.get("metadata") or {}).get("format") == "SafeTensor"
                 and f.get("pickleScanResult") == "Success"
                 and f.get("virusScanResult") == "Success"]
    if not candidates:
        return None
    candidates.sort(key=lambda f: not f.get("primary", False))
    return candidates[0]


def _resolved(info, model_id, name=None):
    """Build the standard resolve() result from a model-version info blob, whether it
    came from a direct lookup or a search result's embedded modelVersions[]."""
    arch = _arch_for(info.get("baseModel"))
    if not arch:
        raise RuntimeError(
            f"unsupported base model {info.get('baseModel')!r} -- only SD 1.5 and SDXL "
            "checkpoints are supported (LCM-LoRA needs a matching architecture)")
    chosen = _pick_safetensor(info.get("files", []))
    if not chosen:
        raise RuntimeError(
            f"model version {info.get('id')} has no scan-clean SafeTensor file "
            "(only PickleTensor/.ckpt available, or the scan did not pass -- refusing)")
    name = name or (info.get("model") or {}).get("name") or info.get("name") or str(model_id)
    log(f"resolved {name!r} v{info.get('id')} ({arch}, {chosen['name']}, "
        f"{chosen.get('sizeKB', 0) / 1024 / 1024:.1f}GB)")
    return {
        "url": chosen["downloadUrl"], "filename": chosen["name"], "arch": arch,
        "name": name, "version_id": info.get("id"),
        "model_id": info.get("modelId", model_id),
    }


def resolve(spec):
    """Everything needed to download and load one checkpoint: url, filename, base
    model architecture, and the human-readable name for logging.

    spec is a CivitAI URL, a bare model id, or "<model id>:<version id>". Anything that
    is not one of those forms -- free text -- is treated as a search, exactly how
    civitai_query already works for prompts: "realistic portrait woman" behaves like a
    query, "4201" or a civitai.com URL behaves like a pin."""
    try:
        model_id, version_id = _parse_spec(spec)
    except ValueError:
        return resolve_from_search(spec)
    info = _version_info(model_id, version_id)
    return _resolved(info, model_id)


# Only Checkpoint-type models with a body of real usage are considered: a model with
# few downloads is more likely mislabeled, broken, or simply untested at scale.
MIN_SEARCH_DOWNLOADS = int(os.environ.get("CIVITAI_MIN_DOWNLOADS", "1000"))
# _ETHNICITY_EXCLUDE_RE (below) only rejects a harvested PROMPT that literally names
# an ethnicity -- it does nothing about a checkpoint that is itself trained/finetuned
# toward one, which generates that way regardless of prompt wording. Live-confirmed
# this is a real gap, not hypothetical: this niche's own queries ("sexy realistic
# woman", "realistic portrait woman") surface "SEX Sexy Eastern Experience v3 ||
# Realistic Asian" as a legitimate high-download candidate, and a batch generated
# from it came out consistently Chinese/Asian-appearing across every image --
# exactly the operator preference _ETHNICITY_EXCLUDE_RE was meant to enforce, just
# via a mechanism that check never covered. Rejected by checkpoint NAME, at
# candidate-selection time, before any prompt is ever evaluated.
# Not "eastern" -- the live-observed offending checkpoint ("SEX Sexy Eastern
# Experience v3 || Realistic Asian") already has "asian" right in its own name, and
# "eastern" alone would also reject "Eastern European"-style checkpoints, which are
# not what this is meant to exclude.
# CJK characters in a checkpoint's own name (e.g. "ppkkmoon_Daemon_Realm_V2
# [ppkkmoon_魔域之墮姫_model_V2]", also live-observed in this niche's own search
# results) are the same ethnicity-bias signal by another route -- a checkpoint named
# in Chinese/Japanese/Korean script is reliably from a community/training set skewed
# the same way.
_ETHNICITY_MODEL_NAME_RE = re.compile(
    r"\b(?:asian|chinese)\b|[一-鿿぀-ヿ가-힣]", re.I)


def search_models(query, limit=20, sort="Most Downloaded", nsfw=False):
    """Candidate checkpoints for a free-text query, most-downloaded first.

    nsfw=False restored (operator request) -- was removed earlier, on the reasoning
    that a live check found the same checkpoint results either way, so it wasn't
    what was surfacing ethnicity-biased ones (still true; _ETHNICITY_MODEL_NAME_RE
    is the actual fix for that). Re-added after a real run showed the secondary QA
    vision model refusing to even look at 10/10 images in a round -- narrowing the
    checkpoint pool back to nsfw=false-tagged models is one lever on how often
    generated content lands in territory that model won't engage with at all,
    though it's a checkpoint-level signal, not a guarantee about any one image."""
    params = {"query": query, "types": "Checkpoint", "limit": limit, "sort": sort,
              "nsfw": "false" if nsfw is False else nsfw}
    return _get(f"{API}/models", params=params).json().get("items", [])


def resolve_from_search(query):
    """Search, then resolve the first candidate that is actually usable: SD1.5/SDXL,
    a scan-clean SafeTensor file, and enough downloads to trust it is not broken.
    Raises if nothing in the result set qualifies."""
    items = search_models(query)
    tried = []
    for item in items:
        versions = item.get("modelVersions") or []
        if not versions:
            continue
        if _ETHNICITY_MODEL_NAME_RE.search(item.get("name") or ""):
            continue
        stats = item.get("stats") or {}
        if stats.get("downloadCount", 0) < MIN_SEARCH_DOWNLOADS:
            continue
        version = versions[0]
        version.setdefault("modelId", item.get("id"))
        try:
            resolved = _resolved(version, item.get("id"), name=item.get("name"))
            log(f"picked {item['name']!r} from {len(items)} search results for "
                f"{query!r} ({stats.get('downloadCount', 0)} downloads)")
            return resolved
        except RuntimeError as e:
            tried.append(f"{item.get('name')}: {e}")
    raise RuntimeError(
        f"no usable checkpoint found for {query!r} among {len(items)} results"
        + (f" (rejected: {'; '.join(tried[:3])})" if tried else ""))


def search_candidates(query, limit=20):
    """Search results as (model_id, name, version_info) tuples, most-downloaded first,
    pre-filtered to the same download floor resolve_from_search() applies.

    The version_info here is the /api/v1/models list view's embedded modelVersions[0]
    -- it carries files[] (what _resolved() needs), but NOT the images' generation
    metadata: the list view only sets a hasMeta flag on each image, the prompt text
    itself is only present behind a direct /api/v1/model-versions/{id} lookup, unlike
    files[] -- verified live, tried assuming otherwise first and every candidate came
    back with 0 usable prompts despite hasMeta: true. Evaluating a candidate's showcase
    prompts is the caller's job (decide_reference below) and costs one extra request
    per candidate looked at; this generator only saves the ones that would be wasted on
    candidates too small or too broken to ever qualify."""
    items = search_models(query, limit=limit)
    for item in items:
        versions = item.get("modelVersions") or []
        if not versions:
            continue
        if _ETHNICITY_MODEL_NAME_RE.search(item.get("name") or ""):
            continue
        stats = item.get("stats") or {}
        if stats.get("downloadCount", 0) < MIN_SEARCH_DOWNLOADS:
            continue
        version = versions[0]
        version.setdefault("modelId", item.get("id"))
        yield item.get("id"), item.get("name"), version


def decide_reference(query, prompt_filter=None, pool_size=8, top_n_prompts=5, weights=None):
    """Evaluate up to `pool_size` search candidates, then pick among whichever qualify
    -- both which model and which of its showcase prompts. One model, one reference
    prompt, for the whole run this feeds, but not the SAME one every run.

    Earlier versions returned the first qualifying candidate's single highest-reaction
    prompt, every time -- since search_candidates() is sorted by download count and a
    given query resolves to the same ranking call after call, that made every run land
    on the exact same model and the exact same prompt (a live run kept coming back to
    one checkpoint's "chef in a kitchen" showcase image). Evaluating the whole pool
    first and choosing among what qualifies is what actually gives each run a
    different look, the way "search CivitAI fresh each time" was meant to.

    weights, an optional {model_id: float} map, lets a caller nudge the odds toward
    checkpoints with a good track record without ruling out the rest -- a plain
    shuffle (all weights equal) when not given. civitai.py doesn't know what "good"
    means (that's imageslides.py's QA-pass-rate history); it only turns numbers it's
    handed into biased odds.

    prompt_filter(prompt_text) -> bool lets a caller apply its own rules (subject,
    gender, whatever a niche cares about) without this module needing to know about
    them -- civitai.py stays subject-agnostic, same as harvest_from_model() and
    resolve_from_search() already are. Returns (resolved, prompt); raises RuntimeError
    with every rejected candidate's reason if none qualify."""
    prompt_filter = prompt_filter or (lambda p: True)
    weights = weights or {}
    tried = []
    qualifying = []  # [(model_id, name, version, [usable prompts])]
    for model_id, name, version in search_candidates(query, limit=pool_size):
        version_id = version.get("id")
        try:
            prompts = [p for p in harvest_from_model(model_id, version_id)
                      if prompt_filter(p["prompt"])]
        except Exception as e:
            tried.append(f"{name}: harvest failed ({type(e).__name__}: {str(e)[:80]})")
            continue
        if not prompts:
            tried.append(f"{name}: no usable on-subject prompt")
            continue
        qualifying.append((model_id, name, version, prompts))

    # Weighted sampling without replacement gives a full visit order, same shape as
    # the shuffle it replaces (every candidate still gets a turn if earlier ones fail
    # to resolve) -- just no longer uniform when a caller has opinions about some of
    # these models. A candidate with no weight entry (never tried, or untracked) gets
    # 1.0, the same baseline as "no preference at all" -- untried checkpoints keep
    # getting a fair shot, not shut out in favour of whatever already has a track record.
    pool = list(qualifying)
    ordered = []
    while pool:
        w = [weights.get(model_id, 1.0) for model_id, *_ in pool]
        pick = random.choices(range(len(pool)), weights=w, k=1)[0]
        ordered.append(pool.pop(pick))

    for model_id, name, version, prompts in ordered:
        try:
            resolved = _resolved(version, model_id, name=name)
        except RuntimeError as e:
            tried.append(f"{name}: {e}")
            continue
        # prompts is already sorted by reactions (harvest_from_model); picking among
        # the top few keeps quality while still varying which exact photo gets used.
        pick = random.choice(prompts[:top_n_prompts])
        return resolved, pick
    raise RuntimeError(
        f"no usable model+prompt for {query!r}"
        + (f" (tried {len(tried)}: {'; '.join(tried[:4])})" if tried else " (no search results)"))


def download(resolved, dest_dir=None, chunk_size=1 << 20):
    """Fetch the checkpoint to a local file, cached by version id so a repeat run (or
    a repeat niche in the same run) does not pay for the same download twice."""
    dest_dir = Path(dest_dir or CACHE_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{resolved['version_id']}-{resolved['filename']}"
    if dest.exists() and dest.stat().st_size > 0:
        log(f"already cached: {dest}")
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    # resolved["url"] is civitai.com's own /api/download endpoint, which 307s to signed
    # storage -- see _download_headers for why the key goes on the first hop only.
    with requests.get(resolved["url"], headers=_download_headers(resolved["url"]),
                      stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
        if total and written != total:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"download incomplete: {written} of {total} bytes")
    tmp.rename(dest)
    log(f"downloaded {dest.stat().st_size / 1024 / 1024:.0f}MB -> {dest}")
    return dest


def resolve_final_url(url, timeout=15):
    """Follow url's redirect chain and return the final URL, without downloading the
    file body. resolve()'s "url" is civitai.com's own download endpoint, which 307s to
    a pre-signed Cloudflare R2 link -- confirmed live that civitai.com's own domain
    451s requests from Kaggle's network while the R2 storage layer underneath is
    independently reachable, so kaggle_imagegen.py resolves the final R2 link here (in
    CI, unblocked) and hands Kaggle that link directly instead of civitai.com's own
    endpoint."""
    with requests.get(url, headers=_download_headers(url),
                      stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        return r.url


def resolve_and_download(spec, dest_dir=None):
    resolved = resolve(spec)
    path = download(resolved, dest_dir)
    resolved["path"] = path
    return resolved


# Posts whose own prompt indicates content this pipeline must never touch, regardless
# of CivitAI's own NSFW rating for the image.
_BAD_PROMPT = re.compile(
    r"\b(?:child|kid|loli|shota|teen(?:ager)?|minor|schoolgirl|infant|toddler)\b", re.I)
# search_models() applies no nsfw filter (operator preference), and a version's own
# showcase images (harvest_from_model, the primary prompt source) are read directly
# off /api/v1/model-versions/{id} with no nsfw filter either way -- a live search
# surfaced a showcase prompt containing unambiguous hardcore sexual-act terms, not
# filtered by anything upstream. This niche wants sexy, not explicit -- reject at
# the source, the same way celebrity-likeness and non-photo-style prompts already
# are.
_EXPLICIT_RE = re.compile(
    r"\b(?:blowjob|cum|cumshot|penetration|anal|orgasm|masturbat\w*|"
    r"deepthroat|creampie|gangbang|bukkake|fellatio|cunnilingus|hentai|rule ?34|"
    r"porn\w*|xxx)\b", re.I)
# Once search picks any qualifying model rather than one of a few pre-vetted presets,
# it can surface "merge"-style checkpoints whose own showcase prompts target a real
# person's likeness -- a live search hit "solo mid shot portrait photo of [Dakota
# Johnson|Maggie Lindemann] as [Hailey Clauson|Hailey Grice] as a real life version of
# ((Queen Elsa))". "[Name|Name]" is Automatic1111 wildcard-alternation syntax used
# specifically for celebrity/character face-swaps; NEGATIVE_HARD already lists
# "celebrity likeness, real person" but a negative term is not a reliable override
# against a strongly-named positive prompt, so this is rejected at the source instead.
_REAL_PERSON_RE = re.compile(
    r"\[[^\[\]|]+\|[^\[\]|]+\]|real[- ]?life version of|deepfake|face ?swap", re.I)
# Real scraped prompts name an explicit age often enough that this needs its own check:
# a live sample included "18 y.o" and "23 y.o" alongside "26 y.o" and "30 y.o". 18 is a
# legal adult almost everywhere, but an explicit low-20s-or-under age callout is exactly
# the kind of borderline signal an automated pipeline representing a real account should
# never carry, regardless of technicality -- reject it at the source rather than lean on
# the SAFETY_PREFIX override or supervisor.py to catch it downstream.
# No upper bound existed before this -- a harvested prompt naming an explicit older
# age (e.g. "45 year old woman") passed through untouched, only ever checked against
# the lower floor. 25-35 is the target range for this niche.
_MIN_PROMPT_AGE = 25
_MAX_PROMPT_AGE = 35
_AGE_RE = re.compile(r"\b(\d{1,2})\s*[-\s]?(?:y\.?o\.?|years?[-\s]old|yo)\b", re.I)
# Operator preference: exclude Chinese-appearance prompts specifically (not a wider
# "Asian" exclusion, which was not asked for) -- reject at the source the same way
# celebrity likeness and explicit terms already are, since a harvested showcase
# prompt naming an ethnicity is a strong, literal signal a negative_prompt term
# alone would not reliably override.
_ETHNICITY_EXCLUDE_RE = re.compile(r"\bchinese\b", re.I)
# Automatic1111/ComfyUI control syntax embedded in the prompt text: <lora:name:weight>,
# <hypernet:...>, <embedding:...>. diffusers does not parse this convention, so it
# passes straight to the CLIP tokenizer as literal junk tokens -- angle brackets, colons
# and all -- rather than doing anything. Harmless to leave in, but it dilutes the prompt
# for no benefit, so it is stripped.
_A1111_TAG_RE = re.compile(r"<(?:lora|lyco|hypernet|embedding):[^>]+>", re.I)


def _clean_prompt_text(text):
    text = _A1111_TAG_RE.sub("", text)
    text = re.sub(r"\s*,\s*(?:,\s*)+", ", ", text)  # collapse ", , ," left by removal
    return re.sub(r"\s{2,}", " ", text).strip(" ,")


def _usable(meta, stats=None):
    prompt = _clean_prompt_text(meta.get("prompt") or "")
    if not prompt or _BAD_PROMPT.search(prompt) or _REAL_PERSON_RE.search(prompt) \
       or _EXPLICIT_RE.search(prompt) or _ETHNICITY_EXCLUDE_RE.search(prompt):
        return None
    ages = [int(m) for m in _AGE_RE.findall(prompt)]
    if any(age < _MIN_PROMPT_AGE or age > _MAX_PROMPT_AGE for age in ages):
        return None
    stats = stats or {}
    reactions = sum(stats.get(k, 0) for k in
                    ("likeCount", "heartCount", "laughCount", "cryCount"))
    width, height = _parse_size(meta.get("Size"))
    return {"prompt": prompt,
           "negative_prompt": _clean_prompt_text(meta.get("negativePrompt") or ""),
           "reactions": reactions,
           # The checkpoint's own creator posted this image and got real engagement
           # for it -- these are the settings that actually worked, not a guess.
           # Only meaningful when present; None means "no opinion, use our defaults".
           "width": width, "height": height,
           "steps": meta.get("steps"), "cfg_scale": meta.get("cfgScale"),
           "sampler": meta.get("sampler")}


def _parse_size(size_str):
    """CivitAI's own "Size" field, e.g. "576x864" -> (576, 864). Malformed or absent
    -> (None, None), meaning the caller falls back to its own defaults."""
    if not size_str or "x" not in str(size_str):
        return None, None
    try:
        w, h = str(size_str).lower().split("x", 1)
        return int(w.strip()), int(h.strip())
    except ValueError:
        return None, None


def _harvest_from_info(info, version_id=None):
    images = info.get("images") or []
    out = [r for img in images if (r := _usable(img.get("meta") or {}, img.get("stats")))]
    out.sort(key=lambda p: p["reactions"], reverse=True)
    log(f"{len(out)} usable prompts from {len(images)} showcase images"
        + (f" (model version {version_id})" if version_id else ""))
    return out


def harvest_from_model(model_id, version_id):
    """A model version's own showcase images and their generation metadata -- verified
    live to work with no API key at all, unlike the general gallery search below.

    This is the primary harvest path, and it is better suited to the job than gallery
    search would be even if that worked cleanly: these are the examples the checkpoint's
    own author chose to demonstrate it, generated with that exact version, so the
    vocabulary and trigger words already match the model being run. The pool is small
    (CivitAI shows roughly 10-20 per version), which is still several times more than
    one video needs."""
    info = _version_info(model_id, version_id)
    return _harvest_from_info(info, version_id)


def harvest_from_gallery(model_version_id=None, query=None, limit=50, period="Month",
                         sort="Most Reactions"):
    """General gallery search. Long-standing CivitAI platform bug (their own tracker:
    civitai/civitai#1037, #1297, #1302): the public /api/v1/images endpoint returns
    meta: null for essentially every post regardless of authentication, engagement, or
    filters -- verified live, 0 of 50 top-reactions-all-time posts carried a prompt.
    Kept only as a secondary top-up when the model-showcase pool above is too small;
    do not rely on this alone."""
    params = {"limit": limit, "sort": sort, "period": period}
    if model_version_id:
        params["modelVersionId"] = model_version_id
    if query:
        params["query"] = query
    items = _get(f"{API}/images", params=params).json().get("items", [])
    out = [r for img in items if (r := _usable(img.get("meta") or {}, img.get("stats")))]
    out.sort(key=lambda p: p["reactions"], reverse=True)
    if items and not out:
        log(f"gallery search returned {len(items)} posts, 0 with usable prompt metadata "
            "(known CivitAI API limitation, not specific to this query)")
    return out


def safe_harvest(model_id=None, model_version_id=None, query=None, limit=50):
    """Never raises. Empty list means the caller falls back to its own prompt source.

    model_id + model_version_id (both known, e.g. from resolve()) uses the reliable
    showcase-image path; anything else falls through to gallery search, which is best
    effort only given the bug above."""
    try:
        if model_id and model_version_id:
            out = harvest_from_model(model_id, model_version_id)
            if len(out) >= 3:
                return out
            log(f"only {len(out)} showcase prompts; topping up from the gallery")
            out += harvest_from_gallery(model_version_id=model_version_id, query=query,
                                        limit=limit)
            return out
        return harvest_from_gallery(model_version_id=model_version_id, query=query, limit=limit)
    except Exception as e:
        log(f"unavailable ({type(e).__name__}: {str(e)[:120]}); "
            "the caller will fall back to its own prompts")
        return []
