"""Discussion-grounded topic research.

Instead of asking the model to invent a topic in a vacuum, pull what people are
actually arguing about today and ground the topic in that. Sources, all free and
keyless:

  reddit json   richest (scores, comments, body text) but Reddit now 403s anonymous
                JSON from datacenter IPs -- including GitHub Actions runners
  reddit rss    same subreddits, titles only, still served to those IPs
  hacker news   Algolia API, no key, no blocking
  stackexchange 300 keyless requests/day, real developer pain points

Every failure path is soft: the caller falls back to prompt-only topic generation,
so a blocked source costs relevance, never a run.
"""
import html, json, os, random, re, time
from pathlib import Path

import requests

from llm import nim_json

ROOT = Path(__file__).resolve().parent
SAAS_FILE = ROOT / "saas_ideas.json"
UA = {"User-Agent": "python:mpt-autopilot:v1.1 (by /u/mpt-autopilot)"}
MIN_POSTS = 5
DIGEST_POSTS = 20   # how many make it into the prompt; the rest only affect ranking


def log(msg): print(f"[research] {msg}", flush=True)


def _get(url, attempts=3, headers=None, **kw):
    """GET with backoff. 429 and 5xx are retried, 403 is not -- it means the source
    is blocking this IP outright and no amount of waiting changes that."""
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=headers or UA, timeout=30, **kw)
            if r.status_code == 429:
                wait = min(float(r.headers.get("Retry-After", 0) or 2 ** i * 5), 60)
                log(f"rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                last = RuntimeError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                raise
            last = e
        except requests.RequestException as e:
            last = e
        if i < attempts - 1:
            time.sleep(2 ** i * 3)
    raise last


_oauth_token = None


def reddit_token():
    """App-only OAuth. Reddit serves oauth.reddit.com to datacenter IPs that it refuses
    on www.reddit.com, so this is the only path that works from a CI runner. Free:
    create a 'script' app at reddit.com/prefs/apps and set REDDIT_CLIENT_ID/SECRET."""
    global _oauth_token
    if _oauth_token:
        return _oauth_token
    cid = (os.environ.get("REDDIT_CLIENT_ID") or "").strip()
    secret = (os.environ.get("REDDIT_CLIENT_SECRET") or "").strip()
    if not cid or not secret or cid.lower() == "xxxx":
        raise RuntimeError("REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET not set")
    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(cid, secret), headers=UA, timeout=30,
        data={"grant_type": "client_credentials"},
    )
    r.raise_for_status()
    _oauth_token = r.json()["access_token"]
    return _oauth_token


def _parse_listing(payload):
    posts = []
    for child in payload["data"]["children"]:
        d = child["data"]
        if d.get("stickied") or d.get("over_18"):
            continue
        posts.append({
            "title": d.get("title", ""),
            "text": (d.get("selftext") or "")[:400],
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "id": f"reddit:{d.get('id')}",
            "url": "https://reddit.com" + (d.get("permalink") or ""),
            "source": f"r/{d.get('subreddit', '')}",
        })
    return posts


def fetch_subreddit_oauth(sub, listing="hot", limit=15):
    token = reddit_token()
    r = _get(f"https://oauth.reddit.com/r/{sub}/{listing}?limit={limit}&raw_json=1",
             headers={**UA, "Authorization": f"bearer {token}"})
    return _parse_listing(r.json())


ARCTIC_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"


def fetch_subreddit_arctic(sub, limit=60, hours_back=None, min_age_hours=18):
    """Arctic Shift -- a free, keyless, open-source Reddit archive API. It answers from
    its own servers, so it works from IPs Reddit refuses, no app registration needed.

    Scores are captured near post creation, so anything newer than min_age_hours reads
    as 1 point. Asking for a window that already accumulated votes gives real ranking
    while staying recent enough to reflect what people are on about this week.

    The window length is randomised: a fixed one returns the same top posts every run,
    which produced the same pain points and, eventually, the same video twice."""
    hours_back = hours_back or random.choice((72, 96, 144, 216))
    now = int(time.time())
    r = _get(ARCTIC_URL, params={
        "subreddit": sub, "limit": limit, "sort": "desc", "sort_type": "created_utc",
        "after": now - hours_back * 3600, "before": now - min_age_hours * 3600,
    })
    posts = []
    for d in r.json().get("data", []):
        if d.get("stickied") or d.get("over_18") or d.get("removed_by_category"):
            continue
        title = (d.get("title") or "").strip()
        if not title:
            continue
        posts.append({
            "title": title,
            "text": (d.get("selftext") or "")[:400],
            "score": d.get("score") or 0,
            "num_comments": d.get("num_comments") or 0,
            "id": f"reddit:{d.get('id')}",
            "url": "https://reddit.com" + (d.get("permalink") or f"/r/{sub}"),
            "source": f"r/{sub}",
        })
    return posts


def fetch_subreddit_json(sub, listing="hot", limit=15):
    r = _get(f"https://www.reddit.com/r/{sub}/{listing}.json?limit={limit}&raw_json=1")
    return _parse_listing(r.json())


def fetch_subreddit_rss(sub, listing="hot", limit=15):
    """Fallback for IPs Reddit blocks from the JSON endpoints. Titles only: the RSS
    feed carries no score, so ranking falls back to feed order. One retry only --
    a blocked IP stays blocked, and other sources are waiting."""
    r = _get(f"https://www.reddit.com/r/{sub}/{listing}.rss?limit={limit}", attempts=2)
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    posts = []
    for e in entries:
        m = re.search(r"<title>(.*?)</title>", e, re.S)
        if not m:
            continue
        title = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        body = re.search(r"<content[^>]*>(.*?)</content>", e, re.S)
        text = html.unescape(re.sub(r"<[^>]+>", " ", body.group(1))) if body else ""
        posts.append({"title": title, "text": re.sub(r"\s+", " ", text)[:400],
                      "score": 0, "num_comments": 0})
    return posts


_reddit_blocked = False


def fetch_subreddit(sub, listing="hot", limit=15):
    """Four ways in, best first:

      oauth    official API, needs free app credentials, richest and most current
      arctic   Arctic Shift, keyless and unblocked, works from CI runners
      json     www.reddit.com, 403s from datacenter IPs
      rss      same, and rate limits almost immediately

    Reddit blocks by IP range rather than per subreddit, so once anonymous access fails
    it is remembered and later subreddits skip straight past it."""
    global _reddit_blocked
    if os.environ.get("REDDIT_CLIENT_ID"):
        try:
            return fetch_subreddit_oauth(sub, listing, limit)
        except Exception as e:
            log(f"r/{sub}/{listing} oauth failed ({type(e).__name__}: {str(e)[:70]})")
    try:
        posts = fetch_subreddit_arctic(sub)
        if posts:
            return posts
        log(f"r/{sub} arctic returned nothing, trying anonymous")
    except Exception as e:
        log(f"r/{sub} arctic failed ({type(e).__name__}: {str(e)[:70]}), trying anonymous")

    if _reddit_blocked:
        raise RuntimeError("reddit is blocking this IP (skipped)")
    try:
        return fetch_subreddit_json(sub, listing, limit)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        log(f"r/{sub}/{listing} json blocked ({code}), trying rss")
    try:
        return fetch_subreddit_rss(sub, listing, limit)
    except Exception:
        _reddit_blocked = True
        raise


def fetch_ask_hn(limit=20):
    """Ask HN threads: people posing real questions, not link submissions."""
    url = f"https://hn.algolia.com/api/v1/search?tags=ask_hn&hitsPerPage={limit}"
    hits = _get(url).json().get("hits", [])
    return [{
        "title": h.get("title") or "",
        "text": (h.get("story_text") or "")[:400],
        "score": h.get("points") or 0,
        "num_comments": h.get("num_comments") or 0,
        "id": f"hn:{h.get('objectID')}",
        "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
        "source": "Ask HN",
    } for h in hits if h.get("title")]


def fetch_hn(query=None, limit=20):
    """Front page when no query, otherwise recent stories matching it."""
    if query:
        # search_by_date and search rank differently; alternating keeps the pool moving
        endpoint = random.choice(("search", "search_by_date"))
        url = (f"https://hn.algolia.com/api/v1/{endpoint}?tags=story&hitsPerPage="
               f"{limit}&query={requests.utils.quote(query)}")
    else:
        url = f"https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage={limit}"
    hits = _get(url).json().get("hits", [])
    return [{
        "title": h.get("title") or h.get("story_title") or "",
        "text": (h.get("story_text") or h.get("comment_text") or "")[:400],
        "score": h.get("points") or 0,
        "num_comments": h.get("num_comments") or 0,
        "id": f"hn:{h.get('objectID')}",
        "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
        "source": "Hacker News",
    } for h in hits if h.get("title") or h.get("story_title")]


def fetch_stackexchange(tag, site="stackoverflow", limit=20, sort=None):
    """Keyless tier is 300 requests/day, plenty for a handful of runs. The sort is
    rotated so consecutive runs do not see an identical question list."""
    sort = sort or random.choice(("week", "month", "activity"))
    r = _get("https://api.stackexchange.com/2.3/questions", params={
        "order": "desc", "sort": sort, "tagged": tag, "site": site,
        "pagesize": limit, "filter": "default",
    })
    return [{
        "title": html.unescape(q.get("title", "")),
        "text": "",
        "score": q.get("score", 0),
        "num_comments": q.get("answer_count", 0),
        "id": f"se:{q.get('question_id')}",
        "url": q.get("link", ""),
        "source": f"Stack Overflow [{tag}]",
    } for q in r.json().get("items", [])]


def gather(niche):
    """Collect from every configured source. One dead source must not sink the rest."""
    posts = []
    subs = list(niche.get("subreddits", []))
    random.shuffle(subs)
    # Only the official API distinguishes hot from rising; Arctic Shift answers with one
    # time-windowed query, so asking twice would just fetch the same rows again.
    listings = ("hot", "rising") if os.environ.get("REDDIT_CLIENT_ID") else ("hot",)
    for sub in subs:
        for listing in listings:
            try:
                posts += fetch_subreddit(sub, listing)
            except Exception as e:
                log(f"r/{sub}/{listing} failed: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)  # polite to an endpoint we do not authenticate against

    for query in niche.get("hn_queries", []):
        try:
            posts += fetch_hn(query or None)
        except Exception as e:
            log(f"hn '{query}' failed: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(1)

    for tag in niche.get("stackexchange_tags", []):
        try:
            posts += fetch_stackexchange(tag)
        except Exception as e:
            log(f"stackexchange '{tag}' failed: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(1)

    seen, out = set(), []
    for p in sorted(posts, key=lambda p: p["score"] + p["num_comments"], reverse=True):
        title = p["title"].strip()
        if title and title.lower() not in seen:
            seen.add(title.lower())
            out.append(p)
    return out[:40]


def research_topic(niche, used_topics):
    """Return (topic, pain_points) grounded in live discussion.
    Raises on failure -- the caller falls back to prompt-only topic generation."""
    posts = gather(niche)
    if len(posts) < MIN_POSTS:
        raise RuntimeError(f"not enough research data ({len(posts)} posts)")
    log(f"{len(posts)} posts gathered")
    # Keep the prompt small. Forty posts with body text each made the request slow
    # enough to hit NIM's read timeout, and the titles carry most of the signal.
    digest = "\n".join(
        f"- [{p['score']}pts/{p['num_comments']}c] {p['title']}"
        + (f" :: {p['text'][:120]}" if p["text"] else "")
        for p in posts[:DIGEST_POSTS]
    )
    avoid = "; ".join(used_topics[-30:]) or "(none)"
    guidance = niche.get("research_prompt") or (
        "You analyze live discussions for a short-video creator. From the posts, extract "
        "the 3 most emotionally charged, recurring pain points -- specific problems people "
        "are struggling with or arguing about -- then craft ONE short-video topic that "
        "speaks directly to the strongest one."
    )
    result = nim_json(
        guidance + ' JSON schema: {"pain_points": ["...", "...", "..."], "topic": "..."} '
        "Topic under 14 words, hook-style, concrete, no clickbait cliches.",
        f"Niche: {niche['name']}\nAlready-used topics to avoid: {avoid}\n\nToday's posts:\n{digest}",
        max_tokens=700,
    )
    topic = (result.get("topic") or "").strip()
    if not topic:
        raise RuntimeError("model returned no topic")
    if niche.get("track_saas_ideas"):
        _log_saas_ideas(niche["id"], result.get("pain_points", []))
    return topic, result.get("pain_points", [])


def _log_saas_ideas(niche_id, pain_points):
    try:
        data = json.loads(SAAS_FILE.read_text()) if SAAS_FILE.exists() else []
    except json.JSONDecodeError:
        data = []
    ts = time.strftime("%Y-%m-%d")
    for p in pain_points:
        data.append({"date": ts, "niche": niche_id, "pain_point": p})
    SAAS_FILE.write_text(json.dumps(data, indent=2))
    log(f"logged {len(pain_points)} pain points to saas_ideas.json")
