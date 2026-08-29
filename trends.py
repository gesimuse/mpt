"""Trend signal for caption_writer's hashtag rubric -- what is popular right now, so
the hashtags on a post are not just fresh-per-post but pointed at live reach.

What was tried and does NOT work, so nobody re-tries it:

  * TikTok Creative Center's own trending-hashtag endpoint
    (ads.tiktok.com/creative_radar_api/v1/popular_trend/hashtag/list) answers every
    unauthenticated request with HTTP 200 and body {"code":40101,"msg":"no permission"}
    -- checked live, with and without the anonymous-user-id/timestamp/user-sign headers
    that circulate for it. It needs a real browser session. So it is attempted ONLY when
    TIKTOK_CC_COOKIE is set (paste the Cookie header from a logged-in Creative Center
    tab); unset, we do not fire a request that is known to fail on every run.

  * CivitAI's `period` filter as a "what's hot this week" signal. /api/v1/models with
    query + period=Week (or Month) returns an EMPTY item list -- checked live across
    Most Downloaded / Most Liked. period only works on an unqueried browse, which is
    useless for a niche-specific search. `sort` likewise doesn't move the results once
    a query is present; relevance dominates.

  * CivitAI's gallery search by query for trending prompt text -- returns posts with no
    usable generation metadata (civitai.py already documents this limitation).

So the working signal is the manual one: `trend_hashtags` in niches.json, a short list
the account owner refreshes from Creative Center in the browser whenever they feel like
it. Stale-but-real beats fabricated. The primary fix for "hashtags always come the same"
is not this module at all -- it is that caption_writer's LLM call now actually completes
(see llm.py); this only aims it at what is currently getting reach.

Nothing here ever raises: a caption must still get written when there is no signal.
"""
import os, time

import requests

CREATIVE_CENTER_URL = (
    "https://ads.tiktok.com/creative_radar_api/v1/popular_trend/hashtag/list")
COUNTRY = os.environ.get("TRENDS_COUNTRY", "US")
PERIOD_DAYS = int(os.environ.get("TRENDS_PERIOD_DAYS", "7"))
LIMIT = int(os.environ.get("TRENDS_LIMIT", "15"))
TIMEOUT = int(os.environ.get("TRENDS_TIMEOUT", "20"))

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/126.0.0.0 Safari/537.36")


def log(msg): print(f"[trends] {msg}", flush=True)


def _normalize(names):
    out = []
    for name in names:
        name = (name or "").strip().lstrip("#").strip()
        if name:
            out.append(f"#{name}")
    return out


def _fetch_creative_center(cookie):
    r = requests.get(
        CREATIVE_CENTER_URL,
        params={"page": 1, "limit": LIMIT, "period": PERIOD_DAYS,
                "country_code": COUNTRY, "sort_by": "popular"},
        headers={"User-Agent": _UA, "Accept": "application/json", "Cookie": cookie,
                 "Referer": "https://ads.tiktok.com/business/creativecenter/"
                            "inspiration/popular/hashtag/pc/en"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    # Creative Center signals auth failure in the BODY with HTTP 200, so status alone
    # proves nothing -- this is the check that actually catches an expired cookie.
    if payload.get("code") not in (0, None):
        raise RuntimeError(f"creative center said {payload.get('code')}: "
                           f"{payload.get('msg')}")
    items = (payload.get("data") or {}).get("list") or []
    return _normalize(item.get("hashtag_name") or item.get("hashtag")
                      for item in items)


def tiktok_hashtags(state=None, niche=None, max_age_hours=24):
    """Trending hashtags to hint the caption rubric with, or [] when there is no
    signal. Cached on `state` so five crons a day cost one request, not five."""
    cache = (state or {}).get("trends") or {}
    fresh = cache.get("hashtags") and (
        time.time() - cache.get("fetched_at", 0)) < max_age_hours * 3600
    if fresh:
        return cache["hashtags"]

    cookie = os.environ.get("TIKTOK_CC_COOKIE", "").strip()
    if cookie:
        try:
            tags = _fetch_creative_center(cookie)
            if tags:
                if state is not None:
                    state["trends"] = {"hashtags": tags, "fetched_at": time.time(),
                                       "source": "creative_center",
                                       "country": COUNTRY, "period_days": PERIOD_DAYS}
                log(f"trending now ({COUNTRY}, {PERIOD_DAYS}d): {' '.join(tags[:8])}")
                return tags
            log("Creative Center returned an empty list")
        except Exception as e:
            log(f"Creative Center unavailable ({type(e).__name__}: {str(e)[:120]}) -- "
                "cookie probably expired; falling back")

    manual = _normalize((niche or {}).get("trend_hashtags") or [])
    if manual:
        return manual
    # A stale cache still beats nothing: last week's popular tags are real tags.
    return cache.get("hashtags") or []
