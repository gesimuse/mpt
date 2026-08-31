#!/usr/bin/env python3
"""Union two versions of posted.json so concurrent writers stop losing runs.

The problem this replaces: posted.json's "uploads" is an append-only array written by
BOTH autopilot workflows and by picker.html's Contents-API writes. Two writers
appending in the same place produce a textual conflict git cannot resolve, and the
persist step's `git pull --rebase` then failed -- five times in a row, because the
retry loop aborted and re-ran the identical rebase, which conflicts identically every
time. A live run (33318817540, 2026-08-30) generated images, pushed a real TikTok
draft, and then lost its own state entirely; commit 5c3bef4 is someone recovering an
earlier occurrence by hand from CI logs.

Rebasing is the wrong tool here. This merges by VALUE instead: take the remote's
version as the base, add anything of ours it doesn't have, and write the result. The
caller then commits on top of the remote with no rebase at all, so there is nothing to
conflict.

Usage:  merge_state.py OURS.json THEIRS.json OUT.json
"""
import json
import sys
from pathlib import Path


def log(msg): print(f"[merge_state] {msg}", flush=True)


def _upload_key(u):
    """What makes an upload entry the same entry on both sides.

    tiktok_post_id is the only truly unique thing TikTok gives us, but it is absent on
    DRY_RUN entries, on picker's manual uploads, and on any run where the init call
    itself failed -- so it cannot be the whole key. ts+niche is stable for everything
    autopilot writes (one entry per push, stamped at write time) and is what the picker
    itself already uses to find an entry to edit."""
    return (u.get("ts"), u.get("niche"), u.get("tiktok_post_id"))


def _merge_uploads(ours, theirs):
    merged = list(theirs)
    seen = {_upload_key(u) for u in theirs}
    added = 0
    for u in ours:
        k = _upload_key(u)
        if k in seen:
            continue
        seen.add(k)
        merged.append(u)
        added += 1
    # Chronological, so CAPTIONS.md's "last N" slice and the picker's reverse scan both
    # see a sensible order regardless of which side an entry came from.
    merged.sort(key=lambda u: u.get("ts") or "")
    return merged, added


def _merge_counters(ours, theirs):
    """model_stats / theme_stats are monotonically increasing counters.

    Take whichever side has seen more samples rather than summing: both sides share a
    common history, so adding them would double-count everything before the fork. The
    loser's few extra observations are worth less than a corrupted denominator -- these
    only ever feed soft selection weights (imageslides._model_weights)."""
    out = dict(theirs)
    for key, mine in (ours or {}).items():
        other = out.get(key)
        if other is None or (mine.get("used", 0) > other.get("used", 0)):
            out[key] = mine
    return out


def merge(ours, theirs):
    out = dict(theirs)
    uploads, added = _merge_uploads(ours.get("uploads") or [],
                                    theirs.get("uploads") or [])
    out["uploads"] = uploads

    topics = dict(theirs.get("topics") or {})
    for niche, items in (ours.get("topics") or {}).items():
        combined = list(topics.get(niche, [])) + list(items)
        # dict.fromkeys de-duplicates while preserving first-seen order.
        topics[niche] = list(dict.fromkeys(combined))
    out["topics"] = topics

    for field in ("model_stats", "theme_stats"):
        if ours.get(field) or theirs.get(field):
            out[field] = _merge_counters(ours.get(field), theirs.get(field) or {})

    # trends is a cache with a timestamp; the fresher fetch wins outright.
    ot, tt = ours.get("trends"), theirs.get("trends")
    if ot or tt:
        out["trends"] = max(
            [x for x in (ot, tt) if x], key=lambda t: t.get("fetched_at", 0))

    # Anything either side has that this function doesn't know about is preserved
    # rather than dropped -- a future field must not be silently deleted by a merge.
    for k, v in ours.items():
        out.setdefault(k, v)
    return out, added


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ours_p, theirs_p, out_p = (Path(a) for a in sys.argv[1:])
    ours = json.loads(ours_p.read_text()) if ours_p.exists() else {}
    theirs = json.loads(theirs_p.read_text()) if theirs_p.exists() else {}
    merged, added = merge(ours, theirs)
    out_p.write_text(json.dumps(merged, indent=2))
    log(f"merged: {len(theirs.get('uploads') or [])} remote + {added} of ours "
        f"= {len(merged['uploads'])} uploads")


if __name__ == "__main__":
    main()
