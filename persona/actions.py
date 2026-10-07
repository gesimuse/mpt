"""The laptop-free schedule, run by .github/workflows/persona_kaggle.yml.

    python -m persona.actions render <slug>         once a day, early
    python -m persona.actions post   <slug> <k> <n> at each of n slots (k = 0..n-1)
    python -m persona.actions votes  <slug>         every 15 min: apply 👍/👎 presses

render: collect pending 👍/👎 presses, apply them to her votes and storyline, sync
        the private Kaggle dataset, write the day's scenes (LLM, HF_TOKEN), and
        push the Kaggle kernel: PERSONA_CLOUD_IMAGES images plus
        PERSONA_CLOUD_VIDEOS videos of the first good ones. Returns at once; the
        kernel takes ~1-2 h on the T4.
post:   download that kernel's output and post this slot's share into
        PERSONA_CHAT_ID with 👍/👎, plus anything an earlier slot could not post
        because the render was not finished yet. Then collect votes again.

State lives in the private dataset, which is the source of truth in cloud mode:
votes.json, storyline.jsonl, and posted.json (ids already sent). The runner has
no persona state of its own; it downloads her each time.

Votes: the bot's pending updates are read with getUpdates. That bot must not be
long-polled anywhere else at the same time (the laptop persona bot and
local/wan2gp_bot.py), or each would take the other's updates. Set the repo
variable PERSONA_CLOUD_VOTES=0 to post without reading votes.

Social lane only: kaggle.push renders lane="social" with models that pass
registry.check(kaggle=True). The Fanvue lane never runs here.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import character, config, kaggle, scenes, tg


def log(msg):
    print(f"[persona.actions] {tg.redact(msg)}", flush=True)


def today():
    return time.strftime("%Y-%m-%d", time.gmtime())


def fetch(slug):
    os.environ.setdefault("PERSONA_HOME", tempfile.mkdtemp(prefix="persona-home-"))
    home = config.home()
    subprocess.run(["kaggle", "datasets", "download", kaggle.dataset_id(slug), "--unzip", "-p", str(home / slug)],
                   env=kaggle._env(), check=True, capture_output=True)
    kaggle.restore_layout(home / slug)
    (home / "settings.json").write_text(json.dumps({"active": slug}))
    return character.Character(slug)


def _posted(char):
    p = char.dir / "posted.json"
    return json.loads(p.read_text()) if p.exists() else {"date": "", "ids": []}


UPDATE_TYPES = ["callback_query", "channel_post", "message"]


def _item_id_from(msg):
    """The #id on the last line of a caption we posted."""
    cap = (msg or {}).get("caption") or ""
    last = cap.strip().splitlines()[-1] if cap.strip() else ""
    return last[1:].strip() if last.startswith("#") else ""


def _videos(char):
    p = char.dir / "videos.json"
    return json.loads(p.read_text()) if p.exists() else {"queue": [], "running": []}


def _save_videos(char, v):
    (char.dir / "videos.json").write_text(json.dumps(v))


def _queue_video(char, item_id, msg, motion, items):
    """A 🎬 tap or a reply: queue that photo for the next video job."""
    photo = (msg.get("photo") or [None])[-1]
    if not photo:
        return "That is not a photo."
    v = _videos(char)
    for q in v["queue"]:
        if q["item"] == item_id:
            q["motion"] = motion or q.get("motion")
            _save_videos(char, v)
            return "Motion prompt updated." if motion else "Already queued."
    item = items.get(item_id, {})
    v["queue"].append({"id": f"{item_id}-{int(time.time())}", "item": item_id, "file_id": photo["file_id"],
                       "chat": msg["chat"]["id"], "message_id": msg["message_id"],
                       "motion": motion or item.get("scene", {}).get("motion") or "",
                       "tags": item.get("tags", [])})
    _save_videos(char, v)
    return "🎬 Queued. The video comes as a reply here in ~40-60 min."


def collect_votes(char, items):
    """Apply pending 👍/👎 presses, 🎬 taps and text replies (motion prompts) to
    photos. `items` maps id -> item meta for what this run knows about (today's
    render). Returns how many changes were made."""
    if not config.flag("PERSONA_CLOUD_VOTES", "1"):
        return 0
    try:
        updates = tg.call("getUpdates", timeout=0, allowed_updates=UPDATE_TYPES)
    except Exception as e:
        log(f"could not read updates: {e}")
        return 0
    chat = str(tg.chat("social"))
    applied, voted = 0, set(_posted(char).get("voted", []))
    for u in updates:
        post = u.get("channel_post") or u.get("message")
        if post:
            # A text reply to one of her photos = a video with that motion prompt.
            parent = post.get("reply_to_message") or {}
            item_id = _item_id_from(parent)
            if str(post.get("chat", {}).get("id")) == chat and item_id and post.get("text") \
                    and not post["text"].startswith("/"):
                note = _queue_video(char, item_id, parent, post["text"].strip(), items)
                tg.text(chat, note, reply_to=post["message_id"])
                applied += 1
            continue
        cq = u.get("callback_query") or {}
        data = cq.get("data", "")
        msg = cq.get("message") or {}
        if data.startswith("pv:vid:"):
            item_id = data.split(":", 2)[2]
            tg.answer(cq.get("id"), _queue_video(char, item_id, msg, None, items))
            applied += 1
            continue
        if not data.startswith(("pv:up:", "pv:dn:")):
            continue
        _, act, item_id = data.split(":", 2)
        if item_id in voted:
            tg.answer(cq.get("id"), "Already voted.")
            continue
        # A photo from an older batch (or made on the laptop) has no tags here; the
        # vote is still recorded and shown, it just cannot steer tags.
        item = items.get(item_id) or {"id": item_id, "kind": "image", "tags": []}
        char.vote(item, 1 if act == "up" else -1)
        if act == "up" and item.get("kind") == "image" and item.get("scene", {}).get("beat"):
            char.add_beat(item["scene"]["beat"], item_id)
        voted.add(item_id)
        applied += 1
        tg.answer(cq.get("id"), "Noted 👍" if act == "up" else "Noted 👎")
        label = "✓ 👍" if act == "up" else "✓ 👎"
        rows = [[(label, f"pv:{act}:{item_id}")]]
        if item.get("kind") == "image":
            rows[0].append(("🎬 Video", f"pv:vid:{item_id}"))
        tg.set_buttons(msg.get("chat", {}).get("id"), msg.get("message_id"), tg.kb(*rows))
    if updates:
        # Confirm them all, so the next run does not see them again.
        tg.call("getUpdates", timeout=0, offset=updates[-1]["update_id"] + 1, allowed_updates=UPDATE_TYPES)
    if applied:
        state = _posted(char)
        state["voted"] = sorted(voted)[-500:]
        (char.dir / "posted.json").write_text(json.dumps(state))
    return applied


def run_videos(char):
    """Advance the on-demand video kernel: post every finished clip not posted yet,
    then start the next batch from the queue. Posting is driven by the kernel's own
    output (each result carries its chat and photo message), not by our record of
    what is running, so a lost or overwritten state file can no longer lose a
    finished video. Returns True if anything changed."""
    import base64
    v = _videos(char)
    v.setdefault("done", [])
    vk = kaggle.video_kernel_id(char)
    st = kaggle.status(vk)
    if "RUNNING" in st or "QUEUED" in st:
        return False
    changed = False
    if "COMPLETE" in st or "ERROR" in st:
        with tempfile.TemporaryDirectory() as tmp:
            status = kaggle.download_output(char, tmp, slug=vk) or {}
            for r in status.get("items", []):
                if r.get("request") in v["done"] or not r.get("chat"):
                    continue
                if r.get("ok"):
                    meta = json.loads((Path(tmp) / "out" / "items" / f"{r['id']}.json").read_text())
                    tg.video(r["chat"], Path(tmp) / "out" / "items" / meta["path"],
                             caption=f"🎬 {char.name}\n{(r.get('motion') or '')[:300]}", reply_to=r["message_id"])
                else:
                    tg.text(r["chat"], f"❌ Video failed: {str(r.get('reason'))[:300]}", reply_to=r["message_id"])
                v["done"].append(r["request"])
                changed = True
        if v["running"]:
            v["running"] = []
            changed = True
    if v["queue"]:
        batch = v["queue"][: int(config.env("PERSONA_VIDEO_BATCH", "3"))]
        jobs = []
        with tempfile.TemporaryDirectory() as tmp:
            for q in batch:
                img = tg.download(q["file_id"], Path(tmp) / f"{q['id']}.jpg")
                jobs.append({"id": q["id"], "image_b64": base64.b64encode(img.read_bytes()).decode(),
                             "motion": q.get("motion", ""), "tags": q.get("tags", []),
                             "chat": q["chat"], "message_id": q["message_id"]})
        kaggle.push_videos(char, jobs)
        v["running"] = batch
        v["queue"] = v["queue"][len(batch):]
        changed = True
    v["done"] = v["done"][-200:]
    _save_videos(char, v)
    return changed


def _output_items(char, tmp):
    status = kaggle.download_output(char, tmp)
    items = {}
    for f in (Path(tmp) / "out" / "items").glob("*.json"):
        meta = json.loads(f.read_text())
        if meta.get("path"):
            meta["path"] = str(f.parent / meta["path"])
        items[meta["id"]] = meta
    return status, items


def render(slug):
    char = fetch(slug)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _, items = _output_items(char, tmp)
        except Exception as e:
            log(f"no previous output to read votes against ({e})")
            items = {}
        log(f"votes applied: {collect_votes(char, items)}")
    n = int(config.env("PERSONA_CLOUD_IMAGES", "8"))
    v = int(config.env("PERSONA_CLOUD_VIDEOS", "0"))
    scene_list = scenes.write(char, lane="social", n=n)
    log(kaggle.sync(char))
    kaggle.push(char, scene_list, videos=v, date=today())


def _caption(char, item):
    lines = ["📸 " + char.name + (" · 🎬" if item.get("kind") == "video" else ""),
             f"📝 {item.get('caption') or ''}".rstrip(),
             f"🎞 {item.get('scene', {}).get('setting', '')}".rstrip()]
    chk = item.get("check") or {}
    if "sim" in chk:
        lines.append(f"face {chk['sim']} · age≈{chk.get('age')}")
    if item.get("kind") != "video":
        lines.append("🎬 tap, or reply with how she should move")
    lines.append(f"#{item['id']}")
    return "\n".join(l for l in lines if l.strip() not in ("", "📝", "🎞"))


def _buttons(item):
    row = [("👍", f"pv:up:{item['id']}"), ("👎", f"pv:dn:{item['id']}")]
    if item.get("kind") != "video":
        row.append(("🎬 Video", f"pv:vid:{item['id']}"))
    return tg.kb(row)


def post(slug, k, n_slots):
    char = fetch(slug)
    chat = tg.chat("social")
    with tempfile.TemporaryDirectory() as tmp:
        status, items = _output_items(char, tmp)
        any_date = config.flag("PERSONA_POST_ANY_DATE")   # post an older batch by hand
        if not status or not status.get("ok") or (status.get("date") != today() and not any_date):
            log(f"today's render is not ready (status: {(status or {}).get('stage')}, "
                f"date {(status or {}).get('date')}); nothing to post this slot")
            log(f"votes applied: {collect_votes(char, items)}")
            if (status or {}).get("date") == today() and not status.get("ok"):
                tg.text(chat, f"⚠️ Today's Kaggle render failed at {status.get('stage')}: "
                              f"{(status.get('error') or '')[:300]}")
            return
        state = _posted(char)
        if state.get("date") != today():
            state.update({"date": today(), "ids": []})
        order = [r for r in status["items"] if r.get("ok")]
        images = [r["id"] for r in order if r.get("kind", "image") == "image"]
        videos = [r["id"] for r in order if r.get("kind") == "video"]
        # Slot k owns images k, k+n, k+2n...; video j goes to slot round((j+0.5)*n/len).
        due = [i for idx, i in enumerate(images) if idx % n_slots <= k]
        due += [i for j, i in enumerate(videos) if int((j + 0.5) * n_slots / len(videos)) <= k]
        sent = 0
        for item_id in due:
            if item_id in state["ids"] or item_id not in items:
                continue
            item = items[item_id]
            try:
                if item.get("kind") == "video":
                    tg.video(chat, item["path"], caption=_caption(char, item), buttons=_buttons(item))
                else:
                    tg.photo(chat, item["path"], caption=_caption(char, item), buttons=_buttons(item))
            except Exception as e:
                log(f"could not post {item_id}: {e}")
                continue
            state["ids"].append(item_id)
            sent += 1
        (char.dir / "posted.json").write_text(json.dumps(state))
        log(f"slot {k}/{n_slots}: posted {sent}")
        log(f"votes applied: {collect_votes(char, items)}")
    log(kaggle.sync(char))


def votes(slug):
    """Every 15 min: apply 👍/👎, queue 🎬 taps and reply prompts, and move the
    on-demand video kernel along. Skips all downloads when there is nothing to do."""
    try:
        pending = tg.call("getUpdates", timeout=0, allowed_updates=UPDATE_TYPES)
    except Exception as e:
        log(f"could not read updates: {e}")
        return
    char = fetch(slug)
    with tempfile.TemporaryDirectory() as tmp:
        items = {}
        if pending:
            try:
                _, items = _output_items(char, tmp)
            except Exception as e:
                log(f"no render output to match votes against ({e})")
        n = collect_votes(char, items) if pending else 0
    log(f"updates applied: {n}")
    if n:
        # Save first: the updates are already confirmed with Telegram, so a crash
        # further down must not lose a vote or a 🎬 request.
        log(kaggle.sync(char))
    if run_videos(char):
        log(kaggle.sync(char))


def main():
    config.load_env()
    cmd, slug = sys.argv[1], sys.argv[2]
    if cmd == "render":
        render(slug)
    elif cmd == "votes":
        votes(slug)
    elif cmd == "post":
        post(slug, int(sys.argv[3]), int(sys.argv[4]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
