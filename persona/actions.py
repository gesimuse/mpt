"""The laptop-free schedule, run by .github/workflows/persona_kaggle.yml.

    python -m persona.actions render <slug>         once a day, early
    python -m persona.actions post   <slug> <k> <n> at each of n slots (k = 0..n-1)

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


def collect_votes(char, items):
    """Apply pending 👍/👎 presses. `items` maps id -> item meta for everything this
    run knows about (today's kernel output). Returns how many were applied."""
    if not config.flag("PERSONA_CLOUD_VOTES", "1"):
        return 0
    try:
        updates = tg.call("getUpdates", timeout=0, allowed_updates=["callback_query"])
    except Exception as e:
        log(f"could not read votes: {e}")
        return 0
    applied, voted = 0, set(_posted(char).get("voted", []))
    for u in updates:
        cq = u.get("callback_query") or {}
        data = cq.get("data", "")
        if not data.startswith(("pv:up:", "pv:dn:")):
            continue
        _, act, item_id = data.split(":", 2)
        item = items.get(item_id)
        msg = cq.get("message") or {}
        if not item or item_id in voted:
            tg.answer(cq.get("id"), "Too old to count." if not item else "Already voted.")
            continue
        char.vote(item, 1 if act == "up" else -1)
        if act == "up" and item.get("kind") == "image" and item.get("scene", {}).get("beat"):
            char.add_beat(item["scene"]["beat"], item_id)
        voted.add(item_id)
        applied += 1
        tg.answer(cq.get("id"), "Noted 👍" if act == "up" else "Noted 👎")
        label = "✓ 👍" if act == "up" else "✓ 👎"
        tg.set_buttons(msg.get("chat", {}).get("id"), msg.get("message_id"),
                       tg.kb([(label, f"pv:{act}:{item_id}")]))
    if updates:
        # Confirm them all, so the next run does not see them again.
        tg.call("getUpdates", timeout=0, offset=updates[-1]["update_id"] + 1, allowed_updates=["callback_query"])
    if applied:
        state = _posted(char)
        state["voted"] = sorted(voted)[-500:]
        (char.dir / "posted.json").write_text(json.dumps(state))
    return applied


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
    v = int(config.env("PERSONA_CLOUD_VIDEOS", "2"))
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
    lines.append(f"#{item['id']}")
    return "\n".join(l for l in lines if l.strip() not in ("", "📝", "🎞"))


def _buttons(item):
    return tg.kb([("👍", f"pv:up:{item['id']}"), ("👎", f"pv:dn:{item['id']}")])


def post(slug, k, n_slots):
    char = fetch(slug)
    chat = tg.chat("social")
    with tempfile.TemporaryDirectory() as tmp:
        status, items = _output_items(char, tmp)
        if not status or not status.get("ok") or status.get("date") != today():
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


def main():
    config.load_env()
    cmd, slug = sys.argv[1], sys.argv[2]
    if cmd == "render":
        render(slug)
    elif cmd == "post":
        post(slug, int(sys.argv[3]), int(sys.argv[4]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
