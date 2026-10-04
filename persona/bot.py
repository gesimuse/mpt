"""The persona daemon: Telegram in, local GPU, Telegram out.

Run with Wan2GP's python (it loads the runtime in-process):
    ~/apps/Wan2GP/.venv/bin/python -m persona.bot

Stop local/wan2gp_bot.py and the Wan2GP web UI first: same runtime, same RAM.

Commands, posted in the review channel (PERSONA_CHAT_ID):
  /new <Name>            create a blank persona and make her active
  /switch <slug>         make another persona active
  /cast [n] [hint]       n candidate faces (default 8); ⭐ one to pick her
  /refs                  re-render master refs from the picked candidate
  /scene [n] [hint]      n social-lane scenes (TikTok/Instagram-safe)
  /fanvue [n] [hint]     n Fanvue-lane scenes (local only, posts to the Fanvue channel)
  /slot                  one scheduled post now (scenes + a video)
  /daily                 PERSONA_DAILY_SOCIAL social + PERSONA_DAILY_FANVUE Fanvue scenes
  /kaggle [n] [hint]     render n social scenes on Kaggle instead of here
  /sync                  push bible+refs+story to the private Kaggle dataset
  /story                 her last storyline beats
  /models                the registry and which model fills each role
  /use <role> <model>    e.g. /use edit qwen-edit-2511
  /status                queue, active persona, refs
  /help

Buttons on every result:
  👍 / 👎    vote: moves her tag scores, and 👍 adds the scene to her storyline
  🎬        animate it with the video model
  📱 / 📷   TikTok draft / Instagram outbox (social lane only)
  💜        Fanvue: posted by API if FANVUE_API_TOKEN is set, otherwise the outbox
Reply to a result with text to animate it with that motion instead.

While it is up it posts on PERSONA_SCHEDULE (e.g. 09:00,12:00,15:00,18:00,21:00):
each slot is PERSONA_SLOT_IMAGES social scenes plus a video of the first good one.
PERSONA_PUBLISH=0 (the default) hides the TikTok/IG/Fanvue buttons while you judge
quality; set it to 1 when she is ready to go out.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from . import character, config, faceid, fanvue, registry, social, tg
from .engine import Engine
from .studio import Studio

STATE = config.home() / "bot-state.json"


def log(msg):
    print(f"[persona.bot] {tg.redact(msg)}", flush=True)


# --------------------------------------------------------------------------- state
def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE)


# ------------------------------------------------------------------------- buttons
def item_buttons(item, done=()):
    def b(label, act):
        return ((f"✓ {label}" if act in done else label), f"pv:{act}:{item['id']}")
    rows = [[b("👍", "up"), b("👎", "dn")] + ([b("🎬 Video", "vid")] if item["kind"] == "image" else [])]
    if not config.flag("PERSONA_PUBLISH", "0"):
        # Evaluation phase: judge consistency and quality before anything is posted.
        return tg.kb(*rows)
    if item.get("lane") == "social":
        rows.append([b("📱 TikTok", "tt"), b("📷 IG", "ig"), b("💜 Fanvue", "fv")])
    else:
        rows.append([b("💜 Fanvue", "fv")])
    return tg.kb(*rows)


def item_caption(char, item):
    lane = "💜 FANVUE" if item.get("lane") == "fanvue" else "📸 SOCIAL"
    model = registry.get(item["model"]) if item.get("model") else {}
    warn = "" if model.get("commercial") else "\n⚠️ non-commercial model licence"
    scene = item.get("scene", {})
    chk = item.get("check", {})
    check = (f"face {chk['sim']}" if "sim" in chk else "") + (f" · age≈{chk['age']}" if "age" in chk else "")
    if chk.get("note"):
        check = chk["note"]
    lines = [f"{lane} · {char.name} · {item.get('model', '')}{warn}",
             f"📝 {item.get('caption') or ''}".rstrip(),
             f"🎞 {scene.get('setting', '')}".rstrip(),
             check, f"#{item['id']}"]
    return "\n".join(l for l in lines if l.strip() and l.strip() not in ("📝", "🎞"))


# ---------------------------------------------------------------------------- bot
class Bot:
    def __init__(self):
        self.state = load_state()
        self.jobs = queue.Queue()
        self.engine = Engine()
        self.studio = Studio(self.engine)
        self.review = tg.chat("social")
        self.current = None
        self.job_started = None

    # ---------------------------------------------------------------- job queue
    def enqueue(self, label, fn, *args, reply_to=None, chat_id=None):
        self.jobs.put((label, fn, args, reply_to, chat_id or self.review))
        n = self.jobs.qsize()
        log(f"queued {label} ({n} waiting)")
        return n

    def worker(self):
        while True:
            label, fn, args, reply_to, chat_id = self.jobs.get()
            self.current, self.job_started = label, time.time()
            log(f"start {label}")
            # Keep the laptop awake while the GPU works. On 2026-10-01 it suspended
            # mid-video and the job never came back: the CUDA context does not
            # survive a suspend.
            inhibit = None
            try:
                inhibit = subprocess.Popen(["systemd-inhibit", "--what=sleep:idle:handle-lid-switch",
                                            "--who=mpt-persona", f"--why=rendering {label}", "--mode=block",
                                            "sleep", "infinity"])
            except Exception as e:
                log(f"could not inhibit sleep ({e})")
            try:
                fn(*args)
            except Exception as e:
                log(f"{label} failed: {traceback.format_exc()[-1500:]}")
                self.say(f"❌ {label} failed: {type(e).__name__}: {str(e)[:400]}", reply_to, chat_id)
            finally:
                if inhibit:
                    inhibit.terminate()
                self.current, self.job_started = None, None
                log(f"done {label}")

    def watchdog(self):
        """Exit (systemd restarts the service) when a job can no longer finish:
        the process was frozen by a suspend while a job ran, or a job outlived
        PERSONA_JOB_TIMEOUT minutes. A stuck GPU job otherwise blocks every later
        slot silently, which is what happened over 2026-10-01..04."""
        limit = int(config.env("PERSONA_JOB_TIMEOUT", "45")) * 60
        last = time.time()
        while True:
            time.sleep(30)
            now = time.time()
            if self.current and now - last > 120:
                log(f"woke from a {int(now - last)}s suspend during {self.current}; restarting")
                self.say(f"⚠️ The laptop slept during {self.current}. Restarting the bot; the next slot runs normally.")
                os._exit(1)
            if self.current and self.job_started and now - self.job_started > limit:
                log(f"{self.current} exceeded {limit // 60} min; restarting")
                self.say(f"⚠️ {self.current} hung for {limit // 60} min. Restarting the bot.")
                os._exit(1)
            last = now

    def say(self, msg, reply_to=None, chat_id=None):
        try:
            return tg.text(chat_id or self.review, msg, reply_to=reply_to)
        except Exception as e:
            log(f"could not send message: {e}")

    # ------------------------------------------------------------------- casting
    def job_cast(self, n, hint):
        char = character.active()
        self.say(f"🎭 Casting {n} candidates for {char.name}"
                 f" with {registry.for_role('cast')['id']}… tap ⭐ on the one who is her.")
        shown = 0
        for c in self.studio.cast(char, n=n, hint=hint):
            if c.get("error"):
                self.say(f"⚠️ candidate failed: {c['error'][:200]}")
                continue
            if c.get("rejected"):
                self.say(f"🚫 candidate dropped: {c['rejected']}")
                continue
            look = " · ".join(c["look"].values())
            tg.photo(self.review, c["path"], caption=f"{look}\n#{c['id']}",
                     buttons=tg.kb([("⭐ This is her", f"pc:{c['id']}")]))
            shown += 1
        self.say(f"Casting done: {shown}/{n} shown. Not her yet? /cast again, or narrow the casting "
                 f"pools in bible.json.")

    def job_refs(self, cid):
        char = character.active()
        self.say(f"📐 Building {char.name}'s master refs from #{cid}…")
        good = []
        for r in self.studio.make_refs(char, cid):
            if r.get("error"):
                self.say(f"⚠️ {r['stem']} failed: {r['error'][:200]}")
                continue
            chk = r["check"]
            note = "✅" if r["ok"] else f"⚠️ {chk.get('reason', '')}"
            sim = f" face {chk['sim']}" if "sim" in chk else ""
            tg.photo(self.review, r["path"], caption=f"{r['stem']}{sim} {note}")
            good.append(r)
        if not good:
            return self.say("No refs came out. /refs to try again.")
        tg.text(self.review, "Lock these as her identity? Every future image is edited from them.",
                buttons=tg.kb([("🔒 Lock refs", "pr:lock"), ("🔁 Redo", f"pr:redo:{cid}")]))

    # -------------------------------------------------------------------- scenes
    def job_scenes(self, lane, n, hint):
        char = character.active()
        chat_id = tg.chat(lane)
        made = 0
        for item in self.studio.render(char, lane=lane, n=n, hint=hint):
            if not item.get("ok"):
                self.say(f"🚫 #{item['id']} dropped: {item.get('reason', '')[:200]}", chat_id=chat_id)
                continue
            msg = tg.photo(chat_id, item["path"], caption=item_caption(char, item), buttons=item_buttons(item))
            char.update_item(item["id"], message=[chat_id, msg["message_id"]])
            made += 1
        if made == 0:
            self.say(f"Nothing passed out of {n}. Check the face threshold (PERSONA_FACE_MIN) or the refs.",
                     chat_id=chat_id)

    def job_video(self, item_id, motion, chat_id):
        char = character.active()
        item = char.item(item_id)
        vid = self.studio.animate(char, item, motion)
        msg = tg.video(chat_id, vid["path"], caption=item_caption(char, vid), buttons=item_buttons(vid),
                       reply_to=(item.get("message") or [None, None])[1])
        char.update_item(vid["id"], message=[chat_id, msg["message_id"]])

    def job_slot(self):
        """One scheduled post: PERSONA_SLOT_IMAGES social scenes, then a video of the
        first one that passed the face gate (PERSONA_SLOT_VIDEO=0 to skip)."""
        char = character.active()
        n = int(config.env("PERSONA_SLOT_IMAGES", "2"))
        first = None
        for item in self.studio.render(char, lane="social", n=n):
            if not item.get("ok"):
                self.say(f"🚫 #{item['id']} dropped: {item.get('reason', '')[:200]}")
                continue
            msg = tg.photo(self.review, item["path"], caption=item_caption(char, item), buttons=item_buttons(item))
            item = char.update_item(item["id"], message=[self.review, msg["message_id"]])
            first = first or item
        if first and config.flag("PERSONA_SLOT_VIDEO", "1"):
            self.job_video(first["id"], None, self.review)

    def job_daily(self):
        social_n = int(config.env("PERSONA_DAILY_SOCIAL", "3"))
        fanvue_n = int(config.env("PERSONA_DAILY_FANVUE", "2"))
        if social_n:
            self.job_scenes("social", social_n, "")
        if fanvue_n:
            self.job_scenes("fanvue", fanvue_n, "")

    def job_kaggle(self, n, hint):
        from . import kaggle
        char = character.active()
        self.say(f"☁️ Rendering {n} social scenes for {char.name} on Kaggle (a cold run takes 30+ min)…")
        items = kaggle.run(char, n=n, hint=hint)
        for item in items:
            if not item.get("ok"):
                self.say(f"🚫 #{item['id']} dropped on Kaggle: {item.get('reason', '')[:200]}")
                continue
            msg = tg.photo(self.review, item["path"], caption=item_caption(char, item), buttons=item_buttons(item))
            char.update_item(item["id"], message=[self.review, msg["message_id"]])

    def job_sync(self):
        from . import kaggle
        char = character.active()
        self.say(f"☁️ {kaggle.sync(char)}")

    # ------------------------------------------------------------------ commands
    def command(self, text, msg):
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].split("@")[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        mid = msg["message_id"]

        def n_and_hint(default):
            bits = rest.split(maxsplit=1)
            if bits and bits[0].isdigit():
                return max(1, min(20, int(bits[0]))), (bits[1] if len(bits) > 1 else "")
            return default, rest

        if cmd in ("/help", "/start"):
            return self.say(__doc__.split("Commands,", 1)[1].split("While it is up", 1)[0].strip(), mid)
        if cmd == "/new":
            if not rest:
                return self.say("Usage: /new <Name>", mid)
            char = character.create(rest.strip())
            character.set_active(char.slug)
            return self.say(f"✨ Created {char.name} ({char.slug}), now active.\n"
                            f"Edit her bible at {char.dir / 'bible.json'} (personality, city, casting pools, "
                            f"lane cues), then /cast.", mid)
        if cmd == "/switch":
            character.set_active(rest.strip())
            return self.say(f"Active persona: {rest.strip()}", mid)
        if cmd == "/models":
            s = config.load_settings()
            roles = "\n".join(f"  {r}: {m}" for r, m in s["roles"].items())
            models = "\n".join(registry.describe(m) for m in registry.all_models().values())
            return self.say(f"Roles:\n{roles}\n\nRegistry:\n{models}", mid)
        if cmd == "/use":
            bits = rest.split()
            if len(bits) != 2:
                return self.say("Usage: /use <role> <model>, roles: " + ", ".join(registry.ROLES), mid)
            role, mid_ = bits
            registry.check(registry.get(mid_), role)
            s = config.load_settings()
            s["roles"][role] = mid_
            config.save_settings(s)
            return self.say(f"{role} -> {mid_}", mid)
        if cmd == "/status":
            try:
                char = character.active()
                who = f"{char.name} ({char.slug}), refs: {len(char.refs())}, items: {len(char.items())}"
            except character.CharacterError as e:
                who = str(e)
            face = "on" if faceid.enabled() else "OFF (no identity or age gate)"
            return self.say(f"Persona: {who}\nRunning: {self.current or 'idle'}\nQueued: {self.jobs.qsize()}\n"
                            f"Face/age gate: {face}\nFanvue API: {'on' if fanvue.enabled() else 'outbox only'}\n"
                            f"TikTok account: {social.account() or 'not set'}", mid)
        if cmd == "/story":
            beats = character.active().story(15)
            return self.say("\n".join(f"• {b['beat']}" for b in beats) or "No story yet -- 👍 a scene to start it.",
                            mid)

        # GPU jobs
        if cmd == "/cast":
            n, hint = n_and_hint(8)
            character.active()
            return self.say(f"🖥 queued ({self.enqueue('cast', self.job_cast, n, hint, reply_to=mid)} waiting)", mid)
        if cmd == "/refs":
            char = character.active()
            cid = char.bible.get("casting_pick")
            if not cid:
                return self.say("Pick a candidate first (/cast, then ⭐).", mid)
            return self.say(f"🖥 queued ({self.enqueue('refs', self.job_refs, cid, reply_to=mid)} waiting)", mid)
        if cmd in ("/scene", "/fanvue"):
            lane = "fanvue" if cmd == "/fanvue" else "social"
            n, hint = n_and_hint(1)
            char = character.active()
            if not char.has_refs():
                return self.say("She has no master refs yet: /cast, ⭐ one, then 🔒 Lock refs.", mid)
            return self.say(f"🖥 queued ({self.enqueue(f'{lane} x{n}', self.job_scenes, lane, n, hint, reply_to=mid)}"
                            f" waiting)", mid)
        if cmd == "/slot":
            return self.say(f"🖥 queued ({self.enqueue('slot', self.job_slot, reply_to=mid)} waiting)", mid)
        if cmd == "/daily":
            return self.say(f"🖥 queued ({self.enqueue('daily', self.job_daily, reply_to=mid)} waiting)", mid)
        if cmd == "/kaggle":
            n, hint = n_and_hint(3)
            return self.say(f"queued ({self.enqueue('kaggle', self.job_kaggle, n, hint, reply_to=mid)} waiting)", mid)
        if cmd == "/sync":
            return self.say(f"queued ({self.enqueue('sync', self.job_sync, reply_to=mid)} waiting)", mid)
        return self.say(f"Unknown command {cmd}. /help", mid)

    def on_reply(self, msg):
        """Text reply to a result = animate it with that motion."""
        parent = msg["reply_to_message"]
        caption = parent.get("caption") or ""
        item_id = caption.rsplit("#", 1)[-1].strip() if "#" in caption else ""
        char = character.active()
        item = char.item(item_id) if item_id else None
        if not item or item.get("kind") != "image":
            return
        chat_id = msg["chat"]["id"]
        self.enqueue("video", self.job_video, item_id, msg["text"], chat_id, reply_to=msg["message_id"],
                     chat_id=chat_id)
        self.say("🎬 queued for the local GPU.", msg["message_id"], chat_id)

    # ----------------------------------------------------------------- callbacks
    def on_callback(self, cq):
        data = cq.get("data", "")
        msg = cq.get("message") or {}
        chat_id = msg.get("chat", {}).get("id")
        try:
            if data.startswith("pc:"):
                cid = data[3:]
                char = character.active()
                self.studio.pick(char, cid)
                tg.answer(cq["id"], "Picked. Building her master refs…")
                tg.set_buttons(chat_id, msg["message_id"], tg.kb([("⭐ Picked", "noop")]))
                self.enqueue("refs", self.job_refs, cid)
                return
            if data == "pr:lock":
                refs = self.studio.lock_refs(character.active())
                tg.answer(cq["id"], "Locked.")
                tg.set_buttons(chat_id, msg["message_id"], tg.kb([("🔒 Locked", "noop")]))
                self.say(f"🔒 {len(refs)} master refs locked. /scene or /fanvue to start her story.")
                return
            if data.startswith("pr:redo"):
                cid = data.split(":", 2)[2]
                tg.answer(cq["id"], "Redoing refs…")
                self.enqueue("refs", self.job_refs, cid)
                return
            if data.startswith("pv:"):
                _, act, item_id = data.split(":", 2)
                return self.on_item_action(cq, act, item_id, chat_id, msg)
            tg.answer(cq["id"])
        except Exception as e:
            log(f"callback {data} failed: {traceback.format_exc()[-1200:]}")
            tg.answer(cq["id"], f"Failed: {str(e)[:150]}")

    def on_item_action(self, cq, act, item_id, chat_id, msg):
        char = character.active()
        item = char.item(item_id)
        if not item and config.env("KAGGLE_USERNAME"):
            # Rendered by the GitHub Actions cron on Kaggle: pull that batch in first.
            try:
                from . import kaggle
                kaggle.import_output(char)
                item = char.item(item_id)
            except Exception as e:
                log(f"kaggle import failed: {e}")
        if not item or not item.get("path") or not Path(item["path"]).exists():
            return tg.answer(cq["id"], "That item is not on this machine any more.")
        done = set(item.get("done", []))
        result = ""
        if act in ("up", "dn"):
            if "up" in done or "dn" in done:
                return tg.answer(cq["id"], "Already voted.")
            char.vote(item, 1 if act == "up" else -1)
            if act == "up" and item.get("scene", {}).get("beat") and item["kind"] == "image":
                char.add_beat(item["scene"]["beat"], item_id)
            result = "Noted 👍" if act == "up" else "Noted 👎"
        elif act == "vid":
            self.enqueue("video", self.job_video, item_id, None, chat_id, chat_id=chat_id)
            result = f"Video queued ({self.jobs.qsize()} waiting)"
        elif act in ("tt", "ig", "fv") and not config.flag("PERSONA_PUBLISH", "0"):
            return tg.answer(cq["id"], "Publishing is off (PERSONA_PUBLISH=0).")
        elif act == "tt":
            result = social.tiktok(char, item, item["path"])
        elif act == "ig":
            result = social.instagram(char, item, item["path"])
        elif act == "fv":
            if config.flag("PERSONA_COMMERCIAL_ONLY") and not item.get("commercial"):
                return tg.answer(cq["id"], "Blocked: made with a non-commercial model (PERSONA_COMMERCIAL_ONLY=1).")
            how, where = fanvue.post(char, item["path"], item.get("caption", ""), item_id)
            result = "Posted to Fanvue." if how == "posted" else f"In the Fanvue outbox: {Path(where).name}"
        else:
            return tg.answer(cq["id"])
        done.add(act)
        item = char.update_item(item_id, done=sorted(done), status="approved" if act != "dn" else "rejected")
        tg.answer(cq["id"], result)
        tg.set_buttons(chat_id, msg["message_id"], item_buttons(item, done))
        if act in ("tt", "ig", "fv"):
            self.say(f"#{item_id}: {result}", msg["message_id"], chat_id)

    # ---------------------------------------------------------------------- loop
    def handle(self, update):
        msg = update.get("channel_post") or update.get("message")
        if update.get("callback_query"):
            cq = update["callback_query"]
            chat = str((cq.get("message") or {}).get("chat", {}).get("id"))
            if chat in self.allowed_chats():
                self.on_callback(cq)
            return
        if not msg or str(msg["chat"]["id"]) not in self.allowed_chats():
            return
        text = msg.get("text") or ""
        if text.startswith("/"):
            try:
                self.command(text, msg)
            except Exception as e:
                self.say(f"❌ {type(e).__name__}: {str(e)[:400]}", msg["message_id"], msg["chat"]["id"])
        elif text and msg.get("reply_to_message"):
            self.on_reply(msg)

    def allowed_chats(self):
        return {str(tg.chat("social")), str(tg.chat("fanvue"))}

    def schedule_tick(self):
        """PERSONA_SCHEDULE: comma-separated HH:MM slots, local time. Runs the most
        recent slot that is due and not yet done today. A slot missed by more than
        PERSONA_SLOT_GRACE minutes (laptop off) is skipped rather than piled up, so
        turning the machine on at 16:00 gives one post, not three."""
        slots = sorted(t.strip() for t in config.env("PERSONA_SCHEDULE").split(",") if t.strip())
        if not slots:
            return
        today, now = time.strftime("%Y-%m-%d"), time.strftime("%H:%M")
        done = self.state.setdefault("slots", {})
        if list(done) != [today]:
            done.clear()
            done[today] = []
        due = [t for t in slots if t <= now and t not in done[today]]
        if not due:
            return
        latest = due[-1]
        done[today].extend(due)
        save_state(self.state)
        h, m = map(int, latest.split(":"))
        late = (time.localtime().tm_hour * 60 + time.localtime().tm_min) - (h * 60 + m)
        if late > int(config.env("PERSONA_SLOT_GRACE", "90")):
            log(f"skipping slot {latest}, {late} min late")
            return
        try:
            character.active()
        except character.CharacterError:
            return
        self.enqueue(f"slot {latest}", self.job_slot)

    def run(self):
        threading.Thread(target=self.worker, daemon=True).start()
        threading.Thread(target=self.watchdog, daemon=True).start()
        offset = self.state.get("offset")
        if offset is None:
            # First start: skip whatever is already pending, like wan2gp_bot.py.
            pending = tg.call("getUpdates", timeout=0)
            offset = (pending[-1]["update_id"] + 1) if pending else 0
        log(f"listening on {sorted(self.allowed_chats())}")
        self.say("🟢 Persona bot up. /help")
        while True:
            self.schedule_tick()
            try:
                updates = tg.call("getUpdates", http_timeout=70, offset=offset, timeout=50,
                                  allowed_updates=["channel_post", "message", "callback_query"])
            except Exception as e:
                log(f"poll failed: {e}")
                time.sleep(10)
                continue
            for u in updates:
                offset = u["update_id"] + 1
                try:
                    self.handle(u)
                except Exception:
                    log(f"update failed: {traceback.format_exc()[-1500:]}")
            self.state["offset"] = offset
            save_state(self.state)


def main():
    config.load_env()
    Bot().run()


if __name__ == "__main__":
    main()
