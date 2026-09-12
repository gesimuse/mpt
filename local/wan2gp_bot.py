"""Telegram -> local Wan2GP -> Telegram. Nothing leaves this machine but the video.

Send a photo into the MPT channel with a motion prompt as its caption (or reply to a
photo with one). While this computer is on, this daemon picks it up, generates the
clip on the local RTX A4500 with the same Wan2GP runtime and the same
i2v_2_2_Enhanced_Lightning_v2 checkpoint the Kaggle path used, and posts the mp4 into
the MPT Videos channel.

DELIBERATELY STANDALONE. It imports nothing from the rest of this repo, writes
nothing into it, and touches no state the cloud flow uses -- no posted.json, no
gh-pages, no GitHub, no TikTok, no Kaggle, no Hugging Face. The Cloudflare Worker,
the buttons on generated photos and the whole autopilot pipeline carry on exactly as
before, unaware this exists. That isolation is a requirement, not a style choice: the
existing flow is the account's publishing pipeline and must not change behaviour
because a local experiment is running.

It also needs its OWN bot token, which is what keeps the two apart at the Telegram
level. The existing bot's updates are delivered to the Cloudflare Worker by webhook,
and Telegram refuses getUpdates for a bot that has one (409) -- so a second bot, added
as an admin to the same two channels, is what makes a local long-poll possible without
disturbing the webhook. Both bots see the channels; each gets its own updates.

What is private here and what is not, plainly:
  * The image and the video exist on this machine and in Telegram's cloud, and
    nowhere else. Both channels are private (no public username, invite link only),
    so only their members can see them.
  * Telegram channels are NOT end-to-end encrypted. Telegram's servers hold the
    photo you send and the video sent back, and can technically read them. That is
    the residual exposure, and it is inherent to using Telegram as the interface at
    all -- no arrangement of this program changes it.
  * Nothing else receives either file or the prompt: generation is local, and the
    timeline formatting below is a plain string template, not a model call.

Setup:
  1. @BotFather -> /newbot -> a SECOND bot, separate from the mpt one.
  2. Add it to both channels as an admin (it needs to read channel posts and post).
  3. Put its token in .env as LOCAL_BOT_TOKEN.
  4. Run it with WAN2GP'S OWN python, since it loads that runtime in-process:
         ~/apps/Wan2GP/.venv/bin/python local/wan2gp_bot.py
     Stop the Wan2GP web UI first -- same runtime, same 30GB of RAM, only one fits.
     See local/README.md for the systemd unit.

Env (all optional except the token; defaults suit this machine):
  LOCAL_BOT_TOKEN      required -- the SECOND bot, never the mpt one
  LOCAL_SRC_CHAT_ID    where photos come from   (default: TELEGRAM_CHAT_ID)
  LOCAL_OUT_CHAT_ID    where videos go          (default: TELEGRAM_VIDEO_CHAT_ID)
  WAN2GP_DIR           default ~/apps/Wan2GP
  WAN2GP_PROFILE       default 4      (memory profile; 5 is the low-RAM failsafe)
  WAN2GP_ATTENTION     default sdpa   (no nvcc here, so no SageAttention)
  LOCAL_VIDEO_FRAMES   default 81     (~5s at Wan2GP's 16fps)
  LOCAL_VIDEO_STEPS    default 8
  LOCAL_RESOLUTION     default: whatever the saved Wan2GP settings file uses
  LOCAL_TIMELINE       default 1  (wrap the prompt in Wan2GP's per-second template;
                       set 0 to send exactly what you typed)
  LOCAL_KEEP_OUTPUTS   default 0  (1 also leaves each clip in Wan2GP's outputs/)
  LOCAL_VERBOSE        default 0  (1 prints wgp.py's own per-step progress)
  WAN2GP_TIMEOUT       default 5400 seconds before a stuck generation is abandoned
"""
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
API = "https://api.telegram.org"
# Not in the repo: this is runtime state, and the repo is public.
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "mpt-wan2gp"
STATE_FILE = STATE_DIR / "state.json"


def log(msg):
    print(f"[wan2gp-bot] {_redact(msg)}", flush=True)


def _redact(text):
    """The token is a path segment of every Telegram URL, so requests puts it into
    its own exception messages. Same rule as telegram.py's redact()."""
    text = str(text)
    token = CONF.get("token") if "CONF" in globals() else None
    if token:
        text = text.replace(token, "<LOCAL_BOT_TOKEN>")
        secret = token.split(":", 1)[-1]
        if len(secret) > 8:
            text = text.replace(secret, "<LOCAL_BOT_TOKEN>")
    return text


def _dotenv(path):
    """The repo's .env, parsed the same shallow way worker/deploy.sh reads it, so
    there is one place to put the token."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _config():
    env = {**_dotenv(REPO / ".env"), **os.environ}
    token = (env.get("LOCAL_BOT_TOKEN") or "").strip()
    if not token:
        sys.exit(
            "LOCAL_BOT_TOKEN is not set.\n"
            "This daemon needs its OWN bot, separate from the mpt one: the mpt bot's\n"
            "updates go to the Cloudflare Worker by webhook, and Telegram will not let\n"
            "a bot with a webhook be long-polled (409).\n"
            "  1. @BotFather -> /newbot\n"
            "  2. add the new bot to both MPT channels as an admin\n"
            "  3. put its token in .env as LOCAL_BOT_TOKEN")
    if token == (env.get("TELEGRAM_BOT_TOKEN") or "").strip():
        sys.exit("LOCAL_BOT_TOKEN is the mpt bot's token. Use a SECOND bot -- long-"
                 "polling this one would fight the Cloudflare Worker's webhook.")
    wan_dir = Path(env.get("WAN2GP_DIR") or Path.home() / "apps/Wan2GP").expanduser()
    return {
        "token": token,
        "src_chat": str(env.get("LOCAL_SRC_CHAT_ID") or env.get("TELEGRAM_CHAT_ID") or "").strip(),
        "out_chat": str(env.get("LOCAL_OUT_CHAT_ID") or env.get("TELEGRAM_VIDEO_CHAT_ID")
                        or env.get("TELEGRAM_CHAT_ID") or "").strip(),
        "wan_dir": wan_dir,
        "profile": env.get("WAN2GP_PROFILE", "4"),
        "attention": env.get("WAN2GP_ATTENTION", "sdpa"),
        "frames": int(env.get("LOCAL_VIDEO_FRAMES", "81")),
        "steps": int(env.get("LOCAL_VIDEO_STEPS", "8")),
        "resolution": (env.get("LOCAL_RESOLUTION") or "").strip(),
        "timeline": env.get("LOCAL_TIMELINE", "1") not in ("0", "false", "no"),
        "timeout": int(env.get("WAN2GP_TIMEOUT", "5400")),
        # Wan2GP writes every clip into its own outputs/ folder as well. Off by
        # default: the video belongs in Telegram, and a folder quietly filling with
        # every clip ever generated is the kind of leftover this flow exists to avoid.
        "keep_outputs": env.get("LOCAL_KEEP_OUTPUTS", "0") not in ("0", "false", "no"),
        # wgp.py is extremely chatty (a progress line per step). Off keeps the
        # journal readable; on is what you want when a generation misbehaves.
        "verbose": env.get("LOCAL_VERBOSE", "0") not in ("0", "false", "no"),
    }


# --------------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------------

def tg(method, timeout=60, files=None, **params):
    url = f"{API}/bot{CONF['token']}/{method}"
    try:
        r = (requests.post(url, data=params, files=files, timeout=timeout) if files
             else requests.post(url, json=params, timeout=timeout))
    except requests.RequestException as e:
        raise RuntimeError(f"telegram {method} unreachable: {_redact(e)[:200]}") from None
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(f"telegram {method} rejected: {_redact(body)[:300]}")
    return body["result"]


def say(chat_id, text, reply_to=None):
    """Never silent. A photo that produced no answer is indistinguishable from a
    daemon that is not running, which is the one thing this has to make obvious."""
    try:
        tg("sendMessage", chat_id=chat_id, text=text[:4096], reply_to_message_id=reply_to,
           disable_web_page_preview=True)
    except Exception as e:
        log(f"could not reply in {chat_id}: {e}")


def download_photo(file_id, dest_dir):
    info = tg("getFile", file_id=file_id)
    path = info["file_path"]
    dest = Path(dest_dir) / "input.jpg"
    # The file URL embeds the token -- never logged, never passed on.
    with requests.get(f"{API}/file/bot{CONF['token']}/{path}", timeout=120,
                      stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    return dest


def send_video(chat_id, path, caption):
    """As a video, so it plays inline in the channel. Falls back to a document if
    Telegram refuses the video (an odd codec, or a file over its video limit) --
    getting the file is what matters, the inline player is a nicety."""
    with open(path, "rb") as f:
        try:
            return tg("sendVideo", timeout=300, files={"video": f},
                      chat_id=chat_id, caption=caption[:1024], supports_streaming=True)
        except RuntimeError as e:
            log(f"sendVideo refused ({e}); sending as a document instead")
    with open(path, "rb") as f:
        return tg("sendDocument", timeout=300, files={"document": f},
                  chat_id=chat_id, caption=caption[:1024])


# --------------------------------------------------------------------------------
# Prompt -> Wan2GP's timeline format
# --------------------------------------------------------------------------------
#
# Wan2GP's own docs/PROMPTS.md: the "(at X seconds: ...)" timeline is "especially
# useful for Wan2.2 Image2video Enhanced Lightning v2 14B, because its own default
# prompt uses that kind of timeline format" -- exactly the checkpoint below, whose
# defaults json also ships a timeline as its own default prompt.
#
# A pure template, no model involved. Three shapes of prompt, all handled the same
# obvious way, so what you type is what you get:
#
#   already a timeline   -> passed through untouched
#   several lines        -> one line per second, in order (write the beats yourself)
#   one sentence         -> dropped into TEMPLATE_BEATS below
#
# The multi-line case is the useful one once a prompt matters: type the seconds
# yourself and nothing reinterprets them.

TIMELINE_RE = re.compile(r"^\(at \d+ seconds?:.+\)\.?$")

# Second 0 carries the motion itself; the rest give it somewhere to go. Deliberately
# generic -- this is a frame around the words you sent, not a rewrite of them.
TEMPLATE_BEATS = [
    "{motion}, the movement just beginning, camera static",
    "{motion}, the movement carrying, weight shifting with it",
    "the movement continues, unhurried, her eyes finding the lens",
    "the movement reaches its fullest point, held there",
    "she settles out of it, hair and fabric still moving",
    "she holds the new pose, breathing, gaze on the lens",
]


def is_timeline(text):
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return bool(lines) and all(TIMELINE_RE.match(ln) for ln in lines)


def _second(n, text):
    return f"(at {n} second{'' if n == 1 else 's'}: {text.rstrip('.')})"


def to_timeline(motion, seconds):
    """`motion` in Wan2GP's per-second format. Deterministic: same input, same
    output, nothing consulted."""
    if not CONF["timeline"] or is_timeline(motion):
        return motion
    lines = [ln.strip() for ln in motion.splitlines() if ln.strip()]
    if len(lines) > 1:
        return "\n".join(_second(i, ln) for i, ln in enumerate(lines[:seconds + 1]))
    beats = TEMPLATE_BEATS[:max(2, seconds + 1)]
    return "\n".join(_second(i, b.format(motion=motion.rstrip(".")))
                     for i, b in enumerate(beats))


# --------------------------------------------------------------------------------
# Wan2GP, in-process
# --------------------------------------------------------------------------------
#
# shared/api.py, not `wgp.py --process` as a subprocess. The difference is not
# tidiness, it is whether this runs at all on this machine: Wan 2.2 14B holds ~26GB
# of the 30GB of host RAM under profile 4, so a second runtime does not fit, and a
# subprocess per job would reload those weights from disk for every single clip --
# minutes of wall clock, repeated, for no reason. `init()` keeps one runtime alive
# for the life of the daemon, so only the FIRST job pays the load.
#
# The consequence to know about: this daemon and the Wan2GP web UI cannot both be
# running. They are the same runtime and the same RAM. Stop the UI to run the bot,
# stop the bot to use the UI.
#
# It also means this script must run under Wan2GP's own venv (torch, mmgp and the
# rest live there), which is what local/README.md and the systemd unit both use:
#     ~/apps/Wan2GP/.venv/bin/python local/wan2gp_bot.py

MODEL_TYPE = "i2v_2_2_Enhanced_Lightning_v2"

SESSION = None
SESSION_LOCK = threading.Lock()


def session():
    """The one live Wan2GP runtime, created on first use.

    Created lazily rather than at startup so the daemon comes up, answers Telegram
    and reports problems in the channel even if the runtime cannot initialise -- a
    bot that dies silently on boot is the failure mode this whole design is trying
    to avoid."""
    global SESSION
    with SESSION_LOCK:
        if SESSION is None:
            sys.path.insert(0, str(CONF["wan_dir"]))
            from shared.api import init
            log(f"loading the Wan2GP runtime from {CONF['wan_dir']} "
                f"(profile {CONF['profile']}, {CONF['attention']}) -- first job only")
            started = time.time()
            SESSION = init(
                root=CONF["wan_dir"],
                cli_args=["--profile", str(CONF["profile"]),
                          "--attention", CONF["attention"]],
                console_output=CONF["verbose"],
            )
            log(f"runtime ready in {int(time.time() - started)}s")
    return SESSION


def _base_settings():
    """The settings this machine's Wan2GP UI last saved for this model, so a clip
    generated from Telegram matches one generated by hand in the UI. Only the few
    fields a job actually decides are overridden below; everything else -- guidance
    phases, flow shift, the switch threshold, LoRAs -- is whatever was tuned there."""
    saved = CONF["wan_dir"] / "settings" / f"{MODEL_TYPE}_settings.json"
    if saved.exists():
        try:
            return json.loads(saved.read_text())
        except Exception as e:
            log(f"could not read {saved.name} ({type(e).__name__}); using bare defaults")
    return {}


def generate(image_path, prompt, out_dir):
    """One clip. Returns (path, seconds taken)."""
    settings = _base_settings()
    settings.update({
        "model_type": MODEL_TYPE,
        "prompt": prompt,
        "image_start": str(image_path),
        "image_prompt_type": "S",          # start image, no end image, no source video
        "video_length": CONF["frames"],
        "num_inference_steps": CONF["steps"],
        "seed": -1,
        "batch_size": 1,
    })
    if CONF["resolution"]:
        settings["resolution"] = CONF["resolution"]
    log(f"generating: {CONF['frames']} frames, {CONF['steps']} steps, "
        f"{settings.get('resolution')}")
    started = time.time()
    job = session().submit_task(settings)
    result = job.result(timeout=CONF["timeout"])
    took = int(time.time() - started)
    if not result.success or not result.generated_files:
        reasons = "; ".join(getattr(e, "message", str(e)) for e in result.errors) \
            or "no file produced and no error reported"
        raise RuntimeError(f"generation failed after {took}s: {reasons[:600]}")
    produced = Path(result.generated_files[-1])
    # WanGP writes into its own configured output folder; the copy into the job's
    # temp dir is what makes the cleanup in run_job() the single place that decides
    # how long a clip lives on disk.
    final = Path(out_dir) / produced.name
    if produced.resolve() != final.resolve():
        shutil.copy(produced, final)
        if CONF["keep_outputs"]:
            log(f"kept Wan2GP's own copy at {produced}")
        else:
            try:
                produced.unlink()
            except OSError as e:
                log(f"could not remove {produced} ({type(e).__name__})")
    log(f"generated in {took // 60}m{took % 60:02d}s: {final.name}")
    return final, took


# --------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------

def run_job(job):
    """One photo + prompt -> one clip in the videos channel. Everything the job
    touches lives in one temp dir, removed whatever happens: the only copies that
    outlive it are the ones in Telegram."""
    work = tempfile.mkdtemp(prefix="wan2gp_job_")
    try:
        image = download_photo(job["file_id"], work)
        prompt = to_timeline(job["prompt"], round(CONF["frames"] / 16))
        video, took = generate(image, prompt, work)
        send_video(CONF["out_chat"], video,
                   f"🖥 local run · {CONF['frames']}f · {CONF['steps']} steps · "
                   f"{took // 60}m{took % 60:02d}s\n\n{prompt}")
        say(job["chat_id"], f"✅ Done in {took // 60}m{took % 60:02d}s — it's in the "
            "videos channel.", reply_to=job["message_id"])
    except Exception as e:
        log(f"job failed: {_redact(e)[:400]}")
        say(job["chat_id"], f"❌ Generation failed:\n{_redact(e)[:1000]}",
            reply_to=job["message_id"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def worker(jobs):
    """One job at a time: there is one GPU, and Wan 2.2 14B on 16GB with 30GB of host
    RAM has no room for a second process. The queue is what makes several photos in a
    row work rather than colliding."""
    while True:
        job = jobs.get()
        try:
            run_job(job)
        except Exception as e:            # never let the worker thread die
            log(f"worker error: {_redact(e)[:300]}")
        finally:
            jobs.task_done()


# --------------------------------------------------------------------------------
# Telegram polling
# --------------------------------------------------------------------------------

def _load_offset():
    try:
        return json.loads(STATE_FILE.read_text()).get("offset", 0)
    except Exception:
        return 0


def _save_offset(offset):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"offset": offset}))


def largest_photo(msg):
    photo = (msg or {}).get("photo") or []
    return photo[-1]["file_id"] if photo else None


def job_from(post):
    """A photo with its prompt as the caption, or a prompt replied onto a photo.

    Bot posts are skipped: this daemon's own confirmations and the mpt bot's cards
    both land in the same channel, and re-ingesting either would be a loop or a
    duplicate run."""
    if post.get("from", {}).get("is_bot") or post.get("reply_markup"):
        return None
    chat_id = str(post.get("chat", {}).get("id"))
    if CONF["src_chat"] and chat_id != CONF["src_chat"]:
        return None
    file_id = largest_photo(post)
    caption = (post.get("caption") or "").strip()
    if file_id and caption:
        return {"file_id": file_id, "prompt": caption, "chat_id": chat_id,
                "message_id": post["message_id"]}
    if file_id:
        say(chat_id, "Reply to this photo with a motion prompt (or send the photo "
            "with the prompt as its caption) and it generates here on this machine.",
            reply_to=post["message_id"])
        return None
    text = (post.get("text") or "").strip()
    replied = post.get("reply_to_message") or {}
    file_id = largest_photo(replied)
    if text and file_id:
        return {"file_id": file_id, "prompt": text, "chat_id": chat_id,
                "message_id": post["message_id"]}
    return None


def main():
    global CONF
    CONF = _config()
    me = tg("getMe")
    log(f"bot @{me.get('username')} | photos from {CONF['src_chat']} | "
        f"videos to {CONF['out_chat']}")
    if not (CONF["wan_dir"] / "wgp.py").exists():
        sys.exit(f"no Wan2GP at {CONF['wan_dir']} -- set WAN2GP_DIR")
    if "shared.api" not in sys.modules and not (CONF["wan_dir"] / "shared/api.py").exists():
        sys.exit(f"{CONF['wan_dir']}/shared/api.py is missing -- this needs a Wan2GP "
                 "recent enough to expose the in-process API")

    jobs = queue.Queue()
    threading.Thread(target=worker, args=(jobs,), daemon=True).start()

    offset = _load_offset()
    if not offset:
        # First start: skip whatever is already sitting in the backlog rather than
        # generating a video for every photo posted before this daemon existed.
        pending = tg("getUpdates", timeout=0, offset=-1)
        offset = (pending[-1]["update_id"] + 1) if pending else 0
        _save_offset(offset)
        log("first start: ignoring anything posted before now")

    log("waiting for photos")
    while True:
        try:
            updates = tg("getUpdates", timeout=70, offset=offset,
                         allowed_updates=["channel_post", "message"])
        except Exception as e:
            log(f"poll failed ({_redact(e)[:200]}); retrying in 10s")
            time.sleep(10)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            post = update.get("channel_post") or update.get("message")
            if not post:
                continue
            try:
                job = job_from(post)
            except Exception as e:
                log(f"could not read update: {_redact(e)[:200]}")
                continue
            if not job:
                continue
            jobs.put(job)
            ahead = jobs.qsize() - 1
            say(job["chat_id"],
                "🖥 Queued for the local GPU."
                + (f" {ahead} job(s) ahead of it." if ahead > 0 else " Starting now."),
                reply_to=job["message_id"])
        if updates:
            _save_offset(offset)


if __name__ == "__main__":
    main()
