"""A talking video: she says one short line on a podcast set, word captions burned in.

The recipe the user picked on 2026-10-27 after eleven tests (memory talking-head-models):
  1. the line: a popular short on her topic (YouTube search by views, or a TikTok
     account), transcribed, and rewritten by the Worker into her own words;
  2. her voice: LTX-2.3 with her reference voice (ID-LoRA, "A1OF"), audio only --
     LTX makes speech that sounds natural, but its moving picture moved too much;
  3. the picture: InfiniteTalk 480p + lightx2v 4 steps on her podcast portrait,
     driven by that audio, the prompt saying her head stays steady (test 11 raw;
     more steps, an upscale or a face stabilizer all made it worse);
  4. captions: Whisper word timings, three words at a time, keywords in yellow.

Her podcast portrait and reference voice are made once and kept in
<character>/talk/ (portrait.jpg, voice_ref.wav).
"""
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import config, hooktext, qa

FPS_TALK = 25
LTX_VOICE_MODEL = {"id": "ltx-voice", "wan2gp_model_type": "ltx2_22B_distilled_1_1"}
LTX_FIRST_VOICE = {"id": "ltx-first-voice", "wan2gp_model_type": "ltx2_25_22B_distilled"}
# Talks are LTX-2 clips at this size since 2026-11-10 (576x1024 looked sharper than 448x768,
# 7 min a talk). InfiniteTalk (animate) is kept for reference, unused.
TALK_RES = "720x1280"
INFINITETALK = {"id": "infinitetalk", "wan2gp_model_type": "infinitetalk",
                "profile": "profiles/wan_i2v/Image2Video lightx2v - 4 Steps.json"}

# Lena/MPT body wording (memory mpt-body-shape: slim and toned, full bust only).
BODY = ("Her body is a flawless toned model physique: a full, round, perky, large bust, a slim narrow waist "
        "and slim arms -- an hourglass figure, perfectly symmetrical left to right: both breasts the same size, shape "
        "and height, both shoulders level, both arms the same.")
ACCENT = {"Prague": "Czech", "Tokyo": "Japanese", "Moscow": "Russian", "London": "British", "Paris": "French",
          "Oslo": "Norwegian", "Los Angeles": "American", "San Diego": "American", "Dubai": "Middle Eastern",
          "Madrid": "Spanish", "Bali": "Indian", "Rome": "Italian"}


# A new first frame for every talk (2026-10-29: one portrait per girl made every talk
# the same shot, and it read as fake). Each: where she sits, its light, the mic.
SCENES = [
    # Always indoors and clean (2026-10-30). Since 2026-11-11 every talk is a front-camera selfie
    # ("Yeah is good" on the selfie test): she holds the phone herself, so no mics, desks or tripods.
    ("bed", "sitting on her neatly made bed",
     "a calm, tidy bedroom behind her: a clean headboard, soft neutral colours", "plain soft daylight from a window to the side"),
    ("couch", "sitting on a cream couch in a bright minimal living room",
     "a tidy, uncluttered living room behind her: white walls, a plant", "soft daylight from a window"),
    ("kitchen", "leaning on a clean kitchen counter",
     "a spotless kitchen behind her: light cabinets, a vase of flowers", "bright soft morning light"),
    ("desk", "sitting at a clean desk in her room",
     "a neat room behind her: a bookshelf with few books, a plant", "soft window light"),
    ("car", "sitting in the driver's seat of her parked car",
     "the clean car interior behind her, a blurred street through the window", "soft daylight through the windscreen"),
    ("nook", "curled up in a big armchair by a window",
     "a tidy reading corner behind her: a floor lamp, a knitted throw", "soft overcast window light"),
    ("evening", "lying back against the pillows on her bed in the evening",
     "a calm bedroom behind her, a bedside lamp on", "warm bedside lamp light"),
]
# Low necklines, all safe for Lena's Instagram (no lingerie, nothing see-through).
OUTFITS = ["a fitted black ribbed top with a low scoop neckline", "a fitted white tank top with a low neckline",
           "a beige knit sweater with a deep V-neck", "an open denim shirt over a low-cut white tank top",
           "a sage-green wrap top with a deep V-neck", "an oversized black blazer over a low-cut fitted top",
           "a fitted grey cropped hoodie, unzipped low over a tank top", "a red fitted top with a sweetheart neckline"]
HAIR = ["loose soft waves", "a sleek high ponytail", "a messy low bun with loose strands", "sleek straight hair",
        "half-up hair with a claw clip", "a loose braid over one shoulder"]
# Phone footage, not a photo shoot.
REAL = ("Candid vertical phone video frame, shot on an iPhone, natural colours, real skin texture with pores, "
        "no retouching, slight sensor noise. The room is clean and tidy, nothing messy, indoors.")
# Her everyday face on a talk (2026-11-11, "very unrealistic"): said positively, Qwen 2.1 skips negatives.
BARE = ("Her face is her everyday face: light everyday makeup, her lips in their own soft natural colour, "
        "natural short lashes, matte skin with visible pores, faint freckles and small imperfections, a real "
        "ordinary woman filming herself at home.")


def log(m):
    print(f"[persona.talk] {m}", flush=True)


def _ff(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *map(str, args)], check=True)


def _duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(path)]).decode())


def _worker(path, body):
    req = urllib.request.Request(config.env("PERSONA_WORKER_URL").rstrip("/") + path, method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.env('PERSONA_ADMIN_SECRET')}",
                                          "Content-Type": "application/json", "User-Agent": "mpt-persona-agent/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"{}")


# ------------------------------------------------------------------ the line
def _search(source, limit=40):
    """Candidate shorts: [{id, url, views, seconds, title}]. `source` is a search
    phrase (YouTube) or tiktok:@account."""
    if source.startswith("tiktok:"):
        url = f"https://www.tiktok.com/{source.split(':', 1)[1]}"
    else:
        url = f"ytsearch{limit}:{source} #shorts"
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--flat-playlist", "--playlist-end", str(limit), "-J", url]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        # Right after the laptop wakes the network isn't back yet (2026-10-28: TikTok's
        # listing failed, then worked a minute later): once more after a pause.
        import time
        time.sleep(30)
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        log(f"search {source!r} failed: {out.stderr.strip()[-200:]}")
        return []
    found = []
    for e in (json.loads(out.stdout).get("entries") or []):
        if not e or not e.get("id"):
            continue
        found.append({"id": e["id"], "url": e.get("url") or e.get("webpage_url") or "",
                      "views": e.get("view_count") or 0, "seconds": e.get("duration") or 0,
                      "title": (e.get("description") or e.get("title") or "")[:600],
                      # A TikTok account listing carries every video's numbers already.
                      "counts": {k: e.get(k) for k in ("like_count", "comment_count", "repost_count", "save_count")}
                                if e.get("like_count") is not None else None})
    return found


def _engagement(c):
    """A short's numbers (full yt-dlp info: likes, comments, shares/reposts, saves),
    scored by what people do with it, not only how many saw it: shares and saves
    weigh most (the user, 2026-10-27: "sayings people share, send, save"). None
    when it isn't viral enough: under 50k views or under 2% interacting."""
    url = c["url"] or c["id"]
    if not url.startswith("http"):
        url = f"https://www.youtube.com/shorts/{c['id']}"
    if c.get("counts"):
        d = {**c["counts"], "view_count": c["views"]}
    else:
        out = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--skip-download", "-J", url],
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return None
        d = json.loads(out.stdout)
    views = d.get("view_count") or c["views"] or 0
    likes, comments = d.get("like_count") or 0, d.get("comment_count") or 0
    shares = d.get("repost_count") or d.get("share_count") or 0
    saves = d.get("save_count") or d.get("collect_count") or 0
    acts = likes + 3 * comments + 5 * shares + 4 * saves
    if views < 50_000 or acts < 0.02 * views:
        return None
    c.update(views=views, likes=likes, comments=comments, shares=shares, saves=saves,
             title=(d.get("title") or d.get("description") or c["title"])[:200], url=d.get("webpage_url") or url)
    return acts


def _transcribe(wav, words=False):
    """Whisper large-v3 (Wan2GP's copy; the tokenizer comes from the Hub)."""
    import torch
    from transformers import AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained("openai/whisper-large-v3")
    # Wan2GP's copy where it is (WAN2GP_DIR: not under ~ on Kaggle), else the Hub's.
    local = Path(config.env("WAN2GP_DIR") or Path.home() / "apps/Wan2GP").expanduser() / "ckpts/whisper-large-v3"
    import gc
    model = str(local) if local.exists() else "openai/whisper-large-v3"
    # The GPU first; on the CPU when the video model before it left the card full
    # (2026-11-10: a finished InfiniteTalk talk died here with CUDA OOM, 37 min lost).
    for device, dtype in (("cuda", torch.float16), ("cpu", torch.float32)):
        p = None
        try:
            p = pipeline("automatic-speech-recognition", model=model, tokenizer=tok, torch_dtype=dtype, device=device)
            r = p(str(wav), return_timestamps="word" if words else True, generate_kwargs={"language": "en"})
            break
        except torch.cuda.OutOfMemoryError:
            if device == "cpu":
                raise
            log("whisper: GPU full, transcribing on the CPU")
        finally:
            del p
            gc.collect()
            torch.cuda.empty_cache()
    return r["chunks"] if words else r["text"].strip()


def find_line(job, work):
    """A popular short on her topic, not used before, transcribed and rewritten into
    her own line by the Worker. Returns {line, keywords, source, source_title}."""
    ask = {"persona": job.get("persona"), "character": job.get("character"), "topic": job.get("topic")}
    # Asked for in the channel: the user's own quote, said as written ...
    if job.get("given_line"):
        # On Kaggle the Worker wrote it all already (no call back: no secret there).
        r = ({"line": job["given_line"], "keywords": job.get("given_keywords") or [], "caption": job.get("given_caption", ""),
              "hook": job.get("given_hook", "")}
             if "given_keywords" in job else _worker("/persona/admin/laptop/talkline", {**ask, "given": job["given_line"]}))
        return {**r, "source": "", "source_title": job.get("given_desc") or "quote sent in Telegram"}
    # ... or the video they sent: its idea, in her words.
    if job.get("given_url"):
        audio = work / "given.wav"
        subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bestaudio/best",
                        "-o", str(work / "given.%(ext)s"), job["given_url"]], check=True, timeout=300)
        got = next(p for p in work.glob("given.*") if p.suffix != ".wav")
        _ff("-i", got, "-t", "90", "-ac", "1", "-ar", "16000", audio)
        text = _transcribe(audio)
        r = _worker("/persona/admin/laptop/talkline", {**ask, "transcript": text[:3000], "title": "", "any_topic": True, "keep": True})
        if not r.get("line"):
            raise RuntimeError("couldn't make a line from that video")
        return {**r, "source": "", "source_title": job["given_url"]}
    # Part 2 of a talk that did well, or her answer to a viewer's comment: no source to find.
    if job.get("follow_line") or job.get("reply_comment"):
        r = _worker("/persona/admin/laptop/talkline", {**ask, "follow_line": job.get("follow_line", ""),
                                                        "reply_comment": job.get("reply_comment", ""),
                                                        "reply_to_line": job.get("reply_to_line", "")})
        if not r.get("line"):
            raise RuntimeError("no line for the Part 2 / reply")
        return {**r, "source": "", "source_title": ""}
    used = set(job.get("used") or [])
    cands = []
    # Sources in their order (the account made for her topic first), most-viewed
    # first within each: mixed together, an off-topic 6.8M-view short won (2026-10-27).
    for src in job.get("sources") or []:
        found = [c for c in _search(src) if c["id"] not in used and 8 <= (c["seconds"] or 30) <= 90]
        scored = []
        for c in sorted(found, key=lambda c: -c["views"])[:15]:
            s = _engagement(c)
            if s:
                c["_score"] = s
                scored.append((s, c))
        cands += [c for _, c in sorted(scored, key=lambda x: -x[0])[:4]]
    # Lena (job "viral"): every account's shorts ranked together, the most shared first,
    # whatever its topic -- the line keeps the short's own topic (any_topic).
    if job.get("viral"):
        cands = sorted(cands, key=lambda c: -c.get("_score", 0))
    for c in cands[:8]:
        # Its own name: a retried job found the last try's .wav as "the download" and
        # ffmpeg refused to write a file over itself (2026-10-30).
        audio = work / f"src-{c['id']}-16k.wav"
        try:
            raw = work / f"src-{c['id']}"
            subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bestaudio/best",
                            "-o", f"{raw}.%(ext)s", c["url"] or c["id"]], check=True, timeout=300)
            got = next(p for p in work.glob(f"src-{c['id']}.*") if not p.name.endswith("-16k.wav"))
            _ff("-i", got, "-t", "75", "-ac", "1", "-ar", "16000", audio)
            text = _transcribe(audio)
        except Exception as e:
            log(f"source {c['id']} unusable: {type(e).__name__}: {str(e)[:150]}")
            continue
        # A talk says something: 30+ words (a song or a skit got through at 15, 2026-11-10).
        if len(text.split()) < 30:
            continue
        log(f"source {c['id']} ({c['views']} views, {c.get('shares', 0)} shares, {c.get('saves', 0)} saves, {c.get('likes', 0)} likes): {c['title'][:60]!r}")
        r = _worker("/persona/admin/laptop/talkline", {"persona": job.get("persona"), "character": job.get("character"),
                                                        "topic": job.get("topic"), "transcript": text[:3000],
                                                        "title": c["title"], "views": c["views"],
                                                        "any_topic": bool(job.get("viral"))})
        if r.get("line"):
            return {**r, "source": c["id"], "source_title": c["title"], "source_views": c["views"], "source_url": c["url"]}
        log(f"source {c['id']} skipped: {r.get('why', 'off topic')}")
    # Nothing usable found: the Worker writes one on the topic by itself.
    r = _worker("/persona/admin/laptop/talkline", {"persona": job.get("persona"), "character": job.get("character"),
                                                    "topic": job.get("topic"), "transcript": "", "title": ""})
    if not r.get("line"):
        raise RuntimeError("no line for the talk")
    return {**r, "source": "", "source_title": ""}


# ------------------------------------------------------------- her assets
def _identity(b):
    """Her bible identity without its makeup and glam words: a talk is her bare everyday face
    (2026-11-11: Skye's "glossy lips, dramatic lashes" and Chloe's red lips looked fake on video)."""
    out = re.sub(r",?\s*(glamorous|soft|natural|glam)[^,]*makeup[^,]*", "", b.get("identity") or "")
    out = re.sub(r",?\s*[^,]*\b(glossy|lips|lashes|lipstick|contour)\b[^,]*", "", out)
    return out.strip(", ")


def portrait(char, studio):
    """Her podcast portrait, made once (desk, boom mic, warm dim studio, looking at the
    host off-camera right, low scoop neckline)."""
    dest = char.dir / "talk" / "portrait.jpg"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    b = char.bible
    heads = char.dir / "refs-heads"
    ref = next((p for p in [heads / "three-quarter-right.jpg", heads / "01-front.jpg", char.dir / "refs" / "01-front.jpg"]
                if p.exists()), None)
    prompt = (f"Photo of the woman from the reference image, a {b.get('age', 24)}-year-old woman ({_identity(b)}), a guest "
              "on a podcast, seated relaxed at a podcast desk, leaning back a little, medium shot from the waist up, seen at "
              "three-quarters, her head turned to the right of the frame, her eyes on the host sitting off-camera to the "
              "right, talking. She wears a fitted black ribbed top with a low scoop neckline showing cleavage, and a thin "
              f"silver necklace. {BODY} One hand rests on the desk. A black broadcast microphone on a boom arm at the lower "
              "left of the frame, her face fully visible. Behind her a dim podcast studio: dark acoustic panels, a warm lamp "
              "and soft bokeh lights far behind her. Moody cinematic side light. Vertical 9:16 photo, realistic skin, 35mm "
              "film look, shallow depth of field.")
    model = studio.model("edit")
    for attempt in range(3):
        p = Path(studio.engine.image(model, prompt, dest.parent, f"portrait-{attempt}", refs=[ref] if ref else [],
                                     resolution="720x1280"))
        good, why, _ = qa.check(p)
        if good:
            p.rename(dest)
            log(f"{char.slug}: podcast portrait made")
            return dest
        log(f"{char.slug}: portrait {attempt + 1} failed its check ({why})")
    p.rename(dest)  # the last one rather than nothing
    return dest


def pick_scene(char):
    """A scene, outfit and hair she hasn't had in her last few talks (talk/recent.json)."""
    import random
    rec_file = char.dir / "talk" / "recent.json"
    try:
        recent = json.loads(rec_file.read_text())
    except (OSError, ValueError):
        recent = {}
    def fresh(pool, key, keep):
        used = recent.get(key, [])[-keep:]
        return random.choice([x for x in pool if (x[0] if isinstance(x, tuple) else x) not in used] or pool)
    scene, outfit, hair = fresh(SCENES, "scene", 4), fresh(OUTFITS, "outfit", 4), fresh(HAIR, "hair", 3)
    for key, v in (("scene", scene[0]), ("outfit", outfit), ("hair", hair)):
        recent[key] = (recent.get(key, []) + [v])[-10:]
    rec_file.parent.mkdir(parents=True, exist_ok=True)
    rec_file.write_text(json.dumps(recent))
    return {"scene": scene, "outfit": outfit, "hair": hair}


def frame(char, studio, look, work):
    """This talk's first frame: her face from her refs, the rest from `look`."""
    b = char.bible
    heads = char.dir / "refs-heads"
    ref = next((p for p in [heads / "01-front.jpg", char.dir / "refs" / "01-front.jpg"] if p.exists()), None)
    _, where, behind, light = look["scene"]
    # A front-camera selfie (2026-11-11 selfie test on Qwen 2.1, the user's pick over Krea):
    # she films herself, face large and centred, looking into the lens.
    prompt = (f"Photo of the woman from the reference image, a {b.get('age', 24)}-year-old woman ({_identity(b)}). "
              f"A selfie video frame taken with the front camera of her phone held at arm's length, {where}. "
              "Framed from just below her shoulders up, her face large and centred, looking straight into the lens, "
              "mid-sentence, a small relaxed half-smile. Her face is bare: no makeup, her own pale pink lips, short natural "
              "lashes, natural eyebrows, real skin with visible pores, faint redness on her cheeks and nose, a few small "
              f"blemishes and freckles. Her hair in {look['hair']}, a little messy, a few flyaway strands. "
              f"She wears {look['outfit']}. {BODY} Behind her {behind}. {light}, slightly uneven. Ordinary front-camera "
              "phone photo: slightly soft focus, mild sensor noise, a little wide-angle, natural muted colours, an everyday "
              "unedited photo, not a professional shot. The room is clean and tidy.")
    model = studio.model("edit")
    for attempt in range(3):
        p = Path(studio.engine.image(model, prompt, work, f"frame-{attempt}", refs=[ref] if ref else [],
                                     resolution="720x1280"))
        good, why, _ = qa.check(p)
        if good:
            break
        log(f"{char.slug}: frame {attempt + 1} failed its check ({why})")
    log(f"{char.slug}: frame {look['scene'][0]}, {look['outfit']}, {look['hair']}")
    return p


def _voice_words(b):
    accent = ACCENT.get(b.get("home_city", ""), "")
    return (f"She has a soft, warm, gentle voice, slightly low and breathy{f', with a light {accent} accent' if accent else ''}, "
            "natural like talking to a close friend.")


def _speech_prompt(b, line):
    # Real talk, not a reading (2026-11-06: "the long pauses are awkward... make it more real"):
    # a normal conversational pace with feeling. Since 2026-11-11 a selfie: she talks into the lens.
    return ("A real selfie video: a young woman holds her phone at arm's length and talks straight into the front camera, "
            "to the viewer, like a friend on FaceTime. The phone moves only very slightly with her hand. "
            f"{_voice_words(b)} She says: \"{line}\" She talks at a normal conversational pace, with real feeling, "
            "the sentences flowing one into the next with short natural breaths. Her eyes stay on the lens, small natural "
            "head nods and tilts, calm and relaxed, small subtle movements only, natural blinks, her lips moving "
            "naturally with every word, soft relaxed lips, her mouth opening only a little as she talks, small closed-lip "
            # 2026-11-12, Sofia: gappy crooked teeth on wide grins ("horrible... scary").
            "smiles; when her teeth show they are straight, even, natural white upper teeth. "
            "Background Sound: quiet room tone.")


def voice_ref(char, studio, face):
    """Her reference voice, made once by LTX-2.5 from a voice description (Lena's is
    the clip the user liked, put there by hand)."""
    dest = char.dir / "talk" / "voice_ref.wav"
    if dest.exists():
        return dest
    s = studio.engine.build_settings(LTX_FIRST_VOICE, prompt=_speech_prompt(
            char.bible, "Honestly? The best thing I ever did... was stop waiting for someone to choose me. I chose myself."),
        image_start=str(face), image_prompt_type="S", audio_prompt_type="", video_length=241,
        resolution="448x768", seed=-1)
    clip = studio.engine.run(s, dest.parent, "voice-ref")[-1]
    _ff("-i", clip, "-vn", "-ac", "1", "-ar", "44100", dest)
    log(f"{char.slug}: reference voice made")
    return dest


# ------------------------------------------------------------------ making it
def _trim_silence(src, dest):
    # Inside the line too: any pause over 0.3s cut down to 0.2s (the long gaps sounded like
    # AI reading, 2026-11-06). InfiniteTalk lip-syncs to this audio, so the mouth follows.
    _ff("-i", src, "-af", "silenceremove=start_periods=1:start_threshold=-45dB,"
        "silenceremove=stop_periods=-1:stop_duration=0.3:stop_threshold=-42dB:stop_silence=0.2,areverse,"
        "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.3,areverse,apad=pad_dur=0.6",
        "-ac", "1", "-ar", "44100", dest)


def speak(char, studio, face, ref, line, work, slack=1.0):
    """Her saying the line: LTX-2.3 makes her voice (from her reference voice) and her face
    talking TOGETHER, and that clip is the talk (2026-11-10: it moved more naturally than
    InfiniteTalk re-animating her to the voice, the lips blurred less, and it is ~5x faster).
    Her soft voice says ~2 words a second; the clip gets room for that plus a margin (`slack`
    > 1 on a retry), so the last word is never past its end (2026-11-05). Returns (voice, clip)."""
    seconds = min(26.0, (len(line.split()) / 2.0 + 2.5) * slack)
    frames = int(seconds * 24) // 8 * 8 + 9
    # Said, not read: "..." and dashes made the model stop dead mid-line.
    said = re.sub(r"\s*(\.\.\.|…|--|—|–)\s*", ", ", line).replace(", ,", ",")
    s = studio.engine.build_settings(LTX_VOICE_MODEL, prompt=_speech_prompt(char.bible, said),
                                     image_start=str(face), image_prompt_type="S", audio_guide=str(ref),
                                     audio_prompt_type="A1OF", video_length=frames, resolution=TALK_RES, seed=-1)
    clip = studio.engine.run(s, work, "speech")[-1]
    voice = work / "speech-raw.wav"
    _ff("-i", clip, "-vn", "-ac", "1", "-ar", "44100", voice)
    return voice, clip


def _trim_ends(clip, dest, end=None):
    """The silence before her first word and after her last cut off, picture and sound
    together. Pauses INSIDE the line stay: cut out they made the clip jump at every one
    (2026-11-10, "a is defo better"). `end`: where her line is over (_line_end); anything
    after it -- LTX saying the line a second time -- goes too."""
    total = min(_duration(clip), end) if end else _duration(clip)
    err = subprocess.run(["ffmpeg", "-i", str(clip), "-af", "silencedetect=noise=-42dB:d=0.35", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    sil = list(zip(st, en + [total] * (len(st) - len(en))))
    head = next((max(0.0, b - 0.1) for a, b in sil if a <= 0.05), 0.0)
    tail = next((min(total, a + 0.6) for a, b in sil if b >= total - 0.05 and a > head), total)
    tail = min(tail, total)
    _ff("-ss", f"{head:.3f}", "-to", f"{tail:.3f}", "-i", clip, "-c:v", "libx264", "-crf", "16", "-c:a", "aac", dest)
    return dest


def animate(studio, face, voice, work, where="sitting relaxed at a podcast desk in a dim warm studio"):
    """InfiniteTalk, the test-11 recipe."""
    frames = -(-int(_duration(voice) * FPS_TALK + 1) // 4) * 4 + 1  # rounded up: never shorter than her voice
    s = studio.engine.build_settings(INFINITETALK,
        prompt=(f"A young woman {where} talks softly to someone off-camera to her right. Calm and natural, accurate "
                "lip sync, her head stays steady, slow natural blinks, a faint smile, her hands stay still. "
                "Static phone camera, realistic."),
        image_start=str(face), image_prompt_type="S", video_prompt_type="",
        audio_guide=str(voice), audio_prompt_type="A",
        video_length=frames, resolution="480x832", sliding_window_size=81, seed=-1)
    return studio.engine.run(s, work, "talk")[-1]


def _ass_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


HOOK_S = 2.6


def captions(video, keywords, work, dest, bw=False, hook="", label="", reply=""):
    """Three words at a time, centre of the frame, keywords in yellow; 1080x1920.
    `bw`: graded black and white (the A/B test of 2026-10-27), captions still in colour.
    `hook`: slammed on screen over the first ~2.5s, with a punch-in on the first frame."""
    wav = work / "final.wav"
    _ff("-i", video, "-vn", "-ac", "1", "-ar", "16000", wav)
    words = _transcribe(wav, words=True)
    # The Worker sometimes gives phrases: every word of 4+ letters in them.
    keys = {w for k in keywords or [] for w in (re.sub(r"[^\w\s]", "", k).lower().split()) if len(w) >= 4}
    groups, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == 3 or re.search(r"[.,!?]$", w["text"].strip()):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    lines = []
    for i, g in enumerate(groups):
        start = g[0]["timestamp"][0] or 0
        end = groups[i + 1][0]["timestamp"][0] if i + 1 < len(groups) else (g[-1]["timestamp"][1] or start + 1) + 0.4
        text = " ".join("{\\c&H00E5FF&}" + w["text"].strip() + "{\\c&HFFFFFF&}"
                        if re.sub(r"\W", "", w["text"]).lower() in keys else w["text"].strip() for w in g)
        # The hook holds the caption's place for its first 2.6s: one text on screen at a time.
        if hook and end <= HOOK_S:
            continue
        start = max(start, HOOK_S) if hook else start
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,{{\\pos(540,1000)\\fad(60,0)}}{text}")
    if hook:
        lines.append(hooktext.event(hook, HOOK_S, y=1000))
    if label:
        lines.append(hooktext.tag_event(label))
    if reply:
        lines.append(hooktext.reply_event(reply))
    elif hook:
        lines.append(hooktext.title_event(hook, HOOK_S, y=200 if label else 120))
    ass = work / "caps.ass"
    ass.write_text("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\n"
                   "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
                   "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
                   "MarginR, MarginV, Encoding\n"
                   "Style: Cap,Lato Black,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,1,0,0,100,100,0,0,1,4,3,5,60,60,0,1\n"
                   + hooktext.STYLE + hooktext.TAG_STYLE + hooktext.REPLY_STYLE + hooktext.TITLE_STYLE +
                   "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                   + "\n".join(lines) + "\n")
    grade = "hue=s=0,eq=contrast=1.12:brightness=-0.02," if bw else ""
    # Edited like a real clip: every other sentence punched in to 113% (a cut, not a
    # zoom), then sharpened, with a little grain over the upscale's smoothness.
    cuts = [0.0] + [w["timestamp"][1] for w in words[:-1]
                    if re.search(r"[.!?]$", w["text"].strip()) and w["timestamp"][1]]
    spans = [(cuts[i], cuts[i + 1] if i + 1 < len(cuts) else 9999) for i in range(1, len(cuts), 2)]
    # With a hook the clip opens punched in, cut back as the hook settles: motion on frame 0.
    if hook and len(cuts) > 1 and cuts[1] > 1.0:
        spans.insert(0, (0.0, 0.8))
    z = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in spans) or "0"
    zoom = (f"scale=w='trunc(max(1080,1920*iw/ih)*(1+0.13*({z}))/2)*2':h='trunc(max(1920,1080*ih/iw)*(1+0.13*({z}))/2)*2'"
            ":flags=lanczos:eval=frame,crop=1080:1920:(iw-1080)/2:(ih-1920)*0.3,")  # her face sits high

    # Under Telegram's 20MB bot download limit, or Buffer can't fetch it from the
    # channel copy ("Video could not be read from its URL", 2026-10-29: grain at
    # crf 18 made a 20s talk too big): capped at 5 Mbit/s, ~13MB for 20s.
    _ff("-i", video, "-vf", f"{grade}{zoom}unsharp=5:5:0.7,noise=alls=3:allf=t,ass={ass}", "-c:v", "libx264", "-crf", "20",
        "-maxrate", "5M", "-bufsize", "10M", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", dest)
    return dest


def _ckpts(studio):
    root = studio.engine.wan_dir / "ckpts"
    return {f for f in root.rglob("*") if f.is_file()} if root.is_dir() else set()


def _done_with(studio, before):
    """Release the stage's model; on Kaggle (PERSONA_PRUNE_CKPTS=1) also delete the
    big weights that stage downloaded. Qwen image + two LTX 22B + InfiniteTalk 14B
    do not fit Kaggle's disk together: once the talk made its own frame, every
    Kaggle talk died on its last download with "Read-only file system" or
    "Background writer channel closed" (2026-11-02)."""
    studio.engine.release()
    if not config.flag("PERSONA_PRUNE_CKPTS", "0"):
        return before
    for f in _ckpts(studio) - before:
        if f.stat().st_size > 500_000_000:
            log(f"disk: dropping {f.name} ({f.stat().st_size >> 30} GB)")
            f.unlink(missing_ok=True)
    return _ckpts(studio)


def _norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def _line_end(voice, line):
    """Where her line ends the FIRST time she says it, in seconds (+ a short breath), or
    None when its last word is never heard (cut off). LTX fills the room it is given:
    when she is done early it starts the line again (2026-11-11: both talks of the day
    said it twice), so the clip is cut here, not at the next silence. Whisper's words
    are matched to the line in order (longest common subsequence); the end is the first
    point where (nearly) all of the line has been said and her last word was just heard."""
    want = [w for w in map(_norm, line.split()) if w]
    try:
        words = [(_norm(x["text"]), x["timestamp"]) for x in _transcribe(voice, words=True)]
    except Exception as e:
        log(f"line-end check skipped: {e}")
        return float("inf")
    words = [(w, t) for w, t in words if w]
    if not want or not words:
        return None
    m = len(want)
    row, lcs = [0] * (m + 1), []
    for w, _ in words:
        new = [0] * (m + 1)
        for j in range(1, m + 1):
            new[j] = row[j - 1] + 1 if w == want[j - 1] else max(row[j], new[j - 1])
        row = new
        lcs.append(row[m])
    full = lcs[-1]
    if full < 0.6 * m:
        log(f"line-end: heard only {full}/{m} words of the line")
    for i, (w, t) in enumerate(words):
        if w == want[-1] and lcs[i] >= full - 1:
            # A breath after her last word, but never into the next one (the repeat's).
            nxt = words[i + 1][1][0] if i + 1 < len(words) else float("inf")
            return max(t[1] or t[0], min((t[1] or t[0]) + 0.35, nxt - 0.05))
    return None


def make(job, char, studio, work):
    """The whole talk. Returns (final mp4 path, line info).

    The video is rendered in a child process (`python -m persona.talk`) that exits before
    TaoMate starts: Wan2GP's release left ~7GB on the card, and TaoMate, short of VRAM,
    ran 8x slower and then hung for 12 hours on Freya (2026-11-12). PERSONA_TALK_SPLIT=0
    renders in this process."""
    work.mkdir(parents=True, exist_ok=True)
    if config.flag("PERSONA_TALK_SPLIT", "1"):
        (work / "render-job.json").write_text(json.dumps(job))
        (work / "render.json").unlink(missing_ok=True)
        subprocess.run([sys.executable, "-m", "persona.talk", str(work / "render-job.json"), str(work), char.slug],
                       cwd=str(Path(__file__).resolve().parent.parent), check=True)
        r = json.loads((work / "render.json").read_text())
    else:
        r = render(job, char, studio, work)
        studio.engine.release()
    return finish(job, char, Path(r["video"]), Path(r["face"]), r["info"], work)


def render(job, char, studio, work):
    """Her line, first frame and LTX clip cut at the line end. Writes and returns
    render.json: {video, face, info}."""
    work.mkdir(parents=True, exist_ok=True)
    info = find_line(job, work)
    log(f"{char.slug}: line {info['line']!r}")
    # One model in RAM at a time (Engine.release), and on Kaggle on disk too.
    have = _ckpts(studio)
    if not (char.dir / "talk" / "voice_ref.wav").exists():
        ref = voice_ref(char, studio, portrait(char, studio))
        have = _done_with(studio, have)
    ref = char.dir / "talk" / "voice_ref.wav"
    look = pick_scene(char)
    # A day theme's room and outfit (Christmas tree and a red sweater ...), from the Worker.
    if job.get("look"):
        th = job["look"]
        name, where, behind, light = look["scene"]
        look = {**look, "outfit": th.get("outfit") or look["outfit"],
                "scene": (job.get("theme") or name, th.get("where") or where, th.get("behind") or behind, th.get("light") or light)}
    face = frame(char, studio, look, work)
    have = _done_with(studio, have)
    voice, clip = speak(char, studio, face, ref, info["line"], work)
    have = _done_with(studio, have)
    # Her last word must be in the voice (2026-11-05: it was sometimes cut): heard by
    # Whisper, and made again with more room once if it is missing.
    end = _line_end(voice, info["line"])
    if end is None:
        log(f"{char.slug}: last word cut, speaking it again with more room")
        voice, clip = speak(char, studio, face, ref, info["line"], work, slack=1.35)
        have = _done_with(studio, have)
        end = _line_end(voice, info["line"])
    if end and end != float("inf"):
        log(f"{char.slug}: line ends at {end:.1f}s of {_duration(clip):.1f}s")
    video = _trim_ends(clip, work / "talk.mp4", end=end if end and end != float("inf") else None)
    r = {"video": str(video), "face": str(face), "info": info}
    (work / "render.json").write_text(json.dumps(r, ensure_ascii=False))
    return r


def finish(job, char, video, face, info, work):
    """Her face refined, then captions. Returns (final mp4 path, line info)."""
    # Her face refined by TaoMate-FVSR (2026-11-11, the user's pick: sharp lips and teeth
    # that follow her voice); GFPGAN (strong blend) if it fails or PERSONA_TAOMATE=0.
    try:
        if not config.flag("PERSONA_TAOMATE", "1"):
            raise RuntimeError("off")
        from . import taomate
        video = taomate.refine(video, work / "talk-tao.mp4", ref=face)
        log(f"{char.slug}: face refined (TaoMate)")
    except Exception as e:
        log(f"TaoMate skipped ({e}); GFPGAN instead")
        try:
            from . import faceswap
            video = faceswap.sharpen_video(video, work / "talk-sharp.mp4", blend=0.8)
        except Exception as e:
            log(f"face sharpening skipped: {e}")
    # Her series label ("Getting over him #14", or "... Part 2"), and the comment she answers.
    label = f"{job['series']} #{job['episode']}" if job.get("series") and job.get("episode") else ""
    if job.get("follow_line") and job.get("series"):
        label = f"{job['series']} · Part 2"
    final = captions(video, info.get("keywords"), work, work / "final.mp4", bw=bool(job.get("bw")),
                     hook=info.get("hook") or "", label=label, reply=job.get("reply_comment") or "")
    return final, info


if __name__ == "__main__":
    # make()'s render step in its own process: python -m persona.talk <job.json> <work> <slug>
    from .character import Character
    from .engine import Engine
    from .studio import Studio
    config.load_env()
    render(json.loads(Path(sys.argv[1]).read_text()), Character(sys.argv[3]), Studio(Engine()), Path(sys.argv[2]))
