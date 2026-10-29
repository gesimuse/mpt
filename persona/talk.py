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

from . import config, qa

FPS_TALK = 25
LTX_VOICE_MODEL = {"id": "ltx-voice", "wan2gp_model_type": "ltx2_22B_distilled_1_1"}
LTX_FIRST_VOICE = {"id": "ltx-first-voice", "wan2gp_model_type": "ltx2_25_22B_distilled"}
INFINITETALK = {"id": "infinitetalk", "wan2gp_model_type": "infinitetalk",
                "profile": "profiles/wan_i2v/Image2Video lightx2v - 4 Steps.json"}

# Lena/MPT body wording (memory mpt-body-shape: slim and toned, full bust only).
BODY = ("Her body is a flawless toned model physique: a full, round, perky, large bust, a slim narrow waist "
        "and slim arms -- an hourglass figure.")
ACCENT = {"Prague": "Czech", "Tokyo": "Japanese", "Moscow": "Russian", "London": "British", "Paris": "French",
          "Oslo": "Norwegian", "Los Angeles": "American", "San Diego": "American", "Dubai": "Middle Eastern",
          "Madrid": "Spanish", "Bali": "Indian", "Rome": "Italian"}


# A new first frame for every talk (2026-10-29: one portrait per girl made every talk
# the same shot, and it read as fake). Each: where she sits, its light, the mic.
SCENES = [
    ("podcast", "seated at a podcast desk, a black broadcast microphone on a boom arm at the lower left of the frame",
     "a dim podcast studio behind her: dark acoustic panels, a warm lamp and soft bokeh lights", "warm moody side light"),
    ("couch", "sitting sideways on a beige couch at home, a small lavalier microphone clipped to her top",
     "a lived-in living room behind her: a window with sheer curtains, a plant, a shelf of books", "soft daylight from the window"),
    ("car", "in the driver's seat of a parked car, seatbelt on, filmed from the phone on the dashboard",
     "the car interior and a street through the side window behind her", "flat overcast daylight through the windshield"),
    ("kitchen", "leaning on a kitchen counter holding a mug, a lavalier microphone clipped to her top",
     "a home kitchen behind her: cabinets, a kettle, warm pendant lights", "warm evening pendant light"),
    ("bedroom", "sitting cross-legged on her bed, filmed from a phone on a tripod",
     "a cosy bedroom behind her: a headboard, fairy lights, a nightstand lamp", "a ring light from the front and warm lamps"),
    ("cafe", "sitting at a small café table by the window, a lavalier microphone clipped to her top",
     "a busy café behind her, people and the counter out of focus", "bright natural window light"),
    ("street", "standing on a city street at golden hour, holding a handheld interview microphone",
     "a city sidewalk, shop fronts and passers-by out of focus behind her", "low warm golden-hour sun"),
    ("balcony", "sitting on a balcony chair at sunset, a lavalier microphone clipped to her top",
     "a balcony railing and rooftops behind her", "soft pink sunset light"),
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
        "no retouching, slight sensor noise, a little imperfect and lived-in.")


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
    p = pipeline("automatic-speech-recognition", model=str(local) if local.exists() else "openai/whisper-large-v3",
                 tokenizer=tok, torch_dtype=torch.float16, device="cuda")
    try:
        r = p(str(wav), return_timestamps="word" if words else True, generate_kwargs={"language": "en"})
    finally:
        del p
        import gc
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
        r = ({"line": job["given_line"], "keywords": job.get("given_keywords") or [], "caption": job.get("given_caption", "")}
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
        r = _worker("/persona/admin/laptop/talkline", {**ask, "transcript": text[:3000], "title": "", "any_topic": True})
        if not r.get("line"):
            raise RuntimeError("couldn't make a line from that video")
        return {**r, "source": "", "source_title": job["given_url"]}
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
                scored.append((s, c))
        cands += [c for _, c in sorted(scored, key=lambda x: -x[0])[:4]]
    for c in cands[:8]:
        audio = work / f"src-{c['id']}.wav"
        try:
            raw = work / f"src-{c['id']}"
            subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bestaudio/best",
                            "-o", f"{raw}.%(ext)s", c["url"] or c["id"]], check=True, timeout=300)
            got = next(work.glob(f"src-{c['id']}.*"))
            _ff("-i", got, "-t", "75", "-ac", "1", "-ar", "16000", audio)
            text = _transcribe(audio)
        except Exception as e:
            log(f"source {c['id']} unusable: {type(e).__name__}: {str(e)[:150]}")
            continue
        if len(text.split()) < 15:
            continue
        log(f"source {c['id']} ({c['views']} views, {c.get('shares', 0)} shares, {c.get('saves', 0)} saves, {c.get('likes', 0)} likes): {c['title'][:60]!r}")
        r = _worker("/persona/admin/laptop/talkline", {"persona": job.get("persona"), "character": job.get("character"),
                                                        "topic": job.get("topic"), "transcript": text[:3000],
                                                        "title": c["title"]})
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
    return re.sub(r",?\s*(glamorous|soft|natural|glam)[^,]*makeup[^,]*", "", b.get("identity") or "").strip(", ")


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
    ref = next((p for p in [heads / "three-quarter-right.jpg", heads / "01-front.jpg", char.dir / "refs" / "01-front.jpg"]
                if p.exists()), None)
    _, where, behind, light = look["scene"]
    prompt = (f"Photo of the woman from the reference image, a {b.get('age', 24)}-year-old woman ({_identity(b)}), "
              f"her hair in {look['hair']}, {where}, medium shot from the waist up, seen at three-quarters, her head "
              "turned a little to the right of the frame, her eyes on someone just off-camera to the right, mid-sentence, "
              f"relaxed. She wears {look['outfit']} showing cleavage. {BODY} Her face fully visible. Behind her {behind}. "
              f"{light}. {REAL}")
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
            "calm and natural like talking to a close friend.")


def _speech_prompt(b, line):
    return ("A calm, quiet, intimate moment, filmed on a locked-off static camera. A young woman "
            "sits still and relaxed, turned toward someone off-camera to the right. "
            f"{_voice_words(b)} She says softly: \"{line}\" Understated, low-energy delivery, her lips move gently and "
            "naturally with every word, slow natural blinks. Background Sound: quiet room tone.")


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
    _ff("-i", src, "-af", "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB,areverse,apad=pad_dur=0.5", "-ac", "1", "-ar", "44100", dest)


def speak(char, studio, face, ref, line, work):
    """Her voice saying the line: LTX-2.3 with her reference voice; only its audio is kept."""
    seconds = min(20.0, len(line.split()) / 2.3 + 1.5)
    frames = int(seconds * 24) // 8 * 8 + 9
    s = studio.engine.build_settings(LTX_VOICE_MODEL, prompt=_speech_prompt(char.bible, line),
                                     image_start=str(face), image_prompt_type="S", audio_guide=str(ref),
                                     audio_prompt_type="A1OF", video_length=frames, resolution="448x768", seed=-1)
    clip = studio.engine.run(s, work, "speech")[-1]
    raw = work / "speech-raw.wav"
    _ff("-i", clip, "-vn", "-ac", "1", "-ar", "44100", raw)
    voice = work / "voice.wav"
    _trim_silence(raw, voice)
    return voice


def animate(studio, face, voice, work, where="sitting relaxed at a podcast desk in a dim warm studio"):
    """InfiniteTalk, the test-11 recipe."""
    frames = int(_duration(voice) * FPS_TALK) // 4 * 4 + 1
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


def captions(video, keywords, work, dest, bw=False):
    """Three words at a time, centre of the frame, keywords in yellow; 1080x1920.
    `bw`: graded black and white (the A/B test of 2026-10-27), captions still in colour."""
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
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,{{\\pos(540,1000)\\fad(60,0)}}{text}")
    ass = work / "caps.ass"
    ass.write_text("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\n"
                   "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
                   "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
                   "MarginR, MarginV, Encoding\n"
                   "Style: Cap,Lato Black,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,1,0,0,100,100,0,0,1,4,3,5,60,60,0,1\n"
                   "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                   + "\n".join(lines) + "\n")
    grade = "hue=s=0,eq=contrast=1.12:brightness=-0.02," if bw else ""
    # Edited like a real clip: every other sentence punched in to 113% (a cut, not a
    # zoom), then sharpened, with a little grain over the upscale's smoothness.
    cuts = [0.0] + [w["timestamp"][1] for w in words[:-1]
                    if re.search(r"[.!?]$", w["text"].strip()) and w["timestamp"][1]]
    spans = [(cuts[i], cuts[i + 1] if i + 1 < len(cuts) else 9999) for i in range(1, len(cuts), 2)]
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


def make(job, char, studio, work):
    """The whole talk. Returns (final mp4 path, line info)."""
    work.mkdir(parents=True, exist_ok=True)
    info = find_line(job, work)
    log(f"{char.slug}: line {info['line']!r}")
    # One model in RAM at a time (Engine.release).
    if not (char.dir / "talk" / "voice_ref.wav").exists():
        ref = voice_ref(char, studio, portrait(char, studio))
        studio.engine.release()
    ref = char.dir / "talk" / "voice_ref.wav"
    look = pick_scene(char)
    face = frame(char, studio, look, work)
    studio.engine.release()
    voice = speak(char, studio, face, ref, info["line"], work)
    studio.engine.release()
    video = animate(studio, face, voice, work, where=look["scene"][1])
    studio.engine.release()
    final = captions(video, info.get("keywords"), work, work / "final.mp4", bw=bool(job.get("bw")))
    return final, info
