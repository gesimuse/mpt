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
    out = subprocess.run([sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "--flat-playlist",
                          "--playlist-end", str(limit), "-J", url], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        log(f"search {source!r} failed: {out.stderr.strip()[-200:]}")
        return []
    found = []
    for e in (json.loads(out.stdout).get("entries") or []):
        if not e or not e.get("id"):
            continue
        found.append({"id": e["id"], "url": e.get("url") or e.get("webpage_url") or "",
                      "views": e.get("view_count") or 0, "seconds": e.get("duration") or 0,
                      "title": (e.get("title") or e.get("description") or "")[:200],
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
    p = pipeline("automatic-speech-recognition", model=str(Path.home() / "apps/Wan2GP/ckpts/whisper-large-v3"),
                 tokenizer=tok, torch_dtype=torch.float16, device="cuda")
    try:
        r = p(str(wav), return_timestamps="word" if words else True, generate_kwargs={"language": "en"})
    finally:
        del p
        torch.cuda.empty_cache()
    return r["chunks"] if words else r["text"].strip()


def find_line(job, work):
    """A popular short on her topic, not used before, transcribed and rewritten into
    her own line by the Worker. Returns {line, keywords, source, source_title}."""
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
            return {**r, "source": c["id"], "source_title": c["title"], "source_views": c["views"]}
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


def _voice_words(b):
    accent = ACCENT.get(b.get("home_city", ""), "")
    return (f"She has a soft, warm, gentle voice, slightly low and breathy{f', with a light {accent} accent' if accent else ''}, "
            "calm and natural like talking to a close friend.")


def _speech_prompt(b, line):
    return ("A calm, quiet late-night podcast moment, filmed on a locked-off static camera. A young woman in a black top "
            "sits still and relaxed at a podcast desk in a dim warm studio, turned toward the host off-camera to the right. "
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


def animate(studio, face, voice, work):
    """InfiniteTalk, the test-11 recipe."""
    frames = int(_duration(voice) * FPS_TALK) // 4 * 4 + 1
    s = studio.engine.build_settings(INFINITETALK,
        prompt=("A young woman on a podcast sits relaxed at a desk in a dim warm studio and talks softly to the host "
                "off-camera to her right. Calm and natural, accurate lip sync, her head stays steady, slow natural "
                "blinks, a faint smile, her hands rest on the desk. Static camera, cinematic, realistic."),
        image_start=str(face), image_prompt_type="S", video_prompt_type="",
        audio_guide=str(voice), audio_prompt_type="A",
        video_length=frames, resolution="480x832", sliding_window_size=81, seed=-1)
    return studio.engine.run(s, work, "talk")[-1]


def _ass_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def captions(video, keywords, work, dest):
    """Three words at a time, centre of the frame, keywords in yellow; 1080x1920."""
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
    _ff("-i", video, "-vf", f"scale=1080:-2:flags=lanczos,crop=1080:1920,ass={ass}", "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", dest)
    return dest


def make(job, char, studio, work):
    """The whole talk. Returns (final mp4 path, line info)."""
    work.mkdir(parents=True, exist_ok=True)
    info = find_line(job, work)
    log(f"{char.slug}: line {info['line']!r}")
    face = portrait(char, studio)
    ref = voice_ref(char, studio, face)
    voice = speak(char, studio, face, ref, info["line"], work)
    video = animate(studio, face, voice, work)
    final = captions(video, info.get("keywords"), work, work / "final.mp4")
    return final, info
