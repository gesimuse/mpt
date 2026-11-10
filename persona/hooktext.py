"""The on-screen hook (2026-11-06, from the koubo packaging playbook: 0-2s is the claim
slammed on screen -- no hello, no name). Black words on a white box below her face,
tilted a little, popping in from 135%: the same look on talks and Reels. ASS, burned in
by ffmpeg's libass."""


def ass_time(s):
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def wrap(text, width=16):
    """ASS line breaks every ~`width` characters, on spaces."""
    out, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\\N".join(out + [cur])


STYLE = "Style: Hook,Lato Black,92,&H00111111,&H00111111,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,3,18,0,5,60,60,0,1\n"


def event(hook, end, y=1000):
    return (f"Dialogue: 1,{ass_time(0)},{ass_time(end)},Hook,,0,0,0,,"
            f"{{\\pos(540,{y})\\frz3\\fscx135\\fscy135\\t(0,140,\\fscx100\\fscy100)\\fad(0,150)}}{wrap(hook.upper())}")


def ass(hook, end, dest, y=1000):
    """A 1080x1920 .ass file with only the hook in it."""
    dest.write_text("[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n[V4+ Styles]\n"
                    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
                    "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding\n" + STYLE +
                    "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                    + event(hook, end, y) + "\n")
    return dest


# Her series label at the top of every talk, "GETTING OVER HIM #14" (2026-11-10: a numbered
# series gives a viewer a reason to follow for the next one), and a viewer's comment she
# answers, in a white bubble like TikTok's own "reply to comment" videos.
TAG_STYLE = "Style: Tag,Lato Black,50,&H00FFFFFF,&H00FFFFFF,&H00000000,&H99000000,1,0,0,0,100,100,2,0,3,14,0,8,60,60,0,1\n"
REPLY_STYLE = "Style: Reply,Lato Black,52,&H00222222,&H00222222,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,3,22,0,8,40,40,0,1\n"


def tag_event(label, y=120):
    return f"Dialogue: 2,{ass_time(0)},9:59:59.00,Tag,,0,0,0,,{{\\pos(540,{y})}}{label.upper()}"


def reply_event(comment, y=215, x=360):
    text = comment.strip()
    if len(text) > 140:
        text = text[:137].rsplit(" ", 1)[0] + "..."
    return (f"Dialogue: 2,{ass_time(0)},9:59:59.00,Reply,,0,0,0,,{{\\pos({x},{y})\\fad(200,0)}}"
            f"{{\\fs40\\c&H777777&}}Reply to a comment{{\\fs52\\c&H222222&}}\\N{wrap(text, 26)}")
