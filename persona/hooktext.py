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
