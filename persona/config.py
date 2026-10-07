"""Paths, .env and the persona's own settings file.

Code lives in the repo; everything about HER lives outside it, under PERSONA_HOME
(default ~/.local/share/mpt-persona). The repo is public, and her face, her
storyline and anything made for the Fanvue lane have no business on GitHub. The
Kaggle path gets what it needs through a private Kaggle dataset (persona/kaggle.py),
never through a commit.

PERSONA_HOME/
  settings.json          active character + which model fills each role
  <slug>/
    bible.json           who she is (see character.py)
    refs/                master reference shots, the identity every image is edited from
    casting/             candidates from /cast, before one is picked
    items/<id>.jpg|.json every generated image/video and what made it
    storyline.jsonl      one approved beat per line
    votes.json           tag scores from 👍/👎
    outbox/fanvue/       approved for Fanvue but not posted by API (no token yet)
    outbox/instagram/    approved for Instagram (no posting API, upload by hand)
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent


def _dotenv(path):
    """Same shallow parser as local/wan2gp_bot.py, so one .env serves both."""
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


def load_env():
    """Fill os.environ from the repo's .env without overriding anything already set.
    Into os.environ rather than a private dict because tiktok.py and llm.py read
    their own credentials from there."""
    for k, v in _dotenv(REPO / ".env").items():
        os.environ.setdefault(k, v)


def env(name, default=""):
    return (os.environ.get(name) or default).strip()


def flag(name, default="0"):
    return env(name, default).lower() not in ("0", "false", "no", "")


def home():
    path = Path(env("PERSONA_HOME") or Path.home() / ".local/share/mpt-persona").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


# Role -> registry id. A role is a job in the pipeline, not a model: swapping the
# model behind one is a /use command (or an edit of settings.json), nothing else.
DEFAULT_ROLES = {
    "cast": "qwen21",              # text-to-image, used only by /cast
    "edit": "qwen21",              # reference-guided image, every social-lane scene
    "fanvue_edit": "qwen21-uncensored",  # same job for the local-only Fanvue lane
    "video": "wan22-i2v-calm",     # image-to-video (calm motion; wan22-i2v is the livelier one)
}


def settings_path():
    return home() / "settings.json"


def load_settings():
    path = settings_path()
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault("active", "")
    data["roles"] = {**DEFAULT_ROLES, **data.get("roles", {})}
    return data


def save_settings(data):
    path = settings_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)
