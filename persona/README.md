# Persona studio

One AI persona who stays the same person from post to post, with an ongoing story.
Claude Code designs her (the `new-persona` skill), Qwen-Image casts her face, every later image is edited from her
master refs, Wan 2.2 animates the good ones, and Telegram is where you approve and
publish.

Separate from the aibeauty autopilot. Nothing here changes `autopilot.py`, the
Cloudflare Worker, or `local/wan2gp_bot.py`.

## How it fits together

```
new-persona skill -> bible.json -> cast (t2i) -> ⭐ pick -> master refs (edit) -> 🔒 lock
                                                                                     |
     storyline + votes -> scene writer (LLM) -> render from refs -> face + age gate -> Telegram review
                                                                                     |
                               👍/👎 votes · 🎬 Wan 2.2 video · 📱 TikTok · 📷 IG outbox · 💜 Fanvue
```

* **Everything runs through Wan2GP** (`~/apps/Wan2GP`, in-process via `shared/api.py`):
  Qwen-Image 2.1, Qwen-Image 2512, Qwen-Image-Edit 2511 and Wan 2.2 i2v are all
  Wan2GP model types. One runtime, one model in RAM at a time.
* **Her data is outside the repo**, under `PERSONA_HOME` (default
  `~/.local/share/mpt-persona/<slug>/`). The repo is public.
* **Two lanes.** `social` must be safe for TikTok and Instagram and may run on
  Kaggle. `fanvue` runs **only on this machine**, may use the uncensored model, and
  can never reach TikTok, Instagram, gh-pages or Kaggle (enforced in
  `registry.check` and `social._guard`).
* **Adult-only gate.** Every prompt carries an adult anchor, and `faceid.py`
  drops any image whose estimated age is under `PERSONA_MIN_AGE` (21), in both lanes.

## Models: `persona/models/*.json`

| id | what | licence | where |
|---|---|---|---|
| `qwen21` | Qwen-Image 2.1 7B, t2i + edit, up to 10 refs, Pruna 8-step | Qwen Research (non-commercial) | local + Kaggle |
| `qwen21-uncensored` | abenzerps re-upload, no added filter | Qwen Research (non-commercial) | local only, Fanvue lane |
| `qwen2512` | Qwen-Image 2512 20B, t2i, Lightning 8-step | Apache-2.0 | local + Kaggle |
| `qwen-edit-2511` | Qwen-Image-Edit 2511 20B, 3 refs, Lightning 8-step | Apache-2.0 | local + Kaggle |
| `wan22-i2v` | Wan 2.2 I2V Enhanced Lightning v2 | Apache-2.0 | local + Kaggle |

**Add a model:** drop a JSON file in `persona/models/` (fields documented in
`registry.py`). If Wan2GP doesn't ship it, add a finetune definition to
`persona/wan2gp_finetunes/`; the engine installs it into Wan2GP on start.

**Roles** decide which model does each job: `cast`, `edit`, `fanvue_edit`, `video`.
Switch with `/use edit qwen-edit-2511`. Defaults are in `config.DEFAULT_ROLES`.

**Licence:** Qwen-Image 2.1 is non-commercial. Captions warn on every item made
with a non-commercial model. Before monetizing, switch `cast` to `qwen2512` and
`edit`/`fanvue_edit` to `qwen-edit-2511`, and re-cast so her master refs are
Apache-made too. Alternatively, get a commercial licence from Qwen
(model-business@notice.qwencloud.com). `PERSONA_COMMERCIAL_ONLY=1` blocks 💜 on
non-commercial items.

## Setup

1. Make a bot with @BotFather, create a private channel, and add the bot as admin.
   Put both in `.env`: `PERSONA_BOT_TOKEN`, `PERSONA_CHAT_ID`. Optionally add a
   second private channel for the Fanvue lane: `PERSONA_FANVUE_CHAT_ID`.
3. Stop `local/wan2gp_bot.py` and the Wan2GP UI (same RAM), then start the bot:
   ```bash
   ~/apps/Wan2GP/.venv/bin/python -m persona.bot
   ```
   The first image job downloads the model (Qwen 2.1: ~7GB plus a ~9GB text encoder).
   To keep it running: `cp persona/mpt-persona-bot.service ~/.config/systemd/user/ && systemctl --user enable --now mpt-persona-bot`
   (it declares `Conflicts=` with the Wan2GP bot, so starting one stops the other).

## Making her

In Claude Code, in this repo: **`/new-persona`** (or just ask for a new persona). The
skill (`.claude/skills/new-persona/SKILL.md`) has Claude write her bible with you in
chat, then cast faces, build master refs and lock them on the local GPU, showing every
step for you to pick. She becomes the active persona; the bot's schedule posts her
from the next slot. `/cast`, `/refs` and the ⭐/🔒 buttons still work in Telegram for
re-casting.

Day to day in the channel: `/scene 3` (social), `/fanvue 2`, `/slot` (one scheduled
post now). `PERSONA_SCHEDULE=09:00,12:00,…` posts a slot at each time.

While `PERSONA_PUBLISH=0` (the default) only 👍/👎/🎬 show: the evaluation phase.
With it set to 1, the buttons on each result: 👍 adds the scene to her storyline and boosts its
tags, 👎 lowers them, 🎬 makes a 5s video, 📱 queues a TikTok draft (on her own
account, `PERSONA_TIKTOK_ACCOUNT`), 📷 copies to the Instagram outbox, and 💜 posts
to Fanvue (or the outbox until the API token arrives). Reply to an image with
text to animate it with that motion.

Fine-tune her any time in `bible.json`: personality, places, styles, and the
lane cues (`lanes.fanvue.cue` is yours to write; it never goes to a hosted LLM).

## Cloud (laptop closed): the Cloudflare Worker

Everything that isn't GPU work runs in the existing Cloudflare Worker
(`worker/src/persona/`); Kaggle does the GPU work; GitHub only holds the code.

* **Schedule** (Cloudflare cron, every 15 min, UTC): 02:30-03:15 render (10 scenes
  written on Workers AI from her bible, votes and storyline, then a Kaggle T4 job;
  retried, and skipped if today's already started); 06/09/12/15/18 post slots into
  `PERSONA_CHAT_ID`; every tick moves the on-demand Kaggle job along.
* **Taps**: 👍 👎 update her tag scores and storyline; 🎬 or a reply (= motion
  prompt) queues a video; 🔁 queues a redraw of the same scene. One Kaggle job runs
  up to 3 of them and the results come back as replies. With the bot's webhook on
  `/persona` it is instant; with `PERSONA_POLL=1` taps are collected every 15 min.
* **State** (votes, storyline, recent scenes, what's posted, the job queue) lives in
  Workers KV. Her identity (bible, refs, face sheet) is in the private Kaggle
  dataset `<KAGGLE_USERNAME>/mpt-persona-<slug>`.
* **Kaggle jobs** are pushed through Kaggle's API with a tiny bootstrap that clones
  this repo and runs `persona/kaggle_kernel.py`. Photos for 🎬 reach Kaggle via a
  signed, 6-hour Worker link, never the bot token.
* After creating or re-casting a persona on the laptop: `python -m persona.cli sync`
  uploads her refs to Kaggle and her bible to the Worker. Set `PERSONA_SLUG` in
  `worker/wrangler.toml` to switch which persona the cloud runs.

Admin (Bearer `PERSONA_ADMIN_SECRET`): `GET/POST /persona/admin/state`,
`POST /persona/admin/bible`, `POST /persona/admin/run?what=render|post&slot=N|jobs|poll`.

`.github/workflows/persona_kaggle.yml` and `persona/actions.py` are the previous
GitHub Actions version, kept for manual use; the workflow is disabled.

## CLI

Same pipeline without Telegram: `python -m persona.cli --help`.

## Tested on this laptop (2026-10-01, RTX A4500 16GB)

| Step | Result |
|---|---|
| Qwen 2.1 text-to-image, Pruna 8-step | ~21s/image once loaded (first load downloads ~17GB) |
| Cast → pick → 4 master refs → lock | refs score 0.62 (profile) to 0.90 (front) against the pick |
| Scene from refs | ~65s/image, face match 0.81-0.90, one person, outfit follows the scene |
| Wan 2.2 video from a scene | 5s clip in ~8.5 min; motion tends to over-act |
| Uncensored model, Fanvue lane | loads via the finetune, face match 0.89 |
| Bot commands and buttons | tested with Telegram mocked, not yet against a live bot |

Lesson baked in: scenes get only the two head refs (`PERSONA_SCENE_REFS`). With
all five, Qwen drew her three times in one shot and copied the ref outfit; the
face check now also rejects any image where she appears more than once.

## Known unknowns

* **Fanvue's API is waitlisted**, so `fanvue.py` is unverified against a live
  account (same as `worker/src/fanvue.js`). Until there's a token, 💜 fills
  `outbox/fanvue/`.
* Kaggle T4 renders images (tested 2026-10-04: ~5-7 min each, face match 0.88-0.90). Videos on the T4 inside the same kernel are not yet tested.
* InsightFace's face models are research-licensed; `PERSONA_FACE_CHECK=0` turns
  the gate off, and that disables the age check too.
