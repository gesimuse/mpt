# Local Wan2GP bot

Send a photo into the **MPT** channel with a motion prompt as its caption (or reply to
a photo with one). While this computer is on, `wan2gp_bot.py` generates the clip here
on the RTX A4500 and posts the mp4 into the **MPT Videos** channel.

Nothing else is involved: no GitHub, no gh-pages, no `posted.json`, no TikTok, no
Kaggle, no Hugging Face, no hosted model of any kind. The image and the video exist on
this machine and in Telegram, and nowhere else.

The cloud flow is untouched. The Cloudflare Worker, the buttons on generated photos
and the autopilot pipeline all behave exactly as before — this runs alongside them on
its own bot and shares no code or state with them.

## Setup

1. **Make a second bot.** @BotFather → `/newbot`. It must be a *different* bot from the
   mpt one: that bot's updates go to the Cloudflare Worker over a webhook, and Telegram
   refuses `getUpdates` for a bot that has one (409). Two bots, one webhook and one
   long-poll, no conflict.
2. **Add it to both channels as an admin** — MPT and MPT Videos. A bot only receives
   channel posts if it is an admin, and it needs post rights to answer and deliver.
3. **Put the token in `.env`:**
   ```
   LOCAL_BOT_TOKEN=123456:AA...
   ```
   That is the only value you have to supply. The rest of the block is already in
   `.env`: `LOCAL_SRC_CHAT_ID` / `LOCAL_OUT_CHAT_ID` are the same MPT and MPT Videos
   channels the cloud flow uses, and the frames/steps match the Motion Forge settings
   in `niches.json` (81 frames ≈ 5s at 16fps, 8 steps).
4. **Stop the Wan2GP web UI** if it is running, then start the bot with Wan2GP's own
   python:
   ```bash
   pkill -f "wgp.py --profile"          # only if the UI is up
   ~/apps/Wan2GP/.venv/bin/python local/wan2gp_bot.py
   ```
   Everything else — both chat ids, frames, steps, resolution — is already in `.env`.

On first start it ignores everything already in the channel, so old photos don't all
generate at once. From then on, its position is remembered in
`~/.local/state/mpt-wan2gp/state.json` — restart it and it picks up where it stopped,
including photos sent while it was down.

## Using it

| You send | What happens |
| --- | --- |
| Photo + caption | Caption is the prompt. Queued immediately. |
| Photo, no caption | It says so. Reply to the photo with a prompt to start. |
| Reply to a photo | That text becomes the prompt for that photo. Repeat for another take. |

**One reply per photo**, and only the acknowledgement — "🖥 Queued for the local GPU."
There is no completion message: the video arriving in MPT Videos is the completion, and
a second reply saying so was just noise. A *failure* still replies, because that is the
one outcome the videos channel cannot show you.

Expect roughly **8 minutes** per clip at 81 frames / 8 steps / 832x480 on this GPU
(measured), plus a few seconds of runtime load on the first job after a restart.

Several photos in a row queue up and run one at a time — one GPU, and Wan 2.2 14B on
16 GB with 30 GB of host RAM has no room for a second process. Each queued photo is
told how many jobs are ahead of it.

### Prompts

Wan2GP's own `docs/PROMPTS.md` recommends a per-second timeline for this exact
checkpoint, "because its own default prompt uses that kind of timeline format", so the
bot puts your prompt into that shape with a plain template — no model, no rewriting of
your words:

- **One sentence** → dropped into the template's beats (`begins`, `carries`, `settles`).
- **Several lines** → one line per second, in the order you wrote them. This is the one
  to use when the prompt matters:
  ```
  she starts to turn away, hips leading
  she looks back over her shoulder
  her hair settles across her back
  a slow smile
  she holds the look
  ```
- **Already `(at X seconds: ...)` lines** → passed through untouched.

Set `LOCAL_TIMELINE=0` in `.env` to send exactly what you typed, with no template.

## The UI and the bot cannot both run

They are the same runtime and the same RAM. Wan 2.2 14B under profile 4 holds ~26 GB
of this machine's 30 GB, so only one of them fits: stop the UI to run the bot, stop the
bot to use the UI.

The bot loads that runtime **in-process** (`shared/api.py`) instead of shelling out per
job, so the model is loaded once and every clip after the first skips it entirely. That
is also why it must run under `~/apps/Wan2GP/.venv/bin/python` — torch and mmgp live
there. The runtime is loaded lazily on the first photo, not at startup, so the bot comes
up and answers in Telegram even if something is wrong with the runtime.

## Operating it by hand

Everything below is the whole job — there is no other moving part.

**Start** (the Wan2GP web UI must be stopped first — same runtime, same RAM):

```bash
pkill -f "wgp.py --profile"                                   # stop the UI if it is up
cd ~/apps/mpt
~/apps/Wan2GP/.venv/bin/python local/wan2gp_bot.py            # foreground, Ctrl-C to stop
```

Backgrounded instead, with a log to read afterwards:

```bash
nohup ~/apps/Wan2GP/.venv/bin/python local/wan2gp_bot.py > /tmp/wan2gp-bot.log 2>&1 &
tail -f /tmp/wan2gp-bot.log
```

A healthy start looks like exactly this:

```
[wan2gp-bot] bot @mpt_local_bot | photos from -1004449497323 | videos to -1004314616972
[wan2gp-bot] first start: ignoring anything posted before now
[wan2gp-bot] waiting for photos
```

**Stop**, and go back to the web UI:

```bash
pkill -f "local/wan2gp_bot.py"
cd ~/apps/Wan2GP && .venv/bin/python wgp.py --profile 4 --attention sdpa
```

**Is it running?**

```bash
pgrep -af "local/wan2gp_bot.py" || echo "not running"
free -g                     # during a job the runtime holds ~26GB of the 30
nvidia-smi                  # and the GPU while it is actually denoising
```

### Setting up a bot yourself (what was done here, repeatable)

1. **@BotFather → `/newbot`**, pick a name and a username ending in `bot`. It replies
   with a token.
2. **Add the bot to both channels as an admin.** Channel → *Administrators* → *Add
   Admin* → search the username. A channel has no plain "add member" for bots, and a
   non-admin bot receives no posts at all — this is the step that silently breaks
   everything if skipped.
3. **`LOCAL_BOT_TOKEN=` in `.env`**, nothing else (the rest of the block is filled in).
4. Verify before starting anything, substituting your own token and ids:

   ```bash
   T=<token>
   curl -s "https://api.telegram.org/bot$T/getMe"
   curl -s "https://api.telegram.org/bot$T/getChatMember?chat_id=-1004449497323&user_id=<bot id>"
   ```

   `getMe` gives the bot id used in the second call. `getChatMember` must say
   `"status":"administrator"` with `"can_post_messages":true` for BOTH channels. While
   the bot is not a member, Telegram answers `Bad Request: chat not found` — that
   message means "not in the channel", not "wrong id".

### When something is wrong

| Symptom | What it is |
| --- | --- |
| `409 Conflict` on getUpdates | The token has a webhook. You used the mpt bot's token; this needs its own bot. |
| `Bad Request: chat not found` | The bot is not in that channel yet. Add it as an admin. |
| Photo posted, bot says nothing | Bot not admin, not running, or the photo is in a chat other than `LOCAL_SRC_CHAT_ID`. |
| The reply says the job failed | The message carries the real reason — RAM, a bad setting, a stuck model. |
| Killed with no message at all | Out of memory: the web UI is probably still running alongside it. |

Chat ids, if they ever change: post anything in the channel and read it back with
`getUpdates` on a bot that has no webhook, or use the mpt bot's existing
`telegram_chats.json` mechanism.

## Run it as a service

So it comes back after a reboot without remembering to start it:

```bash
mkdir -p ~/.config/systemd/user
cp local/mpt-wan2gp-bot.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mpt-wan2gp-bot
systemctl --user status mpt-wan2gp-bot
journalctl --user -u mpt-wan2gp-bot -f      # follow the log
```

`loginctl enable-linger $USER` additionally keeps it running when you are not logged
in. Without that it starts at login and stops at logout.

## Settings

Defaults suit this machine; override any of them in `.env`.

| Var | Default | |
| --- | --- | --- |
| `LOCAL_VIDEO_FRAMES` | `81` | ~5s at Wan2GP's 16fps |
| `LOCAL_VIDEO_STEPS` | `8` | the Lightning checkpoint's own default is 4 |
| `LOCAL_RESOLUTION` | from the saved UI settings | e.g. `832x480`, `512x896` |
| `WAN2GP_PROFILE` | `4` | memory profile; `5` is the low-RAM failsafe |
| `WAN2GP_ATTENTION` | `sdpa` | no nvcc here, so no SageAttention |
| `WAN2GP_TIMEOUT` | `5400` | seconds before a stuck generation is abandoned |
| `LOCAL_TIMELINE` | `1` | `0` sends the prompt verbatim |
| `LOCAL_KEEP_OUTPUTS` | `0` | `1` also leaves each clip in Wan2GP's `outputs/` |
| `LOCAL_VERBOSE` | `0` | `1` prints wgp.py's own per-step progress |

Everything else — guidance phases, flow shift, switch threshold, LoRAs — comes from
`~/apps/Wan2GP/settings/i2v_2_2_Enhanced_Lightning_v2_settings.json`, the file the
Wan2GP UI saves. Tune it in the UI and the bot picks the same values up on its next
job, so a clip from Telegram matches one made by hand.

## Known limits

- **Telegram is not end-to-end encrypted** for channels. Both channels are private (no
  public username, invite link only), so only members see them — but Telegram's servers
  hold the photo and the video and can technically read them. That is inherent to using
  Telegram as the interface; nothing in this program changes it.
- The web UI has to be stopped while the bot runs, and vice versa (see above).
- Only while this computer is on and awake. Photos sent while it is off are picked up
  on the next start, not lost.
- Generation is minutes, not seconds. The bot answers when the job is queued, and again
  when it is done.
