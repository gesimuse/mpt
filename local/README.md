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
   The chat ids are reused from the existing `TELEGRAM_CHAT_ID` / `TELEGRAM_VIDEO_CHAT_ID`,
   so there is nothing else to configure.
4. **Run it:**
   ```bash
   python local/wan2gp_bot.py
   ```

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
| `WAN2GP_TIMEOUT` | `5400` | seconds before a stuck generation is killed |
| `LOCAL_TIMELINE` | `1` | `0` sends the prompt verbatim |

Everything else — guidance phases, flow shift, switch threshold, LoRAs — comes from
`~/apps/Wan2GP/settings/i2v_2_2_Enhanced_Lightning_v2_settings.json`, the file the
Wan2GP UI saves. Tune it in the UI and the bot picks the same values up on its next
job, so a clip from Telegram matches one made by hand.

## Known limits

- **Telegram is not end-to-end encrypted** for channels. Both channels are private (no
  public username, invite link only), so only members see them — but Telegram's servers
  hold the photo and the video and can technically read them. That is inherent to using
  Telegram as the interface; nothing in this program changes it.
- Only while this computer is on and awake. Photos sent while it is off are picked up
  on the next start, not lost.
- Generation is minutes, not seconds. The bot answers when the job is queued, and again
  when it is done.
