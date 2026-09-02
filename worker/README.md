# Telegram callback worker

Receives Telegram button presses and acts on them within seconds. This half exists
because the rest of the project is cron-driven: `telegram.py` (in a GitHub Actions job)
*sends* the images, but nothing in Actions is running when a button is pressed.

What it handles:

| Button / action | What happens |
| --- | --- |
| 🎬 Make video | Dispatches `autopilot_video.yml` with that image's URL and motion prompt, then edits the message to show it was sent |
| 🗑 Skip | Deletes the message from the channel **and** removes the image from `posted.json`, so the pipeline never reuses it |
| 🔄 Retry post | Dispatches the video workflow with `retry_video_url`, republishing the same mp4 without regenerating |
| Reply to a photo | Uses your text as the motion prompt for that image, then makes the video |
| Send a photo | Hosts it on `gh-pages` and registers it in `posted.json`, so it can be animated like any generated image |

An image that is neither skipped nor sent stays in the channel. The channel is the
backlog; nothing expires it.

## Why a Worker and not a polling workflow

GitHub's minimum schedule interval is 5 minutes, and this repo has *measured* GitHub
dropping scheduled runs under load — the `05:00` and `07:00` crons never fired at all
for days. A review surface whose buttons might respond in five minutes, or might not
respond until tomorrow, is not usable.

## Deploy

Everything below is wrapped in `./deploy.sh`, which reads the repo's `.env`, pushes the
secrets, deploys, registers the webhook and verifies it. It refuses to run if
`TELEGRAM_CHAT_ID` is an invite-link hash rather than a numeric channel id — that
mistake otherwise surfaces only later, as "chat not found" on every call.

The one thing it cannot do for you is authenticate to Cloudflare:

```sh
npx wrangler login          # browser OAuth, once
./deploy.sh
```

<details><summary>Manual equivalent</summary>

```sh
cd worker
npm install -g wrangler          # once
wrangler login                   # once
wrangler secret put TELEGRAM_BOT_TOKEN
wrangler secret put TELEGRAM_WEBHOOK_SECRET   # any long random string you invent
wrangler secret put GITHUB_TOKEN              # fine-grained PAT, this repo only:
                                              #   Actions: read+write, Contents: read+write
wrangler deploy
```

Then point Telegram at it (replace both placeholders):

```sh
curl -X POST "https://api.telegram.org/bot<BOT_TOKEN>/setWebhook" \
  -H 'content-type: application/json' \
  -d '{"url":"https://mpt-telegram.<your-subdomain>.workers.dev",
       "secret_token":"<TELEGRAM_WEBHOOK_SECRET>",
       "allowed_updates":["callback_query","message","channel_post"]}'
```

Check it took:

```sh
curl "https://api.telegram.org/bot<BOT_TOKEN>/getWebhookInfo"
```

`pending_update_count` climbing with a `last_error_message` means the Worker is
rejecting or erroring — `wrangler tail` shows why.

</details>

## Security

`secret_token` is not optional. The Worker URL is public, and without it anyone who
finds the URL can forge button presses that dispatch workflows and rewrite
`posted.json`. Telegram sends it back in `X-Telegram-Bot-Api-Secret-Token` on every
request; the Worker rejects anything that doesn't match, before parsing the body.

`ALLOWED_CHAT_ID` in `wrangler.toml` is the second gate: even a correctly-signed
update is ignored unless it came from your own channel.
