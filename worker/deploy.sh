#!/usr/bin/env bash
# One-shot deploy: pushes secrets, deploys the Worker, registers the Telegram webhook
# and verifies it. Reads everything from the repo's .env so there is a single source of
# truth and nothing has to be retyped into three places.
#
# The one prerequisite this cannot do for you: `wrangler login` (browser OAuth), or
# CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID exported in the environment.
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE="../.env"

[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE"; exit 1; }
get() { grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d "\"' "; }

BOT_TOKEN=$(get TELEGRAM_BOT_TOKEN)
CHAT_ID=$(get TELEGRAM_CHAT_ID)
WEBHOOK_SECRET=$(get TELEGRAM_WEBHOOK_SECRET)
GH_PAT=$(get GITHUB_TOKEN)

for pair in "TELEGRAM_BOT_TOKEN:$BOT_TOKEN" "TELEGRAM_CHAT_ID:$CHAT_ID" \
            "TELEGRAM_WEBHOOK_SECRET:$WEBHOOK_SECRET" "GITHUB_TOKEN:$GH_PAT"; do
  [ -n "${pair#*:}" ] || { echo "missing ${pair%%:*} in $ENV_FILE"; exit 1; }
done

# A channel id is numeric and starts with -100. An invite-link hash (+oQB8-aF...) is a
# different thing entirely and every API call made with it fails with "chat not found",
# so it is worth catching here rather than after a deploy.
case "$CHAT_ID" in
  -100*) : ;;
  *) echo "TELEGRAM_CHAT_ID is '$CHAT_ID', which is not a channel id."
     echo "It must be numeric and start with -100 (e.g. -1001234567890)."
     echo "Add the bot to the channel as an admin, post any message there, then:"
     echo "  curl -s \"https://api.telegram.org/bot<TOKEN>/getUpdates\""
     exit 1 ;;
esac

echo "==> writing ALLOWED_CHAT_ID into wrangler.toml"
sed -i.bak -E "s|^ALLOWED_CHAT_ID = .*|ALLOWED_CHAT_ID = \"$CHAT_ID\"|" wrangler.toml
rm -f wrangler.toml.bak

echo "==> pushing secrets to Cloudflare"
for name in TELEGRAM_BOT_TOKEN TELEGRAM_WEBHOOK_SECRET GITHUB_TOKEN; do
  case "$name" in
    TELEGRAM_BOT_TOKEN) val="$BOT_TOKEN" ;;
    TELEGRAM_WEBHOOK_SECRET) val="$WEBHOOK_SECRET" ;;
    GITHUB_TOKEN) val="$GH_PAT" ;;
  esac
  printf '%s' "$val" | npx --yes wrangler@latest secret put "$name" >/dev/null
  echo "    $name"
done

echo "==> deploying"
DEPLOY_OUT=$(npx --yes wrangler@latest deploy 2>&1)
echo "$DEPLOY_OUT" | tail -5
URL=$(echo "$DEPLOY_OUT" | grep -oE 'https://[a-z0-9.-]+\.workers\.dev' | head -1)
[ -n "$URL" ] || { echo "could not find the deployed URL in wrangler's output"; exit 1; }
echo "==> worker at $URL"

echo "==> registering the Telegram webhook"
# allowed_updates is explicit: without it Telegram sends every update type, and this
# Worker only ever acts on three of them.
curl -sS -X POST "https://api.telegram.org/bot$BOT_TOKEN/setWebhook" \
  -H 'content-type: application/json' \
  -d "{\"url\":\"$URL\",\"secret_token\":\"$WEBHOOK_SECRET\",\"allowed_updates\":[\"callback_query\",\"message\",\"channel_post\"]}"
echo

echo "==> verifying"
curl -sS "https://api.telegram.org/bot$BOT_TOKEN/getWebhookInfo"
echo
echo "Done. A non-empty last_error_message above means the Worker is erroring --"
echo "run 'npx wrangler tail' and press a button to see why."
