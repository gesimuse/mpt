/** Telegram Bot API calls the Worker makes back at the user. */

/**
 * Strip the bot token out of anything logged or shown to a user.
 *
 * Telegram has no header auth, so the token is a path segment of every request URL --
 * and getFileUrl returns a URL containing it. A fetch failure, or any error string
 * built from one of those URLs, therefore carries the credential into Worker logs and
 * into messages posted back into the channel. Redact at the boundary; the call sites
 * that would forget are exactly the ones that leak.
 */
export function redact(env, value) {
  let text = String(value);
  const token = env.TELEGRAM_BOT_TOKEN;
  if (token) {
    text = text.split(token).join("<TELEGRAM_BOT_TOKEN>");
    const secret = token.includes(":") ? token.slice(token.indexOf(":") + 1) : token;
    if (secret.length > 8) text = text.split(secret).join("<TELEGRAM_BOT_TOKEN>");
  }
  return text;
}

export function api(env, method, payload) {
  return fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/**
 * Every callback_query MUST be answered, even on failure. Telegram shows the button
 * spinning until it is, so an unanswered press looks like the bot is hung.
 */
export function answer(env, id, text, alert = false) {
  return api(env, "answerCallbackQuery", {
    callback_query_id: id, text: (text || "").slice(0, 200), show_alert: alert,
  });
}

export function deleteMessage(env, chatId, messageId) {
  return api(env, "deleteMessage", { chat_id: chatId, message_id: messageId });
}

export async function getFileUrl(env, fileId) {
  const r = await api(env, "getFile", { file_id: fileId });
  const body = await r.json();
  if (!body.ok) throw new Error(`getFile failed: ${JSON.stringify(body).slice(0, 200)}`);
  return `https://api.telegram.org/file/bot${env.TELEGRAM_BOT_TOKEN}/${body.result.file_path}`;
}
