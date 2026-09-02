/** Telegram Bot API calls the Worker makes back at the user. */

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
