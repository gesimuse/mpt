/** The persona bot (PERSONA_BOT_TOKEN). */
export async function tg(env, method, params) {
  const r = await fetch(`https://api.telegram.org/bot${env.PERSONA_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  const body = await r.json();
  if (!body.ok) throw new Error(`telegram ${method}: ${JSON.stringify(body).slice(0, 200)}`);
  return body.result;
}

export async function answer(env, id, text) {
  try { await tg(env, "answerCallbackQuery", { callback_query_id: id, text: (text || "").slice(0, 200) }); }
  catch { /* too old to answer; the action still happened */ }
}

/** Upload a photo or video: a URL (Kaggle output) or a Blob (sent up by the laptop). */
export async function sendMedia(env, kind, chatId, src, { caption = "", buttons, replyTo } = {}) {
  let blob = src;
  if (typeof src === "string") {
    const file = await fetch(src);
    if (!file.ok) throw new Error(`download ${file.status}`);
    blob = new Blob([await file.arrayBuffer()]);
  }
  const form = new FormData();
  form.append("chat_id", String(chatId));
  form.append("caption", caption.slice(0, 1024));
  if (buttons) form.append("reply_markup", JSON.stringify(buttons));
  if (replyTo) form.append("reply_to_message_id", String(replyTo));
  if (kind === "video") form.append("supports_streaming", "true");
  form.append(kind, blob, kind === "video" ? "clip.mp4" : "photo.jpg");
  const r = await fetch(`https://api.telegram.org/bot${env.PERSONA_BOT_TOKEN}/send${kind === "video" ? "Video" : "Photo"}`,
    { method: "POST", body: form });
  const body = await r.json();
  if (!body.ok) throw new Error(`send ${kind}: ${JSON.stringify(body).slice(0, 200)}`);
  return body.result;
}

export function kb(rows) {
  return { inline_keyboard: rows.map((row) => row.map(([text, data]) => ({ text, callback_data: data }))) };
}
