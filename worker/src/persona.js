/**
 * Persona studio (persona/ in this repo): its own bot, its own webhook path, and the
 * schedule that drives persona_kaggle.yml.
 *
 * Why this lives here: GitHub's own cron for that workflow fired hours late or not
 * at all (2026-10-07: nothing between 13:10 and 16:40 UTC, against a 15-minute
 * schedule), so a 🎬 tap could sit unseen for hours. Cloudflare's cron is punctual
 * and a webhook is instant.
 *
 *  - POST /persona   Telegram webhook of the persona bot (PERSONA_BOT_TOKEN). Each
 *                    button tap or reply is acknowledged at once and handed to the
 *                    workflow as `update` (action=votes), which applies it.
 *  - scheduled()     every 15 min: the daily render at 02:30 UTC, a post slot at
 *                    06/09/12/15/18 UTC, otherwise a votes run that moves the
 *                    on-demand video/redo kernel along.
 *
 * Secrets: PERSONA_BOT_TOKEN, PERSONA_WEBHOOK_SECRET. Vars: PERSONA_CHAT_ID,
 * PERSONA_WORKFLOW.
 */
import { dispatchWorkflow } from "./github.js";

const POST_HOURS = [6, 9, 12, 15, 18];

async function tg(env, method, params) {
  const r = await fetch(`https://api.telegram.org/bot${env.PERSONA_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  return r.json();
}

function itemIdFrom(msg) {
  const lines = (msg?.caption || "").trim().split("\n");
  const last = lines[lines.length - 1] || "";
  return last.startsWith("#") ? last.slice(1).trim() : "";
}

async function dispatch(env, inputs) {
  await dispatchWorkflow(env, inputs, env.PERSONA_WORKFLOW || "persona_kaggle.yml");
}

export async function onPersonaWebhook(request, env) {
  if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.PERSONA_WEBHOOK_SECRET) {
    return new Response("forbidden", { status: 403 });
  }
  const update = await request.json();
  const cq = update.callback_query;
  const post = update.channel_post || update.message;
  const chatId = String(cq?.message?.chat?.id ?? post?.chat?.id ?? "");
  if (chatId !== String(env.PERSONA_CHAT_ID)) return new Response("ok");
  try {
    if (cq?.data?.startsWith("pv:")) {
      const act = cq.data.split(":")[1];
      const note = act === "vid" ? "🎬 Queued. The video comes as a reply here in ~40-60 min."
        : act === "redo" ? "🔁 Queued. The redrawn photo comes as a reply here in ~30-50 min."
        : act === "up" ? "Noted 👍" : act === "dn" ? "Noted 👎" : "OK";
      await tg(env, "answerCallbackQuery", { callback_query_id: cq.id, text: note });
      await dispatch(env, { action: "votes", update: JSON.stringify(update) });
    } else if (post?.text && post.reply_to_message && !post.text.startsWith("/")
               && itemIdFrom(post.reply_to_message)) {
      await tg(env, "sendMessage", {
        chat_id: post.chat.id, reply_to_message_id: post.message_id,
        text: "🎬 Queued with your motion prompt. The video comes as a reply here in ~40-60 min.",
      });
      await dispatch(env, { action: "votes", update: JSON.stringify(update) });
    }
  } catch (err) {
    console.error("persona update failed", String(err).replaceAll(env.PERSONA_BOT_TOKEN || "\u0000", "<TOKEN>"));
  }
  // Always 200: Telegram redelivers on anything else.
  return new Response("ok");
}

export async function onPersonaSchedule(event, env) {
  if (!env.GITHUB_TOKEN) return;
  const now = new Date(event.scheduledTime);
  const h = now.getUTCHours(), m = now.getUTCMinutes();
  let inputs;
  // Render ticks 02:30-03:15: GitHub keeps one pending run per concurrency group, so a
  // tap landing at the same moment could replace the render; later ticks retry, and
  // the render step itself exits if today's batch was already started.
  if ((h === 2 && m >= 30) || (h === 3 && m < 30)) inputs = { action: "render" };
  else if (POST_HOURS.includes(h) && m < 15) inputs = { action: "post", slot: String(POST_HOURS.indexOf(h)) };
  else inputs = { action: "votes" };
  try {
    await dispatch(env, inputs);
  } catch (err) {
    console.error("persona schedule dispatch failed", String(err));
  }
}
