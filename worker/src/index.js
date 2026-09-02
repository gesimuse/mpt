/**
 * Telegram webhook -> GitHub actions.
 *
 * See ../README.md for the button table and deploy steps. The short version: this
 * project is entirely cron-driven, so nothing is running when a button is pressed.
 * This Worker is the only always-on piece, and it exists solely to make the buttons
 * respond in seconds rather than whenever the next scheduled run happens to fire.
 */
import { dispatchWorkflow, mutatePostedJson, hostOnPages } from "./github.js";
import { answer, deleteMessage, getFileUrl, api } from "./telegram.js";

/** callback_data is capped at 64 bytes, so buttons carry ids, not URLs. */
function parseCallback(data) {
  const [action, ts, index] = (data || "").split("|");
  return { action, ts, index: Number(index) };
}

/** The upload entry a button refers to, plus the specific image within it. */
function findImage(state, ts, index) {
  const entry = (state.uploads || []).find((u) => u.ts === ts);
  if (!entry) return {};
  const urls = entry.image_urls || [];
  return { entry, url: urls[index], urls };
}

async function onMakeVideo(env, cq, ts, index, promptOverride) {
  let url = null, prompt = promptOverride || null;
  // Read-only use of the mutate helper: returning false means no PUT is made.
  await mutatePostedJson(env, (state) => {
    const found = findImage(state, ts, index);
    url = found.url;
    if (!prompt) {
      prompt = (found.entry?.motion_prompts || [])[index]
        || (found.entry?.image_prompts || [])[index] || "";
    }
    return false;
  }, "");
  if (!url) {
    await answer(env, cq.id, "That image is no longer in posted.json.", true);
    return;
  }
  await dispatchWorkflow(env, {
    image_url: url,
    motion_prompt: prompt,
    length_s: env.VIDEO_LENGTH_S || "5.0",
    steps: env.VIDEO_STEPS || "4",
  });
  await answer(env, cq.id, "Sent to video generation.");
  // Deliberately NOT markDone: the image stays in the channel with its buttons, so a
  // different motion prompt can be tried on the same still. One photo is worth several
  // attempts, and an image that disappeared the moment it was used made that
  // impossible. Only Done and Skip remove it.
  await noteAttempt(env, cq.message, prompt);
}

/** Append a line recording this attempt, leaving the keyboard in place. */
async function noteAttempt(env, message, prompt) {
  const existing = message.caption || "";
  const line = `🎬 sent: ${prompt}`;
  if (existing.includes(line)) return;
  await api(env, "editMessageCaption", {
    chat_id: message.chat.id, message_id: message.message_id,
    caption: `${existing}\n${line}`.slice(0, 1024),
    reply_markup: message.reply_markup,
  });
}

/**
 * Done and Skip both remove the image; they differ only in the verdict recorded.
 * "posted" means it was used, "skipped" means it was not wanted -- and that is exactly
 * the signal imageslides._owner_theme_rates weights theme and subject choice by, so
 * pressing these keeps the pipeline learning what is worth generating.
 */
async function onResolve(env, cq, ts, index, verdict) {
  await mutatePostedJson(env, (state) => {
    const { entry, url } = findImage(state, ts, index);
    if (!entry || !url) return false;
    const i = entry.image_urls.indexOf(url);
    if (i === -1) return false;
    entry.image_urls.splice(i, 1);
    if (entry.image_prompts) entry.image_prompts.splice(i, 1);
    if (entry.motion_prompts) entry.motion_prompts.splice(i, 1);
    entry.owner_verdict = verdict;
    // The entry is KEPT even with no images left. It still carries vibe and look, and
    // _owner_theme_rates reads exactly those alongside owner_verdict -- deleting it
    // would throw away the feedback the button just produced.
  }, `telegram: ${verdict === "posted" ? "done with" : "skip"} image`);
  await answer(env, cq.id, verdict === "posted" ? "Done." : "Skipped.");
  await deleteMessage(env, cq.message.chat.id, cq.message.message_id);
}

async function onRetry(env, cq, ts) {
  let videoUrl = null;
  await mutatePostedJson(env, (state) => {
    videoUrl = (state.uploads || []).find((u) => u.ts === ts)?.video_url || null;
    return false;
  }, "");
  if (!videoUrl) {
    await answer(env, cq.id, "No hosted mp4 recorded for that entry.", true);
    return;
  }
  // retry_video_url republishes the existing file; it never regenerates, so a
  // TikTok-side rejection costs no ZeroGPU quota to retry.
  await dispatchWorkflow(env, { retry_video_url: videoUrl });
  await answer(env, cq.id, "Retrying the TikTok post.");
}

async function onCallback(env, cq) {
  const { action, ts, index } = parseCallback(cq.data);
  try {
    if (action === "vid") return await onMakeVideo(env, cq, ts, index);
    if (action === "done") return await onResolve(env, cq, ts, index, "posted");
    if (action === "skip") return await onResolve(env, cq, ts, index, "skipped");
    if (action === "retry") return await onRetry(env, cq, ts);
    await answer(env, cq.id, `Unknown action: ${action}`, true);
  } catch (err) {
    // Always answer, or the button spins forever and the bot looks hung.
    await answer(env, cq.id, `Failed: ${String(err).slice(0, 150)}`, true);
  }
}

/** A reply to a photo message = "use my text as the motion prompt for that image". */
async function onReply(env, msg) {
  const target = msg.reply_to_message;
  const data = target?.reply_markup?.inline_keyboard?.[0]?.[0]?.callback_data;
  if (!data) return;
  const { ts, index } = parseCallback(data);
  const fake = { id: null, message: target };
  // answerCallbackQuery needs a real id; there is none here, so acknowledge in chat.
  await onMakeVideoFromReply(env, fake, ts, index, msg.text, msg.chat.id);
}

async function onMakeVideoFromReply(env, fake, ts, index, prompt, chatId) {
  try {
    let url = null;
    await mutatePostedJson(env, (state) => {
      url = findImage(state, ts, index).url;
      return false;
    }, "");
    if (!url) {
      await api(env, "sendMessage",
        { chat_id: chatId, text: "That image is no longer in posted.json." });
      return;
    }
    await dispatchWorkflow(env, {
      image_url: url, motion_prompt: prompt,
      length_s: env.VIDEO_LENGTH_S || "5.0", steps: env.VIDEO_STEPS || "4",
    });
    // Same rule as the button: a reply makes a video and LEAVES the image, so the
    // next reply can try a different prompt on the same still.
    await noteAttempt(env, fake.message, prompt);
    await api(env, "sendMessage",
      { chat_id: chatId, text: `Sent to video generation:\n${prompt}` });
  } catch (err) {
    await api(env, "sendMessage",
      { chat_id: chatId, text: `Failed: ${String(err).slice(0, 300)}` });
  }
}

/** A photo sent to the bot gets hosted and registered, so it can be animated too. */
async function onPhoto(env, msg) {
  try {
    // photo[] is ascending by resolution; the last is the largest Telegram kept.
    const largest = msg.photo[msg.photo.length - 1];
    const fileUrl = await getFileUrl(env, largest.file_id);
    const bytes = new Uint8Array(await (await fetch(fileUrl)).arrayBuffer());
    let bin = "";
    for (const b of bytes) bin += String.fromCharCode(b);
    const name = `media/telegram-${Date.now()}.jpg`;
    const url = await hostOnPages(env, name, btoa(bin));
    const ts = new Date().toISOString().slice(0, 19);
    await mutatePostedJson(env, (state) => {
      // Same shape autopilot writes, so recentBatches/_pick_source_image_url treat it
      // as an ordinary photo with no special-casing.
      state.uploads.push({
        niche: "aibeauty", topic: `telegram upload ${Date.now()}`,
        title: "Uploaded via Telegram", tiktok: true,
        tiktok_via: "telegram_manual_upload",
        image_urls: [url], image_prompts: [null], motion_prompts: [null],
        vibe: null, look: null, ts,
      });
    }, "telegram: register uploaded image");
    await api(env, "sendPhoto", {
      chat_id: msg.chat.id, photo: url,
      caption: "Uploaded and registered.\n\nReply to this message to set a motion prompt.",
      reply_markup: { inline_keyboard: [
        [{ text: "🎬 Make video", callback_data: `vid|${ts}|0` }],
        [{ text: "✅ Done", callback_data: `done|${ts}|0` },
         { text: "🗑 Skip", callback_data: `skip|${ts}|0` }],
      ] },
    });
  } catch (err) {
    await api(env, "sendMessage",
      { chat_id: msg.chat.id, text: `Upload failed: ${String(err).slice(0, 300)}` });
  }
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("ok");
    // The Worker URL is public. Without this check anyone who finds it could dispatch
    // workflows and rewrite posted.json, so it is verified before the body is parsed.
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        !== env.TELEGRAM_WEBHOOK_SECRET) {
      return new Response("forbidden", { status: 403 });
    }
    const update = await request.json();
    const chatId = update.callback_query?.message?.chat?.id
      ?? update.message?.chat?.id ?? update.channel_post?.chat?.id;
    // Second gate: even a correctly-signed update from someone else's chat is ignored.
    // Comma-separated because photos and videos live in separate channels and buttons
    // (Retry) exist in both.
    const allowed = String(env.ALLOWED_CHAT_ID || "")
      .split(",").map((s) => s.trim()).filter(Boolean);
    if (allowed.length && !allowed.includes(String(chatId))) {
      return new Response("ok");
    }
    try {
      if (update.callback_query) await onCallback(env, update.callback_query);
      else if (update.message?.reply_to_message && update.message.text) {
        await onReply(env, update.message);
      } else if (update.message?.photo) await onPhoto(env, update.message);
    } catch (err) {
      console.error("update failed", err);
    }
    // Always 200: a non-2xx makes Telegram redeliver the same update indefinitely,
    // which for a button that dispatches a workflow means dispatching it repeatedly.
    return new Response("ok");
  },
};
