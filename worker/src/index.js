/**
 * Telegram webhook -> GitHub actions.
 *
 * See ../README.md for the button table and deploy steps. The short version: this
 * project is entirely cron-driven, so nothing is running when a button is pressed.
 * This Worker is the only always-on piece, and it exists solely to make the buttons
 * respond in seconds rather than whenever the next scheduled run happens to fire.
 */
import { dispatchWorkflow, mutatePostedJson, hostOnPages, recordUnknownChat } from "./github.js";
import { answer, deleteMessage, getFileUrl, api, redact } from "./telegram.js";

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

/**
 * Re-host a photo from Telegram's own copy back onto gh-pages, and point posted.json
 * at the new file.
 *
 * The problem: tiktok.KEEP_MEDIA prunes gh-pages oldest-first, but a photo can sit in
 * the channel for days waiting to be worked. Telegram keeps its own copy, so the
 * MESSAGE never breaks -- but Make video hands the URL to an HF Space that has to
 * fetch it, so a pruned file looks fine right up until someone presses the button.
 * Half the backlog was already in that state when it was migrated.
 *
 * Mirroring the prune into the channel was the alternative and is worse: it would
 * delete photos nobody had got to yet. Restoring the file keeps the backlog.
 *
 * Telegram's own file URL is deliberately NOT handed to the Space: it embeds the bot
 * token, and the URL becomes a workflow input that lands in Actions logs.
 *
 * Quality is not a concern here: Telegram keeps photos at up to ~1280px and the Space
 * downscales to 832x480 anyway, so its copy is comfortably larger than what is used.
 */
async function rehostFromTelegram(env, message, ts, index) {
  const photo = message?.photo;
  if (!photo?.length) return null;
  // photo[] is ascending by resolution; the last is the largest Telegram kept.
  const fileUrl = await getFileUrl(env, photo[photo.length - 1].file_id);
  const bytes = new Uint8Array(await (await fetch(fileUrl)).arrayBuffer());
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  const path = `media/rehosted-${Date.now()}.jpg`;
  const url = await hostOnPages(env, path, btoa(bin));
  // Kept: this is the one line that says a backlog image was rescued rather than lost,
  // and it is rare enough to be worth finding in a tail.
  console.log("restored a pruned image from Telegram ->", url);
  await mutatePostedJson(env, (state) => {
    const { entry } = findImage(state, ts, index);
    if (!entry?.image_urls || index >= entry.image_urls.length) return false;
    entry.image_urls[index] = url;
  }, "telegram: re-host a pruned image from the channel's own copy");
  return url;
}

async function isLive(url) {
  try {
    const r = await fetch(url, { method: "HEAD", redirect: "manual" });
    return r.status === 200;
  } catch {
    return false;
  }
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
    await answer(env, cq.id, found.entry
      ? "Already marked done or skipped - nothing to generate from."
      : "That batch is no longer in posted.json.", true);
    return;
  }
  if (!(await isLive(url))) {
    // Pruned from gh-pages while it waited in the backlog. Put it back from the copy
    // this very message is holding, rather than failing or dropping it.
    await answer(env, cq.id, "Image had expired - restoring it first...");
    try {
      url = await rehostFromTelegram(env, cq.message, ts, index);
    } catch (err) {
      await answer(env, cq.id, `Could not restore it: ${redact(env, err).slice(0, 120)}`, true);
      return;
    }
    if (!url) {
      await answer(env, cq.id, "That image's file is gone and cannot be restored.", true);
      return;
    }
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
    // TOMBSTONE, never splice. Every message in the channel carries a fixed index in
    // its callback_data, so removing an element renumbers all the later ones: the last
    // message's index falls off the end ("no longer in posted.json") and the ones in
    // between silently point at the WRONG image, which is worse than the error. A
    // batch of ten photos only had to have one resolved for the rest to be wrong.
    //
    // Consumers already tolerate this: autopilot's source picker tests `if url and
    // url not in used`, and the media prune walks gh-pages files rather than state.
    entry.image_urls[i] = null;
    if (entry.image_prompts) entry.image_prompts[i] = null;
    if (entry.motion_prompts) entry.motion_prompts[i] = null;
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
    await answer(env, cq.id, `Failed: ${redact(env, err).slice(0, 150)}`, true);
  }
}

/** A reply to a photo message = "use my text as the motion prompt for that image". */
async function onReply(env, msg) {
  const target = msg.reply_to_message;
  // Any button on the message will do: the photo keyboard's first row is Make video,
  // but reading only [0][0] would break the moment the layout changes again.
  const data = (target?.reply_markup?.inline_keyboard || [])
    .flat().map((b) => b.callback_data).find(Boolean);
  if (!data) {
    // Never silent. A reply to something with no buttons is a user asking for
    // something and getting nothing back, which is indistinguishable from broken.
    await api(env, "sendMessage", {
      chat_id: msg.chat.id,
      reply_to_message_id: msg.message_id,
      text: "Reply to one of the photo posts to use your text as its motion prompt.",
    });
    return;
  }
  const { ts, index } = parseCallback(data);
  const fake = { id: null, message: target };
  // answerCallbackQuery needs a real id; there is none here, so acknowledge in chat.
  await onMakeVideoFromReply(env, fake, ts, index, msg.text, msg.chat.id);
}

async function onMakeVideoFromReply(env, fake, ts, index, prompt, chatId) {
  try {
    let url = null, hadEntry = false;
    await mutatePostedJson(env, (state) => {
      const found = findImage(state, ts, index);
      url = found.url;
      hadEntry = Boolean(found.entry);
      return false;
    }, "");
    if (!url) {
      await api(env, "sendMessage", { chat_id: chatId, text: hadEntry
        ? "Already marked done or skipped - press nothing, it is finished."
        : "That batch is no longer in posted.json." });
      return;
    }
    if (!(await isLive(url))) {
      url = await rehostFromTelegram(env, fake.message, ts, index);
      if (!url) {
        await api(env, "sendMessage", { chat_id: chatId,
          text: "That image's file is gone and cannot be restored." });
        return;
      }
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
      { chat_id: chatId, text: `Failed: ${redact(env, err).slice(0, 300)}` });
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
      { chat_id: msg.chat.id, text: `Upload failed: ${redact(env, err).slice(0, 300)}` });
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
      // Setting up a new channel is otherwise circular: with a webhook active,
      // getUpdates returns nothing (Telegram delivers here instead), and this Worker
      // acks the update, so the chat id is unrecoverable afterwards. Replying with it
      // makes the channel announce itself. Only ever reaches chats the bot is already
      // a member of, and only after the webhook secret has been verified above.
      console.log("update from non-allowed chat", chatId);
      if (chatId) {
        // Written to the repo so the id survives being acked -- see recordUnknownChat.
        const title = update.channel_post?.chat?.title
          ?? update.message?.chat?.title ?? null;
        let isNew = true;
        try { isNew = await recordUnknownChat(env, chatId, title); }
        catch (e) {
          console.error("could not record chat id", redact(env, e));
          // Could not tell whether this is new. Stay quiet: an un-acked id is a
          // nuisance, a reply under every message forever is worse.
          isNew = false;
        }
        // Announce the id ONCE per chat, not on every message. This bot is a member
        // of a channel belonging to a different project, and the hint fired on every
        // post there -- a reply under each card, forever, in a channel that will
        // never be added. recordUnknownChat returns false when the id was already
        // recorded, which is exactly "we have said this before".
        if (isNew) {
          await api(env, "sendMessage", {
            chat_id: chatId,
            text: `This chat's id is ${chatId}.\nIt is not in ALLOWED_CHAT_ID yet, so `
                + `buttons here will be ignored until it is added.`,
          });
        }
      }
      return new Response("ok");
    }
    try {
      // A CHANNEL delivers everything as channel_post; only private chats and groups
      // use `message`. Handling just `message` meant every reply typed in the photos
      // channel -- the documented way to set a different motion prompt -- was dropped
      // without a word. Normalise the two before dispatching.
      const post = update.message ?? update.channel_post;
      if (update.callback_query) await onCallback(env, update.callback_query);
      else if (post?.reply_to_message && post.text) await onReply(env, post);
      else if (post?.photo) await onPhoto(env, post);
    } catch (err) {
      console.error("update failed", redact(env, err));
    }
    // Always 200: a non-2xx makes Telegram redeliver the same update indefinitely,
    // which for a button that dispatches a workflow means dispatching it repeatedly.
    return new Response("ok");
  },
};
