/**
 * Persona studio, run entirely from this Worker. Kaggle does the scheduled GPU work
 * (the daily render); on-demand jobs run on the laptop while it is open
 * (persona/agent.py), on Kaggle otherwise. GitHub only holds the code.
 *
 *   POST /persona               the persona bot's webhook: 👍 👎 🎬 🔁, replies, and 🎭
 *                               (a TikTok link or a video posted in the channel)
 *   GET  /persona/file          signed, short-lived photo links for Kaggle video jobs
 *   POST /persona/admin/*       the laptop uploads her bible / migrates state, and
 *                               claims and returns on-demand jobs (laptop/claim|result)
 *   scheduled() every 15 min    02:30-03:15 UTC render, 06/09/12/15/18 post slots,
 *                               and always: move the on-demand Kaggle job along
 *
 * Why not GitHub Actions: its cron fired hours late or not at all, and its runs
 * raced on shared state. Cloudflare's cron is punctual and KV has one writer.
 *
 * Secrets: PERSONA_BOT_TOKEN, PERSONA_WEBHOOK_SECRET, PERSONA_ADMIN_SECRET,
 * KAGGLE_API_TOKEN. Vars: PERSONA_CHAT_ID, PERSONA_SLUG, KAGGLE_USERNAME.
 * Bindings: PERSONA (KV), AI (Workers AI).
 */
import * as kaggle from "./kaggle.js";
import * as st from "./state.js";
import * as scenes from "./scenes.js";
import { tg, answer, sendMedia, kb } from "./telegram.js";

const POST_HOURS = [6, 9, 12, 15, 18];
const ROLES = { cast: "qwen21", edit: "qwen21", fanvue_edit: "qwen21-uncensored", video: "wan22-i2v-calm" };

const slugOf = (env) => env.PERSONA_SLUG || "lena";
const renderKernel = (env) => `${env.KAGGLE_USERNAME}/mpt-persona-render-${slugOf(env)}`;
const jobKernel = (env) => `${env.KAGGLE_USERNAME}/mpt-persona-video-${slugOf(env)}`;
const dataset = (env) => `${env.KAGGLE_USERNAME}/mpt-persona-${slugOf(env)}`;

function basePayload(env) {
  // 0.62: her scenes score 0.80-0.90; a bowling shot at 0.55 no longer looked like her.
  return { slug: slugOf(env), roles: ROLES, face_min: env.PERSONA_FACE_MIN || "0.62", min_age: "21" };
}

// ------------------------------------------------------------------ captions
function caption(name, item) {
  const lines = [`📸 ${name}${item.kind === "video" ? " · 🎬" : ""}`];
  if (item.caption) lines.push(`📝 ${item.caption}`);
  if (item.scene?.setting) lines.push(`🎞 ${item.scene.setting}`);
  if (item.check?.sim !== undefined) lines.push(`face ${item.check.sim} · age≈${item.check.age}`);
  if (item.kind !== "video") lines.push("🎬 video (or reply with how she moves) · 🔁 redraw");
  lines.push(`#${item.id}`);
  return lines.join("\n");
}

function buttons(item) {
  const row = [["👍", `pv:up:${item.id}`], ["👎", `pv:dn:${item.id}`]];
  if (item.kind !== "video") row.push(["🎬 Video", `pv:vid:${item.id}`], ["🔁 Redo", `pv:redo:${item.id}`]);
  return kb([row]);
}

function remember(state, item) {
  state.items[item.id] = { kind: item.kind || "image", tags: item.tags || [], beat: item.scene?.beat || "",
                           scene: item.scene || {}, caption: item.caption || "" };
}

function itemIdFrom(msg) {
  const lines = (msg?.caption || "").trim().split("\n");
  const last = lines[lines.length - 1] || "";
  return last.startsWith("#") ? last.slice(1).trim() : "";
}

// ---------------------------------------------------------------- signed links
async function sign(env, text) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(env.PERSONA_ADMIN_SECRET),
                                            { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(text));
  return [...new Uint8Array(mac)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fileLink(env, origin, fileId) {
  const exp = Math.floor(Date.now() / 1000) + 6 * 3600;
  return `${origin}/persona/file?f=${encodeURIComponent(fileId)}&e=${exp}&s=${await sign(env, `${fileId}.${exp}`)}`;
}

async function serveFile(request, env) {
  const u = new URL(request.url);
  const f = u.searchParams.get("f"), e = u.searchParams.get("e"), s = u.searchParams.get("s");
  if (!f || !e || Number(e) < Date.now() / 1000 || s !== await sign(env, `${f}.${e}`)) {
    return new Response("forbidden", { status: 403 });
  }
  const info = await tg(env, "getFile", { file_id: f });
  return fetch(`https://api.telegram.org/file/bot${env.PERSONA_BOT_TOKEN}/${info.file_path}`);
}

// ------------------------------------------------------------------- taps
async function onTap(env, cq, state, origin) {
  const [, act, itemId] = cq.data.split(":");
  const msg = cq.message || {};
  const chat = msg.chat?.id;
  if (act === "up" || act === "dn") {
    if (state.voted.includes(itemId)) return answer(env, cq.id, "Already voted.");
    const item = state.items[itemId] || { tags: [] };
    const d = act === "up" ? 1 : -1;
    for (const t of item.tags || []) state.votes.tags[t] = (state.votes.tags[t] || 0) + d;
    if (act === "up" && item.beat) state.story.push({ t: Date.now(), beat: item.beat, item: itemId });
    state.voted.push(itemId);
    await answer(env, cq.id, act === "up" ? "Noted 👍" : "Noted 👎");
    const label = act === "up" ? "✓ 👍" : "✓ 👎";
    try {
      await tg(env, "editMessageReplyMarkup", { chat_id: chat, message_id: msg.message_id,
        reply_markup: kb([[[label, cq.data], ["🎬 Video", `pv:vid:${itemId}`], ["🔁 Redo", `pv:redo:${itemId}`]]]) });
    } catch { /* unchanged markup */ }
    return;
  }
  if (act === "vid") {
    const photo = (msg.photo || []).slice(-1)[0];
    if (!photo) return answer(env, cq.id, "That is not a photo.");
    queueVideo(state, itemId, msg, photo.file_id, "");
    return answer(env, cq.id, "🎬 Queued. The video comes as a reply here in ~40-60 min.");
  }
  if (act === "redo") {
    const scene = state.items[itemId]?.scene;
    if (!scene?.setting) return answer(env, cq.id, "Can't redo this one: its scene isn't known.");
    if (state.queue.some((q) => q.kind === "redo" && q.item === itemId)) return answer(env, cq.id, "Already queued.");
    state.queue.push({ id: `${itemId}-r${Date.now()}`, kind: "redo", item: itemId, scene,
                       chat, message_id: msg.message_id });
    return answer(env, cq.id, "🔁 Queued. The redrawn photo comes as a reply here in ~30-50 min.");
  }
  return answer(env, cq.id, "");
}

// TikTok, or a Facebook reel/video (facebook.com/reel/…, /share/r/…, /watch, fb.watch/…).
const CLIP_LINK = /https?:\/\/(?:(?:www\.|vm\.|vt\.|m\.)?tiktok\.com|(?:www\.|m\.|web\.)?facebook\.com|fb\.watch)\/\S+/i;

/**
 * 🎭 A TikTok or Facebook reel link, or a video uploaded into the channel: recreate that clip with
 * her (persona/recreate.py). "8s" or "first 8" in the text sets the length.
 */
async function onRecreateRequest(env, post, state) {
  const text = post.text || post.caption || "";
  const link = (CLIP_LINK.exec(text) || [])[0];
  const video = post.video || (post.document?.mime_type?.startsWith("video/") ? post.document : null);
  if (!link && !video) return false;
  const secs = /(?:first\s*)?(\d{1,2})\s*(?:s|sec|seconds?)\b/i.exec(text) || /first\s+(\d{1,2})/i.exec(text);
  const seconds = Math.min(Number(secs?.[1] || env.PERSONA_RECREATE_MAX || 10), Number(env.PERSONA_RECREATE_MAX || 10));
  state.queue.push({ id: `rec-${post.message_id}-${Date.now()}`, kind: "recreate", item: `rec-${post.message_id}`,
                     url: link || "", file_id: video?.file_id || "", seconds,
                     chat: post.chat.id, message_id: post.message_id });
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: `🎭 Queued: her version of this clip (first ${seconds}s). It comes as a reply here in ~40-70 min.` });
  return true;
}

function queueVideo(state, itemId, msg, fileId, motion) {
  const existing = state.queue.find((q) => q.kind !== "redo" && q.item === itemId);
  if (existing) {
    if (motion) existing.motion = motion;
    return;
  }
  state.queue.push({ id: `${itemId}-${Date.now()}`, kind: "video", item: itemId, file_id: fileId,
                     motion: motion || state.items[itemId]?.scene?.motion || "",
                     chat: msg.chat.id, message_id: msg.message_id });
}

async function onReply(env, post, state) {
  const parent = post.reply_to_message;
  const itemId = itemIdFrom(parent);
  const photo = (parent.photo || []).slice(-1)[0];
  if (!itemId || !photo) return;
  queueVideo(state, itemId, parent, photo.file_id, post.text.trim());
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: "🎬 Queued with your motion prompt. The video comes as a reply here in ~40-60 min." });
}

export async function onPersonaWebhook(request, env) {
  const u = new URL(request.url);
  if (u.pathname === "/persona/file") return serveFile(request, env);
  if (u.pathname.startsWith("/persona/admin/")) return onAdmin(request, env);
  if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.PERSONA_WEBHOOK_SECRET) {
    return new Response("forbidden", { status: 403 });
  }
  try {
    await handleUpdates(env, [await request.json()], u.origin);
  } catch (err) {
    console.error("persona update failed", String(err).replaceAll(env.PERSONA_BOT_TOKEN || "\u0000", "<TOKEN>"));
  }
  return new Response("ok");
}

/** Apply taps and replies (from the webhook, or polled when no webhook is set). */
async function handleUpdates(env, updates, origin) {
  const slug = slugOf(env);
  const state = await st.load(env, slug);
  let changed = false;
  for (const update of updates) {
    const cq = update.callback_query;
    const post = update.channel_post || update.message;
    const chatId = String(cq?.message?.chat?.id ?? post?.chat?.id ?? "");
    if (chatId !== String(env.PERSONA_CHAT_ID)) continue;
    if (cq?.data?.startsWith("pv:")) { await onTap(env, cq, state, origin); changed = true; }
    // A TikTok/Facebook link or video counts as 🎭 even when it is sent as a reply.
    else if (post && (await onRecreateRequest(env, post, state))) changed = true;
    else if (post?.text && post.reply_to_message && !post.text.startsWith("/")) {
      await onReply(env, post, state); changed = true;
    }
  }
  if (!changed) return;
  await st.save(env, slug, state);
  // Start the job right away if Kaggle is idle, instead of waiting for the cron.
  if (state.queue.length) await advanceJobs(env, origin);
}

/**
 * Until the persona bot has a webhook (PERSONA_POLL=1), its taps are collected
 * here every 15 minutes, so nothing depends on GitHub Actions in the meantime.
 */
async function poll(env, origin) {
  const updates = await tg(env, "getUpdates", { timeout: 0, allowed_updates: ["callback_query", "channel_post", "message"] });
  if (!updates.length) return "no updates";
  await tg(env, "getUpdates", { timeout: 0, offset: updates[updates.length - 1].update_id + 1 });
  await handleUpdates(env, updates, origin);
  return `polled ${updates.length}`;
}

// ------------------------------------------------------------------- admin
async function onAdmin(request, env) {
  if (request.headers.get("Authorization") !== `Bearer ${env.PERSONA_ADMIN_SECRET}`) {
    return new Response("forbidden", { status: 403 });
  }
  const u = new URL(request.url);
  const slug = u.searchParams.get("slug") || slugOf(env);
  if (u.pathname === "/persona/admin/bible" && request.method === "POST") {
    await env.PERSONA.put(`bible:${slug}`, await request.text());
    return new Response("ok");
  }
  if (u.pathname === "/persona/admin/state") {
    if (request.method === "POST") {
      const s = { ...structuredClone(st.EMPTY), ...(await request.json()) };
      await st.save(env, slug, s);
      return new Response("ok");
    }
    return new Response(JSON.stringify(await st.load(env, slug)), { headers: { "content-type": "application/json" } });
  }
  if (u.pathname === "/persona/admin/run" && request.method === "POST") {
    const what = u.searchParams.get("what");
    if (what === "render") return new Response(await render(env, u.searchParams.get("force") === "1"));
    if (what === "post") return new Response(await post(env, Number(u.searchParams.get("slot") || 4)));
    if (what === "jobs") return new Response(await advanceJobs(env, u.origin));
    if (what === "poll") return new Response(await poll(env, u.origin));
  }
  if (u.pathname === "/persona/admin/laptop/claim" && request.method === "POST") return laptopClaim(request, env, u.origin);
  if (u.pathname === "/persona/admin/laptop/result" && request.method === "POST") return laptopResult(request, env);
  return new Response("not found", { status: 404 });
}

// ------------------------------------------------------------------ render
async function render(env, force = false) {
  const slug = slugOf(env);
  const state = await st.load(env, slug);
  if (state.render_date === st.today() && !force) return "already rendered today";
  const b = await st.bible(env, slug);
  if (!b) return "no bible in KV -- run `python -m persona.cli sync` on the laptop";
  const n = Number(env.PERSONA_CLOUD_IMAGES || 10);
  const list = await scenes.write(env, b, state, n);
  state.recent.push(...list.map((s) => `${s.setting} / ${s.outfit} / ${s.action}`));
  state.render_date = st.today();
  await st.save(env, slug, state);
  await kaggle.push(env, renderKernel(env),
    kaggle.bootstrap({ ...basePayload(env), scenes: list, videos: 0, date: st.today() }), [dataset(env)]);
  return `render pushed: ${list.length} scenes`;
}

// -------------------------------------------------------------------- post
async function post(env, k) {
  const slug = slugOf(env);
  const files = await kaggle.outputs(env, renderKernel(env));
  if (!files["status.json"]) return "no render output";
  const status = await kaggle.fetchJson(files["status.json"]);
  if (!status.ok || status.date !== st.today()) return `render not ready (${status.stage} ${status.date})`;
  const state = await st.load(env, slug);
  if (state.posted.date !== st.today()) state.posted = { date: st.today(), ids: [] };
  const images = (status.items || []).filter((r) => r.ok && (r.kind || "image") === "image").map((r) => r.id);
  const due = images.filter((id, i) => i % POST_HOURS.length <= k);
  const b = (await st.bible(env, slug)) || { name: slug };
  let sent = 0;
  for (const id of due) {
    if (state.posted.ids.includes(id) || !files[`out/items/${id}.json`]) continue;
    const item = await kaggle.fetchJson(files[`out/items/${id}.json`]);
    const img = files[`out/items/${id}.jpg`];
    if (!img) continue;
    try {
      await sendMedia(env, "photo", env.PERSONA_CHAT_ID, img, { caption: caption(b.name, item), buttons: buttons(item) });
      remember(state, item);
      state.posted.ids.push(id);
      sent++;
    } catch (e) {
      console.error("post failed", id, String(e));
    }
  }
  await st.save(env, slug, state);
  return `slot ${k}: posted ${sent}`;
}

// ----------------------------------------------------- on-demand Kaggle jobs
/**
 * Two Kaggle job kernels, so they never block each other: "quick" for 🔁 and 🎬
 * (minutes), "recreate" for 🎭 (an hour+). Results only come back when a whole
 * run ends, and on 2026-10-08 a bowling redo sat for hours behind a recreate in
 * one shared run. The original single kernel ("video") is still read for
 * results until its last run is posted.
 */
const LANES = {
  quick: { kernel: (env) => `${env.KAGGLE_USERNAME}/mpt-persona-quick-${slugOf(env)}`, takes: (q) => q.kind !== "recreate", max: 3 },
  recreate: { kernel: (env) => `${env.KAGGLE_USERNAME}/mpt-persona-recreate-${slugOf(env)}`, takes: (q) => q.kind === "recreate", max: 1 },
};

/**
 * Post one finished job (from a Kaggle run or the laptop) under the message that
 * asked for it. `item` is the item JSON, `file` its photo/video (URL or Blob).
 */
async function postResult(env, state, b, r, item, file) {
  if (r.ok && r.kind === "image") {
    await sendMedia(env, "photo", r.chat, file, { caption: "🔁 " + caption(b.name, item), buttons: buttons(item), replyTo: r.message_id });
    remember(state, item);
  } else if (r.ok) {
    const cap = r.recreate ? `🎭 ${b.name} · recreated` : `🎬 ${b.name}\n${(r.motion || "").slice(0, 300)}`;
    const btns = r.recreate ? kb([[["👍", `pv:up:${r.id}`], ["👎", `pv:dn:${r.id}`]]]) : undefined;
    await sendMedia(env, "video", r.chat, file, { caption: cap, buttons: btns, replyTo: r.message_id });
  } else {
    await tg(env, "sendMessage", { chat_id: r.chat, reply_to_message_id: r.message_id,
      text: `❌ ${r.kind === "image" ? "Redraw" : r.recreate ? "Recreate" : "Video"} failed: ${String(r.reason).slice(0, 300)}` });
  }
}

async function postResults(env, state, b, kernel) {
  const files = await kaggle.outputs(env, kernel).catch(() => ({}));
  const status = files["status.json"] ? await kaggle.fetchJson(files["status.json"]).catch(() => ({})) : {};
  let changed = false;
  for (const r of status.items || []) {
    if (!r.request || state.done.includes(r.request) || !r.chat) continue;
    try {
      const item = r.ok ? await kaggle.fetchJson(files[`out/items/${r.id}.json`]) : null;
      const file = item ? files[`out/items/${r.kind === "image" ? `${r.id}.jpg` : item.path}`] : null;
      await postResult(env, state, b, r, item, file);
    } catch (e) {
      console.error("job result post failed", r.request, String(e));
      continue;
    }
    state.done.push(r.request);
    changed = true;
  }
  return changed;
}

// ------------------------------------------------------------------ laptop
/**
 * While the laptop is open, persona/agent.py checks in every 30s and runs the
 * on-demand jobs on its own GPU; Kaggle only gets them when it has gone quiet.
 * The check-in is written to KV at most every 4 minutes (KV's free tier allows
 * 1000 writes a day), so "quiet" means no check-in for LAPTOP_STALE_MS.
 */
const LAPTOP_STALE_MS = 10 * 60 * 1000;
const LAPTOP_WRITE_MS = 4 * 60 * 1000;
const laptopFresh = (state) => Date.now() - (state.laptop?.seen || 0) < LAPTOP_STALE_MS;

async function laptopClaim(request, env, origin) {
  const body = await request.json().catch(() => ({}));
  const slug = slugOf(env);
  let state = await st.load(env, slug);
  if (Date.now() - (state.laptop?.seen || 0) > LAPTOP_WRITE_MS) {
    state.laptop = { seen: Date.now() };
    await st.save(env, slug, state);
  }
  // Then taps (while the bot has no webhook), so a tap made a moment ago is in the
  // queue below -- and, the laptop being marked up, not pushed to Kaggle.
  if (env.PERSONA_POLL === "1" && !body.busy) {
    await poll(env, origin).catch((e) => console.error("poll", String(e)));
    state = await st.load(env, slug);
  }
  state.lanes = state.lanes || {};
  let changed = false;
  let job = null;
  if (!body.busy) {
    // A job it already holds comes back too: the agent restarted mid-job.
    let q = (state.lanes.laptop || [])[0];
    if (!q && state.queue.length) {
      q = state.queue.shift();
      state.lanes.laptop = [q];
      changed = true;
    }
    if (q) job = (await jobPayload(env, origin, [q]))[0];
  }
  if (changed) await st.save(env, slug, state);
  return Response.json({ job });
}

async function laptopResult(request, env) {
  const form = await request.formData();
  const r = JSON.parse(form.get("result"));
  const slug = slugOf(env);
  const state = await st.load(env, slug);
  if (state.done.includes(r.request)) return new Response("already posted");
  const b = (await st.bible(env, slug)) || { name: slug };
  const item = form.get("item") ? JSON.parse(await form.get("item").text()) : null;
  await postResult(env, state, b, r, item, form.get("file"));
  state.done.push(r.request);
  // Taken back from Kaggle's queue too, if it was handed over while the laptop
  // was quiet; a Kaggle result for it is skipped as already done.
  state.lanes.laptop = (state.lanes.laptop || []).filter((q) => q.id !== r.request);
  state.queue = state.queue.filter((q) => q.id !== r.request);
  state.laptop = { seen: Date.now() };
  await st.save(env, slug, state);
  return new Response("posted");
}

async function jobPayload(env, origin, batch) {
  const jobs = [];
  for (const q of batch) {
    if (q.kind === "redo") {
      jobs.push({ id: q.id, kind: "redo", scene: q.scene, chat: q.chat, message_id: q.message_id });
    } else if (q.kind === "recreate") {
      jobs.push({ id: q.id, kind: "recreate", url: q.url, seconds: q.seconds, chat: q.chat, message_id: q.message_id,
                  video_url: q.file_id ? await fileLink(env, origin, q.file_id) : "" });
    } else {
      jobs.push({ id: q.id, image_url: await fileLink(env, origin, q.file_id), motion: q.motion || "",
                  chat: q.chat, message_id: q.message_id });
    }
  }
  return jobs;
}

async function advanceJobs(env, origin) {
  const slug = slugOf(env);
  const state = await st.load(env, slug);
  const b = (await st.bible(env, slug)) || { name: slug };
  state.lanes = state.lanes || {};
  let changed = false;
  const notes = [];

  // The old single kernel: post whatever its last run produced, once.
  if ((state.running || []).length) {
    const s = await kaggle.status(env, jobKernel(env));
    if (s === "COMPLETE" || s === "ERROR") {
      changed = (await postResults(env, state, b, jobKernel(env))) || changed;
      state.running = [];
      changed = true;
    } else notes.push(`legacy ${s}`);
  }

  // The laptop went quiet holding a job (lid closed mid-render): Kaggle takes it.
  const laptopUp = laptopFresh(state);
  if ((state.lanes.laptop || []).length && !laptopUp) {
    state.queue = [...state.lanes.laptop, ...state.queue];
    state.lanes.laptop = [];
    changed = true;
    notes.push("laptop quiet: its job goes to Kaggle");
  }
  if (laptopUp) notes.push("laptop up: new jobs wait for it");

  for (const [name, lane] of Object.entries(LANES)) {
    const kernel = lane.kernel(env);
    const running = state.lanes[name] || [];
    if (running.length) {
      const s = await kaggle.status(env, kernel);
      if (s === "RUNNING" || s === "QUEUED") { notes.push(`${name} ${s}`); continue; }
      if (s === "COMPLETE" || s === "ERROR") {
        changed = (await postResults(env, state, b, kernel)) || changed;
        state.lanes[name] = [];
        changed = true;
      } else if (s.includes("No runs found")) {
        // Saved but never started (no GPU quota free at push time): put the jobs
        // back and push again below, instead of waiting on a run that won't come.
        state.queue = [...running, ...state.queue];
        state.lanes[name] = [];
        changed = true;
      } else { notes.push(`${name} ${s}`); continue; }
    }
    const batch = laptopUp ? [] : state.queue.filter(lane.takes).slice(0, lane.max);
    if (!batch.length) continue;
    const jobs = await jobPayload(env, origin, batch);
    await kaggle.push(env, kernel, kaggle.bootstrap({ ...basePayload(env), mode: "video", jobs }), [dataset(env)]);
    state.lanes[name] = batch;
    const taken = new Set(batch.map((q) => q.id));
    state.queue = state.queue.filter((q) => !taken.has(q.id));
    changed = true;
    notes.push(`${name}: pushed ${jobs.length}`);
  }
  if (changed) await st.save(env, slug, state);
  return notes.join("; ") || "idle";
}

// ---------------------------------------------------------------- schedule
export async function onPersonaSchedule(event, env) {
  if (!env.PERSONA_BOT_TOKEN || !env.KAGGLE_API_TOKEN) return;
  const now = new Date(event.scheduledTime);
  const h = now.getUTCHours(), m = now.getUTCMinutes();
  const origin = env.WORKER_ORIGIN;
  try {
    if ((h === 2 && m >= 30) || (h === 3 && m < 30)) console.log(await render(env));
    else if (POST_HOURS.includes(h) && m < 15) console.log(await post(env, POST_HOURS.indexOf(h)));
    if (env.PERSONA_POLL === "1") console.log(await poll(env, origin));
    console.log(await advanceJobs(env, origin));
  } catch (err) {
    console.error("persona schedule failed", String(err));
  }
}
