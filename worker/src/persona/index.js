/**
 * Persona studio, run entirely from this Worker. Kaggle does the scheduled GPU work
 * (the daily render); on-demand jobs run on the laptop while it is open
 * (persona/agent.py), on Kaggle otherwise. GitHub only holds the code.
 *
 * Each persona has her own Telegram channel (personas.js); the one bot serves them
 * all. A finished photo or video there publishes to her Instagram and TikTok on its
 * own at the next slot (👎 before then cancels it; carousels and albums wait for a 👍
 * or "post"), through social.js, and the post in the channel is marked.
 *
 *   POST /persona               the persona bot's webhook: 👍 👎 🎬 🔁, replies, and 🎭
 *                               (a TikTok link or a video posted in the channel)
 *   GET  /persona/file          signed, short-lived photo links for Kaggle video jobs
 *   GET  /persona/media/…       signed media links Buffer fetches the posts from
 *   POST /persona/admin/*       the laptop uploads her bible / migrates state, links
 *                               channels and Buffer accounts, and claims and returns
 *                               on-demand jobs (laptop/claim|result)
 *   scheduled() every 15 min    02:30-03:15 UTC render, 06/09/12/15/18 post slots,
 *                               and always: move the on-demand jobs and publishing along
 *
 * Why not GitHub Actions: its cron fired hours late or not at all, and its runs
 * raced on shared state. Cloudflare's cron is punctual and KV has one writer.
 *
 * Secrets: PERSONA_BOT_TOKEN, PERSONA_WEBHOOK_SECRET, PERSONA_ADMIN_SECRET,
 * KAGGLE_API_TOKEN, BUFFER_ACCESS_TOKEN. Vars: PERSONA_CHAT_ID,
 * PERSONA_SLUG (the first persona), KAGGLE_USERNAME. Bindings: PERSONA (KV), AI.
 */
import * as kaggle from "./kaggle.js";
import * as st from "./state.js";
import * as scenes from "./scenes.js";
import * as personas from "./personas.js";
import * as social from "./social.js";
import * as captions from "./captions.js";
import * as tiktok from "./tiktok.js";
import * as instagram from "./instagram.js";
import { tg, answer, sendMedia, editPhoto, kb } from "./telegram.js";

const POST_HOURS = [6, 9, 12, 15, 18];
const ROLES = { cast: "qwen21", edit: "qwen21", fanvue_edit: "qwen21-uncensored", video: "wan22-i2v-calm" };

const renderKernel = (env, p) => `${env.KAGGLE_USERNAME}/mpt-persona-render-${p.slug}`;
const jobKernel = (env, p) => `${env.KAGGLE_USERNAME}/mpt-persona-video-${p.slug}`;
const dataset = (env, slug) => `${env.KAGGLE_USERNAME}/mpt-persona-${slug}`;

/**
 * A roster persona (MPT) is a channel shared by several characters ({characters:
 * [slugs]}): each shoot, photo and job belongs to one of them (q.character /
 * item.character), and her own Kaggle dataset and bible are used for it.
 */
const isRoster = (p) => Array.isArray(p.characters) && p.characters.length > 0;
// Her content comes as shoots (one place, one outfit, an album a day): every roster,
// and a single persona whose entry says shoots:true (Lena, 2026-10-23: "the same
// thing we do on mpt with the carousels").
const byShoots = (p) => isRoster(p) || p.shoots === true;
const charOf = (p, q) => q?.character || p.slug;

function basePayload(env, p, slug) {
  // 0.62: her scenes score 0.80-0.90; a bowling shot at 0.55 no longer looked like her.
  return { slug: slug || p.slug, roles: ROLES, face_min: env.PERSONA_FACE_MIN || "0.62", min_age: "21" };
}

// ------------------------------------------------------------------ captions
function caption(name, item) {
  const lines = [`📸 ${name}${item.kind === "video" ? " · 🎬" : ""}`];
  if (item.caption) lines.push(`📝 ${item.caption}`);
  if (item.scene?.setting) lines.push(`🎞 ${item.scene.setting}`);
  if (item.check?.sim !== undefined) lines.push(`face ${item.check.sim} · age≈${item.check.age}`);
  if (item.kind !== "video") lines.push("reply: redo <note> · video <how she moves> · carousel · post");
  lines.push(`#${item.id}`);
  return lines.join("\n");
}

function buttons(item) {
  const row = [["👍", `pv:up:${item.id}`], ["👎", `pv:dn:${item.id}`]];
  if (item.kind !== "video") row.push(["🎬 Video", `pv:vid:${item.id}`], ["🔁 Redo", `pv:redo:${item.id}`]);
  const rows = [row];
  if (item.kind !== "video") rows.push([["🖼 Carousel (+2 angles)", `pv:car:${item.id}`]]);
  return kb(rows);
}

/** `sent`: the Telegram message it went out as -- its file is kept, because a tap on a
 * message older than ~48h no longer carries the photo/video, and 👍 still has to post it. */
function remember(state, item, sent) {
  const media = sent?.video || (sent?.photo || []).slice(-1)[0];
  state.items[item.id] = { kind: item.kind || "image", tags: item.tags || [], beat: item.scene?.beat || "",
                           scene: item.scene || {}, caption: item.caption || "",
                           audio: !!item.audio, recreate: !!item.recreate,
                           ...(item.parent ? { parent: item.parent } : {}), ...(item.source ? { source: item.source } : {}),
                           file_id: media?.file_id || state.items[item.id]?.file_id || "",
                           ...(sent?.message_id ? { message_id: sent.message_id } : state.items[item.id]?.message_id ? { message_id: state.items[item.id].message_id } : {}),
                           character: item.character || state.items[item.id]?.character || undefined };
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

/** For Buffer to fetch: typed by extension (Instagram wants a real image/jpeg), 24h. */
async function mediaLink(env, origin, fileId, ext) {
  // 45 days: Buffer fetches the media when its queue slot comes, which can be days
  // after 👍 -- with 24h the link was dead by then ("Please update the media URL to be
  // publicly accessible", 2026-10-20). Telegram file ids don't expire.
  const exp = Math.floor(Date.now() / 1000) + 45 * 86400;
  return `${origin}/persona/media/${encodeURIComponent(fileId)}.${ext}?e=${exp}&s=${await sign(env, `${fileId}.${exp}`)}`;
}

/** A week-long link to the share page of one publish job (see sharePage). */
async function shareLink(env, origin, p, job) {
  const exp = Math.floor(Date.now() / 1000) + 7 * 86400;
  return `${origin}/persona/share?slug=${p.slug}&j=${encodeURIComponent(job.id)}&e=${exp}&s=${await sign(env, `share.${job.id}.${exp}`)}`;
}

const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/**
 * Opened on the phone from the channel post's "Post on Instagram" button: the video
 * and her caption, and a button that hands the video to Instagram through the phone's
 * share sheet (Web Share API) -- Instagram opens its editor, where a trending song
 * from its library can be added. Instagram's API cannot pick music.
 */
async function sharePage(request, env) {
  const u = new URL(request.url);
  const slug = u.searchParams.get("slug"), jid = u.searchParams.get("j"), e = u.searchParams.get("e");
  if (!jid || !e || Number(e) < Date.now() / 1000 || u.searchParams.get("s") !== await sign(env, `share.${jid}.${e}`)) {
    return new Response("This link has expired.", { status: 403 });
  }
  const state = await st.load(env, slug);
  // The publish job can be gone from the state (a lost update, or pruned) while its
  // link is still in the channel -- the item itself has the video (pub-<item>-<ms>).
  const item = state.items?.[(/^pub-([^-]+)-/.exec(jid) || [])[1]];
  const job = state.publish.find((j) => j.id === jid) || (item?.file_id && { file_id: item.file_id, caption: item.caption });
  if (!job) return new Response("This post is no longer in Lena's list -- save the video from the channel instead.", { status: 404 });
  const video = await mediaLink(env, u.origin, job.file_id, "mp4");
  const caption = job.posts?.instagram?.text || job.caption || "";
  const html = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Post on Instagram</title>
<style>body{font:16px system-ui;margin:0;padding:16px;background:#111;color:#eee;max-width:480px;margin:auto}
video{width:100%;border-radius:12px;background:#000}button,a.b{display:block;width:100%;box-sizing:border-box;margin:10px 0;
padding:14px;border:0;border-radius:10px;font-size:17px;font-weight:600;text-align:center;text-decoration:none}
#share{background:linear-gradient(45deg,#f58529,#dd2a7b,#8134af);color:#fff}#copy,a.b{background:#333;color:#eee}
textarea{width:100%;box-sizing:border-box;height:110px;background:#222;color:#eee;border:1px solid #444;border-radius:8px;padding:8px}
small{color:#999}</style>
<video src="${esc(video)}" controls playsinline loop muted></video>
<button id="copy">1 · Copy caption</button>
<button id="share">2 · Save the video</button>
<small>In the menu pick <b>Save video</b> / <b>Save to gallery</b> (not Instagram: its Reels
share opens the feed with nothing in it, 2026-10-23). Then in Instagram: <b>+ → Reel</b>, pick the
video, tap ♫ for a trending song, paste the caption, share.</small>
<textarea readonly>${esc(caption)}</textarea>
<a class="b" href="${esc(video)}" download="lena.mp4">Or download it</a>
<script>
const url=${JSON.stringify(video)}, cap=${JSON.stringify(caption)};
let file=null; fetch(url).then(r=>r.blob()).then(b=>file=new File([b],"lena.mp4",{type:"video/mp4"}));
document.getElementById("copy").onclick=async(e)=>{try{await navigator.clipboard.writeText(cap);e.target.textContent="✓ Caption copied"}catch{document.querySelector("textarea").select()}};
document.getElementById("share").onclick=async(e)=>{
  if(!file){e.target.textContent="Loading video… tap again";return}
  if(navigator.canShare&&navigator.canShare({files:[file]})){try{await navigator.share({files:[file]})}catch(err){}}
  else{e.target.textContent="Sharing not supported here: open this page in Chrome/Safari, or save the video"}};
</script>`;
  return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
}

async function telegramFile(env, fileId) {
  // Retried once: a busy bot gets "Too Many Requests" now and then.
  const info = await tg(env, "getFile", { file_id: fileId })
    .catch(() => new Promise((r) => setTimeout(r, 1500)).then(() => tg(env, "getFile", { file_id: fileId })));
  return fetch(`https://api.telegram.org/file/bot${env.PERSONA_BOT_TOKEN}/${info.file_path}`);
}

async function serveFile(request, env) {
  const u = new URL(request.url);
  const f = u.searchParams.get("f"), e = u.searchParams.get("e"), s = u.searchParams.get("s");
  if (!f || !e || Number(e) < Date.now() / 1000 || s !== await sign(env, `${f}.${e}`)) {
    return new Response("forbidden", { status: 403 });
  }
  return telegramFile(env, f);
}

async function serveMedia(request, env) {
  const u = new URL(request.url);
  const name = decodeURIComponent(u.pathname.slice("/persona/media/".length));
  if (name.endsWith(".txt")) return tiktok.verifyFile(env, name);
  const dot = name.lastIndexOf(".");
  const f = name.slice(0, dot), ext = name.slice(dot + 1);
  const e = u.searchParams.get("e"), s = u.searchParams.get("s");
  if (!e || Number(e) < Date.now() / 1000 || s !== await sign(env, `${f}.${e}`)) {
    return new Response("forbidden", { status: 403 });
  }
  const file = await telegramFile(env, f);
  const headers = { "content-type": ext === "mp4" ? "video/mp4" : "image/jpeg", "accept-ranges": "bytes" };
  if (!file.ok) return new Response(`media unavailable (${file.status})`, { status: 502 });
  // iPhones play a video only when asked-for byte ranges come back as 206 partial content.
  const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.get("range") || "");
  if (range) {
    const buf = new Uint8Array(await file.arrayBuffer()), n = buf.length;
    let a = range[1] === "" ? n - Number(range[2]) : Number(range[1]);
    let b = range[1] !== "" && range[2] !== "" ? Math.min(Number(range[2]), n - 1) : n - 1;
    a = Math.max(0, a);
    if (a > b || a >= n) return new Response(null, { status: 416, headers: { "content-range": `bytes */${n}` } });
    return new Response(buf.subarray(a, b + 1), { status: 206,
      headers: { ...headers, "content-range": `bytes ${a}-${b}/${n}`, "content-length": String(b - a + 1) } });
  }
  if (file.headers.get("content-length")) headers["content-length"] = file.headers.get("content-length");
  return new Response(file.body, { status: 200, headers });
}

// ------------------------------------------------------------------ publish
/**
 * 👍 on a photo or video: publish it to her Instagram and TikTok. The message's
 * own Telegram copy is what gets posted.
 */
async function queuePublish(env, p, state, cq, itemId) {
  const msg = cq.message || {};
  const known = state.items[itemId] || {};
  const acc = social.accounts(p);
  if (!acc.instagram && !acc.tiktok) return false;
  if (state.publish.some((j) => j.item === itemId)) return "already published";
  if ((state.blocked || []).includes(itemId) || known.private) return "it is private or deleted -- never posted";
  // 🎭 recreates are built on other creators' footage and audio, which Instagram
  // treats as unoriginal: rationed to RECREATE_PER_WEEK so the feed stays mostly
  // original (chosen 2026-10-16).
  if (known.recreate) {
    const week = state.publish.filter((j) => j.recreate && Date.now() - (j.created || 0) < 7 * 86400 * 1000);
    const cap = Number(env.PERSONA_RECREATE_PER_WEEK || 2);
    if (week.length >= cap) {
      const next = new Date(Math.min(...week.map((j) => j.created)) + 7 * 86400 * 1000)
        .toLocaleDateString("en-GB", { timeZone: "Europe/Prague", weekday: "short", day: "numeric", month: "short" });
      return `can't publish: ${cap} 🎭 recreates a week is the Instagram limit (keeps her feed original); next allowed ${next}`;
    }
  }
  const photo = (msg.photo || []).slice(-1)[0];
  const video = msg.video;
  // An old message's tap carries no media: use the file kept when it was posted, or
  // forward the message (silently, deleted at once) -- a forward carries the media.
  const chat = msg.chat?.id;
  let fileId = (photo || video)?.file_id || known.file_id;
  let fwdVideo = null, fwdCaption = null;
  if (!fileId && msg.message_id) {
    try {
      const fwd = await tg(env, "forwardMessage", { chat_id: chat, from_chat_id: chat, message_id: msg.message_id,
                                                     disable_notification: true });
      await tg(env, "deleteMessage", { chat_id: chat, message_id: fwd.message_id }).catch(() => {});
      fwdVideo = fwd.video || null;
      fwdCaption = fwd.caption || "";
      fileId = (fwd.video || (fwd.photo || []).slice(-1)[0])?.file_id;
    } catch (e) {
      console.error("forward for media failed", String(e).slice(0, 200));
    }
  }
  if (!fileId) return "can't publish this one: Telegram didn't give its photo/video";
  const isPhoto = photo ? true : (video || fwdVideo) ? false : known.kind !== "video";
  // MPT: the caption is in the voice of the girl in it.
  const b = (await st.bible(env, known.character || p.slug)) || (await st.bible(env, p.slug)) || { name: p.slug };
  // A caption and hashtags for each platform, never empty (captions.js).
  const posts = await captions.forPost(env, p, b, state, { ...known, kind: isPhoto ? "photo" : "video" });
  const story = isPhoto && feedFull(env, state);
  const silent = !isPhoto && !known.audio && !known.recreate && !known.talk;
  // A day-theme talk (Christmas, Valentine's ...) goes out today, ahead of the others.
  const today = new Date().toLocaleDateString("en-CA", { timeZone: "Europe/Bucharest" });
  const dayOf = known.theme_date && known.theme_date === today;
  state.publish.push({
    id: `pub-${itemId}-${Date.now()}`, item: itemId, kind: isPhoto ? "photo" : "video",
    created: Date.now(), due: social.accounts(p).instagram && !story && !silent && !dayOf ? nextSlot(env, state) : Date.now() + AUTO_WAIT_MS,
    recreate: !!known.recreate, story,
    ...(known.talk ? { talk: true, bw: !!known.bw, tt_due: dayOf ? Date.now() + AUTO_WAIT_MS : nextTalkSlot(env, state) } : {}),
    file_id: fileId, audio: !!known.audio || !!known.recreate || !!known.talk,
    posts, caption: posts.instagram.text,
    chat, message_id: msg.message_id, base: (msg.caption ?? fwdCaption ?? "").split("\n\n━━")[0],
  });
  return true;
}

/**
 * Straight in (2026-10-30, "no more waiting for me"): a finished photo or video
 * publishes on its own, like a 👍, at the next slot -- never sooner than AUTO_WAIT_MS,
 * so a 👎 in the channel can still cancel it. Carousel angles and album photos are left
 * to a 👍 or "post" (an album's TikTok draft is made by itself, postShootAlbum).
 * PERSONA_AUTO=0 turns it off. `sent` is the channel message it went out as.
 */
const AUTO_WAIT_MS = 30 * 60 * 1000;
const autoOn = (env) => env.PERSONA_AUTO !== "0";

async function autoPublish(env, p, state, r, sent) {
  // Only 🎙 talks and 🎭 face swaps go out by themselves (2026-10-30); photos, videos
  // and carousels wait for a 👍 or "post".
  if (!autoOn(env) || !r?.ok || r.private || r.carousel || !(r.talk || r.recreate) || !sent?.message_id || !state.items[r.id]) return;
  const res = await queuePublish(env, p, state, { message: sent }, r.id);
  if (res !== true) return;
  const job = state.publish.at(-1);
  job.auto = true;
  if (job.tt_due) job.tt_due = Math.max(job.tt_due, Date.now() + AUTO_WAIT_MS);
  const when = new Date(Math.max(job.due || 0, Date.now() + AUTO_WAIT_MS))
    .toLocaleString("en-GB", { timeZone: "Europe/Prague", weekday: "short", hour: "2-digit", minute: "2-digit" });
  await noteOnPost(env, sent, `🗓 posts by itself ${when} (Prague time) · 👎 cancels`);
}

/**
 * A 👍'd talk's TikTok time: now, or PERSONA_TT_GAP_H (default 2) hours after the
 * last one. Buffer's own queue had one TikTok slot a day, so talks voted on
 * 2026-10-28 were due on the 29th and 30th.
 */
function nextTalkSlot(env, state) {
  const gap = Number(env.PERSONA_TT_GAP_H || 2) * 3600 * 1000;
  const last = Math.max(0, ...state.publish.map((j) => j.tt_due || 0));
  let t = Math.max(Date.now() + 2 * 60 * 1000, last + gap);
  // Not at night (23:00-07:00 Bucharest, UTC+2): the next 07:00.
  const h = (new Date(t).getUTCHours() + 2) % 24;
  if (h >= 23 || h < 7) {
    const d = new Date(t + 2 * 3600 * 1000 + (h >= 23 ? 86400 * 1000 : 0));
    t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 5);
  }
  return t;
}

/**
 * Photos and carousels in her Instagram feed this week are at PERSONA_IG_FEED_PER_WEEK
 * (default 3): the next one goes to her Story. A new account's photos are shown
 * almost only to followers, Reels to strangers -- Lena's 10-14..10-22 photos and
 * carousels got 2-45 views each (2026-10-23), so the feed is kept mostly for Reels.
 */
function feedFull(env, state) {
  const week = state.publish.filter((j) => (j.kind === "photo" || j.kind === "carousel") && !j.story &&
                                         Date.now() - (j.created || 0) < 7 * 86400 * 1000);
  return week.length >= Number(env.PERSONA_IG_FEED_PER_WEEK ?? 3);
}

/**
 * The next free posting slot (ms). Instagram throttles accounts that post 3+ times a
 * day (each post competes with the last); 1-2 a day is the sweet spot, so 👍 queues
 * a post into the next free slot instead of posting at once. PERSONA_IG_SLOTS: UTC
 * hours, default "10,18" (12:00 and 20:00 in Prague).
 */
const silentReel = (j) => j.kind === "video" && !j.audio && !j.story;

function nextSlot(env, state) {
  const hours = String(env.PERSONA_IG_SLOTS || "10,18").split(",").map(Number).sort((a, b) => a - b);
  // A Story is not a feed post: it never holds a slot (2026-11-05: album Stories
  // held 3 of Lena's slots and pushed her Reels and talks 3-4 days out).
  // Nor a silent Reel: it is shared by hand from Telegram, never at a slot.
  const taken = new Set(state.publish.filter((j) => j.due && !j.story && !silentReel(j)).map((j) => j.due));
  const now = Date.now();
  for (let day = 0; day < 30; day++) {
    for (const h of hours) {
      const d = new Date(now);
      d.setUTCDate(d.getUTCDate() + day);
      d.setUTCHours(h, 0, 0, 0);
      const t = d.getTime();
      if (t > now + 5 * 60 * 1000 && !taken.has(t)) return t;
    }
  }
  return now;
}

/** created/due/story of a new photo post: a Story (the week's feed is full) goes out after the 👎 window, else the next slot. */
function storyOrSlot(env, state) {
  const story = feedFull(env, state);
  return { created: Date.now(), due: story ? Date.now() + AUTO_WAIT_MS : nextSlot(env, state), story };
}

async function advancePublish(env, p, state, origin) {
  let changed = false;
  for (const job of state.publish.filter((j) => !social.finished(j))) {
    const url = job.file_ids
      ? await Promise.all(job.file_ids.map((f) => mediaLink(env, origin, f, "jpg")))
      : await mediaLink(env, origin, job.file_id, job.kind === "video" ? "mp4" : "jpg");
    const before = JSON.stringify([job.ig, job.fb, job.tt, job.retry_at]);
    await social.advance(env, p, job, url);
    // Saved only when something moved: this runs on every 30s laptop check-in, and an
    // unconditional save there overwrote taps and 🖼 requests that the webhook had
    // saved a moment before (2026-10-16, two carousel requests lost).
    if (JSON.stringify([job.ig, job.fb, job.tt, job.retry_at]) === before) continue;
    changed = true;
    if (job.retry_at && Date.now() < job.retry_at) continue;  // rate-limited: nothing to show yet
    if (job.quiet || !job.chat) continue;  // a daily Story: no post of its own to mark
    // Mark the post in the channel: what happened on each platform.
    const cap = `${job.base}\n\n━━ published ━━\n${social.statusLine(job)}`.slice(0, 1024);
    const failed = ["ig", "tt", "fb"].some((k) => job[k]?.status === "failed");
    // Nothing posted yet (only a platform it doesn't go to was settled): 👎 still cancels.
    const waiting = !["ig", "tt"].some((k) => job[k] && job[k].status !== "skipped");
    const markup = job.ig?.mode === "phone"
      ? { inline_keyboard: [[{ text: "📲 Post on Instagram", url: await shareLink(env, origin, p, job) }]] }
      : failed ? kb([[["🔁 Retry posting", `pv:repub:${job.id}`]]])
      : waiting ? kb([[["🗓 queued", `pv:noop:${job.item}`], ["👎 Cancel", `pv:dn:${job.item}`]]])
      : kb([[["✓ 👍 published", `pv:noop:${job.item}`]]]);
    try {
      await tg(env, job.text ? "editMessageText" : "editMessageCaption", { chat_id: job.chat, message_id: job.message_id,
        ...(job.text ? { text: cap.slice(0, 4096) } : { caption: cap }), ...(job.album ? {} : { reply_markup: markup }) });
    } catch (e) {
      console.error("caption update failed", String(e).slice(0, 200));
    }
  }
  // Keep the last 100.
  if (state.publish.length > 100) { state.publish = state.publish.slice(-100); changed = true; }
  return changed;
}

// ------------------------------------------------------------------- taps
async function onTap(env, p, cq, state, origin) {
  const [, act, itemId] = cq.data.split(":");
  const msg = cq.message || {};
  const chat = msg.chat?.id;
  if (act === "noop") {
    const waiting = state.publish.some((j) => j.item === itemId && !social.finished(j));
    return answer(env, cq.id, waiting ? "Queued: it posts by itself at its time (👎 cancels)." : "Already published.");
  }
  if (act === "repub") {
    // 🔁 Retry posting: the failed platform(s) start again.
    const job = state.publish.find((j) => j.id === itemId);
    if (!job) return answer(env, cq.id, "That post isn't known any more.");
    for (const k of ["ig", "tt", "fb"]) if (job[k]?.status === "failed") delete job[k];
    job.tries = 0;
    job.retry_at = 0;
    await advancePublish(env, p, state, origin);
    return answer(env, cq.id, "🔁 Posting again…");
  }
  if (act === "dn") {
    // 👎 on a post waiting for its slot: it won't be published.
    const waiting = state.publish.filter((j) => j.item === itemId && !["ig", "tt"].some((k) => j[k] && j[k].status !== "skipped"));
    if (waiting.length) {
      state.publish = state.publish.filter((j) => !waiting.includes(j));
      await noteOnPost(env, msg, "🚫 cancelled -- won't be posted");
      if (state.voted.includes(itemId)) return answer(env, cq.id, "👎 Cancelled: it won't be posted.");
    }
  }
  if (act === "up" || act === "dn") {
    if (state.voted.includes(itemId)) {
      return answer(env, cq.id, "Already voted.");
    }
    const item = state.items[itemId] || { tags: [] };
    const d = act === "up" ? 1 : -1;
    for (const t of item.tags || []) state.votes.tags[t] = (state.votes.tags[t] || 0) + d;
    learnFrom(state, item, d, { taste: d > 0 });
    if (act === "up" && item.beat) state.story.push({ t: Date.now(), beat: item.beat, item: itemId });
    state.voted.push(itemId);
    const res = act === "up" ? await queuePublish(env, p, state, cq, itemId) : false;
    const publishing = res === true;
    await answer(env, cq.id, act === "dn" ? "Noted 👎" : publishing ? `👍 Added to Buffer's queue for ${social.accounts(p).instagram ? "Instagram" : "TikTok"}`
      : typeof res === "string" ? `Noted 👍 -- ${res}` : "Noted 👍");
    if (typeof res === "string" && res.startsWith("can't")) {
      await tg(env, "sendMessage", { chat_id: chat, reply_to_message_id: msg.message_id, text: `⚠️ 👍 noted, but ${res}.` }).catch(() => {});
    }
    const label = act === "up" ? (publishing ? "⏳ publishing" : "✓ 👍") : "✓ 👎";
    const row = [[label, publishing ? `pv:noop:${itemId}` : cq.data]];
    if (msg.photo && !publishing) row.push(["🎬 Video", `pv:vid:${itemId}`], ["🔁 Redo", `pv:redo:${itemId}`]);
    try {
      await tg(env, "editMessageReplyMarkup", { chat_id: chat, message_id: msg.message_id, reply_markup: kb([row]) });
    } catch { /* unchanged markup */ }
    return;
  }
  if (act === "sredo" || act === "svid") {
    // A photo in a shoot's album, by its button under the album.
    const it = state.items[itemId];
    if (!it?.message_id) return answer(env, cq.id, "Can't: this photo isn't known any more.");
    const at = { chat: { id: chat }, message_id: it.message_id };
    if (act === "svid") {
      queueVideo(state, itemId, at, it.file_id, "");
      return answer(env, cq.id, "🎬 Queued. The video comes as a reply to the photo in ~40-60 min.");
    }
    if (!it.scene?.setting) return answer(env, cq.id, "Can't redo this one: its scene isn't known.");
    if (state.queue.some((q) => q.kind === "redo" && q.item === itemId)) return answer(env, cq.id, "Already queued.");
    // Its shoot, so the redraw is swapped into the album (without it the new photo
    // came as a post of its own).
    const shootId = Object.entries(state.shoots || {}).find(([, x]) => (x.ids || []).includes(itemId))?.[0];
    if (shootId) state.shoots[shootId].redo_wait = (state.shoots[shootId].redo_wait || 0) + 1;
    state.queue.push({ id: `${itemId}-r${Date.now()}`, kind: "redo", item: itemId, scene: it.scene, chat, message_id: it.message_id,
                       ...(shootId ? { shoot: shootId, replaces: itemId } : {}),
                       ...(it.character ? { character: it.character } : {}) });
    return answer(env, cq.id, "🔁 Queued. The new photo replaces it in the album in ~15-30 min.");
  }
  if (act === "pick" || act === "recast") {
    const slug = state.cast_of?.[itemId];
    const build = slug && state.builds?.[slug];
    if (!build) return answer(env, cq.id, "This casting isn't known any more.");
    if (state.queue.some((q) => (q.kind === "lock" || q.kind === "cast") && q.new_slug === slug)) return answer(env, cq.id, "Already working on her.");
    if (act === "recast") {
      state.queue.push({ id: `cast-${slug}-${Date.now()}`, kind: "cast", new_slug: slug, bible: build.bible, n: 4, character: p.characters[0],
                         chat, message_id: msg.message_id, local_only: true });
      return answer(env, cq.id, "🔁 Casting 4 more faces (~5-10 min)");
    }
    state.queue.push({ id: `lock-${slug}-${Date.now()}`, kind: "lock", new_slug: slug, pick: itemId, character: p.characters[0],
                       chat, message_id: msg.message_id, local_only: true });
    await noteOnPost(env, msg, "✅ picked -- making her refs and face sheet (~20-30 min)");
    return answer(env, cq.id, "✅ Picked");
  }
  if (act === "vredo") {
    const res = redoVideo(p, state, itemId, msg);
    if (res === true) await noteOnPost(env, msg, "🔁 making it again -- the new one comes as a reply");
    return answer(env, cq.id, res === true ? "🔁 Queued" : res);
  }
  if (act === "vid") {
    const photo = (msg.photo || []).slice(-1)[0];
    if (!photo) return answer(env, cq.id, "That is not a photo.");
    queueVideo(state, itemId, msg, photo.file_id, "");
    await noteOnPost(env, msg, "🎬 making a video -- it comes as a reply (~40-60 min)");
    return answer(env, cq.id, "🎬 Queued");
  }
  if (act === "car") {
    const scene = state.items[itemId]?.scene;
    if (!scene?.setting) return answer(env, cq.id, "Can't: this photo's scene wasn't recorded.");
    if (state.queue.some((q) => q.kind === "carousel" && q.item === itemId)) return answer(env, cq.id, "Already queued.");
    const fileId = (msg.photo || []).slice(-1)[0]?.file_id || state.items[itemId]?.file_id;
    if (!fileId) return answer(env, cq.id, "Can't: Telegram didn't give this photo.");
    state.queue.push({ id: `${itemId}-car${Date.now()}`, kind: "carousel", item: itemId, scene, file_id: fileId, count: 3,
                       chat, message_id: msg.message_id });
    await noteOnPost(env, msg, "🖼 making 3 more angles for a carousel (~10-20 min)");
    return answer(env, cq.id, "🖼 Queued");
  }
  if (act === "carpub") return publishCarousel(env, p, state, cq, itemId);
  // Which carousel a shot belongs to: its own mark, else the carousel sets (shots
  // made before the mark existed, or whose mark a lost save dropped).
  const carouselOf = state.items[itemId]?.carousel_of
    || Object.entries(state.carousels || {}).find(([, set]) => (set.ids || []).includes(itemId))?.[0]
    || /another angle of #([\w-]+)/.exec(msg.caption || "")?.[1];
  if (act === "redo" && carouselOf) {
    // 🔁 on a carousel angle: a new angle of the same original photo replaces it.
    const parent = carouselOf;
    const fileId = state.items[parent]?.file_id;
    if (!fileId) {
      await tg(env, "sendMessage", { chat_id: chat, reply_to_message_id: msg.message_id,
        text: "⚠️ Can't redo this shot: its original photo isn't known any more. Reply \"carousel\" to the original photo for new shots." }).catch(() => {});
      return answer(env, cq.id, "Can't: the original photo isn't known.");
    }
    state.carousels = state.carousels || {};
    const set = state.carousels[parent] || { ids: [], offered: true };
    if (!set.ids.includes(itemId)) set.ids.push(itemId);
    state.carousels[parent] = set;
    state.queue.push({ id: `${parent}-car${Date.now()}`, kind: "carousel", item: parent, scene: state.items[parent]?.scene || {},
                       file_id: fileId, count: 1, replace: itemId, chat, message_id: msg.message_id });
    return answer(env, cq.id, "🔁 Redrawing this angle (~10 min).");
  }
  if (act === "redo") {
    const scene = state.items[itemId]?.scene;
    if (!scene?.setting) {
      await tg(env, "sendMessage", { chat_id: chat, reply_to_message_id: msg.message_id,
        text: "⚠️ Can't redo this one: its scene wasn't recorded (older post)." }).catch(() => {});
      return answer(env, cq.id, "Can't redo this one: its scene isn't known.");
    }
    if (state.queue.some((q) => q.kind === "redo" && q.item === itemId)) return answer(env, cq.id, "Already queued.");
    learnFrom(state, state.items[itemId], -1);
    state.queue.push({ id: `${itemId}-r${Date.now()}`, kind: "redo", item: itemId, scene,
                       chat, message_id: msg.message_id });
    await noteOnPost(env, msg, "🔁 redrawing -- it replaces this photo (~15-30 min)");
    return answer(env, cq.id, "🔁 Queued");
  }
  return answer(env, cq.id, "");
}

// TikTok, or a Facebook reel/video (facebook.com/reel/…, /share/r/…, /watch, fb.watch/…).
const CLIP_LINK = /https?:\/\/(?:(?:www\.|vm\.|vt\.|m\.)?tiktok\.com|(?:www\.|m\.|web\.)?facebook\.com|fb\.watch)\/\S+/i;

/**
 * 🎭 A TikTok or Facebook reel link, or a video uploaded into the channel: recreate that clip with
 * her (persona/recreate.py). "8s" or "first 8" in the text sets the length.
 */
async function onRecreateRequest(env, post, state, p) {
  const text = post.text || post.caption || "";
  const link = (CLIP_LINK.exec(text) || [])[0];
  const video = post.video || (post.document?.mime_type?.startsWith("video/") ? post.document : null);
  if (!link && !video) return false;
  // MPT: a girl's name before the link picks her; otherwise the laptop picks the girl
  // whose hair and skin fit the clip's person best (persona/fit.py).
  const [named] = isRoster(p) ? await namedGirl(env, p, text.replace(CLIP_LINK, "").trim()) : [null];
  const secs = /(?:first\s*)?(\d{1,2})\s*(?:s|sec|seconds?)\b/i.exec(text) || /first\s+(\d{1,2})/i.exec(text);
  const seconds = Math.min(Number(secs?.[1] || env.PERSONA_RECREATE_MAX || 10), Number(env.PERSONA_RECREATE_MAX || 10));
  state.queue.push({ id: `rec-${post.message_id}-${Date.now()}`, kind: "recreate", item: `rec-${post.message_id}`,
                     url: link || "", file_id: video?.file_id || "", seconds,
                     chat: post.chat.id, message_id: post.message_id,
                     ...(named ? { character: named } : isRoster(p) ? { fit: p.characters } : {}) });
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

// ------------------------------------------------------------------ 🎙 talks
/**
 * Talking videos (2026-10-27): each girl talks about one topic on a podcast set,
 * the line rewritten from a popular short on that topic (persona/talk.py). Persona
 * entry keys: talk_hours (UTC hours, one talk each), talkers (roster girls who do
 * talks, default all), talk_topics ({slug: {topic, sources}} overrides),
 * talk_cross (another persona's slug: each talk is also posted in its channel).
 * Laptop only (local_only): InfiniteTalk + LTX don't fit Kaggle.
 */
// Sources: TikTok accounts whose videos are checked one by one for what people did
// with them (shares, saves, comments, likes -- the user, 2026-10-27: only sayings
// people loved), median per video when picked: jayshetty 52k views/455 saves,
// melrobbins 27k/338, drewafualo 432k/871, hardtruthtalks 17k/219, shetalks.truth
// 10k/148, yourrichbff 11k/65; the user's picks voicebloom1 (top 471k views/9.6k shares)
// and therealtruth93 (top 1.3M/17k shares), riseshine.7 (median 54k views/815 shares,
// top 2.1M/23.5k shares/37k saves), thelasthopeofficial06 (median 42k/419 shares),
// mindfuel_028 (median 120k views/1.3k shares/1.6k saves). A YouTube search last, when none of them has one.
const T = (handles, yt) => [...handles.map((h) => `tiktok:@${h}`), yt];
const TALK_TOPICS = {
  lena: { topic: "dating truths for women", sources: T(["shetalks.truth", "mindfuel_028", "riseshine.7", "voicebloom1", "therealtruth93", "hardtruthtalks", "jayshetty"], "relationship advice for women podcast clip") },
  "mpt-gia": { topic: "confidence, glow-ups and never chasing anyone", sources: T(["mindfuel_028", "riseshine.7", "melrobbins", "kindpush343", "jayshetty"], "self love women motivation podcast clip") },
  "mpt-freya": { topic: "breakups, getting over him and never going back", sources: T(["riseshine.7", "mindfuel_028", "hardtruthtalks", "jayshetty", "melrobbins"], "healing after breakup advice podcast clip") },
  "mpt-aiko": { topic: "female psychology men don't understand", sources: T(["therealtruth93", "voicebloom1", "shetalks.truth", "hardtruthtalks"], "female psychology facts about women podcast") },
  "mpt-leila": { topic: "high-value woman standards", sources: T(["mindfuel_028", "riseshine.7", "thelasthopeofficial06", "shetalks.truth", "kindpush343", "drewafualo"], "high value woman standards podcast clip") },
  "mpt-skye": { topic: "red flags and situationships", sources: T(["drewafualo", "therealtruth93", "hardtruthtalks", "shetalks.truth"], "dating red flags situationship advice podcast") },
  "mpt-chloe": { topic: "what real love looks like vs what women settle for", sources: T(["riseshine.7", "mindfuel_028", "jayshetty", "voicebloom1", "hardtruthtalks", "shetalks.truth"], "deep love quotes relationship truth podcast") },
  "mpt-priya": { topic: "marriage truths wives never say out loud", sources: T(["voicebloom1", "jayshetty", "2bebetterpodcast", "hardtruthtalks"], "marriage advice husband wife podcast clip") },
  "mpt-harper": { topic: "discipline, glow-ups and the excuses women make", sources: T(["thelasthopeofficial06", "mindfuel_028", "melrobbins", "jayshetty", "kindpush343"], "motivational speech discipline women podcast") },
  "mpt-alina": { topic: "money, who pays and women who level up", sources: T(["mindfuel_028", "yourrichbff", "thelasthopeofficial06", "melrobbins"], "women money mindset ambition podcast clip") },
  "mpt-sofia": { topic: "what a real man does for his woman", sources: T(["mindfuel_028", "therealtruth93", "riseshine.7", "thelasthopeofficial06", "voicebloom1", "shetalks.truth", "hardtruthtalks"], "how a man should treat a woman podcast clip") },
  "mpt-lola": { topic: "toxic exes and moving on", sources: T(["riseshine.7", "thelasthopeofficial06", "hardtruthtalks", "drewafualo", "melrobbins"], "moving on from toxic ex advice podcast clip") },
};

// ------------------------------------------------------------- 🗓 Lena's day themes
/**
 * Lena's talks were always the same "dating truths" (2026-11-12: "same thing always.. we need
 * new stuff"). Now: one talk a day on what today is (Christmas, Valentine's, New Year,
 * cuffing season ...), in a room and outfit to match, put first in the queue and posted that
 * same day; the others mostly fresh angles (LENA_FRESH), the classic viral sayings every 3rd.
 * Sources are YouTube searches for the theme; the line keeps to the theme (on_topic).
 */
const LOOK = {
  christmas: { outfit: "a cosy red knit Christmas sweater", where: "sitting on a couch next to a lit Christmas tree",
               behind: "a cosy living room with a decorated Christmas tree and warm fairy lights", light: "warm fairy-light glow and soft lamp light" },
  xmasCream: { outfit: "a cream cable-knit sweater", where: "curled up on her bed under a blanket, fairy lights on the headboard",
               behind: "a cosy bedroom with warm fairy lights and a small Christmas tree", light: "warm fairy-light glow" },
  nye: { outfit: "a sparkly silver sequin top", where: "at home getting ready for a New Year's Eve party",
         behind: "a festive room with gold balloons and fairy lights", light: "warm party lights" },
  winter: { outfit: "an oversized cosy knit sweater", where: "sitting by a window on a snowy day with a warm mug",
            behind: "a cosy room, snow falling outside the window", light: "soft cold winter daylight" },
  valentine: { outfit: "a fitted red top", where: "sitting on her couch",
               behind: "a cosy room with red roses on a table and candles", light: "warm candle light and a soft lamp" },
  autumn: { outfit: "an oversized cosy knit sweater", where: "curled up on the couch under a blanket",
            behind: "a cosy room with a warm candle and a knitted throw", light: "warm lamp light on a grey autumn evening" },
  thanks: { outfit: "a cosy rust-coloured knit sweater", where: "sitting at a dining table",
            behind: "a warm dining room with an autumn table, candles and pumpkins", light: "warm candle light" },
  shopping: { outfit: "a fitted beige top", where: "sitting on her bed next to shopping bags",
              behind: "a tidy bedroom, shopping bags on the bed", light: "soft daylight from a window" },
  halloween: { outfit: "a fitted black top", where: "sitting on her couch",
               behind: "a cosy room with carved glowing pumpkins and candles", light: "warm orange candle light" },
  summer: { outfit: "a light white linen top", where: "sitting on a sunny balcony",
            behind: "a balcony with plants and a bright summer sky", light: "warm golden summer sun" },
};
const yt = (...q) => q;
// The rotating angle of a long season: the day of the year picks one.
const rot = (arr, date) => arr[Math.floor(date.getTime() / 86400000) % arr.length];

function dayTheme(now = new Date()) {
  const ymd = now.toLocaleDateString("en-CA", { timeZone: "Europe/Bucharest" });
  const [y, mo, d] = ymd.split("-").map(Number);
  const date = new Date(Date.UTC(y, mo - 1, d));
  const dow = date.getUTCDay();
  const nth = (month, weekday, n) => { const f = new Date(Date.UTC(y, month - 1, 1)).getUTCDay(); return 1 + ((weekday - f + 7) % 7) + 7 * (n - 1); };
  const md = mo * 100 + d;
  const th = (key, topic, searches, look) => ({ key, date: ymd, topic, searches, look: LOOK[look] || null });
  const thanks = nth(11, 4, 4);
  if (md === 1224) return th("christmas-eve", "Christmas Eve: love, who you spend it with, and being enough on your own",
    yt("christmas eve love quotes women", "spending christmas alone women motivation", "christmas relationship truth"), "christmas");
  if (md === 1225) return th("christmas", "Christmas Day: real love, family and the people who show up",
    yt("christmas love message women", "christmas relationship advice", "christmas single women"), "christmas");
  if (md >= 1226 && md <= 1230) return th("after-christmas", "the days between Christmas and New Year: the ex who texts, looking back on the year",
    yt("ex texts on christmas advice", "end of year reflection women motivation", "leaving this year behind relationship"), "xmasCream");
  if (md === 1231) return th("nye", "New Year's Eve: who you kiss at midnight and who you leave in this year",
    yt("new years eve relationship advice women", "leave them in the old year motivation", "new year new standards dating"), "nye");
  if (md >= 101 && md <= 103) return th("new-year", "a new year: new standards in love, and who doesn't come with you",
    yt("new year new standards women", "new year glow up mindset women", "new year relationship goals"), "winter");
  if (mo === 12) return th("christmas-season", rot([
    "being single at Christmas and loving it", "what his Christmas gift says about him", "meeting his family over the holidays",
    "the holiday party and seeing your ex", "cosy Christmas dates and what real romance looks like", "family asking why you're still single at Christmas",
    "Christmas with someone who makes you feel safe"], date),
    yt("christmas dating advice women", "single at christmas women", "christmas relationship truths", "holidays with your partner advice"),
    rot(["christmas", "xmasCream"], date));
  if (mo === 1 && d === nth(1, 1, 3)) return th("blue-monday", "the saddest day of the year: getting through a heavy winter heart",
    yt("blue monday motivation women", "winter sadness heartbreak advice"), "winter");
  if (md === 213) return th("galentines", "Galentine's Day: the friends who love you better than any man did",
    yt("galentines day female friendship", "best friend love women quotes"), "valentine");
  if (md === 214) return th("valentines", "Valentine's Day: what real love looks like, single or taken",
    yt("valentines day love truth women", "single on valentines day women", "valentines day relationship advice"), "valentine");
  if (md === 215) return th("after-valentines", "the day after Valentine's: what his Valentine's really told you",
    yt("valentines day red flags", "what he did on valentines day says"), "valentine");
  if (mo === 2 && d <= 12) return th("valentine-season", rot(["Valentine's is coming and you're single", "what to really want for Valentine's",
    "the situationship that disappears before Valentine's"], date), yt("valentines day dating advice women", "valentines situationship"), "valentine");
  if (md === 308) return th("womens-day", "Women's Day: what women carry and never get thanked for",
    yt("womens day message women strength", "strong women quotes motivation"), null);
  if (mo === 5 && d === nth(5, 0, 2)) return th("mothers-day", "Mother's Day: what our mothers taught us about love",
    yt("mothers day love message", "what my mother taught me about love"), null);
  if (mo >= 6 && mo <= 8) return dow === 5 || dow === 6 ? th("summer", rot(["summer flings and what they mean", "summer glow and being single",
    "holiday romance truths"], date), yt("summer fling advice women", "summer dating women", "vacation romance truth"), "summer") : null;
  if (md === 1031) return th("halloween", "Halloween: the scariest red flags in men",
    yt("scariest red flags in men", "halloween dating red flags"), "halloween");
  if (md === 1111) return th("singles-day", "Singles' Day: the best things about being single",
    yt("benefits of being single women", "single women happy"), "autumn");
  if (mo === 11 && d === thanks) return th("thanksgiving", "Thanksgiving: being thankful for the love that stayed and the one that left",
    yt("thanksgiving love message", "grateful for heartbreak motivation", "thanksgiving single women family"), "thanks");
  if (mo === 11 && d === thanks + 1) return th("black-friday", "Black Friday: what's worth it in love and what you should never settle for, even on sale",
    yt("dont settle for less women motivation", "know your worth dating women"), "shopping");
  if (md >= 1015 && md <= 1130) return th("cuffing-season", rot(["cuffing season: men who only show up when it gets cold",
    "cosy autumn nights and wanting someone real", "winter coming and texting the ex", "cuffing season vs real love"], date),
    yt("cuffing season dating advice", "cuffing season men", "autumn relationship advice women"), "autumn");
  return null;
}

// Lena's fresh angles (not the classic "dating truths"): one picked, not one of the last 8.
const LENA_FRESH = [
  { topic: "first dates: what really happens and what to watch for", sources: yt("first date stories women", "first date advice women", "first date red flags") },
  { topic: "texting: what his texts really mean", sources: yt("what his texts mean dating advice", "texting rules dating women") },
  { topic: "dating apps in real life", sources: yt("dating app stories women", "dating apps advice women", "dating app red flags") },
  { topic: "green flags: the small things good men do", sources: yt("green flags in a man", "signs he really likes you", "small things good men do") },
  { topic: "being single and loving your own life", sources: yt("enjoying being single women", "single women happy life") },
  { topic: "friendships between women: the friends who stay", sources: yt("female friendship truths", "real friends vs fake friends women") },
  { topic: "meeting his family, his mother and his friends", sources: yt("meeting his family advice", "his mother relationship advice") },
  { topic: "jealousy and trust", sources: yt("jealousy in relationships advice", "trust in a relationship women") },
  { topic: "love languages and the small things that mean love", sources: yt("love languages relationship", "small things that mean love") },
  { topic: "what men actually find attractive, honestly", sources: yt("what men find attractive in women honest", "men answer what makes a woman attractive") },
  { topic: "a calm, peaceful love vs a chaotic one", sources: yt("peaceful relationship vs chaotic", "calm love women") },
  { topic: "your twenties: dating mistakes and growing up", sources: yt("dating in your 20s advice women", "things i wish i knew in my 20s women") },
  { topic: "long-distance love", sources: yt("long distance relationship truths", "long distance relationship advice") },
  { topic: "funny truths about being a girlfriend", sources: yt("funny relationship truths women", "girlfriend problems funny") },
];

/** Lena's talk: today's theme (once a day), else a fresh angle, the classic viral one every 3rd. */
function lenaTopic(state) {
  const th = dayTheme();
  if (th && state.theme_day !== `${th.date}:${th.key}`) {
    state.theme_day = `${th.date}:${th.key}`;
    return { topic: th.topic, sources: th.searches, theme: th };
  }
  state.lena_mix = (state.lena_mix || 0) + 1;
  if (state.lena_mix % 3 === 0) return anyTopic();
  const recent = state.fresh_used || [];
  const pool = LENA_FRESH.filter((f) => !recent.includes(f.topic));
  const pick = (pool.length ? pool : LENA_FRESH)[Math.floor(Math.random() * (pool.length || LENA_FRESH.length))];
  state.fresh_used = [...recent, pick.topic].slice(-8);
  return pick;
}

function anyTopic() {
  const all = Object.values(TALK_TOPICS);
  const handles = [...new Set(all.flatMap((t) => t.sources.filter((x) => x.startsWith("tiktok:"))))];
  const pick = all[Math.floor(Math.random() * all.length)];
  return { topic: pick.topic, sources: [...handles, pick.sources.at(-1)], viral: true };
}

function talkTopic(p, who) {
  return p.talk_topics?.[who] || TALK_TOPICS[who] || { topic: "relationships and self-worth", sources: ["relationship advice podcast clip"] };
}

/** One 🎙 talk queued for the laptop: the next talker in rotation (MPT) or her (Lena). */
// A numbered series per girl (2026-11-10, "real fans"): the same label on every talk of hers,
// "Getting over him #14", so a viewer knows there is a #15 and follows for it.
const TALK_SERIES = {
  lena: "Dating truths", "mpt-gia": "Glow-up rules", "mpt-freya": "Getting over him", "mpt-aiko": "What men don't get",
  "mpt-leila": "High-value habits", "mpt-skye": "Red flag files", "mpt-chloe": "Real love vs settling",
  "mpt-priya": "Wife truths", "mpt-harper": "No excuses", "mpt-alina": "Money talks", "mpt-sofia": "What a real man does",
  "mpt-lola": "The ex files",
};

function queueTalk(p, state, extra = null) {
  // The laptop was off: don't pile up a day's worth (one asked for in the channel always goes).
  // Lena's day theme goes first and always (posted that same day); the rest wait their turn.
  const t0 = !isRoster(p) && !extra && p.talk_any !== false ? lenaTopic(state) : null;
  if (!extra && !t0?.theme && state.queue.filter((q) => q.kind === "talk").length >= 3) return `${p.slug}: talks already waiting`;
  let who = extra?.who || p.slug;
  if (isRoster(p) && !extra?.who) {
    const pool = (p.talkers?.length ? p.talkers : p.characters);
    who = pool[(state.talk_rotation || 0) % pool.length];
    state.talk_rotation = (state.talk_rotation || 0) + 1;
  }
  // Lena (2026-11-06: "lena to get all, with always what is most viral"): every girl's topic
  // and every source account, the most-shared short among all of them, whatever its topic.
  const t = t0 || (!isRoster(p) && p.talk_any !== false ? anyTopic() : talkTopic(p, who));
  // A/B (2026-10-27): every other talk in black and white, like the advice accounts.
  const bw = !!((state.talk_n || 0) % 2);
  state.talk_n = (state.talk_n || 0) + 1;
  state.series_n = state.series_n || {};
  state.series_n[who] = (state.series_n[who] || 0) + 1;
  state.queue[t.theme ? "unshift" : "push"]({ id: `talk-${who}-${Date.now()}`, kind: "talk", character: who, topic: t.topic, sources: t.sources,
                     chat: p.chat_id, local_only: true, bw: t.theme ? false : bw, ...(t.viral ? { viral: true } : {}),
                     ...(t.theme ? { theme: t.theme.key, theme_date: t.theme.date, look: t.theme.look } : {}),
                     series: TALK_SERIES[who] || "", episode: state.series_n[who],
                     ...(extra?.follow ? { follow_line: extra.follow } : {}),
                     ...(extra?.reply ? { reply_comment: extra.reply, reply_to_line: extra.reply_to || "" } : {}),
                     ...(extra && (extra.line || extra.url) ? { given_line: extra.line || "", given_url: extra.url || "", message_id: extra.message_id } : {}) });
  const what = extra?.follow ? "Part 2" : extra?.reply ? "comment reply" : `${t.theme ? "🗓 " : ""}${t.topic}`;
  return `${p.slug}: 🎙 talk queued for ${who} (${what})`;
}

// ------------------------------------------------------------- 🔒 private
/**
 * "private [video] [Name:] <prompt>" in MPT's channel (2026-10-30): her on the laptop's
 * uncensored Fanvue model, the prompt as written plus only her identity. The result
 * comes back to the channel with no buttons and is blocked from every way out --
 * 👍, "post", auto-publish, cross-posts, backfills and Stories (blockItem's list).
 * Laptop only, never Kaggle.
 */
async function onPrivateRequest(env, p, post, state) {
  const m = /^(?:🔒|private|fanvue)\s+(video\b[\s:,-]*)?(.*)$/isu.exec(String(post.text || "").trim());
  if (!m || post.reply_to_message) return false;
  let [who, prompt] = await namedGirl(env, p, m[2].trim());
  if (!who) who = p.characters[Math.floor(Math.random() * p.characters.length)];
  if (prompt.split(/\s+/).length < 2) {
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
      text: "🔒 Say what you want after it: private Gia: <prompt> · private video Gia: <prompt>" });
    return true;
  }
  state.queue.push({ id: `private-${who}-${Date.now()}`, kind: "private", character: who, prompt: prompt.slice(0, 1200),
                     video: !!m[1], chat: post.chat.id, message_id: post.message_id, local_only: true });
  const b = (await st.bible(env, who)) || { name: who };
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: `🔒 Queued (private, never posted): ${b.name}${m[1] ? ", a photo then its video" : ""}. Comes as a reply here.` });
  return true;
}

/** A private result: kept out of every way to publish, and its video queued. */
function privateDone(state, r, sent) {
  if (!r?.ok || !r.private) return;
  state.blocked = [...new Set([...(state.blocked || []), r.id])].slice(-500);
  if (state.items[r.id]) state.items[r.id].private = true;
  const ph = (sent?.photo || []).slice(-1)[0];
  if (r.video && r.kind === "image" && ph) {
    state.queue.push({ id: `${r.id}-pv${Date.now()}`, kind: "video", item: r.id, file_id: ph.file_id, motion: r.prompt || "",
                       private: true, character: r.character, chat: r.chat, message_id: sent.message_id, local_only: true });
  }
}

// ------------------------------------------------------------- 👤 builder
/**
 * "new persona <description>" in MPT's channel (2026-10-30): her bible is written
 * here, the laptop casts 4 faces (jobs.py cast) that come in with ✅ Pick / 🔁 More
 * faces, and the pick makes her refs and face sheet (jobs.py lock); then she joins
 * the roster with a talk topic of her own. Laptop only.
 */
const TALK_SOURCES = ["mindfuel_028", "riseshine.7", "shetalks.truth", "therealtruth93", "hardtruthtalks", "voicebloom1", "jayshetty"];

async function onBuildRequest(env, p, post, state) {
  const m = /^(?:👤\s*)?(?:new\s+(?:persona|girl|model)|persona\s+builder|build\s+(?:a\s+)?persona)\b[\s:,-]*(.*)$/isu.exec(String(post.text || "").trim());
  if (!m || post.reply_to_message) return false;
  const brief = m[1].trim();
  if (brief.split(/\s+/).length < 2) {
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
      text: "👤 Describe her after it, e.g. \"new persona Brazilian fitness girl from Rio, long dark curls, talks about gym motivation\"." });
    return true;
  }
  const taken = [];
  for (const slug of p.characters) taken.push(((await st.bible(env, slug))?.name) || slug);
  const used = Object.values(p.talk_topics || {}).map((t) => t.topic).concat(p.characters.map((c) => TALK_TOPICS[c]?.topic).filter(Boolean));
  const out = await captions.ask(env,
    `Design a NEW fictional AI influencer for an Instagram/TikTok roster from this brief: "${brief.slice(0, 500)}".\n` +
    `Rules: fictional and original (never a real person or celebrity), clearly an adult aged 21-28, glamorous and ` +
    `sexy but Instagram-safe. Her body is ALWAYS "slim toned figure, narrow waist, full bust" (never curvy hips). ` +
    `Her name must not be one of: ${taken.join(", ")}. Her talk topic must be a relationship/self-worth theme that ` +
    `is different from: ${used.join("; ")}.\n` +
    `Return JSON: {"name": "...", "age": 23, "identity": "ethnicity, hair (colour, length, style), eye colour, skin, ` +
    `face details, makeup, slim toned figure, narrow waist, full bust", "tagline": "...", "personality": "...", ` +
    `"backstory": "...", "home_city": "...", "recurring_places": ["5 places"], "style": ["5 outfit styles"], ` +
    `"content_pillars": ["3"], "casting": {"ethnicity": ["one"], "hair": ["one"], "eyes": ["one"], "face": ["one"], ` +
    `"build": ["slim toned figure, narrow waist, full bust"]}, "talk_topic": "..."}`, 1200);
  const name = String(out.name || "").trim().split(/\s+/)[0];
  if (!name || !out.identity || taken.map((t) => t.toLowerCase()).includes(name.toLowerCase())) {
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id, text: "⚠️ Couldn't write her profile -- send it again." });
    return true;
  }
  const slug = `mpt-${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  const bible = { ...out, name, age: Math.min(28, Math.max(21, Number(out.age) || 23)),
    lanes: { social: { cue: "sexy, alluring Instagram model look, glamorous styled hair, a figure-flattering, a little revealing outfit, flirty model pose, no nudity",
                       negative: "nudity, nipples, genitals, see-through clothing, explicit pose" } } };
  state.builds = { ...(state.builds || {}), [slug]: { bible, topic: String(out.talk_topic || "relationships and self-worth").slice(0, 80), chat: post.chat.id } };
  state.queue.push({ id: `cast-${slug}-${Date.now()}`, kind: "cast", new_slug: slug, bible, n: 4, character: p.characters[0],
                     chat: post.chat.id, message_id: post.message_id, local_only: true });
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: `👤 ${name}, ${bible.age} · ${bible.home_city || ""}\n${bible.identity}\n🎙 talks about: ${state.builds[slug].topic}\n\n` +
          `Casting 4 faces on the laptop (~5-10 min). Tap ✅ on the one you want.` });
  return true;
}

/** A cast face remembered for ✅ Pick; a finished girl joins the roster. */
async function builderDone(env, p, state, r) {
  if (r.ok && r.cast) state.cast_of = { ...(state.cast_of || {}), [r.id]: r.cast };
  if (!r.ok || !r.persona_ready) return;
  const slug = r.persona_ready, build = state.builds?.[slug] || {};
  if (r.bible) await env.PERSONA.put(`bible:${slug}`, JSON.stringify(r.bible));
  const entry = await personas.get(env, p.slug);
  if (!entry.characters.includes(slug)) entry.characters.push(slug);
  const topic = build.topic || "relationships and self-worth";
  entry.talk_topics = { ...(entry.talk_topics || {}),
    [slug]: { topic, sources: [...TALK_SOURCES.map((h) => `tiktok:@${h}`), `${topic} podcast clip`] } };
  await personas.save(env, entry);
  Object.assign(p, { characters: entry.characters, talk_topics: entry.talk_topics });
  if (state.builds) delete state.builds[slug];
}

/** A roster girl named at the start of `rest` ("Gia: ...", "gia ..."): [slug, rest]. */
async function namedGirl(env, p, rest, pool = null) {
  for (const slug of pool || p.characters || []) {
    const name = ((await st.bible(env, slug))?.name || slug.replace(/^mpt-/, "")).toLowerCase();
    const n = new RegExp(`^${name}\\b[\\s:,-]*`, "i");
    if (n.test(rest)) return [slug, rest.replace(n, "").trim()];
  }
  return [null, rest];
}

/**
 * "video <prompt>" / "carousel <prompt>" in a channel (2026-10-30): a photoshoot of
 * the prompt (a carousel), or one photo of it animated (a video). In MPT a girl's
 * name first picks her ("video Gia: ..."), else one at random. Both wait for a 👍.
 */
async function onMakeRequest(env, p, post, state) {
  const text = String(post.text || "").trim();
  const m = /^(🎬|video|🖼|carousel|carusel|shoot)\b[\s:,-]*(.*)$/isu.exec(text);
  if (!m || post.reply_to_message || CLIP_LINK.test(text)) return false;
  const video = /^(🎬|video)$/iu.test(m[1]);
  let [who, brief] = isRoster(p) ? await namedGirl(env, p, m[2].trim()) : [p.slug, m[2].trim()];
  if (!who) who = p.characters[Math.floor(Math.random() * p.characters.length)];
  if (brief.split(/\s+/).length < 2) {
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
      text: `${video ? "🎬" : "🖼"} Say what you want after it, e.g. "${video ? "video" : "carousel"} ${isRoster(p) ? "Gia: " : ""}rooftop bar at sunset in a red dress".` });
    return true;
  }
  await st.save(env, p.slug, state);  // shoot() loads and saves on its own
  await shoot(env, p, who, { brief: brief.slice(0, 400), video, message_id: post.message_id });
  Object.assign(state, await st.load(env, p.slug));
  const b = (await st.bible(env, who)) || { name: who };
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: video ? `🎬 Queued: ${b.name}, one photo of this, then its video (~30-60 min). Both wait for your 👍.`
                : `🖼 Queued: a ${p.photos || 6}-photo carousel of ${b.name} (~30-60 min). It waits for your 👍 / "post".` });
  return true;
}

/**
 * A talk asked for in the channel: "🎙 <quote>", "talk <quote>", "talk Gia: <quote>",
 * "talk <tiktok/youtube link>" (its words, rewritten). In MPT a girl's name picks her,
 * else one at random; in Lena's channel it is Lena. The talk comes as a reply.
 */
async function onTalkRequest(env, p, post, state) {
  const text = String(post.text || "").trim();
  const m = /^(?:🎙️?|talk\b)\s*(.*)$/isu.exec(text);
  if (!m || post.reply_to_message) return false;
  let rest = m[1].trim();
  let who = null;
  if (isRoster(p)) {
    const pool = p.talkers?.length ? p.talkers : p.characters;
    for (const slug of pool) {
      const name = ((await st.bible(env, slug))?.name || slug.replace(/^mpt-/, "")).toLowerCase();
      const n = new RegExp(`^${name}\\b[\\s:,-]*`, "i");
      if (n.test(rest)) { who = slug; rest = rest.replace(n, "").trim(); break; }
    }
    if (!who) who = pool[Math.floor(Math.random() * pool.length)];
  }
  const url = (/https?:\/\/\S+/.exec(rest) || [])[0] || "";
  const line = url ? "" : rest.replace(/^["“”']+|["“”']+$/g, "").trim();
  if (!url && line.split(/\s+/).length < 4) {
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
      text: "🎙 Send the quote after it: 🎙 <quote>, talk Gia: <quote>, or talk <TikTok link>." });
    return true;
  }
  const msg = queueTalk(p, state, { who: who || p.slug, line, url, message_id: post.message_id });
  const b = (await st.bible(env, who || p.slug)) || { name: who || p.slug };
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: `🎙 Queued: ${b.name} ${url ? "says this video's idea in her words" : "says your quote"}. It comes as a reply here in ~25-40 min.` });
  console.log(msg);
  return true;
}

/**
 * A talk posted: its source is not used again, and with talk_cross it is posted in
 * that persona's channel too (Lena's talks in MPT's, for the aiqube TikTok).
 */
async function talkDone(env, p, state, r, item, sent) {
  if (r.source) state.talk_used = [...(state.talk_used || []), r.source].slice(-300);
  if (r.line) state.talk_lines = [...(state.talk_lines || []), r.line].slice(-80);
  const it = state.items[r.id];
  // The source's own description and account: its caption style and hashtags are copied.
  const src = { source_desc: String(r.source_title || "").slice(0, 600),
                source_account: (/tiktok\.com\/@([\w.]+)/.exec(r.source_url || "") || [])[1] || "" };
  if (it) Object.assign(it, { talk: true, line: r.line || "", topic: r.topic || "", character: r.character || p.slug, bw: !!r.bw, ...src,
                              ...(r.theme_date ? { theme: r.theme || "", theme_date: r.theme_date } : {}),
                              ...(r.hook_shape ? { hook: r.hook || "", hook_shape: r.hook_shape } : {}) });
}

/**
 * Everything of a persona with talk_cross (Lena) is copied into that channel (MPT's),
 * 2026-10-30. A 🎙 talk or 🎭 face swap there publishes to its TikTok by itself;
 * anything else is approved in her own channel (its TikTok draft goes through MPT's
 * Buffer from there), so the copy has no buttons.
 */
async function crossPost(env, p, state, r, item, sent) {
  const to = p.talk_cross && (await personas.get(env, p.talk_cross));
  const media = sent?.video || (sent?.photo || []).slice(-1)[0];
  if (!to || !r?.ok || r.private || !media?.file_id || r.carousel) return;
  const b = (await st.bible(env, r.character || p.slug)) || { name: p.slug };
  const auto = !!(r.talk || r.recreate);
  const m = resultMessage(b, { ...r, chat: to.chat_id, message_id: null, redo: false, replace: null }, item);
  const cap = `${m.caption || ""}${auto ? "" : `\n↪️ from ${b.name}'s channel -- approve it there`}`.slice(0, 1024);
  try {
    const x = await tg(env, sent.video ? "sendVideo" : "sendPhoto", { chat_id: to.chat_id, [sent.video ? "video" : "photo"]: media.file_id,
      caption: cap, ...(auto ? { reply_markup: m.buttons } : {}) });
    const other = await st.load(env, to.slug);
    const src = state.items[r.id] || {};
    if (r.line) other.talk_lines = [...(other.talk_lines || []), r.line].slice(-80);
    other.items[r.id] = { ...src, kind: sent.video ? "video" : "photo", file_id: media.file_id, message_id: x.message_id,
                          character: r.character || p.slug, from: p.slug, ...(r.talk || r.recreate ? { audio: true } : {}) };
    if (auto) await autoPublish(env, to, other, { ...r, ok: true }, x);
    await st.save(env, to.slug, other);
    state.items[r.id] && (state.items[r.id].crossed = true);
  } catch (e) {
    console.error("cross-post failed", String(e).slice(0, 200));
  }
}

/** Her album, copied into her talk_cross channel (no buttons: approved in hers). */
async function crossPostAlbum(env, p, sh, b) {
  const to = p.talk_cross && (await personas.get(env, p.talk_cross));
  if (!to || sh.crossed) return;
  try {
    await tg(env, "sendMediaGroup", { chat_id: to.chat_id, media: sh.media.slice(0, 10).map((m, i) => ({
      type: "photo", media: m.fileId || m.url, ...(i === 0 ? { caption: `👤 ${b.name} · ${sh.media.length} photos\n↪️ from ${b.name}'s channel -- approve it there` } : {}) })) });
    sh.crossed = true;
  } catch (e) {
    console.error("album cross-post failed", String(e).slice(0, 200));
  }
}

// The day the talks became LTX-2 clips adapted from viral shorts (commits 32812fe, c7a8e86):
// the daily report compares what came before and after.
const NEW_TALKS_AT = Date.UTC(2026, 10, 10, 21, 0);

// A question only: answers to statements came out flat ("I think that's super brave...").
const okComment = (t) => t.length >= 12 && t.length <= 200 && /\?/.test(t) && /[a-z]{3}/i.test(t) && !/https?:|@|\b(sex|nude|onlyfans|dm me|follow me|link in)\b/i.test(t);

/**
 * Her TikTok account's numbers, read by the laptop every few hours (persona/ttstats.py:
 * views, shares, saves, top comments of each video). Each video is matched to her talk by
 * its caption. Once a day: the best talk of the last days (2x the median, 1000+ views)
 * gets a Part 2, the best comment gets her answer as a talk, and the channel gets a report
 * (2026-11-10, "do all 4": hook, volume, real fans, proof).
 */
async function laptopStats(request, env) {
  const { videos = [], comments = [] } = await request.json();
  const p = await personas.get(env, "mpt");
  if (!p || !videos.length) return Response.json({ ok: false, why: "no videos" });
  const state = await st.load(env, p.slug);
  const now = Date.now(), day = st.today(), done = [];
  const norm = (t) => String(t || "").toLowerCase().replace(/#\S+/g, "").replace(/[^a-z0-9 ]/g, "").replace(/\s+/g, " ").trim().slice(0, 40);
  // Her posts' TikTok text -> the talk item (its line and who said it).
  const byText = new Map();
  for (const j of state.publish || []) {
    const it = state.items?.[j.item];
    const txt = j.posts?.tiktok?.text || j.caption;
    if (it?.talk && txt) byText.set(norm(txt), { id: j.item, ...it });
  }
  for (const [id, it] of Object.entries(state.items || {})) if (it.talk && it.caption && !byText.has(norm(it.caption))) byText.set(norm(it.caption), { id, ...it });
  const vids = videos.map((v) => ({ ...v, item: byText.get(norm(v.title)) || null }));
  const sorted = vids.map((v) => v.views || 0).sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)] || 0;
  const queueFor = async (who, extra) => {
    // Lena's talks are on this account too (talk_cross): hers are queued in her own lane.
    if (who === "lena") {
      const lp = await personas.get(env, "lena");
      const ls = await st.load(env, "lena");
      done.push(queueTalk(lp, ls, { who, ...extra }));
      await st.save(env, "lena", ls);
    } else done.push(queueTalk(p, state, { who: who || p.characters[Math.floor(Math.random() * p.characters.length)], ...extra }));
  };
  // Part 2 of the best recent talk.
  if (state.follow_day !== day) {
    const age = (v) => now - (v.ts || 0) * 1000;
    const win = vids.filter((v) => v.item?.line && age(v) > 12 * 3600e3 && age(v) < 5 * 86400e3 && (v.views || 0) >= Math.max(1000, 2 * med)
                                   && !(state.followed || []).includes(v.id)).sort((a, b) => b.views - a.views)[0];
    if (win) {
      await queueFor(win.item.character, { follow: win.item.line });
      state.followed = [...(state.followed || []), win.id].slice(-200);
      state.follow_day = day;
    }
  }
  // Her answer to the best comment.
  if (state.reply_day !== day) {
    const c = comments.map((x) => ({ ...x, text: String(x.text || "").trim(), v: {} }))
      .filter((x) => x.cid && okComment(x.text) && !(state.replied || []).includes(x.cid))
      .sort((a, b) => ((b.likes || 0) + (/\?/.test(b.text) ? 5 : 0)) - ((a.likes || 0) + (/\?/.test(a.text) ? 5 : 0)))[0];
    if (c) {
      await queueFor(c.v.item?.character, { reply: c.text, reply_to: c.v.item?.line || "" });
      state.replied = [...(state.replied || []), c.cid].slice(-300);
      state.reply_day = day;
    }
  }
  // The day's report, once, in the afternoon.
  if (state.report_day !== day && new Date().getUTCHours() >= 16) {
    const week = vids.filter((v) => now - (v.ts || 0) * 1000 < 7 * 86400e3);
    const median = (a) => { const s = a.map((v) => v.views || 0).sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };
    const fresh = week.filter((v) => (v.ts || 0) * 1000 >= NEW_TALKS_AT), old = week.filter((v) => (v.ts || 0) * 1000 < NEW_TALKS_AT);
    const k = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n || 0));
    const top = [...week].sort((a, b) => (b.views || 0) - (a.views || 0)).slice(0, 3)
      .map((v, i) => `${i + 1}. ${k(v.views)} views · ${v.shares || 0} shares · ${v.saves || 0} saves\n   ${String(v.title || "").replace(/#\S+/g, "").trim().slice(0, 80)}`);
    // Which hook shape works (2026-11-12): views, shares and saves per shape over 14 days.
    const shapes = {};
    for (const v of vids) if (v.item?.hook_shape && now - (v.ts || 0) * 1000 < 14 * 86400e3) (shapes[v.item.hook_shape] ||= []).push(v);
    const hookLines = Object.entries(shapes).sort((a, b) => median(b[1]) - median(a[1]))
      .map(([s, vs]) => `${s}: ${vs.length} · median ${k(median(vs))} · ${vs.reduce((t, v) => t + (v.shares || 0), 0)} shares · ${vs.reduce((t, v) => t + (v.saves || 0), 0)} saves`);
    const text = `📊 TikTok, last 7 days: ${week.length} videos, median ${k(median(week))} views\n` +
      `New talks (since 10 Nov): ${fresh.length}, median ${k(median(fresh))} · before: ${old.length}, median ${k(median(old))}\n\n` +
      `Top 3:\n${top.join("\n")}${hookLines.length ? `\n\nHooks (14 days):\n${hookLines.join("\n")}` : ""}${done.length ? `\n\nQueued from the numbers:\n${done.join("\n")}` : ""}`;
    await tg(env, "sendMessage", { chat_id: p.chat_id, text: text.slice(0, 4000) }).catch((e) => console.error("report", String(e)));
    state.report_day = day;
  }
  state.tt_stats = { at: now, n: vids.length, median: med, matched: vids.filter((v) => v.item).length };
  await st.save(env, p.slug, state);
  return Response.json({ ok: true, median: med, matched: state.tt_stats.matched, queued: done });
}

/** The line of a talk, rewritten from a popular short's transcript into her words. */
async function talkLine(request, env) {
  return Response.json(await writeTalkLine(env, await request.json()));
}

const TALK_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

/** Talking-clip formats, one per talk at random. 2026-11-06 ("boring... we need more emotional
 * stuff, stuff that touches the heart... make it sound like real talk"): mostly heart-first
 * formats -- a moment she lived, said to one woman -- with a few of the sharp ones kept. */
const TALK_FORMATS = [
  `THE MOMENT -- one small, exact moment she lived about her topic that changed her (where she was, what was said, what she did after), then what she knows now.`,
  `TO THE WOMAN WHO -- "If you're the one who [a quiet thing many women do, about her topic]..." said gently to her, then the one thing she needs to hear.`,
  `WHAT I WISH SOMEONE TOLD ME -- the thing about her topic she wishes someone had told her at 19, through one real memory.`,
  `THE DAY I KNEW -- the exact day it clicked for her (about her topic), the small detail that did it, and the feeling.`,
  `MY MOM SAID -- something her mom or grandma told her about her topic that she only understood years later.`,
  `THE LETTER -- talks straight to her younger self about one specific day, about her topic.`,
  `THE SOFT TRUTH -- "Nobody talks about this, but..." one true feeling women hide about her topic, and a kind way out.`,
  `THE MINI STORY -- "I used to [a specific thing, about her topic]. And I thought that was normal." Then the twist.`,
  `THE CALLOUT -- "If [an exact situation of her topic], that's not [what women tell themselves]. It's [the blunt truth]." Like a friend who's been there.`,
  `HARD TRUTH -- "Hard truth, and I say this with love:" the thing about her topic women know but don't want to hear, in one concrete situation.`,
];

// The on-screen hook (2026-11-06, from the koubo packaging playbook: 0-2s is the claim
// slammed on screen, no hello): big text over the first seconds of a talk, and over a
// whole Reel. Short, so it reads in one glance with the sound off.
const HOOK_SPEC = "the on-screen title for the first 2 seconds: 3-7 words, a confession, a bold claim, a callout or a question " +
  "that makes a woman stop scrolling (like \"The night I stopped waiting\" or \"Men who do this aren't serious\"); " +
  "not the line's first words, no emojis, no hashtags";
function cleanHook(h) {
  const t = String(h || "").replace(/[#*"“”]|\p{Extended_Pictographic}/gu, "").replace(/\s+/g, " ").trim();
  return t.split(" ").length <= 9 ? t.slice(0, 60) : "";
}

/**
 * The on-screen hook, written on its own once the line is chosen (2026-11-10: hooks written
 * with the line were summaries -- "Walking away is freedom"). Proven shapes of the top
 * women's-advice shorts, filled from THIS line; the model picks the one that opens the
 * biggest question. "" when nothing usable comes back (the line's own hook is kept).
 */
// The hook shapes, each one's share/save numbers compared in the daily report (hookReport).
const HOOK_SHAPES = {
  callout: `"If he makes you beg, he's not the one"`,
  warning: `"Stop explaining yourself to people who don't listen"`,
  secret: `"Nobody tells you why silence works on him"`,
  confession: `"The day I stopped begging him to stay"`,
  pov: `"POV: you finally stopped saying sorry first"`,
  judgment: `"Men who can't decide have already decided"`,
  question: `"Why does he come back when you go quiet?"`,
  number: `"3 things a man does when he's serious"`,
  quote: `"If he says 'I'm just busy', read this"`,
  stopped: `"Things I stopped doing at 25"`,
  send: `"Send this to the friend who keeps going back"`,
  save: `"Save this for the night you miss him"`,
};

/** {hook, hook_shape} for a chosen line: the hook writer's, else the line writer's own. */
async function hookOf(env, best) {
  const h = await writeHook(env, best.line);
  return h ? { hook: h.text, hook_shape: h.shape } : { hook: cleanHook(best.hook), hook_shape: "line" };
}

async function writeHook(env, line, prefix = "") {
  const prompt =
    `A woman says this to camera in a TikTok:\n"${line}"\n\n` +
    `Write one on-screen hook for its first 2 seconds in EACH of these shapes. A hook is a COMPLETE short sentence ` +
    `(4-9 words) that makes a woman stop scrolling because she needs the answer, fully filled in from what she says:\n` +
    Object.entries(HOOK_SHAPES).map(([k, ex]) => `- ${k}, e.g. ${ex}`).join("\n") + `\n` +
    `Never a fragment, never a summary or a moral ("Walking away is freedom" is bad), never her first words, no emojis, no hashtags.\n` +
    // 2026-11-11: "Men who cause stress are toxic" over a calm line; the user wants good men shown too.
    `The hook says what HER line says, in the same tone: a warm or loving line gets a warm hook (e.g. "A man who ` +
    `stays on hard days is rare"), a line praising men gets a hook praising men. Never a hook that bashes men in ` +
    `general or blames men when her line doesn't. A number hook only if her line really has that many points.\n` +
    `Then pick the one that would stop the most women scrolling.\n` +
    `Return JSON only: {"hooks": [{"shape": "callout", "text": "..."}, ...], "best": <index>}`;
  try {
    const out = await captions.ask(env, prompt, 700, TALK_MODEL);
    const hooks = (out.hooks || []).map((h) => ({ shape: HOOK_SHAPES[h?.shape] ? h.shape : "other", text: cleanHook(h?.text ?? h) }))
      .filter((h) => h.text);
    // 1 in 3 another shape than the model's favourite, so every shape gets numbers.
    const h = (Math.random() < 1 / 3 && hooks.length > 1 ? hooks[Math.floor(Math.random() * hooks.length)] : hooks[Number(out.best)]) || hooks[0];
    return h ? { text: `${prefix}${h.text}`.slice(0, 70), shape: h.shape } : null;
  } catch (e) {
    console.error("talk hook", String(e).slice(0, 200));
    return null;
  }
}

/**
 * Part 2 of one of her talks that did well, or her answer to a viewer's comment (2026-11-10,
 * "real fans"): the same voice, the next point / a straight answer, 20-45 words.
 */
async function followTalk(env, b, body) {
  const reply = !!body.reply_comment;
  const prompt = reply
    ? `${b.name}, ${b.age || 24}, answers a viewer's comment in a TikTok, talking to camera.\n` +
      (body.reply_to_line ? `Her clip said: "${body.reply_to_line}"\n` : "") +
      `The comment: "${String(body.reply_comment).slice(0, 300)}"\n` +
      `Each answer is a little spoken speech of 4-5 sentences (25-45 words), in this shape:\n` +
      `1. her straight answer or opinion, taking a side;\n2-3. one real moment from her own life that proves it (what ` +
      `happened, what she did);\n4. a short last line that sticks.\n` +
      `She does not read the comment out or say "great question".\n`
    : `${b.name}, ${b.age || 24}, made a TikTok that did very well:\n"${body.follow_line}"\n` +
      `Write Part 2: a little spoken speech of 4-5 sentences (25-45 words) -- the next point on the same idea, what ` +
      `people asked for after part 1, deeper and more specific, same voice and rhythm, ending on a short line that ` +
      `sticks. Do not repeat part 1's sentences.\n`;
  const rules = `Normal punctuation, spoken by a real woman (contractions, plain ` +
    `words). No therapy words (deserve, worth, healing, boundaries, toxic, validate), no hashtags, no emojis, nothing explicit, no names.\n` +
    `Write 4 versions and pick the most natural, punchy and shareable.\n` +
    `Return JSON only: {"best": <index>, "candidates": [{"line": "...", "keywords": ["3-6 single words from the line"], ` +
    `"caption": "a one-line caption that baits comments, max 90 chars, no hashtags"}]}`;
  // Up to 3 tries; a full answer (2+ sentences, 16-55 words) wins, else the longest usable one.
  let spare = null;
  const done = async (c) => ({ line: c.line, keywords: (c.keywords || []).slice(0, 6).map(String), caption: String(c.caption || "").slice(0, 120),
                               hook: reply ? "" : (await writeHook(env, c.line, "Part 2: "))?.text || "" });
  for (let i = 0; i < 3; i++) {
    try {
      const out = await captions.ask(env, prompt + rules, 1000, TALK_MODEL);
      const cands = (out.candidates || []).map((c, k) => ({ ...c, k, line: String(c.line || "").replace(/\s+/g, " ").trim() }))
        .filter((c) => { const n = c.line.split(" ").length; return n >= 8 && n <= 55 && !WORN.test(c.line) && !/#|https?:/.test(c.line); });
      const full = cands.filter((c) => c.line.split(" ").length >= 16 && (c.line.match(/[.!?]/g) || []).length >= 2);
      const best = full.find((c) => c.k === Number(out.best)) || full[0];
      if (best) return done(best);
      for (const c of cands) if (!spare || c.line.length > spare.line.length) spare = c;
    } catch (e) {
      console.error("talk follow", String(e).slice(0, 200));
    }
  }
  return spare ? done(spare) : { line: "" };
}

// What the model keeps writing however it is told not to; a candidate with any is dropped.
const WORN = /\b(deserve|worth|healing|journey|boundar\w*|validat\w*|toxic|self-love|queen|bestie|let that sink|here's the thing|liberating|finding (me|myself)|cry for help|and it hurts|it's okay to)\b/i;

/**
 * Her version of a short that went viral: same opening, rhythm, length, points and kind
 * of ending, in her words. null when the transcript is not one person talking to camera
 * (song lyrics, a skit, a couple's vlog went through as "sources" -- 'GET EM BANNED',
 * 'forever yours @Pili' -- and the line was made up); false when nothing usable came back.
 */
async function adaptTalk(env, b, topic, transcript, views, said, anyTopic) {
  const prompt =
    `This short went viral${views ? ` (${Number(views).toLocaleString("en-US")} views)` : ""}:\n"""${String(transcript).slice(0, 2500)}"""\n\n` +
    `First: is it ONE person giving an opinion, advice or a hard truth to camera, that any woman watching could ` +
    `apply to her own life? If it is song lyrics, a skit, a vlog, chatter or a couple's video, return {"usable": false}. ` +
    // 2026-11-11: a Trisha Paytas podcast confession became "I'm ashamed of what I did, it's hard to sleep."
    `Also {"usable": false} if it is someone's personal story or confession about events in their own life (an ` +
    `interview, a podcast anecdote, a celebrity talking about her past): out of context it means nothing to a viewer. ` +
    `And {"usable": false} if it is addressed TO men ("your girl", "bro", how to win her): she talks to women. ` +
    // 2026-11-11, the user: "not only bad things... we should also say why men are good".
    `What a good man does, why a good man is worth it, what he feels and needs -- all welcome, said to women. ` +
    `Never bash men in general.` +
    (anyTopic ? "" : ` Also return {"usable": false} if it has nothing to do with ${topic} or women's love lives and self-respect.`) + `\n` +
    // Its OWN idea, not bent onto her topic or her hobbies (2026-11-10: "You think I'm rude?" bent to
    // money became "You think I'm cheap? I stopped paying for guys who can't afford me").
    `Otherwise write ${b.name}'s version for her own TikTok (a woman of ${b.age || 24} talking to women). Keep the ` +
    `short's OWN idea and subject -- do not change what it is about.\n` +
    `Keep what made it work: the same rhythm (short punchy fragments stay short punchy fragments), the same points ` +
    `in the same order, the same kind of last line. HARD LIMIT 20-45 words, 3-6 ` +
    `sentences: if the short says more, keep only its opening and its strongest point and cut the rest.\n` +
    `Written with normal punctuation: full stops, commas, question marks (the transcript has none -- add them).\n` +
    // 2026-11-12: 40 talks in a row stopped at ~700 views with 0 shares -- nobody stayed. Her
    // first words are the hook, said out loud, and the answer only comes at the end.
    `HER FIRST SENTENCE IS A SPOKEN HOOK (4-12 words) that opens a question the rest answers -- the viewer must ` +
    `stay to the end to get it. Shapes that work: a callout ("If he only texts you at night, listen."), a number ` +
    `("There are two kinds of men, and one wastes your time."), his exact words in quotes ("If he says 'I'm just ` +
    `busy', hear this."), a secret ("Nobody tells you this about getting over him."), a confession ("I stopped ` +
    `doing one thing and he came running."), a warning ("Stop doing this if you want him to stay."). Never a soft ` +
    `opener ("I think", "Sometimes", "You know"). Keep the short's own opening if it already hooks this hard. The ` +
    `LAST sentence is the payoff: short, the answer, the line to screenshot.\n` +
    `Change the words so it is clearly hers and not a copy: at least half the words different, never the same sentences.\n` +
    `It must sound spoken by a real woman, not written: contractions, plain words. No therapy words (deserve, worth, ` +
    `healing, boundaries, toxic, validate), no hashtags, no emojis, nothing explicit, no names.\n` +
    (said.length ? `She already said these -- a different idea:\n- ${said.slice(-25).join("\n- ")}\n` : "") +
    `Write 4 versions, then pick the one a TikTok editor would post: the most natural, punchy and shareable.\n` +
    `Return JSON only: {"usable": true, "best": <index>, "candidates": [{"line": "...", ` +
    `"keywords": ["3-6 single words from the line to highlight"], "hook": "${HOOK_SPEC}", ` +
    `"caption": "a one-line caption that baits comments (a question or a hot take), max 90 chars, no hashtags"}]}`;
  for (let i = 0; i < 2; i++) {
    try {
      const out = await captions.ask(env, prompt, 1200, TALK_MODEL);
      if (out.usable === false) return null;
      const cands = (Array.isArray(out.candidates) ? out.candidates : [])
        .map((c, k) => ({ ...c, k, line: String(c.line || "").replace(/\s+/g, " ").trim() }))
        .filter((c) => { const n = c.line.split(" ").length; return n >= 18 && n <= 55 && (c.line.match(/[.!?]/g) || []).length >= 2 && !/#|https?:/.test(c.line) && !WORN.test(c.line); });
      // Read cold by a second pass: a line a stranger can't follow never goes out (2026-11-11).
      const best = await judgeTalk(env, cands, Number(out.best));
      if (best) {
        return ({ line: best.line, keywords: (best.keywords || []).slice(0, 6).map(String),
                  ...(await hookOf(env, best)), caption: String(best.caption || "").slice(0, 120) });
      }
    } catch (e) {
      console.error("talk adapt", String(e).slice(0, 200));
    }
  }
  return false;
}

/**
 * Each candidate line read as a stranger scrolling past would hear it, with no context
 * (2026-11-11: "what she said made absolutely no sense"). Scores 1-10 for: makes complete
 * sense on its own, a woman sees her own life in it, she'd send it to a friend. The best
 * one that clears the bar, or null -- then the next source is tried.
 */
async function judgeTalk(env, cands, preferred) {
  if (!cands.length) return null;
  const prompt =
    `You are a strict TikTok editor. A woman says ONE of these lines to camera. The viewer knows nothing else: ` +
    `no story before it, no names, no context.\n` +
    cands.map((c, i) => `${i}. "${c.line}"`).join("\n") +
    `\n\nScore each 1-10 on:\n` +
    `- clear: does it make complete sense on its own -- a clear point a stranger understands the first time? ` +
    `Vague, confusing, missing context, or a story only the speaker understands = low.\n` +
    `- relatable: would most women watching see their own life in it?\n` +
    `- share: would a woman send it to a friend or comment on it?\n` +
    `A line addressed TO men ("your girl", "bro") scores relatable 1. A line that asks the viewer questions instead ` +
    `of making a point scores clear 4 or less.\n` +
    `Return JSON only: {"scores": [{"i": <index>, "clear": n, "relatable": n, "share": n, "why": "few words"}]}`;
  try {
    const out = await captions.ask(env, prompt, 700, TALK_MODEL);
    const ok = (out.scores || []).map((x) => ({ ...x, c: cands[Number(x.i)] }))
      .filter((x) => x.c && x.clear >= 8 && x.relatable >= 7)
      .sort((a, b) => (b.clear + b.relatable + 2 * b.share) - (a.clear + a.relatable + 2 * a.share));
    if (!ok.length) console.log("talk judge: none clear", JSON.stringify(out.scores || []).slice(0, 400));
    return ok[0]?.c || null;
  } catch (e) {
    console.error("talk judge", String(e).slice(0, 200));
    return cands.find((c) => c.k === preferred) || null;
  }
}

async function writeTalkLine(env, body) {
  const { persona, character, topic, transcript, title, given, any_topic } = body;
  if (given) {
    // The user's own quote: said as written; only its keywords and a caption are made.
    let out = {};
    try {
      out = await captions.ask(env, `For this short spoken quote:\n"${String(given).slice(0, 600)}"\n` +
        `Return JSON: {"keywords": ["3-6 single words from it to highlight"], "hook": "${HOOK_SPEC}", "caption": "a one-line TikTok/Instagram caption, max 90 chars, no hashtags"}`, 260);
    } catch (e) { console.error("talk keywords", String(e).slice(0, 200)); }
    return ({ line: String(given).replace(/\s+/g, " ").trim(), keywords: (out.keywords || []).slice(0, 6).map(String),
                           hook: cleanHook(out.hook), caption: String(out.caption || "").slice(0, 120) });
  }
  const p = await personas.get(env, persona || "lena");
  const b = (await st.bible(env, character || persona)) || { name: character };
  // Never the same idea twice: her persona's last lines (every girl of the roster),
  // and the lines Lena's talks posted into it.
  if (body.follow_line || body.reply_comment) return followTalk(env, b, body);
  const said = [...((await st.load(env, p?.slug || persona || "lena")).talk_lines || []).slice(-40)];
  // A short that went viral: its script adapted, not reinvented (2026-11-10, "I don't like
  // most of them"). The rotated story formats below turned "Let him leave. Really. Stop
  // convincing him..." into a rambling hiking story; kept close, it stays what worked.
  if (transcript) {
    const adapted = await adaptTalk(env, b, topic, transcript, body.views, said, any_topic);
    if (adapted) return adapted;
    // The video they sent is used whatever it is (the old way below).
    if (adapted === null && !body.keep) return ({ line: "", why: "not a talk" });
    // Not adapted (nothing kept the rules): the next source, not the old made-up line --
    // that fallback wrote "And it feels amazing. But it's not always easy." (2026-11-10).
    if (adapted === false && !body.keep) return ({ line: "", why: "no good version" });
  }
  // A viral format, rotated so no two talks sound alike (2026-11-05: "boring, it must be viral").
  const format = TALK_FORMATS[Math.floor(Math.random() * TALK_FORMATS.length)];
  const prompt =
    `You write what a real woman says in a TikTok/Reels talking clip that makes women stop, feel it in their chest, ` +
    `and send it to a friend. ${b.name}, ${b.age || 24}, ${b.personality || "warm, honest, a little sassy"}, says it ` +
    `to a close friend on a podcast. Her topic: ${topic}.\n` +
    (transcript ? `A popular short says:\n"""${String(transcript).slice(0, 2500)}"""\n` +
                  (any_topic ? `Take its most moving or most provocative single idea (whatever it is about), in her own words -- do NOT copy its sentences.\n`
                             : `If it has no clear idea about ${topic}, return {"on_topic": false}. Otherwise take its most moving or ` +
                               `provocative single idea, in her own words -- do NOT copy its sentences.\n`)
                : `Pick the feeling on this topic that women carry quietly and rarely say out loud.\n`) +
    (said.length ? `Already said (a DIFFERENT idea, not a rewording of any of these):\n- ${said.join("\n- ")}\n` : "") +
    `Use this format, adapted to her topic (it need not be about a man): ${format} ` +
    `(the words in capitals are only the format's name -- never say them).\n` +
    `She talks TO WOMEN watching. STAY ON HER TOPIC: it decides what the line is about. Bring a man or an ex in only ` +
    `if her topic is about men or relationships -- money is about money, discipline about discipline, friendship about ` +
    `friends, family about family. Never another "if he ..." dating line when the topic is something else.\n\n` +
    `What makes it land -- all of these:\n` +
    `1. HOOK: the first 3-6 words pull you in on their own: a confession, a moment, a callout -- never a soft ` +
    `opener ("I think", "Sometimes", "It's important", "Remember").\n` +
    `2. FEELING: it touches the heart. Something real that hurts or heals, that women live and rarely say. ` +
    `Vulnerable, warm, honest -- the viewer should think "this is me". Find HER own moment: not crying in the shower, ` +
    `not rereading old texts, not "the backup plan" -- those are worn out.\n` +
    `3. CONCRETE: one exact moment she lived -- a place, a time, his exact words in quotes, what she did after. ` +
    `A small specific detail beats any general truth.\n` +
    `4. ENDING: the last sentence is short and lands -- a soft twist, a truth to screenshot, or a line of comfort.\n` +
    `5. REAL TALK, NOT AI: how a normal woman actually talks when it matters to her. Contractions always (I'm, ` +
    `he'd, didn't). Plain everyday words. Short sentences, some starting with "And" or "But". Allowed, sparingly: ` +
    `one "like", "honestly", "I mean", "you know what?", or a small self-correction ("he was -- no, he IS"). NEVER: ` +
    `poetic metaphors, quotable-poster wisdom, therapy words ("healing journey", "self-worth", "boundaries", ` +
    `"validate", "toxic", "seen and heard", "it's okay to not be okay", "deserve", "worth"), "girl/queen/bestie", "In a world", "Let that sink in", "Here's the thing", "Ladies,".\n` +
    `6. FLOW: written to be said in one breath per sentence -- commas, not "..." or dashes for pauses; no lists, ` +
    `no parentheses, no ALL CAPS.\n` +
    `Rules: 22-40 words, 3-5 short sentences (never just one), talks to "you" or tells her own story. Safe for TikTok and ` +
    `Instagram: never explicit, no slurs, no hate, no hashtags, no emojis, no names, no "subscribe"/"follow".\n` +
    `Write 5 different candidates, then score each 1-10, strictly, as a TikTok editor would: hook (does it stop the ` +
    `scroll), feel (does it hit the heart -- would she get chills or tear up), real (does it sound like a real woman ` +
    `talking, not written; any AI-sounding phrase gets 3 or less), share (will women send it or save it). Anything ` +
    `safe, vague, preachy or obvious gets 4 or less.\n` +
    `Return JSON: {"on_topic": true, "candidates": [{"line": "...", "hook": 8, "feel": 9, "real": 8, "share": 8, ` +
    `"keywords": ["3-6 single words from the line to highlight"], "hook": "${HOOK_SPEC}", ` +
    `"caption": "a one-line TikTok/Instagram caption ` +
    `that baits comments (a question or a hot take), max 90 chars, no hashtags"}]}`;
  for (let i = 0; i < 3; i++) {
    try {
      // A bigger model for the line people hear (2026-11-05: the small one's were bland or
      // garbled); the small one if it fails.
      const out = await captions.ask(env, prompt, 1400, i < 2 ? TALK_MODEL : undefined);
      if (transcript && !any_topic && out.on_topic === false) return ({ line: "", why: "off topic" });
      // The best-scored candidate that keeps the rules (one plain answer from older replies).
      const cands = (Array.isArray(out.candidates) ? out.candidates : [out])
        .map((c) => ({ ...c, line: String(c.line || "").replace(/\s+/g, " ").trim() }))
        .filter((c) => { const n = c.line.split(" ").length; return n >= 14 && n <= 48 && !/#|https?:/.test(c.line) && !WORN.test(c.line); })
        .map((c) => ({ ...c, score: Number(c.score) || ["hook", "feel", "real", "share"].reduce((t, k) => t + (Number(c[k]) || 0), 0) / 4 }))
        // Hook (does anyone hear the rest) and feel (the ask of 2026-11-06) count double.
        .map((c) => ({ ...c, rank: ["hook", "feel"].reduce((t, k) => t + (Number(c[k]) || c.score) * 2, 0) +
                                   (Number(c.real) || c.score) + (Number(c.share) || c.score) }))
        .sort((a, b) => b.rank - a.rank);
      const best = cands[0];
      if (best && best.score >= 6) {
        return ({ line: best.line, keywords: (best.keywords || []).slice(0, 6).map(String),
                  ...(await hookOf(env, best)), caption: String(best.caption || "").slice(0, 120) });
      }
    } catch (e) {
      console.error("talk line", String(e).slice(0, 200));
    }
  }
  return ({ line: "" });
}

const PUBLISH_WORDS = /^\s*(👍|post|publish|post it)\s*[.!]*\s*$/iu;
const BLOCK_WORDS = /^\s*(🗑️?|❌|delete|remove|skip|block|bad|don'?t post|do not post|never post)\s*[.!]*\s*$/iu;

/**
 * "delete" in reply to a post (2026-10-30: a bad video was in the Facebook backfill):
 * it is never posted -- every publish job of it that has not gone out yet is dropped,
 * here and in the channel it was copied to or from, and it is blocked from backfills
 * and Stories. Returns how many jobs were dropped.
 */
async function blockItem(env, p, state, itemId) {
  const drop = (st_) => {
    st_.blocked = [...new Set([...(st_.blocked || []), itemId])].slice(-500);
    const before = st_.publish.length;
    st_.publish = st_.publish.filter((j) => !(j.item === itemId && !["ig", "tt", "fb"].some((k) => j[k]?.status === "done" && j[k]?.mode !== "phone")));
    return before - st_.publish.length;
  };
  let n = drop(state);
  for (const slug of [p.talk_cross, state.items[itemId]?.from].filter(Boolean)) {
    const other = await st.load(env, slug);
    n += drop(other);
    await st.save(env, slug, other);
  }
  for (const other of await personas.list(env)) {
    if (other.talk_cross !== p.slug) continue;
    const o = await st.load(env, other.slug);
    if (!o.items[itemId]) continue;
    n += drop(o);
    await st.save(env, other.slug, o);
  }
  return n;
}
// Any short reply mentioning a carousel ("carousel please", "make carousel 🖼", ...).
const CAROUSEL_WORDS = /^(?=.{0,60}$).*(🖼|car+ou?s+el+|carusel)/isu;

/**
 * MPT: new jobs belong to the girl of the photo they came from; a 🎭 link with no
 * photo goes to the next girl in rotation. A video sent by hand never leaves the
 * laptop (local_only), like every hand-sent media in the MPT channels.
 */
function tagCharacters(p, state, added, post) {
  for (const q of added) {
    if (q.character) continue;
    const parentId = post?.reply_to_message ? itemIdFrom(post.reply_to_message) : "";
    // (A 🎭 with `fit` gets its girl on the laptop; this one is Kaggle's fallback.)
    q.character = state.items[q.item]?.character || state.items[parentId]?.character || nextCharacter(p, state);
    if (q.kind === "recreate" && q.file_id) q.local_only = true;
  }
}

function nextCharacter(p, state) {
  const i = state.rotation || 0;
  state.rotation = i + 1;
  return p.characters[i % p.characters.length];
}

/** A photo uploaded into her channel: 2 more shots of it for a 🖼 carousel. */
async function onPhotoUpload(env, post, state) {
  const itemId = `msg-${post.message_id}`;
  const fileId = post.photo.slice(-1)[0].file_id;
  state.items[itemId] = { kind: "image", tags: [], scene: {}, caption: post.caption || "", file_id: fileId };
  state.queue.push({ id: `${itemId}-car${Date.now()}`, kind: "carousel", item: itemId, scene: {}, file_id: fileId, count: 3,
                     chat: post.chat.id, message_id: post.message_id });
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: "🖼 Making 2 more shots of this photo for a carousel (~5-10 min). Only photos of her pass the face check." });
}

async function onReply(env, p, post, state, origin) {
  const parent = post.reply_to_message;
  if (await albumRedo(env, p, post, state)) return;
  // Any other reply to a shoot album is not a "how she moves" note for a video
  // (2026-10-19: a "redo" sent before albumRedo existed made a video of photo 1).
  if (albumOf(state, parent)) {
    const a = albumOf(state, parent), sh = a && state.shoots?.[a.shoot];
    if (sh) await albumStatus(env, p, sh, "💡 reply: post · redo 3 <note> · video 3 <how she moves>", state);
    await tidy(env, post);
    return;
  }
  // Replies to a single post (Lena's channel, and any old single photos): one word
  // says what to do -- the same words as on MPT albums (2026-10-20: any other text
  // used to become a video's motion prompt, so "redo 3 ..." made a video).
  const text = post.text.trim();
  const itemId = itemIdFrom(parent) || `msg-${parent.message_id}`;
  const photo = (parent.photo || []).slice(-1)[0];
  if (CAROUSEL_WORDS.test(text) && photo) {
    state.items[itemId] = { kind: "image", tags: [], scene: {}, caption: "", ...(state.items[itemId] || {}), file_id: photo.file_id };
    const busy = state.queue.some((q) => q.kind === "carousel" && q.item === itemId);
    if (!busy) {
      state.queue.push({ id: `${itemId}-car${Date.now()}`, kind: "carousel", item: itemId, scene: state.items[itemId].scene || {},
                         file_id: photo.file_id, count: 3, chat: post.chat.id, message_id: parent.message_id });
    }
    await noteOnPost(env, parent, busy ? "🖼 already making this carousel" : "🖼 making 3 more angles for a carousel (~10-20 min)");
    return tidy(env, post);
  }
  if (BLOCK_WORDS.test(text) && (parent.photo || parent.video)) {
    const n = await blockItem(env, p, state, itemId);
    await noteOnPost(env, parent, `🗑 won't be posted anywhere${n ? ` (${n} pending post${n > 1 ? "s" : ""} cancelled)` : ""}`);
    return tidy(env, post);
  }
  if (PUBLISH_WORDS.test(text) && (parent.photo || parent.video)) {
    const res = await queuePublish(env, p, state, { message: parent }, itemId);
    if (res === true && !state.voted.includes(itemId)) state.voted.push(itemId);
    if (res !== true) await noteOnPost(env, parent, `⚠️ not published: ${res || "no Instagram/TikTok linked"}`);
    if (res === true) await advancePublish(env, p, state, origin);
    return tidy(env, post);
  }
  const redo = /^\s*(?:redo|🔁)\b\s*(.*)$/isu.exec(text);
  if (redo && parent.video) {
    const res = redoVideo(p, state, itemId, parent);
    await noteOnPost(env, parent, res === true ? "🔁 making it again -- the new one comes as a reply" : `⚠️ ${res}`);
    return tidy(env, post);
  }
  if (redo && photo) {
    const note = redo[1].trim().slice(0, 300);
    const scene = state.items[itemId]?.scene;
    if (!scene?.setting) {
      await noteOnPost(env, parent, "⚠️ can't redo this one: its scene wasn't recorded (older post)");
      return tidy(env, post);
    }
    if (!state.queue.some((q) => q.kind === "redo" && q.item === itemId)) {
      learnFrom(state, state.items[itemId], -1);
      state.queue.push({ id: `${itemId}-r${Date.now()}`, kind: "redo", item: itemId, scene: { ...scene, ...(note ? { note } : {}) },
                         chat: post.chat.id, message_id: parent.message_id });
    }
    await noteOnPost(env, parent, `🔁 redrawing${note ? ` (${note.slice(0, 60)})` : ""} -- it replaces this photo (~15-30 min)`);
    return tidy(env, post);
  }
  const video = /^\s*(?:video|🎬)\b\s*(.*)$/isu.exec(text);
  if (video && photo) {
    queueVideo(state, itemId, parent, photo.file_id, video[1].trim());
    await noteOnPost(env, parent, "🎬 making a video -- it comes as a reply (~40-60 min)");
    return tidy(env, post);
  }
  // Anything else is left alone (a note to yourself, a question): nothing happens.
}

/** A one-line status at the end of a post's caption, its buttons kept. */
async function noteOnPost(env, msg, line) {
  const base = (msg.caption || "").replace(/\n(?:🔁|🎬|🖼|⚠️|📤|📥|⏳|❌)[^\n]*$/u, "");
  await tg(env, "editMessageCaption", { chat_id: msg.chat.id, message_id: msg.message_id,
    caption: `${base}\n${line}`.slice(0, 1024), ...(msg.reply_markup ? { reply_markup: msg.reply_markup } : {}) }).catch(() => {});
}

export async function onPersonaWebhook(request, env, ctx) {
  const u = new URL(request.url);
  if (u.pathname === "/persona/ig/webhook") return instagram.onWebhook(request, env, ctx);
  if (u.pathname === "/persona/privacy") {
    // The privacy policy URL Meta asks for before an app goes Live.
    return new Response(`<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Privacy</title>
<body style="font:16px/1.5 system-ui;max-width:640px;margin:40px auto;padding:0 16px">
<h1>Privacy policy</h1><p>This app answers comments and direct messages on its owner's own Instagram account.
It reads the text of a comment or message only to write a reply, and stores nothing but the id of what it answered
(for 30 days, so nothing is answered twice). No data is sold or shared. To have anything removed, message the account.</p>`,
      { headers: { "content-type": "text/html; charset=utf-8" } });
  }
  if (u.pathname === "/persona/file") return serveFile(request, env);
  if (u.pathname.startsWith("/persona/media/")) return serveMedia(request, env);
  if (u.pathname === "/persona/share") return sharePage(request, env);
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

/** Apply taps and replies (from the webhook, or polled), each in its persona's channel. */
async function handleUpdates(env, updates, origin) {
  const byChat = new Map((await personas.list(env)).map((p) => [p.chat_id, p]));
  const groups = new Map();
  for (const update of updates) {
    const cq = update.callback_query;
    const post = update.channel_post || update.message;
    const rc = update.message_reaction_count;
    const chatId = String(cq?.message?.chat?.id ?? post?.chat?.id ?? rc?.chat?.id ?? "");
    const p = byChat.get(chatId);
    console.log("update", update.update_id, cq ? `tap ${cq.data}` : post?.photo ? "photo" : post?.video ? "video" : post?.text ? `text "${post.text.slice(0, 40)}"` : "other",
                "chat", chatId, post?.chat?.title || cq?.message?.chat?.title || "", p ? `-> ${p.slug}` : "-> no persona channel, ignored");
    if (!p) continue;
    if (!groups.has(p.slug)) groups.set(p.slug, { p, list: [] });
    groups.get(p.slug).list.push(update);
  }
  for (const { p, list } of groups.values()) {
    const state = await st.load(env, p.slug);
    let changed = false;
    for (const update of list) {
      const cq = update.callback_query;
      const post = update.channel_post || update.message;
      const queued = state.queue.length;
      try {
        // MPT: photos sent by hand stay private (never processed in the cloud).
        if (isRoster(p) && post?.photo && !post.reply_to_message && !post.reply_markup) continue;
        if (update.message_reaction_count) { if (await onReaction(env, p, update.message_reaction_count, state, origin)) changed = true; }
        else if (cq?.data?.startsWith("pv:")) { await onTap(env, p, cq, state, origin); changed = true; }
        // A TikTok/Facebook link or video counts as 🎭 even when it is sent as a reply.
        // "🎙 <quote>" / "talk Gia: <quote>" / "talk <tiktok link>": a talk on request.
        else if (post && (await onStatusRequest(env, p, post))) continue;
        else if (post && isRoster(p) && (await onPrivateRequest(env, p, post, state))) changed = true;
        else if (post && isRoster(p) && (await onBuildRequest(env, p, post, state))) changed = true;
        else if (post && (await onMakeRequest(env, p, post, state))) changed = true;
        else if (post && (await onTalkRequest(env, p, post, state))) changed = true;
        else if (post && (await onRecreateRequest(env, post, state, p))) changed = true;
        // A photo uploaded into her channel (not by the bot): 🖼 a carousel of it.
        // (Bot posts, aibeauty's included, carry buttons; a hand upload has none.)
        else if (post?.photo && !post.reply_to_message && !post.via_bot && !post.reply_markup) {
          await onPhotoUpload(env, post, state); changed = true;
        }
        else if (post?.text && post.reply_to_message && !post.text.startsWith("/")) {
          await onReply(env, p, post, state, origin); changed = true;
        }
        if (isRoster(p)) tagCharacters(p, state, state.queue.slice(queued), post);
      } catch (e) {
        // One bad tap must not take the others down with it, and must not vanish:
        // on 2026-10-14 three 👍 taps were dropped without a trace.
        console.error("update failed", update.update_id, String(e?.stack || e).slice(0, 500));
        const msg = cq?.message || post;
        await tg(env, "sendMessage", { chat_id: p.chat_id, reply_to_message_id: msg?.message_id,
          text: `⚠️ Couldn't handle that ${cq ? "tap" : "message"}: ${String(e?.message || e).slice(0, 300)}` }).catch(() => {});
      }
    }
    if (!changed) continue;
    if (state.publish.some((j) => !social.finished(j))) await advancePublish(env, p, state, origin);
    await st.save(env, p.slug, state);
    // Start the job right away if Kaggle is idle, instead of waiting for the cron.
    if (state.queue.length) await advanceJobs(env, p, origin);
  }
}

/**
 * Until the persona bot has a webhook (PERSONA_POLL=1), its taps are collected
 * here every 15 minutes (and every 30s while the laptop agent runs).
 */
async function poll(env, origin) {
  const updates = await tg(env, "getUpdates", { timeout: 0, allowed_updates: ["callback_query", "channel_post", "message"] });
  if (!updates.length) return "no updates";
  // Handled first, marked read after: if handling dies (e.g. saving state fails),
  // the taps come back at the next poll instead of being lost.
  await handleUpdates(env, updates, origin);
  await tg(env, "getUpdates", { timeout: 0, offset: updates[updates.length - 1].update_id + 1 });
  return `polled ${updates.length}`;
}

// ------------------------------------------------------------------- admin
async function onAdmin(request, env) {
  if (request.headers.get("Authorization") !== `Bearer ${env.PERSONA_ADMIN_SECRET}`) {
    return new Response("forbidden", { status: 403 });
  }
  const u = new URL(request.url);
  const slug = u.searchParams.get("slug") || env.PERSONA_SLUG || "lena";
  const p = (await personas.get(env, slug)) || { slug, chat_id: "" };
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
  if (u.pathname === "/persona/admin/personas") {
    if (request.method === "POST") return Response.json(await personas.save(env, await request.json()));
    const all = await personas.list(env);
    for (const q of all) q.accounts = social.accounts(q);
    return Response.json(all);
  }
  if (u.pathname === "/persona/admin/scenes") {
    // A dry run of the daily scene writer: nothing is saved or rendered.
    const b = await st.bible(env, slug);
    return Response.json(await scenes.write(env, b, await st.load(env, slug), Number(u.searchParams.get("n") || 2)));
  }
  if (u.pathname === "/persona/admin/caption") {
    // A dry run: what a 👍 on this item would post, nothing is published.
    const state = await st.load(env, slug);
    const item = state.items[u.searchParams.get("item")] || { kind: u.searchParams.get("kind") || "video", recreate: true };
    const b = (await st.bible(env, slug)) || { name: slug };
    return Response.json(await captions.forPost(env, p, b, state, item));
  }
  if (u.pathname === "/persona/admin/tiktok" && request.method === "POST") {
    // {refresh_token} for this persona's TikTok account, or {verify_name, verify_body}
    // for TikTok's URL-prefix verification file.
    const t = await request.json();
    if (t.refresh_token) await env.PERSONA.put(`tt:${slug}`, JSON.stringify({ refresh_token: t.refresh_token }));
    if (t.verify_name) await env.PERSONA.put(`ttverify:${t.verify_name}`, t.verify_body || "");
    return new Response("ok");
  }
  if (u.pathname === "/persona/admin/igtoken" && request.method === "POST") {
    // Her long-lived Instagram token (comment and DM replies, instagram.js); never returned.
    return Response.json(await instagram.setToken(env, slug, (await request.text()).trim()));
  }
  if (u.pathname === "/persona/admin/buffer-token" && request.method === "POST") {
    // Her own Buffer account's API key; never returned by any endpoint.
    await env.PERSONA.put(`buffertoken:${slug}`, (await request.text()).trim());
    return new Response("ok");
  }
  if (u.pathname === "/persona/admin/tags") {
    // POST {trend: [...]} replaces her hand-kept trending list; GET shows her tag scores.
    if (request.method === "POST") {
      const t = await request.json();
      if (t.trend) await env.PERSONA.put(`trend_tags:${slug}`, JSON.stringify(t.trend));
      if (t.core) await env.PERSONA.put(`coretags:${slug}`, JSON.stringify(t.core));
      return new Response("ok");
    }
    const state = await st.load(env, slug);
    return Response.json({ context: state.context || {}, core: JSON.parse((await env.PERSONA.get(`coretags:${slug}`)) || "null"),
                           trend: JSON.parse((await env.PERSONA.get(`trend_tags:${slug}`)) || "[]"),
                           scores: state.hashtags || {} });
  }
  if (u.pathname === "/persona/admin/status") return new Response(await dayStatus(env));
  if (u.pathname === "/persona/admin/run" && request.method === "POST") {
    const what = u.searchParams.get("what");
    if (what === "render") return new Response(await render(env, p, u.searchParams.get("force") === "1"));
    if (what === "post") return new Response(await post(env, p, Number(u.searchParams.get("slot") || 4)));
    if (what === "jobs") return new Response(await advanceJobs(env, p, u.origin));
    if (what === "shoot") return new Response(await shoot(env, p, u.searchParams.get("character")));
    if (what === "albumcmd") {
      // An album reply sent from here (?shoot=&text=redo 5 / video 1 2), as if typed under it.
      const state = await st.load(env, slug);
      const shoot = u.searchParams.get("shoot");
      if (!state.shoots?.[shoot]) return new Response("unknown shoot", { status: 404 });
      // &reset=1: a redo that died on the laptop (out of memory) left the album waiting for it.
      if (u.searchParams.get("reset")) Object.assign(state.shoots[shoot], { redo_wait: 0, redo_failed: [] });
      const done = await albumRedoCommands(env, p, { text: u.searchParams.get("text") || "", chat: { id: p.chat_id }, message_id: 0 },
                                           state, { shoot, item: null });
      await st.save(env, slug, state);
      return Response.json({ done, queue: state.queue.map((q) => q.id) });
    }
    if (what === "drop") {
      // Messages taken out of her channel (?ids=1,2,3), and a dropped shoot forgotten
      // (&shoot=), e.g. a bad album and its Reels.
      const ids = (u.searchParams.get("ids") || "").split(",").map(Number).filter(Boolean);
      const gone = [];
      for (const id of ids) if (await tg(env, "deleteMessage", { chat_id: p.chat_id, message_id: id }).then(() => true).catch(() => false)) gone.push(id);
      const shoot = u.searchParams.get("shoot");
      if (shoot) {
        const state = await st.load(env, slug);
        delete state.shoots?.[shoot];
        for (const [mid, a] of Object.entries(state.album_msgs || {})) if (a.shoot === shoot) delete state.album_msgs[mid];
        await st.save(env, slug, state);
      }
      return Response.json({ deleted: gone });
    }
    if (what === "albumswap") {
      // A redo that came back as a photo of its own: put it into its album
      // (?shoot=&old=&new=, and &drop=<message id> of the stray photo).
      const state = await st.load(env, slug);
      const [shoot, oldId, newId] = ["shoot", "old", "new"].map((k) => u.searchParams.get(k));
      const sh = state.shoots?.[shoot], it = state.items[newId];
      if (!sh || !it?.file_id || !(sh.ids || []).includes(oldId)) return new Response("unknown shoot/photo", { status: 404 });
      sh.redo_wait = 1;
      await albumRedoDone(env, p, state, { ok: true, shoot, replaces: oldId, id: newId, request: "albumswap", how: "redrawn" },
                          { ...it, id: newId }, it.file_id);
      if (u.searchParams.get("drop")) await tg(env, "deleteMessage", { chat_id: p.chat_id, message_id: Number(u.searchParams.get("drop")) }).catch(() => {});
      await st.save(env, slug, state);
      return new Response(`album ${sh.album}`);
    }
    if (what === "albumbuttons") {
      // Albums posted before they had a button message: buttons now (album photos
      // have consecutive message ids from the album's first).
      const state = await st.load(env, p.slug), done = [];
      for (const [id, sh] of Object.entries(state.shoots || {})) {
        if (typeof sh.album !== "number" || sh.buttons || !sh.media?.length) continue;
        state.album_msgs = state.album_msgs || {};
        sh.media.forEach((m, i) => {
          state.album_msgs[sh.album + i] = { shoot: id, item: m.id };
          if (state.items[m.id]) state.items[m.id].message_id = sh.album + i;
        });
        await albumButtons(env, p, sh);
        done.push(id);
      }
      await st.save(env, p.slug, state);
      return Response.json(done);
    }
    if (what === "buffermutations") {
      const t = u.searchParams.get("type");
      return Response.json(await social.raw(env, p, t ? `{ __type(name: "${t.replace(/[^\w]/g, "")}") { inputFields { name type { name kind ofType { name } } } } }`
                                                      : `{ __schema { mutationType { fields { name args { name type { name ofType { name } } } } } } }`, {}));
    }
    if (what === "bufferposts") {
      // Every post in her Buffer channel (any status), to find what fills the queue.
      const org = (await social.raw(env, p, `{ account { organizations { id } } }`, {}))?.account?.organizations?.[0]?.id;
      // &channel=facebook|tiktok (default instagram), &after=<cursor> for the next 100.
      const ch = p.buffer?.[u.searchParams.get("channel") || "instagram"];
      return Response.json(await social.raw(env, p, `query($i: PostsInput!, $a: String) { posts(input: $i, first: 100, after: $a) { pageInfo { hasNextPage endCursor } edges { node { id status dueAt createdAt text assets { source } } } } }`,
        { i: { organizationId: org, filter: { channelIds: [ch].filter(Boolean) } }, a: u.searchParams.get("after") || null }));
    }
    if (what === "attach") {
      // A publish job whose Buffer post exists but the job lost it: linked again (its
      // links refreshed by the next recheck), so it stops making new copies.
      const state = await st.load(env, p.slug);
      const job = state.publish.find((j) => j.id === u.searchParams.get("j"));
      if (!job) return new Response("no such job", { status: 404 });
      job.ig = { mode: "queued", status: "done", post: u.searchParams.get("post"), sent_at: Date.now() - 7 * 3600 * 1000, links: 1 };
      delete job.retry_at;
      await st.save(env, p.slug, state);
      return new Response("attached");
    }
    if (what === "bufferdelete") {
      // ?id= one post, or ?ids=a,b,c (at most 40: the Worker's subrequest limit).
      const ids = (u.searchParams.get("ids") || u.searchParams.get("id") || "").split(",").filter(Boolean).slice(0, 40);
      const out = {};
      for (const id of ids) out[id] = await social.raw(env, p, `mutation($i: DeletePostInput!) { deletePost(input: $i) { __typename } }`, { i: { id } });
      return Response.json(ids.length === 1 ? out[ids[0]] : out);
    }
    if (what === "bufferpost") {
      return Response.json(await social.raw(env, p, `query($input: PostInput!) { post(input: $input) { id status dueAt sentAt error { message } } }`,
                                             { input: { id: u.searchParams.get("id") } }));
    }
    if (what === "postnow") {
      const state = await st.load(env, p.slug);
      const job = state.publish.find((j) => j.id === u.searchParams.get("j"));
      if (!job?.ig?.post) return new Response("no such queued post", { status: 404 });
      const url = job.file_ids ? await Promise.all(job.file_ids.map((f) => mediaLink(env, u.origin, f, "jpg")))
        : await mediaLink(env, u.origin, job.file_id, job.kind === "video" ? "mp4" : "jpg");
      if (u.searchParams.get("at")) await social.moveTo(env, p, job, "ig", url, u.searchParams.get("at"));
      else await social.sendNow(env, p, job, "ig", url);
      await st.save(env, p.slug, state);
      return new Response("sent now");
    }
    if (what === "enhance") {
      // The prompt enhancer's scenes for &brief= and &who= (no render), to check it.
      const who = u.searchParams.get("who") || p.slug, n = Number(u.searchParams.get("n") || 1);
      const b = (await st.bible(env, who)) || { name: who };
      return Response.json(await scenes.writeShoot(env, b, await st.load(env, p.slug), n,
        { sexy: isRoster(p) ? true : p.style || "flirty", brief: u.searchParams.get("brief") || "" }));
    }
    if (what === "fbbackfill") {
      // Her Facebook Page filled with her 👍'd photos and videos (2026-10-30, a new
      // Page): &n= posts, &per_day= a day at 09/14/19 Prague time, Facebook only.
      const state = await st.load(env, p.slug);
      if (!p.buffer?.facebook) return new Response("no facebook channel", { status: 400 });
      const n = Number(u.searchParams.get("n") || 12), perDay = Number(u.searchParams.get("per_day") || 3);
      const hours = [7, 12, 17].slice(0, perDay);  // UTC = 09/14/19 Prague (UTC+2)
      const done = new Set([...state.publish.filter((j) => j.fb || j.fb_only).map((j) => j.item), ...(state.blocked || [])]);
      const picks = [...state.voted].reverse().map((id) => [id, state.items[id]])
        // Photos and 🎭 face swaps (they keep the original's sound); never a silent video.
        .filter(([id, it]) => it?.file_id && !it.talk && !done.has(id) &&
                ((it.kind || "photo") === "photo" || (it.kind === "video" && (it.audio || it.recreate)))).slice(0, n);
      const b = (await st.bible(env, p.slug)) || { name: p.slug };
      let t = Date.now(), made = [];
      for (const [id, it] of picks) {
        // The next free hour of the slots.
        do { t += 3600 * 1000; } while (!hours.includes(new Date(t).getUTCHours()));
        const due = Date.UTC(new Date(t).getUTCFullYear(), new Date(t).getUTCMonth(), new Date(t).getUTCDate(), new Date(t).getUTCHours());
        const posts = await captions.forPost(env, p, b, state, it);
        state.publish.push({ id: `pub-fb-${id}-${Date.now()}`, item: id, kind: it.kind || "photo", created: Date.now(), due,
          file_id: it.file_id, audio: !!it.audio || !!it.recreate, posts, caption: posts.instagram.text, fb_only: true, quiet: true,
          ig: { status: "skipped", note: "Facebook backfill" }, tt: { status: "skipped", note: "Facebook backfill" } });
        made.push([id, new Date(due).toISOString().slice(0, 16)]);
      }
      await st.save(env, p.slug, state);
      return Response.json(made);
    }
    if (what === "postitem") {
      // One posted item (&item=) through Buffer now, as its 👍 would (a vote made before
      // the persona's Buffer channel was linked).
      const state = await st.load(env, p.slug);
      const id = u.searchParams.get("item"), it = state.items[id];
      if (!it?.file_id) return new Response("unknown item", { status: 404 });
      // &replace=1: forget its earlier publish job (its Buffer post deleted separately).
      if (u.searchParams.get("replace")) state.publish = state.publish.filter((x) => x.item !== id);
      const cq = { message: { chat: { id: p.chat_id }, message_id: it.message_id,
                              ...(it.kind === "video" ? { video: { file_id: it.file_id } } : {}) } };
      const res = await queuePublish(env, p, state, cq, id);
      if (res === true) await advancePublish(env, p, state, u.origin);
      await st.save(env, p.slug, state);
      const j = state.publish.find((x) => x.item === id);
      return Response.json({ res, caption: j?.posts?.tiktok?.text, tt: j?.tt });
    }
    if (what === "publish") {
      // Move every publish job along now (incl. the Buffer queue recheck).
      const state = await st.load(env, p.slug);
      if (u.searchParams.get("force")) for (const j of state.publish) for (const k of ["ig", "tt"]) if (j[k]?.mode === "queued" && !j[k].final) { delete j[k].checked_at; j[k].sent_at = Math.min(j[k].sent_at || Date.now(), Date.now() - 7 * 3600 * 1000); }
      const changed = await advancePublish(env, p, state, u.origin);
      if (changed) await st.save(env, p.slug, state);
      return Response.json(state.publish.slice(-12).map((j) => ({ id: j.id, retry_at: j.retry_at, ig: j.ig && { mode: j.ig.mode, status: j.ig.status, links: j.ig.links, resent: j.ig.resent, err: j.ig.recheck_error, note: j.ig.note } })));
    }
    if (what === "prefs") {
      const prefs = (await st.load(env, p.slug)).prefs || {};
      return Response.json(Object.fromEntries(Object.entries(prefs).sort((a, b) => b[1] - a[1])));
    }
    if (what === "sendtiktok") {
      // The same as replying "post" to the album (also after a redo).
      const state = await st.load(env, p.slug), sh = state.shoots?.[u.searchParams.get("id")];
      if (!sh) return new Response("no such shoot", { status: 404 });
      Object.assign(sh, { send_tiktok: true, draft: null, draft_error: null, retry_at: null, urls: null, changed_after_draft: false });
      sh.tiktok_status = "📤 TikTok: sending to the aiqube inbox (~15-45 min)";
      await albumStatus(env, p, sh, sh.tiktok_status);
      await st.save(env, p.slug, state);
      return new Response("sending");
    }
    if (what === "sharelink") {
      const state = await st.load(env, p.slug);
      const job = state.publish.find((j) => j.id === u.searchParams.get("j"));
      return new Response(job ? await shareLink(env, u.origin, p, job) : "no such job", { status: job ? 200 : 404 });
    }
    if (what === "writeshoot") return new Response(String(await writeRequestedShoot(env, p)));
    if (what === "abtest") {
      // 🎙 colour vs black and white: Instagram numbers of posted talks (Buffer), TikTok drafts sent.
      const state = await st.load(env, p.slug);
      const rows = [];
      for (const j of state.publish.filter((x) => x.talk && x.ig?.post)) {
        const post = await social.metrics(env, p, j.ig.post).catch(() => null);
        rows.push({ bw: j.bw, created: new Date(j.created).toISOString().slice(0, 10),
                    metrics: Object.fromEntries((post?.metrics || []).map((m) => [m.type, m.value])) });
      }
      return Response.json({ instagram: rows });
    }
    if (what === "reslot") {
      // Waiting posts packed into the earliest free slots again, oldest first; a waiting
      // Story goes now (2026-11-05: Stories had held feed slots, a 4-day backlog).
      const state = await st.load(env, p.slug);
      const moved = [];
      const wait = state.publish.filter((j) => !j.ig && j.due > Date.now()).sort((a, b) => a.created - b.created);
      for (const j of wait) delete j.due;
      for (const j of wait) {
        j.due = j.story || silentReel(j) ? Date.now() : nextSlot(env, state);
        moved.push(`${j.id} ${j.story ? "story" : silentReel(j) ? "silent reel (phone)" : j.kind} -> ${new Date(j.due).toISOString().slice(5, 16)}`);
      }
      await advancePublish(env, p, state, u.origin);
      await st.save(env, p.slug, state);
      return Response.json(moved);
    }
    if (what === "talk") {
      // A 🎙 talk now (&who= a roster girl, else the next in rotation).
      const state = await st.load(env, p.slug);
      const who = u.searchParams.get("who");
      const msg = who ? queueTalk({ ...p, talkers: [who] }, state) : queueTalk(p, state);
      await st.save(env, p.slug, state);
      return new Response(msg);
    }
    if (what === "dropshoot") {
      // A shoot that will never finish (its job is gone): removed from the state.
      const p = await personas.get(env, u.searchParams.get("slug") || "mpt");
      const state = await st.load(env, p.slug);
      const id = u.searchParams.get("id");
      if (!state.shoots?.[id] || state.shoots[id].finished) return new Response("no such unfinished shoot", { status: 404 });
      delete state.shoots[id];
      state.queue = state.queue.filter((q) => q.shoot !== id);
      await st.save(env, p.slug, state);
      return new Response(`dropped ${id}`);
    }
    if (what === "skipdrafts") {
      // Finished shoots started before `before` (ms) get no TikTok draft (an old backlog).
      const before = Number(u.searchParams.get("before") || 0), out = [];
      for (const p of (await personas.list(env)).filter(isRoster)) {
        const state = await st.load(env, p.slug);
        for (const [id, sh] of Object.entries(state.shoots || {})) {
          const only = u.searchParams.get("id");
          if (!sh.draft && (only ? id === only : (sh.started || 0) < before)) { sh.draft = "skipped"; out.push(id); }
        }
        await st.save(env, p.slug, state);
      }
      return Response.json(out);
    }
    if (what === "ttwho") return Response.json(await tiktok.whoami(env, u.searchParams.get("slug") || "mpt"));
    if (what === "ttstatus") {
      const out = {};
      for (const p of (await personas.list(env)).filter(isRoster)) {
        const state = await st.load(env, p.slug);
        for (const [id, sh] of Object.entries(state.shoots || {})) {
          if (sh.draft && !["off", "skipped"].includes(sh.draft)) out[id] = await tiktok.draftStatus(env, p.slug, sh.draft).catch((e) => String(e));
        }
      }
      return Response.json(out);
    }
    if (what === "draft") {
      // now=1: the inbox was just cleared, so drafts waiting on TikTok go without the hourly wait.
      if (u.searchParams.get("now")) {
        for (const p of (await personas.list(env)).filter(isRoster)) {
          const state = await st.load(env, p.slug);
          const waiting = Object.values(state.shoots || {}).filter((sh) => sh.retry_at);
          waiting.forEach((sh) => delete sh.retry_at);
          if (waiting.length) await st.save(env, p.slug, state);
        }
      }
      await onDraftSchedule(null, env);
      return new Response("draft step done");
    }
  }
  if (u.pathname === "/persona/admin/laptop/claim" && request.method === "POST") return laptopClaim(request, env, u.origin);
  if (u.pathname === "/persona/admin/laptop/result" && request.method === "POST") return laptopResult(request, env);
  if (u.pathname === "/persona/admin/laptop/prepare" && request.method === "POST") return laptopPrepare(request, env);
  if (u.pathname === "/persona/admin/laptop/render-item" && request.method === "POST") return laptopRenderItem(request, env);
  if (u.pathname === "/persona/admin/laptop/talkline" && request.method === "POST") return talkLine(request, env);
  if (u.pathname === "/persona/admin/laptop/stats" && request.method === "POST") return laptopStats(request, env);
  return new Response("not found", { status: 404 });
}

// ------------------------------------------------------------------ render
async function render(env, p, force = false) {
  const state = await st.load(env, p.slug);
  if (state.render_date === st.today() && !force) return `${p.slug}: already rendered today`;
  const b = await st.bible(env, p.slug);
  if (!b) return `${p.slug}: no bible in KV -- run \`python -m persona.cli sync\` on the laptop`;
  const n = Number(env.PERSONA_CLOUD_IMAGES || 10);
  // Her Instagram lane: flirty, Instagram-safe (persona entry style, default "flirty").
  const list = await scenes.write(env, b, state, n, { sexy: p.style === "recommendable" ? false : p.style || "flirty" });
  state.recent.push(...list.map((s) => `${s.setting} / ${s.outfit} / ${s.action}`));
  state.render_date = st.today();
  // The laptop renders when it is open (its GPU, no Kaggle quota); Kaggle otherwise,
  // or later if the laptop goes quiet before finishing (advanceJobs).
  if (await laptopFresh(env)) {
    state.queue.unshift({ id: `render-${p.slug}-${st.today()}`, kind: "render", scenes: list, date: st.today(),
                          chat: p.chat_id, videos: dailyReels(env) });
    state.render_via = "laptop";
    await env.PERSONA.put(`rstatus:${p.slug}`, JSON.stringify({ date: st.today(), ok: true, items: [] }));
    await st.save(env, p.slug, state);
    return `${p.slug}: render queued for the laptop, ${list.length} scenes`;
  }
  state.render_via = "kaggle";
  await st.save(env, p.slug, state);
  await pushRender(env, p, list);
  return `${p.slug}: render pushed, ${list.length} scenes`;
}

/** Reels a day: the render's first N good photos animated (PERSONA_DAILY_REELS, 2). */
const dailyReels = (env) => Number(env.PERSONA_DAILY_REELS ?? 2);

async function pushRender(env, p, list, character = null, videos = null) {
  await kaggle.push(env, renderKernel(env, p),
    kaggle.bootstrap({ ...basePayload(env, p, character), scenes: list, videos: videos ?? dailyReels(env), date: st.today() }),
    [dataset(env, character || p.slug)]);
}

/**
 * A daily-render photo from the laptop: kept in KV (3 days) until its post slot,
 * where post() sends it like a Kaggle render output.
 */
async function laptopRenderItem(request, env) {
  // JSON {result, item, file_id}: the laptop already put the photo/Reel on Telegram.
  const { result: r, item, file_id } = await request.json();
  if (r.shoot) {
    // An MPT shoot photo: posted to the channel now (by its Telegram file).
    const p = await personas.get(env, r.persona || env.PERSONA_SLUG);
    const state = await st.load(env, p.slug);
    learnQA(state, r.qa);
    if (!r.ok && !r.replaces) { await st.save(env, p.slug, state); return new Response("noted"); }
    if (r.replaces) {
      await albumRedoDone(env, p, state, r, item, file_id);
      await st.save(env, p.slug, state);
      return new Response("posted");
    }
    if (r.ok && item && file_id && !state.items[r.id]) {
      await postShootPhoto(env, p, state, r.shoot, item, { fileId: file_id }, r.character);
      await st.save(env, p.slug, state);
    }
    return new Response("posted");
  }
  const slug = r.slug || env.PERSONA_SLUG || "lena";
  const ttl = { expirationTtl: 3 * 86400 };
  if (r.ok && item && file_id) await env.PERSONA.put(`ritem:${slug}:${r.id}`, JSON.stringify({ ...item, file_id }), ttl);
  const status = JSON.parse((await env.PERSONA.get(`rstatus:${slug}`)) || "null") || { date: r.render, ok: true, items: [] };
  if (!status.items.some((x) => x.id === r.id)) {
    status.items.push({ id: r.id, ok: !!r.ok, kind: r.kind === "video" ? "video" : "image", parent: r.parent || null, reason: r.reason || null });
  }
  status.date = r.render || status.date;
  await env.PERSONA.put(`rstatus:${slug}`, JSON.stringify(status));
  return new Response("kept");
}

// -------------------------------------------------------------------- post
async function post(env, p, k) {
  const state = await st.load(env, p.slug);
  // Today's photos: from KV when the laptop rendered them, else Kaggle's output.
  let status, getItem, getImage;
  if (state.render_via === "laptop") {
    status = JSON.parse((await env.PERSONA.get(`rstatus:${p.slug}`)) || "null");
    if (!status) return `${p.slug}: no laptop render`;
    getItem = async (id) => JSON.parse((await env.PERSONA.get(`ritem:${p.slug}:${id}`)) || "null");
    getImage = async (id) => {
      const it = await getItem(id);
      return it?.file_id ? { fileId: it.file_id } : null;
    };
  } else {
    const files = await kaggle.outputs(env, renderKernel(env, p)).catch(() => ({}));
    if (!files["status.json"]) return `${p.slug}: no render output`;
    status = await kaggle.fetchJson(files["status.json"]);
    getItem = async (id) => (files[`out/items/${id}.json`] ? kaggle.fetchJson(files[`out/items/${id}.json`]) : null);
    getImage = async (id) => {
      const it = await getItem(id);
      return it?.path ? files[`out/items/${it.path}`] || null : files[`out/items/${id}.jpg`] || null;
    };
  }
  if (!status.ok || status.date !== st.today()) return `${p.slug}: render not ready (${status.stage} ${status.date})`;
  if (state.posted.date !== st.today()) state.posted = { date: st.today(), ids: [] };
  const images = (status.items || []).filter((r) => r.ok && (r.kind || "image") === "image").map((r) => r.id);
  // The day's Reels go out between the photos (2nd and 4th place).
  const reels = (status.items || []).filter((r) => r.ok && r.kind === "video").map((r) => r.id);
  const order = [...images];
  reels.forEach((id, j) => order.splice(Math.min(order.length, 1 + 3 * j), 0, id));
  const isReel = new Set(reels);
  const due = order.filter((id, i) => i % POST_HOURS.length <= k);
  const b = (await st.bible(env, p.slug)) || { name: p.slug };
  let sent = 0;
  for (const id of due) {
    if (state.posted.ids.includes(id)) continue;
    const item = await getItem(id);
    const img = item && (await getImage(id));
    if (!img) continue;
    try {
      const reel = isReel.has(id);
      const msg = await sendMedia(env, reel ? "video" : "photo", p.chat_id, img,
        { caption: caption(b.name, { ...item, kind: reel ? "video" : item.kind }), buttons: buttons({ ...item, kind: reel ? "video" : item.kind }) });
      remember(state, { ...item, kind: reel ? "video" : (item.kind || "image") }, msg);
      state.posted.ids.push(id);
      sent++;
    } catch (e) {
      console.error("post failed", id, String(e));
    }
  }
  await st.save(env, p.slug, state);
  return `${p.slug} slot ${k}: posted ${sent}`;
}

// ----------------------------------------------------- on-demand Kaggle jobs
/**
 * Two Kaggle job kernels per persona, so they never block each other: "quick" for
 * 🔁 and 🎬 (minutes), "recreate" for 🎭 (an hour+). Results only come back when a
 * whole run ends, and on 2026-10-08 a bowling redo sat for hours behind a recreate
 * in one shared run. The original single kernel ("video") is still read for
 * results until its last run is posted.
 */
const LANES = {
  quick: { kernel: (env, p) => `${env.KAGGLE_USERNAME}/mpt-persona-quick-${p.slug}`, takes: (q) => !["recreate", "render"].includes(q.kind), max: 3 },
  recreate: { kernel: (env, p) => `${env.KAGGLE_USERNAME}/mpt-persona-recreate-${p.slug}`, takes: (q) => q.kind === "recreate", max: 1 },
};

/** persona/faceswap.py's check of a 🎭 video: how much of it looks like her. */
function faceLine(face) {
  if (!face) return "\n⚠️ face not checked (no face swap ran)";
  if (!face.judged) return "\n⚠️ face not checked: she never faces the camera";
  const line = `\nface match ${face.median} (lowest ${face.min})`;
  if (face.low_share > 0.15 || face.low_seconds >= 1)
    return `${line}\n⚠️ ${Math.round(face.low_share * 100)}% doesn't look like her (longest ${face.low_seconds}s)`;
  return line;
}

/**
 * Post one finished job (from a Kaggle run or the laptop) under the message that
 * asked for it. `item` is the item JSON, `file` its photo/video (URL or Blob).
 */
/** How a finished job is posted: {kind: "photo"|"video"|"text", caption|text, buttons, chat, replyTo}. */
function resultMessage(b, r, item) {
  if (r.private) {
    return r.ok ? { kind: r.kind === "image" ? "photo" : "video", chat: r.chat, replyTo: r.message_id,
                    caption: `🔒 ${b.name} · private, never posted${r.video && r.kind === "image" ? " · video coming" : ""}\n#${r.id}` }
                : { kind: "text", chat: r.chat, replyTo: r.message_id, text: `❌ 🔒 Private one failed: ${String(r.reason).slice(0, 200)}` };
  }
  if (r.ok && r.cast) {
    return { kind: "photo", chat: r.chat, replyTo: r.message_id,
             caption: `👤 face for ${r.cast.replace(/^mpt-/, "")}\n${String(r.look || "").slice(0, 300)}\n#${r.id}`,
             buttons: kb([[["✅ Pick", `pv:pick:${r.id}`], ["🔁 More faces", `pv:recast:${r.id}`]]]) };
  }
  if (!r.ok && r.cast) return { kind: "text", chat: r.chat, replyTo: r.message_id, text: `❌ 👤 Casting failed: ${String(r.reason).slice(0, 300)}` };
  if (r.ok && r.persona_ready) {
    b = r.bible || b;
    return { kind: "photo", chat: r.chat, replyTo: r.message_id,
             caption: `✅ ${b.name || r.persona_ready} joined the roster: refs and face sheet done. She talks from the next rotation; ` +
                      `"talk ${b.name || ""}: <quote>", "video ${b.name || ""}: <prompt>" and "carousel ${b.name || ""}: <prompt>" work now.` };
  }
  if (r.ok && r.kind === "image" && r.carousel) {
    // A 🔁 of one shot replaces it in its own message.
    return { kind: "photo", chat: r.chat, replyTo: r.message_id, edit: r.replace ? r.message_id : null,
             caption: `🖼 another angle of #${r.carousel}\n🎭 ${item?.scene?.angle || ""}\n#${r.id}`,
             buttons: kb([[["👎", `pv:dn:${r.id}`], ["🔁 Redo this angle", `pv:redo:${r.id}`]]]) };
  }
  if (r.ok && r.kind === "image") {
    return { kind: "photo", chat: r.chat, replyTo: r.message_id, edit: r.redo ? r.message_id : null,
             caption: `🔁 ${r.redo ? `${redoSummary(r)}\n` : ""}${caption(b.name, item)}`, buttons: buttons(item) };
  }
  if (r.ok && r.talk) {
    return { kind: "video", chat: r.chat, replyTo: r.message_id,
             caption: `🎙 ${b.name} · ${r.topic || "talk"} · ${r.bw ? "⚫ B&W" : "🎨 colour"}\n“${String(r.line || "").slice(0, 600)}”\n#${r.id}`,
             buttons: kb([[["👍", `pv:up:${r.id}`], ["👎", `pv:dn:${r.id}`], ["🔁 Redo", `pv:vredo:${r.id}`]]]) };
  }
  if (!r.ok && r.talk) {
    return { kind: "text", chat: r.chat, replyTo: r.message_id, text: `❌ 🎙 Talk failed: ${String(r.reason).slice(0, 300)}` };
  }
  if (r.ok) {
    const cap = r.recreate ? `🎭 ${b.name} · recreated${faceLine(item?.face)}` : `🎬 ${b.name}\n${(r.motion || "").slice(0, 300)}`;
    return { kind: "video", chat: r.chat, replyTo: r.message_id, caption: `${cap}\n#${r.id}`,
             buttons: kb([[["👍", `pv:up:${r.id}`], ["👎", `pv:dn:${r.id}`], ["🔁 Redo", `pv:vredo:${r.id}`]]]) };
  }
  if (r.carousel) {
    return { kind: "text", chat: r.chat, replyTo: r.message_id,
             text: `🖼 One carousel shot didn't pass the checks (${String(r.reason).slice(0, 120)}); the carousel goes ahead with the others. Reply "carousel" for more.` };
  }
  return { kind: "text", chat: r.chat, replyTo: r.message_id,
           text: `❌ ${r.kind === "image" ? "Redraw" : r.recreate ? "Recreate" : "Video"} failed: ${String(r.reason).slice(0, 300)}` };
}

/**
 * 🖼 angles arrive one by one; when the set is complete, a message under the
 * original photo offers to post photo + angles as one Instagram carousel.
 */
async function carouselProgress(env, state, r) {
  if (!r.carousel) return;
  const set = (state.carousels = state.carousels || {})[r.carousel] || { ids: [] };
  if (!r.ok) {
    // A shot that failed its checks: offer the carousel with the shots that worked.
    set.failed = (set.failed || 0) + 1;
    state.carousels[r.carousel] = set;
    if (set.ids.length && !set.offered && set.ids.length + set.failed >= (r.of || 2)) {
      set.offered = true;
      await tg(env, "sendMessage", { chat_id: r.chat, reply_to_message_id: r.message_id,
        text: `🖼 Carousel ready: this photo + ${set.ids.length} more. Tap to post them as one Instagram carousel.`,
        reply_markup: kb([[["🖼 Post carousel", `pv:carpub:${r.carousel}`]]]) });
    }
    return;
  }
  if (r.replace && set.ids.includes(r.replace)) {
    // 🔁 on one angle: the new one takes its place in the set.
    set.ids = set.ids.map((id) => (id === r.replace ? r.id : id));
    state.carousels[r.carousel] = set;
    // The shot was replaced in its own message; the existing "Post carousel"
    // button now posts the new set.
    return;
  }
  if (!set.ids.includes(r.id)) set.ids.push(r.id);
  state.carousels[r.carousel] = set;
  if (set.ids.length + (set.failed || 0) < (r.of || 2) || set.offered) return;
  set.offered = true;
  await tg(env, "sendMessage", { chat_id: r.chat, reply_to_message_id: r.message_id,
    text: `🖼 Carousel ready: this photo + ${set.ids.length} more angles. Tap to post them as one Instagram carousel.`,
    reply_markup: kb([[["🖼 Post carousel", `pv:carpub:${r.carousel}`]]]) });
}

const albumHint = (p) => `reply post → ${!isRoster(p) && p.buffer?.instagram ? "Instagram carousel" : "TikTok"} · redo 3 · video 3`;

/** A shoot album to her Instagram as one carousel (its first 10 photos): a "post" reply or a 👍. */
async function publishAlbum(env, p, state, shootId) {
  const sh = state.shoots?.[shootId];
  if (!sh) return;
  const fileIds = (sh.media || []).map((m) => state.items[m.id]?.file_id).filter(Boolean).slice(0, 10);
  if (fileIds.length < 2) return albumStatus(env, p, sh, "⚠️ can't post: the album's photos aren't known", state);
  if (state.publish.some((j) => j.item === `car-${shootId}` && !["failed"].includes(j.ig?.status))) {
    return albumStatus(env, p, sh, "📤 Instagram: already queued", state);
  }
  const b = (await st.bible(env, p.slug)) || { name: p.slug };
  const first = state.items[sh.media[0].id] || {};
  const posts = await captions.forPost(env, p, b, state, { ...first, kind: "photo" });
  if (!sh.learned) { for (const id of sh.ids) learnFrom(state, state.items[id], 0.5, { taste: true }); sh.learned = true; }
  const drafted = state.publish.some((j) => j.item === `ttcar-${shootId}`);
  state.publish.push({ id: `pub-car-${shootId}-${Date.now()}`, item: `car-${shootId}`, kind: "carousel", file_ids: fileIds,
    ...storyOrSlot(env, state), audio: false, posts, caption: posts.instagram.text,
    ...(drafted ? { tt: { status: "skipped", note: "its draft went with the album" } } : {}),
    chat: p.chat_id, message_id: sh.album, album: true, base: sh.caption_text || `👤 ${b.name}` });
  await advancePublish(env, p, state, env.WORKER_ORIGIN);
}

/** 🖼 Post carousel: the original photo and its angles, one Instagram post. */
async function publishCarousel(env, p, state, cq, itemId) {
  const set = state.carousels?.[itemId];
  const fileIds = [state.items[itemId]?.file_id, ...(set?.ids || []).map((id) => state.items[id]?.file_id)].filter(Boolean);
  if (fileIds.length < 2) return answer(env, cq.id, "Can't: the carousel's photos aren't known.");
  if (state.publish.some((j) => j.item === `car-${itemId}`)) return answer(env, cq.id, "Already queued.");
  const b = (await st.bible(env, p.slug)) || { name: p.slug };
  const known = state.items[itemId] || {};
  const posts = await captions.forPost(env, p, b, state, { ...known, kind: "photo" });
  state.publish.push({ id: `pub-car-${itemId}-${Date.now()}`, item: `car-${itemId}`, kind: "carousel", file_ids: fileIds,
    ...storyOrSlot(env, state), audio: false, posts, caption: posts.instagram.text,
    chat: cq.message.chat.id, message_id: cq.message.message_id, base: cq.message.text || "🖼 Carousel", text: true });
  await advancePublish(env, p, state, env.WORKER_ORIGIN);
  return answer(env, cq.id, state.publish.at(-1).story
    ? "🖼 This week's feed is full (kept for Reels): its first photo goes to her Story."
    : `🖼 Carousel of ${fileIds.length} queued for Instagram.`);
}

/** One job can return several results (🖼 carousel angles): each is posted once. */
const resultKey = (r) => (r.carousel ? `${r.request}/${r.id}` : r.request);

/** Remember a posted result (`sent`: the Telegram message it went out as). */
/**
 * 🔁 on a talk, face swap or video (button or a "redo" reply): made again the same way
 * -- a talk says the same line in a new scene -- as a reply to it. Posts from before
 * redo_job was kept are rebuilt from the item (2026-10-30: a redo did nothing).
 * True, or why it can't.
 */
function redoVideo(p, state, itemId, msg) {
  const it = state.items[itemId];
  if (!it) return "Can't redo this one: it isn't known any more.";
  if (state.queue.some((q) => q.redo_of === itemId)) return "Already queued.";
  const who = it.character || p.slug;
  let job = it.redo_job;
  if (!job && it.talk) {
    const t = talkTopic(p, who);
    job = { kind: "talk", character: who, topic: it.topic || t.topic, sources: t.sources, local_only: true, bw: !!it.bw };
  } else if (!job && it.recreate && /^https?:/.test(it.source || "")) {
    job = { kind: "recreate", url: it.source, seconds: 10, character: who };
  } else if (!job) {
    // A 🎬 video: its photo, by the item or the photo it was posted in reply to.
    const ph = (msg.reply_to_message?.photo || []).slice(-1)[0];
    const parent = it.parent || (msg.reply_to_message ? itemIdFrom(msg.reply_to_message) : "");
    const fileId = state.items[parent]?.file_id || ph?.file_id;
    if (fileId && !it.talk && !it.recreate) job = { kind: "video", item: parent || itemId, file_id: fileId, motion: state.items[parent]?.scene?.motion || "" };
  }
  if (!job) return "Can't redo this one: how it was made isn't known.";
  const again = { ...job, id: `${job.kind}-${who}-${Date.now()}`, redo_of: itemId, character: who,
                  chat: msg.chat.id, message_id: msg.message_id };
  if (job.kind === "talk") Object.assign(again, { given_line: it.line || job.given_line || "", given_url: "" });
  delete again.given_keywords;
  state.queue.push(again);
  return true;
}

/**
 * How a video/talk/face swap was asked for, kept on its item so 🔁 Redo can ask again
 * (2026-10-30: redo for everything). `q` is its queue entry, found before it is dropped.
 */
function keepRedo(state, r, q) {
  if (!r?.ok || !q || !state.items[r.id] || !["talk", "recreate", "video"].includes(q.kind || "video")) return;
  const { id, ...rest } = q;
  state.items[r.id].redo_job = { ...rest, kind: q.kind || "video", character: r.character || q.character };
}

const queuedFor = (state, r) => [...Object.values(state.lanes || {}).flat(), ...state.queue].find((q) => q?.id === r.request);

function recordResult(state, r, item, sent) {
  if (!r.ok || !item) return;
  if (r.kind === "image") remember(state, item, sent);
  else remember(state, { ...item, kind: "video", recreate: !!r.recreate }, sent);
  if (r.carousel && state.items[r.id]) state.items[r.id].carousel_of = r.carousel;
  if (r.character && state.items[r.id]) state.items[r.id].character = r.character;
}

/**
 * Post one finished job (from a Kaggle run or the laptop) under the message that
 * asked for it. `item` is the item JSON, `file` its photo/video (URL or Blob).
 */
async function postResult(env, state, b, r, item, file) {
  const m = albumAware(state, resultMessage(b, r, item));
  if (m.kind === "text") {
    await tg(env, "sendMessage", { chat_id: m.chat, reply_to_message_id: m.replyTo, text: m.text });
    await carouselProgress(env, state, r);
    return;
  }
  let sent = null;
  if (m.edit) {
    sent = await editPhoto(env, m.chat, m.edit, file, { caption: m.caption, buttons: m.buttons })
      .catch((e) => { console.error("in-place replace failed", String(e).slice(0, 200)); return null; });
  }
  const replaced = !!sent;
  if (!sent) sent = await sendMedia(env, m.kind, m.chat, file, { caption: m.caption, buttons: m.buttons || undefined, replyTo: m.replyTo });
  recordResult(state, r, item, sent);
  if (replaced) swapAlbumPhoto(state, r, sent);
  await carouselProgress(env, state, r);
  return sent;
}

async function postResults(env, p, state, b, kernel) {
  const files = await kaggle.outputs(env, kernel).catch(() => ({}));
  const status = files["status.json"] ? await kaggle.fetchJson(files["status.json"]).catch(() => ({})) : {};
  let changed = false;
  for (const r of status.items || []) {
    if (!r.request || state.done.includes(resultKey(r)) || !r.chat) continue;
    try {
      const item = r.ok ? await kaggle.fetchJson(files[`out/items/${r.id}.json`]) : null;
      const file = item ? files[`out/items/${r.kind === "image" ? `${r.id}.jpg` : item.path}`] : null;
      // A redo of an MPT album photo: swapped into the album like the laptop's
      // (2026-10-23: Kaggle's came back as a photo of its own, the album unchanged).
      if (r.shoot && r.replaces) await albumRedoDone(env, p, state, r, item, null, file);
      else {
        const sent = await postResult(env, state, r.character ? (await st.bible(env, r.character)) || b : b, r, item, file);
        keepRedo(state, r, queuedFor(state, r));
        if (r.talk && r.ok) await talkDone(env, p, state, r, item, sent);
        await autoPublish(env, p, state, r, sent);
        await crossPost(env, p, state, r, item, sent);
      }
    } catch (e) {
      console.error("job result post failed", r.request, String(e));
      continue;
    }
    state.done.push(resultKey(r));
    changed = true;
  }
  return changed;
}

// ------------------------------------------------------------------ laptop
/**
 * While the laptop is open, persona/agent.py checks in every 30s and runs the
 * on-demand jobs of every persona on its own GPU; Kaggle only gets them when it has
 * gone quiet. The check-in (KV "laptop", one for all personas) is written at most
 * every 4 minutes (KV's free tier allows 1000 writes a day), so "quiet" means no
 * check-in for LAPTOP_STALE_MS.
 */
const LAPTOP_STALE_MS = 10 * 60 * 1000;
const LAPTOP_WRITE_MS = 4 * 60 * 1000;

async function laptopSeen(env) {
  return JSON.parse((await env.PERSONA.get("laptop")) || "{}").seen || 0;
}
const laptopFresh = async (env) => Date.now() - (await laptopSeen(env)) < LAPTOP_STALE_MS;

async function laptopClaim(request, env, origin) {
  const body = await request.json().catch(() => ({}));
  if (Date.now() - (await laptopSeen(env)) > LAPTOP_WRITE_MS) {
    await env.PERSONA.put("laptop", JSON.stringify({ seen: Date.now() }));
  }
  // Then taps (while the bot has no webhook), so a tap made a moment ago is in a
  // queue below -- and, the laptop being marked up, not pushed to Kaggle.
  if (env.PERSONA_POLL === "1" && !body.busy) await poll(env, origin).catch((e) => console.error("poll", String(e)));
  const all = await personas.list(env);
  // Posts are NOT moved along here any more (2026-11-10): every 30s, on state that KV
  // can serve a minute stale, the same video went to Buffer up to 4 times. The crons
  // (every ~7 min) and taps post; social.js once() guards each post as well.
  if (body.busy) return Response.json({ job: null });
  // A job it already holds comes back first: the agent restarted mid-job.
  for (const pass of ["held", "queued"]) {
    for (const p of all) {
      const state = await st.load(env, p.slug);
      state.lanes = state.lanes || {};
      let q = (state.lanes.laptop || [])[0];
      if (!q && pass === "queued" && state.queue.length) {
        q = state.queue.shift();
        q.claimed_at = Date.now();
        state.lanes.laptop = [q];
        await st.save(env, p.slug, state);
      }
      if (q) return Response.json({ job: { ...(await jobPayload(env, origin, state, [q]))[0], slug: charOf(p, q), persona: p.slug } });
    }
  }
  return Response.json({ job: null });
}

/**
 * The laptop posts finished photos/videos to Telegram itself (laptop/prepare gives
 * it the caption and buttons), then reports the sent message here as JSON. Big
 * videos sent THROUGH the Worker failed with HTTP 500 on 2026-10-16 (a 10s recreate,
 * five tries), and the agent deleted it and rendered it again.
 */
async function laptopPrepare(request, env) {
  const { result: r, item } = await request.json();
  const p = await personas.get(env, r.persona || r.slug || env.PERSONA_SLUG || "lena");
  const b = (await st.bible(env, r.character || p.slug)) || { name: p.slug };
  return Response.json(albumAware(await st.load(env, p.slug), resultMessage(b, r, item)));
}

/** A redo of a photo inside an album: replaced in place, with no buttons (albums can't). */
function albumAware(state, m) {
  if (m.edit && state.album_msgs?.[m.edit]) return { ...m, buttons: null, caption: "" };
  return m;
}

/** After an album photo was redrawn: the shoot (and its TikTok draft) uses the new one. */
function swapAlbumPhoto(state, r, sent) {
  const a = sent && r.ok && state.album_msgs?.[r.message_id];
  if (!a || !(r.redo || r.replace)) return;
  const sh = state.shoots?.[a.shoot];
  if (sh) {
    sh.ids = sh.ids.map((x) => (x === a.item ? r.id : x));
    sh.media = (sh.media || []).map((x) => (x.id === a.item ? { ...x, id: r.id } : x));
  }
  if (state.items[r.id]) state.items[r.id].message_id = r.message_id;
  state.album_msgs[r.message_id] = { ...a, item: r.id };
}

async function laptopResult(request, env) {
  let r, item, file = null, sent = null;
  if ((request.headers.get("content-type") || "").startsWith("application/json")) {
    ({ result: r, item, sent } = await request.json());
  } else {
    const form = await request.formData();
    r = JSON.parse(form.get("result"));
    item = form.get("item") ? JSON.parse(await form.get("item").text()) : null;
    file = form.get("file");
  }
  const p = (await personas.get(env, r.persona || r.slug || env.PERSONA_SLUG || "lena"));
  const state = await st.load(env, p.slug);
  if (state.done.includes(resultKey(r))) return new Response("already posted");
  learnQA(state, r.qa);
  const b = (await st.bible(env, r.character || p.slug)) || { name: p.slug };
  if (r.kind === "render-done") {
    // The photos came in one by one (laptop/render-item); a finished MPT shoot
    // becomes one TikTok draft.
    // (The draft itself is made by the next cron run: uploading the photos and
    // waiting for gh-pages takes minutes, longer than the laptop waits here.)
    if (r.shoot && state.shoots?.[r.shoot]) {
      state.shoots[r.shoot].finished = true;
      await postShootAlbum(env, p, state, r.shoot).catch((e) => console.error("album failed", String(e).slice(0, 200)));
    }
  }
  else if (sent) {
    await builderDone(env, p, state, r);
    recordResult(state, r, item, sent);
    privateDone(state, r, sent);
    keepRedo(state, r, queuedFor(state, r));
    swapAlbumPhoto(state, r, sent);
    await carouselProgress(env, state, r);
    if (r.talk && r.ok) await talkDone(env, p, state, r, item, sent);
    await autoPublish(env, p, state, r, sent);
    await crossPost(env, p, state, r, item, sent);
  }
  else {
    await builderDone(env, p, state, r);
    const posted = await postResult(env, state, b, r, item, file);
    privateDone(state, r, posted);
    keepRedo(state, r, queuedFor(state, r));
    await autoPublish(env, p, state, r, posted);
    await crossPost(env, p, state, r, item, posted);
  }
  state.done.push(resultKey(r));
  // Taken back from Kaggle's queue too, if it was handed over while the laptop
  // was quiet; a Kaggle result for it is skipped as already done.
  state.lanes.laptop = (state.lanes.laptop || []).filter((q) => q.id !== r.request);
  state.queue = state.queue.filter((q) => q.id !== r.request);
  await st.save(env, p.slug, state);
  await env.PERSONA.put("laptop", JSON.stringify({ seen: Date.now() }));
  return new Response("posted");
}

async function jobPayload(env, origin, state, batch) {
  const jobs = [];
  for (const q of batch) {
    const before = jobs.length;
    if (q.kind === "carousel") {
      jobs.push({ id: q.id, kind: "carousel", item: q.item, scene: q.scene, count: q.count || 3, replace: q.replace || null,
                  image_url: await fileLink(env, origin, q.file_id), chat: q.chat, message_id: q.message_id });
    } else if (q.kind === "render") {
      jobs.push({ id: q.id, kind: "render", scenes: q.scenes, date: q.date, chat: q.chat, videos: q.videos ?? 0,
                  shoot: q.shoot || null });
    } else if (q.kind === "redo") {
      // The photo itself goes along: the laptop looks at what is wrong with it and
      // fixes that in an edit of the same photo (2026-10-20).
      const fid = q.file_id || state.items[q.item]?.file_id;
      jobs.push({ id: q.id, kind: "redo", scene: q.scene, chat: q.chat, message_id: q.message_id,
                  ...(fid ? { image_url: await fileLink(env, origin, fid) } : {}),
                  ...(q.shoot ? { shoot: q.shoot, replaces: q.replaces } : {}) });
    } else if (q.kind === "talk") {
      jobs.push({ id: q.id, kind: "talk", topic: q.topic, sources: q.sources, chat: q.chat, bw: !!q.bw, viral: !!q.viral,
                  given_line: q.given_line || "", given_url: q.given_url || "", message_id: q.message_id || null,
                  series: q.series || "", episode: q.episode || 0, follow_line: q.follow_line || "",
                  reply_comment: q.reply_comment || "", reply_to_line: q.reply_to_line || "",
                  ...(q.theme ? { theme: q.theme, theme_date: q.theme_date, look: q.look || null } : {}),
                  ...(q.given_keywords ? { given_keywords: q.given_keywords, given_caption: q.given_caption || "", given_hook: q.given_hook || "" } : {}),
                  used: (state.talk_used || []).slice(-300) });
    } else if (q.kind === "private") {
      jobs.push({ id: q.id, kind: "private", prompt: q.prompt, video: q.video, character: q.character,
                  chat: q.chat, message_id: q.message_id });
    } else if (q.kind === "cast" || q.kind === "lock") {
      jobs.push({ id: q.id, kind: q.kind, new_slug: q.new_slug, bible: q.bible, n: q.n, pick: q.pick,
                  character: q.character, chat: q.chat, message_id: q.message_id });
    } else if (q.kind === "recreate") {
      jobs.push({ id: q.id, kind: "recreate", url: q.url, seconds: q.seconds, chat: q.chat, message_id: q.message_id,
                  video_url: q.file_id ? await fileLink(env, origin, q.file_id) : "",
                  ...(q.character ? { character: q.character } : {}), ...(q.fit ? { fit: q.fit } : {}) });
    } else {
      jobs.push({ id: q.id, image_url: await fileLink(env, origin, q.file_id), motion: q.motion || "", wish: q.wish || "",
                  ...(q.private ? { private: true, character: q.character } : {}),
                  caption: state.items[q.item]?.caption || "", tags: state.items[q.item]?.tags || [],
                  hook: q.private ? "" : state.items[q.item]?.scene?.hook || "",
                  chat: q.chat, message_id: q.message_id });
    }
    if (jobs.length > before && q.character) jobs[jobs.length - 1].character = q.character;
  }
  return jobs;
}

// ------------------------------------------------------------- MPT shoots
/**
 * A roster persona's shoot: the next girl in rotation, p.photos (10) scenes in as
 * many different places, rendered on the laptop when it is open, else on Kaggle
 * (at most p.kaggle_shoots a day -- 5 shoots of 10 would need ~5 GPU hours a day,
 * more than Kaggle's 30h a week). Photos post to the channel as they come; the
 * finished shoot becomes one TikTok photo draft (shootDone).
 */
/**
 * Ask for a shoot: only recorded here (state.shoot_requests). Writing its scenes is
 * an AI call that can take longer than a request may (Cloudflare cuts it at ~100s,
 * and a shoot asked for that way vanished without a word, 2026-10-18/19), so the
 * draft cron (onDraftSchedule, its own invocation every 15 min) writes them --
 * writeRequestedShoot -- and queues the render.
 */
async function shoot(env, p, who = null, ask = null) {
  const state = await st.load(env, p.slug);
  const character = who || (isRoster(p) ? nextCharacter(p, state) : p.slug);
  if (!(await st.bible(env, character))) { await st.save(env, p.slug, state); return `${p.slug}: no bible for ${character} (persona.cli sync)`; }
  // `ask`: from the channel -- {brief, video, message_id} (a carousel or a video of a prompt).
  state.shoot_requests = [...(state.shoot_requests || []), { character, at: Date.now(), ...(ask || {}) }];
  await st.save(env, p.slug, state);
  return `${p.slug}: shoot of ${character} requested (scenes written within 15 min)`;
}

/** One requested shoot: its scenes written and its render queued. True when it did one. */
async function writeRequestedShoot(env, p) {
  const req = (await st.load(env, p.slug)).shoot_requests?.[0];
  if (!req) return false;
  const b = await st.bible(env, req.character);
  const recentState = await st.load(env, p.slug);
  // One photoshoot: one place, one outfit, one day -- a carousel.
  const list = b ? await scenes.writeShoot(env, b, recentState, req.video ? 1 : Number(p.photos || 6),
    { sexy: p.style === "recommendable" ? false : p.style || (isRoster(p) ? true : "flirty"), brief: req.brief || "" }) : null;
  // Applied onto the state as it is now: other writers moved while the AI wrote.
  const state = await st.load(env, p.slug);
  state.shoot_requests = (state.shoot_requests || []).filter((r) => !(r.character === req.character && r.at === req.at));
  if (list?.length) {
    state.recent.push(`${list[0].shoot_place} / ${list[0].outfit}`, ...list.map((x) => `${x.setting} / ${x.action}`));
    const id = `shoot-${req.character}-${Date.now()}`;
    state.shoots = state.shoots || {};
    state.shoots[id] = { character: req.character, ids: [], started: Date.now(),
                         ...(req.brief ? { brief: req.brief, video: !!req.video, asked: req.message_id || null } : {}) };
    state.queue.push({ id, kind: "render", scenes: list, date: st.today(), chat: p.chat_id, videos: 0, character: req.character, shoot: id });
    console.log(`${p.slug}: shoot of ${req.character} queued (${list.length} scenes)`);
  }
  await st.save(env, p.slug, state);
  return true;
}

/** Post one shoot photo to the channel and remember it under its shoot. */
async function postShootPhoto(env, p, state, shootId, item, file, character) {
  const it = { ...item, character };
  const sh = (state.shoots = state.shoots || {})[shootId] || { character, ids: [] };
  if (p.album === false) {
    const b = (await st.bible(env, character)) || { name: character };
    remember(state, it, await sendMedia(env, "photo", p.chat_id, file, { caption: caption(b.name, it), buttons: buttons(it) }));
  } else {
    // Kept until the shoot is done, then posted as ONE album -- a carousel a shoot,
    // 5 a day, not each photo on its own with buttons (2026-10-18).
    remember(state, it, null);
    if (file?.fileId) state.items[item.id].file_id = file.fileId;
    sh.media = [...(sh.media || []).filter((m) => m.id !== item.id), { id: item.id, ...(file?.fileId ? { fileId: file.fileId } : { url: file }) }];
  }
  if (!sh.ids.includes(item.id)) sh.ids.push(item.id);
  state.shoots[shootId] = sh;
}

/**
 * An album can't carry buttons, so they come in a message under it: 🔁 n redraws
 * photo n in place, 🎬 n animates it (2026-10-19: "no options to redo").
 */
async function albumButtons(env, p, sh) {
  const ids = sh.media.slice(0, 10).map((m) => m.id);
  const row = (icon, act) => ids.map((id, i) => [`${icon}${i + 1}`, `pv:${act}:${id}`]);
  const chunk = (r) => (r.length > 5 ? [r.slice(0, 5), r.slice(5)] : [r]);
  await tg(env, "sendMessage", { chat_id: p.chat_id, text: "🔁 n redraws photo n · 🎬 n makes it a video",
    reply_parameters: { message_id: sh.album, allow_sending_without_reply: true },
    reply_markup: kb([...chunk(row("🔁", "sredo")), ...chunk(row("🎬", "svid"))]) })
    .then(() => { sh.buttons = true; })
    .catch((e) => console.error("album buttons", String(e).slice(0, 150)));
}

/**
 * Learning from you (2026-10-20, "always"): a redo counts against what that photo
 * was (its place, outfit, head direction, body view, pose), a carousel you send on
 * or a 👍 counts for it. scenes.js weights its picks by state.prefs.
 */
function learnFrom(state, item, delta, { taste = false } = {}) {
  // Risk keys (what tends to break anatomy): pose, expression, framing, view, head
  // direction. Taste keys (place, outfit) only ever go UP, from what you send on, so
  // learning can't narrow the variety (2026-10-20: "not always the same pics").
  const sc = item?.scene || {};
  const keys = [sc.action && `pose:${scenes.poseKey(sc.action)}`, sc.expression && `expr:${sc.expression}`,
                sc.shot && `shot:${String(sc.shot).toLowerCase().split(/\s+/).slice(0, 3).join(" ")}`,
                `view:${sc.view || "front"}`, `angle:${sc.angle_sheet || ""}`].filter(Boolean);
  if (taste && delta > 0) keys.push(...[sc.shoot_place && `place:${sc.shoot_place}`, sc.outfit && `outfit:${sc.outfit}`].filter(Boolean));
  bump(state, keys, delta);
}

function bump(state, keys, delta) {
  state.prefs = state.prefs || {};
  for (const k of keys) state.prefs[k] = Math.max(-5, Math.min(5, (state.prefs[k] || 0) + delta));
  const all = Object.keys(state.prefs);
  if (all.length > 400) for (const k of all.sort((x, y) => Math.abs(state.prefs[x]) - Math.abs(state.prefs[y])).slice(0, all.length - 400)) delete state.prefs[k];
}

/**
 * The checks as teacher (the main signal): every attempt that failed for anatomy --
 * extra/broken limbs, hands holding too much, a twisted head, head size -- counts
 * against what that photo was asked to be; a photo that passed first time counts for
 * it. `qa` = {keys, fails, ok} from the laptop (persona/scenes.learn_keys).
 */
function learnQA(state, qa) {
  if (!qa?.keys?.length) return;
  const anatomy = (qa.fails || []).filter((w) => /limb|hand|arm|finger|object|head|twist|proportion|shoulder/i.test(w)).length;
  if (anatomy) bump(state, qa.keys, -0.5 * anatomy);
  else if (qa.ok && !(qa.fails || []).length) bump(state, qa.keys, 0.2);
}

/**
 * "redo 3" / "redo 2 5" as a reply to a shoot's album: those photos are drawn again
 * and, when all are back, the album is posted again with them in place (asked for
 * 2026-10-19 instead of buttons under the album).
 */
/**
 * Which shoot album a message belongs to: the saved map, else the album's own
 * message range (its photos have consecutive ids from sh.album) -- the map lost
 * older albums once (2026-10-20, "post" did nothing on them).
 */
function albumOf(state, msg) {
  const id = msg?.message_id;
  if (!id) return null;
  if (state.album_msgs?.[id]) return state.album_msgs[id];
  for (const [shoot, sh] of Object.entries(state.shoots || {})) {
    const n = sh.media?.length || sh.ids?.length || 0;
    if (typeof sh.album === "number" && id >= sh.album && id < sh.album + n) {
      return { shoot, item: (sh.media || [])[id - sh.album]?.id || sh.ids[id - sh.album] };
    }
  }
  const tag = /#(shoot-[\w-]+)/.exec(msg.caption || "")?.[1];
  return tag && state.shoots?.[tag] ? { shoot: tag, item: null } : null;
}

async function albumRedo(env, p, post, state) {
  const a = albumOf(state, post.reply_to_message);
  // "post" / "tiktok" / 👍: this carousel goes to TikTok as a draft (2026-10-20: sent
  // only when asked, after any redos, so the 5 daily slots go to approved ones).
  if (a && /^\s*(post|tiktok|tik tok|send|publish|👍)\b/iu.test(post.text || "") && !/\d/.test(post.text || "")) {
    await sendAlbum(env, p, state, a);
    await tidy(env, post);
    return true;
  }
  return albumRedoCommands(env, p, post, state, a);
}

/**
 * A reaction in the channel (Telegram sends channels only the counts): 👍 ❤️ 🔥 on
 * an MPT album sends it to TikTok; on a Lena post it publishes it, like the 👍 button.
 */
const LIKES = new Set(["👍", "❤", "❤️", "🔥", "😍", "🥰"]);
async function onReaction(env, p, rc, state, origin) {
  const liked = (rc.reactions || []).some((r) => r.type?.type === "emoji" && LIKES.has(r.type.emoji) && r.total_count > 0);
  if (!liked) return false;
  const msg = { message_id: rc.message_id, chat: rc.chat };
  const a = albumOf(state, msg);
  if (a) {
    const sh = state.shoots?.[a.shoot];
    if (sh?.send_tiktok || (sh?.draft && !["skipped", "off"].includes(sh.draft) && !sh.changed_after_draft)) return false;
    await sendAlbum(env, p, state, a);
    return true;
  }
  if (isRoster(p)) return false;
  // Lena: the post's item from what we sent it as.
  const itemId = Object.entries(state.items || {}).find(([, it]) => it.message_id === rc.message_id)?.[0];
  if (!itemId || state.voted.includes(itemId)) return false;
  const res = await queuePublish(env, p, state, { message: msg }, itemId);
  if (res === true) { state.voted.push(itemId); await advancePublish(env, p, state, origin); }
  return true;
}

/** An album to TikTok: a "post" reply or a 👍/❤️/🔥 reaction on it. */
async function sendAlbum(env, p, state, a) {
    if (!isRoster(p) && p.buffer?.instagram) return publishAlbum(env, p, state, a.shoot);
    const sh = state.shoots?.[a.shoot];
    if (sh) {
      if (sh.send_tiktok && !sh.draft && !sh.draft_error) {
        await albumStatus(env, p, sh, sh.tiktok_status || "📤 TikTok: already on its way", state);
      } else {
        // Sent again even if it went before: you asked (a draft deleted in the app, or
        // sent automatically before the photos were redone).
        // Redone since it was sent: the new photos go up (urls cleared) as a new draft.
        Object.assign(sh, { send_tiktok: true, draft: null, draft_error: null, retry_at: null, urls: null, changed_after_draft: false });
        sh.tiktok_status = "📤 TikTok: sending to the aiqube inbox (~15-45 min)";
        if (!sh.learned) { for (const id of sh.ids) learnFrom(state, state.items[id], 0.5, { taste: true }); sh.learned = true; }
        await albumStatus(env, p, sh, sh.tiktok_status, state);
      }
    }
}

async function albumRedoCommands(env, p, post, state, a) {
  // "redo 3" / "redo 2 5" / "redo 3 red lipstick, hair up" (a note added to the prompt);
  // "video 3" / "video 3 she spins slowly" (how she moves).
  const m = /^\s*(redo|🔁|video|🎬)\s*((?:\d+[\s,]*)+)(.*)$/isu.exec(post.text || "");
  const note = (m?.[3] || "").trim().slice(0, 300);
  if (!a || !m) return false;
  const sh = state.shoots?.[a.shoot];
  if (/video|🎬/iu.test(m[1])) {
    const ids = [...new Set(m[2].split(/[\s,]+/).filter(Boolean).map(Number))].map((n) => [n, sh?.media?.[n - 1]?.id]).filter(([, id]) => id);
    for (const [, id] of ids) {
      const it = state.items[id];
      if (it?.file_id) queueVideo(state, id, { chat: { id: post.chat.id }, message_id: it.message_id || sh.album }, it.file_id, note);
    }
    if (ids.length) await albumStatus(env, p, sh, `🎬 making a video of ${ids.map(([n]) => n).join(", ")} -- it comes as a reply (~40-60 min)`, state);
    await tidy(env, post);
    return true;
  }
  const nums = [...new Set(m[2].split(/[\s,]+/).filter(Boolean).map(Number))];
  const picked = nums.map((n) => [n, sh?.media?.[n - 1]?.id]).filter(([, id]) => id && state.items[id]?.scene?.setting);
  if (!sh || !picked.length) {
    if (sh) await albumStatus(env, p, sh, `⚠️ redo: this album has photos 1-${sh.media?.length || 0}`, state);
    await tidy(env, post);
    return true;
  }
  for (const [, id] of picked) {
    if (state.queue.some((q) => q.kind === "redo" && q.item === id)) continue;
    learnFrom(state, state.items[id], -1);
    state.queue.push({ id: `${id}-r${Date.now()}`, kind: "redo", item: id,
                       scene: { ...state.items[id].scene, ...(note ? { note } : {}) }, chat: post.chat.id,
                       message_id: post.message_id, shoot: a.shoot, replaces: id, character: sh.character });
    sh.redo_wait = (sh.redo_wait || 0) + 1;
  }
  await albumStatus(env, p, sh, `🔁 redrawing ${picked.map(([n]) => n).join(", ")}${note ? ` (${note.slice(0, 60)})` : ""} -- the carousel comes again (~15-30 min)`, state);
  await tidy(env, post);
  return true;
}

/**
 * A shoot's news (TikTok draft, redraws) as one line in its album's caption, edited
 * in place -- not a new message each time (2026-10-20: "too much text in the channel").
 */
async function albumStatus(env, p, sh, line, state = null) {
  sh.status = line;
  if (typeof sh.album !== "number") return;
  if (!sh.caption_text) {
    // Albums posted before the status line existed: their caption rebuilt the same way.
    const b = (await st.bible(env, sh.character)) || { name: sh.character };
    const first = (state?.items || {})[sh.media?.[0]?.id || sh.ids?.[0]] || {};
    const shootId = Object.entries(state?.shoots || {}).find(([, x]) => x === sh)?.[0] || "";
    sh.caption_text = `👤 ${b.name} · ${(sh.media || sh.ids || []).length} photos${first.caption ? `\n${first.caption}` : ""}${shootId ? `\n#${shootId}` : ""}`;
  }
  await tg(env, "editMessageCaption", { chat_id: p.chat_id, message_id: sh.album,
    caption: `${sh.caption_text}\n${line}`.slice(0, 1024) }).catch(() => {});
}

/** A reply the bot has acted on goes, so the channel keeps only the posts. */
async function tidy(env, post) {
  await tg(env, "deleteMessage", { chat_id: post.chat.id, message_id: post.message_id }).catch(() => {});
}

/** What a redo did, in words for the post: what it saw and fixed, or that it redrew it. */
function redoSummary(r) {
  const fixed = (r.fixed || []).slice(0, 3).join("; ");
  if (r.how === "fixed") return fixed ? `fixed: ${fixed}` : "no defect seen -- edited per your note";
  return fixed ? `redrawn (couldn't fix: ${fixed})` : "redrawn from scratch";
}

/** One redrawn album photo back: swapped into the shoot; the last one reposts the album. */
async function albumRedoDone(env, p, state, r, item, fileId, url = null) {
  const sh = state.shoots?.[r.shoot];
  state.queue = state.queue.filter((q) => q.id !== r.request);
  state.lanes.laptop = (state.lanes.laptop || []).filter((q) => q.id !== r.request);
  if (!sh) return;
  if (!r.ok) sh.redo_failed = [...(sh.redo_failed || []), (sh.media || []).findIndex((x) => x.id === r.replaces) + 1];
  if (r.ok) {
    const n = (sh.media || []).findIndex((x) => x.id === r.replaces) + 1;
    sh.redo_notes = [...(sh.redo_notes || []), `🔁 photo ${n} redone -- ${redoSummary(r)}`];
  }
  if (r.ok && item && (fileId || url)) {
    remember(state, { ...item, character: sh.character }, null);
    // A Kaggle result comes as its output URL: the album's repost gives it a file_id.
    if (fileId) state.items[r.id].file_id = fileId;
    sh.media = (sh.media || []).map((x) => (x.id === r.replaces ? { id: r.id, ...(fileId ? { fileId } : { url }) } : x));
    sh.ids = sh.ids.map((x) => (x === r.replaces ? r.id : x));
  }
  sh.redo_wait = Math.max(0, (sh.redo_wait || 1) - 1);
  if (sh.redo_wait) return;
  const failed = (sh.redo_failed || []).filter((n) => n > 0);
  sh.redo_failed = [];
  if (failed.length && !sh.media.some((x) => x.id === r.id)) {
    // Nothing new to show: the album stays, and says what happened.
    await albumStatus(env, p, sh, `⚠️ redraw of ${failed.join(", ")} failed its checks -- reply redo ${failed.join(" ")} to try again`, state);
    return;
  }
  if (sh.draft && !["skipped", "off"].includes(sh.draft)) {
    sh.changed_after_draft = true;
    sh.tiktok_status = "↻ TikTok has the old version -- reply post to send this one";
    sh.status = sh.tiktok_status;
  }
  // All back: the old album goes, the new one is posted.
  for (const [mid, a] of Object.entries(state.album_msgs || {})) {
    if (a.shoot !== r.shoot) continue;
    await tg(env, "deleteMessage", { chat_id: p.chat_id, message_id: Number(mid) }).catch(() => {});
    delete state.album_msgs[mid];
  }
  sh.album = null;
  if (/^🔁/.test(sh.status || "")) {
    const notes = (sh.redo_notes || []).join("\n");
    sh.status = [notes, sh.tiktok_status || albumHint(p)].filter(Boolean).join("\n");
  }
  sh.redo_notes = [];
  await postShootAlbum(env, p, state, r.shoot);
}

/** A finished shoot's photos as one Telegram album (its carousel). */
async function postShootAlbum(env, p, state, shootId) {
  const sh = state.shoots?.[shootId];
  if (!sh || sh.album || !sh.media?.length) return;
  const b = (await st.bible(env, sh.character)) || { name: sh.character };
  const first = state.items[sh.media[0].id] || {};
  if (sh.video) {
    // "video <prompt>": its one photo as a post of its own (🔁 redraws it), animated
    // as a reply; both wait for a 👍.
    const m = sh.media[0], it = state.items[m.id] || {};
    const msg = await tg(env, "sendPhoto", { chat_id: p.chat_id, photo: m.fileId || m.url, caption: caption(b.name, it).slice(0, 1024),
      reply_markup: buttons(it), ...(sh.asked ? { reply_parameters: { message_id: sh.asked, allow_sending_without_reply: true } } : {}) });
    const ph = (msg.photo || []).slice(-1)[0];
    if (state.items[m.id]) Object.assign(state.items[m.id], { file_id: ph?.file_id || it.file_id, message_id: msg.message_id });
    sh.album = msg.message_id;
    // Its scene's motion (the prompt made safe, scenes.readBrief), never the raw prompt.
    queueVideo(state, m.id, msg, ph?.file_id || it.file_id, it.scene?.motion || "");
    // Her video, not the persona's ("no character 'mpt'", 2026-10-30).
    const vq = state.queue.find((q) => q.kind !== "redo" && q.item === m.id);
    // The laptop writes its movement from the photo, with the prompt as the wish (qa.motion_for).
    if (vq) Object.assign(vq, { character: sh.character, wish: sh.brief || "" });
    await crossPost(env, p, state, { ok: true, id: m.id, kind: "image", character: sh.character }, it, msg);
    return;
  }
  sh.caption_text = `👤 ${b.name} · ${sh.media.length} photos${first.caption ? `\n${first.caption}` : ""}\n#${shootId}`;
  if (!sh.status && !p.tiktok_auto) sh.status = albumHint(p);
  const text = sh.status ? `${sh.caption_text}\n${sh.status}` : sh.caption_text;
  const msgs = await tg(env, "sendMediaGroup", { chat_id: p.chat_id, media: sh.media.slice(0, 10).map((m, i) => ({
    type: "photo", media: m.fileId || m.url, ...(i === 0 ? { caption: text.slice(0, 1024) } : {}) })) });
  // The album's own copies are what the TikTok draft downloads.
  state.album_msgs = state.album_msgs || {};
  (msgs || []).forEach((msg, i) => {
    const id = sh.media[i]?.id, ph = (msg.photo || []).slice(-1)[0];
    if (id && ph && state.items[id]) state.items[id].file_id = ph.file_id;
    if (id && state.items[id]) state.items[id].message_id = msg.message_id;
    if (id) state.album_msgs[msg.message_id] = { shoot: shootId, item: id };
  });
  sh.album = (msgs || [])[0]?.message_id || true;
  // A single persona's shoot (Lena): its first photos also become Reels, as replies
  // to the album (her old daily render made PERSONA_DAILY_REELS a day).
  if (!isRoster(p) && !sh.reels_queued) {
    sh.reels_queued = true;
    for (const m of sh.media.slice(0, dailyReels(env))) {
      const it = state.items[m.id];
      if (it?.file_id) queueVideo(state, m.id, { chat: { id: p.chat_id }, message_id: it.message_id || sh.album }, it.file_id, "");
    }
  }
  // Lena's albums are copied into MPT's channel (2026-10-30: everything of hers there too).
  await crossPostAlbum(env, p, sh, b);
  // Old album records go after 5 days.
  const keep = Object.entries(state.album_msgs);
  if (keep.length > 300) state.album_msgs = Object.fromEntries(keep.slice(-300));
}

/**
 * A finished shoot's TikTok inbox draft, one step per call, never waiting inside a
 * request (Cloudflare closes one after ~100s, and an invocation may make only 50
 * subrequests): first its photos go to gh-pages in one commit (~35 requests), then
 * a later call, once Pages serves them, makes the draft. Runs in its own cron
 * invocation (onDraftSchedule). Returns true when it did something.
 */
async function shootDone(env, p, state, shootId) {
  const sh = state.shoots?.[shootId];
  if (!sh || sh.draft || !sh.ids.length) return false;
  // Persona entry tiktok_drafts:false keeps shoots in Telegram only.
  if (p.tiktok_drafts === false) { sh.draft = "off"; return true; }
  const b = (await st.bible(env, sh.character)) || { name: sh.character };
  try {
    if (!sh.urls) {
      const photos = [];
      for (const id of sh.ids) {
        // The album keeps each photo too (a Kaggle one as its output URL).
        const m = (sh.media || []).find((x) => x.id === id) || {};
        const fileId = state.items[id]?.file_id || m.fileId;
        const f = fileId ? await telegramFile(env, fileId) : m.url ? await fetch(m.url) : { ok: false };
        if (f.ok) photos.push({ name: `${Date.now()}-${id}.jpg`, bytes: await f.arrayBuffer() });
      }
      sh.urls = await tiktok.hostAll(env, photos.slice(0, 35));
      return true;
    }
    if (!(await tiktok.live(sh.urls))) {
      if (Date.now() - (sh.started || 0) > 24 * 3600 * 1000) throw new Error("gh-pages never served the photos");
      return false;
    }
    const first = state.items[sh.ids[0]] || {};
    const posts = await captions.forPost(env, p, b, state, { ...first, kind: "photo" });
    sh.draft = await tiktok.photoDraft(env, p.slug, sh.urls, posts.tiktok.text);
    sh.tiktok_status = "📥 TikTok: in the aiqube inbox -- add a sound and post";
    await albumStatus(env, p, sh, sh.tiktok_status);
  } catch (e) {
    const msg = String(e.message || e).slice(0, 200);
    if (/too_many_pending|rate_limit|spam_risk/i.test(msg)) {
      // TikTok's inbox is full of unposted drafts: retried every hour until there
      // is room (post or delete drafts in the app), told once.
      if (!sh.retry_at) {
        sh.tiktok_status = "⏳ TikTok: waiting for a free draft slot (5 a day), retried hourly";
        await albumStatus(env, p, sh, sh.tiktok_status);
      }
      sh.retry_at = Date.now() + 3600 * 1000;
      return true;
    }
    sh.draft_error = msg;
    sh.tiktok_status = `❌ TikTok draft failed: ${msg.slice(0, 120)}`;
    await albumStatus(env, p, sh, sh.tiktok_status);
  }
  return true;
}

/** The draft cron: one step of one finished shoot's TikTok draft per run. */
export async function onDraftSchedule(event, env) {
  // A requested shoot's scenes first, for every persona (one AI call each): a
  // persona later in the list was starved while the one before it had a draft
  // step every run (Lena's shoot waited behind MPT's, 2026-10-23).
  let wrote = false;
  for (const p of (await personas.list(env)).filter(byShoots)) {
    try {
      wrote = (await writeRequestedShoot(env, p)) || wrote;
    } catch (e) {
      console.error("writing a requested shoot failed", String(e).slice(0, 200));
    }
  }
  if (wrote) return;
  for (const p of await personas.list(env)) {
    if (!byShoots(p)) continue;
    const state = await st.load(env, p.slug);
    // A finished shoot with no photos (a failed Kaggle run) never gets a draft; it
    // used to be picked every run and stall every shoot behind it (2026-10-19).
    const pending = Object.entries(state.shoots || {})
      .find(([, sh]) => sh.finished && sh.ids?.length && (sh.send_tiktok || p.tiktok_auto) && !sh.draft && !sh.draft_error
                        && !(sh.retry_at && Date.now() < sh.retry_at));
    if (!pending) continue;
    if (await shootDone(env, p, state, pending[0])) {
      // Saved onto the current state: other writers may have moved meanwhile.
      const now = await st.load(env, p.slug);
      if (now.shoots?.[pending[0]]) {
        const sh = state.shoots[pending[0]];
        Object.assign(now.shoots[pending[0]], { urls: sh.urls, draft: sh.draft, draft_error: sh.draft_error, retry_at: sh.retry_at,
                                                status: sh.status, tiktok_status: sh.tiktok_status });
        await st.save(env, p.slug, now);
      }
    }
    return;  // one step a run
  }
}

/** MPT on Kaggle: push a waiting shoot (daily cap), and post a finished one. */
async function advanceShoots(env, p, state, laptopUp) {
  const notes = [];
  const run = state.kaggle_shoot;
  if (run) {
    const s = await kaggle.status(env, renderKernel(env, p));
    if (s === "COMPLETE" || s === "ERROR" || s.startsWith("CANCEL") || Date.now() - run.started > 13 * 3600 * 1000) {
      const files = await kaggle.outputs(env, renderKernel(env, p)).catch(() => ({}));
      const status = files["status.json"] ? await kaggle.fetchJson(files["status.json"]).catch(() => ({})) : {};
      for (const r of (status.items || []).filter((x) => x.ok && (x.kind || "image") === "image")) {
        try {
          const item = await kaggle.fetchJson(files[`out/items/${r.id}.json`]);
          await postShootPhoto(env, p, state, run.id, item, files[`out/items/${r.id}.jpg`], run.character);
        } catch (e) { console.error("shoot photo post failed", r.id, String(e).slice(0, 200)); }
      }
      if (state.shoots?.[run.id]) state.shoots[run.id].finished = true;
      await postShootAlbum(env, p, state, run.id).catch((e) => console.error("album failed", String(e).slice(0, 200)));
      state.kaggle_shoot = null;
      notes.push(`shoot ${run.id}: ${s}`);
    } else {
      notes.push(`shoot on Kaggle ${s}`);
      return notes;
    }
  }
  // Old shoot records go after 3 days.
  for (const [id, sh] of Object.entries(state.shoots || {})) if (Date.now() - (sh.started || 0) > 3 * 86400 * 1000) delete state.shoots[id];
  if (laptopUp) return notes;
  const day = st.today();
  if (state.kaggle_shoots?.date !== day) state.kaggle_shoots = { date: day, n: 0 };
  const job = state.queue.find((q) => q.kind === "render" && q.shoot);
  if (job && state.kaggle_shoots.n < Number(p.kaggle_shoots ?? 2) && !kaggleFull(state)) {
    try {
      await pushRender(env, p, job.scenes, job.character, 0);
    } catch (e) {
      notes.push(await kaggleRefused(env, p, state, e));
      return notes;
    }
    state.kaggle_shoot = { id: job.shoot, character: job.character, started: Date.now() };
    state.kaggle_shoots.n += 1;
    state.queue = state.queue.filter((q) => q !== job);
    notes.push(`shoot ${job.shoot} pushed to Kaggle`);
  }
  // A shoot nobody could render today is dropped rather than piling up.
  state.queue = state.queue.filter((q) => !(q.shoot && q.date !== day && Date.now() - (state.shoots?.[q.shoot]?.started || 0) > 20 * 3600 * 1000));
  return notes;
}

/**
 * Kaggle refused a run (its weekly GPU quota, 30h): no push is tried for 6 hours, the
 * jobs stay queued for the laptop, and her channel is told once a day.
 */
const kaggleFull = (state) => (state.kaggle_full?.until || 0) > Date.now();
async function kaggleRefused(env, p, state, err) {
  const msg = String(err.message || err).replace(/^kaggle: /, "").slice(0, 200);
  const day = st.today();
  const told = state.kaggle_full?.told === day;
  state.kaggle_full = { until: Date.now() + 6 * 3600 * 1000, note: msg, told: day };
  if (!told && p.chat_id) {
    await tg(env, "sendMessage", { chat_id: p.chat_id, disable_notification: true,
      text: `⚠️ Kaggle refused a run: ${msg}\nJobs wait for the laptop (persona agent) until it frees up.` }).catch(() => {});
  }
  return `kaggle refused: ${msg}`;
}

async function advanceJobs(env, p, origin) {
  const state = await st.load(env, p.slug);
  const b = (await st.bible(env, p.slug)) || { name: p.slug };
  state.lanes = state.lanes || {};
  let changed = false;
  const notes = [];

  // The old single kernel: post whatever its last run produced, once.
  if ((state.running || []).length) {
    const s = await kaggle.status(env, jobKernel(env, p));
    if (s === "COMPLETE" || s === "ERROR") {
      changed = (await postResults(env, p, state, b, jobKernel(env, p))) || changed;
      state.running = [];
      changed = true;
    } else notes.push(`legacy ${s}`);
  }

  // The laptop went quiet holding a job (lid closed mid-render): Kaggle takes it.
  const laptopUp = await laptopFresh(env);
  if ((state.lanes.laptop || []).length && !laptopUp) {
    state.queue = [...state.lanes.laptop, ...state.queue];
    state.lanes.laptop = [];
    changed = true;
    notes.push("laptop quiet: its job goes to Kaggle");
  }
  if (laptopUp && state.queue.length) notes.push("laptop up: new jobs wait for it");
  if (byShoots(p)) {
    const n = await advanceShoots(env, p, state, laptopUp);
    if (n.length) { notes.push(...n); changed = true; }
  }
  // The day's render waiting for a laptop that went quiet: Kaggle renders it.
  const renderJob = !isRoster(p) && state.queue.find((q) => q.kind === "render" && !q.shoot);
  if (renderJob && !laptopUp && !kaggleFull(state)) {
    try {
      await pushRender(env, p, renderJob.scenes);
      state.queue = state.queue.filter((q) => q !== renderJob);
      state.render_via = "kaggle";
      notes.push("render: laptop quiet, pushed to Kaggle");
    } catch (e) {
      notes.push(await kaggleRefused(env, p, state, e));
    }
    changed = true;
  }

  // 🎙 A talk waiting an hour with the laptop off goes to Kaggle, up to the day's
  // talk_hours count (was only after a whole day off, one a day -- 2026-10-28: the
  // laptop asleep overnight held the morning's talks until noon). Its line is
  // written here (Kaggle gets no secret to ask for one).
  const day = st.today();
  const kt = state.kaggle_talks?.day === day ? state.kaggle_talks.n : 0;
  const ktCap = Number(env.PERSONA_KAGGLE_TALKS_PER_DAY || (p.talk_hours || []).length || 1);
  if (!laptopUp && !kaggleFull(state) && Date.now() - (await laptopSeen(env)) > 3600 * 1000 && kt < ktCap &&
      !state.queue.some((x) => x.kind === "talk" && !x.local_only)) {
    const q = state.queue.find((x) => x.kind === "talk" && x.local_only && Date.now() - (x.created || 0) > 3600 * 1000);
    if (q) {
      const r = await writeTalkLine(env, { persona: p.slug, character: q.character, topic: q.topic,
                                           ...(q.given_line ? { given: q.given_line } : { transcript: "" }) });
      if (r.line) {
        Object.assign(q, { given_line: r.line, given_keywords: r.keywords || [], given_caption: r.caption || "", given_hook: r.hook || "", local_only: false });
        state.kaggle_talks = { day, n: kt + 1 };
        changed = true;
        notes.push(`talk ${q.id}: laptop off, to Kaggle`);
      }
    }
  }
  for (const [name, lane] of Object.entries(LANES)) {
    const kernel = lane.kernel(env, p);
    const running = state.lanes[name] || [];
    if (running.length) {
      const s = await kaggle.status(env, kernel);
      const started = state.lane_started?.[name] || 0;
      // A cancelled run, or a lane busy past Kaggle's 12h run limit, will never
      // post: free the lane and say so. On 2026-10-11 a run ended CANCEL_ACKNOWLEDGED,
      // a status this did not know, and the recreate lane stayed blocked for 5 days.
      const dead = s.startsWith("CANCEL") || (started && Date.now() - started > 13 * 3600 * 1000);
      if (dead) {
        for (const q of running) {
          if (state.done.includes(q.id)) continue;
          await tg(env, "sendMessage", { chat_id: q.chat, reply_to_message_id: q.message_id,
            text: `❌ ${q.kind === "recreate" ? "Recreate" : q.kind === "redo" ? "Redraw" : "Video"} failed: the Kaggle run was ${s.startsWith("CANCEL") ? "cancelled" : "stuck"}. Send it again.` }).catch(() => {});
          state.done.push(q.id);
        }
        state.lanes[name] = [];
        changed = true;
        notes.push(`${name}: dead run (${s}) cleared`);
        continue;
      }
      if (s === "RUNNING" || s === "QUEUED") { notes.push(`${name} ${s}`); continue; }
      if (s === "COMPLETE" || s === "ERROR") {
        changed = (await postResults(env, p, state, b, kernel)) || changed;
        // Jobs the finished run never had (its push was refused, or a newer push
        // replaced it) go back in the queue, at most twice, instead of vanishing.
        const back = running.filter((q) => !state.done.some((d) => d === q.id || d.startsWith(`${q.id}/`)) &&
                                           (q.requeued || 0) < 2);
        if (back.length) {
          state.queue = [...back.map((q) => ({ ...q, requeued: (q.requeued || 0) + 1 })), ...state.queue];
          notes.push(`${name}: ${back.length} not in the run, queued again`);
        }
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
    // One character per Kaggle run (her dataset is the one mounted); jobs on media
    // someone sent by hand in a roster channel never leave the laptop.
    const ready = laptopUp || kaggleFull(state) ? [] : state.queue.filter(lane.takes).filter((q) => !q.local_only);
    const who = ready.length ? charOf(p, ready[0]) : null;
    const batch = ready.filter((q) => charOf(p, q) === who).slice(0, lane.max);
    if (!batch.length) continue;
    const jobs = await jobPayload(env, origin, state, batch);
    try {
      await kaggle.push(env, kernel, kaggle.bootstrap({ ...basePayload(env, p, who), mode: "video", jobs }), [dataset(env, who)]);
    } catch (e) {
      notes.push(await kaggleRefused(env, p, state, e));
      changed = true;
      continue;
    }
    state.lanes[name] = batch;
    state.lane_started = { ...(state.lane_started || {}), [name]: Date.now() };
    const taken = new Set(batch.map((q) => q.id));
    state.queue = state.queue.filter((q) => !taken.has(q.id));
    changed = true;
    notes.push(`${name}: pushed ${jobs.length}`);
  }
  if (changed) await st.save(env, p.slug, state);
  return `${p.slug}: ${notes.join("; ") || "idle"}`;
}

/**
 * A single persona's daily Story (Lena, 2026-10-23): a shoot photo of the last 3 days
 * that went to Instagram neither as a post nor a Story, through Buffer. Daily
 * Stories keep a new account looking active between feed posts.
 */
/**
 * Her Instagram Stories, PERSONA story_hours (default 3 a day, 2026-10-30: "improve
 * the stories as much as possible"): Stories keep followers seeing her between posts.
 * The middle one shares her newest Reel or talk (a Story of a fresh video sends
 * viewers to it); the others are photos from her recent shoots that never went to
 * the feed -- behind-the-scenes of the day. Nothing is used twice.
 */
async function dailyStory(env, p, origin) {
  if (isRoster(p) || !p.buffer?.instagram || p.daily_story === false) return `${p.slug}: no daily story`;
  const state = await st.load(env, p.slug);
  const day = st.today();
  const n = state.story_day?.day === day ? state.story_day.n : 0;
  const used = new Set([...state.publish.flatMap((j) => [j.file_id, ...(j.file_ids || [])]),
                        ...(state.blocked || []).map((id) => state.items[id]?.file_id)].filter(Boolean));
  const storied = new Set(state.publish.filter((j) => j.story).map((j) => j.file_id));
  const fresh = Object.values(state.shoots || {}).filter((sh) => Date.now() - (sh.started || 0) < 3 * 86400 * 1000)
    .sort((a, b) => (b.started || 0) - (a.started || 0))
    .flatMap((sh) => (sh.media || []).map((m) => ({ id: m.id, fileId: state.items[m.id]?.file_id, kind: "photo" })))
    .filter((x) => x.fileId && !used.has(x.fileId));
  // Her newest video that went out in the last 2 days, not yet in a Story.
  const video = state.publish.filter((j) => j.kind === "video" && !j.story && j.ig?.status === "done" && j.file_id &&
                                            !storied.has(j.file_id) && Date.now() - (j.ig.sent_at || 0) < 2 * 86400 * 1000)
    .sort((a, b) => (b.ig.sent_at || 0) - (a.ig.sent_at || 0))[0];
  state.story_day = { day, n: n + 1 };
  const x = n === 1 && video ? { id: video.item, fileId: video.file_id, kind: "video" }
    : fresh.length ? fresh[Math.floor(Math.random() * Math.min(3, fresh.length))]
    : video ? { id: video.item, fileId: video.file_id, kind: "video" } : null;
  if (!x) { await st.save(env, p.slug, state); return `${p.slug}: nothing new for a story`; }
  state.publish.push({ id: `pub-story-${x.id}-${Date.now()}`, item: x.id, kind: x.kind, story: true, quiet: true,
                       created: Date.now(), file_id: x.fileId, audio: x.kind === "video", caption: "", posts: {} });
  await advancePublish(env, p, state, origin);
  await st.save(env, p.slug, state);
  return `${p.slug}: story ${x.id} queued`;
}

// ------------------------------------------------------------------ status
/**
 * "status" / "laptop" / "I opened the laptop" in either channel (2026-11-05): is the
 * laptop agent up, can Kaggle run, what is running and queued, what today's schedule
 * still has, and roughly when all of it is done. Times are rough averages from the
 * agent's logs (laptop) and Kaggle runs (setup included).
 */
const MIN = 60 * 1000;
const JOB_MIN = { laptop: { talk: 27, shoot: 17, render: 46, recreate: 23, video: 5, redo: 5, cast: 10, lock: 10, other: 5 },
                  kaggle: { talk: 85, shoot: 45, render: 60, recreate: 40, video: 20, redo: 20, cast: 30, lock: 30, other: 20 } };
const jobKind = (q) => (q.shoot ? "shoot" : q.kind in JOB_MIN.laptop ? q.kind : "other");
const STATUS_RE = /^(?:⏱️?|status|left|how long|laptop(?:\s+(?:on|open|opened|up))?|i\s+opened\s+(?:the\s+)?laptop)\b/iu;

async function onStatusRequest(env, p, post) {
  if (post.reply_to_message || !STATUS_RE.test(String(post.text || "").trim())) return false;
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id, text: await dayStatus(env) });
  return true;
}

async function dayStatus(env) {
  const now = Date.now();
  const clock = (t) => new Date(t).toLocaleTimeString("en-GB", { timeZone: "Europe/Bucharest", hour: "2-digit", minute: "2-digit" });
  const span = (ms) => { const m = Math.max(0, Math.round(ms / MIN)); return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`; };
  const seen = await laptopSeen(env);
  const up = now - seen < LAPTOP_STALE_MS;
  const todayEnd = new Date(); todayEnd.setUTCHours(23, 59, 59, 999);
  const h = new Date(now).getUTCHours(), m = new Date(now).getUTCMinutes();
  const ahead = (hours) => (hours || []).filter((x) => x > h || (x === h && m < 15));
  const at = (x) => { const d = new Date(now); d.setUTCHours(x, 0, 0, 0); return d.getTime(); };
  // Gather every persona's work first: the laptop takes all queued jobs (any persona)
  // before a scheduled one exists, so the timeline is: running, queued, then by hour.
  const sections = [], work = [];
  let last = now;
  for (const p of await personas.list(env)) {
    const state = await st.load(env, p.slug);
    const sec = { p, head: [], rows: [], tail: [] };
    sections.push(sec);
    const lane = up ? "laptop" : kaggleFull(state) ? null : "kaggle";
    if (kaggleFull(state)) sec.head.push(`⛔ Kaggle: ${state.kaggle_full.note} (retry ${clock(state.kaggle_full.until)})`);
    for (const q of state.lanes?.laptop || []) {
      const left = Math.max(2 * MIN, JOB_MIN.laptop[jobKind(q)] * MIN - (now - (q.claimed_at || now)));
      work.push({ sec, order: 0, at: now, lane: "laptop", fixed: left, label: `▶️ ${jobKind(q)} ${(q.character || p.slug).replace(/^mpt-/, "")} on the laptop` });
    }
    for (const [name, run] of Object.entries(state.lanes || {})) {
      if (name === "laptop" || !run.length) continue;
      const left = Math.max(5 * MIN, Math.max(...run.map((q) => JOB_MIN.kaggle[jobKind(q)])) * MIN - (now - (state.lane_started?.[name] || now)));
      work.push({ sec, order: 0, at: now, lane: "kaggle", fixed: left, label: `▶️ Kaggle: ${run.map(jobKind).join(", ")}` });
    }
    if (state.kaggle_shoot) {
      const left = Math.max(5 * MIN, JOB_MIN.kaggle.shoot * MIN - (now - state.kaggle_shoot.started));
      work.push({ sec, order: 0, at: now, lane: "kaggle", fixed: left, label: "▶️ Kaggle: shoot" });
    }
    for (const q of state.queue) {
      work.push({ sec, order: 1, at: now, lane, kind: jobKind(q), label: `⏳ ${jobKind(q)} ${(q.character || p.slug).replace(/^mpt-/, "")}` });
    }
    if (byShoots(p)) {
      for (const x of ahead(p.shoot_hours ?? [5, 7, 9, 11, 13])) work.push({ sec, order: 2, at: at(x), lane, kind: "shoot", label: `🗓 shoot at ${clock(at(x))}` });
      for (const x of ahead(p.talk_hours)) work.push({ sec, order: 2, at: at(x), lane, kind: "talk", label: `🗓 talk at ${clock(at(x))}` });
      if (!isRoster(p) && p.buffer?.instagram) {
        const n = ahead(p.story_hours || [7, 12, 17]).length;
        if (n) sec.tail.push(`🗓 ${n} Instagram Stor${n > 1 ? "ies" : "y"} still today`);
      }
    }
    const due = state.publish.filter((j) => !social.finished(j))
      .map((j) => Math.min(...[!j.ig && j.due, !j.tt && ((j.talk && j.tt_due) || j.due)].filter(Boolean)))
      .filter((t) => Number.isFinite(t) && t <= todayEnd.getTime());
    if (due.length) {
      last = Math.max(last, ...due);
      sec.tail.push(`📤 ${due.length} post(s) still going out today, last ~${clock(Math.max(...due))}`);
    }
  }
  // The laptop does one job at a time; Kaggle runs each lane on its own.
  let cursor = now, waiting = 0;
  for (const w of work.sort((x, y) => x.order - y.order || x.at - y.at)) {
    let end;
    if (w.fixed) end = now + w.fixed;
    else if (!w.lane) { waiting++; w.sec.rows.push(`${w.label}: waits for the laptop`); continue; }
    else if (w.lane === "laptop") end = cursor = Math.max(cursor, w.at) + JOB_MIN.laptop[w.kind] * MIN;
    else end = Math.max(now, w.at) + JOB_MIN.kaggle[w.kind] * MIN;
    if (w.lane === "laptop" && w.fixed) cursor = Math.max(cursor, end);
    last = Math.max(last, end);
    w.sec.rows.push(w.fixed ? `${w.label}, ~${span(end - now)} left` : `${w.label}: done ~${clock(end)}`);
  }
  const lines = [up ? "💻 Laptop agent: up ✅ — jobs run on the laptop"
                    : `💻 Laptop agent: off${seen ? ` (last seen ${clock(seen)})` : ""} — on the laptop run: mpt-on`];
  for (const { p, head, rows, tail } of sections) {
    const out = [...head, ...rows, ...tail];
    lines.push("", `👤 ${p.slug.toUpperCase()}`, ...(out.length ? out : ["✅ nothing running or left today"]));
  }
  lines.push("", last > now + MIN ? `⏱ All of today done ~${clock(last)} (in ${span(last - now)})` : "⏱ All of today is done",
             ...(waiting ? [`⚠️ ${waiting} job(s) can't start until the laptop agent runs (Kaggle is out)`] : []),
             "Times: Bucharest, rough averages.");
  return lines.join("\n").slice(0, 4000);
}

// ---------------------------------------------------------------- schedule
export async function onPersonaSchedule(event, env) {
  if (!env.PERSONA_BOT_TOKEN || !env.KAGGLE_API_TOKEN) return;
  const now = new Date(event.scheduledTime);
  const h = now.getUTCHours(), m = now.getUTCMinutes();
  const origin = env.WORKER_ORIGIN;
  try {
    if (env.PERSONA_POLL === "1") console.log(await poll(env, origin));
  } catch (err) {
    console.error("persona poll failed", String(err));
  }
  if (h === 4 && m < 15) await instagram.refreshTokens(env);
  for (const p of await personas.list(env)) {
    try {
      if (byShoots(p)) {
        if ((p.shoot_hours || [5, 7, 9, 11, 13]).includes(h) && m < 15) console.log(await shoot(env, p));
        if ((p.story_hours || [7, 12, 17]).includes(h) && m < 15) console.log(await dailyStory(env, p, origin));
        if ((p.talk_hours || []).includes(h) && m < 15) {
          const state = await st.load(env, p.slug);
          console.log(queueTalk(p, state));
          await st.save(env, p.slug, state);
        }
      } else if ((h === 2 && m >= 30) || (h === 3 && m < 30)) console.log(await render(env, p));
      else if (POST_HOURS.includes(h) && m < 15) console.log(await post(env, p, POST_HOURS.indexOf(h)));
      console.log(await advanceJobs(env, p, origin));
      const state = await st.load(env, p.slug);
      const dirty = state.publish.some((j) => !social.finished(j)) && (await advancePublish(env, p, state, origin));
      if (dirty) await st.save(env, p.slug, state);
    } catch (err) {
      console.error(`persona ${p.slug} schedule failed`, String(err));
    }
  }
}

// Test hook (scripts/talk_lines_test.mjs): the line writer run outside the Worker.
export const _talkTest = { adaptTalk, judgeTalk, writeHook };
