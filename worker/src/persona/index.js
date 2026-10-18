/**
 * Persona studio, run entirely from this Worker. Kaggle does the scheduled GPU work
 * (the daily render); on-demand jobs run on the laptop while it is open
 * (persona/agent.py), on Kaggle otherwise. GitHub only holds the code.
 *
 * Each persona has her own Telegram channel (personas.js); the one bot serves them
 * all. 👍 on a photo or video there publishes it to her Instagram and TikTok
 * through Buffer (social.js) and marks the post in the channel.
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
  if (item.kind !== "video") lines.push("🎬 video (or reply with how she moves) · 🔁 redraw");
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
                           file_id: media?.file_id || state.items[item.id]?.file_id || "",
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
  const exp = Math.floor(Date.now() / 1000) + 24 * 3600;
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
  const job = (await st.load(env, slug)).publish.find((j) => j.id === jid);
  if (!job) return new Response("Not found.", { status: 404 });
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
<button id="share">2 · Share to Instagram</button>
<small>Pick <b>Instagram → Reels</b>, tap ♫ to add a trending song, paste the caption, share.</small>
<textarea readonly>${esc(caption)}</textarea>
<a class="b" href="${esc(video)}" download="lena.mp4">Or save the video</a>
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
  const info = await tg(env, "getFile", { file_id: fileId });
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
  const headers = { "content-type": ext === "mp4" ? "video/mp4" : "image/jpeg" };
  if (file.headers.get("content-length")) headers["content-length"] = file.headers.get("content-length");
  return new Response(file.body, { status: file.status, headers });
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
  const b = (await st.bible(env, p.slug)) || { name: p.slug };
  // A caption and hashtags for each platform, never empty (captions.js).
  const posts = await captions.forPost(env, p, b, state, { ...known, kind: isPhoto ? "photo" : "video" });
  state.publish.push({
    id: `pub-${itemId}-${Date.now()}`, item: itemId, kind: isPhoto ? "photo" : "video",
    created: Date.now(), due: nextSlot(env, state), recreate: !!known.recreate,
    file_id: fileId, audio: !!known.audio || !!known.recreate,
    posts, caption: posts.instagram.text,
    chat, message_id: msg.message_id, base: (msg.caption ?? fwdCaption ?? "").split("\n\n━━")[0],
  });
  return true;
}

/**
 * The next free posting slot (ms). Instagram throttles accounts that post 3+ times a
 * day (each post competes with the last); 1-2 a day is the sweet spot, so 👍 queues
 * a post into the next free slot instead of posting at once. PERSONA_IG_SLOTS: UTC
 * hours, default "10,18" (12:00 and 20:00 in Prague).
 */
function nextSlot(env, state) {
  const hours = String(env.PERSONA_IG_SLOTS || "10,18").split(",").map(Number).sort((a, b) => a - b);
  const taken = new Set(state.publish.filter((j) => j.due).map((j) => j.due));
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

async function advancePublish(env, p, state, origin) {
  let changed = false;
  for (const job of state.publish.filter((j) => !social.finished(j))) {
    // Buffer said "too many requests": no more calls this round, for any post.
    if (state.publish.some((j) => j.retry_at && Date.now() < j.retry_at)) break;
    const url = job.file_ids
      ? await Promise.all(job.file_ids.map((f) => mediaLink(env, origin, f, "jpg")))
      : await mediaLink(env, origin, job.file_id, job.kind === "video" ? "mp4" : "jpg");
    const before = JSON.stringify([job.ig, job.tt, job.retry_at]);
    await social.advance(env, p, job, url);
    // Saved only when something moved: this runs on every 30s laptop check-in, and an
    // unconditional save there overwrote taps and 🖼 requests that the webhook had
    // saved a moment before (2026-10-16, two carousel requests lost).
    if (JSON.stringify([job.ig, job.tt, job.retry_at]) === before) continue;
    changed = true;
    if (job.retry_at && Date.now() < job.retry_at) continue;  // rate-limited: nothing to show yet
    // Mark the post in the channel: what happened on each platform.
    const cap = `${job.base}\n\n━━ published ━━\n${social.statusLine(job)}`.slice(0, 1024);
    const failed = ["ig", "tt"].some((k) => job[k]?.status === "failed");
    const markup = job.ig?.mode === "phone"
      ? { inline_keyboard: [[{ text: "📲 Post on Instagram", url: await shareLink(env, origin, p, job) }]] }
      : failed ? kb([[["🔁 Retry posting", `pv:repub:${job.id}`]]])
      : kb([[["✓ 👍 published", `pv:noop:${job.item}`]]]);
    try {
      await tg(env, job.text ? "editMessageText" : "editMessageCaption", { chat_id: job.chat, message_id: job.message_id,
        ...(job.text ? { text: cap.slice(0, 4096) } : { caption: cap }), reply_markup: markup });
    } catch (e) {
      console.error("caption update failed", String(e).slice(0, 200));
    }
  }
  // Keep the last 100.
  if (state.publish.length > 100) { state.publish = state.publish.slice(-100); changed = true; }
  return changed;
}

/**
 * 48h after a post went out, its Buffer numbers score its hashtags (captions.learn),
 * so the next posts lean on the tags that worked for her. Given up after 7 days
 * (a TikTok notification post that was never posted from the phone has none).
 */
async function learnFromPosts(env, p, state) {
  let changed = false;
  const now = Date.now();
  for (const job of state.publish) {
    for (const [key, platform] of [["ig", "instagram"], ["tt", "tiktok"]]) {
      const rec = job[key];
      if (!rec?.post || rec.status !== "done" || rec.learned || now - (rec.sent_at || now) < 48 * 3600 * 1000) continue;
      try {
        const post = await social.metrics(env, p, rec.post);
        const value = captions.postValue(post?.metrics);
        if (value > 0) {
          const hour = Number(new Date(rec.sent_at).toLocaleString("en-GB", { timeZone: "Europe/Prague", hour: "2-digit", hour12: false }));
          captions.learn(state, platform, job.posts?.[platform]?.tags || [], value, { kind: job.kind, hour });
          rec.learned = true;
        } else if (now - rec.sent_at > 7 * 86400 * 1000) rec.learned = "no numbers";
        changed = true;
      } catch (e) {
        console.error("metrics failed", rec.post, String(e).slice(0, 200));
      }
    }
  }
  return changed;
}

// ------------------------------------------------------------------- taps
async function onTap(env, p, cq, state, origin) {
  const [, act, itemId] = cq.data.split(":");
  const msg = cq.message || {};
  const chat = msg.chat?.id;
  if (act === "noop") return answer(env, cq.id, "Already published.");
  if (act === "repub") {
    // 🔁 Retry posting: the failed platform(s) start again.
    const job = state.publish.find((j) => j.id === itemId);
    if (!job) return answer(env, cq.id, "That post isn't known any more.");
    for (const k of ["ig", "tt"]) if (job[k]?.status === "failed") delete job[k];
    job.tries = 0;
    job.retry_at = 0;
    await advancePublish(env, p, state, origin);
    return answer(env, cq.id, "🔁 Posting again…");
  }
  if (act === "up" || act === "dn") {
    if (state.voted.includes(itemId)) return answer(env, cq.id, "Already voted.");
    const item = state.items[itemId] || { tags: [] };
    const d = act === "up" ? 1 : -1;
    for (const t of item.tags || []) state.votes.tags[t] = (state.votes.tags[t] || 0) + d;
    if (act === "up" && item.beat) state.story.push({ t: Date.now(), beat: item.beat, item: itemId });
    state.voted.push(itemId);
    const res = act === "up" ? await queuePublish(env, p, state, cq, itemId) : false;
    const publishing = res === true;
    await answer(env, cq.id, act === "dn" ? "Noted 👎" : publishing ? "👍 Added to Buffer's queue for Instagram"
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
  if (act === "vid") {
    const photo = (msg.photo || []).slice(-1)[0];
    if (!photo) return answer(env, cq.id, "That is not a photo.");
    queueVideo(state, itemId, msg, photo.file_id, "");
    return answer(env, cq.id, "🎬 Queued. The video comes as a reply here in ~40-60 min.");
  }
  if (act === "car") {
    const scene = state.items[itemId]?.scene;
    if (!scene?.setting) return answer(env, cq.id, "Can't: this photo's scene wasn't recorded.");
    if (state.queue.some((q) => q.kind === "carousel" && q.item === itemId)) return answer(env, cq.id, "Already queued.");
    const fileId = (msg.photo || []).slice(-1)[0]?.file_id || state.items[itemId]?.file_id;
    if (!fileId) return answer(env, cq.id, "Can't: Telegram didn't give this photo.");
    state.queue.push({ id: `${itemId}-car${Date.now()}`, kind: "carousel", item: itemId, scene, file_id: fileId, count: 2,
                       chat, message_id: msg.message_id });
    return answer(env, cq.id, "🖼 Making 2 more angles of this scene for a carousel (~20-30 min).");
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

const PUBLISH_WORDS = /^\s*(👍|post|publish|post it)\s*[.!]*\s*$/iu;
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
  state.queue.push({ id: `${itemId}-car${Date.now()}`, kind: "carousel", item: itemId, scene: {}, file_id: fileId, count: 2,
                     chat: post.chat.id, message_id: post.message_id });
  await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: post.message_id,
    text: "🖼 Making 2 more shots of this photo for a carousel (~5-10 min). Only photos of her pass the face check." });
}

async function onReply(env, p, post, state, origin) {
  const parent = post.reply_to_message;
  // "👍" / "post" as a reply publishes any photo or video, including ones posted
  // without buttons (older 🎬 videos, hand-posted versions).
  // "🖼" / "carousel" as a reply: a carousel of any photo, including old posts
  // without the 🖼 button.
  if (CAROUSEL_WORDS.test(post.text) && parent.photo) {
    const itemId = itemIdFrom(parent) || `msg-${parent.message_id}`;
    const fileId = parent.photo.slice(-1)[0].file_id;
    state.items[itemId] = { kind: "image", tags: [], scene: {}, caption: "", ...(state.items[itemId] || {}), file_id: fileId };
    const busy = state.queue.some((q) => q.kind === "carousel" && q.item === itemId);
    if (!busy) {
      state.queue.push({ id: `${itemId}-car${Date.now()}`, kind: "carousel", item: itemId, scene: state.items[itemId].scene || {},
                         file_id: fileId, count: 2, chat: post.chat.id, message_id: parent.message_id });
    }
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: parent.message_id,
      text: busy ? "🖼 Already making this carousel." : "🖼 Making 2 more shots of this photo for a carousel (~5-10 min on the laptop)." });
    return;
  }
  if (PUBLISH_WORDS.test(post.text) && (parent.photo || parent.video)) {
    const itemId = itemIdFrom(parent) || `msg-${parent.message_id}`;
    const res = await queuePublish(env, p, state, { message: parent }, itemId);
    if (res === true && !state.voted.includes(itemId)) state.voted.push(itemId);
    await tg(env, "sendMessage", { chat_id: post.chat.id, reply_to_message_id: parent.message_id,
      text: res === true ? "👍 Publishing…" : `⚠️ Not published: ${res || "no Instagram/TikTok linked"}.` });
    if (res === true) await advancePublish(env, p, state, origin);
    return;
  }
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
    const chatId = String(cq?.message?.chat?.id ?? post?.chat?.id ?? "");
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
        if (cq?.data?.startsWith("pv:")) { await onTap(env, p, cq, state, origin); changed = true; }
        // A TikTok/Facebook link or video counts as 🎭 even when it is sent as a reply.
        else if (post && (await onRecreateRequest(env, post, state))) changed = true;
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
  if (u.pathname === "/persona/admin/run" && request.method === "POST") {
    const what = u.searchParams.get("what");
    if (what === "render") return new Response(await render(env, p, u.searchParams.get("force") === "1"));
    if (what === "post") return new Response(await post(env, p, Number(u.searchParams.get("slot") || 4)));
    if (what === "jobs") return new Response(await advanceJobs(env, p, u.origin));
    if (what === "shoot") return new Response(await shoot(env, p, u.searchParams.get("character")));
    if (what === "dropshoot") {
      // A shoot that will never finish (its job is gone): removed from the state.
      const p = await personas.get(env, u.searchParams.get("slug") || "mpt");
      const state = await st.load(env, p.slug);
      const id = u.searchParams.get("id");
      if (!state.shoots?.[id] || state.shoots[id].finished) return new Response("no such unfinished shoot", { status: 404 });
      delete state.shoots[id];
      await st.save(env, p.slug, state);
      return new Response(`dropped ${id}`);
    }
    if (what === "ttstatus") {
      const out = {};
      for (const p of (await personas.list(env)).filter(isRoster)) {
        const state = await st.load(env, p.slug);
        for (const [id, sh] of Object.entries(state.shoots || {})) {
          if (sh.draft && sh.draft !== "off") out[id] = await tiktok.draftStatus(env, p.slug, sh.draft).catch((e) => String(e));
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
  return new Response("not found", { status: 404 });
}

// ------------------------------------------------------------------ render
async function render(env, p, force = false) {
  const state = await st.load(env, p.slug);
  if (state.render_date === st.today() && !force) return `${p.slug}: already rendered today`;
  const b = await st.bible(env, p.slug);
  if (!b) return `${p.slug}: no bible in KV -- run \`python -m persona.cli sync\` on the laptop`;
  const n = Number(env.PERSONA_CLOUD_IMAGES || 10);
  const list = await scenes.write(env, b, state, n);
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
  if (r.ok && r.kind === "image" && r.carousel) {
    // A 🔁 of one shot replaces it in its own message.
    return { kind: "photo", chat: r.chat, replyTo: r.message_id, edit: r.replace ? r.message_id : null,
             caption: `🖼 another angle of #${r.carousel}\n🎭 ${item?.scene?.angle || ""}\n#${r.id}`,
             buttons: kb([[["👎", `pv:dn:${r.id}`], ["🔁 Redo this angle", `pv:redo:${r.id}`]]]) };
  }
  if (r.ok && r.kind === "image") {
    return { kind: "photo", chat: r.chat, replyTo: r.message_id, edit: r.redo ? r.message_id : null,
             caption: "🔁 " + caption(b.name, item), buttons: buttons(item) };
  }
  if (r.ok) {
    const cap = r.recreate ? `🎭 ${b.name} · recreated${faceLine(item?.face)}` : `🎬 ${b.name}\n${(r.motion || "").slice(0, 300)}`;
    return { kind: "video", chat: r.chat, replyTo: r.message_id, caption: `${cap}\n#${r.id}`,
             buttons: kb([[["👍", `pv:up:${r.id}`], ["👎", `pv:dn:${r.id}`]]]) };
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
    if (set.ids.length && !set.offered && set.ids.length + set.failed >= 2) {
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
  if (set.ids.length + (set.failed || 0) < 2 || set.offered) return;
  set.offered = true;
  await tg(env, "sendMessage", { chat_id: r.chat, reply_to_message_id: r.message_id,
    text: `🖼 Carousel ready: this photo + ${set.ids.length} more angles. Tap to post them as one Instagram carousel.`,
    reply_markup: kb([[["🖼 Post carousel", `pv:carpub:${r.carousel}`]]]) });
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
    created: Date.now(), due: nextSlot(env, state), audio: false, posts, caption: posts.instagram.text,
    chat: cq.message.chat.id, message_id: cq.message.message_id, base: cq.message.text || "🖼 Carousel", text: true });
  await advancePublish(env, p, state, env.WORKER_ORIGIN);
  return answer(env, cq.id, `🖼 Carousel of ${fileIds.length} queued for Instagram.`);
}

/** One job can return several results (🖼 carousel angles): each is posted once. */
const resultKey = (r) => (r.carousel ? `${r.request}/${r.id}` : r.request);

/** Remember a posted result (`sent`: the Telegram message it went out as). */
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
  const m = resultMessage(b, r, item);
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
  if (!sent) sent = await sendMedia(env, m.kind, m.chat, file, { caption: m.caption, buttons: m.buttons, replyTo: m.replyTo });
  recordResult(state, r, item, sent);
  await carouselProgress(env, state, r);
}

async function postResults(env, state, b, kernel) {
  const files = await kaggle.outputs(env, kernel).catch(() => ({}));
  const status = files["status.json"] ? await kaggle.fetchJson(files["status.json"]).catch(() => ({})) : {};
  let changed = false;
  for (const r of status.items || []) {
    if (!r.request || state.done.includes(resultKey(r)) || !r.chat) continue;
    try {
      const item = r.ok ? await kaggle.fetchJson(files[`out/items/${r.id}.json`]) : null;
      const file = item ? files[`out/items/${r.kind === "image" ? `${r.id}.jpg` : item.path}`] : null;
      const rb = r.character ? (await st.bible(env, r.character)) || b : b;
      await postResult(env, state, rb, r, item, file);
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
  // Posts still on their way to Instagram/TikTok: checked on every check-in, so the
  // channel shows "posted" a minute after 👍 rather than at the next 15-min cron.
  for (const p of all) {
    const state = await st.load(env, p.slug);
    if (state.publish.some((j) => !social.finished(j)) && (await advancePublish(env, p, state, origin))) {
      await st.save(env, p.slug, state);
    }
  }
  if (body.busy) return Response.json({ job: null });
  // A job it already holds comes back first: the agent restarted mid-job.
  for (const pass of ["held", "queued"]) {
    for (const p of all) {
      const state = await st.load(env, p.slug);
      state.lanes = state.lanes || {};
      let q = (state.lanes.laptop || [])[0];
      if (!q && pass === "queued" && state.queue.length) {
        q = state.queue.shift();
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
  return Response.json(resultMessage(b, r, item));
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
  const b = (await st.bible(env, r.character || p.slug)) || { name: p.slug };
  if (r.kind === "render-done") {
    // The photos came in one by one (laptop/render-item); a finished MPT shoot
    // becomes one TikTok draft.
    // (The draft itself is made by the next cron run: uploading the photos and
    // waiting for gh-pages takes minutes, longer than the laptop waits here.)
    if (r.shoot && state.shoots?.[r.shoot]) state.shoots[r.shoot].finished = true;
  }
  else if (sent) { recordResult(state, r, item, sent); await carouselProgress(env, state, r); }
  else await postResult(env, state, b, r, item, file);
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
      jobs.push({ id: q.id, kind: "carousel", item: q.item, scene: q.scene, count: q.count || 2, replace: q.replace || null,
                  image_url: await fileLink(env, origin, q.file_id), chat: q.chat, message_id: q.message_id });
    } else if (q.kind === "render") {
      jobs.push({ id: q.id, kind: "render", scenes: q.scenes, date: q.date, chat: q.chat, videos: q.videos ?? 0,
                  shoot: q.shoot || null });
    } else if (q.kind === "redo") {
      jobs.push({ id: q.id, kind: "redo", scene: q.scene, chat: q.chat, message_id: q.message_id });
    } else if (q.kind === "recreate") {
      jobs.push({ id: q.id, kind: "recreate", url: q.url, seconds: q.seconds, chat: q.chat, message_id: q.message_id,
                  video_url: q.file_id ? await fileLink(env, origin, q.file_id) : "" });
    } else {
      jobs.push({ id: q.id, image_url: await fileLink(env, origin, q.file_id), motion: q.motion || "",
                  caption: state.items[q.item]?.caption || "", tags: state.items[q.item]?.tags || [],
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
async function shoot(env, p, who = null) {
  const state = await st.load(env, p.slug);
  const character = who || nextCharacter(p, state);
  const b = await st.bible(env, character);
  if (!b) { await st.save(env, p.slug, state); return `${p.slug}: no bible for ${character} (persona.cli sync)`; }
  // MPT stays sexy (its girls' choice, 2026-10-17); Lena's Instagram lane does not.
  const list = await scenes.write(env, b, state, Number(p.photos || 10), { sexy: p.style !== "recommendable" });
  state.recent.push(...list.map((x) => `${x.setting} / ${x.outfit} / ${x.action}`));
  const id = `shoot-${character}-${Date.now()}`;
  state.shoots = state.shoots || {};
  state.shoots[id] = { character, ids: [], started: Date.now() };
  state.queue.push({ id, kind: "render", scenes: list, date: st.today(), chat: p.chat_id, videos: 0, character, shoot: id });
  await st.save(env, p.slug, state);
  return `${p.slug}: shoot of ${character} queued (${list.length} scenes)`;
}

/** Post one shoot photo to the channel and remember it under its shoot. */
async function postShootPhoto(env, p, state, shootId, item, file, character) {
  const b = (await st.bible(env, character)) || { name: character };
  const it = { ...item, character };
  const sent = await sendMedia(env, "photo", p.chat_id, file, { caption: caption(b.name, it), buttons: buttons(it) });
  remember(state, it, sent);
  const sh = (state.shoots = state.shoots || {})[shootId] || { character, ids: [] };
  if (!sh.ids.includes(item.id)) sh.ids.push(item.id);
  state.shoots[shootId] = sh;
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
        const f = await telegramFile(env, state.items[id]?.file_id);
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
    await tg(env, "sendMessage", { chat_id: p.chat_id,
      text: `📥 ${b.name}'s shoot (${sh.ids.length} photos) is in the TikTok inbox as a draft -- add a sound and post.\n\n${posts.tiktok.text}` });
  } catch (e) {
    const msg = String(e.message || e).slice(0, 200);
    if (/too_many_pending|rate_limit|spam_risk/i.test(msg)) {
      // TikTok's inbox is full of unposted drafts: retried every hour until there
      // is room (post or delete drafts in the app), told once.
      if (!sh.retry_at) {
        await tg(env, "sendMessage", { chat_id: p.chat_id,
          text: `⏳ ${b.name}'s shoot waits for room in the TikTok inbox (too many unposted drafts). Post or delete some drafts in the app; it retries every hour.` }).catch(() => {});
      }
      sh.retry_at = Date.now() + 3600 * 1000;
      return true;
    }
    sh.draft_error = msg;
    await tg(env, "sendMessage", { chat_id: p.chat_id, text: `❌ ${b.name}'s shoot: TikTok draft failed: ${sh.draft_error}` }).catch(() => {});
  }
  return true;
}

/** The draft cron: one step of one finished shoot's TikTok draft per run. */
export async function onDraftSchedule(event, env) {
  for (const p of await personas.list(env)) {
    if (!isRoster(p)) continue;
    const state = await st.load(env, p.slug);
    const pending = Object.entries(state.shoots || {})
      .find(([, sh]) => sh.finished && !sh.draft && !sh.draft_error && !(sh.retry_at && Date.now() < sh.retry_at));
    if (!pending) continue;
    if (await shootDone(env, p, state, pending[0])) {
      // Saved onto the current state: other writers may have moved meanwhile.
      const now = await st.load(env, p.slug);
      if (now.shoots?.[pending[0]]) {
        const sh = state.shoots[pending[0]];
        Object.assign(now.shoots[pending[0]], { urls: sh.urls, draft: sh.draft, draft_error: sh.draft_error, retry_at: sh.retry_at });
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
      await shootDone(env, p, state, run.id);
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
  if (job && state.kaggle_shoots.n < Number(p.kaggle_shoots ?? 2)) {
    await pushRender(env, p, job.scenes, job.character, 0);
    state.kaggle_shoot = { id: job.shoot, character: job.character, started: Date.now() };
    state.kaggle_shoots.n += 1;
    state.queue = state.queue.filter((q) => q !== job);
    notes.push(`shoot ${job.shoot} pushed to Kaggle`);
  }
  // A shoot nobody could render today is dropped rather than piling up.
  state.queue = state.queue.filter((q) => !(q.shoot && q.date !== day && Date.now() - (state.shoots?.[q.shoot]?.started || 0) > 20 * 3600 * 1000));
  return notes;
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
      changed = (await postResults(env, state, b, jobKernel(env, p))) || changed;
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
  if (isRoster(p)) {
    const n = await advanceShoots(env, p, state, laptopUp);
    if (n.length) { notes.push(...n); changed = true; }
  }
  // The day's render waiting for a laptop that went quiet: Kaggle renders it.
  const renderJob = !isRoster(p) && state.queue.find((q) => q.kind === "render");
  if (renderJob && !laptopUp) {
    await pushRender(env, p, renderJob.scenes);
    state.queue = state.queue.filter((q) => q !== renderJob);
    state.render_via = "kaggle";
    changed = true;
    notes.push("render: laptop quiet, pushed to Kaggle");
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
    // One character per Kaggle run (her dataset is the one mounted); jobs on media
    // someone sent by hand in a roster channel never leave the laptop.
    const ready = laptopUp ? [] : state.queue.filter(lane.takes).filter((q) => !q.local_only);
    const who = ready.length ? charOf(p, ready[0]) : null;
    const batch = ready.filter((q) => charOf(p, q) === who).slice(0, lane.max);
    if (!batch.length) continue;
    const jobs = await jobPayload(env, origin, state, batch);
    await kaggle.push(env, kernel, kaggle.bootstrap({ ...basePayload(env, p, who), mode: "video", jobs }), [dataset(env, who)]);
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
  for (const p of await personas.list(env)) {
    try {
      if (isRoster(p)) {
        if ((p.shoot_hours || [5, 7, 9, 11, 13]).includes(h) && m < 15) console.log(await shoot(env, p));
      } else if ((h === 2 && m >= 30) || (h === 3 && m < 30)) console.log(await render(env, p));
      else if (POST_HOURS.includes(h) && m < 15) console.log(await post(env, p, POST_HOURS.indexOf(h)));
      console.log(await advanceJobs(env, p, origin));
      const state = await st.load(env, p.slug);
      let dirty = state.publish.some((j) => !social.finished(j)) && (await advancePublish(env, p, state, origin));
      // Numbers settle slowly: scoring once an hour is plenty.
      if (m < 15) dirty = (await learnFromPosts(env, p, state)) || dirty;
      if (dirty) await st.save(env, p.slug, state);
    } catch (err) {
      console.error(`persona ${p.slug} schedule failed`, String(err));
    }
  }
}
