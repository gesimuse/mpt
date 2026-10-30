/**
 * A publish job (index.js queuePublish: a 👍, or on its own since 2026-10-30) goes to
 * her Instagram and TikTok, from the copy Telegram already holds.
 *
 *   Instagram: straight to her account at the job's slot (instagram.js publishStep),
 *     no Buffer -- its free API ran out on status checks (2026-10-27).
 *   TikTok, through Buffer, one call per post and never checked again: a 🎙 talk is
 *     published at its time (tt_due); everything else goes to Buffer as a draft, to
 *     post from the app. A persona with no TikTok of her own posts through her
 *     talk_cross persona's (Lena's through MPT's), unless it is a talk -- those reach
 *     that persona's channel anyway (index.js talkDone) and go out from there.
 *
 * Each persona can have her own Buffer account (KV buffertoken:<slug>; the shared
 * BUFFER_ACCESS_TOKEN otherwise); her entry names her channels there:
 * {buffer: {instagram: <channel id>, tiktok: <id>}}
 * (persona.cli accounts <slug> buffer). Buffer fetches the media from a signed
 * Worker link (/persona/media/).
 */

import * as personas from "./personas.js";
import * as instagram from "./instagram.js";

const API = "https://api.buffer.com";

/** Her own Buffer account's token (KV buffertoken:<slug>), else the shared BUFFER_ACCESS_TOKEN. */
async function token(env, slug) {
  return (await env.PERSONA.get(`buffertoken:${slug}`)) || env.BUFFER_ACCESS_TOKEN;
}

/**
 * One Buffer API call. Buffer's free API allows 1000 calls per 30 days per token
 * (also 600 a day, 150 per 15 min); Lena's ran out on 2026-10-27 with 17 days to the
 * reset, from status polling. Once Buffer says a window is used up, no call is made
 * until its retry-after (KV bufferblock:<slug>): each refused call counts too.
 */
async function gql(env, slug, query, variables) {
  const blocked = Number((await env.PERSONA.get(`bufferblock:${slug}`)) || 0);
  if (blocked > Date.now()) throw new Error(`Buffer rate limit: no calls until ${new Date(blocked).toISOString().slice(0, 16)}`);
  const r = await fetch(API, {
    method: "POST",
    headers: { Authorization: `Bearer ${await token(env, slug)}`, "content-type": "application/json" },
    body: JSON.stringify({ query, variables }),
  });
  const d = await r.json();
  if (d.errors?.some((e) => e.extensions?.code === "RATE_LIMIT_EXCEEDED")) {
    const wait = Number(r.headers.get("retry-after") || 3600) * 1000;
    await env.PERSONA.put(`bufferblock:${slug}`, String(Date.now() + wait), { expirationTtl: Math.max(60, Math.ceil(wait / 1000) + 60) });
  }
  if (d.errors?.length) throw new Error(`Buffer: ${d.errors.map((e) => e.message).join("; ")}`.slice(0, 300));
  return d.data;
}

const CREATE = `mutation($input: CreatePostInput!) { createPost(input: $input) {
  __typename
  ... on PostActionSuccess { post { id status externalLink error { message } } }
  ... on NotFoundError { message } ... on UnauthorizedError { message } ... on UnexpectedError { message }
  ... on LimitReachedError { message } ... on InvalidInputError { message } } }`;

const STATUS = `query($input: PostInput!) { post(input: $input) { id status dueAt externalLink error { message } } }`;

/**
 * Her Instagram or Facebook Page post through her own Buffer, published now (it is
 * only called at the job's time): one call, never checked again. Instagram goes this
 * way while she has no Instagram API token (igtoken:<slug>); 2026-10-30 Buffer raised
 * her limit to 2000 calls a month, and with no polling a post is one call.
 */
async function startBuffer(env, p, job, mediaUrl, platform) {
  const channelId = p.buffer?.[platform];
  if (platform === "facebook" && job.story) return { status: "skipped", note: "a Story stays on Instagram" };
  const text = job.posts?.[platform]?.text || job.posts?.instagram?.text || job.caption || "";
  const alt = (job.posts?.alt || text.split("\n")[0]).slice(0, 200);
  const urls = Array.isArray(mediaUrl) ? mediaUrl : [mediaUrl];
  const video = job.kind === "video";
  const metadata = platform === "instagram"
    ? { instagram: job.story ? { type: "story", shouldShareToFeed: false, isAiGenerated: true }
                             : { type: video ? "reel" : "post", shouldShareToFeed: true, isAiGenerated: true } }
    : { facebook: { type: video ? "reel" : "post" } };
  const assets = job.story ? assetsFor(job, urls.slice(0, 1), alt) : assetsFor(job, urls, alt);
  const d = await gql(env, p.slug, CREATE, { input: {
    channelId, text: job.story ? "" : text, assets, metadata, aiAssisted: true, schedulingType: "automatic", mode: "shareNow",
  } });
  const res = d.createPost;
  if (res.__typename !== "PostActionSuccess") throw new Error(`${res.__typename}: ${res.message || ""}`);
  if (res.post.status === "error") return { status: "failed", post: res.post.id, note: res.post.error?.message || "Buffer error" };
  return { mode: "buffer", post: res.post.id, status: "done", link: res.post.externalLink || "", sent_at: Date.now() };
}

export function accounts(p) {
  return { instagram: !!p.buffer?.instagram, tiktok: !!p.buffer?.tiktok || !!(p.tiktok_via || p.talk_cross) };
}

/** Where her TikTok posts go: her own Buffer channel, else her talk_cross persona's. */
async function tiktokTarget(env, p) {
  if (p.buffer?.tiktok) return { channel: p.buffer.tiktok, slug: p.slug };
  const via = (p.tiktok_via || p.talk_cross) && (await personas.get(env, p.tiktok_via || p.talk_cross));
  return via?.buffer?.tiktok ? { channel: via.buffer.tiktok, slug: via.slug, via: via.slug } : null;
}

/** Her TikTok post in Buffer: published at its time if it is a talk, else a draft. */
async function startTiktok(env, p, job, mediaUrl) {
  if (job.story) return { status: "skipped", note: "a Story is Instagram only" };
  if (job.talk && !p.buffer?.tiktok && p.talk_cross) return { status: "skipped", note: `from ${p.talk_cross}'s channel` };
  const to = await tiktokTarget(env, p);
  if (!to) return { status: "skipped", note: "not linked" };
  const text = job.posts?.tiktok?.text || job.caption;
  const alt = (job.posts?.alt || text.split("\n")[0]).slice(0, 200);
  const urls = Array.isArray(mediaUrl) ? mediaUrl : [mediaUrl];
  const live = !!(job.talk && job.audio && job.kind === "video");
  // TikTok photo posts refuse the AI flag ("do not support AI content disclosure");
  // the caption's #aigenerated stays.
  const metadata = { tiktok: { title: text.split("\n")[0].slice(0, 90), ...(job.kind === "video" ? { isAiGenerated: true } : {}) } };
  const d = await gql(env, to.slug, CREATE, { input: {
    channelId: to.channel, text, assets: assetsFor(job, urls, alt), metadata, aiAssisted: true, schedulingType: "automatic",
    ...(live ? { mode: "shareNow" } : { mode: "addToQueue", saveToDraft: true }),
  } });
  const res = d.createPost;
  if (res.__typename !== "PostActionSuccess") throw new Error(`${res.__typename}: ${res.message || ""}`);
  if (res.post.status === "error") return { status: "failed", post: res.post.id, note: res.post.error?.message || "Buffer error" };
  return { mode: live ? "sent" : "draft", post: res.post.id, status: "done", sent_at: Date.now(), via: to.via || "" };
}

/**
 * Move one publish job along, nothing before its time (`due`, a talk's TikTok at
 * `tt_due`): Instagram one step at a time (at most once a minute while Instagram
 * processes a video), TikTok one Buffer call. Failures are recorded on the job.
 */
export async function advance(env, p, job, mediaUrl) {
  const now = Date.now();
  // Left "processing" in Buffer before Instagram went direct: not posted again.
  for (const k of ["ig", "tt"]) {
    if (job[k]?.status === "processing" && !job[k].container) job[k] = { ...job[k], status: "failed", note: "left in Buffer -- check the app" };
  }
  if (!p.buffer?.instagram) job.ig = job.ig || { status: "skipped", note: "not linked" };
  else if ((!job.ig || job.ig.status === "processing") && now >= (job.due || 0) && now - (job.ig?.checked_at || 0) >= 60 * 1000) {
    try {
      // Straight through Instagram's API with her token; through her Buffer without one.
      job.ig = job.ig?.container || (await env.PERSONA.get(`igtoken:${p.slug}`))
        ? await instagram.publishStep(env, p.slug, job, mediaUrl, job.ig)
        : await startBuffer(env, p, job, mediaUrl, "instagram");
    } catch (e) {
      job.ig = { ...(job.ig || {}), status: "failed", note: String(e.message || e).slice(0, 200) };
    }
  }
  // Her Facebook Page (2026-10-30), at the same time as Instagram.
  if (!job.fb && p.buffer?.facebook && now >= (job.due || 0)) {
    try {
      job.fb = await startBuffer(env, p, job, mediaUrl, "facebook");
    } catch (e) {
      job.fb = { status: "failed", note: String(e.message || e).slice(0, 200) };
    }
  }
  if (!job.tt && now >= ((job.talk && job.tt_due) || job.due || 0) && !(job.retry_at && now < job.retry_at)) {
    try {
      job.tt = await startTiktok(env, p, job, mediaUrl);
    } catch (e) {
      const msg = String(e.message || e);
      // Buffer escalates to a 24h block when hammered: wait an hour.
      if (/too many requests|rate limit|429/i.test(msg)) job.retry_at = now + 60 * 60 * 1000;
      else job.tt = { status: "failed", note: msg.slice(0, 200) };
    }
  }
  return job;
}

const METRICS = `query($input: PostInput!) { post(input: $input) { status metrics { type value } } }`;

/** Buffer's numbers for a post: {status, metrics: [{type, value}]}. */
export async function metrics(env, p, postId) {
  return (await gql(env, p.slug, METRICS, { input: { id: postId } })).post;
}

/** One GraphQL call, for admin checks. */
export async function raw(env, p, query, variables) {
  try { return await gql(env, p.slug, query, variables); } catch (e) { return { error: String(e) }; }
}

const EDIT = `mutation($input: EditPostInput!) { editPost(input: $input) {
  __typename
  ... on PostActionSuccess { post { id status dueAt } }
  ... on NotFoundError { message } ... on UnauthorizedError { message } ... on UnexpectedError { message }
  ... on InvalidInputError { message } } }`;

function assetsFor(job, urls, alt) {
  return job.kind === "video" ? [{ video: { url: urls[0] } }] : urls.map((url) => ({ image: { url, metadata: { altText: alt } } }));
}

/**
 * Fresh media links on a post already in Buffer, keeping its place: a queued post
 * whose link was made before links lasted 45 days (they died after 24h), or one that
 * failed at its slot -- that one is sent now (2026-10-21: a re-created post went to
 * the back of the queue, days later).
 */
async function refresh(env, p, job, key, mediaUrl, post, sendNow) {
  const urls = Array.isArray(mediaUrl) ? mediaUrl : [mediaUrl];
  const alt = (job.posts?.alt || (job.posts?.instagram?.text || job.caption || "").split("\n")[0]).slice(0, 200);
  const when = sendNow ? { mode: "shareNow" } : { mode: "customScheduled", dueAt: post.dueAt };
  const metadata = key === "ig"
    ? { instagram: { type: job.kind === "video" ? "reel" : "post", shouldShareToFeed: true, isAiGenerated: true } }
    : { tiktok: { title: (job.posts?.tiktok?.text || job.caption || "").split("\n")[0].slice(0, 90) } };
  const d = await gql(env, p.slug, EDIT, { input: { id: job[key].post, assets: assetsFor(job, urls, alt), metadata, ...when } });
  const res = d.editPost;
  if (res.__typename !== "PostActionSuccess") throw new Error(`${res.__typename}: ${res.message || ""}`);
  job[key] = { ...job[key], links: 2, checked_at: Date.now(), ...(sendNow ? { resent: (job[key].resent || 0) + 1 } : {}) };
}

/** Admin: a queued post moves to `dueAt` (ISO). */
export async function moveTo(env, p, job, key, mediaUrl, dueAt) {
  await refresh(env, p, job, key, mediaUrl, { dueAt }, false);
}

/** Admin: a queued post goes out now. */
export async function sendNow(env, p, job, key, mediaUrl) {
  const post = (await gql(env, p.slug, STATUS, { input: { id: job[key].post } })).post || {};
  await refresh(env, p, job, key, mediaUrl, post, true);
}

// A Facebook-only job (the Page's backfill) waits for its fb; others post it in the same pass as ig.
export const finished = (job) => ["ig", "tt"].every((k) => job[k] && job[k].status !== "processing") && !(job.fb_only && !job.fb);

/** The status lines added to the post's caption in the channel. */
export function statusLine(job) {
  const at = (t) => new Date(t).toLocaleString("en-GB", { timeZone: "Europe/Prague", weekday: "short", hour: "2-digit", minute: "2-digit" });
  const line = (name, rec, due, doneText) => {
    if (!rec) return due > Date.now() + 60 * 1000 ? `🗓 ${name}: posts ${at(due)} (Prague time) · 👎 cancels` : `⏳ ${name}: starting…`;
    if (rec.status === "done") return `${doneText(rec)}${rec.link ? `\n${rec.link}` : ""}`;
    if (rec.status === "failed") return `❌ ${name}: ${rec.note}`;
    if (rec.status === "skipped") return `— ${name}: ${rec.note || "not linked"}`;
    return `⏳ ${name}: posting…`;
  };
  return [
    line("Instagram", job.ig, job.due, (r) => r.mode === "phone"
      ? "📲 Instagram: tap «Post on Instagram» below on your phone, add a trending song, share"
      : r.mode === "queued" ? "🗓 Instagram: in Buffer's queue"
      : job.story ? "✅ Instagram Story: posted" : "✅ Instagram: posted"),
    ...(job.fb ? [line("Facebook", job.fb, job.due, () => "✅ Facebook: posted")] : []),
    line("TikTok", job.tt, (job.talk && job.tt_due) || job.due, (r) => r.mode === "draft"
      ? `📝 TikTok: a draft in ${r.via ? `${r.via}'s ` : ""}Buffer -- post it from the app`
      : r.mode === "notification" ? "📲 TikTok: Buffer sent it to your phone -- add a song and post"
      : r.mode === "queued" ? "🗓 TikTok: in Buffer's queue"
      : `✅ TikTok: posted${r.via ? ` on ${r.via}` : ""} with its sound`),
  ].join("\n");
}
