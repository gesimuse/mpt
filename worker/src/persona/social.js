/**
 * 👍 in a persona's channel publishes the photo/video to her Instagram and TikTok
 * through Buffer (api.buffer.com, GraphQL), from the copy Telegram already holds.
 *
 *   Instagram: published now, public (a photo post or a Reel with its own sound),
 *     labelled AI. A silent video is shared from the phone instead (a button on the
 *     channel post), to add a trending song from Instagram's library.
 *   TikTok, video WITH sound (🎭 recreates keep the original's audio): published
 *     now, public, with that sound (Buffer's "automatic" scheduling), labelled AI.
 *   TikTok, no sound (🎬 videos, photos): Buffer's "notification" scheduling --
 *     a push on the phone opens TikTok with the media, to add a song and post.
 *
 * Each persona can have her own Buffer account (KV buffertoken:<slug>; the shared
 * BUFFER_ACCESS_TOKEN otherwise); her entry names her channels there:
 * {buffer: {instagram: <channel id>, tiktok: <id>}}
 * (persona.cli accounts <slug> buffer). Buffer fetches the media from a signed
 * Worker link (/persona/media/).
 */

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

export function accounts(p) {
  return { instagram: !!p.buffer?.instagram, tiktok: !!p.buffer?.tiktok };
}

/** Start one platform's post. Returns its record {status, post, mode, link, note}. */
async function start(env, p, job, platform, mediaUrl) {
  const channelId = p.buffer?.[platform];
  if (!channelId) return { status: "skipped", note: "not linked" };
  if (job.story && platform !== "instagram") return { status: "skipped", note: "a Story is Instagram only" };
  // Her caption and hashtags for this platform (captions.js).
  const text = job.posts?.[platform]?.text || job.caption;
  const alt = (job.posts?.alt || text.split("\n")[0]).slice(0, 200);
  const urls = Array.isArray(mediaUrl) ? mediaUrl : [mediaUrl];
  // A carousel is an Instagram "post" with several images.
  const assets = job.kind === "video" ? [{ video: { url: urls[0] } }]
    : (platform === "instagram" && job.story ? urls.slice(0, 1) : urls).map((url) => ({ image: { url, metadata: { altText: alt } } }));
  let mode = "automatic", metadata;
  if (platform === "instagram") {
    // A silent video is shared from the phone instead: a viral song can only be added
    // from Instagram's own music library, in the app (a song mixed into the file gets
    // the Reel muted or blocked). The channel post gets a "Post on Instagram" button
    // (index.js sharePage) that hands the video to Instagram via the share sheet.
    if (job.kind === "video" && !job.audio) return { status: "done", mode: "phone" };
    // Over the week's feed cap (index.js feedFull) a photo or carousel becomes a Story
    // instead: the feed is kept for Reels, which reach people who don't follow her.
    metadata = { instagram: job.story ? { type: "story", shouldShareToFeed: false, isAiGenerated: true }
      : { type: job.kind === "video" ? "reel" : "post", shouldShareToFeed: true, isAiGenerated: true } };
  } else {
    // Without sound it goes to the phone, to add a song before posting.
    if (!(job.kind === "video" && job.audio)) mode = "notification";
    // TikTok photo posts refuse the AI flag ("do not support AI content
    // disclosure"); the caption's #aigenerated stays.
    metadata = { tiktok: { title: text.split("\n")[0].slice(0, 90),
                           ...(job.kind === "video" ? { isAiGenerated: true } : {}) } };
  }
  // Into Buffer's own queue: it posts at the times in her Buffer posting schedule,
  // and nothing here polls it (polling scheduled posts got "Too many requests",
  // 2026-10-17). PERSONA_BUFFER_QUEUE=0 goes back to our own slots.
  // A talk on TikTok goes out at its own spaced time (index.js nextTalkSlot), not at
  // Buffer's slots; looked at like a queued post (twice), not polled.
  const talkAt = platform === "tiktok" && job.talk && job.tt_due;
  const queue = env.PERSONA_BUFFER_QUEUE !== "0" && mode === "automatic" && !talkAt;
  const when = talkAt ? (talkAt > Date.now() + 60 * 1000 ? { mode: "customScheduled", dueAt: new Date(talkAt).toISOString() } : { mode: "shareNow" })
    : queue ? { mode: "addToQueue" }
    : job.due && job.due > Date.now() + 60 * 1000 ? { mode: "customScheduled", dueAt: new Date(job.due).toISOString() }
    : { mode: "shareNow" };
  const d = await gql(env, p.slug, CREATE, { input: {
    channelId, text, ...when, schedulingType: mode, assets, metadata, aiAssisted: true,
  } });
  const res = d.createPost;
  if (res.__typename !== "PostActionSuccess") throw new Error(`${res.__typename}: ${res.message || ""}`);
  if (queue || (talkAt && res.post.status !== "sent" && res.post.status !== "error"))
    return { mode: "queued", post: res.post.id, status: "done", sent_at: Date.now(), links: 2, ...(talkAt ? { at: talkAt } : {}) };
  return settle({ mode, post: res.post.id }, res.post);
}

function settle(rec, post) {
  if (post.status === "error") return { ...rec, status: "failed", note: post.error?.message || "Buffer error" };
  if (post.status === "sent") return { ...rec, status: "done", link: post.externalLink || "", sent_at: rec.sent_at || Date.now() };
  // A notification post waits for you on the phone: nothing more to watch here.
  if (rec.mode === "notification") return { ...rec, status: "done", sent_at: rec.sent_at || Date.now() };
  return { ...rec, status: "processing" };
}

async function poll(env, p, rec) {
  const d = await gql(env, p.slug, STATUS, { input: { id: rec.post } });
  return settle(rec, d.post);
}

/**
 * Move one publish job along: start each platform once, then poll until both are
 * done or failed. Failures are recorded on the job, not thrown.
 */
const POLL_AT = [10, 30, 120, 360].map((m) => m * 60 * 1000);

export async function advance(env, p, job, mediaUrl) {
  const now = Date.now();
  // Buffer rate-limits: polling every scheduled post on every 30s laptop check-in
  // got "Too many requests" (2026-10-17). Nothing to check before a post is due,
  // and after that at most every 5 minutes; a rate limit waits and retries.
  if (job.retry_at && now < job.retry_at) return job;
  for (const [key, platform] of [["ig", "instagram"], ["tt", "tiktok"]]) {
    try {
      if (job[key]?.mode === "queued" && needsRecheck({ due: (key === "tt" && job.tt_due) || job.due, [key]: job[key] })) await recheck(env, p, job, key, mediaUrl);
    } catch (e) {
      if (/too many requests|rate limit|429/i.test(String(e))) { job.retry_at = now + 60 * 60 * 1000; continue; }
      if (job[key]) job[key] = { ...job[key], recheck_error: String(e.message || e).slice(0, 200), checked_at: now };
    }
    const cur = job[key];
    try {
      if (!cur) job[key] = await start(env, p, job, platform, mediaUrl);
      else if (cur.status === "processing") {
        // Asked 10 min, 30 min, 2h and 6h after it was due (was every 5 min for 6h).
        const n = cur.polls || 0;
        if (n >= POLL_AT.length) {
          job[key] = { ...cur, status: "failed", note: "Buffer didn't confirm it within 6h -- check the app" };
          continue;
        }
        if (now < (job.due || cur.sent_at || 0) + POLL_AT[n]) continue;
        job[key] = { ...(await poll(env, p, cur)), polled_at: now, polls: n + 1 };
      }
    } catch (e) {
      const msg = String(e.message || e);
      if (/too many requests|rate limit|429/i.test(msg)) {
        job.retry_at = now + 60 * 60 * 1000;  // Buffer escalates to a 24h block when hammered
        continue;
      }
      job[key] = { ...(cur || {}), status: "failed", note: msg.slice(0, 200) };
    }
  }
  job.tries = (job.tries || 0) + 1;
  // Give up watching 6h after it was due.
  for (const key of ["ig", "tt"]) {
    if (job[key]?.status === "processing" && Date.now() > (job.due || job.created || 0) + 6 * 3600 * 1000 && job.tries > 24) {
      job[key] = { ...job[key], status: "failed", note: "timed out" };
    }
  }
  return job;
}

const METRICS = `query($input: PostInput!) { post(input: $input) { status metrics { type value } } }`;

/** Buffer's numbers for a post: {status, metrics: [{type, value}]}. */
export async function metrics(env, p, postId) {
  return (await gql(env, p.slug, METRICS, { input: { id: postId } })).post;
}

/**
 * A post handed to Buffer's queue is looked at again every 6h until Buffer has sent
 * it (for 14 days): one that failed at its slot (an expired media link, 2026-10-20)
 * is made again with a fresh link.
 */
/** One GraphQL call, for admin checks. */
export async function raw(env, p, query, variables) {
  try { return await gql(env, p.slug, query, variables); } catch (e) { return { error: String(e) }; }
}

/** A post handed to Buffer's queue is looked at twice: an hour after it was due, and a
 * day after (was every 6h for 14 days -- most of the 1000 calls a month). */
export function needsRecheck(job) {
  const now = Date.now();
  return ["ig", "tt"].some((k) => {
    const r = job[k];
    if (r?.mode !== "queued" || r.final || !r.post) return false;
    const base = job.due || r.sent_at || 0, c = r.checked_at || 0;
    return (!c && now > base + 3600 * 1000) || (c && c < base + 24 * 3600 * 1000 && now > base + 24 * 3600 * 1000);
  });
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

async function recheck(env, p, job, key, mediaUrl) {
  const r = job[key];
  let post;
  try {
    post = (await gql(env, p.slug, STATUS, { input: { id: r.post } })).post;
  } catch (e) {
    if (/too many requests|rate limit|429/i.test(String(e))) throw e;
    post = null;
  }
  if (!post) {
    // Gone from Buffer (failed and removed, or deleted there): shown on the post with
    // a 🔁 Retry posting button, so you see which and can send it again.
    job[key] = { ...r, status: "failed", note: "not in Buffer any more -- tap 🔁 Retry posting", final: true };
    return;
  }
  if (post.status === "sent") job[key] = { ...r, final: true, link: post.externalLink || "", checked_at: Date.now() };
  else if (post.status === "error") {
    if ((r.resent || 0) >= 2) job[key] = { ...r, status: "failed", final: true, note: post.error?.message || "Buffer error" };
    else await refresh(env, p, job, key, mediaUrl, post, true);
  } else if (r.links !== 2) await refresh(env, p, job, key, mediaUrl, post, false);
  else job[key] = { ...r, checked_at: Date.now() };
}

export const finished = (job) => ["ig", "tt"].every((k) => job[k] && job[k].status !== "processing");

/** The status lines added to the post's caption in the channel. */
export function statusLine(job) {
  const line = (name, rec, doneText) => {
    if (!rec) return `⏳ ${name}: starting…`;
    if (rec.status === "processing" && job.due > Date.now() + 60 * 1000) {
      const t = new Date(job.due).toLocaleString("en-GB", { timeZone: "Europe/Prague", weekday: "short", hour: "2-digit", minute: "2-digit" });
      return `🗓 ${name}: scheduled for ${t} (Prague time)`;
    }
    if (rec.status === "done") return `${doneText(rec)}${rec.link ? `\n${rec.link}` : ""}`;
    if (rec.status === "failed") return `❌ ${name}: ${rec.note}`;
    if (rec.status === "skipped") return `— ${name}: not linked`;
    return `⏳ ${name}: posting…`;
  };
  return [
    line("Instagram", job.ig, (r) => r.mode === "phone"
      ? "📲 Instagram: tap «Post on Instagram» below on your phone, add a trending song, share"
      : r.mode === "queued" && job.story ? "🗓 Instagram Story: in Buffer's queue (this week's feed is kept for Reels)"
      : r.mode === "queued" ? "🗓 Instagram: in Buffer's queue -- posts at her next Buffer time slot"
      : "✅ Instagram: posted"),
    line("TikTok", job.tt, (r) => r.mode === "notification"
      ? "📲 TikTok: Buffer sent it to your phone -- add a song and post"
      : r.mode === "queued" && r.at && !r.final ? `🗓 TikTok: posts ${new Date(r.at).toLocaleString("en-GB", { timeZone: "Europe/Bucharest", weekday: "short", hour: "2-digit", minute: "2-digit" })}`
      : r.mode === "queued" && !r.final ? "🗓 TikTok: in Buffer's queue"
      : "✅ TikTok: posted with its sound"),
  ].join("\n");
}
