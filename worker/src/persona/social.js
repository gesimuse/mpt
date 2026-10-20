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

async function gql(env, slug, query, variables) {
  const r = await fetch(API, {
    method: "POST",
    headers: { Authorization: `Bearer ${await token(env, slug)}`, "content-type": "application/json" },
    body: JSON.stringify({ query, variables }),
  });
  const d = await r.json();
  if (d.errors?.length) throw new Error(`Buffer: ${d.errors.map((e) => e.message).join("; ")}`.slice(0, 300));
  return d.data;
}

const CREATE = `mutation($input: CreatePostInput!) { createPost(input: $input) {
  __typename
  ... on PostActionSuccess { post { id status externalLink error { message } } }
  ... on NotFoundError { message } ... on UnauthorizedError { message } ... on UnexpectedError { message }
  ... on LimitReachedError { message } ... on InvalidInputError { message } } }`;

const STATUS = `query($input: PostInput!) { post(input: $input) { id status externalLink error { message } } }`;

export function accounts(p) {
  return { instagram: !!p.buffer?.instagram, tiktok: !!p.buffer?.tiktok };
}

/** Start one platform's post. Returns its record {status, post, mode, link, note}. */
async function start(env, p, job, platform, mediaUrl) {
  const channelId = p.buffer?.[platform];
  if (!channelId) return { status: "skipped", note: "not linked" };
  // Her caption and hashtags for this platform (captions.js).
  const text = job.posts?.[platform]?.text || job.caption;
  const alt = (job.posts?.alt || text.split("\n")[0]).slice(0, 200);
  const urls = Array.isArray(mediaUrl) ? mediaUrl : [mediaUrl];
  // A carousel is an Instagram "post" with several images.
  const assets = job.kind === "video" ? [{ video: { url: urls[0] } }] : urls.map((url) => ({ image: { url, metadata: { altText: alt } } }));
  let mode = "automatic", metadata;
  if (platform === "instagram") {
    // A silent video is shared from the phone instead: a viral song can only be added
    // from Instagram's own music library, in the app (a song mixed into the file gets
    // the Reel muted or blocked). The channel post gets a "Post on Instagram" button
    // (index.js sharePage) that hands the video to Instagram via the share sheet.
    if (job.kind === "video" && !job.audio) return { status: "done", mode: "phone" };
    metadata = { instagram: { type: job.kind === "video" ? "reel" : "post", shouldShareToFeed: true, isAiGenerated: true } };
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
  const queue = env.PERSONA_BUFFER_QUEUE !== "0" && mode === "automatic";
  const when = queue ? { mode: "addToQueue" }
    : job.due && job.due > Date.now() + 60 * 1000 ? { mode: "customScheduled", dueAt: new Date(job.due).toISOString() }
    : { mode: "shareNow" };
  const d = await gql(env, p.slug, CREATE, { input: {
    channelId, text, ...when, schedulingType: mode, assets, metadata, aiAssisted: true,
  } });
  const res = d.createPost;
  if (res.__typename !== "PostActionSuccess") throw new Error(`${res.__typename}: ${res.message || ""}`);
  if (queue) return { mode: "queued", post: res.post.id, status: "done", sent_at: Date.now() };
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
export async function advance(env, p, job, mediaUrl) {
  const now = Date.now();
  // Buffer rate-limits: polling every scheduled post on every 30s laptop check-in
  // got "Too many requests" (2026-10-17). Nothing to check before a post is due,
  // and after that at most every 5 minutes; a rate limit waits and retries.
  if (job.retry_at && now < job.retry_at) return job;
  for (const [key, platform] of [["ig", "instagram"], ["tt", "tiktok"]]) {
    try {
      if (job[key]?.mode === "queued" && needsRecheck({ [key]: job[key] })) await recheck(env, p, job, key);
    } catch (e) {
      if (/too many requests|rate limit|429/i.test(String(e))) { job.retry_at = now + 60 * 60 * 1000; continue; }
    }
    const cur = job[key];
    try {
      if (!cur) job[key] = await start(env, p, job, platform, mediaUrl);
      else if (cur.status === "processing") {
        if ((job.due || 0) > now || now - (cur.polled_at || 0) < 5 * 60 * 1000) continue;
        job[key] = { ...(await poll(env, p, cur)), polled_at: now };
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
export function needsRecheck(job) {
  const now = Date.now();
  return ["ig", "tt"].some((k) => {
    const r = job[k];
    return r?.mode === "queued" && !r.final && r.post && now - (r.sent_at || 0) < 14 * 86400 * 1000
      && now - (r.checked_at || r.sent_at || 0) > 6 * 3600 * 1000;
  });
}

async function recheck(env, p, job, key) {
  const r = job[key];
  const d = await gql(env, p.slug, STATUS, { input: { id: r.post } });
  const post = d.post || {};
  if (post.status === "sent") job[key] = { ...r, final: true, link: post.externalLink || "", checked_at: Date.now() };
  else if (post.status === "error") job[key] = undefined;   // made again below, with a fresh link
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
      : r.mode === "queued" ? "🗓 Instagram: in Buffer's queue -- posts at her next Buffer time slot"
      : "✅ Instagram: posted"),
    line("TikTok", job.tt, (r) => r.mode === "notification"
      ? "📲 TikTok: Buffer sent it to your phone -- add a song and post"
      : "✅ TikTok: posted with its sound"),
  ].join("\n");
}
