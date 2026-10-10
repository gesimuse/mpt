/**
 * 👍 in a persona's channel publishes the photo/video to her Instagram and TikTok
 * through Buffer (api.buffer.com, GraphQL), from the copy Telegram already holds.
 *
 *   Instagram: published now, public (a photo post or a Reel), labelled AI.
 *   TikTok, video WITH sound (🎭 recreates keep the original's audio): published
 *     now, public, with that sound (Buffer's "automatic" scheduling), labelled AI.
 *   TikTok, no sound (🎬 videos, photos): Buffer's "notification" scheduling --
 *     a push on the phone opens TikTok with the media, to add a song and post.
 *
 * One Buffer account (BUFFER_ACCESS_TOKEN) holds every persona's channels; each
 * persona entry names hers: {buffer: {instagram: <channel id>, tiktok: <id>}}
 * (persona.cli accounts <slug> buffer). Buffer fetches the media from a signed
 * Worker link (/persona/media/).
 */

const API = "https://api.buffer.com";

async function gql(env, query, variables) {
  const r = await fetch(API, {
    method: "POST",
    headers: { Authorization: `Bearer ${env.BUFFER_ACCESS_TOKEN}`, "content-type": "application/json" },
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
  if (!channelId || !env.BUFFER_ACCESS_TOKEN) return { status: "skipped", note: "not linked" };
  const asset = job.kind === "photo"
    ? { image: { url: mediaUrl, metadata: { altText: job.caption.split("\n")[0].slice(0, 100) } } }
    : { video: { url: mediaUrl } };
  let mode = "automatic", metadata;
  if (platform === "instagram") {
    metadata = { instagram: { type: job.kind === "photo" ? "post" : "reel", shouldShareToFeed: true, isAiGenerated: true } };
  } else {
    // Without sound it goes to the phone, to add a song before posting.
    if (!(job.kind === "video" && job.audio)) mode = "notification";
    // TikTok photo posts refuse the AI flag ("do not support AI content
    // disclosure"); the caption's #aigenerated stays.
    metadata = { tiktok: { title: job.caption.split("\n")[0].slice(0, 90),
                           ...(job.kind === "photo" ? {} : { isAiGenerated: true }) } };
  }
  const d = await gql(env, CREATE, { input: {
    channelId, text: job.caption, mode: "shareNow", schedulingType: mode, assets: [asset], metadata, aiAssisted: true,
  } });
  const res = d.createPost;
  if (res.__typename !== "PostActionSuccess") throw new Error(`${res.__typename}: ${res.message || ""}`);
  return settle({ mode, post: res.post.id }, res.post);
}

function settle(rec, post) {
  if (post.status === "error") return { ...rec, status: "failed", note: post.error?.message || "Buffer error" };
  if (post.status === "sent") return { ...rec, status: "done", link: post.externalLink || "" };
  // A notification post waits for you on the phone: nothing more to watch here.
  if (rec.mode === "notification") return { ...rec, status: "done" };
  return { ...rec, status: "processing" };
}

async function poll(env, rec) {
  const d = await gql(env, STATUS, { input: { id: rec.post } });
  return settle(rec, d.post);
}

/**
 * Move one publish job along: start each platform once, then poll until both are
 * done or failed. Failures are recorded on the job, not thrown.
 */
export async function advance(env, p, job, mediaUrl) {
  for (const [key, platform] of [["ig", "instagram"], ["tt", "tiktok"]]) {
    const cur = job[key];
    try {
      if (!cur) job[key] = await start(env, p, job, platform, mediaUrl);
      else if (cur.status === "processing") job[key] = await poll(env, cur);
    } catch (e) {
      job[key] = { ...(cur || {}), status: "failed", note: String(e.message || e).slice(0, 200) };
    }
  }
  job.tries = (job.tries || 0) + 1;
  // Give up watching after ~6h of cron ticks.
  for (const key of ["ig", "tt"]) {
    if (job[key]?.status === "processing" && job.tries > 24) job[key] = { ...job[key], status: "failed", note: "timed out" };
  }
  return job;
}

export const finished = (job) => ["ig", "tt"].every((k) => job[k] && job[k].status !== "processing");

/** The status lines added to the post's caption in the channel. */
export function statusLine(job) {
  const line = (name, rec, doneText) => {
    if (!rec) return `⏳ ${name}: starting…`;
    if (rec.status === "done") return `${doneText(rec)}${rec.link ? `\n${rec.link}` : ""}`;
    if (rec.status === "failed") return `❌ ${name}: ${rec.note}`;
    if (rec.status === "skipped") return `— ${name}: not linked`;
    return `⏳ ${name}: posting…`;
  };
  return [
    line("Instagram", job.ig, () => "✅ Instagram: posted"),
    line("TikTok", job.tt, (r) => r.mode === "notification"
      ? "📲 TikTok: Buffer sent it to your phone -- add a song and post"
      : "✅ TikTok: posted with its sound"),
  ].join("\n");
}
