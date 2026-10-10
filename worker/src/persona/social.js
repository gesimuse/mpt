/**
 * 👍 in a persona's channel publishes the photo/video to her own Instagram and
 * TikTok, from the copy Telegram already holds.
 *
 * Instagram (Instagram API with Instagram Login): public post, labelled
 * is_ai_generated. A photo or a Reel container is made from a signed Worker link,
 * then published once Instagram has processed it.
 *
 * TikTok (Content Posting API, the same app as the aibeauty account):
 *   - a video WITH sound (🎭 recreates keep the original's audio) is posted public
 *     with that sound, labelled is_aigc -- once the app has passed TikTok's audit
 *     (persona config tiktok_direct). Before the audit TikTok only allows private
 *     (SELF_ONLY) direct posts, so it goes to her inbox as a draft instead.
 *   - a video without sound (🎬) goes to her inbox as a draft, to add a song.
 *   - a photo goes as a photo draft too (TikTok takes photos only by URL, from a
 *     verified URL prefix: /persona/media/, see verifyFile()).
 * Inbox drafts carry no caption (TikTok's inbox endpoint takes none), so the
 * caption is shown in the channel to paste.
 *
 * Credentials live in KV, per persona: tt:<slug> {refresh_token, access_token,
 * expires_at}, ig:<slug> {token, user_id, refreshed}. Set through
 * POST /persona/admin/account (persona/cli.py accounts).
 */

const TT = "https://open.tiktokapis.com/v2";
const IG = "https://graph.instagram.com/v23.0";

// ------------------------------------------------------------------ tokens
async function tiktokToken(env, slug) {
  const raw = await env.PERSONA.get(`tt:${slug}`);
  if (!raw) return null;
  const t = JSON.parse(raw);
  if (t.access_token && t.expires_at > Date.now() + 10 * 60 * 1000) return t.access_token;
  const r = await fetch(`${TT}/oauth/token/`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ client_key: env.TIKTOK_CLIENT_KEY, client_secret: env.TIKTOK_CLIENT_SECRET,
                                grant_type: "refresh_token", refresh_token: t.refresh_token }),
  });
  const d = await r.json();
  if (!d.access_token) throw new Error(`TikTok token refresh: ${JSON.stringify(d).slice(0, 200)}`);
  await env.PERSONA.put(`tt:${slug}`, JSON.stringify({ refresh_token: d.refresh_token || t.refresh_token,
    access_token: d.access_token, expires_at: Date.now() + (d.expires_in || 86400) * 1000 }));
  return d.access_token;
}

async function instagramToken(env, slug) {
  const raw = await env.PERSONA.get(`ig:${slug}`);
  if (!raw) return null;
  const t = JSON.parse(raw);
  // Long-lived tokens last 60 days; refreshed every 20.
  if (Date.now() - (t.refreshed || 0) > 20 * 86400 * 1000) {
    const r = await fetch(`https://graph.instagram.com/refresh_access_token?grant_type=ig_refresh_token&access_token=${t.token}`);
    const d = await r.json();
    if (d.access_token) {
      t.token = d.access_token;
      t.refreshed = Date.now();
      await env.PERSONA.put(`ig:${slug}`, JSON.stringify(t));
    }
  }
  return t;
}

export async function accounts(env, slug) {
  return { tiktok: !!(await env.PERSONA.get(`tt:${slug}`)) && !!env.TIKTOK_CLIENT_KEY,
           instagram: !!(await env.PERSONA.get(`ig:${slug}`)) };
}

// ----------------------------------------------------------------- TikTok
async function ttCall(token, path, body) {
  const r = await fetch(`${TT}${path}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json; charset=UTF-8" },
    body: JSON.stringify(body),
  });
  const d = await r.json();
  if (d.error?.code && d.error.code !== "ok") throw new Error(`TikTok ${path}: ${d.error.code} ${d.error.message || ""}`.slice(0, 300));
  return d.data || {};
}

/** Start the TikTok side of a publish job. Returns its tt record. */
async function tiktokStart(env, p, job, bytes, mediaUrl) {
  const token = await tiktokToken(env, p.slug);
  if (!token) return { status: "skipped", note: "no TikTok account linked" };
  if (job.kind === "photo") {
    const d = await ttCall(token, "/post/publish/content/init/", {
      post_mode: "MEDIA_UPLOAD", media_type: "PHOTO",
      post_info: { title: job.caption.split("\n")[0].slice(0, 90), description: job.caption.slice(0, 4000) },
      source_info: { source: "PULL_FROM_URL", photo_images: [mediaUrl], photo_cover_index: 0 },
    });
    return { status: "processing", mode: "draft", publish_id: d.publish_id };
  }
  const size = bytes.byteLength;
  const source = { source: "FILE_UPLOAD", video_size: size, chunk_size: size, total_chunk_count: 1 };
  let mode = "draft", d;
  if (job.audio && p.tiktok_direct) {
    const info = await ttCall(token, "/post/publish/creator_info/query/", {});
    if ((info.privacy_level_options || []).includes("PUBLIC_TO_EVERYONE")) {
      d = await ttCall(token, "/post/publish/video/init/", {
        post_info: { title: job.caption.slice(0, 2200), privacy_level: "PUBLIC_TO_EVERYONE", is_aigc: true },
        source_info: source,
      });
      mode = "public";
    }
  }
  if (!d) d = await ttCall(token, "/post/publish/inbox/video/init/", { source_info: source });
  const up = await fetch(d.upload_url, {
    method: "PUT",
    headers: { "content-type": "video/mp4", "content-range": `bytes 0-${size - 1}/${size}` },
    body: bytes,
  });
  if (!up.ok) throw new Error(`TikTok upload ${up.status}: ${(await up.text()).slice(0, 200)}`);
  return { status: "processing", mode, publish_id: d.publish_id };
}

async function tiktokPoll(env, p, tt) {
  const token = await tiktokToken(env, p.slug);
  const d = await ttCall(token, "/post/publish/status/fetch/", { publish_id: tt.publish_id });
  if (d.status === "SEND_TO_USER_INBOX") return { ...tt, status: "done" };
  if (d.status === "PUBLISH_COMPLETE") return { ...tt, status: "done" };
  if (d.status === "FAILED") return { ...tt, status: "failed", note: d.fail_reason || "failed" };
  return tt;
}

// --------------------------------------------------------------- Instagram
async function igCall(method, url, params) {
  const r = await fetch(method === "GET" ? `${url}?${new URLSearchParams(params)}` : url, method === "GET" ? {} : {
    method: "POST", body: new URLSearchParams(params),
  });
  const d = await r.json();
  if (d.error) throw new Error(`Instagram: ${d.error.message || JSON.stringify(d.error)}`.slice(0, 300));
  return d;
}

async function instagramStart(env, p, job, mediaUrl) {
  const t = await instagramToken(env, p.slug);
  if (!t) return { status: "skipped", note: "no Instagram account linked" };
  const params = { access_token: t.token, caption: job.caption.slice(0, 2200), is_ai_generated: "true" };
  if (job.kind === "photo") params.image_url = mediaUrl;
  else Object.assign(params, { media_type: "REELS", video_url: mediaUrl, share_to_feed: "true" });
  const d = await igCall("POST", `${IG}/${t.user_id}/media`, params);
  return { status: "processing", container: d.id };
}

async function instagramPoll(env, p, ig) {
  const t = await instagramToken(env, p.slug);
  const s = await igCall("GET", `${IG}/${ig.container}`, { fields: "status_code,status", access_token: t.token });
  if (s.status_code === "ERROR" || s.status_code === "EXPIRED") return { ...ig, status: "failed", note: s.status || s.status_code };
  if (s.status_code !== "FINISHED") return ig;
  const d = await igCall("POST", `${IG}/${t.user_id}/media_publish`, { creation_id: ig.container, access_token: t.token });
  const m = await igCall("GET", `${IG}/${d.id}`, { fields: "permalink", access_token: t.token }).catch(() => ({}));
  return { ...ig, status: "done", media_id: d.id, link: m.permalink || "" };
}

// --------------------------------------------------------------- the job
/**
 * Move one publish job along: start each platform once, then poll until both are
 * done or failed. `media` = { bytes(), url } for the Telegram copy. Returns the
 * updated job; failures are recorded on it, not thrown.
 */
export async function advance(env, p, job, media) {
  for (const [key, start, poll] of [["ig", instagramStart, instagramPoll], ["tt", tiktokStart, tiktokPoll]]) {
    const cur = job[key];
    try {
      if (!cur) {
        job[key] = key === "ig" ? await start(env, p, job, media.url)
                                : await start(env, p, job, job.kind === "photo" ? null : await media.bytes(), media.url);
      } else if (cur.status === "processing") {
        job[key] = await poll(env, p, cur);
      }
    } catch (e) {
      job[key] = { ...(cur || {}), status: "failed", note: String(e.message || e).slice(0, 200) };
    }
  }
  job.tries = (job.tries || 0) + 1;
  // Give up polling after ~6h of cron ticks.
  for (const key of ["ig", "tt"]) {
    if (job[key]?.status === "processing" && job.tries > 24) job[key] = { ...job[key], status: "failed", note: "timed out" };
  }
  return job;
}

export const finished = (job) => ["ig", "tt"].every((k) => job[k] && job[k].status !== "processing");

/** The status line added to the post's caption in the channel. */
export function statusLine(job) {
  const ig = job.ig || {}, tt = job.tt || {};
  const igText = { done: "✅ Instagram: posted", processing: "⏳ Instagram: posting…", skipped: "— Instagram: not linked" }[ig.status]
    || (ig.status === "failed" ? `❌ Instagram: ${ig.note}` : "⏳ Instagram: starting…");
  let ttText;
  if (tt.status === "done") ttText = tt.mode === "public" ? "✅ TikTok: posted with its sound"
    : job.kind === "video" && job.audio ? "📥 TikTok: draft in her inbox, with its sound (tap Post)"
    : "📥 TikTok: draft in her inbox (add a song, then Post)";
  else if (tt.status === "failed") ttText = `❌ TikTok: ${tt.note}`;
  else if (tt.status === "skipped") ttText = "— TikTok: not linked";
  else ttText = "⏳ TikTok: uploading…";
  return `${igText}\n${ttText}`;
}

/** TikTok's URL-prefix verification file, served at /persona/media/<name>.txt. */
export async function verifyFile(env, name) {
  const body = await env.PERSONA.get(`ttverify:${name}`);
  return body ? new Response(body, { headers: { "content-type": "text/plain" } }) : new Response("not found", { status: 404 });
}
