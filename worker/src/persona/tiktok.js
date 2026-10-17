/**
 * A finished MPT shoot goes to the aibeauty TikTok account's inbox as one photo
 * carousel draft, like the old autopilot did (tiktok.py). TikTok takes photos only
 * by URL from a verified URL prefix: they are served from Telegram's own copy
 * through this Worker's signed /persona/media/ links (that prefix verified in the
 * TikTok developer portal, file served by verifyFile) -- nothing on gh-pages.
 *
 * KV tt:<persona slug> holds the account's refresh token (set from
 * TIKTOK_REFRESH_TOKEN_AIBEAUTY); TIKTOK_CLIENT_KEY/SECRET are Worker secrets.
 */

const TT = "https://open.tiktokapis.com/v2";

async function accessToken(env, slug) {
  const raw = await env.PERSONA.get(`tt:${slug}`);
  if (!raw) throw new Error("no TikTok account linked");
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

/**
 * `urls`: signed /persona/media/ links to the shoot's photos, in order. Returns the
 * publish_id of the inbox draft (the account owner adds a sound and posts it).
 */
export async function photoDraft(env, slug, urls, caption) {
  const token = await accessToken(env, slug);
  const r = await fetch(`${TT}/post/publish/content/init/`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json; charset=UTF-8" },
    body: JSON.stringify({
      post_mode: "MEDIA_UPLOAD", media_type: "PHOTO",
      post_info: { title: caption.split("\n")[0].slice(0, 90), description: caption.slice(0, 4000) },
      source_info: { source: "PULL_FROM_URL", photo_images: urls.slice(0, 35), photo_cover_index: 0 },
    }),
  });
  const d = await r.json();
  if (d.error?.code && d.error.code !== "ok") throw new Error(`TikTok draft: ${d.error.code} ${d.error.message || ""}`.slice(0, 300));
  return d.data?.publish_id;
}

/** TikTok's URL-prefix verification file, served at /persona/media/<name>.txt. */
export async function verifyFile(env, name) {
  const body = await env.PERSONA.get(`ttverify:${name}`);
  return body ? new Response(body, { headers: { "content-type": "text/plain" } }) : new Response("not found", { status: 404 });
}
