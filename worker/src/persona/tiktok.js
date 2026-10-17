/**
 * A finished MPT shoot goes to the aibeauty TikTok account's inbox as one photo
 * carousel draft, like the old autopilot did (tiktok.py). TikTok takes photos only
 * by URL from a verified domain, and gesimuse.github.io/mpt is the one verified for
 * this app, so the shoot's photos are hosted on the repo's gh-pages branch first
 * (kept that way 2026-10-17 rather than verifying the Worker's domain). Generated
 * photos only -- never media someone sent by hand.
 *
 * KV tt:<persona slug> holds the account's refresh token (set from
 * TIKTOK_REFRESH_TOKEN_AIBEAUTY); TIKTOK_CLIENT_KEY/SECRET are Worker secrets.
 */

import { hostOnPages } from "../github.js";

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

function b64(buf) {
  let bin = "";
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

/** Wait until GitHub Pages serves every URL (TikTok's fetch fails on a 404). */
async function waitLive(urls, timeoutMs = 240000) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    const codes = await Promise.all(urls.map((u) => fetch(u, { method: "HEAD" }).then((r) => r.status).catch(() => 0)));
    if (codes.every((c) => c === 200)) return true;
    await new Promise((res) => setTimeout(res, 10000));
  }
  return false;
}

/**
 * `photos`: [{name, bytes: ArrayBuffer}] in order. Returns the publish_id of the
 * inbox draft (the account owner adds a sound and posts it from the app).
 */
export async function photoDraft(env, slug, photos, caption) {
  const urls = [];
  for (const ph of photos.slice(0, 35)) urls.push(await hostOnPages(env, `media/${ph.name}`, b64(ph.bytes)));
  if (!(await waitLive(urls))) throw new Error("gh-pages did not serve the photos in time");
  const token = await accessToken(env, slug);
  const r = await fetch(`${TT}/post/publish/content/init/`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json; charset=UTF-8" },
    body: JSON.stringify({
      post_mode: "MEDIA_UPLOAD", media_type: "PHOTO",
      post_info: { title: caption.split("\n")[0].slice(0, 90), description: caption.slice(0, 4000) },
      source_info: { source: "PULL_FROM_URL", photo_images: urls, photo_cover_index: 0 },
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
