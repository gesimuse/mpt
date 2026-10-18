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

/**
 * Put every photo on gh-pages in ONE commit (git data API): one file per commit
 * meant ten queued Pages builds, and they were not live within 4 minutes
 * (2026-10-17, the first shoot's draft failed). Returns the photos' Pages URLs.
 */
async function hostAll(env, photos) {
  const api = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}`;
  const h = { Authorization: `Bearer ${env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json",
              "User-Agent": "mpt-persona-worker", "content-type": "application/json" };
  const gh = async (path, method = "GET", body) => {
    const r = await fetch(`${api}${path}`, { method, headers: h, body: body && JSON.stringify(body) });
    if (!r.ok) throw new Error(`GitHub ${method} ${path}: ${r.status} ${(await r.text()).slice(0, 150)}`);
    return r.json();
  };
  const tree = [];
  for (const ph of photos) {
    const blob = await gh("/git/blobs", "POST", { content: b64(ph.bytes), encoding: "base64" });
    tree.push({ path: `media/${ph.name}`, mode: "100644", type: "blob", sha: blob.sha });
  }
  // The autopilot also commits to gh-pages: if the branch moved under us ("not a
  // fast forward"), build the commit again on its new head.
  for (let attempt = 0; ; attempt++) {
    const ref = await gh("/git/ref/heads/gh-pages");
    const head = await gh(`/git/commits/${ref.object.sha}`);
    const t = await gh("/git/trees", "POST", { base_tree: head.tree.sha, tree });
    const c = await gh("/git/commits", "POST", { message: `media: ${photos.length} shoot photos`, tree: t.sha, parents: [ref.object.sha] });
    try {
      await gh("/git/refs/heads/gh-pages", "PATCH", { sha: c.sha });
      break;
    } catch (e) {
      if (attempt >= 2 || !/fast forward/i.test(String(e))) throw e;
    }
  }
  const base = env.PAGES_BASE_URL.replace(/\/$/, "");
  return photos.map((ph) => `${base}/media/${ph.name}`);
}

export { hostAll };

/** Whether gh-pages already serves the photos (the last URL stands for the commit). */
export async function live(urls) {
  return fetch(urls[urls.length - 1], { method: "HEAD" }).then((r) => r.status === 200).catch(() => false);
}

/** The inbox draft from photos already live on gh-pages (`urls`). */
export async function photoDraft(env, slug, urls, caption) {
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

/** What TikTok did with a draft: {status, fail_reason, ...} (post/publish/status/fetch). */
export async function draftStatus(env, slug, publishId) {
  const token = await accessToken(env, slug);
  const r = await fetch(`${TT}/post/publish/status/fetch/`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "content-type": "application/json; charset=UTF-8" },
    body: JSON.stringify({ publish_id: publishId }),
  });
  const d = await r.json();
  return { ...(d.data || {}), error: d.error?.code !== "ok" ? d.error : undefined };
}

/** Which TikTok account the token posts to. */
export async function whoami(env, slug) {
  const token = await accessToken(env, slug);
  // display_name needs user.info.basic (username would need user.info.profile).
  const r = await fetch(`${TT}/user/info/?fields=open_id,display_name`, { headers: { Authorization: `Bearer ${token}` } });
  return r.json();
}

/** TikTok's URL-prefix verification file, served at /persona/media/<name>.txt. */
export async function verifyFile(env, name) {
  const body = await env.PERSONA.get(`ttverify:${name}`);
  return body ? new Response(body, { headers: { "content-type": "text/plain" } }) : new Response("not found", { status: 404 });
}
