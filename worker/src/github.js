/**
 * GitHub REST calls the Worker needs. Kept separate from Telegram handling because
 * these are the operations that WRITE -- dispatching workflows and editing
 * posted.json -- and they deserve to be read on their own.
 */

const API = "https://api.github.com";

function headers(env) {
  return {
    Authorization: `Bearer ${env.GITHUB_TOKEN}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    // GitHub rejects API requests without a User-Agent.
    "User-Agent": "mpt-telegram-worker",
  };
}

export async function dispatchWorkflow(env, inputs) {
  const url = `${API}/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}`
    + `/actions/workflows/${env.VIDEO_WORKFLOW}/dispatches`;
  const r = await fetch(url, {
    method: "POST",
    headers: { ...headers(env), "content-type": "application/json" },
    body: JSON.stringify({ ref: env.GITHUB_BRANCH, inputs }),
  });
  if (!r.ok) throw new Error(`workflow dispatch failed: ${r.status} ${await r.text()}`);
}

async function getFile(env, path) {
  const url = `${API}/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/contents/${path}`
    + `?ref=${env.GITHUB_BRANCH}`;
  const r = await fetch(url, { headers: headers(env) });
  if (!r.ok) throw new Error(`GET ${path}: ${r.status}`);
  return r.json();
}

async function putFile(env, path, contentB64, sha, message) {
  const url = `${API}/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/contents/${path}`;
  return fetch(url, {
    method: "PUT",
    headers: { ...headers(env), "content-type": "application/json" },
    body: JSON.stringify({ message, content: contentB64, branch: env.GITHUB_BRANCH, sha }),
  });
}

// The Contents API is byte-oriented; btoa/atob alone mangle anything outside Latin-1,
// and posted.json is full of captions that are not.
function utf8ToB64(str) {
  const bytes = new TextEncoder().encode(str);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}
function b64ToUtf8(b64) {
  const bin = atob(b64.replace(/\n/g, ""));
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export { utf8ToB64, b64ToUtf8, getFile, putFile };

/**
 * Read-modify-write posted.json with retry on 409.
 *
 * It genuinely does move under us: two autopilot workflows and this Worker all write
 * it. `mutate` edits the parsed state in place and returns false to mean "nothing to
 * do", in which case no PUT is made at all.
 */
export async function mutatePostedJson(env, mutate, message) {
  const MAX = 5;
  for (let attempt = 0; attempt < MAX; attempt++) {
    const file = await getFile(env, "posted.json");
    const state = JSON.parse(b64ToUtf8(file.content));
    state.uploads = state.uploads || [];
    if (mutate(state) === false) return null;
    const r = await putFile(env, "posted.json",
      utf8ToB64(JSON.stringify(state, null, 2)), file.sha, message);
    if (r.ok) return state;
    if (r.status !== 409 || attempt === MAX - 1) {
      throw new Error(`posted.json write failed: ${r.status} ${await r.text()}`);
    }
    await new Promise((res) => setTimeout(res, 400 * (attempt + 1)));
  }
}

/** Commit a binary file to gh-pages and return its public Pages URL. */
export async function hostOnPages(env, path, contentB64) {
  const url = `${API}/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/contents/${path}`;
  const r = await fetch(url, {
    method: "PUT",
    headers: { ...headers(env), "content-type": "application/json" },
    body: JSON.stringify({
      message: `media: ${path.split("/").pop()} (via telegram)`,
      content: contentB64,
      branch: "gh-pages",
    }),
  });
  if (!r.ok) throw new Error(`gh-pages upload failed: ${r.status} ${await r.text()}`);
  return `${env.PAGES_BASE_URL.replace(/\/$/, "")}/${path}`;
}


/**
 * Record a chat id the Worker does not recognise, into a small file in the repo.
 *
 * Setting up a new channel is otherwise a coordination problem: with a webhook active
 * getUpdates returns nothing, the Worker acks each update so Telegram drops it, and
 * `wrangler tail` only shows what happens while someone is watching. Writing the id
 * down makes it retroactive -- post whenever, read it whenever.
 *
 * Deliberately a separate file rather than a key in posted.json: this is transient
 * setup scaffolding, and posted.json is written by three other things already.
 */
// Returns true when this chat id had NOT been recorded before, so the caller can
// announce it once instead of replying under every message in a channel that is never
// going to be added -- which is what happened in a sibling project's channel this bot
// is a member of.
export async function recordUnknownChat(env, chatId, title) {
  const path = "telegram_chats.json";
  let sha, seen = {};
  try {
    const f = await getFile(env, path);
    sha = f.sha;
    seen = JSON.parse(b64ToUtf8(f.content));
  } catch {
    // 404 on first use is expected: PUT with no sha creates the file.
  }
  if (seen[chatId]) return false;
  seen[chatId] = { title: title || null, seen_at: new Date().toISOString() };
  const body = {
    message: `telegram: record chat id ${chatId}`,
    content: utf8ToB64(JSON.stringify(seen, null, 2)),
    branch: env.GITHUB_BRANCH,
  };
  if (sha) body.sha = sha;
  const r = await fetch(
    `${API}/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/contents/${path}`, {
      method: "PUT",
      headers: { ...headers(env), "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  // A failed write means the id was not persisted, so the next message would look new
  // again. Report not-new so the announcement does not repeat on every post.
  return r.ok;
}
