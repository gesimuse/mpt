/**
 * Fanvue API integration -- posting Telegram-approved images/videos to Fanvue.
 *
 * UNVERIFIED AGAINST A LIVE ACCOUNT. Fanvue's public creator API is currently
 * waitlisted: help.fanvue.com/en/articles/11363222-creator-settings-request-api-keys
 * says "Fanvue does not offer API keys for general use... Creators can request an API
 * key to join a waitlist" (checked 2026-09-08, no ETA given). There is no
 * FANVUE_API_TOKEN to test this against yet for most accounts.
 *
 * The flow below follows Fanvue's own documented image-post upload flow as closely as
 * the public docs describe it (api.fanvue.com/docs/mcp-server/custom-tools --
 * "reserve an upload slot, PUT raw bytes to a short-lived URL with no Authorization
 * header, keep the ETag, create the post referencing mediaUuid+uploadId+etag"):
 *   1. POST START_UPLOAD_PATH            -> { mediaUuid, uploadId, uploadUrl }
 *   2. PUT <uploadUrl>, raw bytes         -> keep the ETag response header
 *   3. POST /posts { image: {mediaUuid, uploadId, etag}, text, audience }
 *
 * START_UPLOAD_PATH is the one line most likely wrong: the docs describe this flow
 * precisely through their MCP tool wrapper, but never name the underlying REST path
 * outside it. Fanvue's own v1 API reference (api.fanvue.com/docs/_llms/api-reference/
 * v1.md) separately documents a DIFFERENT-shaped upload flow at "/creators/
 * {creatorUserUuid}/media/multipart-upload" (S3-style multipart, for an agency
 * posting media on a creator's behalf) and a create-post body shaped as
 * {mediaUuids: [...]} (plural, referencing already-uploaded media) rather than the
 * single nested {image: {...}} object used below -- the two docs describe the same
 * capability inconsistently, and without a live token there is no way to tell which
 * is current. Whichever field name is wrong will surface immediately as a 400 on the
 * first live call; fix it in createPost() below.
 */

const API = "https://api.fanvue.com";
const API_VERSION = "2025-06-26"; // X-Fanvue-API-Version -- required per api.fanvue.com/docs
// Best-guess REST path for the MCP tool's "start-image-upload" -- see the module
// docstring above. Verify against a live account before relying on this.
const START_UPLOAD_PATH = "/media/images";

function headers(env, extra) {
  return {
    Authorization: `Bearer ${env.FANVUE_API_TOKEN}`,
    "X-Fanvue-API-Version": API_VERSION,
    ...extra,
  };
}

async function startUpload(env) {
  const r = await fetch(`${API}${START_UPLOAD_PATH}`, {
    method: "POST",
    headers: headers(env, { "content-type": "application/json" }),
  });
  if (!r.ok) throw new Error(`fanvue start-upload failed: ${r.status} ${await r.text()}`);
  const body = await r.json();
  if (!body.mediaUuid || !body.uploadId || !body.uploadUrl) {
    throw new Error(`fanvue start-upload: unexpected response shape ${JSON.stringify(body).slice(0, 200)}`);
  }
  return body;
}

async function putBytes(uploadUrl, bytes, contentType) {
  // No Authorization header on this PUT -- uploadUrl is itself a short-lived signed
  // URL per Fanvue's docs; sending the Fanvue API bearer token to whatever storage
  // host it actually points at would be a real credential leak to a third party.
  const r = await fetch(uploadUrl, {
    method: "PUT",
    headers: { "content-type": contentType },
    body: bytes,
  });
  if (!r.ok) throw new Error(`fanvue upload PUT failed: ${r.status} ${await r.text()}`);
  const etag = r.headers.get("etag");
  if (!etag) throw new Error("fanvue upload PUT returned no ETag header");
  return etag;
}

async function createPost(env, media, caption) {
  const body = {
    text: (caption || "").slice(0, 5000),
    audience: env.FANVUE_AUDIENCE || "subscribers",
    image: media, // {mediaUuid, uploadId, etag} -- see module docstring re: field-name risk
  };
  const r = await fetch(`${API}/posts`, {
    method: "POST",
    headers: headers(env, { "content-type": "application/json" }),
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`fanvue create-post failed: ${r.status} ${await r.text()}`);
  return r.json();
}

/** bytes: an ArrayBuffer/Uint8Array of the raw image file. Returns the created post. */
export async function postImage(env, bytes, caption, contentType = "image/jpeg") {
  if (!env.FANVUE_API_TOKEN) {
    throw new Error("FANVUE_API_TOKEN not configured (Fanvue's creator API is "
      + "waitlisted -- see this file's module docstring)");
  }
  const { mediaUuid, uploadId, uploadUrl } = await startUpload(env);
  const etag = await putBytes(uploadUrl, bytes, contentType);
  return createPost(env, { mediaUuid, uploadId, etag }, caption);
}

/**
 * Same flow as postImage -- Fanvue's docs don't separately describe a video-post
 * upload path, so this assumes it mirrors the image one. Unconfirmed; the likeliest
 * thing to need a real fix once a video is actually tried live.
 */
export async function postVideo(env, bytes, caption, contentType = "video/mp4") {
  return postImage(env, bytes, caption, contentType);
}
