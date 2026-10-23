/**
 * Replies on her Instagram: every comment on her posts and every DM gets a short
 * answer in her voice (mostly emojis), through Instagram's own API ("Instagram API
 * with Instagram Login": instagram_business_manage_comments / _manage_messages).
 * Asked for 2026-10-23: "auto reply ... no confirmation needed, reply with emojis".
 *
 *   GET/POST /persona/ig/webhook   Meta's webhook: the verify handshake, then
 *                                  comments and messages events
 *   POST /persona/admin/igtoken?slug=   her long-lived token; looks up her
 *                                  Instagram user id and subscribes to the events
 *
 * Secrets: IG_APP_SECRET (signs Meta's webhook calls), IG_VERIFY_TOKEN (the
 * handshake). Her token: KV igtoken:<slug> = {token, user_id, username, refreshed},
 * refreshed weekly (it lasts 60 days). Nothing is replied twice (KV igdone:<id>).
 */
import * as personas from "./personas.js";

const GRAPH = "https://graph.instagram.com";
const MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct";
const EMOJI = ["😘", "🥰", "💕", "😍", "🙈", "✨", "🔥", "😉", "💋", "🤍"];
const pick = (a) => a[Math.floor(Math.random() * a.length)];

async function graph(path, token, { method = "GET", body = null, params = {} } = {}) {
  const u = new URL(`${GRAPH}${path}`);
  for (const [k, v] of Object.entries({ ...params, access_token: token })) u.searchParams.set(k, v);
  const r = await fetch(u, { method, ...(body ? { headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {}) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(`Instagram ${path}: ${d.error?.message || r.status}`.slice(0, 300));
  return d;
}

/** Her token and ids: POST the long-lived token (text body). */
export async function setToken(env, slug, token) {
  const me = await graph("/me", token, { params: { fields: "user_id,username" } });
  await graph("/me/subscribed_apps", token, { method: "POST", params: { subscribed_fields: "comments,messages" } });
  const rec = { token, user_id: String(me.user_id), username: me.username, refreshed: Date.now() };
  await env.PERSONA.put(`igtoken:${slug}`, JSON.stringify(rec));
  return { user_id: rec.user_id, username: rec.username };
}

/** Weekly: a long-lived token refreshed before its 60 days run out. */
export async function refreshTokens(env) {
  for (const p of await personas.list(env)) {
    const rec = JSON.parse((await env.PERSONA.get(`igtoken:${p.slug}`)) || "null");
    if (!rec || Date.now() - rec.refreshed < 7 * 86400 * 1000) continue;
    try {
      const d = await graph("/refresh_access_token", rec.token, { params: { grant_type: "ig_refresh_token" } });
      await env.PERSONA.put(`igtoken:${p.slug}`, JSON.stringify({ ...rec, token: d.access_token, refreshed: Date.now() }));
    } catch (e) {
      console.error(`ig token refresh ${p.slug}`, String(e).slice(0, 200));
    }
  }
}

async function verifySignature(env, request, raw) {
  const sig = (request.headers.get("x-hub-signature-256") || "").replace(/^sha256=/, "");
  if (!env.IG_APP_SECRET || !sig) return false;
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(env.IG_APP_SECRET),
                                            { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(raw)));
  const hex = [...mac].map((b) => b.toString(16).padStart(2, "0")).join("");
  return hex.length === sig.length && hex === sig.toLowerCase();
}

/** Who she is on Instagram: the persona whose token has this Instagram user id. */
async function personaFor(env, igId) {
  for (const p of await personas.list(env)) {
    const rec = JSON.parse((await env.PERSONA.get(`igtoken:${p.slug}`)) || "null");
    if (rec && rec.user_id === String(igId)) return { p, rec };
  }
  return null;
}

/**
 * Her answer: one short line, mostly emojis. Instagram-safe: no links, no meeting up,
 * no money, nothing explicit. She never brings up being AI (her posts carry
 * Instagram's AI label) and a "are you real?" gets a coy emoji -- never a claim to
 * be human (2026-10-23).
 */
async function answer(env, bible, kind, text) {
  const fallback = pick(EMOJI) + (Math.random() < 0.5 ? pick(EMOJI) : "");
  if (!String(text || "").trim()) return fallback;
  try {
    const r = await env.AI.run(MODEL, { max_tokens: 60, temperature: 0.9, messages: [{ role: "user", content:
      `You are ${bible?.name || "Lena"}, a flirty, playful fictional Instagram model (an AI character) from ` +
      `${bible?.home_city || "Prague"}. Someone wrote this ${kind === "dm" ? "direct message to you" : "comment on your post"}:\n` +
      `"${String(text).slice(0, 400)}"\n` +
      `Reply as her: at most 8 words, mostly emojis (often just 1-3 emojis), warm and playful, in the language they wrote in. ` +
      `Never explicit, never a link, never agree to meet, never talk about money or prices. Never bring up AI yourself. ` +
      `If they ask whether you are real, a bot or AI, don't answer it: reply only with a coy emoji like "😏" or "🙈" -- ` +
      `never claim to be human. If the message is rude or hateful, reply only "🙏". ` +
      `Return ONLY the reply text.` }] });
    const out = String(typeof r.response === "string" ? r.response : "").trim().replace(/^["']|["']$/g, "");
    return out && out.length <= 120 && !/https?:|www\./i.test(out) ? out : fallback;
  } catch {
    return fallback;
  }
}

async function once(env, key) {
  if (await env.PERSONA.get(`igdone:${key}`)) return false;
  await env.PERSONA.put(`igdone:${key}`, "1", { expirationTtl: 30 * 86400 });
  return true;
}

async function onEvent(env, body) {
  for (const entry of body.entry || []) {
    const who = await personaFor(env, entry.id);
    if (!who) continue;
    const { p, rec } = who;
    const bible = JSON.parse((await env.PERSONA.get(`bible:${p.slug}`)) || "null");
    for (const ch of entry.changes || []) {
      // A comment on her post (not her own reply).
      const v = ch.value || {};
      if (ch.field !== "comments" || !v.id || String(v.from?.id) === rec.user_id || v.parent_id) continue;
      if (!(await once(env, `c:${v.id}`))) continue;
      const msg = await answer(env, bible, "comment", v.text);
      await graph(`/${v.id}/replies`, rec.token, { method: "POST", params: { message: msg } })
        .catch((e) => console.error("ig comment reply", String(e).slice(0, 200)));
    }
    for (const m of entry.messaging || []) {
      // A DM to her (not her own echo, not a reaction or read receipt).
      if (!m.message || m.message.is_echo || String(m.sender?.id) === rec.user_id) continue;
      if (!(await once(env, `m:${m.message.mid}`))) continue;
      const msg = await answer(env, bible, "dm", m.message.text || "");
      await graph("/me/messages", rec.token, { method: "POST", body: { recipient: { id: m.sender.id }, message: { text: msg } } })
        .catch((e) => console.error("ig dm reply", String(e).slice(0, 200)));
    }
  }
}

export async function onWebhook(request, env, ctx) {
  const u = new URL(request.url);
  if (request.method === "GET") {
    // Meta's subscription handshake.
    const ok = u.searchParams.get("hub.mode") === "subscribe" && env.IG_VERIFY_TOKEN &&
               u.searchParams.get("hub.verify_token") === env.IG_VERIFY_TOKEN;
    return ok ? new Response(u.searchParams.get("hub.challenge") || "") : new Response("forbidden", { status: 403 });
  }
  const raw = await request.text();
  if (!(await verifySignature(env, request, raw))) return new Response("bad signature", { status: 403 });
  // Answered at once (Meta retries slow webhooks); the replies are written after.
  const work = onEvent(env, JSON.parse(raw)).catch((e) => console.error("ig webhook", String(e).slice(0, 300)));
  if (ctx?.waitUntil) ctx.waitUntil(work);
  else await work;
  return new Response("ok");
}
