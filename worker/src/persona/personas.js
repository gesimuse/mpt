/**
 * The personas this Worker runs, one Telegram channel each: KV "personas" =
 * [{slug, chat_id, tiktok_direct?}]. The persona from wrangler.toml
 * (PERSONA_SLUG / PERSONA_CHAT_ID) is always included unless KV lists it too.
 * Register a channel with `python -m persona.cli channel <slug> <chat_id>`.
 */
export async function list(env) {
  const raw = await env.PERSONA.get("personas");
  const out = raw ? JSON.parse(raw) : [];
  const slug = env.PERSONA_SLUG || "lena";
  if (!out.some((p) => p.slug === slug)) out.unshift({ slug, chat_id: env.PERSONA_CHAT_ID });
  return out.map((p) => ({ ...p, chat_id: String(p.chat_id) }));
}

export async function get(env, slug) {
  return (await list(env)).find((p) => p.slug === slug) || null;
}

export async function save(env, entry) {
  const raw = await env.PERSONA.get("personas");
  const all = (raw ? JSON.parse(raw) : []).filter((p) => p.slug !== entry.slug);
  all.push(entry);
  await env.PERSONA.put("personas", JSON.stringify(all));
  return all;
}
