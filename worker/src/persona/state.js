/**
 * Her changing state in Workers KV, one JSON document per persona. Her identity
 * (bible, refs, face sheet) stays in the private Kaggle dataset the GPU jobs read.
 *
 * Moved here from the Kaggle dataset, where two overlapping runs overwrote each
 * other twice on 2026-10-07 (a lost 🎬 tap, a lost "video running" record).
 */
export const EMPTY = {
  votes: { tags: {}, models: {} },
  story: [],          // [{t, beat, item}]
  recent: [],         // ["setting / outfit / action"], newest last
  posted: { date: "", ids: [] },
  render_date: "",
  voted: [],
  queue: [],          // on-demand jobs waiting: 🎬 / reply / 🔁
  running: [],        // the old single job kernel's last run
  lanes: {},          // quick / recreate kernel, or laptop -> jobs in its current run
  publish: [],        // 👍 -> Instagram / TikTok jobs (social.js), newest last
  hashtags: {},       // platform -> tag -> {n, sum of log-lift} (captions.js)
  values: {},         // platform -> her last 30 posts' values, for the median
  done: [],
  items: {},          // id -> {kind, tags, beat, scene, caption, motion}
};

export async function load(env, slug) {
  const raw = await env.PERSONA.get(`state:${slug}`);
  return { ...structuredClone(EMPTY), ...(raw ? JSON.parse(raw) : {}) };
}

export async function save(env, slug, s) {
  s.story = s.story.slice(-200);
  s.recent = s.recent.slice(-200);
  s.voted = s.voted.slice(-500);
  s.done = s.done.slice(-300);
  // Oldest first, but never a photo of a kept shoot (its TikTok draft and redos need
  // it). An all-digits id ("61213512") is listed before every other key, whatever its
  // age: such photos were dropped a day after their shoot (2026-10-26).
  const kept = new Set(Object.values(s.shoots || {}).flatMap((sh) => [...(sh.ids || []), ...(sh.media || []).map((m) => m.id)]));
  const ids = Object.keys(s.items).filter((id) => !kept.has(id));
  for (const id of ids.slice(0, Math.max(0, Object.keys(s.items).length - 400))) delete s.items[id];
  await env.PERSONA.put(`state:${slug}`, JSON.stringify(s));
}

export async function bible(env, slug) {
  const raw = await env.PERSONA.get(`bible:${slug}`);
  return raw ? JSON.parse(raw) : null;
}

export function today() {
  return new Date().toISOString().slice(0, 10);
}
