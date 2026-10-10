/**
 * The caption and hashtags of every 👍-published post, per platform.
 *
 * Every post gets a real caption: the scene's own (written with the photo), else
 * Workers AI writes one from what is in the post, else a template from the scene
 * -- never empty, never a lone emoji.
 *
 * Hashtags (5 per platform, #aigenerated always one of them) come from three
 * pools -- her core tags (persona entry core_tags, else written once from her
 * bible and cached), tags for this post's scene, and a hand-kept trending list
 * (KV trend_tags:<slug>; TikTok's trending API needs a logged-in browser, see
 * trends.py) -- and are ranked by what has worked for HER: learn() scores each
 * tag from Buffer's numbers for the posts it was on, 48h after posting, against
 * her usual post on that platform. Each post keeps one slot for a tag not tried
 * yet, so new tags keep getting tested.
 */

// llama-3.1-8b-instruct was retired 2026-05-30 (every call failed: "5028 ... deprecated"),
// so scenes and captions had been falling back to templates. Mistral Small 3.1 wrote
// the most specific captions and tags in a 2026-10-10 comparison.
const MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct";
const text = (r) => (typeof r.response === "string" ? r.response : JSON.stringify(r.response || ""));
const PER_POST = 5;
const ALWAYS = "aigenerated";

const clean = (t) => String(t).toLowerCase().replace(/^#/, "").replace(/[^a-z0-9_]/g, "");
const pick = (a) => a[Math.floor(Math.random() * a.length)];

async function ask(env, prompt, maxTokens = 400) {
  const r = await env.AI.run(MODEL, { messages: [{ role: "user", content: prompt }], max_tokens: maxTokens, temperature: 0.8 });
  const m = /\{[\s\S]*\}/.exec(text(r));
  if (!m) throw new Error("no JSON");
  return JSON.parse(m[0].replace(/,\s*([\]}])/g, "$1"));
}

function describe(item) {
  const s = item.scene || {};
  if (s.setting) return `a ${item.kind === "video" ? "short video" : "photo"} of her: ${s.setting}, wearing ${s.outfit || "an outfit"}, ${s.action || ""}, mood ${s.mood || ""}`;
  if (item.recreate) return "a short video of her doing a popular TikTok trend";
  return `a ${item.kind === "video" ? "short video" : "photo"} of her`;
}

function okCaption(t) {
  const s = String(t || "").trim();
  return s.length >= 8 && s.length <= 220 && !s.includes("#") && /[a-z]/i.test(s) ? s : "";
}

function templateCaption(item) {
  const place = item.scene?.setting;
  if (place) return pick([`${place} kind of day ✨`, `Caught the light at ${place} 🌅`, `Soft mood, ${place} 💫`,
                          `Me, ${place}, nothing else 🤍`, `Saving this ${place} moment 📸`]);
  if (item.recreate) return pick(["Couldn't resist this trend 😏", "Tried this one, rate it 1-10 👀", "Doing it my way ✨"]);
  return pick(["Just me being me ✨", "Feeling this one 💫", "A little moment for you 🤍"]);
}

/** {caption, tiktok, tags}: the post's caption, a TikTok version with search words, scene tags. */
async function write(env, b, item) {
  const prompt = `You write social posts for ${b.name}, a ${b.age}-year-old fictional social-media persona from ${b.home_city}.
Personality: ${b.personality}.
The post: ${describe(item)}.${item.caption ? `\nIts caption: "${item.caption}"` : ""}
Return ONLY JSON:
{"caption": ${item.caption ? "the caption above, unchanged" : "\"first-person caption in her voice, max 20 words, 1-2 emojis, no hashtags\""},
 "tiktok": "the same caption plus a few words people would search for on TikTok, naturally, max 30 words, no hashtags",
 "tags": ["6 specific lowercase hashtags for THIS post's place, outfit and mood, no # sign, no generic tags like love or instagood"]}`;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const d = await ask(env, prompt);
      const caption = okCaption(item.caption) || okCaption(d.caption);
      if (!caption) continue;
      return { caption, tiktok: okCaption(d.tiktok) || caption,
               tags: (Array.isArray(d.tags) ? d.tags : []).map(clean).filter((t) => t.length >= 3).slice(0, 8) };
    } catch (e) {
      console.log("caption writing failed", String(e).slice(0, 120));
    }
  }
  const caption = okCaption(item.caption) || templateCaption(item);
  return { caption, tiktok: caption, tags: (item.tags || []).map(clean) };
}

/** Her core tags: the persona entry's core_tags, else written once from her bible. */
async function coreTags(env, p, b) {
  if (p.core_tags?.length) return p.core_tags.map(clean);
  const cached = await env.PERSONA.get(`coretags:${p.slug}`);
  if (cached) return JSON.parse(cached);
  let tags = [];
  try {
    const d = await ask(env, `List 6 hashtags (lowercase, no # sign) that describe the niche of ${b.name}, a fictional ` +
      `glam Instagram/TikTok model persona: ${b.personality}; style: ${(b.style || []).join(", ")}; based in ${b.home_city}. ` +
      `Specific to her niche, not generic like love or instagood. Return ONLY JSON: {"tags": [...]}`);
    tags = (d.tags || []).map(clean).filter((t) => t.length >= 3).slice(0, 6);
  } catch (e) {
    console.log("core tags failed", String(e).slice(0, 120));
  }
  if (!tags.length) tags = ["model", "fashion", "ootd", "glam"];
  await env.PERSONA.put(`coretags:${p.slug}`, JSON.stringify(tags));
  return tags;
}

/** Rank by her results: mean log-lift with a small bonus for few trials; untried tags get a prior. */
function score(stats, tag, prior) {
  const s = stats[tag];
  if (!s || s.n < 2) return prior + 0.3;
  return s.sum / s.n + 0.3 / Math.sqrt(s.n);
}

function choose(stats, pools) {
  const prior = { scene: 0.15, trend: 0.2, core: 0.1 };
  const cands = new Map();
  for (const [kind, tags] of Object.entries(pools)) {
    for (const t of tags) if (t && t !== ALWAYS && !cands.has(t)) cands.set(t, kind);
  }
  const ranked = [...cands.entries()].map(([t, kind]) => ({ t, kind, s: score(stats, t, prior[kind]) }))
    .sort((a, b) => b.s - a.s);
  const out = [ALWAYS];
  // One tag for this post's own scene, so the tags always fit the post.
  const scene = ranked.find((r) => r.kind === "scene");
  if (scene) out.push(scene.t);
  // One slot for a tag not tried yet.
  const untried = ranked.filter((r) => !stats[r.t] && !out.includes(r.t));
  if (untried.length) out.push(pick(untried).t);
  for (const r of ranked) {
    if (out.length >= PER_POST) break;
    if (!out.includes(r.t)) out.push(r.t);
  }
  return out;
}

/**
 * {instagram: {text, tags}, tiktok: {text, tags}} for one post. `state.hashtags`
 * holds her per-platform tag scores.
 */
export async function forPost(env, p, b, state, item) {
  const w = await write(env, b, item);
  const core = await coreTags(env, p, b);
  const trend = JSON.parse((await env.PERSONA.get(`trend_tags:${p.slug}`)) || "[]").map(clean);
  const stats = state.hashtags || {};
  const out = {};
  for (const platform of ["instagram", "tiktok"]) {
    const tags = choose(stats[platform] || {}, { scene: w.tags, trend, core });
    const text = platform === "tiktok" ? w.tiktok : w.caption;
    out[platform] = { tags, text: `${text}\n\n${tags.map((t) => "#" + t).join(" ")}` };
  }
  return out;
}

// ------------------------------------------------------------------ learning
const WEIGHTS = { likes: 1, reactions: 1, comments: 3, shares: 4, reposts: 4, saves: 4, follows: 8 };

/** One number for how a post did: its reach plus weighted engagement. */
export function postValue(metrics) {
  const m = Object.fromEntries((metrics || []).map((x) => [x.type || x.name, x.value]));
  const reach = m.views || m.reach || m.impressions || 0;
  let eng = 0;
  for (const [k, w] of Object.entries(WEIGHTS)) eng += (m[k] || 0) * w;
  return reach + 10 * eng;
}

/**
 * Score the tags of a post that has had 48h: log of its value against her median
 * post on that platform, added to each tag it carried.
 */
export function learn(state, platform, tags, value) {
  state.hashtags = state.hashtags || {};
  state.values = state.values || {};
  const hist = (state.values[platform] = [...(state.values[platform] || []), value].slice(-30));
  const sorted = [...hist].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] || 1;
  const lift = Math.log((value + 1) / (median + 1));
  const stats = (state.hashtags[platform] = state.hashtags[platform] || {});
  for (const t of tags) {
    if (t === ALWAYS) continue;
    const s = (stats[t] = stats[t] || { n: 0, sum: 0 });
    s.n += 1;
    s.sum += lift;
  }
  // Keep the 300 most-used.
  const keys = Object.keys(stats);
  if (keys.length > 300) {
    for (const k of keys.sort((a, b) => stats[a].n - stats[b].n).slice(0, keys.length - 300)) delete stats[k];
  }
}
