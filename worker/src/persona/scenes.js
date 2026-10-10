/**
 * Scene writing on Workers AI, from the same idea bank as persona/scenes.py
 * (persona/ideas.json). Each scene is seeded with a place she has not been to
 * recently and an outfit that belongs there; the LLM writes around them, and the
 * template fallback uses them directly if the LLM answer is unusable.
 */
import IDEAS from "../../../persona/ideas.json";

const MODEL = "@cf/meta/llama-3.1-8b-instruct";
const pick = (a) => a[Math.floor(Math.random() * a.length)];

function outfitFor(place, used) {
  const low = place.toLowerCase();
  let pool = null;
  for (const [kind, words] of IDEAS.place_kinds) {
    if (words.some((w) => low.includes(w))) { pool = IDEAS.outfit_kinds[kind]; break; }
  }
  if (!pool) {
    const excluded = new Set([...IDEAS.outfit_kinds.swim, ...IDEAS.outfit_kinds.sport, ...(IDEAS.outfit_kinds.snow || [])]);
    pool = IDEAS.outfits.filter((o) => !excluded.has(o));
  }
  const free = pool.filter((o) => !used.has(o));
  return pick(free.length ? free : pool);
}

function seeds(state, n) {
  const recent = state.recent.slice(-40).join(" ").toLowerCase();
  const places = [], outfits = [], usedPlaces = new Set(), usedOutfits = new Set();
  for (let i = 0; i < n; i++) {
    const free = IDEAS.places.filter((p) => !recent.includes(p.toLowerCase()) && !usedPlaces.has(p));
    const place = pick(free.length ? free : IDEAS.places);
    usedPlaces.add(place);
    const outfit = outfitFor(place, usedOutfits);
    usedOutfits.add(outfit);
    places.push(place); outfits.push(outfit);
  }
  return { places, outfits };
}

function leaning(state) {
  const ranked = Object.entries(state.votes.tags).sort((a, b) => b[1] - a[1]);
  return { liked: ranked.filter(([, v]) => v > 0).slice(0, 6).map(([t]) => t),
           disliked: ranked.filter(([, v]) => v < 0).slice(-6).map(([t]) => t) };
}

function prompt(b, state, places, outfits) {
  const { liked, disliked } = leaning(state);
  const story = state.story.slice(-8).map((s) => `- ${s.beat}`).join("\n") || "- (nothing yet)";
  const recent = state.recent.slice(-30).map((r) => `- ${r}`).join("\n") || "- (none yet)";
  const pairs = places.map((p, i) => `${p} wearing a ${outfits[i]}`).join("; ");
  const n = places.length;
  return `You plan photos for ${b.name}, a fictional social-media persona.
Who she is: ${b.age}-year-old woman, ${b.identity || ""}.
Personality: ${b.personality}. Lives in ${b.home_city}.
Her story so far (oldest first):
${story}
People liked: ${liked.join(", ") || "no data yet"}.
People did not like: ${disliked.join(", ") || "no data yet"}.

She is a glamorous, sexy Instagram model and every photo is a flattering shoot of HER, not a
travel photo: she is the subject, filling most of the frame. Outfits are figure-flattering and
revealing, but there is NO nudity. Poses are confident and model-like.

Write ${n} NEW scenes, one per place and outfit, in this order: ${pairs}.
Do not describe her face or hair; that is fixed.
Recent scenes -- do NOT repeat these:
${recent}

Return ONLY a JSON array of ${n} objects with these keys:
"setting" (the place, specific), "outfit" (the given outfit, with colours), "action" (what she is
doing, her hands doing ONE simple thing: on her hip, in her hair, holding one drink; never several
objects; her body facing roughly toward the camera, no twisting; she is POSING for the photo at the place, never
caught mid-action: no throwing, swinging, running or jumping), "shot" (camera framing), "light",
"mood", "motion" (one sentence: how she moves slowly in a 5-second video), "caption" (first-person
social caption, max 20 words, 1-2 emojis, no hashtags), "beat" (one past-tense sentence for her
storyline), "tags" (3-5 short lowercase tags).`;
}

function parse(text, n) {
  const m = /\[[\s\S]*\]/.exec(text || "");
  if (!m) throw new Error("no JSON array");
  const arr = JSON.parse(m[0].replace(/,\s*([\]}])/g, "$1"));
  const keys = ["setting", "outfit", "action", "shot", "light", "mood", "motion", "caption", "beat", "tags"];
  const good = arr.filter((s) => s && s.setting && s.outfit && s.action).map((s) => {
    for (const k of keys) if (s[k] === undefined) s[k] = k === "tags" ? [] : "";
    s.tags = (Array.isArray(s.tags) ? s.tags : []).map((t) => String(t).toLowerCase().trim()).filter(Boolean).slice(0, 6);
    return s;
  });
  if (!good.length) throw new Error("no usable scenes");
  return good.slice(0, n);
}

function fallback(places, outfits) {
  return places.map((place, i) => ({
    setting: place, outfit: outfits[i], action: pick(IDEAS.simple_actions), shot: pick(IDEAS.shots),
    light: pick(IDEAS.times), mood: pick(IDEAS.moods),
    motion: "she shifts her weight slowly and smiles softly", caption: "", beat: `Spent time at ${place}.`,
    tags: [place.split(" ").pop(), outfits[i].split(" ").pop()],
  }));
}

function assignGazes(list) {
  const g = IDEAS.gazes, start = Math.floor(Math.random() * g.length);
  list.forEach((s, i) => { [s.gaze, s.sheet] = g[(start + i * 3) % g.length]; });
  return list;
}

/** n scenes, in chunks of 4 (an 8B model breaks long JSON), each chunk retried once. */
export async function write(env, b, state, n) {
  const out = [];
  while (out.length < n) {
    const k = Math.min(4, n - out.length);
    const { places, outfits } = seeds({ ...state, recent: [...state.recent, ...out.map((s) => s.setting)] }, k);
    let got = null;
    for (let attempt = 0; attempt < 2 && !got; attempt++) {
      try {
        const r = await env.AI.run(MODEL, { messages: [{ role: "user", content: prompt(b, state, places, outfits) }],
                                            max_tokens: 2000, temperature: 0.9 });
        got = parse(r.response, k);
      } catch (e) {
        console.log("scene writing failed", String(e).slice(0, 150));
      }
    }
    out.push(...(got || fallback(places, outfits)));
  }
  return assignGazes(out.slice(0, n));
}

/**
 * A post caption when the item has none (🎭 recreates): first person, short, in
 * her voice. Falls back to a plain one if Workers AI fails.
 */
export async function caption(env, b) {
  try {
    const r = await env.AI.run(MODEL, { messages: [{ role: "user", content:
      `Write ONE Instagram/TikTok caption for a short video of ${b.name}, a ${b.age}-year-old social-media persona ` +
      `(${b.personality}). First person, max 15 words, 1-2 emojis, no hashtags, no quotes. Answer with the caption only.` }],
      max_tokens: 60, temperature: 0.9 });
    const text = String(r.response || "").trim().replace(/^["']|["']$/g, "").split("\n")[0];
    if (text) return text;
  } catch (e) {
    console.log("caption writing failed", String(e).slice(0, 150));
  }
  return "✨";
}
