/**
 * Scene writing on Workers AI, from the same idea bank as persona/scenes.py
 * (persona/ideas.json). Each scene is seeded with a place she has not been to
 * recently and an outfit that belongs there; the LLM writes around them, and the
 * template fallback uses them directly if the LLM answer is unusable.
 */
import IDEAS from "../../../persona/ideas.json";

// llama-3.1-8b-instruct was retired 2026-05-30 (every call failed: "5028 ... deprecated"),
// so scenes and captions had been falling back to templates. Mistral Small 3.1 wrote
// the most specific captions and tags in a 2026-10-10 comparison.
const MODEL = "@cf/mistralai/mistral-small-3.1-24b-instruct";
const text = (r) => (typeof r.response === "string" ? r.response : JSON.stringify(r.response || ""));
const pick = (a) => a[Math.floor(Math.random() * a.length)];

// Lena's Instagram lane (2026-10-20, "not as sexy as she used to be"): the sexy list
// without what Instagram demotes (lingerie, see-through).
const IG_SAFE = (o) => !/lingerie|bralette|sheer|mesh|see-through|bodysuit/i.test(o);

function outfitFor(place, used, sexy = false, state = null) {
  const low = place.toLowerCase();
  let pool = null;
  for (const [kind, words] of IDEAS.place_kinds) {
    if (words.some((w) => low.includes(w))) { pool = IDEAS.outfit_kinds[kind]; break; }
  }
  // Sexy-style personas (MPT): the sexier list everywhere except swim spots, which
  // keep their swimwear and only they get bikinis (2026-10-17: "more sexy" -- a vineyard got a hoodie, a golf
  // course a gym set).
  if (sexy && IDEAS.sexy_outfits && pool !== IDEAS.outfit_kinds.swim) {
    pool = IDEAS.sexy_outfits.filter((o) => !/bikini|swim/i.test(o) && (sexy !== "flirty" || IG_SAFE(o)));
  }
  if (!pool) {
    const excluded = new Set([...IDEAS.outfit_kinds.swim, ...IDEAS.outfit_kinds.sport, ...(IDEAS.outfit_kinds.snow || [])]);
    pool = IDEAS.outfits.filter((o) => !excluded.has(o));
  }
  const free = pool.filter((o) => !used.has(o));
  return state ? pickPref(state, free.length ? free : pool, (o) => `outfit:${o}`) : pick(free.length ? free : pool);
}

/**
 * What you liked and redid (state.prefs, learnFrom in index.js): "place:…", "outfit:…",
 * "view:…", "angle:…", "pose:…" -> a score; each redo -1, each photo sent on +0.5.
 * Picks are weighted by it, so what you redo comes less and what you send comes more
 * -- never banned, so nothing disappears for good on a few votes.
 */
const prefW = (state, key) => {
  const v = (state.prefs || {})[key] || 0;
  // Places and outfits are only ever pushed up (variety stays); risk keys both ways.
  const lo = /^(place|outfit):/.test(key) ? 0 : -3;
  return Math.exp(Math.max(lo, Math.min(3, v)) * 0.5);
};
function pickPref(state, list, key) {
  const w = list.map((x) => prefW(state, key(x)));
  let r = Math.random() * w.reduce((a, b) => a + b, 0);
  for (let i = 0; i < list.length; i++) if ((r -= w[i]) < 0) return list[i];
  return list[list.length - 1];
}
export const poseKey = (action) => String(action || "").toLowerCase().replace(/[^a-z ]/g, "").split(" ").filter((w) => w.length > 3).slice(0, 3).join(" ");

/**
 * The real season in Prague (northern hemisphere), and the places that would look
 * wrong in it: 2026-10-23 her October feed had summer bikinis, a sunflower field and a
 * Christmas market next to each other, which read as random reposts, not one person.
 */
export function season(now = new Date()) {
  const m = now.getUTCMonth() + 1, d = now.getUTCDate();
  const name = m === 12 || m <= 2 ? "winter" : m <= 5 ? "spring" : m <= 8 ? "summer" : "autumn";
  return { name, month: now.toLocaleString("en-GB", { month: "long", timeZone: "Europe/Prague" }),
           christmas: (m === 11 && d >= 20) || m === 12 };
}
const OUT_OF_SEASON = {
  winter: /sunflower|lavender|cherry blossom|music festival|beach bonfire|rooftop picnic|autumn forest|farmers market/i,
  spring: /christmas|ski|snow|chalet|autumn forest|sunflower/i,
  summer: /christmas|ski|snow|chalet|fireplace|autumn forest|cozy cabin/i,
  autumn: /sunflower|lavender|cherry blossom|music festival|ski|snow|rooftop picnic/i,
};
const inSeason = (place, sn = season()) => !OUT_OF_SEASON[sn.name].test(place) && (sn.christmas || !/christmas/i.test(place));
const isSwim = (place) => IDEAS.place_kinds.find(([k]) => k === "swim")[1].some((w) => place.toLowerCase().includes(w));

function seeds(state, n, taken = [], sexy = false) {
  const recent = state.recent.slice(-40).join(" ").toLowerCase();
  // `taken`: scenes already written in this batch -- their places and outfits are
  // not repeated (a shoot had the same pool and bikini twice, 2026-10-17).
  const places = [], outfits = [], usedPlaces = new Set(taken.map((t) => t.place)),
        usedOutfits = new Set(taken.map((t) => t.outfit));
  for (let i = 0; i < n; i++) {
    // In season only; and outside summer, Lena's Instagram (flirty) gets at most one
    // pool/beach scene per batch -- a warm trip now and then, not every other photo.
    const swimDone = sexy === "flirty" && season().name !== "summer" && [...usedPlaces].some(isSwim);
    const ok = IDEAS.places.filter((p) => inSeason(p) && !(swimDone && isSwim(p)));
    const free = ok.filter((p) => !recent.includes(p.toLowerCase()) && !usedPlaces.has(p));
    const place = pickPref(state, free.length ? free : ok, (x) => `place:${x}`);
    usedPlaces.add(place);
    const outfit = outfitFor(place, usedOutfits, sexy, state);
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

function prompt(b, state, places, outfits, sexy = false) {
  const { liked, disliked } = leaning(state);
  const story = state.story.slice(-8).map((s) => `- ${s.beat}`).join("\n") || "- (nothing yet)";
  const recent = state.recent.slice(-30).map((r) => `- ${r}`).join("\n") || "- (none yet)";
  const pairs = places.map((p, i) => `${p} wearing a ${outfits[i]}`).join("; ");
  const n = places.length;
  const sn = season();
  return `You plan photos for ${b.name}, a fictional social-media persona.
Who she is: ${b.age}-year-old woman, ${b.identity || ""}.
Personality: ${b.personality}. Lives in ${b.home_city}.
It is ${sn.month} now (${sn.name} in ${b.home_city}): outfits, light and setting fit this time of year.
${sexy === "flirty" ? `She is ONE real person with ONE life: a place that is not obviously abroad is in ${b.home_city}
(say so in "setting", e.g. "a cozy bookstore loft in ${b.home_city}'s Old Town"); a beach or a city abroad is a
short trip she is on right now, and its caption says so.
` : ""}Her story so far (oldest first):
${story}
People liked: ${liked.join(", ") || "no data yet"}.
People did not like: ${disliked.join(", ") || "no data yet"}.

${sexy === "flirty" ? `She is a glamorous, sexy Instagram model and every photo is a flattering shoot of HER, not a
travel photo: she is the subject, filling most of the frame. Outfits are figure-flattering and a
little revealing -- bodycon and mini dresses, crop tops, low-cut and backless tops, bikinis at the
beach or pool -- but never lingerie, underwear or see-through. Poses are alluring, flirty and
model-like -- hip popped, back slightly arched, a hand in her hair, a sultry gaze or a teasing smile --
never explicit (Instagram must still recommend her to people who don't follow her).` : sexy ? `She is a glamorous, sexy social-media model and every photo is a flattering shoot of HER, not a
travel photo: she is the subject, filling most of the frame. Outfits are figure-flattering and
revealing and show off her body (micro bikinis, tiny crop tops, micro mini dresses, plunging necklines,
cut-outs, backless), but there is NO nudity. Poses are seductive and model-like: hip popped, back
arched, hands in her hair or tracing her waist, looking over her shoulder, a sultry, seductive gaze.` : `She is a glamorous Instagram fashion and travel model and every photo is a flattering shoot of
HER, not a travel photo: she is the subject, filling most of the frame. Outfits are stylish and
flattering -- never lingerie, underwear or see-through; swimwear only at the beach or pool. Poses
are confident and elegant, never sexual, never lying in bed (Instagram does not recommend
sexually suggestive posts to people who don't follow her).`}

Write ${n} NEW scenes, one per place and outfit, in this order: ${pairs}.
Do not describe her face or hair; that is fixed.
Recent scenes -- do NOT repeat these:
${recent}

Return ONLY a JSON array of ${n} objects with these keys:
"setting" (the place, specific), "outfit" (the given outfit, with colours, fabric and fit, e.g. "a black satin slip mini dress with thin straps, softly clinging"), "action" (what she is
doing, saying where BOTH of her hands are, e.g. "one hand on her hip, the other relaxed at her side"
or "holding a glass in her right hand, her left hand in her hair"; at most ONE object; never several
objects; her body facing roughly toward the camera, no twisting; she is POSING for the photo at the place, never
caught mid-action: no throwing, swinging, running or jumping), "shot" (camera framing), "light" (where
the light comes from and what it does, e.g. "warm low sun from her left, a soft glow on her cheek and a
long shadow behind her"),
"mood", "motion" (one sentence: how she moves slowly in a 5-second video), "caption" (first-person
social caption, max 20 words, 1-2 emojis, no hashtags; never starting with "Sun's out", "Summer vibes",
"Who's ready" or "Who's joining"), "hook" (2-6 words of on-screen text for the first second of her Reel,
in her voice, that makes a stranger stop scrolling: a tease, a confession or a POV, e.g. "POV: our first
date", "he said I wouldn't wear it", "guess where I woke up"; no emojis, no hashtags, never asking people
to like, comment or follow), "beat" (one past-tense sentence for her storyline), "tags" (3-5 short
lowercase tags).`;
}

// Gloves read as extra arms (Lena's black opera gloves, 2026-10-23): dropped from
// every outfit the model writes; persona/scenes.py no_gloves does the same.
const GLOVES = /(?:,\s*|\s+(?:with|and|plus)\s+)[^,;.]*?\b(?:gloves?|mittens?|arm[- ]warmers?|gauntlets?)\b[^,;.]*/gi;
export const noGloves = (o) => String(o || "").replace(GLOVES, "").replace(/^[\s,;]+|[\s,;]+$/g, "") || o;

function parse(text, n) {
  const m = /\[[\s\S]*\]/.exec(text || "");
  if (!m) throw new Error("no JSON array");
  const arr = JSON.parse(m[0].replace(/,\s*([\]}])/g, "$1"));
  const keys = ["setting", "outfit", "action", "shot", "light", "mood", "motion", "caption", "hook", "beat", "tags"];
  const good = arr.filter((s) => s && s.setting && s.outfit && s.action).map((s) => {
    for (const k of keys) if (s[k] === undefined) s[k] = k === "tags" ? [] : "";
    s.tags = (Array.isArray(s.tags) ? s.tags : []).map((t) => String(t).toLowerCase().trim()).filter(Boolean).slice(0, 6);
    s.outfit = noGloves(s.outfit);
    s.hook = String(s.hook || "").replace(/[#\p{Extended_Pictographic}]/gu, "").trim().split(/\s+/).slice(0, 7).join(" ");
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

/**
 * Head angle and expression, picked separately (ideas.json angles/expressions, the
 * same as persona/scenes.py assign_gazes): about a quarter face the lens, the rest
 * are turned, each with its face-sheet portrait. The old mild gazes made nearly
 * every photo face the camera.
 */
let prefState = {};
function assignGazes(list, state = {}) {
  prefState = state;
  if (!IDEAS.angles?.length) {
    const g = IDEAS.gazes, start = Math.floor(Math.random() * g.length);
    list.forEach((s, i) => { [s.gaze, s.sheet] = g[(start + i * 3) % g.length]; });
    return list;
  }
  // Every head direction once before any repeats (2026-10-18: weighted random picks
  // gave whole shoots the same angle): a shuffled deck of directions, dealt in turn.
  const bySheet = {};
  for (const a of IDEAS.angles) (bySheet[a[1]] = bySheet[a[1]] || []).push(a);
  let deck = [];
  const used = new Set();
  for (const s of list) {
    if (!deck.length) deck = Object.keys(bySheet).sort(() => Math.random() - 0.5);
    // A direction you keep redoing goes to the back of the deck (seen less in short shoots).
    deck.sort((x, y) => prefW(prefState, `angle:${y}`) - prefW(prefState, `angle:${x}`) + (Math.random() - 0.5) * 0.6);
    const angle = pick(bySheet[deck.shift()]);
    const free = IDEAS.expressions.filter((e) => !used.has(e[0]));
    const expr = pickPref(prefState, free.length ? free : IDEAS.expressions, (e) => `expr:${e[0]}`);
    used.add(expr[0]);
    s.angle = angle[0]; s.expression = expr[0]; s.angle_sheet = angle[1];
    s.gaze = `${angle[0]}; ${expr[0]}`;
    s.sheet = angle[1] || expr[1];
  }
  return list;
}

/** n scenes, in chunks of 4 (an 8B model breaks long JSON), each chunk retried once. */
export async function write(env, b, state, n, { sexy = false } = {}) {
  const out = [], taken = [];
  while (out.length < n) {
    const k = Math.min(4, n - out.length);
    const { places, outfits } = seeds({ ...state, recent: [...state.recent, ...out.map((s) => s.setting)] }, k, taken, sexy);
    taken.push(...places.map((place, i) => ({ place, outfit: outfits[i] })));
    let got = null;
    for (let attempt = 0; attempt < 2 && !got; attempt++) {
      try {
        const r = await env.AI.run(MODEL, { messages: [{ role: "user", content: prompt(b, state, places, outfits, sexy) }],
                                            max_tokens: 2000, temperature: 0.9 });
        got = parse(text(r), k);
      } catch (e) {
        console.log("scene writing failed", String(e).slice(0, 150));
      }
    }
    out.push(...(got || fallback(places, outfits)));
  }
  const list = assignGazes(out.slice(0, n));
  // Sexy style: most shots show her whole figure (asked for, 2026-10-17).
  if (sexy) list.forEach((sc, i) => { if (i % 5 !== 4) sc.shot = pick(FULL_SHOTS); });
  return list;
}

// The makeup level for a whole shoot, by the same rules as persona/scenes.py
// makeup_for (glam out at night, none at home or the gym, light otherwise).
const BARE = /\b(bed|bedroom|home|apartment|couch|sofa|kitchen|in bed|waking up|sauna|spa|gym|pilates|yoga|workout|hot tub|bath|pajamas?|loungewear|lounge set|hoodie|sweatpants|sports bra|laundromat)\b/i;
const GLAM = /\b(night|nightclub|bar|gala|casino|party|cocktail|red carpet|jazz|karaoke|evening|opera|cinema|theat(er|re)|dinner|restaurant|sequins?|gown|festival|fashion week|premiere|neon|limo|yacht party|ball)\b/i;
function makeupFor(place, outfit) {
  if (BARE.test(place) || (BARE.test(outfit) && !GLAM.test(place))) return "none";
  return GLAM.test(place) || GLAM.test(outfit) ? "glam" : "light";
}

/**
 * One photoshoot: ONE place, ONE outfit, ONE day -- the photos of a carousel
 * (2026-10-18: "1 carousel should be literally the same girl, same clothes, 1 day";
 * a shoot of 9 places and 9 outfits read as 9 posts). The scenes move between spots
 * of that place and change pose, framing and head angle; outfit and makeup are the
 * same on every photo.
 */
/** The poses you send on and the ones you redo, for the scene writer. */
function poseNotes(state) {
  const poses = Object.entries(state.prefs || {}).filter(([k]) => k.startsWith("pose:")).sort((a, b) => b[1] - a[1]);
  const good = poses.filter(([, v]) => v >= 1).slice(0, 5).map(([k]) => k.slice(5));
  const bad = poses.filter(([, v]) => v <= -1).slice(-5).map(([k]) => k.slice(5));
  if (!good.length && !bad.length) return "";
  return `${good.length ? `Poses that worked well before (use similar ones): ${good.join("; ")}.\n` : ""}${bad.length ? `Poses that did NOT work (avoid): ${bad.join("; ")}.\n` : ""}\n`;
}

export async function writeShoot(env, b, state, n, { sexy = false } = {}) {
  const { places, outfits } = seeds(state, 1, [], sexy);
  const place = places[0], outfit = outfits[0];
  const p = `${prompt(b, state, [place], [outfit], sexy)}

${poseNotes(state)}IMPORTANT, this replaces the count above: write ${n} photos of ONE photoshoot on ONE day at ${place}, all
in the SAME outfit (${outfit}, the same colours in every photo). Each photo is a different spot of that
same place or right next to it (e.g. the entrance, a table, a window, the terrace, the street outside),
with a different pose and framing, in time order through that one outing. Return ONLY a JSON array of
${n} objects with the keys above.`;
  let list = null;
  for (let attempt = 0; attempt < 2 && !list; attempt++) {
    try {
      const r = await env.AI.run(MODEL, { messages: [{ role: "user", content: p }], max_tokens: 3000, temperature: 0.9 });
      list = parse(text(r), n);
      if (list.length < Math.min(n, 4)) list = null;
    } catch (e) {
      console.log("shoot writing failed", String(e).slice(0, 150));
    }
  }
  list = list || fallback(Array(n).fill(place), Array(n).fill(outfit));
  while (list.length < n) list.push({ ...list[list.length % Math.max(1, list.length)] });
  // The outfit word for word on every photo: the model drifted colours otherwise.
  const outfitText = list[0].outfit || outfit;
  const makeup = makeupFor([place, ...list.map((sc) => `${sc.setting} ${sc.light}`)].join(" "), outfitText);
  list.forEach((sc) => { sc.outfit = outfitText; sc.makeup = makeup; sc.shoot_place = place; });
  list = assignGazes(list.slice(0, n), state);
  if (sexy) list.forEach((sc, i) => { if (i % 5 !== 4) sc.shot = pickPref(state, FULL_SHOTS, (x) => `shot:${x.toLowerCase().split(/\s+/).slice(0, 3).join(" ")}`); });
  // Body views (2026-10-19, asked for): one photo from behind and one side-on,
  // the rest facing the camera. From behind her face is not seen (nothing to check
  // or twist); side-on her head is in profile like her body.
  if (n >= 4) {
    const back = list[2], side = list[n - 2], dir = pick(["left", "right"]);
    Object.assign(back, { view: "back", shot: pick(BACK_SHOTS), angle: "her back to the camera, glancing back over her shoulder, face in profile",
                          angle_sheet: "", sheet: "", gaze: "glancing back over her shoulder" });
    Object.assign(side, { view: "side", shot: `full-body side view, her whole body in profile facing ${dir === "left" ? "the left" : "the right"} of the frame`,
                          angle: `face in side profile like her body, looking ahead`, angle_sheet: `profile-${dir}`, sheet: `profile-${dir}`,
                          gaze: `side profile, looking ahead` });
  }
  return list;
}

const BACK_SHOTS = ["full-body shot from behind, her back to the camera, glancing back over her shoulder",
                    "full-body shot from behind as she walks away, looking back over her shoulder",
                    "three-quarter-body shot from behind, hand in her hair, glancing back over her shoulder"];

const FULL_SHOTS = ["full-body shot, head to toe, showing her whole figure", "three-quarter-body shot from head to knees",
                    "full-body shot from a slightly low angle, head to toe"];

