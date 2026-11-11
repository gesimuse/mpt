// Talk lines, tested without the Worker or a GPU: transcripts of real viral shorts
// (sources.json from the collect step) go through the Worker's own adaptTalk + judge,
// with Workers AI's Llama 3.3 70B swapped for the same model on GitHub Models.
//   node scripts/talk_lines_test.mjs sources.json [name] > lines.json
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";
const { _talkTest } = await import(process.env.WORKER_BUNDLE || "../worker/src/persona/index.js");
const token = execSync("gh auth token").toString().trim();
const env = { AI: { run: async (_m, { messages, max_tokens, temperature }) => {
  for (let i = 0; i < 4; i++) {
    const r = await fetch("https://models.github.ai/inference/chat/completions", { method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: "meta/Llama-3.3-70B-Instruct", messages, max_tokens, temperature }) });
    if (r.ok) return { response: (await r.json()).choices[0].message.content };
    await new Promise((ok) => setTimeout(ok, 15000 * (i + 1)));
  }
  throw new Error("GitHub Models failed");
} } };
const sources = JSON.parse(readFileSync(process.argv[2], "utf8"));
const b = { name: process.argv[3] || "Lena", age: 24 };
const out = [];
for (const s of sources) {
  let r;
  try { r = await _talkTest.adaptTalk(env, b, "dating truths for women", s.text, s.views, [], true); }
  catch (e) { r = { error: String(e) }; }
  out.push({ src: s.src, title: s.title, views: s.views, said: s.text.slice(0, 300), result: r });
  console.error(`${s.id}: ${r === null ? "REJECTED (not a talk)" : r === false ? "REJECTED (no clear line)" : r.line || r.error}`);
}
console.log(JSON.stringify(out, null, 1));
