/**
 * Kaggle's API, called directly: the same three calls the kaggle CLI makes
 * (kagglesdk's KernelsApiService), with KAGGLE_API_TOKEN as a bearer token.
 */
const API = "https://api.kaggle.com/v1/kernels.KernelsApiService";

async function call(env, method, body) {
  const r = await fetch(`${API}/${method}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${env.KAGGLE_API_TOKEN}`, "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`kaggle ${method} ${r.status}: ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) : {};
}

function split(slug) {
  const [userName, kernelSlug] = slug.split("/");
  return { userName, kernelSlug };
}

/** "RUNNING" | "QUEUED" | "COMPLETE" | "ERROR" | "CANCEL..." | "" */
export async function status(env, slug) {
  try {
    return (await call(env, "GetKernelSessionStatus", split(slug))).status || "";
  } catch (e) {
    return `UNKNOWN: ${e.message}`;
  }
}

/** The latest completed run's output files: { "out/items/x.jpg": url, ... }. */
export async function outputs(env, slug) {
  const r = await call(env, "ListKernelSessionOutput", split(slug));
  const files = {};
  for (const f of r.files || []) files[f.fileName] = f.url;
  return files;
}

export async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${r.status}`);
  return r.json();
}

/** Save and run a private T4 script kernel. */
export async function push(env, slug, text, datasets) {
  return call(env, "SaveKernel", {
    slug,
    newTitle: slug.split("/")[1],
    text,
    language: "python",
    kernelType: "script",
    isPrivate: true,
    enableGpu: true,
    enableInternet: true,
    datasetDataSources: datasets || [],
    machineShape: env.KAGGLE_ACCELERATOR || "NvidiaTeslaT4",
  });
}

/**
 * The script Kaggle runs: clone the public repo and run persona/kaggle_kernel.py
 * with this payload. The persona code is not bundled into the Worker.
 */
export function bootstrap(payload) {
  const b64 = btoa(unescape(encodeURIComponent(JSON.stringify(payload))));
  return `import os, runpy, subprocess
PAYLOAD = "${b64}"
subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/gesimuse/mpt", "/kaggle/tmp/boot"], check=True)
src = open("/kaggle/tmp/boot/persona/kaggle_kernel.py").read()
src = src.replace('PAYLOAD_B64 = "__PAYLOAD_B64__"', 'PAYLOAD_B64 = "' + PAYLOAD + '"', 1)
open("/kaggle/tmp/persona_kernel.py", "w").write(src)
runpy.run_path("/kaggle/tmp/persona_kernel.py", run_name="__main__")
`;
}
