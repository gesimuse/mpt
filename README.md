# MPT Autopilot — aibeauty

Free, local AI-portrait TikTok pipeline. Each run: search CivitAI's public gallery for
a real, well-liked checkpoint and the actual prompt behind one of its showcase photos
(`civitai.py`), generate camera/lighting variations of that one prompt locally with
Stable Diffusion (`sdgen.py`, CPU or GPU), run every image through a vision-model QA
pass before anything ships (`supervisor.py`), then queue the survivors as a **native
TikTok inbox draft** (`tiktok.py`) -- not auto-published. The account owner opens the
TikTok app, adds trending sound and the caption (printed to `CAPTIONS.md`), and posts
by hand.

No paid APIs anywhere. CivitAI's key is read-only (search/gallery only, never near
their separate paid generation API) -- every image is generated on this machine.

**Not yet verified against a real TikTok account** -- see `tiktok.py`'s module
docstring for what's assumed and what to check on the first real run.

## One-time setup

1. **CivitAI API key** at civitai.com/user/account -- read-only, fixes a region block
   on the search/gallery endpoints (model-version lookups and downloads work without it).
2. **TikTok developer app** at developers.tiktok.com: add "Login Kit" + "Content
   Posting API" products, platform Desktop, redirect URI `http://localhost:8723/callback`.
   Copy the Client Key/Secret into `.env` as `TIKTOK_CLIENT_KEY` / `TIKTOK_CLIENT_SECRET`.
3. **Verify the fetch domain** for Content Posting API's PULL_FROM_URL under the app's
   settings (Content Posting API → URL properties) -- TikTok requires this before it
   will fetch images from a URL; without it every draft push fails with a
   domain-verification error from TikTok's own API.
4. Run `python3 get_tiktok_token.py` locally, **signed into the aibeauty TikTok
   account**, and save the printed refresh token as `TIKTOK_REFRESH_TOKEN_AIBEAUTY`.

## Running

### GitHub Actions -- recommended, free, true autopilot
Push this repo, add every key from `.env.example` as a repo **Secret**, done. The photo
run fires 5x/day (05/07/09/11/13 UTC) and is capped at 5 drafts per **calendar day** --
a midnight reset, not a rolling 24h window, so yesterday's late runs never occupy this
morning's budget. Trigger manually from the Actions tab, optionally scoped to
`aibeauty` via the `niches` input. Public repo = unlimited free minutes.

The video run (`autopilot_video.yml`) has no cron on purpose: every generated image is
posted to a Telegram channel with **🎬 Make video / ✅ Done / 🗑 Skip** buttons, and a
video is made when you press one (or reply with your own motion prompt). See
`telegram.py` for the sending half and `worker/` for the Cloudflare Worker that handles
the button presses -- nothing in this repo is running when a button is pressed, which is
why that half exists.

### Local GPU -- generate, look at it yourself, then push
`sdgen.py` auto-detects a local CUDA GPU (falls back to CPU otherwise) -- generation
that takes ~130s/image on GitHub's CPU runner takes ~1s/image on a real GPU.

```bash
DRY_RUN=1 python3 autopilot.py          # generate + QA, write to ./out, queue nothing
# look at ./out/aibeauty-<stamp>/*.png yourself
python3 push_draft.py out/aibeauty-<stamp>   # host + queue that batch as an inbox draft
```

## Captions and hashtags
Written per post by an LLM from the theme that batch actually used -- not the fixed
string in `niches.json` (that is only the last-resort fallback). `llm.py` tries HF's
router first (`HF_TOKEN`, rotating `HF_TOKENS` when an account runs out of free
Inference Providers credit), then a local Ollama. The HF token has to be a
**fine-grained** one with *"Make calls to Inference Providers"*; a plain read token 401s
and the run falls back to Ollama.

`niches.json`'s `trend_hashtags` is a short list you refresh by hand from TikTok's
Creative Center; it is offered to the caption model as a hint, not pasted verbatim.
There is no automated trending feed: Creative Center's endpoint answers every
unauthenticated request with `{"code":40101,"msg":"no permission"}`, and CivitAI's
`period` filter returns nothing at all alongside a search query. Both were checked live
-- see `trends.py`'s docstring before re-attempting either.

## Video generation
`videogen.py` walks a ladder of HF ZeroGPU Spaces (Wan 2.2 I2V rCM -> Wan 2.1 fast ->
LTX-Video distilled). Within each, `HF_TOKENS` rotates across HF **accounts** on a
quota error -- ZeroGPU gives roughly 5 GPU-minutes per account per day, so more
accounts is the only real way to raise the ceiling.

Running a model on our own Kaggle T4 instead was built, worked end to end, and was
removed: nothing that fits a 15 GB card produced output worth posting next to Wan 2.2.
When quota is gone the run fails and waits rather than shipping something worse.

## What the pipeline learns
Two loops write back into `posted.json` and bias future runs:
- **QA pass rate**, per checkpoint (`model_stats`) and per theme (`theme_stats`) -- how
  often a batch's images came out well-formed.
- **Your own verdict**, from the Telegram channel's *Done / Skip* buttons (`owner_verdict`).
  QA only knows an image was well-formed; only you know it was worth posting. Real
  TikTok analytics would need the `video.list` scope, which requires an app audit this
  unaudited app cannot get -- this is the closest available signal.

## Tuning
- `niches.json`: `civitai_query` steers which checkpoint gets picked; `outfits` /
  `mood` / `camera_modifiers` steer tone; `images_per_video` (variations generated per
  batch), `min_images` (floor to post), `max_images` (cap on what ships).
- `SUPERVISOR_MIN_REALISTIC` (default 6/10) -- the vision QA's pass threshold.
- `SUPERVISOR_ENABLED=0` skips vision QA entirely -- local testing only, never for a
  real run.
