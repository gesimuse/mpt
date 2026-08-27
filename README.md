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
Push this repo, add every key from `.env.example` as a repo **Secret**, done. Runs
07:00/19:00 UTC daily. Trigger manually from the Actions tab, optionally scoped to
`aibeauty` via the `niches` input. Public repo = unlimited free minutes.

### Local GPU -- generate, look at it yourself, then push
`sdgen.py` auto-detects a local CUDA GPU (falls back to CPU otherwise) -- generation
that takes ~130s/image on GitHub's CPU runner takes ~1s/image on a real GPU.

```bash
DRY_RUN=1 python3 autopilot.py          # generate + QA, write to ./out, queue nothing
# look at ./out/aibeauty-<stamp>/*.png yourself
python3 push_draft.py out/aibeauty-<stamp>   # host + queue that batch as an inbox draft
```

## Tuning
- `niches.json`: `civitai_query` steers which checkpoint gets picked; `outfits` /
  `mood` / `camera_modifiers` steer tone; `images_per_video` (variations generated per
  batch), `min_images` (floor to post), `max_images` (cap on what ships).
- `SUPERVISOR_MIN_REALISTIC` (default 6/10) -- the vision QA's pass threshold.
- `SUPERVISOR_ENABLED=0` skips vision QA entirely -- local testing only, never for a
  real run.
