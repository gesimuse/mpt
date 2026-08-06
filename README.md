# MPT Autopilot

Fully automated faceless-channel pipeline: NVIDIA NIM generates a topic per niche, MoneyPrinterTurbo builds the video (free Pexels footage + free Edge TTS), then it auto-publishes to YouTube (free API) and optionally TikTok (Upload-Post). Four niches included in `niches.json`. Topic history is stored in `posted.json` so it never repeats itself.

## One-time setup (~45 min)

1. **API keys (all free):**
   - NVIDIA NIM key at build.nvidia.com (free credits, OpenAI-compatible endpoint).
   - Pexels API key at pexels.com/api.
2. **Create 4 YouTube channels** (they can live under one Google account as brand channels, or separate accounts — separate is safer).
3. **Google Cloud:** create a project, enable *YouTube Data API v3*, create an OAuth client (Desktop type) → gives you `YT_CLIENT_ID` and `YT_CLIENT_SECRET`. Add yourself as a test user on the consent screen.
4. **Per channel**, run `python get_youtube_token.py` locally, sign in as that channel, and save the printed refresh token as `YT_REFRESH_TOKEN_<NICHEID>`.
5. *(Optional, TikTok)* create an upload-post.com account, connect the TikTok accounts using the niche ids as profile names, and set `UPLOAD_POST_API_KEY`. Free tier = 10 uploads/month, so treat TikTok as a teaser channel or upgrade later.

## Choose ONE runtime

### A. GitHub Actions — recommended, free, true autopilot
1. Push this folder to a GitHub repo.
2. Add every key from `.env.example` as a repo **Secret** (Settings → Secrets → Actions).
3. Done. It runs at 06/12/18/23 UTC daily. Each run commits posted.json back to the repo, which persists your topic history and keeps the schedule alive forever (GitHub disables cron on repos inactive 60+ days — this sidesteps that). Make the repo public for unlimited free minutes; secrets stay encrypted regardless. Trigger manually anytime from the Actions tab.

### B. Docker — for a home PC / VPS that stays on
```bash
cp .env.example .env   # fill it in
touch posted.json && echo '{"topics":{},"uploads":[]}' > posted.json
docker compose up -d --build
```

### C. Colab — testing only
Open `colab_autopilot.ipynb`. Free Colab cannot schedule itself; sessions die. Fine for verifying the pipeline, not for autopilot.

## Tuning
- `RUN_HOURS` (Docker) or the cron lines (Actions) control frequency. Default: 4 runs/day × 4 niches = 16 videos/day. **Start with 1–2 runs/day** — brand-new channels that suddenly post 4x/day get flagged as spam.
- Edit `niches.json` to change prompts, voices, hashtags, or add niches; add a matching `YT_REFRESH_TOKEN_<ID>` secret.
- YouTube API default quota allows ~6 uploads/day per Google Cloud project. With 4 niches × 1 video/run you'll hit that at 2 runs/day; use one Cloud project per channel to scale past it.

## Known limits (honesty section)
- TikTok "free at scale" doesn't exist: TikTok's own API posts privately until your app passes an audit, and Upload-Post's free tier is 10/month.
- YouTube may limit monetization of mass-produced content; channels with a consistent editing hook and occasional human-made videos survive far better. Check `posted.json` weekly and prune topic styles that flop.
