# Replacing a raw TikTok integration with Buffer

Notes for any system that posts to TikTok from a server and keeps hitting the same
wall. This is what we learned wiring it into MPT Autopilot; the reasoning transfers to
any pipeline, language, or scheduler.

## The wall

TikTok's Content Posting API gives an unaudited app exactly two options, and neither is
what an automated pipeline wants:

| endpoint | what you get | the catch |
|---|---|---|
| `/v2/post/publish/inbox/video/init/` | video lands in the creator's drafts | **no caption field exists.** Title, description, hashtags and privacy are rejected in `MEDIA_UPLOAD` mode by design — a human writes them in the app |
| `/v2/post/publish/video/init/` | direct post, caption included | needs `video.publish`, and until the app passes TikTok's audit `privacy_level` may only be `SELF_ONLY`, i.e. the post is invisible |

So before audit you choose between *public but hand-captioned* and *auto-captioned but
private*. There is no third setting. Chasing the audit is often the wrong move: review
weighs whether a real product exists, whether it serves users other than the developer,
and how mass-produced the content is. A single-operator automation usually fails on the
first two regardless of code quality.

## Why Buffer sidesteps it

Buffer publishes to TikTok through **Buffer's own approved app**. Their audit, not
yours. A post created through their API arrives on TikTok public, with caption and
hashtags attached. You are a Buffer user, not a TikTok developer, so the audit question
never arises.

The same reasoning applies to any TikTok-approved intermediary — Later, Metricool,
upload-post.com. What matters is that the approval belongs to them.

## What this costs you

Be clear-eyed before switching:

- **A dependency and a price.** Buffer's free tier covers 3 channels; beyond that it is
  per channel per month. Their API is bundled, not sold separately.
- **A beta surface.** The GraphQL API is in public beta. The schema can move under you.
- **No new OAuth clients.** Buffer stopped accepting developer app registrations, so
  there is no way to onboard third parties. Personal API keys work, which is fine for a
  single-operator pipeline and useless for a multi-tenant product.
- **You must host the video.** See below — this is the part that surprises people.

## Discovering the API instead of trusting docs

The published documentation lagged reality on every point that mattered. Introspection
answered each question in seconds:

```python
gql('{ __schema { mutationType { fields { name } } } }')          # what can I call
gql('query($n:String!){ __type(name:$n){ inputFields { name type { name kind } } } }',
    {"n": "CreatePostInput"})                                      # what does it take
gql('query($n:String!){ __type(name:$n){ possibleTypes { name } } }',
    {"n": "PostActionPayload"})                                    # what comes back
```

Three findings worth knowing up front:

1. **Endpoint.** `https://api.buffer.com/graphql`. `graph.buffer.com` answers
   `Please use api.buffer.com`, and the legacy REST API rejects these tokens outright
   with `Public API tokens are not accepted for REST API access`.
2. **Results are unions, not nullable objects.** `createPost` returns
   `PostActionSuccess | NotFoundError | UnauthorizedError | UnexpectedError |
   RestProxyError | LimitReachedError | InvalidInputError`. Query `__typename` plus
   `message` on each error member, or failures surface as missing fields instead of
   readable errors.
3. **No upload endpoint.** `VideoAssetInput` requires `url: String!`. Buffer fetches
   the file; it will not accept bytes. Nothing in the schema mentions upload, signed
   URLs, or media, and multipart posts to `api.buffer.com` return
   `Unsupported Content-Type`.

## The shape of a post

```graphql
mutation($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id dueAt } }
    ... on InvalidInputError { message }
    # ...and the other error members
  }
}
```

```json
{
  "channelId": "<the TikTok channel>",
  "text": "caption including #hashtags",
  "assets": [{ "video": { "url": "https://public/host/video.mp4",
                          "thumbnailUrl": "https://public/host/cover.jpg" } }],
  "mode": "addToQueue",
  "schedulingType": "automatic",
  "needsApproval": false,
  "metadata": { "tiktok": { "title": "...", "isAiGenerated": true } }
}
```

Things that bite:

- `schedulingType: "automatic"` means Buffer publishes it. `"notification"` sends a
  phone reminder instead — which reintroduces the manual step you came here to remove.
- `mode` is `addToQueue` (next free slot in your schedule), `shareNow`, or
  `customScheduled` with `dueAt`. Queue mode spreads posts and respects plan limits.
- `saveToDraft: true` creates a Buffer draft that never publishes. **Use it for every
  integration test**, then `deletePost`. Testing against a real channel is otherwise a
  good way to publish nonsense.
- `metadata.tiktok.isAiGenerated` exists — set it. TikTok requires AI-generated content
  to be disclosed, and this is one boolean.

## Hosting the video

Because `VideoAssetInput` needs a URL, a pipeline that renders on an ephemeral runner
must publish the file somewhere public first. Requirements are stricter than they look:

- must return `Content-Type: video/mp4` and the actual bytes. Several popular "temporary
  file host" services return an HTML landing page, which Buffer rejects with
  `Invalid post: Video could not be read from its URL.` That error nearly always means
  the host, not the file.
- must be reachable without authentication, since Buffer's servers do the fetching.

Options, roughly in order of friction: a GitHub release asset on a public repo (free,
stable URL, keeps binaries out of git history, and `GITHUB_TOKEN` already has
`contents: write` inside Actions); object storage such as R2 or S3; or any static host
you already run. Whatever you choose, verify it with
`curl -o /dev/null -w '%{content_type}' <url>` before blaming the API.

## Migration shape

Keep the old path as a fallback rather than deleting it. The failure modes are
different — Buffer can be down, over quota, or reject a URL, while the TikTok inbox
draft only needs a refresh token — and a fallback turns an outage into a mild
inconvenience.

```python
def publish_tiktok(video, meta, niche, caption):
    if buffer.enabled():                       # presence of a token is the switch
        try:
            return True, "buffer", buffer.publish(video, caption, title=meta["title"])
        except Exception as e:
            log(f"Buffer failed ({e}); falling back to an inbox draft")
    return bool(upload_tiktok(video, meta, niche)), "inbox", ...
```

Record which path ran. Downstream behaviour depends on it: a Buffer post already carries
its caption, while an inbox draft still needs the human to paste one, so anything that
prepares captions for manual use should skip the Buffer rows.

## Verification checklist

1. `{ account { organizations { id } } }` — token valid.
2. `channels(input: {organizationId})` — the TikTok channel is connected, note its `id`.
3. `curl` your hosted URL, confirm `video/mp4`.
4. `createPost` with `saveToDraft: true`, confirm `PostActionSuccess`, then `deletePost`.
5. Only then enable the real mode.

Step 4 is the one people skip, and it is the only step that proves the whole chain.
