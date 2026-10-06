---
name: new-persona
description: Create a new AI persona (a new "model" girl) for the persona studio -- design her identity in chat, write her bible, cast her face on the local GPU, build and lock her master refs. Use when the user wants a new persona/model/character, or to re-cast an existing one.
---

# New persona

You (Claude, in this session) are the designer. No API key or Telegram command is
involved: you write her bible, then drive `persona.cli` on the local GPU and show the
results here so the user picks.

Background: `persona/README.md`. Her data lives in `~/.local/share/mpt-persona/<slug>/`,
never in the repo (it is public).

## 1. Concept

If the user gave a brief, use it. Otherwise ask at most 3 quick questions (vibe and
niche, look and ethnicity, home city). Then propose 2-3 distinct concepts in a few
lines each and let the user pick or mix. Rules, non-negotiable:
- fictional and original: never modelled on, or named after, a real person,
  celebrity or existing influencer
- clearly adult: age 21-35, and she must read as an adult in every image
- the user wants her sexy: glamorous Instagram/lifestyle model type, but the social
  lane stays Instagram-safe (no nudity); explicit content belongs only to the local
  Fanvue lane cue, which the user writes or approves

## 2. Bible

Write the chosen concept as JSON to the scratchpad, then import it. Fields:

```json
{
  "name": "Valentina",
  "age": 24,
  "identity": "Latina, long dark-brown waves, warm brown eyes, high cheekbones and full lips, glamorous makeup, slim waist, curvy hips, toned body",
  "tagline": "Miami fitness girl who lives in a bikini",
  "personality": "confident, flirty, playful, disciplined about training, loves nights out",
  "backstory": "Personal trainer who started posting her workouts and beach days.",
  "home_city": "Miami",
  "recurring_places": ["her gym", "South Beach at golden hour", "a rooftop pool", "her bedroom mirror", "a cocktail bar at night"],
  "style": ["string bikinis", "matching gym sets", "bodycon mini dresses", "crop tops with low-rise jeans", "silk slip dresses"],
  "content_pillars": ["fitness", "beach life", "nights out"],
  "casting": {
    "ethnicity": ["Latina"],
    "hair": ["long dark-brown waves"],
    "eyes": ["warm brown eyes"],
    "face": ["high cheekbones and full lips, glamorous makeup"],
    "build": ["slim waist, curvy hips, toned body"]
  },
  "lanes": {
    "social": {"cue": "sexy and confident Instagram model look, glamorous makeup, styled hair, figure-flattering revealing outfit, no nudity", "negative": "nudity, nipples, genitals"}
  }
}
```

- `identity` is repeated in every prompt next to her refs: concrete and visual.
- `casting` pools: ONE value each pins the concept so every candidate is a take on
  the same girl; give 2-3 values in a pool only where the user is undecided.
- Do not set `lanes.fanvue` unless the user gives that wording.

```bash
~/apps/Wan2GP/.venv/bin/python -m persona.cli import <scratchpad>/bible.json
```

This creates her and makes her the ACTIVE persona, which the scheduled bot posts as
from the next slot. Say so.

## 3. Cast (GPU)

The laptop bot service (if running) holds the same GPU and RAM, so stop it first:

```bash
systemctl --user stop mpt-persona-bot
cd ~/apps/mpt && ~/apps/Wan2GP/.venv/bin/python -m persona.cli cast 6
```

About 25s per candidate after the first load. Each line prints `<id> <path> <look>` or
a rejection reason. Read every image and show them: SendUserFile with all candidate
paths so the user sees them on any device, plus your own one-line take on each.
Re-cast with tweaks (edit `casting`/`identity` in her bible.json, or pass a hint:
`cast 6 "softer jawline"`) until the user picks one.

## 4. Master refs

```bash
~/apps/Wan2GP/.venv/bin/python -m persona.cli pick <id>
~/apps/Wan2GP/.venv/bin/python -m persona.cli refs
```

Four refs (front, three-quarter, profile, full body) land in `casting/refs-pending/`,
each with a face similarity against the pick. Show them. If one drifted (front or
three-quarter below ~0.75), run `refs` again. When the user is happy:

```bash
~/apps/Wan2GP/.venv/bin/python -m persona.cli lock
```

## 5. Face sheet

Twelve portraits of her at different head angles and expressions (profiles, over the
shoulder, looking down/up, laughing, big smile, sultry, kiss, shy). Each scene uses
the one matching its pose as a second reference, so she is not always front-facing
with the same smile:

```bash
~/apps/Wan2GP/.venv/bin/python -m persona.cli sheet
```

About 1 min each. Show them; re-render a bad one with `sheet <stem>` (e.g.
`sheet laughing kiss`). Rejected ones (face drifted) are not kept.

## 6. Test and hand back

Render two scenes so the user sees her in the wild, show them, sync her to the cloud
dataset (the laptop-free schedule renders from it), then restart the bot if used:

```bash
~/apps/Wan2GP/.venv/bin/python -m persona.cli scene 2
PATH=$HOME/.local/bin:$PATH ~/apps/Wan2GP/.venv/bin/python -m persona.cli sync
```

To switch back to an earlier persona: `python -m persona.cli switch <slug>`
(`python -m persona.cli list` shows them).
