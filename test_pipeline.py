#!/usr/bin/env python3
"""Regression tests for the aibeauty autopilot flow.

No network, no API keys: every external call (CivitAI, NIM vision QA, TikTok, GitHub
release hosting) is stubbed. The point is to prove the wiring still holds -- that a
run reaches sdgen with the right prompt, that a rejected batch triggers a fresh round
instead of shipping, and that DRY_RUN never queues a draft.

Run: python3 test_pipeline.py
"""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

os.environ.pop("DRY_RUN", None)

import autopilot  # noqa: E402
import caption_writer  # noqa: E402
import civitai  # noqa: E402
import clip_encode  # noqa: E402
import hand_pose  # noqa: E402
import imageslides  # noqa: E402
import push_draft  # noqa: E402
import refine  # noqa: E402
import sdgen  # noqa: E402
import supervisor  # noqa: E402
import tiktok  # noqa: E402
import upscale  # noqa: E402
import videogen  # noqa: E402

NICHE = {"id": "aibeauty", "content_type": "images", "hashtags": "#aiart",
        "ai_disclosure": "Created with AI. Not a real person.", "captions": ["Soft light."]}


class Response:
    def __init__(self, status=200, payload=None, text="{}"):
        self.status_code, self._payload, self.text = status, payload or {}, text
        self.ok = status < 400
        self.headers = {}

    def json(self):
        return self._payload


class ImageSlideshowTest(unittest.TestCase):
    AIBEAUTY = {"id": "aibeauty", "content_type": "images",
                "hashtags": "#aiart", "ai_disclosure": "Created with AI. Not a real person.",
                "captions": ["Slow mornings."], "scenes": ["a street"], "styles": ["35mm"]}

    def test_swimwear_and_lingerie_are_not_blocked(self):
        """The policy line is nudity, not how much skin an outfit shows. Checks for
        "bikini"/"lingerie" rather than "swimsuit" -- one-piece swimsuits were dropped
        from the themes entirely (operator preference), so "swimsuit" itself no longer
        appears, but swim/lingerie-style outfits in general are still represented."""
        outfits = [t["outfit"] for t in imageslides.DEFAULT_THEMES]
        self.assertTrue(any("bikini" in o for o in outfits))
        self.assertTrue(any("lingerie" in o for o in outfits))
        self.assertNotIn("swimwear", imageslides.NEGATIVE_HARD)
        self.assertNotIn("lingerie", imageslides.NEGATIVE_HARD)

    def test_one_piece_swimsuits_are_excluded(self):
        """Explicit operator preference: no one-piece swimsuits."""
        outfits = [t["outfit"] for t in imageslides.DEFAULT_THEMES]
        self.assertFalse(any("one-piece" in o or "one piece" in o for o in outfits))

    def test_generate_raises_when_civitai_is_unavailable(self):
        """No static-formula fallback: a batch's model and prompt must come from a
        real CivitAI search, so an unreachable CivitAI must fail the run loudly
        instead of silently shipping generic images."""
        with mock.patch.object(civitai, "decide_reference",
                               mock.Mock(side_effect=RuntimeError("down"))), \
             tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertIn("down", str(ctx.exception))

    def test_query_is_rotated_across_a_list(self):
        """A fixed single query biases toward whatever CivitAI ranks highest for it
        every run -- civitai_queries lets the theme itself vary, not just which exact
        photo gets picked within one theme."""
        niche = {**self.AIBEAUTY,
                "civitai_queries": ["query one", "query two", "query three"]}
        seen = set()

        def fake_decide(query, prompt_filter=None, weights=None):
            seen.add(query)
            return ({"model_id": 1, "version_id": 1, "name": "X"},
                   {"prompt": "portrait of a woman", "negative_prompt": ""})

        with mock.patch.object(civitai, "decide_reference", fake_decide):
            for _ in range(30):
                imageslides.decide_reference(niche)
        self.assertEqual(seen, {"query one", "query two", "query three"},
                         "all three queries should get picked eventually")

    def test_build_variations_all_derive_from_the_same_base_prompt(self):
        """Ten variations of one decided prompt, not ten unrelated prompts -- only the
        camera/lighting framing should differ between them."""
        prompts, negatives = imageslides.build_variations(
            "adult woman, red gown", "a long reference description", "nude, blurry",
            10, self.AIBEAUTY)
        self.assertEqual(len(prompts), 10)
        for p in prompts:
            self.assertTrue(p.startswith("adult woman, red gown"))
            self.assertTrue(p.endswith("a long reference description"))
        self.assertEqual(negatives, ["nude, blurry"] * 10)
        self.assertGreater(len(set(prompts)), 1, "framing must actually vary")

    def test_camera_modifiers_include_full_body_framing(self):
        """The original list only ever had close-up/medium/three-quarter-style
        framing -- effectively portrait/closeup no matter which one got picked."""
        self.assertTrue(any("full body" in m or "full length" in m
                            for m in imageslides.CAMERA_MODIFIERS))

    def test_prefix_and_camera_modifier_precede_the_reference_text(self):
        """CLIP hard-truncates at 77 tokens -- a live run hit a showcase prompt long
        enough on its own to push everything appended after it (mood, camera framing)
        clean off the end. The controlled, short part must come first so a long
        reference prompt loses its own tail, not ours."""
        long_reference = "word " * 60  # long enough to risk truncation on its own
        prompts, _ = imageslides.build_variations(
            "adult woman, confident mood", long_reference, "", 3, self.AIBEAUTY)
        for p in prompts:
            head = p.split(long_reference.strip())[0]
            self.assertIn("adult woman, confident mood", head)
            self.assertTrue(p.rstrip().endswith(long_reference.strip()))

    def test_decide_reference_scopes_the_search_and_applies_the_subject_filter(self):
        """The niche's gender/portrait rules are passed to civitai.py as a filter
        function -- that module stays subject-agnostic by design."""
        captured = {}

        def fake_decide(query, prompt_filter=None, weights=None):
            captured["query"] = query
            captured["accepts_woman"] = prompt_filter("portrait of a woman")
            captured["rejects_man_only"] = prompt_filter("portrait of a man")
            return ({"model_id": 4201, "version_id": 130072, "name": "Test Model"},
                   {"prompt": "portrait of a woman", "negative_prompt": ""})

        with mock.patch.object(civitai, "decide_reference", fake_decide):
            resolved, reference = imageslides.decide_reference(self.AIBEAUTY)
        self.assertEqual(resolved["name"], "Test Model")
        self.assertEqual(reference["prompt"], "portrait of a woman")
        self.assertTrue(captured["accepts_woman"])
        self.assertFalse(captured["rejects_man_only"])

    def test_caption_discloses_ai(self):
        caption = imageslides.image_caption(self.AIBEAUTY)
        self.assertIn("Created with AI", caption)
        self.assertIn("#aiart", caption)

    def test_caption_uses_the_llm_writer_when_a_vibe_is_given(self):
        with mock.patch.object(imageslides.caption_writer, "write",
                               lambda vibe: (f"About {vibe}.", "#freshtag #vibecheck")):
            caption = imageslides.image_caption(self.AIBEAUTY, vibe="a rainy afternoon")
        self.assertIn("About a rainy afternoon.", caption)
        self.assertIn("#freshtag #vibecheck", caption)
        self.assertIn("Created with AI", caption)

    def test_caption_falls_back_to_the_static_pool_when_the_writer_fails(self):
        """A caption-writing hiccup must never block an otherwise-good batch of
        images from getting posted."""
        with mock.patch.object(imageslides.caption_writer, "write",
                               mock.Mock(side_effect=RuntimeError("NIM down"))):
            caption = imageslides.image_caption(self.AIBEAUTY, vibe="a rainy afternoon")
        self.assertIn("Slow mornings.", caption)

    def test_caption_falls_back_to_the_static_pool_without_a_vibe(self):
        """No vibe (e.g. a niche override with no vibe field) skips the LLM call
        entirely rather than calling it with nothing meaningful to write about."""
        with mock.patch.object(imageslides.caption_writer, "write",
                               mock.Mock(side_effect=AssertionError("must not be called"))):
            caption = imageslides.image_caption(self.AIBEAUTY, vibe=None)
        self.assertIn("Slow mornings.", caption)

    def test_theme_bundles_outfit_location_and_mood_coherently(self):
        """Independent random picks could land a bikini with 'candlelit bathtub' and
        'walking toward camera' in the same image -- individually fine, reads as a
        random recombination, not a scene. A theme's outfit/location/mood must come
        from the SAME bundle, not be shuffled independently."""
        for theme in imageslides.DEFAULT_THEMES:
            self.assertIn("vibe", theme)
            self.assertIn("outfit", theme)
            self.assertIn("location", theme)
            self.assertIn("mood", theme)

    def test_build_prefix_returns_the_chosen_themes_own_vibe(self):
        reference = {"prompt": "closeup portrait, dramatic lighting, studio"}
        niche = {"id": "aibeauty",
                "themes": [{"vibe": "a specific test vibe", "outfit": "wearing a coat",
                           "location": "in a room", "mood": "a calm pose"}]}
        prefix, vibe = imageslides._build_prefix(niche, reference)
        self.assertEqual(vibe, "a specific test vibe")
        self.assertIn("wearing a coat", prefix)
        self.assertIn("in a room", prefix)
        self.assertIn("a calm pose", prefix)

    @staticmethod
    def _fake_decide(query, prompt_filter=None, weights=None):
        return ({"model_id": 4201, "version_id": 130072, "name": "Test Model"},
               {"prompt": "studio portrait, dramatic lighting", "negative_prompt": ""})

    def test_generate_runs_qa_and_returns_only_approved_images(self):
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(10)]
        with mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             mock.patch.object(imageslides.supervisor, "filter_images",
                               lambda paths: paths[:4]), \
             tempfile.TemporaryDirectory() as tmp:
            approved, vibe, _prompts = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertEqual(len(approved), 4)

    def test_short_round_triggers_a_second_round_of_fresh_variations(self):
        """Ten generations, fewer than three pass: try again with a fresh batch of
        variations rather than reusing the failed images, per the run-again-until-3
        spec. Both rounds must actually run, not just the first."""
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(10)]
        filter_results = [[], fake_paths[:4]]

        with mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             mock.patch.object(imageslides.supervisor, "filter_images",
                               lambda paths: filter_results.pop(0)), \
             tempfile.TemporaryDirectory() as tmp:
            approved, vibe, _prompts = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertEqual(len(approved), 4)
        self.assertEqual(filter_results, [], "both rounds must have run")

    def test_broken_checkpoint_triggers_a_fresh_decision_next_round(self):
        """A live run hit this exactly: a gated CivitAI model 401'd on every download
        attempt, and since the checkpoint was decided once and reused for every round,
        all rounds failed identically -- zero images generated, forever. A round that
        generates NOTHING (not 'some generated but rejected', literally none) must
        trigger a fresh decide_reference() call for the next round, not a retry of the
        same broken model."""
        decisions = [
            ({"model_id": 1, "version_id": 1, "name": "Broken"},
             {"prompt": "portrait, broken checkpoint", "negative_prompt": ""}),
            ({"model_id": 2, "version_id": 2, "name": "Working"},
             {"prompt": "portrait, working checkpoint", "negative_prompt": ""}),
        ]

        def fake_decide(niche_arg, state=None):
            return decisions.pop(0)

        def fake_generate_batch(prompts, workdir, civitai_model=None, **kw):
            if civitai_model == "1:1":
                raise RuntimeError("no images were generated")
            return [Path(f"/tmp/img_{i}.png") for i in range(len(prompts))]

        with mock.patch.object(imageslides, "decide_reference", fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            approved, vibe, _prompts = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertGreater(len(approved), 0)
        self.assertEqual(decisions, [], "both decide_reference calls must have happened")

    def test_too_few_approved_images_raises(self):
        """A carousel needs at least min_images; publishing fewer is not worth it, even
        after every retry round is exhausted."""
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(10)]
        with mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: []), \
             tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                imageslides.generate({**self.AIBEAUTY, "min_images": 3}, workdir=tmp)
        self.assertIn("passed review", str(ctx.exception))

    def test_broken_supervisor_falls_back_to_raw_generations(self):
        """A live run wasted a full CI slot generating 20 images that all got
        rejected because the local Ollama couldn't load llama3.2-vision (mllama
        arch not supported by its llama-server engine) -- a supervisor infra bug,
        not a real content signal. When EVERY round's supervisor call was broken
        (no image ever got a real verdict), push the raw generations rather than
        throw a good batch away; the TikTok app's own draft review is still the
        gating human step, so the safety gate isn't fully bypassed."""
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(10)]
        # A FilterResult that unpacks like the old empty-list return but carries
        # the supervisor_broken flag -- exactly the shape supervisor.filter_images
        # returns when every image's issues start with 'gave no usable verdict'.
        broken_result = supervisor.FilterResult([], supervisor_broken=True)

        with mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             mock.patch.object(imageslides.supervisor, "filter_images",
                               lambda paths: broken_result), \
             tempfile.TemporaryDirectory() as tmp:
            approved, vibe, _prompts = imageslides.generate(
                {**self.AIBEAUTY, "min_images": 3, "max_images": 5}, workdir=tmp)
        # The generated images must come back so downstream can push them --
        # never mind that supervisor didn't approve any, because supervisor
        # didn't manage to say anything at all.
        self.assertEqual(len(approved), 5)

    def test_real_reject_from_supervisor_still_raises(self):
        """The fallback above only fires when supervisor was BROKEN. A real
        rejection (supervisor answered but said 'no' -- realistic score too low,
        anatomy_ok false, etc.) must still block the batch; that's the whole
        point of having the supervisor. Guards against a future edit that
        accidentally widens the fallback to include real rejections."""
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(10)]
        real_reject = supervisor.FilterResult([], supervisor_broken=False)

        with mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             mock.patch.object(imageslides.supervisor, "filter_images",
                               lambda paths: real_reject), \
             tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                imageslides.generate({**self.AIBEAUTY, "min_images": 3}, workdir=tmp)
        self.assertIn("passed review", str(ctx.exception))

    def test_supervisor_can_be_disabled_for_local_testing(self):
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(3)]
        with mock.patch.dict(os.environ, {"SUPERVISOR_ENABLED": "0"}), \
             mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             tempfile.TemporaryDirectory() as tmp:
            approved, vibe, _prompts = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertEqual(approved, fake_paths)

    def test_sexy_cue_is_always_present(self):
        """Every image must read as sexy -- not just when the random mood pick happens
        to land on strong wording. SEXY_CUE is the guaranteed baseline."""
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured["prompts"] = prompts
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        with mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            imageslides.generate(self.AIBEAUTY, workdir=tmp)
        for p in captured["prompts"]:
            self.assertIn(imageslides.SEXY_CUE, p)


class FakePipe:
    """Stands in for a diffusers pipeline: callable, returns an object with .images[0]."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kw):
        self.calls.append((prompt, kw))
        image = mock.Mock()
        # sdgen.py saves via image.convert("RGB").save(dest, "JPEG", ...) now (PNG-
        # sourced photo posts failed TikTok's own format check, live-confirmed) --
        # the chained call needs mocking on convert's return value, not image itself.
        image.convert.return_value.save = lambda dest, *a, **k: Path(dest).write_bytes(b"fake")
        result = mock.Mock()
        result.images = [image]
        return result


class SdgenTest(unittest.TestCase):
    def setUp(self):
        # generate_image() now runs refine.refine() and upscale.upscale() on every
        # image -- both download real models and expect a real PIL image / diffusers
        # pipe, neither of which FakePipe's Mock().images[0] provides. Identity stubs
        # keep these tests hermetic; refine.py/upscale.py's own behaviour is covered
        # separately in RefineTest/UpscaleTest below.
        refine_patcher = mock.patch.object(refine, "refine", lambda image, pipe, *a, **kw: image)
        upscale_patcher = mock.patch.object(upscale, "upscale", lambda image, *a, **kw: image)
        refine_patcher.start()
        upscale_patcher.start()
        self.addCleanup(refine_patcher.stop)
        self.addCleanup(upscale_patcher.stop)

    def test_every_preset_has_a_repo_id(self):
        for key, (repo, lora, name) in sdgen.MODELS.items():
            self.assertTrue(repo, key)

    @staticmethod
    def _passthrough_encode(pipe, arch, prompt, negative):
        return {"prompt": prompt, "negative_prompt": negative}

    def test_load_civitai_passes_an_explicit_base_config_matching_arch(self):
        """Live-caught bug: from_single_file() without an explicit config= infers the
        pipeline structure from the checkpoint's own weights, and a real checkpoint
        (genuinely SD1.5, no config override) got misidentified as a ControlNet repo
        (lllyasviel/control_v11p_sd15_canny), which has no model_index.json --
        diffusers raised trying to load one from it. Passing a known-good base config
        repo per architecture sidesteps the guess."""
        import diffusers
        seen = {}

        class FakePipeline:
            @classmethod
            def from_single_file(cls, path, **kw):
                seen.update(kw)
                p = mock.Mock()
                p.scheduler = mock.Mock(config={})
                return p

        with mock.patch.object(sdgen.civitai, "resolve_and_download", lambda spec: {
                "path": "/tmp/fake.safetensors", "arch": "sdxl", "name": "Fake"}), \
             mock.patch.object(diffusers, "StableDiffusionXLPipeline", FakePipeline), \
             mock.patch.object(diffusers, "LCMScheduler",
                               mock.Mock(from_config=lambda c: mock.Mock())), \
             mock.patch.dict(sdgen._pipes, {}, clear=True):
            sdgen._load_civitai("some-spec")
        self.assertEqual(seen["config"], sdgen._BASE_CONFIG["sdxl"])

    def test_civitai_model_takes_precedence_over_preset(self):
        """An operator-named checkpoint always wins over the built-in presets."""
        fake = FakePipe()
        with mock.patch.object(sdgen, "_load_civitai", lambda spec: fake), \
             mock.patch.object(sdgen, "_load",
                               mock.Mock(side_effect=AssertionError("must not use presets"))), \
             mock.patch.dict(sdgen._pipe_arch, {"4201:130072": "sd15"}), \
             mock.patch.object(sdgen, "_encode", self._passthrough_encode), \
             tempfile.TemporaryDirectory() as tmp:
            sdgen.generate_image("a prompt", Path(tmp) / "out.png", civitai_model="4201:130072")
        self.assertEqual(fake.calls[0][0], "a prompt")

    def test_preset_used_when_no_civitai_model_given(self):
        fake = FakePipe()
        with mock.patch.object(sdgen, "_load", lambda key: fake), \
             mock.patch.dict(os.environ, {"CIVITAI_MODEL": ""}), \
             mock.patch.object(sdgen, "CIVITAI_MODEL", ""), \
             mock.patch.dict(sdgen._pipe_arch, {"dreamshaper": "sd15"}), \
             mock.patch.object(sdgen, "_encode", self._passthrough_encode), \
             tempfile.TemporaryDirectory() as tmp:
            sdgen.generate_image("a prompt", Path(tmp) / "out.png", model_key="dreamshaper")
        self.assertEqual(len(fake.calls), 1)

    def test_width_height_steps_guidance_overrides_reach_the_pipeline(self):
        """imageslides.py's _adopted_settings() feeds a harvested checkpoint's own
        posted resolution/steps/cfg_scale through these -- confirm they actually reach
        the pipe() call, not just get accepted and dropped."""
        fake = FakePipe()
        with mock.patch.object(sdgen, "_load", lambda key: fake), \
             mock.patch.dict(os.environ, {"CIVITAI_MODEL": ""}), \
             mock.patch.object(sdgen, "CIVITAI_MODEL", ""), \
             mock.patch.dict(sdgen._pipe_arch, {"dreamshaper": "sd15"}), \
             mock.patch.object(sdgen, "_encode", self._passthrough_encode), \
             tempfile.TemporaryDirectory() as tmp:
            sdgen.generate_image("a prompt", Path(tmp) / "out.png", model_key="dreamshaper",
                                 width=576, height=864, steps=10, guidance=1.0)
        kw = fake.calls[0][1]
        self.assertEqual(kw["width"], 576)
        self.assertEqual(kw["height"], 864)
        self.assertEqual(kw["num_inference_steps"], 10)
        self.assertEqual(kw["guidance_scale"], 1.0)

    def test_overrides_default_to_module_constants_when_not_given(self):
        fake = FakePipe()
        with mock.patch.object(sdgen, "_load", lambda key: fake), \
             mock.patch.dict(os.environ, {"CIVITAI_MODEL": ""}), \
             mock.patch.object(sdgen, "CIVITAI_MODEL", ""), \
             mock.patch.dict(sdgen._pipe_arch, {"dreamshaper": "sd15"}), \
             mock.patch.object(sdgen, "_encode", self._passthrough_encode), \
             tempfile.TemporaryDirectory() as tmp:
            sdgen.generate_image("a prompt", Path(tmp) / "out.png", model_key="dreamshaper")
        kw = fake.calls[0][1]
        self.assertEqual(kw["width"], sdgen.WIDTH)
        self.assertEqual(kw["height"], sdgen.HEIGHT)
        self.assertEqual(kw["num_inference_steps"], sdgen.STEPS)
        self.assertEqual(kw["guidance_scale"], sdgen.GUIDANCE)

    def test_encode_falls_back_to_truncated_prompt_when_compel_is_unavailable(self):
        """compel is an optional dependency -- if it is not installed, generation must
        still work, just with the 77-token CLIP truncation instead of the long-prompt
        workaround."""
        import builtins
        real_import = builtins.__import__

        def blocked_import(name, *a, **kw):
            if name == "compel":
                raise ImportError("no module named compel")
            return real_import(name, *a, **kw)

        with mock.patch.object(builtins, "__import__", blocked_import):
            result = sdgen._encode(FakePipe(), "sd15", "a prompt", "a negative")
        self.assertEqual(result, {"prompt": "a prompt", "negative_prompt": "a negative"})

    # The compel-available path (real chunked embeddings for a >77-token prompt) is
    # proven against the real checkpoint in a live generation run instead of a unit
    # test here -- a synthetic tiny CLIP fixture (tried: hf-internal-testing/tiny-
    # random-clip) pairs a full-size tokenizer with a truncated embedding table and
    # throws IndexError on realistic input, which tests the fixture, not _encode.

    def test_refine_is_called_with_the_actual_generation_settings(self):
        """refine.refine() must see the same steps/guidance/prompt actually used for
        the base image, not module defaults -- otherwise an adopted checkpoint setting
        (imageslides._adopted_settings) would apply to the base render but not the
        touch-up pass."""
        fake = FakePipe()
        seen = {}

        def spy_refine(image, pipe, prompt, negative_prompt, steps, guidance, **kw):
            seen.update(prompt=prompt, negative_prompt=negative_prompt,
                       steps=steps, guidance=guidance, pipe=pipe, arch=kw.get("arch"))
            return image

        with mock.patch.object(sdgen, "_load", lambda key: fake), \
             mock.patch.dict(os.environ, {"CIVITAI_MODEL": ""}), \
             mock.patch.object(sdgen, "CIVITAI_MODEL", ""), \
             mock.patch.dict(sdgen._pipe_arch, {"dreamshaper": "sd15"}), \
             mock.patch.object(sdgen, "_encode", self._passthrough_encode), \
             mock.patch.object(refine, "refine", spy_refine), \
             tempfile.TemporaryDirectory() as tmp:
            sdgen.generate_image("a prompt", Path(tmp) / "out.png", model_key="dreamshaper",
                                 negative_prompt="extra negative", steps=10, guidance=1.0)
        self.assertEqual(seen["steps"], 10)
        self.assertEqual(seen["guidance"], 1.0)
        self.assertEqual(seen["prompt"], "a prompt")
        self.assertIn("extra negative", seen["negative_prompt"])
        # Without this, refine()'s inpaint calls default to "sd15" regardless of the
        # checkpoint actually loaded -- harmless for the common case, wrong for an
        # SDXL checkpoint (clip_encode.encode() builds the wrong tokenizer/text_
        # encoder pairing for the pipe it's actually given).
        self.assertEqual(seen["arch"], "sd15")
        self.assertIs(seen["pipe"], fake)

    def test_upscale_runs_on_the_refined_image_before_saving(self):
        """upscale.upscale() must run on refine.refine()'s output, not the raw base
        render -- it's the last quality step before the file hits disk."""
        fake = FakePipe()
        marker = mock.Mock()
        marker.convert.return_value.save = lambda dest, *a, **k: Path(dest).write_bytes(b"fake")
        seen = {}

        def spy_upscale(image, *a, **kw):
            seen["image"] = image
            return marker

        with mock.patch.object(sdgen, "_load", lambda key: fake), \
             mock.patch.dict(os.environ, {"CIVITAI_MODEL": ""}), \
             mock.patch.object(sdgen, "CIVITAI_MODEL", ""), \
             mock.patch.dict(sdgen._pipe_arch, {"dreamshaper": "sd15"}), \
             mock.patch.object(sdgen, "_encode", self._passthrough_encode), \
             mock.patch.object(refine, "refine", lambda image, pipe, *a, **kw: image), \
             mock.patch.object(upscale, "upscale", spy_upscale), \
             tempfile.TemporaryDirectory() as tmp:
            sdgen.generate_image("a prompt", Path(tmp) / "out.png", model_key="dreamshaper")
        # upscale.upscale()'s return value (marker), not its input, must be what
        # actually gets saved to disk.
        marker.convert.assert_called_once_with("RGB")


class RefineTest(unittest.TestCase):
    """refine.py's own logic: the crop/pad/feather math (pure, no model needed) and
    the detect -> inpaint -> paste-back flow with the detector and inpaint pipe
    mocked out -- real YOLO/diffusers calls are covered by the live verification in
    this session's history (crop-based inpaint on a real generated image, both face
    and hand regions, confirmed a genuine visible improvement with no seam), not
    re-run here since that needs real model weights and a GPU/CPU-minutes budget a
    unit test shouldn't spend."""

    def setUp(self):
        # refine()'s inpaint calls go through clip_encode.encode() now (a live check
        # found them silently truncating at CLIP's 77-token limit otherwise -- see
        # clip_encode.py). The mocked inpaint pipes here have no real tokenizer for
        # it to use, so this stubs it to the same passthrough shape clip_encode.encode
        # itself falls back to when compel isn't installed -- keeps prompt/negative
        # inspectable in tests exactly as before this change.
        patcher = mock.patch.object(
            clip_encode, "encode",
            lambda pipe, arch, prompt, negative: {"prompt": prompt, "negative_prompt": negative})
        patcher.start()
        self.addCleanup(patcher.stop)
        # Hands route through hand_pose.py now (MediaPipe landmarks -> ControlNet
        # skeleton -- see that module's docstring for why). Default stub returns a
        # real control image for any canvas, so hand tests exercise the inpaint path
        # by default; tests of the no-landmarks skip path override this explicitly.
        from PIL import Image
        skeleton_patcher = mock.patch.object(
            hand_pose, "skeleton_for", lambda canvas: Image.new("RGB", canvas.size))
        skeleton_patcher.start()
        self.addCleanup(skeleton_patcher.stop)

    def test_crop_box_pads_and_clamps_to_image_bounds(self):
        # A box already touching the left/top edge must not pad past 0.
        x0, y0, x1, y1 = refine._crop_box((200, 300), [0, 0, 40, 60])
        self.assertEqual((x0, y0), (0, 0))
        self.assertLessEqual(x1, 200)
        self.assertLessEqual(y1, 300)

    def test_crop_box_pads_a_centered_box_on_every_side(self):
        x0, y0, x1, y1 = refine._crop_box((1000, 1000), [400, 400, 500, 500])
        # 30% padding of a 100px box = 30px each side.
        self.assertEqual((x0, y0, x1, y1), (370, 370, 530, 530))

    def test_crop_box_never_returns_a_degenerate_region(self):
        x0, y0, x1, y1 = refine._crop_box((100, 100), [50, 50, 50, 50])
        self.assertGreater(x1 - x0, 0)
        self.assertGreater(y1 - y0, 0)

    def test_canvas_size_preserves_aspect_ratio_not_forced_square(self):
        """Live-suspected real contributor to hands looking worse after refinement,
        not just failing to improve them: a hand crop is usually tall/narrow (e.g.
        102x157, seen in a real run), and the canvas used to always be a forced
        INPAINT_SIZE x INPAINT_SIZE square -- squishing it horizontally before
        inpainting and stretching back after, distorting finger proportions on every
        single hand refine."""
        w, h = refine._canvas_size(100, 155)
        self.assertAlmostEqual(w / h, 100 / 155, delta=0.05)
        self.assertEqual(max(w, h), refine.INPAINT_SIZE)
        self.assertEqual(w % 8, 0)
        self.assertEqual(h % 8, 0)

    def test_canvas_size_stays_square_for_a_square_crop(self):
        w, h = refine._canvas_size(64, 64)
        self.assertEqual(w, h)
        self.assertEqual(w, refine.INPAINT_SIZE)

    def test_feathered_mask_is_bright_in_the_center_and_dim_at_the_edge(self):
        from PIL import Image
        mask = refine._feathered_mask((100, 100))
        self.assertIsInstance(mask, Image.Image)
        self.assertGreater(mask.getpixel((50, 50)), 200)
        self.assertLess(mask.getpixel((0, 0)), 50)

    def test_disabled_returns_the_original_image_untouched(self):
        from PIL import Image
        image = Image.new("RGB", (64, 64))
        with mock.patch.object(refine, "REFINE_ENABLED", False):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8)
        self.assertIs(result, image)

    def test_no_detection_returns_the_original_image_untouched(self):
        from PIL import Image
        image = Image.new("RGB", (64, 64))
        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes", lambda img, kind: []):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8)
        self.assertIs(result, image)

    def test_hand_with_no_landmarks_is_left_untouched_not_guessed_at(self):
        """The core safety property of hand_pose.py: a hand malformed enough that
        MediaPipe can't find any landmarks on it has no recognizable structure to
        guide toward. Live-confirmed this correlates with a hand blind inpainting
        makes WORSE, not better (a real severely-malformed hand turned into a
        formless smudge). Must skip the region entirely and leave supervisor.py's
        QA gate to catch it, not paste an unguided guess over it."""
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))
        inpaint_pipe = mock.Mock(side_effect=AssertionError("must not be called"))
        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([100, 150, 140, 190], 0.9)]), \
             mock.patch.object(hand_pose, "skeleton_for", lambda canvas: None), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: inpaint_pipe):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8,
                                   kinds=("hand",))
        self.assertEqual(result.getpixel((120, 170)), (10, 10, 10), "must be left untouched")
        inpaint_pipe.assert_not_called()

    def test_glitched_hand_output_is_discarded_not_pasted(self):
        """Live-caught: a real production run produced a hand replaced by chaotic
        rainbow/static noise even with a valid pose skeleton and the tuned strength
        (see hand_pose.py's docstring for the elimination process that ruled out the
        skeleton/ControlNet as the cause). Caught on the actual OUTPUT via
        hand_pose._looks_glitched(), not predicted from the input -- must discard
        the result and leave the original region in place, same as no-landmarks."""
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))
        inpaint_pipe = mock.Mock(return_value=mock.Mock(
            images=[Image.new("RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE), color=(200, 50, 50))]))
        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([100, 150, 140, 190], 0.9)]), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: inpaint_pipe), \
             mock.patch.object(hand_pose, "_looks_glitched", lambda img: True):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8,
                                   kinds=("hand",))
        self.assertEqual(result.getpixel((120, 170)), (10, 10, 10),
                         "glitched result must not be pasted")

    def test_hand_landmark_detection_failure_falls_back_to_the_original_image(self):
        """Never raises -- a broken landmark detector (network error downloading the
        model, etc.) must not lose an otherwise-good generation."""
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))

        def boom(canvas):
            raise RuntimeError("model download failed")

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([100, 150, 140, 190], 0.9)]), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: mock.Mock()), \
             mock.patch.object(hand_pose, "skeleton_for", boom):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8,
                                   kinds=("hand",))
        self.assertEqual(result.getpixel((120, 170)), (10, 10, 10))

    def test_detection_failure_falls_back_to_the_original_image(self):
        """Never raises -- a broken detector must not lose an otherwise-good
        generation over a post-process step."""
        from PIL import Image
        image = Image.new("RGB", (64, 64))

        def boom(img, kind):
            raise RuntimeError("model download failed")

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes", boom):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8)
        self.assertIs(result, image)

    def test_a_detected_region_is_inpainted_and_pasted_back_at_full_resolution(self):
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))
        pipe = mock.Mock()

        def fake_inpaint(**kw):
            result = mock.Mock()
            # A distinct color so the paste-back is verifiable below.
            result.images = [Image.new("RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE),
                                       color=(255, 0, 0))]
            return result

        inpaint_pipe = mock.Mock(side_effect=fake_inpaint)
        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([100, 150, 140, 190], 0.9)]), \
             mock.patch.object(refine, "_inpaint_pipe_for", lambda pipe: inpaint_pipe):
            result = refine.refine(image, pipe, "p", "n", steps=6, guidance=1.8, kinds=("face",))
        self.assertEqual(result.size, image.size)
        # Center of the padded box must now show the inpainted color, not the original.
        self.assertEqual(result.getpixel((120, 170)), (255, 0, 0))
        # Far corner, outside the refined region, must be untouched.
        self.assertEqual(result.getpixel((5, 5)), (10, 10, 10))

    def test_inpaint_failure_for_one_kind_does_not_block_the_others(self):
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))

        def fake_inpaint(**kw):
            result = mock.Mock()
            result.images = [Image.new("RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE),
                                       color=(0, 255, 0))]
            return result

        call_count = {"n": 0}

        def flaky_pipe(**kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first pass fails")
            return fake_inpaint(**kw)

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([100, 150, 140, 190], 0.9)]), \
             mock.patch.object(refine, "_inpaint_pipe_for", lambda pipe: flaky_pipe), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: fake_inpaint):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8,
                                   kinds=("face", "hand"))
        # Second kind's inpaint succeeded and is visible; first kind's failure didn't
        # raise or abort the loop.
        self.assertEqual(result.getpixel((120, 170)), (0, 255, 0))

    def test_both_detected_hands_get_refined_when_the_cap_allows_it(self):
        """The mechanism supports refining more than one region per kind (a selfie
        holding a phone, hands on hips, etc. commonly show both hands at once) --
        MAX_REGIONS_PER_KIND itself defaults to 1 for CI time-budget reasons (a real
        GH Actions run got killed by the workflow timeout with it at 2), not because
        the capability is gone. Explicitly opts into 2 here to prove the mechanism
        still works when configured that way."""
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))

        def fake_inpaint(**kw):
            result = mock.Mock()
            result.images = [Image.new("RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE),
                                       color=(0, 0, 255))]
            return result

        inpaint_pipe = mock.Mock(side_effect=fake_inpaint)
        boxes = [([20, 20, 60, 60], 0.9), ([200, 300, 240, 340], 0.8)]
        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "MAX_REGIONS_PER_KIND", 2), \
             mock.patch.object(refine, "_detect_boxes", lambda img, kind: boxes), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: inpaint_pipe):
            result = refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8,
                                   kinds=("hand",))
        self.assertEqual(result.getpixel((40, 40)), (0, 0, 255))
        self.assertEqual(result.getpixel((220, 320)), (0, 0, 255))
        self.assertEqual(inpaint_pipe.call_count, 2)

    def test_regions_beyond_the_per_kind_cap_are_not_refined(self):
        from PIL import Image
        image = Image.new("RGB", (300, 400), color=(10, 10, 10))
        inpaint_pipe = mock.Mock(side_effect=lambda **kw: mock.Mock(
            images=[Image.new("RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE))]))
        boxes = [([x, x, x + 20, x + 20], 0.9 - x / 1000) for x in (10, 60, 110, 160)]
        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes", lambda img, kind: boxes), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: inpaint_pipe):
            refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8, kinds=("hand",))
        self.assertEqual(inpaint_pipe.call_count, refine.MAX_REGIONS_PER_KIND)

    def test_hand_region_gets_a_minimal_anatomy_only_prompt(self):
        """Live-caught bug: a "holding phone" base prompt (true of one hand in the
        shot) bled into the OTHER hand's inpaint and hallucinated a phone-shaped
        object into a hand that was just resting. The base prompt's action/scene
        wording applies to the whole image, not to any one region a local crop
        can't tell it doesn't apply to -- so for hands it must be REPLACED, not
        appended to, unlike faces."""
        from PIL import Image
        image = Image.new("RGB", (300, 400))
        seen = {}

        def fake_inpaint(**kw):
            seen.update(kw)
            return mock.Mock(images=[Image.new(
                "RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE))])

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([20, 20, 60, 60], 0.9)]), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: fake_inpaint):
            refine.refine(image, mock.Mock(), "selfie, holding phone", "base negative",
                         steps=6, guidance=1.8, kinds=("hand",))
        self.assertNotIn("holding phone", seen["prompt"])
        self.assertIn("five fingers", seen["prompt"])
        self.assertIn("base negative", seen["negative_prompt"])
        self.assertIn("fused fingers", seen["negative_prompt"])
        self.assertEqual(seen["strength"], hand_pose.STRENGTH)
        self.assertNotEqual(hand_pose.STRENGTH, refine.STRENGTH_BY_KIND["face"])
        # Effective denoising steps = num_inference_steps * strength -- inheriting the
        # base render's own steps (6) at hand_pose's old 0.6 strength gave ~4
        # effective steps, FEWER than the base render itself used to produce the
        # malformed hand in the first place. Hands get their own, decoupled step
        # count and a ControlNet pose skeleton now (see hand_pose.py).
        self.assertEqual(seen["num_inference_steps"], hand_pose.STEPS)
        self.assertNotEqual(seen["num_inference_steps"], 6)
        # The pose skeleton must actually reach the inpaint call, not just get built
        # and dropped -- this is the real structural-guidance mechanism, not the
        # text prompt above.
        self.assertIn("control_image", seen)
        self.assertIsNotNone(seen["control_image"])

    def test_non_square_hand_crop_gets_a_non_square_canvas(self):
        """Wiring check for the aspect-ratio fix: a real hand box, once padded, is
        clearly non-square, and the canvas actually handed to the inpaint pipe must
        reflect that -- not silently forced back to square somewhere in refine()."""
        from PIL import Image
        image = Image.new("RGB", (300, 400))
        seen = {}

        def fake_inpaint(**kw):
            seen.update(kw)
            w, h = kw["width"], kw["height"]
            return mock.Mock(images=[Image.new("RGB", (w, h))])

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([100, 150, 130, 280], 0.9)]), \
             mock.patch.object(hand_pose, "inpaint_pipe_for", lambda pipe: fake_inpaint):
            refine.refine(image, mock.Mock(), "p", "n", steps=6, guidance=1.8, kinds=("hand",))
        self.assertNotEqual(seen["width"], seen["height"])
        self.assertEqual(max(seen["width"], seen["height"]), refine.INPAINT_SIZE)

    def test_face_region_still_keeps_the_base_prompt(self):
        """Unlike hands, faces have no live-observed hallucination risk from scene
        text, and keeping it helps style/identity consistency with the rest of the
        image -- only hands are replaced, not appended."""
        from PIL import Image
        image = Image.new("RGB", (300, 400))
        seen = {}

        def fake_inpaint(**kw):
            seen.update(kw)
            return mock.Mock(images=[Image.new(
                "RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE))])

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([20, 20, 60, 60], 0.9)]), \
             mock.patch.object(refine, "_inpaint_pipe_for", lambda pipe: fake_inpaint):
            refine.refine(image, mock.Mock(), "base pose prompt", "base negative",
                         steps=6, guidance=1.8, kinds=("face",))
        self.assertIn("base pose prompt", seen["prompt"])
        self.assertIn("sharp focus", seen["prompt"])

    def test_inpaint_prompt_goes_through_clip_encode_with_the_right_arch(self):
        """Live-caught bug: refine()'s inpaint calls used to pass prompt=/
        negative_prompt= as plain strings straight to diffusers, silently truncated
        at CLIP's 77-token limit -- exactly the problem sdgen.py's base render
        already solved via clip_encode.encode()'s compel chunking, just never
        applied here. This confirms the wiring: arch reaches clip_encode.encode(),
        not just that a passthrough dict happens to look right."""
        from PIL import Image
        image = Image.new("RGB", (300, 400))
        seen = {}

        def fake_encode(pipe, arch, prompt, negative):
            seen["arch"] = arch
            seen["prompt"] = prompt
            return {"prompt": prompt, "negative_prompt": negative}

        with mock.patch.object(refine, "REFINE_ENABLED", True), \
             mock.patch.object(refine, "_detect_boxes",
                               lambda img, kind: [([20, 20, 60, 60], 0.9)]), \
             mock.patch.object(refine, "_inpaint_pipe_for", lambda pipe: (
                 lambda **kw: mock.Mock(images=[Image.new(
                     "RGB", (refine.INPAINT_SIZE, refine.INPAINT_SIZE))]))), \
             mock.patch.object(clip_encode, "encode", fake_encode):
            refine.refine(image, mock.Mock(), "base pose prompt", "base negative",
                         steps=6, guidance=1.8, kinds=("face",), arch="sdxl")
        self.assertEqual(seen["arch"], "sdxl")
        self.assertIn("base pose prompt", seen["prompt"])


class HandPoseGlitchDetectionTest(unittest.TestCase):
    """hand_pose._looks_glitched() -- see that module's docstring for the live
    elimination process (garbled skeletons, ControlNet scale, plain inpainting with
    no ControlNet at all, and crop brightness were all tried and ruled out as
    predictors on the input side; this measures the actual output instead). Real
    threshold calibration (4 confirmed-clean canvas outputs at 3.1-4.6, the
    confirmed-bad production-default case at 6.05) lives in the module itself, not
    duplicated here -- these tests cover the function's own logic with synthetic
    images at the extremes, not the exact live threshold value."""

    def test_smooth_uniform_image_does_not_look_glitched(self):
        from PIL import Image
        image = Image.new("RGB", (100, 100), color=(180, 140, 120))
        self.assertFalse(hand_pose._looks_glitched(image))

    def test_smooth_gradient_does_not_look_glitched(self):
        """A real photo has smooth gradients (lighting falloff, skin tone shading),
        not flat color -- must not itself trigger the glitch check."""
        import numpy as np
        from PIL import Image
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for x in range(100):
            arr[:, x] = [120 + x // 2, 90 + x // 3, 80 + x // 4]
        self.assertFalse(hand_pose._looks_glitched(Image.fromarray(arr)))

    def test_chaotic_per_pixel_noise_looks_glitched(self):
        import numpy as np
        from PIL import Image
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)
        self.assertTrue(hand_pose._looks_glitched(Image.fromarray(arr)))

    def test_threshold_is_env_overridable(self):
        from PIL import Image
        image = Image.new("RGB", (100, 100), color=(180, 140, 120))
        with mock.patch.object(hand_pose, "GLITCH_THRESHOLD", -1.0):
            self.assertTrue(hand_pose._looks_glitched(image),
                           "a threshold below any real image's noise must always trigger")


class UpscaleTest(unittest.TestCase):
    """upscale.py's own logic, with the actual super-resolution model mocked out --
    real model behaviour (a 2x upscale of a real generated image producing a visibly
    sharper, artifact-free result in 3.3s on CPU) was verified live in this session's
    history, not re-run here since it needs the real ~/.cache-downloaded weights."""

    def test_disabled_returns_the_original_image_untouched(self):
        from PIL import Image
        image = Image.new("RGB", (32, 32))
        with mock.patch.object(upscale, "UPSCALE_ENABLED", False):
            result = upscale.upscale(image)
        self.assertIs(result, image)

    def test_success_returns_an_image_of_the_model_output_size(self):
        # upscale.upscale imports `super_image.ImageLoader` at call time; skip
        # rather than fail when the CI-only package isn't available locally.
        # The failure path (super-image missing -> fall back to original image)
        # is already covered by test_failure_falls_back_to_the_original_image
        # below, so we're not losing coverage here.
        try:
            import super_image  # noqa: F401
        except ImportError:
            self.skipTest("super_image not installed (CI-only dependency)")
        import torch
        from PIL import Image
        image = Image.new("RGB", (10, 10), color=(50, 100, 150))
        fake_model = mock.Mock(return_value=torch.rand(1, 3, 20, 20))
        with mock.patch.object(upscale, "UPSCALE_ENABLED", True), \
             mock.patch.object(upscale, "_model", lambda scale: fake_model):
            result = upscale.upscale(image, scale=2)
        self.assertEqual(result.size, (20, 20))
        self.assertEqual(result.mode, "RGB")

    def test_failure_falls_back_to_the_original_image(self):
        """Never raises -- a broken/unreachable model must not lose an otherwise-good
        generation over a resolution bump."""
        from PIL import Image
        image = Image.new("RGB", (10, 10))

        def boom(scale):
            raise RuntimeError("model download failed")

        with mock.patch.object(upscale, "UPSCALE_ENABLED", True), \
             mock.patch.object(upscale, "_model", boom):
            result = upscale.upscale(image)
        self.assertIs(result, image)


class CivitaiPromptCleanupTest(unittest.TestCase):
    """Real harvested prompts carry Automatic1111/ComfyUI syntax diffusers cannot parse
    -- <lora:name:weight> and similar -- which becomes literal junk tokens if left in."""

    def test_lora_tag_is_stripped(self):
        cleaned = civitai._clean_prompt_text(
            "city street, neon, fog, <lora:epiNoiseoffset_v2-pynoise:-1>")
        self.assertEqual(cleaned, "city street, neon, fog")

    def test_multiple_tags_and_stray_commas_are_cleaned(self):
        cleaned = civitai._clean_prompt_text(
            "portrait, <lyco:foo:0.8>, <embedding:bar>, dramatic light")
        self.assertEqual(cleaned, "portrait, dramatic light")

    def test_plain_prompt_is_untouched(self):
        self.assertEqual(civitai._clean_prompt_text("a woman in a coat"), "a woman in a coat")

    def test_usable_applies_the_cleanup(self):
        result = civitai._usable({"prompt": "woman, <lora:x:1> in a coat"})
        self.assertEqual(result["prompt"], "woman, in a coat")


class CivitaiHarvestSafetyTest(unittest.TestCase):
    """Real scraped prompts named explicit ages ("18 y.o", "23 y.o") alongside adult
    ones ("26 y.o", "30 y.o") in a live sample -- this is the filter that caught it."""

    def test_explicit_low_age_is_rejected(self):
        for bad in ("RAW photo, 18 y.o woman in dress", "closeup, 19yo girl portrait",
                   "photo of a 16 years old student", "20 y.o. woman smiling"):
            self.assertIsNone(civitai._usable({"prompt": bad}), bad)

    def test_age_within_target_range_is_kept(self):
        for ok in ("RAW photo, 26 y.o woman in dress", "30 y.o european man",
                  "professional photo, 35 years old woman"):
            self.assertIsNotNone(civitai._usable({"prompt": ok}), ok)

    def test_age_above_target_range_is_rejected(self):
        """25-35 is the target range for this niche -- an explicit older age (e.g.
        '45 year old woman') used to pass through untouched, only ever checked
        against the lower floor."""
        for old in ("professional photo, 45 years old woman", "50yo woman portrait",
                   "60 y.o. woman smiling"):
            self.assertIsNone(civitai._usable({"prompt": old}), old)

    def test_no_age_mentioned_is_kept(self):
        self.assertIsNotNone(civitai._usable({"prompt": "woman walking in a city street"}))

    def test_bad_prompt_terms_still_reject_regardless_of_age(self):
        self.assertIsNone(civitai._usable({"prompt": "26 y.o woman, schoolgirl outfit"}))

    def test_celebrity_faceswap_wildcard_syntax_is_rejected(self):
        """A live search hit a real prompt: 'portrait of [Dakota Johnson|Maggie
        Lindemann] as [Hailey Clauson|Hailey Grice] as a real life version of ((Queen
        Elsa))' -- [Name|Name] is Automatic1111 wildcard-alternation syntax used
        specifically for celebrity/character face-swaps, and must be rejected at the
        source, not left to a negative prompt to (unreliably) override."""
        self.assertIsNone(civitai._usable(
            {"prompt": "portrait of [Dakota Johnson|Maggie Lindemann], studio light"}))
        self.assertIsNone(civitai._usable(
            {"prompt": "woman as a real life version of Elsa, fantasy background"}))
        self.assertIsNone(civitai._usable(
            {"prompt": "deepfake portrait, cinematic lighting"}))

    def test_explicit_sexual_act_terms_are_rejected(self):
        """A live search's showcase prompt for a qualifying model contained explicit
        hardcore sexual-act terms, unfiltered by anything upstream -- search_models'
        nsfw=false only filters the model list, not a version's own showcase images.
        This niche wants sexy, not explicit."""
        self.assertIsNone(civitai._usable({"prompt": "photo of a woman, cumshot, explicit"}))
        self.assertIsNone(civitai._usable({"prompt": "hentai style, anime girl"}))

    def test_chinese_appearance_prompts_are_rejected(self):
        """Operator preference: exclude Chinese-appearance prompts specifically, not a
        wider ethnicity exclusion."""
        self.assertIsNone(civitai._usable({"prompt": "portrait of a chinese woman, studio"}))
        self.assertIsNone(civitai._usable({"prompt": "Chinese model, cinematic lighting"}))

    def test_other_ethnicities_are_not_rejected(self):
        """The exclusion is scoped to what was actually asked for -- it must not
        silently sweep up unrelated prompts."""
        self.assertIsNotNone(civitai._usable({"prompt": "portrait of a korean woman, studio"}))
        self.assertIsNotNone(civitai._usable({"prompt": "european woman, studio light"}))

    def test_ordinary_bracket_use_is_not_flagged(self):
        """Only the |-alternation wildcard form is a celebrity-swap signal -- plain
        brackets used for emphasis/weighting must not be rejected."""
        self.assertIsNotNone(civitai._usable(
            {"prompt": "woman in a coat [detailed background], studio light"}))

    def test_harvest_from_model_uses_the_versions_own_showcase_images(self):
        info = {"images": [
            {"meta": {"prompt": "adult woman in a coat"}, "stats": {"likeCount": 10}},
            {"meta": {"prompt": "18 y.o woman"}, "stats": {"likeCount": 999}},  # rejected
            {"meta": {}, "stats": {}},  # no prompt shared
        ]}
        with mock.patch.object(civitai, "_version_info", lambda m, v: info):
            out = civitai.harvest_from_model(4201, 130072)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["prompt"], "adult woman in a coat")

    def test_small_showcase_pool_tops_up_from_gallery(self):
        with mock.patch.object(civitai, "harvest_from_model", lambda m, v: [
                {"prompt": "p1", "negative_prompt": "", "reactions": 1}]), \
             mock.patch.object(civitai, "harvest_from_gallery", lambda **kw: [
                {"prompt": "p2", "negative_prompt": "", "reactions": 1}]):
            out = civitai.safe_harvest(model_id=4201, model_version_id=130072)
        self.assertEqual(len(out), 2)

    def test_healthy_showcase_pool_skips_the_gallery(self):
        with mock.patch.object(civitai, "harvest_from_model", lambda m, v: [
                {"prompt": f"p{i}", "negative_prompt": "", "reactions": 1} for i in range(5)]), \
             mock.patch.object(civitai, "harvest_from_gallery",
                               mock.Mock(side_effect=AssertionError("must not be called"))):
            out = civitai.safe_harvest(model_id=4201, model_version_id=130072)
        self.assertEqual(len(out), 5)


class ImageSlidesSubjectFilterTest(unittest.TestCase):
    """SAFETY_PREFIX always asserts a female adult subject; a harvested prompt describing
    only a man would contradict it in the same prompt."""

    def test_male_only_prompt_is_dropped(self):
        self.assertFalse(imageslides._matches_subject(
            "closeup portrait of a man in a black suit", {"id": "aibeauty"}))

    def test_female_subject_prompt_is_kept(self):
        self.assertTrue(imageslides._matches_subject(
            "closeup portrait of a woman in a black dress", {"id": "aibeauty"}))

    def test_gender_neutral_portrait_cue_is_kept(self):
        self.assertTrue(imageslides._matches_subject(
            "closeup portrait, editorial fashion photo", {"id": "aibeauty"}))

    def test_filter_can_be_disabled_per_niche(self):
        self.assertTrue(imageslides._matches_subject(
            "portrait of a man", {"id": "x", "subject_gender": "any"}))

    def test_creature_or_character_merge_prompt_is_dropped(self):
        """A live search landed on a 'merge' checkpoint whose showcase prompt was 'a
        humanoid boar Electrode hybrid creature ... portrait photo' -- has 'portrait'
        and no male-only term, so the gender check alone would have let it through."""
        self.assertFalse(imageslides._matches_subject(
            "a humanoid boar Electrode hybrid creature, highly detailed portrait photo",
            {"id": "aibeauty"}))

    def test_3d_render_style_prompt_is_dropped(self):
        """A live search decided on 'NLIGHT Realistic's own showcase prompt: 'high
        quality,8K,a girl,blender,3d model,...FASHION SHOOT...' -- explicitly asking
        for a 3D render directly contradicts NEGATIVE_QUALITY's own '3d render' term,
        positive and negative prompt fighting each other in the same call."""
        self.assertFalse(imageslides._matches_subject(
            "high quality, a girl, blender, 3d model, fashion shoot, portrait photo",
            {"id": "aibeauty"}))

    def test_full_body_prompt_without_the_word_portrait_is_kept(self):
        """None of CAMERA_MODIFIERS ever asked for full-body framing, and this regex
        didn't recognize "full body"/"standing"/"full length" as valid subject
        material either -- together the whole pipeline skewed portrait/closeup no
        matter which reference prompt got picked."""
        for ok in ("full body shot of a woman on a beach", "full length photo, standing",
                  "standing pose, natural light"):
            self.assertTrue(imageslides._matches_subject(ok, {"id": "aibeauty"}), ok)

    def test_prompt_with_no_person_at_all_is_dropped(self):
        """A live sample harvested a pure scenery prompt with no person in it --
        SAFETY_PREFIX would glue a subject onto it and render a landscape shot with an
        incidental figure, not a portrait."""
        self.assertFalse(imageslides._matches_subject(
            "autumn landscape, dramatic lighting, gloomy, cloudy weather",
            {"id": "aibeauty"}))


class OutfitConflictTest(unittest.TestCase):
    """Forcing an outfit onto a decided reference prompt that already names one gave
    the model two contradictory outfits in a single prompt -- e.g. jeans-and-coat
    glued onto a prompt that already said 'wearing a black dress'."""

    AIBEAUTY = {"id": "aibeauty",
               "themes": [{"vibe": "a night out", "outfit": "wearing a red gown",
                          "location": "in a room", "mood": "a confident pose"}],
               "min_images": 1, "images_per_video": 2}

    def _prompts_for_reference(self, prompt_text):
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured["prompts"] = prompts
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        with mock.patch.object(civitai, "decide_reference", lambda q, prompt_filter=None, weights=None: (
                {"model_id": 1, "version_id": 2, "name": "X"},
                {"prompt": prompt_text, "negative_prompt": ""})), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            imageslides.generate(self.AIBEAUTY, workdir=tmp)
        return captured["prompts"]

    def test_outfit_not_duplicated_when_source_already_names_one(self):
        prompts = self._prompts_for_reference(
            "portrait photo, wearing a black dress, studio light")
        self.assertIn("black dress", prompts[0])
        self.assertNotIn("red gown", prompts[0], "must not inject a second, conflicting outfit")

    def test_outfit_is_injected_when_source_has_none(self):
        prompts = self._prompts_for_reference("closeup portrait, dramatic lighting, studio")
        self.assertIn("red gown", prompts[0])

    def test_mood_is_appended_regardless_of_source(self):
        """Unlike outfit, mood/pose is appended unconditionally -- it never conflicts
        with what the reference prompt already says, and it is what keeps the tone
        consistent even when the decided reference is mundane (a live search once
        decided on a showcase prompt describing a chef in a kitchen)."""
        niche = {**self.AIBEAUTY,
                "themes": [{"vibe": "a night out", "outfit": "wearing a red gown",
                           "location": "in a room", "mood": "sultry confident gaze"}]}
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured["prompts"] = prompts
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        with mock.patch.object(civitai, "decide_reference", lambda q, prompt_filter=None, weights=None: (
                {"model_id": 1, "version_id": 2, "name": "X"},
                {"prompt": "chef in a kitchen, studio light", "negative_prompt": ""})), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            imageslides.generate(niche, workdir=tmp)
        self.assertIn("sultry confident gaze", captured["prompts"][0])


class LocationConflictTest(unittest.TestCase):
    """Same failure mode as OutfitConflictTest, for setting/location instead of
    clothing: injecting 'poolside cabana' onto a reference that already says 'in a
    bustling gourmet kitchen' gives the model two contradictory settings at once."""

    AIBEAUTY = {"id": "aibeauty",
               "themes": [{"vibe": "a day on the water", "outfit": "wearing a swimsuit",
                          "location": "on a private yacht deck", "mood": "a confident pose"}],
               "min_images": 1, "images_per_video": 2}

    def _prompts_for_reference(self, prompt_text):
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured["prompts"] = prompts
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        with mock.patch.object(civitai, "decide_reference", lambda q, prompt_filter=None, weights=None: (
                {"model_id": 1, "version_id": 2, "name": "X"},
                {"prompt": prompt_text, "negative_prompt": ""})), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            imageslides.generate(self.AIBEAUTY, workdir=tmp)
        return captured["prompts"]

    def test_location_not_duplicated_when_source_already_names_one(self):
        prompts = self._prompts_for_reference(
            "portrait photo in a bustling gourmet kitchen, studio light")
        self.assertIn("kitchen", prompts[0])
        self.assertNotIn("yacht deck", prompts[0],
                         "must not inject a second, conflicting location")

    def test_location_is_injected_when_source_has_none(self):
        prompts = self._prompts_for_reference("closeup portrait, dramatic lighting, studio")
        self.assertIn("yacht deck", prompts[0])


class ModelPreferenceTest(unittest.TestCase):
    """A checkpoint's QA pass-rate history, kept in posted.json's model_stats, nudges
    future selection odds both up (good track record) and down (bad one) from the
    neutral 1.0 baseline an untested checkpoint gets. Not a hard switch: even a
    checkpoint with a 0% pass rate keeps a small positive weight (civitai.py's
    random.choices() needs one to ever reconsider it), not zero -- odds, not a ban
    list. The floor used to be the same 1.0 baseline as untested (MIN_WEIGHT_
    MULTIPLIER didn't exist, effectively 1.0), which meant a proven-bad checkpoint
    was NEVER actually less likely than a brand-new one -- caught after a real batch
    came out consistently biased from a checkpoint whose pass rate should have
    tanked its odds and didn't."""

    def test_no_state_means_no_preference(self):
        self.assertEqual(imageslides._model_weights(None), {})
        self.assertEqual(imageslides._model_weights({}), {})

    def test_below_sample_floor_is_ignored(self):
        """One lucky or unlucky early batch must not permanently tilt selection."""
        state = {"model_stats": {"1:1": {"name": "X", "used": 2, "passed": 2}}}
        self.assertEqual(imageslides._model_weights(state), {})

    def test_good_track_record_weighs_above_baseline(self):
        state = {"model_stats": {"1:1": {"name": "X", "used": 10, "passed": 9}}}
        weights = imageslides._model_weights(state)
        self.assertGreater(weights[1], 1.0)
        self.assertLessEqual(weights[1], imageslides.MAX_WEIGHT_MULTIPLIER)

    def test_poor_track_record_weighs_below_baseline(self):
        """A checkpoint that never passes must become LESS likely to be picked again
        than an untested one, not just no-more-likely -- this is what actually lets a
        bad checkpoint's odds tank over a few batches without anyone needing to have
        named it in advance (see supervisor.py's ethnicity_excluded, which exists
        specifically to feed this)."""
        state = {"model_stats": {"1:1": {"name": "X", "used": 10, "passed": 0}}}
        weights = imageslides._model_weights(state)
        self.assertEqual(weights[1], imageslides.MIN_WEIGHT_MULTIPLIER)
        self.assertLess(weights[1], 1.0)
        self.assertGreater(weights[1], 0)

    def test_record_model_result_accumulates_across_calls(self):
        state = {}
        imageslides._record_model_result(state, "1:1", "X", generated=6, approved=4)
        imageslides._record_model_result(state, "1:1", "X", generated=6, approved=5)
        entry = state["model_stats"]["1:1"]
        self.assertEqual(entry["used"], 12)
        self.assertEqual(entry["passed"], 9)

    def test_record_model_result_does_nothing_without_state(self):
        imageslides._record_model_result(None, "1:1", "X", generated=6, approved=4)  # must not raise


class CivitaiWeightedSelectionTest(unittest.TestCase):
    """civitai.decide_reference()'s own end of the preference loop: it doesn't know
    what "good" means, only turns whatever weights it's handed into biased odds."""

    def _candidate(self, id_, name, downloads=5000):
        files = [{"downloadUrl": "u", "name": f"{name}.safetensors",
                 "metadata": {"format": "SafeTensor"},
                 "pickleScanResult": "Success", "virusScanResult": "Success", "primary": True}]
        return {"id": id_, "name": name, "stats": {"downloadCount": downloads},
               "modelVersions": [{"id": id_ * 100, "baseModel": "SD 1.5", "files": files}]}

    def test_no_weights_behaves_like_a_plain_shuffle(self):
        """Every qualifying candidate should still eventually get picked with no
        weights given -- unchanged behaviour from before this feature existed."""
        items = [self._candidate(1, "A"), self._candidate(2, "B"), self._candidate(3, "C")]

        def fake_harvest(model_id, version_id):
            return [{"prompt": "portrait of a woman", "negative_prompt": "", "reactions": 1}]

        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", fake_harvest):
            seen = {civitai.decide_reference("q")[0]["name"] for _ in range(40)}
        self.assertEqual(seen, {"A", "B", "C"})

    def test_heavily_weighted_candidate_is_picked_far_more_often(self):
        items = [self._candidate(1, "Preferred"), self._candidate(2, "Other")]

        def fake_harvest(model_id, version_id):
            return [{"prompt": "portrait of a woman", "negative_prompt": "", "reactions": 1}]

        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", fake_harvest):
            picks = [civitai.decide_reference("q", weights={1: 20.0, 2: 1.0})[0]["name"]
                    for _ in range(30)]
        self.assertGreater(picks.count("Preferred"), picks.count("Other"))

    def test_unresolvable_weighted_candidate_still_falls_through(self):
        """A heavily-preferred candidate that turns out broken (bad file, no usable
        prompt) must not block the fallback to whatever else qualifies."""
        items = [self._candidate(1, "BrokenButPreferred"), self._candidate(2, "Working")]

        def fake_harvest(model_id, version_id):
            if model_id == 1:
                return []  # nothing usable -- disqualified before weighting even matters
            return [{"prompt": "portrait of a woman", "negative_prompt": "", "reactions": 1}]

        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", fake_harvest):
            resolved, _ = civitai.decide_reference("q", weights={1: 100.0})
        self.assertEqual(resolved["name"], "Working")


class AdoptedSettingsTest(unittest.TestCase):
    """A live check on a real checkpoint's showcase metadata found real, portable
    generation settings we were ignoring entirely -- BEAUTY_BY_STABLE_YOGI's own
    posted example used sampler=LCM, steps=10, cfgScale=1, Size=576x864, which is
    directly compatible with our own fused-LCM pipeline (unlike a typical posted
    25-40-step DPM++ example, which is NOT portable without also switching schedulers)."""

    def test_resolution_adopted_when_in_range(self):
        settings = imageslides._adopted_settings(
            {"width": 576, "height": 864, "sampler": "", "steps": None, "cfg_scale": None})
        self.assertEqual(settings, {"width": 576, "height": 864})

    def test_steps_and_cfg_adopted_only_when_samplers_own_family_is_lcm(self):
        settings = imageslides._adopted_settings(
            {"width": None, "height": None, "sampler": "LCM", "steps": 10, "cfg_scale": 1})
        self.assertEqual(settings, {"steps": 10, "guidance": 1.0})

    def test_steps_and_cfg_ignored_when_sampler_is_not_lcm(self):
        """A typical 30-step DPM++/Euler posted example is not portable to our
        fused-LCM pipeline -- copying its step count without switching schedulers
        would not reproduce their result, just run needlessly slow."""
        settings = imageslides._adopted_settings(
            {"width": None, "height": None, "sampler": "DPM++ 2M Karras",
            "steps": 30, "cfg_scale": 7})
        self.assertEqual(settings, {})

    def test_out_of_range_values_are_not_adopted(self):
        """A bad data point (typo, outlier) must not be allowed to blow the CI time
        budget across the whole batch that shares this one reference."""
        settings = imageslides._adopted_settings(
            {"width": 2048, "height": 2048, "sampler": "LCM", "steps": 50, "cfg_scale": 9})
        self.assertEqual(settings, {})

    def test_missing_values_adopt_nothing(self):
        settings = imageslides._adopted_settings(
            {"width": None, "height": None, "sampler": "", "steps": None, "cfg_scale": None})
        self.assertEqual(settings, {})

    def test_size_not_divisible_by_8_is_rounded_not_used_raw(self):
        """A real run crashed every image in a round on this exact gap: a checkpoint's
        own posted width (513) passed the min/max range check but isn't a multiple of
        8, and diffusers raises rather than rounding -- 'height and width have to be
        divisible by 8 but are 768 and 513'."""
        settings = imageslides._adopted_settings(
            {"width": 513, "height": 768, "sampler": "", "steps": None, "cfg_scale": None})
        self.assertEqual(settings["width"] % 8, 0)
        self.assertEqual(settings["height"] % 8, 0)
        # 513 rounds to 512, the nearest multiple of 8 -- still much closer to what
        # the creator posted than falling back to our own fixed default.
        self.assertEqual(settings["width"], 512)
        self.assertEqual(settings["height"], 768)

    def test_rounding_never_pushes_size_outside_the_trusted_range(self):
        settings = imageslides._adopted_settings(
            {"width": 385, "height": 895, "sampler": "", "steps": None, "cfg_scale": None})
        self.assertGreaterEqual(settings["width"], imageslides._SIZE_MIN)
        self.assertLessEqual(settings["height"], imageslides._SIZE_MAX)

    def test_adopted_settings_reach_generate_batch(self):
        """End-to-end: whatever _adopted_settings() returns for the decided reference
        must actually reach sdgen.generate_batch(), not just be computed and dropped."""
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured.update(kw)
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        def fake_decide(q, prompt_filter=None, weights=None):
            return ({"model_id": 1, "version_id": 2, "name": "X"},
                   {"prompt": "portrait, studio light", "negative_prompt": "",
                    "width": 576, "height": 864, "sampler": "LCM",
                    "steps": 10, "cfg_scale": 1})

        niche = {"id": "aibeauty", "min_images": 1, "images_per_video": 2}
        with mock.patch.object(civitai, "decide_reference", fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            imageslides.generate(niche, workdir=tmp)
        self.assertEqual(captured.get("width"), 576)
        self.assertEqual(captured.get("height"), 864)
        self.assertEqual(captured.get("steps"), 10)
        self.assertEqual(captured.get("guidance"), 1.0)


class CivitaiSearchTest(unittest.TestCase):
    """civitai_model accepts free text -- "realistic portrait woman" -- the same way
    civitai_query already works for prompts: search, then pick a real, usable result."""

    def _items(self, *entries):
        """entries: (name, downloads, base_model, safetensor_ok)"""
        out = []
        for i, (name, downloads, base_model, ok) in enumerate(entries):
            files = [{"downloadUrl": f"u{i}", "name": f"{name}.safetensors",
                      "metadata": {"format": "SafeTensor" if ok else "PickleTensor"},
                      "pickleScanResult": "Success", "virusScanResult": "Success",
                      "primary": True}]
            out.append({
                "id": i, "name": name, "stats": {"downloadCount": downloads},
                "modelVersions": [{"id": 100 + i, "baseModel": base_model, "files": files}],
            })
        return out

    def test_free_text_spec_triggers_search(self):
        search = mock.Mock(return_value={"name": "found via search"})
        with mock.patch.object(civitai, "resolve_from_search", search):
            result = civitai.resolve("realistic portrait woman")
        search.assert_called_once_with("realistic portrait woman")
        self.assertEqual(result["name"], "found via search")

    def test_numeric_spec_does_not_trigger_search(self):
        with mock.patch.object(civitai, "_version_info", lambda m, v: {
                "id": 1, "baseModel": "SD 1.5", "model": {"name": "T"}, "files": [
                    {"downloadUrl": "u", "name": "t.safetensors", "primary": True,
                     "metadata": {"format": "SafeTensor"},
                     "pickleScanResult": "Success", "virusScanResult": "Success"}]}), \
             mock.patch.object(civitai, "resolve_from_search",
                               mock.Mock(side_effect=AssertionError("must not search"))):
            civitai.resolve("4201")

    def test_picks_the_most_downloaded_qualifying_result(self):
        items = self._items(
            ("TooFewDownloads", 10, "SD 1.5", True),      # below the download floor
            ("WrongArch", 50_000, "SD 2.0", True),         # unsupported architecture
            ("PickleOnly", 40_000, "SD 1.5", False),       # no usable SafeTensor
            ("GoodOne", 30_000, "SD 1.5", True),           # first real candidate
        )
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000):
            result = civitai.resolve_from_search("portrait woman")
        self.assertEqual(result["name"], "GoodOne")

    def test_no_qualifying_result_raises_with_reasons(self):
        items = self._items(("TooSmall", 5, "SD 1.5", True))
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000):
            with self.assertRaises(RuntimeError) as ctx:
                civitai.resolve_from_search("obscure query")
        self.assertIn("no usable checkpoint", str(ctx.exception))

    def test_empty_search_results_raise_cleanly(self):
        with mock.patch.object(civitai, "search_models", lambda q, **kw: []):
            with self.assertRaises(RuntimeError):
                civitai.resolve_from_search("nothing matches this")

    def test_ethnicity_biased_checkpoint_names_are_skipped(self):
        """Live-observed: this niche's own queries surfaced "SEX Sexy Eastern
        Experience v3 || Realistic Asian" as a legitimate high-download candidate,
        and a batch generated from it came out consistently Chinese/Asian-appearing
        -- the prompt-text filter (_ETHNICITY_EXCLUDE_RE) cannot catch this since the
        bias lives in the checkpoint itself, not the prompt. "eastern" alone is
        intentionally NOT matched -- it would also reject Eastern European-style
        checkpoints, which are fine."""
        items = self._items(
            ("SEX Sexy Eastern Experience v3 || Realistic Asian", 50_000, "SD 1.5", True),
            ("Eastern European Realism", 40_000, "SD 1.5", True),
            ("GoodOne", 30_000, "SD 1.5", True),
        )
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000):
            result = civitai.resolve_from_search("sexy realistic woman")
        self.assertEqual(result["name"], "Eastern European Realism")

    def test_cjk_characters_in_checkpoint_name_are_skipped(self):
        items = self._items(
            ("ppkkmoon_Daemon_Realm_V2 [ppkkmoon_魔域之墮姫_model_V2]", 50_000, "SD 1.5", True),
            ("GoodOne", 30_000, "SD 1.5", True),
        )
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000):
            result = civitai.resolve_from_search("realistic beauty model")
        self.assertEqual(result["name"], "GoodOne")


class CivitaiDecideReferenceTest(unittest.TestCase):
    """Step 1/2 of the "search for a good photo, decide on it, download its model"
    flow: evaluate the search pool, then pick RANDOMLY among whichever candidates
    both resolve to a downloadable checkpoint and have a real showcase prompt worth
    using -- not always the first (see test_picks_randomly below for why: a fixed
    query resolved deterministically made every run land on the same model and the
    same prompt, confirmed live on a real search).

    The search list view only flags hasMeta: true on each image -- the actual prompt
    text needs a direct model-version lookup, i.e. harvest_from_model() -- so that is
    mocked here per candidate rather than embedding images in the search fixture."""

    def _candidate(self, id_, name, downloads, base_model, ok_file=True):
        files = [{"downloadUrl": "u", "name": f"{name}.safetensors",
                 "metadata": {"format": "SafeTensor" if ok_file else "PickleTensor"},
                 "pickleScanResult": "Success", "virusScanResult": "Success", "primary": True}]
        return {"id": id_, "name": name, "stats": {"downloadCount": downloads},
               "modelVersions": [{"id": id_ * 100, "baseModel": base_model, "files": files}]}

    def _harvest_by_model_id(self, prompts_by_model_id):
        def fake(model_id, version_id):
            prompt = prompts_by_model_id.get(model_id)
            return [{"prompt": prompt, "negative_prompt": "", "reactions": 1}] if prompt else []
        return fake

    def test_picks_the_only_qualifying_candidate(self):
        items = [
            self._candidate(1, "NoShowcase", 5000, "SD 1.5"),
            self._candidate(2, "GoodOne", 4000, "SD 1.5"),
        ]
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model",
                               self._harvest_by_model_id({2: "portrait of a woman"})):
            resolved, prompt = civitai.decide_reference("portrait woman")
        self.assertEqual(resolved["name"], "GoodOne")
        self.assertEqual(prompt["prompt"], "portrait of a woman")

    def test_picks_randomly_among_qualifying_candidates(self):
        """Earlier behaviour always returned the first qualifying candidate -- since
        search results are sorted deterministically, that means the exact same model
        every run for a fixed query, confirmed live (a run kept landing on the same
        checkpoint's "chef in a kitchen" showcase image). Given several qualifying
        candidates, results must vary across repeated calls."""
        items = [self._candidate(i, f"Model{i}", 5000 - i, "SD 1.5") for i in range(1, 6)]
        prompts_by_id = {i: f"portrait of a woman, look {i}" for i in range(1, 6)}
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model",
                               self._harvest_by_model_id(prompts_by_id)):
            seen = {civitai.decide_reference("portrait woman")[0]["name"] for _ in range(40)}
        self.assertGreater(len(seen), 1, "must not always pick the same candidate")

    def test_picks_randomly_among_a_candidates_top_prompts(self):
        """Same failure mode within one model: always taking prompts[0] means the same
        single showcase photo every run even when the checkpoint has several usable ones."""
        def fake_harvest(model_id, version_id):
            return [{"prompt": f"portrait of a woman, variant {i}", "negative_prompt": "",
                     "reactions": 10 - i} for i in range(5)]

        items = [self._candidate(1, "OneModel", 5000, "SD 1.5")]
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", fake_harvest):
            seen = {civitai.decide_reference("portrait woman")[1]["prompt"] for _ in range(40)}
        self.assertGreater(len(seen), 1, "must not always pick the same showcase prompt")

    def test_prompt_filter_skips_candidates_that_fail_it(self):
        items = [
            self._candidate(1, "WrongSubject", 5000, "SD 1.5"),
            self._candidate(2, "RightSubject", 4000, "SD 1.5"),
        ]
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", self._harvest_by_model_id(
                 {1: "portrait of a man", 2: "portrait of a woman"})):
            resolved, prompt = civitai.decide_reference(
                "portrait woman", prompt_filter=lambda p: "woman" in p)
        self.assertEqual(resolved["name"], "RightSubject")

    def test_ethnicity_biased_checkpoint_is_never_offered_as_a_candidate(self):
        """Same check as resolve_from_search's (search_candidates is decide_reference's
        own candidate source) -- must be skipped before harvest_from_model is ever
        called on it, not just excluded after the fact."""
        items = [
            self._candidate(1, "SEX Sexy Eastern Experience v3 || Realistic Asian",
                           50_000, "SD 1.5"),
            self._candidate(2, "GoodOne", 4000, "SD 1.5"),
        ]
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model",
                               self._harvest_by_model_id(
                                   {1: "should never be looked at",
                                    2: "portrait of a woman"})):
            resolved, prompt = civitai.decide_reference("sexy realistic woman")
        self.assertEqual(resolved["name"], "GoodOne")

    def test_unresolvable_candidate_is_skipped_in_favour_of_the_next(self):
        """A candidate can have a good showcase prompt but no scan-clean SafeTensor --
        that must not stop the search, the next candidate should still be tried."""
        items = [
            self._candidate(1, "PickleOnly", 5000, "SD 1.5", ok_file=False),
            self._candidate(2, "GoodOne", 4000, "SD 1.5"),
        ]
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", self._harvest_by_model_id(
                 {1: "portrait of a woman", 2: "portrait of a woman"})):
            resolved, _ = civitai.decide_reference("portrait woman")
        self.assertEqual(resolved["name"], "GoodOne")

    def test_raises_with_reasons_when_nothing_qualifies(self):
        items = [self._candidate(1, "NoShowcase", 5000, "SD 1.5")]
        with mock.patch.object(civitai, "search_models", lambda q, **kw: items), \
             mock.patch.object(civitai, "MIN_SEARCH_DOWNLOADS", 1000), \
             mock.patch.object(civitai, "harvest_from_model", self._harvest_by_model_id({})):
            with self.assertRaises(RuntimeError) as ctx:
                civitai.decide_reference("nothing")
        self.assertIn("no usable model+prompt", str(ctx.exception))


class CivitaiTest(unittest.TestCase):
    """Resolving an operator-named model, and refusing anything unsafe to load."""

    def test_url_with_version_is_parsed(self):
        self.assertEqual(
            civitai._parse_spec("https://civitai.com/models/4201?modelVersionId=130072"),
            (4201, 130072))

    def test_bare_url_without_version(self):
        self.assertEqual(civitai._parse_spec("https://civitai.com/models/4201"), (4201, None))

    def test_model_colon_version_form(self):
        self.assertEqual(civitai._parse_spec("4201:130072"), (4201, 130072))

    def test_bare_id(self):
        self.assertEqual(civitai._parse_spec("4201"), (4201, None))

    def test_picks_the_primary_safetensor_and_skips_pickle(self):
        info = {
            "id": 130072, "baseModel": "SD 1.5", "model": {"name": "Test Model"},
            "files": [
                {"name": "m.ckpt", "downloadUrl": "u1", "primary": False,
                 "metadata": {"format": "PickleTensor"},
                 "pickleScanResult": "Success", "virusScanResult": "Success"},
                {"name": "m-full.safetensors", "downloadUrl": "u2", "primary": False,
                 "metadata": {"format": "SafeTensor"}, "sizeKB": 4_000_000,
                 "pickleScanResult": "Success", "virusScanResult": "Success"},
                {"name": "m-pruned.safetensors", "downloadUrl": "u3", "primary": True,
                 "metadata": {"format": "SafeTensor"}, "sizeKB": 2_000_000,
                 "pickleScanResult": "Success", "virusScanResult": "Success"},
            ],
        }
        with mock.patch.object(civitai, "_version_info", lambda m, v: info):
            resolved = civitai.resolve("4201:130072")
        self.assertEqual(resolved["url"], "u3")
        self.assertEqual(resolved["arch"], "sd15")

    def test_refuses_a_failed_scan(self):
        info = {"id": 1, "baseModel": "SD 1.5", "model": {"name": "T"}, "files": [
            {"name": "m.safetensors", "downloadUrl": "u", "primary": True,
             "metadata": {"format": "SafeTensor"},
             "pickleScanResult": "Success", "virusScanResult": "Failed"},
        ]}
        with mock.patch.object(civitai, "_version_info", lambda m, v: info):
            with self.assertRaises(RuntimeError) as ctx:
                civitai.resolve("1")
        self.assertIn("scan-clean", str(ctx.exception))

    def test_pickle_only_model_is_refused(self):
        info = {"id": 1, "baseModel": "SD 1.5", "model": {"name": "T"}, "files": [
            {"name": "m.ckpt", "downloadUrl": "u", "primary": True,
             "metadata": {"format": "PickleTensor"},
             "pickleScanResult": "Success", "virusScanResult": "Success"},
        ]}
        with mock.patch.object(civitai, "_version_info", lambda m, v: info):
            with self.assertRaises(RuntimeError):
                civitai.resolve("1")

    def test_unsupported_architecture_is_refused(self):
        info = {"id": 1, "baseModel": "Pony", "model": {"name": "T"}, "files": []}
        with mock.patch.object(civitai, "_version_info", lambda m, v: info):
            with self.assertRaises(RuntimeError) as ctx:
                civitai.resolve("1")
        self.assertIn("unsupported base model", str(ctx.exception))

    def test_regional_block_is_not_retried_forever(self):
        calls = []

        def get(url, **kw):
            calls.append(1)
            return Response(451)

        with mock.patch.object(civitai.requests, "get", get):
            with self.assertRaises(RuntimeError) as ctx:
                civitai._get("https://civitai.com/api/v1/models", {})
        self.assertEqual(len(calls), 1, "a regional block will not clear on retry")
        self.assertIn("network", str(ctx.exception))


class RetryTest(unittest.TestCase):
    def test_transient_failure_is_retried_within_the_run(self):
        calls = []

        def run_niche(niche, state):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("NIM read timeout")

        with mock.patch.object(autopilot, "run_niche", run_niche), \
             mock.patch.object(autopilot.time, "sleep"):
            autopilot.run_niche_with_retries(NICHE, {}, attempts=3)
        self.assertEqual(len(calls), 3)

    def test_persistent_failure_still_raises(self):
        with mock.patch.object(autopilot, "run_niche",
                               mock.Mock(side_effect=RuntimeError("still broken"))), \
             mock.patch.object(autopilot.time, "sleep"):
            with self.assertRaises(RuntimeError):
                autopilot.run_niche_with_retries(NICHE, {}, attempts=2)


class RunNicheTest(unittest.TestCase):
    AIBEAUTY = {"id": "aibeauty", "hashtags": "#aiart",
               "ai_disclosure": "Created with AI.", "captions": ["Soft light."]}

    def test_skipped_without_tiktok_credentials(self):
        with mock.patch.object(tiktok, "enabled", lambda niche_id: False):
            # must not raise, and must not touch imageslides at all
            with mock.patch.object(imageslides, "generate",
                                   mock.Mock(side_effect=AssertionError("must not run"))):
                autopilot.run_niche(self.AIBEAUTY, {"topics": {}, "uploads": []})

    def test_real_run_queues_a_draft_and_records_state(self):
        fake_images = [Path(f"/tmp/i{i}.png") for i in range(5)]
        state = {"topics": {}, "uploads": []}
        with mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(tiktok, "enabled", lambda niche_id: True), \
             mock.patch.object(imageslides, "generate", lambda n, state=None: (fake_images, None, [None] * len(fake_images))), \
             mock.patch.object(tiktok, "host_file", lambda p: f"https://pages/media/{Path(p).name}"), \
             mock.patch.object(tiktok, "publish_photos_draft",
                               lambda imgs, niche_id, image_urls=None, caption=None, title=None: "publish1"), \
             mock.patch.object(tiktok, "check_publish_status",
                               lambda pid, niche_id: ("SEND_TO_USER_INBOX", None)), \
             mock.patch.object(autopilot.os, "remove", lambda p: None), \
             mock.patch.object(autopilot, "save_state", lambda s: None), \
             tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "ROOT", Path(tmp)):
                autopilot.run_niche(self.AIBEAUTY, state)
        entry = state["uploads"][-1]
        self.assertEqual(entry["tiktok_via"], "inbox")
        self.assertEqual(entry["tiktok_post_id"], "publish1")
        self.assertTrue(entry["tiktok"])
        # Image URLs are recorded on the upload so the video niche can pick from them.
        self.assertEqual(len(entry["image_urls"]), 5)

    def test_publish_that_fails_downstream_is_not_recorded_as_success(self):
        """init returning a publish_id only means TikTok accepted the job -- live-
        confirmed this can still fail downstream (photo_pull_failed etc.) and never
        reach the inbox. tiktok:true must reflect the polled outcome, not just
        whether init returned an id."""
        fake_images = [Path(f"/tmp/i{i}.png") for i in range(5)]
        state = {"topics": {}, "uploads": []}
        with mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(tiktok, "enabled", lambda niche_id: True), \
             mock.patch.object(imageslides, "generate", lambda n, state=None: (fake_images, None, [None] * len(fake_images))), \
             mock.patch.object(tiktok, "host_file", lambda p: f"https://pages/media/{Path(p).name}"), \
             mock.patch.object(tiktok, "publish_photos_draft",
                               lambda imgs, niche_id, image_urls=None, caption=None, title=None: "publish1"), \
             mock.patch.object(tiktok, "check_publish_status",
                               lambda pid, niche_id: ("FAILED", "photo_pull_failed")), \
             mock.patch.object(autopilot.os, "remove", lambda p: None), \
             mock.patch.object(autopilot, "save_state", lambda s: None), \
             tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "ROOT", Path(tmp)):
                autopilot.run_niche(self.AIBEAUTY, state)
        entry = state["uploads"][-1]
        self.assertFalse(entry["tiktok"])
        self.assertEqual(entry["tiktok_status"], "FAILED")

    def test_dry_run_writes_files_and_never_queues_a_draft(self):
        fake_images = [Path(f"/tmp/i{i}.png") for i in range(5)]
        with mock.patch.object(autopilot, "DRY_RUN", True), \
             mock.patch.object(imageslides, "generate", lambda n, state=None: (fake_images, None, [None] * len(fake_images))), \
             mock.patch.object(tiktok, "publish_photos_draft",
                               mock.Mock(side_effect=AssertionError("must not push"))), \
             mock.patch.object(autopilot.shutil, "copy", lambda a, b: None), \
             tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "OUT_DIR", Path(tmp)):
                autopilot.run_niche(self.AIBEAUTY, {"topics": {}, "uploads": []})

    def test_write_pending_captions_lists_only_tiktok_uploads(self):
        state = {"uploads": [
            {"tiktok": True, "tiktok_caption": "caption one", "ts": "t1", "niche": "aibeauty"},
            {"tiktok": False, "tiktok_caption": "", "ts": "t2", "niche": "aibeauty"},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "ROOT", Path(tmp)):
                autopilot.write_pending_captions(state)
                text = (Path(tmp) / "CAPTIONS.md").read_text()
        self.assertIn("caption one", text)

    def _upload(self, niche_id="aibeauty", hours_ago=1, tiktok_ok=True):
        ts = (autopilot.datetime.now() - autopilot.timedelta(hours=hours_ago))
        return {"niche": niche_id, "tiktok": tiktok_ok, "ts": ts.strftime("%Y-%m-%dT%H:%M:%S")}

    def test_pending_count_only_counts_this_niche_within_24h(self):
        state = {"uploads": [
            self._upload("aibeauty", hours_ago=1),
            self._upload("aibeauty", hours_ago=23),
            self._upload("aibeauty", hours_ago=25),      # outside the window
            self._upload("other-niche", hours_ago=1),     # different niche
            self._upload("aibeauty", hours_ago=1, tiktok_ok=False),  # never posted
        ]}
        self.assertEqual(autopilot._pending_drafts_last_24h(state, "aibeauty"), 2)

    def test_run_skipped_entirely_once_at_the_cap(self):
        """TikTok's own API caps at 5 pending drafts per rolling 24h -- there is no
        endpoint to ask it how many are still untouched, so posted.json's own recent
        history is the proxy. At the cap, skip without touching imageslides at all."""
        state = {"uploads": [self._upload() for _ in range(5)]}
        with mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(tiktok, "enabled", lambda niche_id: True), \
             mock.patch.object(imageslides, "generate",
                               mock.Mock(side_effect=AssertionError("must not run"))):
            autopilot.run_niche(self.AIBEAUTY, state)

    def test_partial_cap_room_clamps_videos_per_run(self):
        """3 already pushed in the last 24h, cap is 5 -> only 2 more get generated
        this run, not the full videos_per_run, so a mid-run push can't overshoot the
        cap and hit spam_risk_too_many_pending_share."""
        state = {"topics": {}, "uploads": [self._upload() for _ in range(3)]}
        calls = []

        def fake_generate(n, state=None):
            calls.append(1)
            return [Path(f"/tmp/i{len(calls)}.png")], None, [None]

        with mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(tiktok, "enabled", lambda niche_id: True), \
             mock.patch.object(imageslides, "generate", fake_generate), \
             mock.patch.object(tiktok, "host_file", lambda p: f"https://pages/media/{Path(p).name}"), \
             mock.patch.object(tiktok, "publish_photos_draft",
                               lambda imgs, niche_id, image_urls=None, caption=None, title=None: "p1"), \
             mock.patch.object(tiktok, "check_publish_status",
                               lambda pid, niche_id: ("SEND_TO_USER_INBOX", None)), \
             mock.patch.object(autopilot.os, "remove", lambda p: None), \
             mock.patch.object(autopilot, "save_state", lambda s: None):
            autopilot.run_niche({**self.AIBEAUTY, "videos_per_run": 5}, state)
        self.assertEqual(len(calls), 2)


class SupervisorRubricTest(unittest.TestCase):
    """No live-model unit test exists for review_image() itself (it calls a real NIM
    API, verified live during development instead), but the rubric wording is what
    actually determines the fully_clothed verdict -- a live check found a pose with no
    visible bottom garment at all (bare hip/rear, no waistband/hem/fabric) scored
    fully_clothed=True, because the rubric only asked about exposed breasts/genitals,
    not whether a real garment was actually present. This guards the fix in the
    wording, not the model's behavior."""

    def test_rubric_requires_a_real_garment_on_both_halves(self):
        self.assertIn("hips/groin/rear", supervisor.RUBRIC)
        self.assertIn("chest/breasts", supervisor.RUBRIC)

    def test_rubric_rejects_bare_skin_regardless_of_camera_angle(self):
        self.assertIn("camera angle", supervisor.RUBRIC.lower())

    def test_rubric_asks_about_ethnicity_exclusion(self):
        self.assertIn("ethnicity_excluded", supervisor.RUBRIC)


class CaptionWriterTest(unittest.TestCase):
    """caption_writer.write() replaces the old fixed-pool caption/hashtags -- a
    single "hashtags" string used on literally every post, forever, and a small
    static "captions" pool that cycled back to the same 14 lines regardless of how
    different the actual images were. Hermetic: _ask is mocked, no real NIM call."""

    def test_parses_a_well_formed_response(self):
        with mock.patch.object(caption_writer, "_ask", lambda prompt, **kw:
                               "CAPTION: Golden hour, no filter needed.\n"
                               "HASHTAGS: #aiart #beachday #goldenhour #confident"):
            caption, tags = caption_writer.write("a sunny beach day")
        self.assertEqual(caption, "Golden hour, no filter needed.")
        self.assertEqual(tags, "#aiart #beachday #goldenhour #confident")

    def test_strips_surrounding_quotes_from_the_caption(self):
        with mock.patch.object(caption_writer, "_ask", lambda prompt, **kw:
                               'CAPTION: "Quoted line."\nHASHTAGS: #aiart #vibe'):
            caption, tags = caption_writer.write("a vibe")
        self.assertEqual(caption, "Quoted line.")

    def test_ignores_hashtag_like_words_missing_the_hash(self):
        with mock.patch.object(caption_writer, "_ask", lambda prompt, **kw:
                               "CAPTION: A line.\nHASHTAGS: #aiart notahashtag #confident"):
            caption, tags = caption_writer.write("a vibe")
        self.assertEqual(tags, "#aiart #confident")

    def test_unparseable_response_raises(self):
        """Must raise, not silently return something empty -- image_caption()'s
        fallback to the static pool depends on this actually failing loudly."""
        with mock.patch.object(caption_writer, "_ask", lambda prompt, **kw: "not the expected format at all"):
            with self.assertRaises(RuntimeError):
                caption_writer.write("a vibe")

    def test_transient_failures_are_retried(self):
        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 2:
                return mock.Mock(status_code=503, text="busy")
            return mock.Mock(status_code=200, ok=True,
                             raise_for_status=lambda: None,
                             json=lambda: {"choices": [{"message": {"content":
                                 "CAPTION: Retried fine.\nHASHTAGS: #aiart #vibe"}}]})

        with mock.patch.object(caption_writer.requests, "post", fake_post), \
             mock.patch.object(caption_writer.time, "sleep", lambda s: None):
            result = caption_writer._ask("prompt")
        self.assertIn("Retried fine", result)
        self.assertEqual(calls["n"], 2)


class ReviewImageTest(unittest.TestCase):
    """review_image() itself, hermetic: _ask_vision is mocked per model so the
    multi-model agreement logic is under test, not any live model's judgment.

    A live check found the primary model confidently and wrongly scoring an image
    with full, unambiguous nudity as fully_clothed=True, while the secondary model
    refused to even discuss the same image ("I'm not going to engage in this
    conversation topic"). That refusal used to only matter as a fallback when the
    primary's response failed to PARSE -- never when it parsed fine but was simply
    wrong, which is exactly what let the bad image through. Now every configured
    model is always consulted and must all agree, so a wrong-but-confident primary
    can no longer single-handedly pass an image the secondary would refuse."""

    GOOD = '{"realistic": 9, "anatomy_ok": true, "fully_clothed": true, "age_appears_adult": true, "issues": []}'
    BAD_BUT_CONFIDENT = ('{"realistic": 9, "anatomy_ok": true, "fully_clothed": true, '
                         '"age_appears_adult": true, "issues": []}')

    def setUp(self):
        patch = mock.patch.object(supervisor, "_b64", lambda path: "fakebase64")
        patch.start()
        self.addCleanup(patch.stop)
        # ponytail: default VISION_MODELS is one model; force two so the multi-model
        # aggregation tests below exercise the aggregation, not just the trivial single case.
        models_patch = mock.patch.object(supervisor, "VISION_MODELS", ["m1", "m2"])
        models_patch.start()
        self.addCleanup(models_patch.stop)

    def test_all_models_agreeing_pass_actually_passes(self):
        with mock.patch.object(supervisor, "_ask_vision",
                               lambda model, prompt, b64, **kw: self.GOOD):
            result = supervisor.review_image("x.jpg")
        self.assertTrue(supervisor.passes(result))

    def test_a_refusal_from_any_model_rejects_even_if_another_confidently_passed(self):
        """Reproduces the exact live failure: primary model returns clean, parseable,
        wrong JSON; secondary model refuses outright. Must still reject."""
        def fake_ask(model, prompt, b64, **kw):
            if model == supervisor.VISION_MODELS[0]:
                return self.BAD_BUT_CONFIDENT
            raise RuntimeError("no JSON object in response: I'm not going to engage in this topic.")

        with mock.patch.object(supervisor, "_ask_vision", fake_ask):
            result = supervisor.review_image("x.jpg")
        self.assertFalse(supervisor.passes(result))

    def test_fully_clothed_is_recorded_but_no_longer_gates(self):
        """The account owner reviews every draft in the TikTok app before posting and
        can remove individual images from the carousel there -- nudity is caught
        downstream by a human either way, so gating it here was mostly costing
        variety without adding real protection past that point. anatomy_ok and
        age_appears_adult still gate; only fully_clothed was loosened."""
        def fake_ask(model, prompt, b64, **kw):
            if model == supervisor.VISION_MODELS[0]:
                return self.GOOD
            return ('{"realistic": 8, "anatomy_ok": true, "fully_clothed": false, '
                   '"age_appears_adult": true, "issues": ["exposed skin"]}')

        with mock.patch.object(supervisor, "_ask_vision", fake_ask):
            result = supervisor.review_image("x.jpg")
        self.assertFalse(result["fully_clothed"], "still recorded for visibility in logs")
        self.assertTrue(supervisor.passes(result), "must not block on fully_clothed alone")

    def test_anatomy_not_ok_still_rejects(self):
        def fake_ask(model, prompt, b64, **kw):
            if model == supervisor.VISION_MODELS[0]:
                return self.GOOD
            return ('{"realistic": 8, "anatomy_ok": false, "fully_clothed": true, '
                   '"age_appears_adult": true, "issues": ["extra finger"]}')

        with mock.patch.object(supervisor, "_ask_vision", fake_ask):
            result = supervisor.review_image("x.jpg")
        self.assertFalse(supervisor.passes(result))

    def test_age_not_adult_still_rejects(self):
        def fake_ask(model, prompt, b64, **kw):
            if model == supervisor.VISION_MODELS[0]:
                return self.GOOD
            return ('{"realistic": 8, "anatomy_ok": true, "fully_clothed": true, '
                   '"age_appears_adult": false, "issues": ["looks young"]}')

        with mock.patch.object(supervisor, "_ask_vision", fake_ask):
            result = supervisor.review_image("x.jpg")
        self.assertFalse(supervisor.passes(result))

    def test_ethnicity_excluded_rejects(self):
        """Exists specifically so a checkpoint that produces this pattern gets a low
        QA pass rate recorded, which is what actually drives it out of future
        selection (imageslides._model_weights) without anyone needing to have
        blocklisted its name in advance."""
        def fake_ask(model, prompt, b64, **kw):
            if model == supervisor.VISION_MODELS[0]:
                return self.GOOD
            return ('{"realistic": 8, "anatomy_ok": true, "fully_clothed": true, '
                   '"age_appears_adult": true, "ethnicity_excluded": true, "issues": []}')

        with mock.patch.object(supervisor, "_ask_vision", fake_ask):
            result = supervisor.review_image("x.jpg")
        self.assertTrue(result["ethnicity_excluded"])
        self.assertFalse(supervisor.passes(result))

    def test_ethnicity_excluded_false_across_the_board_still_passes(self):
        with mock.patch.object(supervisor, "_ask_vision",
                               lambda model, prompt, b64, **kw: self.GOOD):
            result = supervisor.review_image("x.jpg")
        self.assertFalse(result["ethnicity_excluded"])
        self.assertTrue(supervisor.passes(result))

    def test_realistic_score_is_the_minimum_across_models(self):
        def fake_ask(model, prompt, b64, **kw):
            score = 9 if model == supervisor.VISION_MODELS[0] else 5
            return (f'{{"realistic": {score}, "anatomy_ok": true, "fully_clothed": true, '
                   f'"age_appears_adult": true, "issues": []}}')

        with mock.patch.object(supervisor, "_ask_vision", fake_ask):
            result = supervisor.review_image("x.jpg")
        self.assertEqual(result["realistic"], 5)


class StaticCheckTest(unittest.TestCase):
    def test_no_undefined_names(self):
        """pyflakes catches a NameError class of bug without executing anything."""
        try:
            import pyflakes
            del pyflakes
        except ImportError:
            self.skipTest("pyflakes not installed (pip install pyflakes)")
        files = ["autopilot.py", "tiktok.py", "push_draft.py", "civitai.py",
                "imageslides.py", "sdgen.py", "supervisor.py"]
        r = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                           cwd=Path(__file__).parent, capture_output=True, text=True)
        undefined = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
        self.assertEqual(undefined, [], "undefined names:\n" + "\n".join(undefined))


class TikTokHostingTest(unittest.TestCase):
    """host_file() drives real git commands (fetch, worktree add, commit, push) --
    exercised here against a real local git remote rather than mocked, since a mocked
    git call would not catch an actual command-syntax bug. GitHub Pages was chosen
    over GitHub Releases after a live check found release-asset download URLs
    302-redirect to a signed, ~1h-expiring URL, which TikTok's PULL_FROM_URL disallows;
    Pages serves files directly with no redirect."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.remote = root / "remote.git"
        self.repo = root / "repo"

        subprocess.run(["git", "init", "--bare", str(self.remote)],
                       check=True, capture_output=True)
        subprocess.run(["git", "clone", str(self.remote), str(self.repo)],
                       check=True, capture_output=True)
        self._git("config", "user.name", "test")
        self._git("config", "user.email", "test@example.com")
        (self.repo / "README.md").write_text("main branch")
        self._git("add", "README.md")
        self._git("commit", "-m", "init")
        self._git("push", "-u", "origin", "HEAD:main")

        self._git("checkout", "--orphan", "gh-pages")
        self._git("rm", "-rf", ".", allow_fail=True)
        (self.repo / "media").mkdir()
        (self.repo / "media" / ".gitkeep").write_text("")
        self._git("add", "-A")
        self._git("commit", "-m", "init gh-pages")
        self._git("push", "-u", "origin", "gh-pages")
        self._git("checkout", "main")

        self.image = root / "source.png"
        self.image.write_bytes(b"fake-image-bytes")

        patches = [mock.patch.object(tiktok, "REPO_ROOT", self.repo),
                  mock.patch.object(tiktok, "PAGES_WORKTREE", root / "pages-wt"),
                  # host_file() polls the real URL before returning it (a live check
                  # found TikTok's fetch racing GitHub Pages' own build/deploy delay);
                  # these tests use a fake base_url that never resolves, so that check
                  # is exercised separately in WaitUntilLiveTest instead.
                  mock.patch.object(tiktok, "_wait_until_live", lambda url, **kw: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _git(self, *args, allow_fail=False):
        r = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True)
        if r.returncode != 0 and not allow_fail:
            raise RuntimeError(r.stderr)
        return r.stdout

    def test_host_file_commits_and_returns_a_working_url(self):
        url = tiktok.host_file(self.image, base_url="https://example.test/repo")
        self.assertTrue(url.startswith("https://example.test/repo/media/"))
        self.assertTrue(url.endswith("source.png"))
        files = list((tiktok.PAGES_WORKTREE / "media").glob("*source.png"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), b"fake-image-bytes")

    def test_host_file_requires_base_url(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                tiktok.host_file(self.image, base_url=None)
        self.assertIn("PAGES_BASE_URL", str(ctx.exception))

    def test_pushed_file_is_actually_on_the_remote(self):
        """Not just committed locally -- the whole point is TikTok's servers fetching
        it, which only works once it's actually pushed."""
        tiktok.host_file(self.image, base_url="https://example.test/repo")
        r = subprocess.run(["git", "ls-tree", "-r", "--name-only", "origin/gh-pages"],
                           cwd=tiktok.PAGES_WORKTREE, capture_output=True, text=True)
        self.assertIn("source.png", r.stdout)

    def test_pruning_keeps_only_the_newest_files(self):
        with mock.patch.object(tiktok, "KEEP_MEDIA", 3):
            for i in range(5):
                img = Path(self.tmp.name) / f"src{i}.png"
                img.write_bytes(f"data{i}".encode())
                tiktok.host_file(img, base_url="https://example.test/repo")
        remaining = list((tiktok.PAGES_WORKTREE / "media").glob("*.png"))
        self.assertLessEqual(len(remaining), 3)


class WaitUntilLiveTest(unittest.TestCase):
    """A live check found TikTok's PULL_FROM_URL failing with photo_pull_failed right
    after a successful push -- git push succeeding only means GitHub has the commit,
    not that Pages has finished building and serving it. host_file() must not hand a
    URL to TikTok until this confirms it's actually live."""

    def test_returns_once_the_url_is_live(self):
        responses = [mock.Mock(status_code=404), mock.Mock(status_code=404),
                    mock.Mock(status_code=200)]
        with mock.patch.object(tiktok.requests, "head",
                               mock.Mock(side_effect=responses)), \
             mock.patch.object(tiktok.time, "sleep"):
            tiktok._wait_until_live("https://example.test/x.png", timeout=5, interval=0.01)

    def test_raises_after_timeout_if_never_live(self):
        with mock.patch.object(tiktok.requests, "head",
                               lambda *a, **k: mock.Mock(status_code=404)), \
             mock.patch.object(tiktok.time, "sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                tiktok._wait_until_live("https://example.test/x.png", timeout=0.05, interval=0.01)
        self.assertIn("did not go live", str(ctx.exception))

    def test_connection_errors_are_retried_not_raised_immediately(self):
        responses = [tiktok.requests.RequestException("dns fail"), mock.Mock(status_code=200)]

        def fake_head(*a, **k):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.object(tiktok.requests, "head", fake_head), \
             mock.patch.object(tiktok.time, "sleep"):
            tiktok._wait_until_live("https://example.test/x.png", timeout=5, interval=0.01)


class TikTokEnabledTest(unittest.TestCase):
    def test_enabled_requires_all_three_credentials(self):
        with mock.patch.dict(os.environ, {"TIKTOK_CLIENT_KEY": "k", "TIKTOK_CLIENT_SECRET": "s",
                                          "TIKTOK_REFRESH_TOKEN_AIBEAUTY": "r"}):
            self.assertTrue(tiktok.enabled("aibeauty"))

    def test_disabled_when_this_niches_token_is_missing(self):
        with mock.patch.dict(os.environ, {"TIKTOK_CLIENT_KEY": "k", "TIKTOK_CLIENT_SECRET": "s"},
                             clear=False):
            os.environ.pop("TIKTOK_REFRESH_TOKEN_AIBEAUTY", None)
            self.assertFalse(tiktok.enabled("aibeauty"))


class TikTokPublishDraftTest(unittest.TestCase):
    """Native TikTok inbox draft: PULL_FROM_URL only (photos don't support
    FILE_UPLOAD), MEDIA_UPLOAD post_mode (draft, not a live post). Not yet verified
    against a real TikTok account -- see tiktok.py's module docstring."""

    def _env(self):
        return mock.patch.dict(os.environ, {
            "TIKTOK_CLIENT_KEY": "ck", "TIKTOK_CLIENT_SECRET": "cs",
            "TIKTOK_REFRESH_TOKEN_AIBEAUTY": "rt",
        })

    def test_returns_none_without_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(tiktok.publish_photos_draft(["a.png", "b.png"], "aibeauty"))

    def test_single_image_is_refused(self):
        with self._env():
            with self.assertRaises(RuntimeError) as ctx:
                tiktok.publish_photos_draft(None, "aibeauty", image_urls=["u1"])
        self.assertIn("at least 2", str(ctx.exception))

    def test_sends_pull_from_url_photo_media_upload(self):
        captured = {}

        def fake_post(url, headers=None, json=None, data=None, timeout=None):
            if "oauth/token" in url:
                return mock.Mock(ok=True, raise_for_status=lambda: None,
                                 json=lambda: {"access_token": "acc"})
            captured["url"] = url
            captured["json"] = json
            return mock.Mock(ok=True, json=lambda: {"data": {"publish_id": "p1"}})

        with self._env(), mock.patch.object(tiktok.requests, "post", fake_post):
            publish_id = tiktok.publish_photos_draft(
                None, "aibeauty", image_urls=["u1", "u2", "u3"])
        self.assertEqual(publish_id, "p1")
        self.assertIn("content/init", captured["url"])
        self.assertEqual(captured["json"]["media_type"], "PHOTO")
        self.assertEqual(captured["json"]["post_mode"], "MEDIA_UPLOAD")
        self.assertEqual(captured["json"]["source_info"]["source"], "PULL_FROM_URL")
        self.assertEqual(captured["json"]["source_info"]["photo_images"], ["u1", "u2", "u3"])
        self.assertNotIn("post_info", captured["json"], "no caption given, no post_info sent")

    def test_caption_prefills_post_info(self):
        """MEDIA_UPLOAD now accepts post_info same as DIRECT_POST -- checked live
        against TikTok's current docs, since this module's original design predated
        that and assumed no caption could be attached to a draft at all."""
        captured = {}

        def fake_post(url, headers=None, json=None, data=None, timeout=None):
            if "oauth/token" in url:
                return mock.Mock(ok=True, raise_for_status=lambda: None,
                                 json=lambda: {"access_token": "acc"})
            captured["json"] = json
            return mock.Mock(ok=True, json=lambda: {"data": {"publish_id": "p1"}})

        long_caption = "My hook line\n\n" + ("x" * 4100)
        with self._env(), mock.patch.object(tiktok.requests, "post", fake_post):
            tiktok.publish_photos_draft(None, "aibeauty", image_urls=["u1", "u2"],
                                        caption=long_caption)
        self.assertEqual(captured["json"]["post_info"]["title"], "My hook line")
        self.assertEqual(len(captured["json"]["post_info"]["description"]), 4000)
        self.assertTrue(long_caption.startswith(captured["json"]["post_info"]["description"]))

    def test_explicit_title_overrides_captions_first_line(self):
        captured = {}

        def fake_post(url, headers=None, json=None, data=None, timeout=None):
            if "oauth/token" in url:
                return mock.Mock(ok=True, raise_for_status=lambda: None,
                                 json=lambda: {"access_token": "acc"})
            captured["json"] = json
            return mock.Mock(ok=True, json=lambda: {"data": {"publish_id": "p1"}})

        with self._env(), mock.patch.object(tiktok.requests, "post", fake_post):
            tiktok.publish_photos_draft(None, "aibeauty", image_urls=["u1", "u2"],
                                        caption="first line caption", title="custom title")
        self.assertEqual(captured["json"]["post_info"]["title"], "custom title")

    def test_api_failure_raises_with_response_body(self):
        def fake_post(url, headers=None, json=None, data=None, timeout=None):
            if "oauth/token" in url:
                return mock.Mock(ok=True, raise_for_status=lambda: None,
                                 json=lambda: {"access_token": "acc"})
            return mock.Mock(ok=False, status_code=403, text="domain not verified")

        with self._env(), mock.patch.object(tiktok.requests, "post", fake_post):
            with self.assertRaises(RuntimeError) as ctx:
                tiktok.publish_photos_draft(None, "aibeauty", image_urls=["u1", "u2"])
        self.assertIn("domain not verified", str(ctx.exception))

    def test_hosts_images_when_no_urls_given(self):
        with self._env(), \
             mock.patch.object(tiktok, "host_file", lambda p: f"hosted://{p}"), \
             mock.patch.object(tiktok, "_access_token", lambda ck, cs, rt: "acc"), \
             mock.patch.object(tiktok.requests, "post",
                               lambda *a, **k: mock.Mock(ok=True, json=lambda: {
                                   "data": {"publish_id": "p1"}})):
            tiktok.publish_photos_draft(["a.png", "b.png"], "aibeauty")


class CheckPublishStatusTest(unittest.TestCase):
    """init's 200 OK only means TikTok ACCEPTED the job -- live-confirmed a real
    batch of "successful" pushes included several that had actually failed
    downstream and were never polled for the real outcome. check_publish_status()
    is what closes that gap."""

    def _env(self):
        return mock.patch.dict(os.environ, {
            "TIKTOK_CLIENT_KEY": "ck", "TIKTOK_CLIENT_SECRET": "cs",
            "TIKTOK_REFRESH_TOKEN_AIBEAUTY": "rt",
        })

    def test_terminal_success_is_returned_immediately(self):
        with self._env(), \
             mock.patch.object(tiktok, "_access_token", lambda ck, cs, rt: "acc"), \
             mock.patch.object(tiktok.requests, "post", lambda *a, **k: mock.Mock(
                 raise_for_status=lambda: None,
                 json=lambda: {"data": {"status": "SEND_TO_USER_INBOX"}})):
            status, reason = tiktok.check_publish_status("p1", "aibeauty")
        self.assertEqual(status, "SEND_TO_USER_INBOX")
        self.assertIsNone(reason)

    def test_terminal_failure_returns_the_reason(self):
        with self._env(), \
             mock.patch.object(tiktok, "_access_token", lambda ck, cs, rt: "acc"), \
             mock.patch.object(tiktok.requests, "post", lambda *a, **k: mock.Mock(
                 raise_for_status=lambda: None,
                 json=lambda: {"data": {"status": "FAILED",
                                        "fail_reason": "photo_pull_failed"}})):
            status, reason = tiktok.check_publish_status("p1", "aibeauty")
        self.assertEqual(status, "FAILED")
        self.assertEqual(reason, "photo_pull_failed")

    def test_polls_through_non_terminal_states(self):
        calls = {"n": 0}

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            status = "PROCESSING_DOWNLOAD" if calls["n"] < 3 else "SEND_TO_USER_INBOX"
            return mock.Mock(raise_for_status=lambda: None,
                             json=lambda: {"data": {"status": status}})

        with self._env(), \
             mock.patch.object(tiktok, "_access_token", lambda ck, cs, rt: "acc"), \
             mock.patch.object(tiktok.requests, "post", fake_post), \
             mock.patch.object(tiktok.time, "sleep", lambda s: None):
            status, _ = tiktok.check_publish_status("p1", "aibeauty", interval=0)
        self.assertEqual(status, "SEND_TO_USER_INBOX")
        self.assertEqual(calls["n"], 3)

    def test_never_reaching_terminal_times_out(self):
        with self._env(), \
             mock.patch.object(tiktok, "_access_token", lambda ck, cs, rt: "acc"), \
             mock.patch.object(tiktok.requests, "post", lambda *a, **k: mock.Mock(
                 raise_for_status=lambda: None,
                 json=lambda: {"data": {"status": "PROCESSING_DOWNLOAD"}})), \
             mock.patch.object(tiktok.time, "sleep", lambda s: None):
            status, reason = tiktok.check_publish_status(
                "p1", "aibeauty", timeout=0.01, interval=0.01)
        self.assertEqual(status, "TIMEOUT")


class PushDraftTest(unittest.TestCase):
    def _make_batch(self, tmp, n=3, caption="hello"):
        folder = Path(tmp) / "aibeauty-20260101-120000"
        folder.mkdir()
        for i in range(n):
            (folder / f"sd_{i}.jpg").write_bytes(b"fake")
        (folder / "caption.txt").write_text(caption)
        return folder

    def test_pushes_and_records_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_batch(tmp)
            with mock.patch.object(tiktok, "publish_photos_draft",
                                   lambda imgs, niche_id, caption=None: "p1"), \
                 mock.patch.object(tiktok, "check_publish_status",
                                   lambda pid, niche_id: ("SEND_TO_USER_INBOX", None)), \
                 mock.patch.object(autopilot, "ROOT", Path(tmp)), \
                 mock.patch.object(autopilot, "STATE_FILE", Path(tmp) / "posted.json"):
                publish_id = push_draft.push_draft(folder)
            self.assertEqual(publish_id, "p1")
            state = json.loads((Path(tmp) / "posted.json").read_text())
            self.assertEqual(state["uploads"][-1]["tiktok_post_id"], "p1")
            self.assertEqual(state["uploads"][-1]["niche"], "aibeauty")
            self.assertTrue(state["uploads"][-1]["tiktok"])

    def test_downstream_failure_is_recorded_as_not_queued(self):
        """Mirrors the real incident: init accepted the job (returned a publish_id)
        but it failed downstream and never reached the inbox. Must not be recorded
        as a success, or it silently eats a slot in the pending-drafts cap forever."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_batch(tmp)
            with mock.patch.object(tiktok, "publish_photos_draft",
                                   lambda imgs, niche_id, caption=None: "p1"), \
                 mock.patch.object(tiktok, "check_publish_status",
                                   lambda pid, niche_id: ("FAILED", "photo_pull_failed")), \
                 mock.patch.object(autopilot, "ROOT", Path(tmp)), \
                 mock.patch.object(autopilot, "STATE_FILE", Path(tmp) / "posted.json"):
                push_draft.push_draft(folder)
            state = json.loads((Path(tmp) / "posted.json").read_text())
            self.assertFalse(state["uploads"][-1]["tiktok"])
            self.assertEqual(state["uploads"][-1]["tiktok_status"], "FAILED")

    def test_niche_id_inferred_from_folder_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_batch(tmp)
            captured = {}

            def fake_publish(imgs, niche_id, caption=None):
                captured["niche"] = niche_id
                return "p1"

            with mock.patch.object(tiktok, "publish_photos_draft", fake_publish), \
                 mock.patch.object(tiktok, "check_publish_status",
                                   lambda pid, niche_id: ("SEND_TO_USER_INBOX", None)), \
                 mock.patch.object(autopilot, "ROOT", Path(tmp)), \
                 mock.patch.object(autopilot, "STATE_FILE", Path(tmp) / "posted.json"):
                push_draft.push_draft(folder)
            self.assertEqual(captured["niche"], "aibeauty")

    def test_too_few_images_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_batch(tmp, n=1)
            with self.assertRaises(ValueError) as ctx:
                push_draft.push_draft(folder)
        self.assertIn("at least 2", str(ctx.exception))

    def test_missing_credentials_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self._make_batch(tmp)
            with mock.patch.object(tiktok, "publish_photos_draft", lambda imgs, niche_id, caption=None: None):
                with self.assertRaises(RuntimeError) as ctx:
                    push_draft.push_draft(folder)
        self.assertIn("no TikTok credentials", str(ctx.exception))


class SupervisorBrokenDetectionTest(unittest.TestCase):
    """filter_images returns supervisor_broken=True only when every image failed
    with a supervisor-model error (not with a real content rejection). If a future
    edit to review_image's error message shape breaks this detection, the fallback
    in imageslides.generate silently stops firing and CI wastes real generation
    time on broken infra again -- so pin the shape here."""

    def test_broken_pattern_is_detected(self):
        broken = {"realistic": 0, "anatomy_ok": False, "fully_clothed": False,
                 "age_appears_adult": False,
                 "issues": ["llava:7b gave no usable verdict: 500 error"]}
        self.assertTrue(supervisor._is_broken_verdict(broken))

    def test_real_reject_is_not_flagged_broken(self):
        real = {"realistic": 3, "anatomy_ok": False, "fully_clothed": True,
               "age_appears_adult": True, "issues": ["extra finger"]}
        self.assertFalse(supervisor._is_broken_verdict(real))

    def test_filter_images_flags_supervisor_broken_when_every_image_errored(self):
        with mock.patch.object(supervisor, "review_image",
                              lambda p: {"realistic": 0, "anatomy_ok": False,
                                         "fully_clothed": False, "age_appears_adult": False,
                                         "issues": ["llava:7b gave no usable verdict: X"]}):
            result = supervisor.filter_images(["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(list(result), [])
        self.assertTrue(result.supervisor_broken)

    def test_filter_images_does_not_flag_broken_when_any_verdict_is_real(self):
        """If ONE image got a real (rejecting) verdict and the rest were errors,
        we still can't conclude the supervisor is broken -- maybe the model just
        crashed on some inputs but is otherwise judging content. Safer path is
        to not fall back and let the run fail loudly."""
        results = [
            {"realistic": 3, "anatomy_ok": False, "issues": ["extra finger"]},
            {"realistic": 0, "anatomy_ok": False, "issues": ["llava:7b gave no usable verdict: X"]},
        ]
        it = iter(results)
        with mock.patch.object(supervisor, "review_image", lambda p: next(it)):
            result = supervisor.filter_images(["a.jpg", "b.jpg"])
        self.assertFalse(result.supervisor_broken)


class TikTokPublishVideoDraftTest(unittest.TestCase):
    """publish_video_draft: PULL_FROM_URL to /v2/post/publish/inbox/video/init/.
    token_niche defaults to niche_id but the video niche points it at the source
    (aibeauty) so the video draft lands in the same channel as the photos."""

    def _fake_post(self, captured):
        def post(url, headers=None, json=None, timeout=None, **kw):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            if url.endswith("/oauth/token/"):
                return mock.Mock(status_code=200, ok=True,
                                 raise_for_status=lambda: None,
                                 json=lambda: {"access_token": "acc"})
            return mock.Mock(status_code=200, ok=True,
                             raise_for_status=lambda: None,
                             json=lambda: {"data": {"publish_id": "pv_1"}})
        return post

    def test_pull_from_url_body_and_token_niche_override(self):
        captured = {}
        env = {"TIKTOK_CLIENT_KEY": "ck", "TIKTOK_CLIENT_SECRET": "cs",
              "TIKTOK_REFRESH_TOKEN_AIBEAUTY": "rt"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(tiktok.requests, "post", self._fake_post(captured)), \
             mock.patch.object(tiktok, "host_file", lambda p: "https://pages/media/vid.mp4"):
            pid = tiktok.publish_video_draft(
                "/tmp/vid.mp4", "aibeautyvideo", token_niche="aibeauty",
                caption="Some days just look like this.")

        self.assertEqual(pid, "pv_1")
        self.assertIn("/v2/post/publish/inbox/video/init/", captured["url"])
        self.assertEqual(captured["json"]["source_info"]["source"], "PULL_FROM_URL")
        self.assertEqual(captured["json"]["source_info"]["video_url"],
                        "https://pages/media/vid.mp4")
        self.assertIn("post_info", captured["json"])

    def test_missing_credentials_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(tiktok.publish_video_draft(
                "/tmp/vid.mp4", "aibeautyvideo", token_niche="aibeauty"))


class AutopilotVideoNicheTest(unittest.TestCase):
    """The video niche reuses an image URL that the photo niche recorded in
    state["uploads"], calls videogen.generate (mocked), and publishes via
    tiktok.publish_video_draft using the SOURCE niche's TikTok token."""

    VIDEO_NICHE = {
        "id": "aibeautyvideo", "content_type": "video_via_motionforge",
        "source_niche": "aibeauty", "tiktok_token_niche": "aibeauty",
        "motionforge_prompt": "she smiles", "motionforge_length_s": "5.0",
        "motionforge_steps": "4", "captions": ["Golden hour."],
        "hashtags": "#aiart", "ai_disclosure": "Created with AI.",
    }
    ENV_WITH_TOKEN = {"TIKTOK_CLIENT_KEY": "ck", "TIKTOK_CLIENT_SECRET": "cs",
                     "TIKTOK_REFRESH_TOKEN_AIBEAUTY": "rt"}

    def _photo_upload(self, image_urls, tiktok_ok=True):
        return {"niche": "aibeauty", "tiktok": tiktok_ok, "image_urls": image_urls,
               "ts": "2026-08-27T14:00:00"}

    def test_picks_most_recent_unused_aibeauty_image(self):
        state = {"topics": {}, "uploads": [
            self._photo_upload(["https://pages/media/a.jpg", "https://pages/media/b.jpg"]),
            self._photo_upload(["https://pages/media/c.jpg"]),
        ]}
        # `c.jpg` is newest -- should be picked first.
        self.assertEqual(autopilot._pick_source_image_url(self.VIDEO_NICHE, state),
                        "https://pages/media/c.jpg")

    def test_skips_urls_already_animated(self):
        state = {"topics": {}, "uploads": [
            self._photo_upload(["https://pages/media/a.jpg"]),
            {"niche": "aibeautyvideo", "motionforge_source_url": "https://pages/media/a.jpg",
             "ts": "2026-08-27T15:00:00"},
        ]}
        # a.jpg already animated -- no fresh URL available.
        self.assertIsNone(autopilot._pick_source_image_url(self.VIDEO_NICHE, state))

    def test_skips_source_uploads_that_never_reached_tiktok(self):
        """Failed drafts (tiktok=false) don't count as valid source images -- their
        URL may still resolve but the batch as a whole didn't ship, so reusing one
        of its photos on video would surface a batch that was rejected."""
        state = {"topics": {}, "uploads": [
            self._photo_upload(["https://pages/media/a.jpg"], tiktok_ok=False),
        ]}
        self.assertIsNone(autopilot._pick_source_image_url(self.VIDEO_NICHE, state))

    def test_no_source_image_skips_run(self):
        state = {"topics": {}, "uploads": []}
        called = {}
        with mock.patch.dict(os.environ, self.ENV_WITH_TOKEN, clear=True), \
             mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(autopilot.videogen, "generate",
                              lambda *a, **kw: called.setdefault("gen", True)):
            autopilot.run_niche(self.VIDEO_NICHE, state)
        self.assertNotIn("gen", called)
        self.assertEqual(state["uploads"], [])

    def test_missing_tiktok_token_skips_silently(self):
        """Photo cron doesn't have the video niche's TikTok token in its env by
        accident? Silent skip -- same pattern the photo path uses for its niche."""
        state = {"topics": {}, "uploads": [
            self._photo_upload(["https://pages/media/a.jpg"]),
        ]}
        called = {}
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(autopilot.videogen, "generate",
                              lambda *a, **kw: called.setdefault("gen", True)):
            autopilot.run_niche(self.VIDEO_NICHE, state)
        self.assertNotIn("gen", called)
        self.assertEqual(len(state["uploads"]), 1)  # unchanged

    def test_happy_path_generates_and_publishes_video(self):
        state = {"topics": {}, "uploads": [
            self._photo_upload(["https://pages/media/a.jpg"]),
        ]}
        calls = {}

        def fake_generate(image_url, prompt, length_s="5.0", steps="4", **kw):
            calls["gen"] = {"image_url": image_url, "prompt": prompt,
                           "length_s": length_s, "steps": steps}
            return "/tmp/final.mp4"

        def fake_publish(video_path, niche_id, video_url=None, caption=None, token_niche=None, **kw):
            calls["publish"] = {"video_path": video_path, "niche_id": niche_id,
                                "video_url": video_url, "token_niche": token_niche,
                                "caption": caption}
            return "pv_1"

        with mock.patch.dict(os.environ, self.ENV_WITH_TOKEN, clear=True), \
             mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(autopilot.videogen, "generate", fake_generate), \
             mock.patch.object(autopilot.tiktok, "host_file",
                              lambda p: "https://pages/media/vid.mp4"), \
             mock.patch.object(autopilot.tiktok, "publish_video_draft", fake_publish), \
             mock.patch.object(autopilot.tiktok, "check_publish_status",
                              lambda pid, niche_id, token_niche=None: ("SEND_TO_USER_INBOX", None)), \
             mock.patch.object(autopilot, "save_state", lambda s: None), \
             mock.patch.object(autopilot, "write_pending_captions", lambda s: None), \
             mock.patch.object(autopilot.os, "remove", lambda p: None):
            autopilot.run_niche(self.VIDEO_NICHE, state)

        self.assertEqual(calls["gen"]["image_url"], "https://pages/media/a.jpg")
        self.assertEqual(calls["gen"]["prompt"], "she smiles")
        self.assertEqual(calls["publish"]["token_niche"], "aibeauty")
        # publish_video_draft gets the URL we already hosted, not a bare video_path
        # for it to host again itself.
        self.assertEqual(calls["publish"]["video_url"], "https://pages/media/vid.mp4")
        # The video upload records the source url so subsequent runs don't reuse it,
        # plus the hosted mp4 url so it stays visible/downloadable/retriable even if
        # TikTok's own check later rejects it.
        video_up = state["uploads"][-1]
        self.assertEqual(video_up["niche"], "aibeautyvideo")
        self.assertEqual(video_up["motionforge_source_url"], "https://pages/media/a.jpg")
        self.assertEqual(video_up["video_url"], "https://pages/media/vid.mp4")
        self.assertTrue(video_up["tiktok"])

    def test_video_image_url_env_override_beats_auto_pick(self):
        """push_video.py fires workflow_dispatch with the picked image_url as an
        input; the workflow surfaces it as VIDEO_IMAGE_URL. That must override the
        auto-pick (most-recent-in-state) -- the whole point of the manual UI is
        that the user chose a specific image, not the newest."""
        state = {"topics": {}, "uploads": [
            self._photo_upload(["https://pages/media/newest.jpg"]),
        ]}
        calls = {}

        env = dict(self.ENV_WITH_TOKEN)
        env["VIDEO_IMAGE_URL"] = "https://pages/media/user-picked.jpg"
        env["VIDEO_PROMPT"] = "she looks up and laughs"

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(autopilot.videogen, "generate",
                              lambda url, prompt, **kw: (calls.setdefault(
                                  "gen", {"url": url, "prompt": prompt}) or "/tmp/final.mp4")), \
             mock.patch.object(autopilot.tiktok, "host_file",
                              lambda p: "https://pages/media/vid.mp4"), \
             mock.patch.object(autopilot.tiktok, "publish_video_draft",
                              lambda *a, **kw: "pv_1"), \
             mock.patch.object(autopilot.tiktok, "check_publish_status",
                              lambda *a, **kw: ("SEND_TO_USER_INBOX", None)), \
             mock.patch.object(autopilot, "save_state", lambda s: None), \
             mock.patch.object(autopilot, "write_pending_captions", lambda s: None), \
             mock.patch.object(autopilot.os, "remove", lambda p: None):
            autopilot.run_niche(self.VIDEO_NICHE, state)

        self.assertEqual(calls["gen"]["url"], "https://pages/media/user-picked.jpg")
        self.assertEqual(calls["gen"]["prompt"], "she looks up and laughs")
        self.assertEqual(state["uploads"][-1]["motionforge_source_url"],
                        "https://pages/media/user-picked.jpg")

    def test_video_retry_url_skips_regeneration(self):
        """The picker's Retry button on an already-generated (but TikTok-rejected)
        video sets VIDEO_RETRY_URL -- must republish that exact mp4 straight to
        TikTok without ever touching videogen.generate() or picking a source image."""
        state = {"topics": {}, "uploads": []}
        calls = {}

        env = dict(self.ENV_WITH_TOKEN)
        env["VIDEO_RETRY_URL"] = "https://pages/media/1787842736-final.mp4"

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(autopilot.videogen, "generate",
                              mock.Mock(side_effect=AssertionError("must not regenerate"))), \
             mock.patch.object(autopilot.tiktok, "host_file",
                              mock.Mock(side_effect=AssertionError("must not re-host"))), \
             mock.patch.object(autopilot.tiktok, "publish_video_draft",
                              lambda *a, **kw: calls.setdefault("publish", kw) or "pv_retry"), \
             mock.patch.object(autopilot.tiktok, "check_publish_status",
                              lambda *a, **kw: ("SEND_TO_USER_INBOX", None)), \
             mock.patch.object(autopilot, "save_state", lambda s: None), \
             mock.patch.object(autopilot, "write_pending_captions", lambda s: None):
            autopilot.run_niche(self.VIDEO_NICHE, state)

        self.assertEqual(calls["publish"]["video_url"],
                        "https://pages/media/1787842736-final.mp4")
        video_up = state["uploads"][-1]
        self.assertEqual(video_up["tiktok_via"], "inbox_video_retry")
        self.assertEqual(video_up["video_url"], "https://pages/media/1787842736-final.mp4")
        self.assertTrue(video_up["tiktok"])
        self.assertNotIn("motionforge_source_url", video_up)


class PickerHtmlTest(unittest.TestCase):
    """picker.html is a static page on gh-pages -- no Python glue to test, but
    a few invariants are worth catching if a future edit breaks them: the
    hardcoded owner/repo/workflow, the raw.githubusercontent URLs, and the
    workflow_dispatch endpoint shape."""

    HTML = (Path(__file__).parent / "picker.html").read_text()

    def test_targets_correct_workflow(self):
        self.assertIn('WORKFLOW = "autopilot_video.yml"', self.HTML)

    def test_fetches_posted_and_niches_from_raw(self):
        self.assertIn("raw.githubusercontent.com/${OWNER}/${REPO}/main/posted.json", self.HTML)
        self.assertIn("raw.githubusercontent.com/${OWNER}/${REPO}/main/niches.json", self.HTML)

    def test_hits_workflow_dispatch_api(self):
        self.assertIn("actions/workflows/${WORKFLOW}/dispatches", self.HTML)
        # PAT is stored in localStorage (deliberate tradeoff for zero-setup UX --
        # scope it to actions:write on this repo only per the setup blurb).
        self.assertIn('localStorage.getItem("mpt_pat")', self.HTML)


class VideoGenTest(unittest.TestCase):
    """videogen.generate shells out to motionforge's own scripts (prepare_kernel.py,
    `kaggle kernels push`, poll_kaggle.py). Tests assert the subprocess plumbing
    (env passed, commands invoked, mp4 copied out) with everything mocked."""

    def _setup_fake_motionforge(self, tmp):
        """Build a fake motionforge tree under tmp with the scripts videogen expects."""
        mf = Path(tmp) / "motionforge"
        (mf / "scripts").mkdir(parents=True)
        (mf / "kaggle_output").mkdir(parents=True)
        (mf / "scripts" / "prepare_kernel.py").write_text("# fake")
        (mf / "scripts" / "poll_kaggle.py").write_text("# fake")
        (mf / "kaggle_output" / "final.mp4").write_bytes(b"fakemp4payload")
        return mf

    def test_shells_out_and_copies_mp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf = self._setup_fake_motionforge(tmp)
            out_dir = Path(tmp) / "out"
            calls = []

            def fake_run(cmd, cwd=None, env=None, check=None, capture_output=None,
                        text=None):
                calls.append({"cmd": cmd, "cwd": cwd, "env": env})
                if cmd[0] == "ffmpeg":
                    # Real ffmpeg would transcode; simulate that by writing the
                    # dest path (the last arg) so the caller sees a real file.
                    Path(cmd[-1]).write_bytes(b"normalizedmp4payload")
                return mock.Mock(returncode=0, stderr="")

            env = {"KAGGLE_USERNAME": "u", "KAGGLE_API_TOKEN": "t", "HF_TOKEN": "hf",
                  "MOTIONFORGE_DIR": str(mf)}
            with mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(videogen.subprocess, "run", fake_run):
                out = videogen.generate("https://x/img.jpg", "she smiles",
                                       length_s=5.0, steps=4, seed=42,
                                       out_dir=str(out_dir))

            self.assertTrue(Path(out).exists())
            self.assertEqual(Path(out).read_bytes(), b"normalizedmp4payload")
            # Four subprocess calls: prepare_kernel, kaggle push, poll_kaggle, then
            # the ffmpeg frame-rate normalize before handing the mp4 back.
            self.assertEqual(len(calls), 4)
            self.assertIn("prepare_kernel.py", calls[0]["cmd"][-1])
            self.assertEqual(calls[1]["cmd"][:3], ["kaggle", "kernels", "push"])
            self.assertIn("poll_kaggle.py", calls[2]["cmd"][-1])
            self.assertEqual(calls[3]["cmd"][0], "ffmpeg")
            self.assertIn("-r", calls[3]["cmd"])
            # motionforge scripts read IMAGE_URL/PROMPT/LENGTH_S/STEPS from env.
            passed = calls[0]["env"]
            self.assertEqual(passed["IMAGE_URL"], "https://x/img.jpg")
            self.assertEqual(passed["PROMPT"], "she smiles")
            self.assertEqual(passed["LENGTH_S"], "5.0")
            self.assertEqual(passed["STEPS"], "4")
            self.assertEqual(passed["IMAGE_SEED"], "42")
            # Kaggle CLI wants KAGGLE_KEY; motionforge's secret name is
            # KAGGLE_API_TOKEN -- videogen aliases one to the other.
            self.assertEqual(passed["KAGGLE_KEY"], "t")

    def test_missing_motionforge_dir_raises(self):
        with mock.patch.dict(os.environ, {"MOTIONFORGE_DIR": "/nope/does/not/exist"},
                             clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                videogen.generate("https://x/img.jpg", "hi")
        self.assertIn("motionforge script missing", str(ctx.exception))

    def test_final_mp4_missing_after_poll_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            mf = self._setup_fake_motionforge(tmp)
            # Kaggle "succeeded" but never wrote the file: real-world failure mode
            # where poll_kaggle exits 0 on some transitional states without an mp4.
            (mf / "kaggle_output" / "final.mp4").unlink()
            with mock.patch.dict(os.environ, {"MOTIONFORGE_DIR": str(mf)},
                                 clear=True), \
                 mock.patch.object(videogen.subprocess, "run",
                                  lambda *a, **kw: mock.Mock(returncode=0)):
                with self.assertRaises(RuntimeError) as ctx:
                    videogen.generate("https://x/img.jpg", "hi")
        self.assertIn("final.mp4", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
