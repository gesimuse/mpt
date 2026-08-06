#!/usr/bin/env python3
"""Regression tests for the autopilot flow.

No network, no API keys, no MoneyPrinterTurbo checkout: every external call is stubbed.
The point is to prove the wiring still holds -- that a run reaches MPT with the right
arguments, that each failing dependency degrades instead of killing the run, and that
DRY_RUN never uploads.

Run: python3 test_pipeline.py
"""
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("NIM_API_KEY", "test-key")
os.environ.pop("DRY_RUN", None)

import autopilot  # noqa: E402
import imageslides  # noqa: E402
import buffer  # noqa: E402
import civitai  # noqa: E402
import critic  # noqa: F401,E402
import questions  # noqa: E402
import llm  # noqa: E402
import research  # noqa: E402
import sdgen  # noqa: E402

SCRIPT = ("Two waiters just sold the same table twice. Picture a Friday night restaurant "
          "with one table left and a paper reservation book. Each waiter is a thread. The "
          "last table is the shared resource. The book is your database row. A lock is the "
          "rule that only one waiter touches it. Skip it and two customers are charged for "
          "one seat, which is a duplicate payment under load. Lock the row.")

NICHE = {
    "id": "codeaz", "name": "CodeAZ", "voice": "en-GB-RyanNeural-Male",
    "topic_prompt": "topics", "hashtags": "#programming #coding",
    "youtube_tags": ["programming"], "videos_per_run": 1, "video_mode": "stock",
}


class Response:
    def __init__(self, status=200, payload=None, text="{}"):
        self.status_code, self._payload, self.text = status, payload or {}, text
        self.ok = status < 400
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)


def chat_payload(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


class NimClientTest(unittest.TestCase):
    def test_reasoning_model_null_content_grows_budget(self):
        """gpt-oss returns content: null when reasoning eats the token budget."""
        budgets = []

        def post(url, **kw):
            budgets.append(kw["json"]["max_tokens"])
            if len(budgets) < 3:
                return Response(200, {"choices": [{"finish_reason": "length", "message": {
                    "content": None, "reasoning_content": "thinking"}}]})
            return Response(200, chat_payload("done"))

        with mock.patch.object(llm.requests, "post", post), mock.patch.object(llm.time, "sleep"):
            self.assertEqual(llm.nim_chat("s", "u"), "done")
        self.assertEqual(budgets, [512, 2048, 8000])

    def test_unknown_model_fails_immediately(self):
        calls = []

        def post(url, **kw):
            calls.append(1)
            return Response(404, text='{"detail":"not found"}')

        with mock.patch.dict(os.environ, {"NIM_API_KEY": "a"}, clear=True), \
             mock.patch.object(llm.requests, "post", post), \
             mock.patch.object(llm.requests, "get", lambda *a, **k: Response(200, {"data": []})), \
             mock.patch.object(llm.time, "sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                llm.nim_chat("s", "u")
        self.assertEqual(len(calls), 1, "a 404 is permanent and must not be retried")
        self.assertIn("does not serve model", str(ctx.exception))

    def test_timeout_is_retried(self):
        import requests
        calls = []

        def post(url, **kw):
            calls.append(1)
            if len(calls) < 2:
                raise requests.Timeout("read timed out")
            return Response(200, chat_payload("recovered"))

        with mock.patch.object(llm.requests, "post", post), mock.patch.object(llm.time, "sleep"):
            self.assertEqual(llm.nim_chat("s", "u"), "recovered")

    def test_json_helper_survives_prose_around_the_object(self):
        payload = chat_payload('Sure, here you go:\n```json\n{"topic": "x"}\n```')
        with mock.patch.object(llm.requests, "post", lambda *a, **k: Response(200, payload)):
            self.assertEqual(llm.nim_json("s", "u"), {"topic": "x"})


class ProviderFailoverTest(unittest.TestCase):
    """NIM read-timing-out three times used to kill the run. With a second key set it
    should cost a log line instead."""

    def test_falls_over_to_the_next_provider(self):
        import requests as rq
        seen = []

        def post(url, **kw):
            seen.append(url)
            if "nvidia" in url:
                raise rq.Timeout("read timed out")
            return Response(200, chat_payload("from the fallback"))

        with mock.patch.dict(os.environ, {"NIM_API_KEY": "a", "GROQ_API_KEY": "b"}, clear=True), \
             mock.patch.object(llm.requests, "post", post), \
             mock.patch.object(llm.time, "sleep"):
            self.assertEqual(llm.nim_chat("s", "u"), "from the fallback")
        self.assertTrue(any("nvidia" in u for u in seen))
        self.assertTrue(any("groq" in u for u in seen))

    def test_chain_order_and_opt_in(self):
        with mock.patch.dict(os.environ, {"NIM_API_KEY": "a", "OPENROUTER_API_KEY": "b",
                                          "GROQ_API_KEY": "c"}, clear=True), \
             mock.patch.object(llm, "openrouter_is_free", lambda m: True):
            self.assertEqual([p.name for p in llm.providers()],
                             ["nim", "groq", "openrouter"],
                             "Groq outranks OpenRouter: 14,400 free requests/day vs 50")
        with mock.patch.dict(os.environ, {"NIM_API_KEY": "a"}, clear=True):
            self.assertEqual([p.name for p in llm.providers()], ["nim"])

    def test_paid_openrouter_model_is_refused(self):
        """OpenRouter serves paid and free models through one key, so a wrong model id
        is the only way this pipeline could start spending money."""
        catalogue = {"data": [
            {"id": "free/model:free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "paid/model", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
        ]}
        with mock.patch.object(llm.requests, "get",
                               lambda *a, **k: Response(200, catalogue)):
            self.assertTrue(llm.openrouter_is_free("free/model:free"))
            self.assertFalse(llm.openrouter_is_free("paid/model"))
            self.assertFalse(llm.openrouter_is_free("retired/model:free"))
        # if pricing cannot be checked at all, refuse rather than assume
        with mock.patch.object(llm.requests, "get",
                               mock.Mock(side_effect=RuntimeError("offline"))):
            self.assertFalse(llm.openrouter_is_free("anything:free"))
        with mock.patch.object(llm.requests, "get",
                               lambda *a, **k: Response(200, catalogue)):
            with mock.patch.dict(os.environ, {"NIM_API_KEY": "a", "OPENROUTER_API_KEY": "b",
                                              "OPENROUTER_MODEL": "paid/model"}, clear=True):
                self.assertEqual([p.name for p in llm.providers()], ["nim"])
            with mock.patch.dict(os.environ, {"NIM_API_KEY": "a", "OPENROUTER_API_KEY": "b",
                                              "OPENROUTER_MODEL": "free/model:free"}, clear=True):
                self.assertEqual([p.name for p in llm.providers()], ["nim", "openrouter"])

    def test_default_openrouter_model_is_a_free_id(self):
        self.assertTrue(llm.OPENROUTER_DEFAULT_MODEL.endswith(":free"))

    def test_placeholder_keys_are_ignored(self):
        with mock.patch.dict(os.environ, {"NIM_API_KEY": "real", "GROQ_API_KEY": "xxxx"},
                             clear=True):
            self.assertEqual([p.name for p in llm.providers()], ["nim"])

    def test_every_provider_down_raises(self):
        import requests as rq
        with mock.patch.dict(os.environ, {"NIM_API_KEY": "a", "GROQ_API_KEY": "b"}, clear=True), \
             mock.patch.object(llm.requests, "post",
                               mock.Mock(side_effect=rq.Timeout("down"))), \
             mock.patch.object(llm.time, "sleep"):
            with self.assertRaises(Exception):
                llm.nim_chat("s", "u")


class ScriptQualityTest(unittest.TestCase):
    def test_drift_terms_are_per_niche(self):
        """A finance script must not be judged against programming vocabulary."""
        finance = {"id": "moneymech", "drift_terms": ["crypto", "day trading", "get rich"]}
        hype = ("Your savings account is like a bucket. Put crypto in it and day trading "
                "will get rich fast, guaranteed.")
        self.assertTrue(autopilot._drift_terms(hype, "Compound interest", finance))
        # the programming list must not fire on a finance topic
        threads = "Each teller is a thread holding a lock on the shared resource."
        self.assertEqual(autopilot._drift_terms(threads, "Compound interest", finance), [])

    def test_dormant_niche_is_skipped_not_failed(self):
        """A configured niche whose channel does not exist yet must not fail the run."""
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(autopilot.niche_is_ready({"id": "moneymech"}))
        with mock.patch.dict(os.environ, {"YT_REFRESH_TOKEN_MONEYMECH": "1//real"}):
            self.assertTrue(autopilot.niche_is_ready({"id": "moneymech"}))
        with mock.patch.dict(os.environ, {"YT_REFRESH_TOKEN_MONEYMECH": "xxxx"}):
            self.assertFalse(autopilot.niche_is_ready({"id": "moneymech"}))

    def test_every_configured_niche_has_the_required_fields(self):
        niches = json.loads((Path(__file__).parent / "niches.json").read_text())["niches"]
        for n in niches:
            self.assertIn("id", n)
            self.assertIn("name", n)
            self.assertIn("hashtags", n)
            if n.get("content_type") == "images":
                for field in ("images_per_video", "min_images", "captions", "ai_disclosure"):
                    self.assertIn(field, n, f"{n['id']} is missing {field}")
                continue
            for field in ("topic_prompt", "voice", "video_mode"):
                self.assertIn(field, n, f"{n.get('id')} is missing {field}")
            self.assertTrue(n["voice"].endswith(("-Male", "-Female")),
                            f"{n['id']} voice must carry the gender suffix MPT expects")
            self.assertIn("Neural", n["voice"], f"{n['id']} voice must be a real edge-tts id")

    def test_offtopic_concurrency_terms_are_rejected(self):
        drifted = ("A drive-thru is like a program. The cashier is the main thread and the "
                   "menu board is the API. The last car is the shared resource, which causes "
                   "a deadlock under load, and that is a race condition you must lock.")
        terms = autopilot._drift_terms(drifted, "Loops explained like a coffee shop drive-thru")
        self.assertGreater(len(terms), autopilot.MAX_DRIFT, f"drift not detected: {terms}")

    def test_concurrency_terms_allowed_when_topic_is_concurrency(self):
        self.assertEqual(
            autopilot._drift_terms(SCRIPT, "Race conditions, explained by two waiters"), [])

    def _approve(self):
        return mock.patch.object(critic, "review",
                                 lambda *a, **k: ("publish", {"hook": 9}, [], ""))

    def test_regenerates_until_on_topic(self):
        bad = ("A drive-thru is like a program. The cashier is the main thread. The menu is "
               "the API. The last car is the shared resource and causes a deadlock. " * 3)
        good = ("A drive-thru is like a loop. Each car is an iteration. The lane is the "
                "condition. The window is the loop body. Miss the exit and it is an infinite "
                "loop, so the queue never clears and orders stall. Loops repeat until done. " * 4)
        replies = [bad, bad, good]
        with mock.patch.object(autopilot, "nim_chat", lambda *a, **k: replies.pop(0)), \
             self._approve():
            out = autopilot.generate_script("Loops explained like a drive-thru", NICHE)
        self.assertIn("iteration", out)
        self.assertEqual(autopilot._drift_terms(out, "Loops explained like a drive-thru"), [])

    def test_thinking_block_never_reaches_narration(self):
        cleaned = autopilot._clean_script("<think>plan: map waiter to thread</think>Each waiter is a thread.")
        self.assertNotIn("plan", cleaned)
        self.assertTrue(cleaned.startswith("Each waiter"))


USED_TOPICS = [
    "Loops explained like a coffee shop drive-thru",
    "Functions like a recipe book",
    "Variables like labeled kitchen jars",
    "Garbage collection explained by city recycling trucks on scheduled routes",
    "Deadlocks explained by two cars stuck on intersecting one-way bridges",
    "Why lazy functions stay idle, explained by a restaurant's order ticket system",
]


class TopicNoveltyTest(unittest.TestCase):
    """A researched topic once slipped through that was the same video as an earlier one
    in different words. Exact-match checking cannot see that."""

    def test_reworded_repeats_are_caught(self):
        for topic in [
            # the real regression: 77% identical phrasing to the lazy-functions video
            "Why your async function stalls, explained by a restaurant reservation system",
            "Loops explained like a coffee shop drive thru",       # punctuation only
            "Functions explained like a recipe book",              # one word added
            "Garbage collection, explained by recycling trucks on their routes",
        ]:
            clash, why = autopilot.too_similar(topic, USED_TOPICS)
            self.assertIsNotNone(clash, f"missed repeat: {topic}")

    def test_same_concept_with_a_new_analogy_is_still_a_repeat(self):
        for topic in [
            "Loops explained like a subway turnstile queue",
            "Garbage collection, explained by a hotel housekeeping rota",
            "Deadlocks, explained by two people in a narrow doorway",
        ]:
            clash, why = autopilot.too_similar(topic, USED_TOPICS)
            self.assertIsNotNone(clash, f"swapping the analogy is not a new video: {topic}")
            self.assertIn("concept", why)

    def test_genuinely_new_topics_pass(self):
        for topic in [
            "Database indexes, explained like a library card catalogue",
            "Cache invalidation, explained by a stale specials board",
            "Hash collisions, explained by two guests with the same locker key",
            "Why your API rate limits, explained by a nightclub door policy",
            "Type inference, explained like a librarian's catalog system",
            "Event bubbling, explained by a postal mail sorting office",
        ]:
            clash, why = autopilot.too_similar(topic, USED_TOPICS)
            self.assertIsNone(clash, f"false positive: {topic} vs {clash} ({why})")

    def test_topic_must_come_from_a_harvested_question(self):
        """Ideas come from humans: with no questions available the run stops rather than
        letting the model invent a subject."""
        niche = {**NICHE, "subreddits": ["learnprogramming"]}
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(questions, "harvest", lambda n: []):
            with self.assertRaises(RuntimeError) as ctx:
                autopilot.pick_topic(niche, USED_TOPICS)
        self.assertIn("no unanswered questions", str(ctx.exception))

    def test_already_answered_questions_are_filtered(self):
        harvested = [
            {"id": "reddit:1", "title": "Why does my loop run forever?", "score": 9,
             "num_comments": 4, "url": "u1"},
            {"id": "reddit:2", "title": "How does a hash map find a key so fast?",
             "score": 12, "num_comments": 6, "url": "u2"},
        ]
        left = questions.unused(harvested, {"reddit:1"}, [], autopilot.too_similar)
        self.assertEqual([q["id"] for q in left], ["reddit:2"])

    def test_question_matching_an_existing_video_is_filtered(self):
        harvested = [{"id": "reddit:9", "title": "Functions like a recipe book",
                      "score": 9, "num_comments": 4, "url": "u"}]
        self.assertEqual(questions.unused(harvested, set(), USED_TOPICS,
                                          autopilot.too_similar), [])


class HookTest(unittest.TestCase):
    """The topic is itself a question, so the model kept opening by restating it. The
    critic scored that 6 forever without ever forcing a rewrite."""

    def test_weak_openers_are_caught(self):
        for opener in [
            "When normalizing RGB values, should you divide by 255 or 256?",
            "Have you ever wondered how a hash map works? It uses buckets.",
            "In this video we look at loops.",
            "Did you know your cache can lie?",
            "Let's talk about closures.",
            # hedged abstractions: statements that promise nothing
            "Using the wrong divisor can lead to inaccurate color representation.",
            "Choosing the right divisor is important for color accuracy.",
            "Race conditions can be tricky to debug.",
        ]:
            self.assertIsNotNone(autopilot.weak_hook(opener), opener)

    def test_real_hooks_pass(self):
        for opener in [
            "Divide by 256 and every colour in your app shifts one shade dark.",
            "Two waiters just sold the same table twice.",
            "Your cache is lying to you right now, and the fix is one line.",
            "Your loop never exits because the counter resets on every pass.",
        ]:
            self.assertIsNone(autopilot.weak_hook(opener), opener)

    def test_weak_opener_is_repaired_by_the_dedicated_hook_call(self):
        """One call cannot answer the question, map the analogy, stay accurate and land a
        hook -- the hook is what it drops. So the hook gets its own call."""
        sent = []

        def chat(system, user, **kw):
            sent.append(system)
            if "opening line" in system:
                return "Divide by 256 and every colour comes out one shade dark."
            return "Should you divide by 255 or 256? " + "word " * 120

        with mock.patch.object(autopilot, "nim_chat", chat), \
             mock.patch.object(critic, "review",
                               lambda *a, **k: ("publish", {"hook": 9}, [], "")):
            out = autopilot.generate_script("Divide RGB by 255 or 256?", NICHE)
        self.assertTrue(out.startswith("Divide by 256"), f"opener not replaced: {out[:60]}")
        self.assertIsNone(autopilot.weak_hook(out))
        self.assertTrue(any("opening line" in x for x in sent), "hook call never happened")

    def test_hook_rewrite_keeps_the_body_intact(self):
        script = "Should you divide by 255 or 256? The ruler has 256 marks. Divide by 255."
        with mock.patch.object(autopilot, "nim_chat",
                               lambda *a, **k: "Every colour comes out one shade dark."):
            out = autopilot._rewrite_hook(script, "topic")
        self.assertTrue(out.startswith("Every colour comes out one shade dark."))
        self.assertIn("The ruler has 256 marks.", out)

    def test_hook_rewrite_gives_up_rather_than_ruining_the_script(self):
        script = "Should you divide by 255 or 256? The ruler has 256 marks."
        with mock.patch.object(autopilot, "nim_chat", lambda *a, **k: "Have you ever wondered?"):
            out = autopilot._rewrite_hook(script, "topic", attempts=2)
        self.assertEqual(out, script, "a bad candidate must not replace the opener")


class CriticTest(unittest.TestCase):
    """A model asked to write and approve its own work approves nearly everything, so the
    critic is a separate call that can send a script back."""

    def test_low_score_forces_a_revision_even_if_the_verdict_says_publish(self):
        payload = {"scores": {"hook": 3, "accuracy": 9}, "verdict": "publish",
                   "problems": ["generic opener"], "fix": "Open with the failure."}
        with mock.patch.object(critic, "nim_json", lambda *a, **k: payload):
            verdict, scores, problems, fix = critic.review("t", "s")
        self.assertEqual(verdict, "revise")

    def test_loop_stops_as_soon_as_it_passes(self):
        drafts = []

        def write(feedback):
            drafts.append(feedback)
            return "draft " + str(len(drafts))

        verdicts = [("revise", {"hook": 4}, ["weak hook"], "open with the failure"),
                    ("publish", {"hook": 9}, [], "")]
        with mock.patch.object(critic, "review", lambda *a, **k: verdicts.pop(0)):
            script, verdict, scores = critic.refine("topic", write)
        self.assertEqual(verdict, "publish")
        self.assertEqual(len(drafts), 2)
        self.assertEqual(drafts[0], None, "first draft gets no feedback")
        self.assertIn("open with the failure", drafts[1], "the fix must reach the writer")

    def test_never_passing_returns_the_best_attempt_flagged_as_revise(self):
        with mock.patch.object(critic, "review",
                               lambda *a, **k: ("revise", {"hook": 4}, [], "sharper")):
            script, verdict, scores = critic.refine("topic", lambda f: "draft", rounds=2)
        self.assertEqual(verdict, "revise")

    def test_a_failing_critic_cannot_approve(self):
        with mock.patch.object(critic, "nim_json", mock.Mock(side_effect=RuntimeError("down"))):
            verdict, scores, problems, fix = critic.review("t", "s")
        self.assertEqual(verdict, "revise")

    def test_script_that_never_passes_is_not_published(self):
        with mock.patch.object(autopilot, "nim_chat", lambda *a, **k: "word " * 120), \
             mock.patch.object(critic, "review",
                               lambda *a, **k: ("revise", {"hook": 3}, ["weak"], "fix it")):
            with self.assertRaises(RuntimeError) as ctx:
                autopilot.generate_script("topic", NICHE)
        self.assertIn("never passed review", str(ctx.exception))


class QuestionHarvestTest(unittest.TestCase):
    def test_only_real_questions_survive(self):
        keep = [{"title": "Why does my loop run forever?"},
                {"title": "How does a hash map find a key so fast"},
                {"title": "Can someone explain closures"}]
        drop = [{"title": "Show HN: my new database"},
                {"title": "What laptop should I buy for coding"},   # off-topic shape
                {"title": "Rust 2.0 released"},
                {"title": "Should I quit my job to learn programming"}]
        for q in keep:
            self.assertTrue(questions.looks_like_a_question(q), q["title"])
        for q in drop:
            self.assertFalse(questions.looks_like_a_question(q), q["title"])

    def test_low_engagement_questions_are_dropped(self):
        posts = [{"title": "Why does my loop run forever?", "score": 0, "num_comments": 0,
                  "id": "a"},
                 {"title": "How does a hash map work?", "score": 5, "num_comments": 4,
                  "id": "b"}]
        with mock.patch.object(questions.research, "fetch_subreddit", lambda *a, **k: posts), \
             mock.patch.object(questions.research, "fetch_ask_hn", lambda *a, **k: []):
            got = questions.harvest({"id": "x", "subreddits": ["s"]})
        self.assertEqual([q["id"] for q in got], ["b"])

    def test_selection_must_come_from_the_shortlist(self):
        qs = [{"title": "Why does my loop run forever?", "url": "u", "id": "a"}]
        with mock.patch.object(questions, "nim_json",
                               lambda *a, **k: {"index": 7, "topic": "Something"}):
            with self.assertRaises(RuntimeError):
                questions.choose({"name": "n"}, qs)


class ResearchTest(unittest.TestCase):
    def test_arctic_shift_parses_and_filters(self):
        payload = {"data": [
            {"title": "Real post", "selftext": "body", "score": 42, "num_comments": 7},
            {"title": "Pinned", "selftext": "", "score": 10, "num_comments": 1, "stickied": True},
            {"title": "", "selftext": "", "score": 5, "num_comments": 0},
        ]}
        with mock.patch.object(research, "_get", lambda *a, **k: Response(200, payload)):
            posts = research.fetch_subreddit_arctic("learnprogramming")
        self.assertEqual([p["title"] for p in posts], ["Real post"])

    def test_all_sources_down_raises_so_caller_can_fall_back(self):
        niche = {"id": "codeaz", "name": "CodeAZ", "subreddits": ["x"], "hn_queries": [""],
                 "stackexchange_tags": ["python"]}
        boom = mock.Mock(side_effect=RuntimeError("down"))
        with mock.patch.object(research, "fetch_subreddit", boom), \
             mock.patch.object(research, "fetch_hn", boom), \
             mock.patch.object(research, "fetch_stackexchange", boom), \
             mock.patch.object(research.time, "sleep"):
            with self.assertRaises(RuntimeError):
                research.research_topic(niche, [])


class StaticCheckTest(unittest.TestCase):
    def test_no_undefined_names(self):
        """A NameError in write_mpt_config once killed a run after the script had already
        been generated -- it only fires when MPT is actually invoked, which no stubbed
        test reaches. pyflakes catches that class of bug without executing anything."""
        try:
            import pyflakes
            del pyflakes
        except ImportError:
            self.skipTest("pyflakes not installed (pip install pyflakes)")
        files = ["autopilot.py", "llm.py", "research.py"]
        r = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                           cwd=Path(__file__).parent, capture_output=True, text=True)
        undefined = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
        self.assertEqual(undefined, [], "undefined names:\n" + "\n".join(undefined))


class ConfigTest(unittest.TestCase):
    def test_written_config_has_model_voice_and_keys(self):
        """write_mpt_config interpolates module-level constants; exercise it for real."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "MPT_DIR", Path(tmp)), \
                 mock.patch.dict(os.environ, {"PEXELS_API_KEY": "pk", "PIXABAY_API_KEY": "xk",
                                              "NIM_API_KEY": "nk"}):
                autopilot.write_mpt_config(NICHE, "pexels")
            cfg = (Path(tmp) / "config.toml").read_text()
        self.assertIn('voice_name = "en-GB-RyanNeural-Male"', cfg)
        self.assertIn(f'openai_model_name = "{autopilot.NIM_MODEL}"', cfg)
        self.assertIn('video_source = "pexels"', cfg)
        self.assertIn('pexels_api_keys = ["pk"]', cfg)
        self.assertNotIn("{NIM", cfg, "every placeholder must be interpolated")

    def test_placeholder_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "MPT_DIR", Path(tmp)), \
                 mock.patch.dict(os.environ, {"PEXELS_API_KEY": "xxxx", "PIXABAY_API_KEY": "",
                                              "NIM_API_KEY": "nk"}):
                autopilot.write_mpt_config(NICHE, "pexels")
                cfg = (Path(tmp) / "config.toml").read_text()
                self.assertIn('pexels_api_keys = [""]', cfg)
                with self.assertRaises(RuntimeError):
                    autopilot.pick_sources(NICHE)


class ImageSlideshowTest(unittest.TestCase):
    AIBEAUTY = {"id": "aibeauty", "content_type": "images",
                "hashtags": "#aiart", "ai_disclosure": "Created with AI. Not a real person.",
                "captions": ["Slow mornings."], "scenes": ["a street"], "styles": ["35mm"],
                "outfits": ["wearing a wool coat and jeans"]}

    def test_swimwear_and_lingerie_are_not_blocked(self):
        """The policy line is nudity, not how much skin an outfit shows."""
        self.assertTrue(any("swimsuit" in o for o in imageslides.DEFAULT_OUTFITS))
        self.assertNotIn("swimwear", imageslides.NEGATIVE_HARD)
        self.assertNotIn("lingerie", imageslides.NEGATIVE_HARD)

    def test_formula_reference_builds_from_the_niches_own_scenes_and_styles(self):
        """The fallback used only when CivitAI cannot be reached at all."""
        _, reference = imageslides._formula_reference(self.AIBEAUTY)
        self.assertIn("a street", reference["prompt"])
        self.assertEqual(reference["negative_prompt"], "")

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

        def fake_decide(query, prompt_filter=None):
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

    @staticmethod
    def _fake_decide(query, prompt_filter=None):
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
            approved = imageslides.generate(self.AIBEAUTY, workdir=tmp)
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
            approved = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertEqual(len(approved), 4)
        self.assertEqual(filter_results, [], "both rounds must have run")

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

    def test_supervisor_can_be_disabled_for_local_testing(self):
        fake_paths = [Path(f"/tmp/img_{i}.png") for i in range(3)]
        with mock.patch.dict(os.environ, {"SUPERVISOR_ENABLED": "0"}), \
             mock.patch.object(civitai, "decide_reference", self._fake_decide), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: fake_paths), \
             tempfile.TemporaryDirectory() as tmp:
            approved = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertEqual(approved, fake_paths)

    def test_generate_falls_back_to_formula_reference_when_civitai_unavailable(self):
        with mock.patch.object(civitai, "decide_reference",
                               mock.Mock(side_effect=RuntimeError("down"))), \
             mock.patch.object(imageslides.sdgen, "generate_batch",
                               lambda *a, **k: [Path(f"/tmp/img_{i}.png") for i in range(3)]), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            approved = imageslides.generate(self.AIBEAUTY, workdir=tmp)
        self.assertEqual(len(approved), 3)


class FakePipe:
    """Stands in for a diffusers pipeline: callable, returns an object with .images[0]."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, **kw):
        self.calls.append((prompt, kw))
        image = mock.Mock()
        image.save = lambda dest: Path(dest).write_bytes(b"fake")
        result = mock.Mock()
        result.images = [image]
        return result


class SdgenTest(unittest.TestCase):
    def test_every_preset_has_a_repo_id(self):
        for key, (repo, lora, name) in sdgen.MODELS.items():
            self.assertTrue(repo, key)

    @staticmethod
    def _passthrough_encode(pipe, arch, prompt, negative):
        return {"prompt": prompt, "negative_prompt": negative}

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

    def test_adult_age_is_kept(self):
        for ok in ("RAW photo, 26 y.o woman in dress", "30 y.o european man",
                  "professional photo, 45 years old woman"):
            self.assertIsNotNone(civitai._usable({"prompt": ok}), ok)

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

    AIBEAUTY = {"id": "aibeauty", "outfits": ["wearing a red gown"],
               "min_images": 1, "images_per_video": 2}

    def _prompts_for_reference(self, prompt_text):
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured["prompts"] = prompts
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        with mock.patch.object(civitai, "decide_reference", lambda q, prompt_filter=None: (
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
        niche = {**self.AIBEAUTY, "mood": ["sultry confident gaze"]}
        captured = {}

        def fake_generate_batch(prompts, workdir, **kw):
            captured["prompts"] = prompts
            return [Path(f"/tmp/i{i}.png") for i in range(len(prompts))]

        with mock.patch.object(civitai, "decide_reference", lambda q, prompt_filter=None: (
                {"model_id": 1, "version_id": 2, "name": "X"},
                {"prompt": "chef in a kitchen, studio light", "negative_prompt": ""})), \
             mock.patch.object(imageslides.sdgen, "generate_batch", fake_generate_batch), \
             mock.patch.object(imageslides.supervisor, "filter_images", lambda paths: paths), \
             tempfile.TemporaryDirectory() as tmp:
            imageslides.generate(niche, workdir=tmp)
        self.assertIn("sultry confident gaze", captured["prompts"][0])


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


class CivitaiDecideReferenceTest(unittest.TestCase):
    """Step 1/2 of the "search for a good photo, decide on it, download its model"
    flow: search once, walk ranked candidates, commit to the first that both resolves
    to a downloadable checkpoint and has a real showcase prompt worth using.

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

    def test_picks_first_candidate_with_a_usable_showcase_prompt(self):
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


class BufferCarouselTest(unittest.TestCase):
    def test_publish_photos_sends_one_asset_per_image(self):
        sent = {}

        def gql(query, variables=None):
            if "account" in query:
                return {"account": {"organizations": [{"id": "org1"}]}}
            if "channels" in query:
                return {"channels": [{"id": "ch1", "service": "tiktok"}]}
            sent["input"] = variables["input"]
            return {"createPost": {"__typename": "PostActionSuccess",
                                   "post": {"id": "post1"}}}

        with mock.patch.object(buffer, "gql", gql):
            post_id = buffer.publish_photos(
                None, "caption #tag", image_urls=["u1", "u2", "u3"])
        self.assertEqual(post_id, "post1")
        self.assertEqual(sent["input"]["assets"],
                         [{"image": {"url": u}} for u in ("u1", "u2", "u3")])

    def test_single_image_is_refused(self):
        with mock.patch.object(buffer, "tiktok_channel", lambda niche_id=None: ("ch1", "o")):
            with self.assertRaises(RuntimeError) as ctx:
                buffer.publish_photos(None, "caption", image_urls=["u1"])
        self.assertIn("at least 2", str(ctx.exception))


class RenderedVideoCheckTest(unittest.TestCase):
    """A three second clip carrying only the opening line once reached a channel, and
    nothing between the renderer and the upload noticed."""

    SCRIPT = "word " * 150   # ~58s of narration

    def test_truncated_render_is_refused(self):
        with mock.patch.object(autopilot, "video_duration", lambda p: 3.0), \
             mock.patch.object(autopilot.subprocess, "run",
                               lambda *a, **k: mock.Mock(stdout="video\naudio\n")):
            with self.assertRaises(RuntimeError) as ctx:
                autopilot.check_rendered_video("/tmp/v.mp4", self.SCRIPT)
        self.assertIn("cut short", str(ctx.exception))

    def test_silent_video_is_refused(self):
        with mock.patch.object(autopilot, "video_duration", lambda p: 60.0), \
             mock.patch.object(autopilot.subprocess, "run",
                               lambda *a, **k: mock.Mock(stdout="video\n")):
            with self.assertRaises(RuntimeError) as ctx:
                autopilot.check_rendered_video("/tmp/v.mp4", self.SCRIPT)
        self.assertIn("no audio", str(ctx.exception))

    def test_a_healthy_render_passes(self):
        with mock.patch.object(autopilot, "video_duration", lambda p: 58.0), \
             mock.patch.object(autopilot.subprocess, "run",
                               lambda *a, **k: mock.Mock(stdout="video\naudio\n")):
            self.assertEqual(autopilot.check_rendered_video("/tmp/v.mp4", self.SCRIPT), 58.0)

    def test_short_but_proportionate_video_still_fails_the_floor(self):
        """Even a proportionate render is refused below the absolute floor: a 15s short
        is not what this pipeline is for."""
        with mock.patch.object(autopilot, "video_duration", lambda p: 15.0), \
             mock.patch.object(autopilot.subprocess, "run",
                               lambda *a, **k: mock.Mock(stdout="video\naudio\n")):
            with self.assertRaises(RuntimeError):
                autopilot.check_rendered_video("/tmp/v.mp4", "word " * 40)


class RenderTest(unittest.TestCase):
    def test_mpt_command_has_voice_and_subtitle_flags(self):
        captured = {}

        def run(cmd, **kw):
            captured["cmd"] = cmd
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(autopilot.subprocess, "run", run), \
             mock.patch.object(autopilot.glob, "glob", return_value=["/tmp/final-1.mp4"]), \
             mock.patch.object(autopilot.os.path, "getmtime", return_value=9e9):
            autopilot.generate_video("topic", NICHE, "pexels", script="s", terms="t")

        cmd = captured["cmd"]
        # The voice regression: MPT's CLI default overrides config.toml, so this flag
        # must always be present or narration reverts to Chinese.
        self.assertIn("--voice-name", cmd)
        self.assertEqual(cmd[cmd.index("--voice-name") + 1], "en-GB-RyanNeural-Male")
        for flag in ("--video-script", "--video-terms", "--font-name", "--font-size",
                     "--subtitle-position", "--custom-position", "--stroke-width"):
            self.assertIn(flag, cmd)
        # MPT's bundled background music draws YouTube copyright claims.
        self.assertEqual(cmd[cmd.index("--bgm-type") + 1], "none")
        self.assertEqual(cmd[cmd.index("--bgm-volume") + 1], "0")
        self.assertEqual(cmd[cmd.index("--font-name") + 1], "BeVietnamPro-Bold.ttf")

    def test_falls_back_to_other_stock_source(self):
        tried = []

        def gen(topic, niche, source, script=None, terms=None):
            tried.append(source)
            if len(tried) == 1:
                raise RuntimeError("MPT failed:\nstage=materials, error: failed to "
                                   "download video materials from pixabay")
            return "/tmp/final.mp4"

        with mock.patch.dict(os.environ, {"PEXELS_API_KEY": "a", "PIXABAY_API_KEY": "b"}), \
             mock.patch.object(autopilot, "generate_video", gen), \
             mock.patch.object(autopilot, "write_mpt_config", lambda *a: None):
            self.assertEqual(autopilot.render_with_fallback("t", NICHE, "s", "x"), "/tmp/final.mp4")
        self.assertEqual(len(tried), 2)

    def test_real_mpt_failure_is_not_retried(self):
        tried = []

        def gen(topic, niche, source, script=None, terms=None):
            tried.append(source)
            raise RuntimeError("MPT failed:\nffmpeg: invalid codec")

        with mock.patch.dict(os.environ, {"PEXELS_API_KEY": "a", "PIXABAY_API_KEY": "b"}), \
             mock.patch.object(autopilot, "generate_video", gen), \
             mock.patch.object(autopilot, "write_mpt_config", lambda *a: None):
            with self.assertRaises(RuntimeError):
                autopilot.render_with_fallback("t", NICHE, "s", "x")
        self.assertEqual(len(tried), 1, "a codec error must not burn the other source")

    def test_unknown_video_mode_is_rejected(self):
        with self.assertRaises(RuntimeError):
            autopilot.render_video("t", {**NICHE, "video_mode": "slideshow"}, "s")


class YouTubeCredentialsTest(unittest.TestCase):
    """Quota is per Cloud project, so a channel with its own Google account has its own
    OAuth client -- and a refresh token only works with the client that minted it."""

    def test_per_niche_client_overrides_the_shared_one(self):
        with mock.patch.dict(os.environ, {
                "YT_CLIENT_ID": "shared", "YT_CLIENT_SECRET": "shared-secret",
                "YT_CLIENT_ID_MONEYMECH": "own", "YT_CLIENT_SECRET_MONEYMECH": "own-secret",
                "YT_REFRESH_TOKEN_MONEYMECH": "tok"}, clear=True):
            self.assertEqual(autopilot.youtube_credentials("moneymech"),
                             ("own", "own-secret", "tok"))

    def test_falls_back_to_the_shared_client(self):
        with mock.patch.dict(os.environ, {
                "YT_CLIENT_ID": "shared", "YT_CLIENT_SECRET": "shared-secret",
                "YT_REFRESH_TOKEN_CODEAZ": "tok"}, clear=True):
            self.assertEqual(autopilot.youtube_credentials("codeaz"),
                             ("shared", "shared-secret", "tok"))

    def test_placeholder_refresh_token_counts_as_absent(self):
        with mock.patch.dict(os.environ, {"YT_REFRESH_TOKEN_AIWORKS": "xxxx"}, clear=True):
            self.assertEqual(autopilot.youtube_credentials("aiworks")[2], "")


class YouTubeDisclosureTest(unittest.TestCase):
    def test_upload_declares_synthetic_media(self):
        """Model-written narration with a synthetic voice must carry YouTube's altered
        media disclosure; leaving it off is itself a monetisation risk."""
        captured = {}

        class Insert:
            def next_chunk(self):
                return None, {"id": "vid123"}

        class Videos:
            def insert(self, part, body, media_body):
                captured["body"] = body
                captured["part"] = part
                return Insert()

        class YT:
            def videos(self):
                return Videos()

        with mock.patch.dict("sys.modules", {
                "google.oauth2.credentials": mock.MagicMock(),
                "googleapiclient.discovery": mock.MagicMock(build=lambda *a, **k: YT()),
                "googleapiclient.http": mock.MagicMock(
                    MediaFileUpload=lambda *a, **k: object())}), \
             mock.patch.dict(os.environ, {"YT_CLIENT_ID": "i", "YT_CLIENT_SECRET": "s",
                                          "YT_REFRESH_TOKEN_CODEAZ": "r"}):
            autopilot.upload_youtube("/tmp/v.mp4",
                                     {"title": "T", "description": "D"}, NICHE)
        self.assertTrue(captured["body"]["status"]["containsSyntheticMedia"])
        self.assertIn("status", captured["part"])


class TikTokCaptionTest(unittest.TestCase):
    def test_caption_merges_title_description_and_hashtags(self):
        meta = {"title": "Race conditions", "description": "Two waiters, one table."}
        caption = autopilot.tiktok_caption(meta, NICHE)
        self.assertIn("Race conditions", caption)
        self.assertTrue(caption.endswith(NICHE["hashtags"]))

    def test_long_caption_keeps_hashtags(self):
        meta = {"title": "T" * 80, "description": "D" * 4000}
        caption = autopilot.tiktok_caption(meta, NICHE)
        self.assertLessEqual(len(caption), 2200)
        self.assertTrue(caption.endswith(NICHE["hashtags"]),
                        "hashtags must survive truncation")


class BufferChannelRoutingTest(unittest.TestCase):
    """Several niches share one Buffer account, so the channel must be chosen per niche
    rather than guessed -- otherwise one niche's video reaches another's audience."""

    def _gql(self, channels):
        def gql(query, variables=None):
            if "account" in query:
                return {"account": {"organizations": [{"id": "org1"}]}}
            return {"channels": channels}
        return gql

    def test_per_niche_channel_env_wins(self):
        with mock.patch.dict(os.environ, {"BUFFER_CHANNEL_ID_AIWORKS": "ch-ai"}), \
             mock.patch.object(buffer, "gql", self._gql([])):
            self.assertEqual(buffer.tiktok_channel("aiworks")[0], "ch-ai")

    def test_several_channels_without_mapping_is_an_error(self):
        channels = [{"id": "a", "name": "codeaz", "service": "tiktok"},
                    {"id": "b", "name": "aiworks", "service": "tiktok"}]
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(buffer, "gql", self._gql(channels)):
            with self.assertRaises(RuntimeError) as ctx:
                buffer.tiktok_channel("aiworks")
        self.assertIn("BUFFER_CHANNEL_ID_AIWORKS", str(ctx.exception))

    def test_single_channel_still_resolves(self):
        channels = [{"id": "only", "name": "codeaz", "service": "tiktok"}]
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(buffer, "gql", self._gql(channels)):
            self.assertEqual(buffer.tiktok_channel("codeaz")[0], "only")


class BufferTest(unittest.TestCase):
    def test_publish_sends_caption_and_ai_disclosure(self):
        sent = {}

        def gql(query, variables=None):
            if "account" in query:
                return {"account": {"organizations": [{"id": "org1"}]}}
            if "channels" in query:
                return {"channels": [{"id": "ch1", "name": "codeazorg", "service": "tiktok"}]}
            sent["input"] = variables["input"]
            return {"createPost": {"__typename": "PostActionSuccess",
                                   "post": {"id": "post1", "dueAt": None}}}

        with mock.patch.object(buffer, "gql", gql):
            post_id = buffer.publish(None, "caption #tag", title="T",
                                     video_url="https://example.com/v.mp4")
        self.assertEqual(post_id, "post1")
        inp = sent["input"]
        self.assertEqual(inp["channelId"], "ch1")
        self.assertEqual(inp["text"], "caption #tag")
        self.assertEqual(inp["assets"], [{"video": {"url": "https://example.com/v.mp4"}}])
        self.assertEqual(inp["schedulingType"], "automatic")
        # queueing it put videos days out behind an existing backlog
        self.assertEqual(inp["mode"], "shareNow")
        # TikTok requires AI-generated content to be disclosed.
        self.assertTrue(inp["metadata"]["tiktok"]["isAiGenerated"])

    def test_error_union_is_reported(self):
        def gql(query, variables=None):
            if "account" in query:
                return {"account": {"organizations": [{"id": "org1"}]}}
            if "channels" in query:
                return {"channels": [{"id": "ch1", "service": "tiktok"}]}
            return {"createPost": {"__typename": "InvalidInputError",
                                   "message": "Video could not be read from its URL."}}

        with mock.patch.object(buffer, "gql", gql):
            with self.assertRaises(RuntimeError) as ctx:
                buffer.publish(None, "c", video_url="https://example.com/bad.mp4")
        self.assertIn("could not be read", str(ctx.exception))

    def test_missing_tiktok_channel_is_explicit(self):
        def gql(query, variables=None):
            if "account" in query:
                return {"account": {"organizations": [{"id": "org1"}]}}
            return {"channels": [{"id": "c", "service": "mastodon"}]}

        with mock.patch.object(buffer, "gql", gql):
            with self.assertRaises(RuntimeError) as ctx:
                buffer.tiktok_channel()
        self.assertIn("no TikTok channel", str(ctx.exception))


class PublishDispatchTest(unittest.TestCase):
    def test_buffer_replaces_the_inbox_draft(self):
        with mock.patch.object(buffer, "enabled", lambda: True), \
             mock.patch.object(buffer, "publish",
                               lambda v, c, title=None, niche_id=None: "post1"), \
             mock.patch.object(autopilot, "upload_tiktok",
                               mock.Mock(side_effect=AssertionError("inbox must not run"))):
            ok, via, pid = autopilot.publish_tiktok("/tmp/v.mp4", {"title": "T"}, NICHE, "cap")
        self.assertEqual((ok, via, pid), (True, "buffer", "post1"))

    def test_buffer_failure_falls_back_to_inbox(self):
        def boom(*a, **k):
            raise RuntimeError("buffer down")

        with mock.patch.object(buffer, "enabled", lambda: True), \
             mock.patch.object(buffer, "publish", boom), \
             mock.patch.object(autopilot, "upload_tiktok", lambda v, m, n: "inbox1"):
            ok, via, pid = autopilot.publish_tiktok("/tmp/v.mp4", {"title": "T"}, NICHE, "cap")
        self.assertEqual((ok, via, pid), (True, "inbox", "inbox1"))

    def test_without_token_it_uses_the_inbox(self):
        with mock.patch.object(buffer, "enabled", lambda: False), \
             mock.patch.object(autopilot, "upload_tiktok", lambda v, m, n: "inbox1"):
            ok, via, pid = autopilot.publish_tiktok("/tmp/v.mp4", {"title": "T"}, NICHE, "cap")
        self.assertEqual(via, "inbox")

    def test_captions_file_skips_posts_buffer_already_captioned(self):
        import tempfile as tf
        with tf.TemporaryDirectory() as tmp:
            state = {"uploads": [
                {"ts": "t1", "niche": "codeaz", "tiktok": True, "tiktok_via": "buffer",
                 "tiktok_caption": "already published"},
                {"ts": "t2", "niche": "codeaz", "tiktok": True, "tiktok_via": "inbox",
                 "tiktok_caption": "needs pasting"},
            ]}
            with mock.patch.object(autopilot, "ROOT", Path(tmp)):
                autopilot.write_pending_captions(state)
                text = (Path(tmp) / "CAPTIONS.md").read_text()
        self.assertIn("needs pasting", text)
        self.assertNotIn("already published", text)


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


class RunImageNicheTest(unittest.TestCase):
    AIBEAUTY = {"id": "aibeauty", "content_type": "images", "hashtags": "#aiart",
                "ai_disclosure": "Created with AI.", "captions": ["Soft light."]}

    def test_dispatch_skips_without_buffer(self):
        with mock.patch.object(buffer, "enabled", lambda: False):
            # must not raise, and must not touch imageslides at all
            with mock.patch.object(imageslides, "generate",
                                   mock.Mock(side_effect=AssertionError("must not run"))):
                autopilot.run_niche(self.AIBEAUTY, {"topics": {}, "uploads": []})

    def test_real_run_publishes_a_carousel_and_records_state(self):
        fake_images = [Path(f"/tmp/i{i}.png") for i in range(5)]
        state = {"topics": {}, "uploads": []}
        with mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(buffer, "enabled", lambda: True), \
             mock.patch.object(imageslides, "generate", lambda n: fake_images), \
             mock.patch.object(buffer, "publish_photos",
                               lambda imgs, cap, title=None, niche_id=None: "post1"), \
             mock.patch.object(autopilot.os, "remove", lambda p: None), \
             mock.patch.object(autopilot, "save_state", lambda s: None):
            autopilot.run_niche(self.AIBEAUTY, state)
        entry = state["uploads"][-1]
        self.assertEqual(entry["tiktok_via"], "buffer")
        self.assertEqual(entry["tiktok_post_id"], "post1")
        self.assertIsNone(entry["youtube"])

    def test_dry_run_writes_files_and_never_publishes(self):
        fake_images = [Path(f"/tmp/i{i}.png") for i in range(5)]
        with mock.patch.object(autopilot, "DRY_RUN", True), \
             mock.patch.object(buffer, "enabled", lambda: True), \
             mock.patch.object(imageslides, "generate", lambda n: fake_images), \
             mock.patch.object(buffer, "publish_photos",
                               mock.Mock(side_effect=AssertionError("must not publish"))), \
             mock.patch.object(autopilot.shutil, "copy", lambda a, b: None), \
             tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(autopilot, "OUT_DIR", Path(tmp)):
                autopilot.run_niche(self.AIBEAUTY, {"topics": {}, "uploads": []})


class RunNicheTest(unittest.TestCase):
    """The full loop, with MPT and both upload targets stubbed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(autopilot, "ROOT", root),
            mock.patch.object(autopilot, "STATE_FILE", root / "posted.json"),
            mock.patch.object(autopilot, "OUT_DIR", root / "out"),
            mock.patch.object(autopilot, "pick_topic",
                              lambda n, u, ids=(): ("Race conditions, two waiters",
                                                   {"id": "reddit:x", "title": "why?", "url": "u"})),
            mock.patch.object(autopilot, "generate_script", lambda t, n, q=None: SCRIPT),
            mock.patch.object(autopilot, "generate_terms", lambda t, s, n: "waiter,table"),
            mock.patch.object(autopilot, "render_with_fallback",
                              lambda t, n, s, x: str(root / "video.mp4")),
            # the stub file is not a real video; the render check has its own tests
            mock.patch.object(autopilot, "check_rendered_video", lambda p, s: 60.0),
            mock.patch.object(autopilot, "make_metadata",
                              lambda t, n: {"title": "Race Conditions", "description": "desc"}),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        (root / "video.mp4").write_bytes(b"x")
        self.root = root

    def test_uploads_and_state(self):
        uploads = {}
        with mock.patch.object(autopilot, "DRY_RUN", False), \
             mock.patch.object(autopilot, "upload_youtube",
                               lambda v, m, n: (uploads.__setitem__("yt", True), "yt123")[1]), \
             mock.patch.object(autopilot, "upload_tiktok",
                               lambda v, m, n: (uploads.__setitem__("tt", True), "pub1")[1]), \
             mock.patch.object(buffer, "enabled", lambda: False):
            state = {"topics": {}, "uploads": []}
            autopilot.run_niche(NICHE, state)

        self.assertEqual(uploads, {"yt": True, "tt": True})
        entry = state["uploads"][-1]
        self.assertEqual(entry["youtube"], "yt123")
        self.assertTrue(entry["tiktok"])
        self.assertIn("#programming", entry["tiktok_caption"])
        self.assertTrue((self.root / "CAPTIONS.md").exists())
        self.assertIn("Race Conditions", (self.root / "CAPTIONS.md").read_text())

    def test_dry_run_uploads_nothing_and_keeps_state_clean(self):
        def fail(*a, **k):
            raise AssertionError("DRY_RUN must not upload")

        with mock.patch.object(autopilot, "DRY_RUN", True), \
             mock.patch.object(autopilot, "upload_youtube", fail), \
             mock.patch.object(buffer, "publish", fail), \
             mock.patch.object(autopilot, "upload_tiktok", fail):
            state = {"topics": {}, "uploads": []}
            autopilot.run_niche(NICHE, state)

        self.assertEqual(state["uploads"], [], "a dry run must not record an upload")
        out = list((self.root / "out").glob("*.mp4"))
        self.assertEqual(len(out), 1, "the video should be left in ./out for review")
        sidecar = out[0].with_suffix(".txt").read_text()
        self.assertIn("tiktok caption:", sidecar)
        self.assertIn("script:", sidecar)

    def test_one_failing_niche_does_not_stop_the_others(self):
        calls = []

        def run_niche(niche, state):
            calls.append(niche["id"])
            if niche["id"] == "bad":
                raise RuntimeError("boom")

        niches = {"niches": [{**NICHE, "id": "bad"}, {**NICHE, "id": "good"}]}
        (self.root / "niches.json").write_text(json.dumps(niches))
        with mock.patch.object(autopilot, "run_niche", run_niche), \
             mock.patch.object(autopilot, "load_state", lambda: {"topics": {}, "uploads": []}), \
             mock.patch.object(autopilot.time, "sleep"):
            with self.assertRaises(SystemExit) as ctx:
                autopilot.main()
        # the bad niche is retried, then the good one still runs
        self.assertEqual(calls, ["bad"] * autopilot.RUN_ATTEMPTS + ["good"],
                         "a failing niche must not stop the rest")
        self.assertIn("bad", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
