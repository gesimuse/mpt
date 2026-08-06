"""Judge a script before it becomes a video, and send it back until it earns a pass.

Writing and judging in one call does not work: a model asked to produce and approve its
own output approves nearly everything. This is a separate call with a rubric, no sight
of the previous verdict, and the power to reject.

The loop is generate -> critique -> revise, and a script that never passes is not
published. A run that produces nothing is cheaper than a channel of filler, which is
also what YouTube's inauthentic-content review penalises.
"""
import os

from llm import nim_json

# A script passes on its average, provided nothing is actually bad. Requiring every
# metric to clear the bar meant one stubborn score -- usually the hook -- vetoed
# scripts scoring 9 everywhere else, and the run published nothing at all.
MIN_SCORE = float(os.environ.get("CRITIC_MIN_SCORE", "7"))
HARD_FLOOR = float(os.environ.get("CRITIC_FLOOR", "6"))
MAX_ROUNDS = int(os.environ.get("CRITIC_ROUNDS", "4"))

RUBRIC = """You are a harsh script editor for a short-video channel. You did not write this
script and you gain nothing by approving it. Most drafts should be sent back.

Score each from 1 to 10:

  answers_question  Does it actually answer the viewer's question? A script about a
                    neighbouring subject scores 1, however well written.
  hook              Would the first sentence stop someone scrolling? A generic opener
                    ("Have you ever wondered...") scores 3 or less.
  mapping           Does it name each part of the analogy and say which part of the real
                    mechanism it stands for? An analogy left unmapped scores 3 or less.
  accuracy          Is every technical claim true, and does every term belong to this
                    subject? One borrowed or wrong term caps this at 4.
  specificity       Concrete detail over vague summary. "It can cause problems" scores 2;
                    "the second write silently overwrites the first" scores 9.
  payoff            Does it end on something a viewer could repeat to a friend?

Then decide:
  "publish" only if the scores average at least {min_score}, none is below 6, and
  nothing is factually wrong.
  "revise" otherwise.

problems: what is wrong, concretely, quoting the offending phrases.
fix: direct instructions to the writer for the next attempt. Name what to cut and what
to add. Do not rewrite the script yourself."""


def log(msg): print(f"[critic] {msg}", flush=True)


def review(topic, script, question=None, min_score=None):
    """Return (verdict, scores, problems, fix). Never raises on a bad response: a critic
    that fails must not block the pipeline, it just cannot approve anything either."""
    min_score = min_score or MIN_SCORE
    asked = f"\n\nThe viewer's question was: {question}" if question else ""
    try:
        result = nim_json(
            RUBRIC.format(min_score=min_score) +
            ' JSON schema: {"scores": {"answers_question": n, "hook": n, "mapping": n, '
            '"accuracy": n, "specificity": n, "payoff": n}, "verdict": "publish"|"revise", '
            '"problems": ["..."], "fix": "..."}',
            f"Topic: {topic}{asked}\n\nScript:\n{script}",
            temperature=0.3,
            max_tokens=1200,
        )
    except Exception as e:
        log(f"review failed ({type(e).__name__}: {str(e)[:90]})")
        return "revise", {}, [f"critic unavailable: {type(e).__name__}"], ""

    scores = {k: v for k, v in (result.get("scores") or {}).items() if isinstance(v, (int, float))}
    problems = [p for p in (result.get("problems") or []) if isinstance(p, str)]
    fix = (result.get("fix") or "").strip()
    verdict = (result.get("verdict") or "").strip().lower()
    # Trust the scores over the verdict: models say "publish" while scoring a 4.
    if scores:
        mean = sum(scores.values()) / len(scores)
        verdict = ("publish" if mean >= min_score and min(scores.values()) >= HARD_FLOOR
                   else "revise")
    return verdict, scores, problems, fix


def summarise(scores):
    if not scores:
        return "no scores"
    worst = min(scores, key=scores.get)
    return (", ".join(f"{k}={v:.0f}" for k, v in scores.items())
            + f" (worst: {worst})")


def refine(topic, write, question=None, rounds=None, min_score=None, review_fn=None):
    """Generate, critique, revise. `write(feedback)` returns a script; feedback is None
    on the first attempt and the critic's instructions afterwards.

    Returns (script, verdict, scores). The caller decides what to do with a script that
    never passed -- publishing it is not this module's decision to make."""
    rounds = rounds or MAX_ROUNDS
    min_score = min_score or MIN_SCORE
    review_fn = review_fn or review
    best = None
    feedback = None
    for attempt in range(rounds):
        script = write(feedback)
        verdict, scores, problems, fix = review_fn(topic, script, question, min_score)
        worst = (sum(scores.values()) / len(scores)) if scores else 0
        log(f"round {attempt + 1}/{rounds}: {verdict} -- {summarise(scores)}")
        for p in problems[:3]:
            log(f"  - {p[:150]}")
        if best is None or worst > best[3]:
            best = (script, verdict, scores, worst)
        if verdict == "publish":
            return script, verdict, scores
        feedback = fix or "; ".join(problems) or "Make it sharper and more concrete."
    script, verdict, scores, _ = best
    return script, verdict, scores
