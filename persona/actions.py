"""GitHub Actions entry: render a social-lane batch on Kaggle and post it for review.

    python -m persona.actions <slug> [n]

The runner has no persona state of its own, so her bible, refs and storyline come
down from the private Kaggle dataset the laptop last synced (/sync, or /lock).
Results go to the review channel with the normal buttons. Pressing one is handled
by the local bot, which pulls the same kernel output in (kaggle.import_output) the
first time it sees one of these ids, so this works even if the laptop was off
when the batch ran.

Social lane only, by construction: kaggle.run renders lane="social" with a model
that passes registry.check(kaggle=True).
"""
import os
import subprocess
import sys
import tempfile

from . import character, config, kaggle, tg
from .bot import item_buttons, item_caption


def main():
    config.load_env()
    slug = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else int(config.env("PERSONA_DAILY_SOCIAL", "3"))
    os.environ.setdefault("PERSONA_HOME", tempfile.mkdtemp(prefix="persona-home-"))
    home = config.home()
    env = kaggle._env()
    subprocess.run(["kaggle", "datasets", "download", kaggle.dataset_id(slug),
                    "--unzip", "-p", str(home / slug)], env=env, check=True)
    kaggle.restore_layout(home / slug)
    char = character.Character(slug)
    review = tg.chat("social")
    items = kaggle.run(char, n=n, do_sync=False)
    for item in items:
        if not item.get("ok"):
            tg.text(review, f"🚫 #{item['id']} dropped on Kaggle: {(item.get('reason') or '')[:200]}")
            continue
        tg.photo(review, item["path"], caption="☁️ " + item_caption(char, item), buttons=item_buttons(item))
    print(f"posted {sum(1 for i in items if i.get('ok'))}/{len(items)}")


if __name__ == "__main__":
    main()
