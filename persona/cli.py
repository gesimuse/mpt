"""The same pipeline without Telegram, for testing and scripting.

GPU commands need Wan2GP's python:
    ~/apps/Wan2GP/.venv/bin/python -m persona.cli cast 4

  import <bible.json>       create a persona from a full bible (new-persona skill)
  new <Name>                blank persona
  list | switch <slug>
  cast [n] [hint]           candidates into casting/
  pick <candidate-id>       make a candidate her face
  refs                      render master refs from the pick into casting/refs-pending/
  lock                      promote pending refs to refs/
  sheet [stem ...]          her face sheet: 12 angles/expressions into refs/sheet/
  scene [n] [hint]          social-lane scenes
  fanvue [n] [hint]         Fanvue-lane scenes
  video <item-id> [motion]
  models | use <role> <model>
  sync                      push her bible+refs+story to the private Kaggle dataset
  kaggle [n] [hint]         render social scenes on Kaggle and import them
  channel <slug> <chat id>  her own Telegram channel (the bot must be an admin there)
  accounts [<slug> tiktok [--direct] | instagram <token> | tiktok-verify <name> <body>]
                            link her accounts for 👍 publishing (see persona/accounts.py)
"""
import sys

from . import character, config, registry


def _engine_studio():
    from .engine import Engine
    from .studio import Studio
    return Studio(Engine())


def _n_hint(args, default):
    if args and args[0].isdigit():
        return int(args[0]), " ".join(args[1:])
    return default, " ".join(args)


def main(argv=None):
    config.load_env()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, args = argv[0], argv[1:]

    if cmd == "import":
        import json
        char = character.import_bible(json.loads(open(args[0]).read()))
        print(f"created {char.slug} at {char.dir} (active); next: cast")
    elif cmd == "new":
        char = character.create(" ".join(args))
        character.set_active(char.slug)
        print(f"created {char.slug} at {char.dir}")
    elif cmd == "list":
        active = config.load_settings().get("active")
        for slug in character.list_characters():
            print(("* " if slug == active else "  ") + slug)
    elif cmd == "switch":
        character.set_active(args[0])
    elif cmd == "models":
        for role, mid in config.load_settings()["roles"].items():
            print(f"{role}: {mid}")
        for m in registry.all_models().values():
            print(registry.describe(m))
    elif cmd == "use":
        role, mid = args
        registry.check(registry.get(mid), role)
        s = config.load_settings()
        s["roles"][role] = mid
        config.save_settings(s)
    elif cmd == "cast":
        n, hint = _n_hint(args, 4)
        for c in _engine_studio().cast(character.active(), n=n, hint=hint):
            print(c.get("error") or c.get("rejected") or f"{c['id']}  {c['path']}  {' · '.join(c['look'].values())}")
    elif cmd == "pick":
        meta = _engine_studio().pick(character.active(), args[0])
        print(f"picked {meta['id']}; next: refs")
    elif cmd == "refs":
        char = character.active()
        for r in _engine_studio().make_refs(char, char.bible["casting_pick"]):
            print(r)
    elif cmd == "lock":
        for p in _engine_studio().lock_refs(character.active()):
            print(p)
    elif cmd == "sheet":
        for r in _engine_studio().make_sheet(character.active(), only=args or None):
            print(r.get("stem"), r.get("path") if r.get("ok") else (r.get("error") or r.get("check")))
    elif cmd in ("scene", "fanvue"):
        n, hint = _n_hint(args, 1)
        lane = "fanvue" if cmd == "fanvue" else "social"
        for item in _engine_studio().render(character.active(), lane=lane, n=n, hint=hint):
            print(item["id"], item.get("path") or item.get("reason"))
    elif cmd == "video":
        char = character.active()
        vid = _engine_studio().animate(char, char.item(args[0]), " ".join(args[1:]) or None)
        print(vid["path"])
    elif cmd == "sync":
        from . import kaggle
        print(kaggle.sync(character.active()))
    elif cmd == "kaggle":
        from . import kaggle
        n, hint = _n_hint(args, 3)
        for item in kaggle.run(character.active(), n=n, hint=hint):
            print(item["id"], item.get("path") or item.get("reason"))
    elif cmd == "channel":
        from . import accounts
        print(accounts.channel(args[0], args[1]))
    elif cmd == "accounts":
        from . import accounts
        if not args:
            print(accounts.show())
        elif args[1] == "tiktok":
            print(accounts.tiktok(args[0], direct="--direct" in args))
        elif args[1] == "instagram":
            print(accounts.instagram(args[0], args[2]))
        elif args[1] == "tiktok-verify":
            print(accounts.tiktok_verify(args[2], " ".join(args[3:])))
    else:
        sys.exit(f"unknown command {cmd!r}\n{__doc__}")


if __name__ == "__main__":
    main()
