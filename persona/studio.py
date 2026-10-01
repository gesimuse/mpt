"""The pipeline itself, with no Telegram in it: cast her, build her master refs,
render scenes, animate them. bot.py, cli.py and the Kaggle kernel all drive this.

Every long operation is a generator that yields each result as soon as it exists,
so the bot can post image 1 while image 2 is still rendering.
"""
import json
import shutil
import time

from . import config, faceid, registry, scenes


def log(msg):
    print(f"[persona.studio] {msg}", flush=True)


class Studio:
    def __init__(self, engine, kaggle=False):
        self.engine = engine
        self.kaggle = kaggle

    def model(self, role, lane=None):
        return registry.for_role(role, lane=lane, kaggle=self.kaggle)

    # ------------------------------------------------------------------ casting
    def cast(self, char, n=8, hint=""):
        """Yields candidate dicts: {id, path, look, prompt, model}."""
        model = self.model("cast", lane="social")
        out = char.dir / "casting"
        for _ in range(n):
            look = scenes.casting_look(char)
            prompt = scenes.casting_prompt(char, look, hint)
            cid = f"c{int(time.time() * 1000) % 10**9:09d}"
            try:
                path = self.engine.image(model, prompt, out, cid,
                                         negative=scenes.negative(char, "social"))
            except Exception as e:
                yield {"id": cid, "error": str(e)[:300]}
                continue
            ok, info = faceid.check(path, None)
            meta = {"id": cid, "path": str(path), "look": look, "prompt": prompt, "model": model["id"],
                    "check": info}
            (out / f"{cid}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            if not ok:
                path.unlink(missing_ok=True)
                meta["rejected"] = info.get("reason")
            yield meta

    def pick(self, char, cid):
        """Make candidate `cid` her: its look becomes bible.identity."""
        meta_path = char.dir / "casting" / f"{cid}.json"
        if not meta_path.exists():
            raise ValueError(f"no casting candidate {cid}")
        meta = json.loads(meta_path.read_text())
        bible = char.bible
        bible["identity"] = ", ".join(meta["look"].values())
        bible["casting_pick"] = cid
        char.save_bible(bible)
        return meta

    def make_refs(self, char, cid):
        """Yields each master ref, rendered from the picked candidate, into
        casting/refs-pending/. Nothing replaces refs/ until lock_refs()."""
        model = self.model("edit", lane="social")
        src = char.dir / "casting" / f"{cid}.jpg"
        if not src.exists():
            src = next((char.dir / "casting").glob(f"{cid}.*"))
        pending = char.dir / "casting" / "refs-pending"
        shutil.rmtree(pending, ignore_errors=True)
        pending.mkdir(parents=True)
        shutil.copyfile(src, pending / f"00-source{src.suffix}")
        anchor = faceid.reference_embedding([src])
        for stem, shot in scenes.REF_SHOTS:
            try:
                path = self.engine.image(model, scenes.ref_prompt(char, shot), pending, stem, refs=[src],
                                         negative=scenes.negative(char, "social"))
            except Exception as e:
                yield {"stem": stem, "error": str(e)[:300]}
                continue
            ok, info = faceid.check(path, anchor)
            yield {"stem": stem, "path": str(path), "ok": ok, "check": info}

    def lock_refs(self, char):
        """Promote refs-pending/ to refs/. The old set is archived, not deleted."""
        pending = char.dir / "casting" / "refs-pending"
        if not pending.exists() or not any(pending.glob("0[1-9]-*")):
            raise ValueError("no pending refs to lock -- pick a candidate first")
        if char.has_refs():
            archive = char.dir / f"refs-archive-{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.move(str(char.refs_dir), archive)
            char.refs_dir.mkdir()
        for p in sorted(pending.glob("0[1-9]-*")):
            shutil.copyfile(p, char.refs_dir / p.name)
        # The source portrait goes last: it is the picked face, but its framing and
        # background are not studio-neutral, so it should not be <image1>.
        src = next(pending.glob("00-source*"), None)
        if src:
            shutil.copyfile(src, char.refs_dir / f"99-source{src.suffix}")
        return char.refs()

    # ------------------------------------------------------------------- scenes
    def render(self, char, lane="social", n=1, hint="", scene_list=None):
        """Yields one item per scene: the item dict, with item['ok'] False and a
        reason when the face/age gate dropped it."""
        if not char.has_refs():
            raise ValueError(f"{char.name} has no master refs yet -- /cast, pick one, then lock the refs")
        role = "edit" if lane == "social" else "fanvue_edit"
        model = self.model(role, lane=lane)
        # Head refs only. Tested with all five: Qwen drew one woman per reference
        # (three of her at one cafe table), and the full-body ref's t-shirt and
        # jeans leaked into scene outfits. Face from refs, body from the bible text.
        stems = config.env("PERSONA_SCENE_REFS", "01-,02-").split(",")
        refs = [p for p in char.refs() if p.name.startswith(tuple(stems))] or char.refs()[:1]
        refs = refs[: max(1, model.get("max_refs", 1))]
        ref_emb = faceid.reference_embedding(char.refs())
        for scene in (scene_list or scenes.write(char, lane=lane, hint=hint, n=n)):
            prompt = scenes.scene_prompt(char, scene, lane, len(refs))
            item = char.new_item(kind="image", lane=lane, model=model["id"], scene=scene, prompt=prompt,
                                 tags=scene.get("tags", []), caption=scene.get("caption", ""),
                                 commercial=bool(model.get("commercial")))
            try:
                path = self.engine.image(model, prompt, char.dir / "items", item["id"], refs=refs,
                                         negative=scenes.negative(char, lane))
            except Exception as e:
                yield char.update_item(item["id"], status="failed", ok=False, reason=str(e)[:300])
                continue
            ok, info = faceid.check(path, ref_emb)
            if not ok:
                path.unlink(missing_ok=True)
                yield char.update_item(item["id"], status="rejected", ok=False, reason=info.get("reason"),
                                       check=info)
                continue
            yield char.update_item(item["id"], status="review", ok=True, path=str(path), check=info)

    # -------------------------------------------------------------------- video
    def animate(self, char, item, motion=None):
        """A video of an approved image. Returns the new video item."""
        model = self.model("video", lane=item.get("lane", "social"))
        motion = (motion or item.get("scene", {}).get("motion") or
                  "she moves naturally and looks into the camera, hair and clothes moving slightly")
        prompt = _timeline(motion)
        vid = char.new_item(kind="video", lane=item.get("lane", "social"), model=model["id"], parent=item["id"],
                            prompt=prompt, tags=item.get("tags", []), caption=item.get("caption", ""),
                            scene=item.get("scene", {}), commercial=bool(model.get("commercial")))
        path = self.engine.video(model, item["path"], prompt, char.dir / "items", vid["id"])
        return char.update_item(vid["id"], status="review", ok=True, path=str(path))


TIMELINE_BEATS = [
    "{motion}, the movement just beginning, camera static",
    "{motion}, the movement carrying, weight shifting with it",
    "the movement continues, unhurried, her eyes finding the lens",
    "the movement reaches its fullest point, held there",
    "she settles out of it, hair and fabric still moving",
    "she holds the new pose, breathing, gaze on the lens",
]


def _timeline(motion):
    """Wan 2.2 Enhanced Lightning's per-second prompt format, same template as
    local/wan2gp_bot.py's to_timeline() for a one-line motion."""
    motion = motion.strip().rstrip(".")
    if motion.startswith("(at "):
        return motion
    return "\n".join(f"(at {i} second{'' if i == 1 else 's'}: {b.format(motion=motion)})"
                     for i, b in enumerate(TIMELINE_BEATS))
