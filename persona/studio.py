"""The pipeline itself, with no Telegram in it: cast her, build her master refs,
render scenes, animate them. bot.py, cli.py and the Kaggle kernel all drive this.

Every long operation is a generator that yields each result as soon as it exists,
so the bot can post image 1 while image 2 is still rendering.
"""
import json
from pathlib import Path
import shutil
import time

from . import config, faceid, proportions, qa, registry, scenes


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

    def make_sheet(self, char, only=None):
        """Her face sheet (scenes.FACE_SHEET) into refs/sheet/, rendered from her
        refs. Yields each result; a portrait whose face drifts too far from her refs
        is not kept. `only` limits to some stems.

        Angle portraits must really show their angle (headpose.TARGETS, measured):
        on 2026-10-18 most girls' "three-quarter"/"profile-right"/"up"/"down"
        portraits faced the camera like her front ref, so all her photos did too.
        Each gets up to SHEET_TRIES draws, starting from the ref closest to that
        angle; a side that never turns is her other side mirrored."""
        from . import headpose
        model = self.model("edit", lane="social")
        refs = {p.name[:2]: p for p in char.refs()}
        front = [refs[k] for k in ("01", "02") if k in refs] or char.refs()[:1]
        out = char.refs_dir / "sheet"
        out.mkdir(exist_ok=True)
        ref_emb = faceid.reference_embedding(char.refs())
        front_pitch = 0.0
        try:
            front_pitch = (headpose.pose(refs.get("01", front[0])) or (0.0, 0.0))[0]
        except Exception:
            pass
        tries = int(config.env("PERSONA_SHEET_TRIES", "4"))
        results = {}
        for stem, text in scenes.FACE_SHEET:
            if only and stem not in only:
                continue
            angled = stem in headpose.TARGETS
            side = stem.rsplit("-", 1)[-1]
            prof = out / f"profile-{side}.jpg"
            # A three-quarter view drawn from her profile at that side inherits a turn;
            # from the front refs it came back facing the camera.
            base = ([refs["03"]] + front[:1] if stem.startswith("profile") and "03" in refs
                    else [prof] + front[:1] if stem.startswith("three-quarter") and results.get(f"profile-{side}", {}).get("ok")
                    else [refs["02"]] + front[:1] if stem.startswith("three-quarter") and "02" in refs
                    else front)
            prompt = scenes.sheet_prompt(char, scenes.SHEET_ANGLES.get(stem, text))
            res = None
            for attempt in range(tries if angled else 1):
                try:
                    path = self.engine.image(model, prompt, out, stem, refs=base,
                                             negative=scenes.negative(char, "social"))
                except Exception as e:
                    res = {"stem": stem, "error": str(e)[:300]}
                    continue
                # Side views score lower against front-heavy refs, so profiles get a lower bar.
                floor = 0.3 if "profile" in stem or stem == "over-shoulder" else None
                ok, info = faceid.check(path, ref_emb, min_sim=floor)
                if ok and angled:
                    ok, got = headpose.matches(stem, path, front_pitch)
                    if not ok and stem[-4:] in ("left", "ight"):
                        # Turned the other way: the mirror image is this side.
                        import cv2
                        cv2.imwrite(str(path), cv2.flip(cv2.imread(str(path)), 1))
                        ok, got = headpose.matches(stem, path, front_pitch)
                    info = {**info, "pose": got}
                    if not ok:
                        info["reason"] = f"head not turned ({got})"
                res = {"stem": stem, "path": str(path), "ok": ok, "check": info}
                if ok:
                    break
                path.unlink(missing_ok=True)
            results[stem] = res
        # A side that never turned: the other side, mirrored.
        import cv2
        for stem, other in (("profile-left", "profile-right"), ("profile-right", "profile-left"),
                            ("three-quarter-left", "three-quarter-right"),
                            ("three-quarter-right", "three-quarter-left")):
            r, o = results.get(stem), results.get(other) or {}
            src = out / f"{other}.jpg"
            if r and not r.get("ok") and (o.get("ok") or (not results.get(other) and src.exists())):
                dest = out / f"{stem}.jpg"
                cv2.imwrite(str(dest), cv2.flip(cv2.imread(str(src)), 1))
                ok, got = headpose.matches(stem, dest, front_pitch)
                results[stem] = {"stem": stem, "path": str(dest), "ok": ok, "mirrored": True, "check": {"pose": got}}
                if not ok:
                    dest.unlink(missing_ok=True)
        yield from results.values()
        shutil.rmtree(char.dir / "refs-heads", ignore_errors=True)

    def lock_refs(self, char):
        """Promote refs-pending/ to refs/. The old set is archived, not deleted."""
        pending = char.dir / "casting" / "refs-pending"
        if not pending.exists() or not any(pending.glob("0[1-9]-*")):
            raise ValueError("no pending refs to lock -- pick a candidate first")
        if char.has_refs():
            archive = char.dir / f"refs-archive-{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.move(str(char.refs_dir), archive)
            char.refs_dir.mkdir()
        shutil.rmtree(char.dir / "refs-heads", ignore_errors=True)
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
        # Each scene gets her front ref plus the face-sheet portrait for its gaze
        # (falling back to the three-quarter ref), cropped to the head.
        heads = char.dir / "refs-heads"
        heads.mkdir(exist_ok=True)

        def head(p):
            dest = heads / p.name
            return dest if dest.exists() else faceid.head_crop(p, dest)

        stems = config.env("PERSONA_SCENE_REFS", "01-,02-").split(",")
        default_refs = [p for p in char.refs() if p.name.startswith(tuple(stems))] or char.refs()[:1]
        front = [p for p in default_refs if p.name.startswith("01-")] or default_refs[:1]

        def refs_for(scene):
            # From behind: her full-body ref only (hair, figure, skin) -- her face refs
            # pulled the head round to the camera.
            full = char.refs_dir / "04-full-body.jpg"
            if scene.get("view") == "back" and full.exists():
                return [full]
            sheet = char.refs_dir / "sheet" / f"{scene.get('sheet', '')}.jpg"
            if scene.get("sheet") and sheet.exists():
                # A turned head: its portrait FIRST, so the model takes the head angle
                # from it rather than copying the front ref's straight-on pose.
                turned = scene.get("angle_sheet") in scenes.ANGLE_FLOOR
                chosen = [sheet] + front if turned else front + [sheet]
            else:
                chosen = default_refs
            return [head(p) for p in chosen[: max(1, model.get("max_refs", 1))]]

        ref_emb = faceid.reference_embedding(char.refs())
        from . import headpose
        try:
            front_pitch = (headpose.pose(front[0]) or (0.0, 0.0))[0]
        except Exception:
            front_pitch = 0.0
        two_step = config.flag("PERSONA_TWO_STEP", "0")

        def floor_for(scene):
            if not scene.get("angle"):
                return None
            base = float(config.env("PERSONA_FACE_MIN", "0.62"))
            return min(base, scenes.ANGLE_FLOOR.get(scene.get("angle_sheet", ""), base))
        for scene in (scene_list or scenes.write(char, lane=lane, hint=hint, n=n)):
            refs = refs_for(scene)
            prompt = scenes.scene_prompt(char, scene, lane, len(refs), style=model.get("prompt_style", "qwen"))
            item = char.new_item(kind="image", lane=lane, model=model["id"], scene=scene, prompt=prompt,
                                 tags=scene.get("tags", []), caption=scene.get("caption", ""),
                                 commercial=bool(model.get("commercial")))
            tries = 1 + int(config.env("PERSONA_QA_RETRIES", "1"))
            for attempt in range(tries):
                try:
                    # Two-step first; its retry is one-step, so a two-step result that
                    # does not look enough like her still ends in a usable photo.
                    if two_step and attempt == 0:
                        path = self._two_step(char, model, scene, lane, item["id"],
                                              [head(p) for p in default_refs])
                    else:
                        path = self.engine.image(model, prompt, char.dir / "items", item["id"], refs=refs,
                                                 negative=scenes.negative(char, lane, scene))
                except Exception as e:
                    path = None
                    reason = str(e)[:300]
                    break
                good, why, counts = qa.check(path)
                if good:
                    good, why, counts["proportions"] = proportions.check(path, ref=char.refs_dir / "04-full-body.jpg")
                log(f"{item['id']}: anatomy check {'passed' if good else 'FAILED'} {why} {counts}")
                if good:
                    # Off-model faces are redrawn too, not just dropped. A turned head
                    # gets a lower floor (scenes.ANGLE_FLOOR), the age check stays.
                    # A photo from behind shows no face to check (2026-10-19, asked for).
                    face_ok, info = ((True, {"reason": "from behind"}) if scene.get("view") == "back"
                                     else faceid.check(path, ref_emb, min_sim=floor_for(scene)))
                    if face_ok:
                        # Her head must show the scene's angle (measured); on the last
                        # try a photo that only missed the angle is kept.
                        angle_ok, got = headpose.matches(scene.get("angle_sheet", "") if scene.get("angle") else None,
                                                         path, front_pitch, either_side=True)
                        if angle_ok or attempt == tries - 1:
                            break
                        face_ok, info = False, {"reason": f"head not at {scene.get('angle_sheet') or 'front'} ({got})"}
                    good, why = False, info.get("reason", "face check")
                    log(f"{item['id']}: face check FAILED {why}")
                log(f"{item['id']}: anatomy check failed ({why}), attempt {attempt + 1}/{tries}")
                # Kept (local only, last 40) so rejections can be looked at later.
                rej = char.dir / "rejected"
                rej.mkdir(exist_ok=True)
                shutil.move(str(path), rej / f"{item['id']}-{attempt}.jpg")
                for old in sorted(rej.glob("*.jpg"), key=lambda f: f.stat().st_mtime)[:-40]:
                    old.unlink(missing_ok=True)
                path, reason = None, f"anatomy check: {why}"
            if path is None:
                yield char.update_item(item["id"], status="failed", ok=False, reason=reason)
                continue
            ok, info = ((True, {"reason": "from behind"}) if scene.get("view") == "back"
                        else faceid.check(path, ref_emb, min_sim=floor_for(scene)))
            if not ok:
                path.unlink(missing_ok=True)
                yield char.update_item(item["id"], status="rejected", ok=False, reason=info.get("reason"),
                                       check=info)
                continue
            yield char.update_item(item["id"], status="review", ok=True, path=str(path), check=info)

    def _two_step(self, char, model, scene, lane, stem, faces):
        """Pose first, then her: the scene is drawn WITHOUT her face photos -- with them
        the model copies their straight-on pose and her head angle stays front-facing
        (2026-10-16: asked for profiles, three-quarters, glances, nearly every photo
        faced the lens) -- and her identity is then edited onto that photo, keeping its
        pose, head angle and expression, the same edit 🎭 recreates use."""
        out = char.dir / "items"
        neg = scenes.negative(char, lane)
        pose = self.engine.image(model, scenes.scene_prompt(char, scene, lane, 0), out, f"{stem}-pose", refs=(),
                                 negative=neg)
        others = " and ".join(f"image {i + 2}" for i in range(len(faces)))
        edit = (f"Replace the woman in image 1 with the woman shown in {others} (the same woman). Keep image 1's "
                "exact pose, head angle, gaze direction, facial expression, hands, clothing, camera framing, background "
                "and lighting; only her identity -- her exact face, facial features, hair and skin -- becomes that "
                f"woman ({char.bible.get('identity', '')}).")
        try:
            # Her head-cropped refs (front, three-quarter): with one full-body photo the
            # face match came out 0.44-0.46 on 2026-10-16.
            return self.engine.image(model, edit, out, stem, refs=[str(pose)] + [str(f) for f in faces], negative=neg)
        finally:
            Path(pose).unlink(missing_ok=True)

    # -------------------------------------------------------------------- video
    def animate(self, char, item, motion=None):
        """A video of an approved image. Returns the new video item."""
        model = self.model("video", lane=item.get("lane", "social"))
        motion = (motion or item.get("scene", {}).get("motion") or
                  "she moves naturally and looks into the camera, hair and clothes moving slightly")
        prompt = calm_prompt(motion)
        vid = char.new_item(kind="video", lane=item.get("lane", "social"), model=model["id"], parent=item["id"],
                            prompt=prompt, tags=item.get("tags", []), caption=item.get("caption", ""),
                            scene=item.get("scene", {}), commercial=bool(model.get("commercial")))
        path = self.engine.video(model, item["path"], prompt, char.dir / "items", vid["id"])
        path = slow_down(path, model.get("slowdown"))
        return char.update_item(vid["id"], status="review", ok=True, path=str(path))


# Every second asks for the same slow continuation. The earlier beats escalated
# ("reaches its fullest point", "hair flip") and the Lightning model over-acted on
# them: users found the motion far too much.
TIMELINE_BEATS = [
    "{motion}, very slow and subtle, smooth natural movement, camera static",
    "the same movement continues slowly and gently, small natural motion",
    "slow, calm continuation, only slight movement, she stays in the same place",
    "gentle slow movement, soft breathing, hair moving slightly",
    "slow and smooth, minimal motion, relaxed",
    "she holds the pose calmly, almost still, slow camera push-in",
]


def calm_prompt(motion):
    """One plain sentence. Tested 2026-10-07 on the same photo and seed: Wan2GP's
    per-second "(at N seconds: ...)" timeline made her lunge at the camera with a
    push-in and flying hair (mean frame difference 7.8); the same motion as one
    sentence stayed natural (4.4). A timeline typed by hand is still passed through."""
    motion = motion.strip().rstrip(".")
    if motion.startswith("(at "):
        return motion
    return f"{motion}, slow subtle natural movement, realistic, camera static"


def slow_down(path, factor=None):
    """Motion-interpolate the clip to 24fps and play it PERSONA_VIDEO_SLOWDOWN times
    slower (default 1.6: Wan's 5s at 16fps becomes ~8s at 24fps, smoother and calmer).
    Falls back to the original if ffmpeg is missing or fails."""
    import subprocess
    # A model may set its own (the calm model is slow already); the env var wins.
    factor = float(config.env("PERSONA_VIDEO_SLOWDOWN") or (factor if factor is not None else 1.6))
    # No slow-down still gets interpolated to 24fps: Wan renders 16fps, and Instagram
    # refuses Reels under 23fps ("Video frame rate must be at least 23 fps", 2026-10-14).
    filt = (f"minterpolate=fps=32:mi_mode=mci:mc_mode=aobmc:vsbmc=1,setpts={factor}*PTS" if factor > 1
            else "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    out = path.with_name(path.stem + "-slow.mp4")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-filter:v", filt,
                        "-r", "24", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        print(f"[persona.studio] slow-down failed, keeping the original: {r.stderr[-300:]}", flush=True)
        return path
    path.unlink(missing_ok=True)
    final = path.with_suffix(".mp4")
    out.rename(final)
    return final


def _timeline(motion):
    """Wan 2.2 Enhanced Lightning's per-second prompt format, same template as
    local/wan2gp_bot.py's to_timeline() for a one-line motion."""
    motion = motion.strip().rstrip(".")
    if motion.startswith("(at "):
        return motion
    return "\n".join(f"(at {i} second{'' if i == 1 else 's'}: {b.format(motion=motion)})"
                     for i, b in enumerate(TIMELINE_BEATS))
