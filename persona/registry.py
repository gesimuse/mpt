"""The model registry: one JSON file per model in persona/models/.

Adding a model means adding a file. Nothing else in the pipeline names a model; it
asks for a ROLE (config.DEFAULT_ROLES) and gets whatever settings.json maps there.

Fields a file can carry:
  id                  registry id, must match the filename
  name, hf, notes     for humans (/models lists them)
  kind                "image" | "video"
  roles               which roles it may fill: cast, edit, fanvue_edit, video
  lanes               "social" and/or "fanvue" -- a social-lane job refuses a model
                      that only lists "fanvue"
  local_only          true = never shipped to Kaggle or anywhere else
  license, commercial whether output may be monetized (warned on, see bot.py)
  kaggle_ok           whether persona/kaggle.py may select it
  max_refs            reference images it accepts (0 = text-to-image only)
  wan2gp_model_type   the Wan2GP model id that actually runs
  finetune            a file in persona/wan2gp_finetunes/ to install into Wan2GP
                      first (how a model Wan2GP does not ship gets in)
  profile             a Wan2GP accelerator profile (path inside the Wan2GP dir),
                      merged under `settings`
  settings            Wan2GP settings overrides, applied last
  slowdown            video only: playback slow-down after interpolation (1 = none)
"""
import json

from . import config

MODELS_DIR = config.PACKAGE / "models"
ROLES = ("cast", "edit", "fanvue_edit", "video")


class RegistryError(ValueError):
    pass


def all_models():
    models = {}
    for path in sorted(MODELS_DIR.glob("*.json")):
        m = json.loads(path.read_text())
        if m.get("id") != path.stem:
            raise RegistryError(f"{path.name}: id {m.get('id')!r} must match the filename")
        models[m["id"]] = m
    return models


def get(model_id):
    models = all_models()
    if model_id not in models:
        raise RegistryError(f"unknown model {model_id!r}; known: {', '.join(models)}")
    return models[model_id]


def for_role(role, settings=None, lane=None, kaggle=False):
    """The model mapped to `role`, checked against what the job is about to do with
    it. Raises rather than silently substituting: a Fanvue-only model must never end
    up rendering a TikTok post, and a local-only one never on Kaggle."""
    if role not in ROLES:
        raise RegistryError(f"unknown role {role!r}; roles: {', '.join(ROLES)}")
    settings = settings or config.load_settings()
    model = get(settings["roles"][role])
    check(model, role, lane=lane, kaggle=kaggle)
    return model


def check(model, role, lane=None, kaggle=False):
    if role not in model.get("roles", []):
        raise RegistryError(f"{model['id']} cannot fill role {role!r} "
                            f"(it fills: {', '.join(model.get('roles', []))})")
    if lane and lane not in model.get("lanes", []):
        raise RegistryError(f"{model['id']} is not allowed in the {lane!r} lane")
    if kaggle and (model.get("local_only") or not model.get("kaggle_ok")):
        raise RegistryError(f"{model['id']} is local-only and cannot run on Kaggle")


def describe(model):
    lic = model.get("license", "?")
    money = "commercial OK" if model.get("commercial") else "NON-commercial"
    where = "local only" if model.get("local_only") else "local+kaggle"
    return (f"{model['id']} -- {model.get('name', '')}\n"
            f"   roles: {', '.join(model.get('roles', []))} | lanes: "
            f"{', '.join(model.get('lanes', []))} | {lic}, {money} | {where}")
