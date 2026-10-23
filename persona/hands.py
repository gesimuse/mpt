"""Her hands counted by a hand detector, not by the vision model's reading of the
photo (2026-10-23: Qwen2.5-VL listed 2 hands on a photo with a third hand on a
door, at both resolutions, and the user kept getting extra hands).

Bingsu/adetailer's YOLOv9c hand model over the whole photo and 12 overlapping
zoomed tiles (a small hand on a door frame is only found zoomed in), boxes merged
when they are the same hand, kept at confidence >= 0.6 and inside her body's box
(DWPose, padded) -- a cactus at the top edge read as a hand. Tested on the 80
latest photos: the one with a third hand flagged, no false alarm.

    python -m persona.hands <photo>...
"""
import sys
import threading

CONF = 0.6
_M = None
_LOCK = threading.Lock()


def _model():
    global _M
    with _LOCK:
        if _M is None:
            try:
                from huggingface_hub import hf_hub_download
                from ultralytics import YOLO
                _M = YOLO(hf_hub_download("Bingsu/adetailer", "hand_yolov9c.pt"))
            except Exception as e:
                print(f"[persona.hands] detector unavailable: {type(e).__name__}: {str(e)[:150]}", flush=True)
                _M = False
        return _M


def _boxes(im, m):
    W, H = im.size
    tiles = [(0, 0, W, H)] + [(int(x * W / 3), int(y * H / 4), int(min(W, (x + 1.6) * W / 3)), int(min(H, (y + 1.6) * H / 4)))
                              for x in range(3) for y in range(4)]
    found = []
    for x0, y0, x1, y1 in tiles:
        r = m(im.crop((x0, y0, x1, y1)), verbose=False, conf=0.35, imgsz=640)[0]
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            found.append((b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0, float(c)))
    hands = []
    for b in sorted(found, key=lambda b: -b[4]):
        cx, cy, size = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1])
        same = any((o[0] <= cx <= o[2] and o[1] <= cy <= o[3]) or
                   ((cx - (o[0] + o[2]) / 2) ** 2 + (cy - (o[1] + o[3]) / 2) ** 2) ** 0.5
                   < 0.8 * max(size, o[2] - o[0], o[3] - o[1]) for o in hands)
        if not same:
            hands.append(b)
    return hands


def find(path):
    """Her hands as [(x, y, conf)], or None when the detector can't run."""
    from PIL import Image
    m = _model()
    if not m:
        return None
    im = Image.open(path).convert("RGB")
    hands = [b for b in _boxes(im, m) if b[4] >= CONF]
    # Only hands on or next to her: her body's box from the pose model, padded.
    try:
        from . import proportions
        person = proportions._person(path)
    except Exception:
        person = None
    if person:
        x1, y1, x2, y2 = person[2][:4]
        pw, ph = x2 - x1, y2 - y1
        x1, x2, y1, y2 = x1 - 0.15 * pw, x2 + 0.15 * pw, y1 - 0.1 * ph, y2 + 0.1 * ph
        hands = [b for b in hands if x1 <= (b[0] + b[2]) / 2 <= x2 and y1 <= (b[1] + b[3]) / 2 <= y2]
    return [(int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2), round(b[4], 2)) for b in hands]


def extra(path):
    """'' when she has at most 2 hands (or it can't tell), else what is wrong, in
    words for a redo ("a third hand at the upper right")."""
    try:
        hands = find(path)
    except Exception as e:
        print(f"[persona.hands] check failed: {type(e).__name__}: {str(e)[:150]}", flush=True)
        return ""
    if not hands or len(hands) <= 2:
        return ""
    from PIL import Image
    W, H = Image.open(path).size
    # The least confident is most likely the extra one.
    x, y, _ = min(hands, key=lambda h: h[2])
    where = f"{'upper' if y < H / 2 else 'lower'} {'left' if x < W / 2 else 'right'}"
    return f"{len(hands)} hands (an extra hand at the {where} of the photo)"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, find(p), extra(p) or "ok")
