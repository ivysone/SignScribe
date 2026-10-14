"""
SignScribe - fingerspelling practice with your webcam.

Uses MediaPipe to find 21 landmarks on your hand, then a simple rule-based
check (which fingers are sticking out) to guess the letter. Hold the target
letter for about a second to score a point.

Keys: s = skip, q = quit
"""

import math
import os
import random
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HOLD_SECONDS = 1.1   # how long you have to hold a sign
HISTORY = 18         # frames to look back over when smoothing

# fingers order: thumb, index, middle, ring, pinky (True = sticking out)
SIGNS = [
    {"key": "A", "hint": "fist, thumb against the side",       "ext": (1, 0, 0, 0, 0)},
    {"key": "S", "hint": "fist, thumb over the fingers",       "ext": (0, 0, 0, 0, 0)},
    {"key": "B", "hint": "flat hand, thumb tucked in",         "ext": (0, 1, 1, 1, 1)},
    {"key": "5", "hint": "open hand, everything spread",       "ext": (1, 1, 1, 1, 1)},
    {"key": "L", "hint": "index up and thumb out",             "ext": (1, 1, 0, 0, 0)},
    {"key": "V", "hint": "index + middle up, apart",           "ext": (0, 1, 1, 0, 0), "apart": True},
    {"key": "U", "hint": "index + middle up, together",        "ext": (0, 1, 1, 0, 0), "apart": False},
    {"key": "W", "hint": "index, middle, ring up",             "ext": (0, 1, 1, 1, 0)},
    {"key": "Y", "hint": "thumb and pinky out",                "ext": (1, 0, 0, 0, 1)},
    {"key": "I", "hint": "pinky up only",                      "ext": (0, 0, 0, 0, 1)},
]

# pairs of landmark indices to draw as lines
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def get_model():
    if not os.path.exists(MODEL_PATH):
        print("downloading hand model (one time, ~10MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def finger_flags(lm):
    """Work out which fingers are out. Returns (flags, hand_size)."""
    wrist = lm[0]
    size = dist(wrist, lm[9]) or 1e-6   # wrist to middle knuckle, used to normalise

    def finger_out(tip, pip):
        return (dist(lm[tip], wrist) - dist(lm[pip], wrist)) / size > 0.28

    # thumb is awkward - compare against the base of the pinky instead of the wrist
    thumb_out = (dist(lm[4], lm[17]) - dist(lm[3], lm[17])) / size > 0.22

    flags = (
        int(thumb_out),
        int(finger_out(8, 6)),
        int(finger_out(12, 10)),
        int(finger_out(16, 14)),
        int(finger_out(20, 18)),
    )
    return flags, size


def classify(lm):
    flags, size = finger_flags(lm)
    spread = dist(lm[8], lm[12]) / size
    for sign in SIGNS:
        if sign["ext"] != flags:
            continue
        # V and U have the same fingers up, only the gap between them differs
        if "apart" in sign and (spread > 0.55) != sign["apart"]:
            continue
        return sign["key"]
    return None


def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in BONES:
        cv2.line(frame, pts[a], pts[b], (89, 122, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (242, 239, 237), -1)


def draw_ring(frame, centre, radius, fraction):
    cv2.circle(frame, centre, radius, (76, 58, 46), 6)
    if fraction > 0:
        cv2.ellipse(frame, centre, (radius, radius), -90, 0, 360 * min(fraction, 1),
                    (89, 122, 255), 6)


def pick_new(current):
    options = [s for s in SIGNS if s is not current]
    return random.choice(options)


def main():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=get_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("couldn't open the webcam")
        return

    target = pick_new(None)
    streak = 0
    hold_start = None
    recent = deque(maxlen=HISTORY)
    last_tick = time.time()
    fps = 0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)   # mirror so it feels like a mirror

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(image, timestamp)

        label = None
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            draw_hand(frame, lm)
            label = classify(lm)
        recent.append(label)

        # smooth it out - only trust a label if it shows up in at least half the recent frames
        counts = Counter(l for l in recent if l)
        stable = None
        stability = 0
        if counts:
            best, n = counts.most_common(1)[0]
            stability = n / len(recent)
            if stability >= 0.5:
                stable = best

        # hold timer
        fraction = 0
        if stable == target["key"]:
            if hold_start is None:
                hold_start = time.time()
            fraction = (time.time() - hold_start) / HOLD_SECONDS
            if fraction >= 1:
                streak += 1
                target = pick_new(target)
                hold_start = None
                fraction = 0
        else:
            hold_start = None

        # fps
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last_tick, 1e-6))
        last_tick = now

        # ui
        draw_ring(frame, (70, 70), 45, fraction)
        cv2.putText(frame, target["key"], (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    (242, 239, 237), 3)
        cv2.putText(frame, target["hint"], (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (242, 239, 237), 1)
        cv2.putText(frame, f"streak {streak}", (140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (157, 198, 116), 1)
        detected = stable if stable else ("..." if result.hand_landmarks else "no hand")
        cv2.putText(frame, f"seeing: {detected}  ({int(stability * 100)}%)",
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (224, 209, 127), 1)
        cv2.putText(frame, f"{fps:.0f} fps   s = skip   q = quit",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (169, 152, 139), 1)

        cv2.imshow("SignScribe", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            target = pick_new(target)
            hold_start = None

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
