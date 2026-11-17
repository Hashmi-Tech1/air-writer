"""Air Writer - write in the air with your index finger and have it typed on the computer.

Pipeline:
    camera -> MediaPipe hand landmarks -> gesture logic -> glowing ink canvas
           -> CNN character recognition -> text buffer (+ optional typing into any app)

Gestures (palm facing the camera):
    index finger only ............ WRITE
    index + middle ............... MOVE without drawing / pick a colour or the eraser in the top bar
    index + pinky ................ ERASE ink under the fingertip
    3 fingers (hold ~1 s) ........ SPACE
    fist (hold ~1 s) ............. BACKSPACE (removes typed text)
    open palm (hold ~1 s) ........ CLEAR all ink

The ink stays on screen until you erase or clear it; erasing ink never changes the typed text.

Keys: ESC / q quit | Tab show or hide the side panel | 1-7 pick a colour / eraser | e toggle eraser
      c clear canvas | x clear text | s save text to output.txt
(With --type only ESC and Tab are active, so typed characters can never trigger a shortcut.)
"""

import argparse
import colorsys
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Air Writer"

# ----------------------------------------------------------------------------- behaviour
HOLD_SECONDS = 0.9       # how long a hold-gesture (space / backspace / clear) must be kept
COMMIT_DELAY = 1.0       # pause before the written character is recognised
MOVE_TOLERANCE = 9       # px the fingertip may drift and still count as "paused"
MIN_INK_PIXELS = 150     # ignore accidental tiny strokes
MIN_CONFIDENCE = 0.30    # reject very unsure predictions
BRUSH = 9                # stroke thickness of the mask fed to the recogniser
DWELL_SECONDS = 0.55     # hover time needed to pick a colour
LEAVE_TRIM = 4           # stroke points dropped when the hand leaves the write pose
ERASE_RADIUS = 30        # eraser size in px (at 720p)

# ----------------------------------------------------------------------------- theme (BGR)
PANEL = (30, 24, 22)
BORDER = (92, 78, 70)
TRACK = (72, 62, 58)
TEXT = (245, 240, 236)
TEXT_DIM = (196, 188, 182)
MUTED = (152, 142, 136)
ACCENT = (255, 214, 0)
GREEN = (120, 222, 90)
AMBER = (0, 184, 255)
RED = (96, 96, 255)
BLUE = (255, 150, 90)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

PALETTE = [
    ("Cyan", (255, 229, 0)),
    ("Magenta", (200, 64, 255)),
    ("Lime", (70, 255, 150)),
    ("Amber", (0, 184, 255)),
    ("White", (255, 255, 255)),
    ("Rainbow", None),
    ("Eraser", "eraser"),
]

STATUS = {
    "WRITE": ("WRITING", GREEN),
    "HOVER": ("MOVE / PICK", BLUE),
    "ERASE": ("ERASING", RED),
    "SPACE": ("SPACE", AMBER),
    "FIST": ("BACKSPACE", AMBER),
    "PALM": ("CLEAR", AMBER),
    "NONE": ("READY", MUTED),
    "NO HAND": ("NO HAND", RED),
}
HOLD_LABEL = {"SPACE": "SPACE", "FIST": "BACKSPACE", "PALM": "CLEAR"}

GESTURE_ROWS = [
    ("WRITE", "Index finger", "Write"),
    ("HOVER", "Index + middle", "Move / pick colour"),
    ("ERASE", "Index + pinky", "Eraser"),
    ("SPACE", "3 fingers (hold)", "Space"),
    ("FIST", "Fist (hold)", "Backspace"),
    ("PALM", "Open palm (hold)", "Clear canvas"),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)


# ----------------------------------------------------------------------------- drawing helpers
def ip(p):
    return int(round(float(p[0]))), int(round(float(p[1])))


def text_width(text, scale, thick=1, font=FONT):
    return cv2.getTextSize(text, font, scale, thick)[0][0]


def put_text(img, text, org, scale=0.6, color=TEXT, thick=1, font=FONT, anchor="l", shadow=False):
    x, y = int(org[0]), int(org[1])
    tw = text_width(text, scale, thick, font)
    if anchor == "c":
        x -= tw // 2
    elif anchor == "r":
        x -= tw
    if shadow:  # dark outline keeps the text readable on any camera background
        cv2.putText(img, text, (x, y), font, scale, (12, 10, 10), thick + 3, AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, AA)
    return tw


def panel(img, x1, y1, x2, y2, color=PANEL, alpha=0.66, radius=14, border=BORDER):
    """Translucent rounded panel (only the region of interest is blended)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    r = max(0, min(int(radius), h // 2, w // 2))
    mask = np.zeros((h, w), np.uint8)
    if r > 0:
        cv2.rectangle(mask, (r, 0), (w - r - 1, h - 1), 255, -1)
        cv2.rectangle(mask, (0, r), (w - 1, h - r - 1), 255, -1)
        for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
    else:
        mask[:] = 255
    tint = np.empty_like(roi)
    tint[:] = color
    blended = cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0)
    roi[mask > 0] = blended[mask > 0]
    if border is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(roi, cnts, -1, border, 1, AA)


def bar(img, y1, y2, color=PANEL, alpha=0.74):
    roi = img[y1:y2]
    tint = np.empty_like(roi)
    tint[:] = color
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, roi)


def arc(img, center, radius, frac, color, thick=3, track=None):
    c, r = ip(center), int(radius)
    if track is not None:
        cv2.circle(img, c, r, track, max(1, thick - 1), AA)
    if frac > 0:
        cv2.ellipse(img, c, (r, r), 0, -90, -90 + int(360 * min(frac, 1.0)), color, thick, AA)


def hue_bgr(h, sat=0.8, val=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, sat, val)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_rainbow(img, center, radius):
    for a in range(0, 360, 15):
        cv2.ellipse(img, center, (radius, radius), 0, a, a + 17, hue_bgr(a / 360.0, 0.85, 1.0), -1, AA)


def draw_eraser_icon(img, center, r):
    cv2.circle(img, center, r, (66, 56, 52), -1, AA)
    ang = math.radians(-35)
    ca, sa = math.cos(ang), math.sin(ang)
    u = r / 15.0

    def box(x0, x1, y0, y1):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return np.array([[center[0] + x * ca - y * sa, center[1] + x * sa + y * ca] for x, y in pts], np.int32)

    cv2.fillConvexPoly(img, box(-10 * u, 0, -5.5 * u, 5.5 * u), (96, 96, 255), AA)    # rubber tip
    cv2.fillConvexPoly(img, box(0, 10 * u, -5.5 * u, 5.5 * u), (236, 232, 228), AA)   # body


def make_vignette(w, h):
    """Dims the camera image and darkens the corners (uint8 multiplier, cool tint)."""
    y, x = np.ogrid[:h, :w]
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    v = np.clip(0.72 - 0.42 * d ** 2, 0.22, 1.0)
    chans = [(v * k * 255).astype(np.uint8) for k in (1.0, 0.96, 0.90)]  # B, G, R
    return cv2.merge(chans)


def fit_cover(frame, w, h):
    """Resize + centre-crop the camera frame so it exactly fills w x h."""
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    scale = max(w / fw, h / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(resized[y0:y0 + h, x0:x0 + w])


# ----------------------------------------------------------------------------- hand + gestures
@dataclass
class Hand:
    pts: np.ndarray          # (21, 2) landmark positions in pixels
    fingers: list            # [thumb, index, middle, ring, pinky] extended?

    @property
    def tip(self):
        return self.pts[8]


def fingers_up(pts):
    thumb = abs(pts[4][0] - pts[5][0]) > abs(pts[3][0] - pts[5][0])
    others = [pts[tip][1] < pts[tip - 2][1] for tip in FINGERTIPS[1:]]
    return [bool(thumb)] + [bool(o) for o in others]


def classify_gesture(f):
    _, idx, mid, ring, pinky = f
    if idx and pinky and not (mid or ring):
        return "ERASE"
    if idx and not (mid or ring or pinky):
        return "WRITE"
    if idx and mid and not (ring or pinky):
        return "HOVER"
    if idx and mid and ring and not pinky:
        return "SPACE"
    if idx and mid and ring and pinky:
        return "PALM"
    if not (idx or mid or ring or pinky):
        return "FIST"
    return "NONE"


class GestureStabilizer:
    """Debounce: a new gesture must be seen on consecutive frames before it counts,
    so one bad frame cannot break a stroke. Leaving WRITE reacts a bit faster."""

    def __init__(self, enter=3, leave_write=2):
        self.enter = enter
        self.leave_write = leave_write
        self.current = "NO HAND"
        self.cand = None
        self.count = 0

    def update(self, raw):
        if raw == self.current:
            self.cand, self.count = None, 0
            return self.current
        if raw == self.cand:
            self.count += 1
        else:
            self.cand, self.count = raw, 1
        need = self.leave_write if self.current == "WRITE" else self.enter
        if self.count >= need:
            self.current, self.cand, self.count = raw, None, 0
        return self.current


class OneEuro:
    """One Euro filter: strong smoothing when the finger is slow (no jitter), little lag when it
    moves fast. Works on timestamps, so it behaves the same at 10 or 30 FPS."""

    def __init__(self, min_cutoff=1.0, beta=0.010, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = None
        self.dx = np.zeros(2)
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None:
            self.x, self.t = x.copy(), t
            return self.x
        dt = max(t - self.t, 1e-3)
        self.dx += self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.hypot(*self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class HandTracker:
    def __init__(self):
        import mediapipe as mp  # imported here so the UI code can be tested without it

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, model_complexity=0,
            min_detection_confidence=0.7, min_tracking_confidence=0.6,
        )

    def process(self, frame_bgr):
        result = self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lm = result.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        return Hand(pts, fingers_up(pts))

    def close(self):
        self.hands.close()


# ----------------------------------------------------------------------------- recognition
class CharRecognizer:
    def __init__(self, model_path: Path):
        import torch
        from train_model import CLASSES, CharCNN

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Put emnist_cnn.pt next to air_writer.py "
                "(train it with train_model.py, or on Google Colab)."
            )
        self.torch = torch
        self.classes = CLASSES
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def preprocess(self, canvas):
        """Crop to the stroke, centre in a square, thicken, resize to EMNIST style 28x28."""
        ys, xs = np.where(canvas > 0)
        crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = crop.shape
        side = max(h, w)
        square = np.zeros((side, side), np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        square[y0:y0 + h, x0:x0 + w] = crop

        k = max(3, side // 7)
        square = cv2.dilate(square, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        img = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return self.torch.from_numpy(img).float().div(255.0)[None, None]

    def predict(self, canvas):
        x = self.preprocess(canvas).to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        top = self.torch.topk(probs, 3)
        alts = [(self.classes[int(i)], float(p)) for p, i in zip(top.values, top.indices)]
        return self.classes[int(idx)], float(conf), alts


# ----------------------------------------------------------------------------- ink
def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _curve(p0, c, p2):
    """Points of the quadratic Bezier curve p0 -> p2 with control point c."""
    n = max(3, int(math.hypot(p2[0] - p0[0], p2[1] - p0[1]) / 4))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    p0, c, p2 = (np.asarray(p, dtype=float) for p in (p0, c, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2


class InkCanvas:
    """Ink stays on screen until it is erased or cleared.

    color : the stroke being written right now (bright)
    keep  : strokes that were already recognised (slightly dimmer, permanent)
    mask  : black/white image of the current stroke only, used for recognition
    """

    def __init__(self, w, h, brush):
        self.w, self.h, self.brush = w, h, brush
        self.color = np.zeros((h, w, 3), np.uint8)
        self.keep = np.zeros((h, w, 3), np.uint8)
        self.keep_any = False
        self.mask = np.zeros((h, w), np.uint8)
        self.strokes = []     # strokes of the current character: lists of (point, colour, thickness)
        self.tail = None      # unfinished last piece of the stroke, only drawn on screen
        self.path_len = 0.0

    # ---- drawing
    def _poly(self, pts, color, thickness):
        arr = np.round(np.asarray(pts, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.color, [arr], False, color, thickness, AA)
        cv2.polylines(self.mask, [arr], False, 255, self.brush)

    def _dot(self, pt, color, thickness):
        cv2.circle(self.color, pt, max(1, thickness // 2), color, -1, AA)
        cv2.circle(self.mask, pt, max(1, self.brush // 2), 255, -1)

    def _draw_piece(self, st, n):
        """Draw the piece of stroke `st` that becomes final when its n-th point arrives.
        Curves run through the midpoints between points (quadratic Bezier), so the corners of the
        hand-drawn polyline turn into one smooth line."""
        pt, color, thickness = st[n - 1]
        if n == 1:
            self._dot(pt, color, thickness)
        elif n == 2:
            self._poly([st[0][0], _mid(st[0][0], pt)], color, thickness)
        else:
            a, b = st[n - 3][0], st[n - 2][0]
            self._poly(_curve(_mid(a, b), b, _mid(b, pt)), color, thickness)

    def _finish(self, st):
        if len(st) >= 2:
            pt, color, thickness = st[-1]
            self._poly([_mid(st[-2][0], pt), pt], color, thickness)

    def begin_stroke(self):
        self.strokes.append([])

    def add_point(self, pt, color, thickness):
        if not self.strokes:
            self.begin_stroke()
        st = self.strokes[-1]
        st.append((pt, color, thickness))
        self._draw_piece(st, len(st))
        self.tail = (_mid(st[-2][0], pt) if len(st) > 1 else pt, pt, color, thickness)

    def finish_stroke(self):
        if self.strokes:
            self._finish(self.strokes[-1])
        self.tail = None

    def trim(self, k):
        """Drop the last k points of the current stroke (the hand was already leaving the
        'write' pose, so they are not intentional) and redraw the rest."""
        if not self.strokes:
            return
        st = self.strokes[-1]
        del st[max(0, len(st) - k):]
        if len(st) < 2:
            self.strokes.pop()
        self.color[:] = 0
        self.mask[:] = 0
        for st in self.strokes:
            for n in range(1, len(st) + 1):
                self._draw_piece(st, n)
            self._finish(st)
        self.tail = None

    # ---- erasing / committing
    def erase(self, p0, p1, radius):
        for layer in (self.color, self.keep, self.mask):
            cv2.line(layer, p0, p1, 0, 2 * radius)
            cv2.circle(layer, p0, radius, 0, -1)
            cv2.circle(layer, p1, radius, 0, -1)
        self.strokes = []     # erased ink must never be redrawn by trim()
        self.tail = None

    def commit_stroke(self):
        """The character was recognised: its ink stays on screen but no longer counts
        as part of the next character."""
        self.finish_stroke()
        cv2.add(self.keep, cv2.convertScaleAbs(self.color, alpha=0.8), dst=self.keep)
        self.keep_any = True
        self.color[:] = 0
        self.mask[:] = 0
        self.strokes = []

    def discard_stroke(self):
        self.color[:] = 0
        self.mask[:] = 0
        self.strokes = []
        self.tail = None

    def clear(self):
        self.discard_stroke()
        self.keep[:] = 0
        self.keep_any = False
        self.path_len = 0.0

    # ---- display
    def glow(self):
        if self.keep_any:
            base = cv2.add(self.keep, self.color)
        else:
            base = self.color.copy() if self.tail is not None else self.color
        if self.tail is not None:
            start, end, color, thickness = self.tail
            cv2.line(base, ip(start), ip(end), color, thickness, AA)
        small = cv2.resize(base, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3.5)
        halo = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(base, 1.0, halo, 1.5, 0)


# ----------------------------------------------------------------------------- application
class AirWriterApp:
    def __init__(self, recognizer, width, height, typer=None):
        self.rec = recognizer
        self.typer = typer
        self.W, self.H = width, height
        self.s = height / 720.0
        s = self.s

        self.canvas = InkCanvas(width, height, max(3, int(round(BRUSH * s))))
        self.vignette = make_vignette(width, height)
        self.stab = GestureStabilizer()
        self.gesture = "NO HAND"

        self.text = ""
        self.last_pred = None     # (char, confidence, alternatives, time)
        self.popup = None         # (char, (x, y), time)
        self.toast = None         # (message, colour, until)
        self.show_panel = True
        self.fps = 0.0
        self._last_t = None

        self.smooth = None
        self.prev_pt = None
        self.last_tip = None
        self.anchor = None
        self.last_active = 0.0
        self.idle = 0.0
        self.ink_px = 0

        self.color_idx = 0
        self.prev_color_idx = 0
        self.erase_prev = None
        self.erase_mode = False
        self.filter = OneEuro()
        self.dwell_idx = None
        self.dwell_start = 0.0
        self.dwell_prog = 0.0

        self.hold_gesture = None
        self.hold_start = 0.0
        self.hold_fired = False
        self.hold_prog = 0.0

        self.top_h = int(64 * s)
        self.out_h = int(88 * s)
        n, gap, r = len(PALETTE), int(56 * s), int(15 * s)
        x0 = width // 2 - ((n - 1) * gap) // 2
        self.swatches = [(x0 + i * gap, self.top_h // 2, r) for i in range(n)]

    # ------------------------------------------------------------------ logic
    def ink_color(self, now):
        c = PALETTE[self.color_idx][1]
        if isinstance(c, str):
            return (255, 255, 255)
        if c is None:
            return hue_bgr(now * 0.25 + self.canvas.path_len * 0.0012, 0.8, 1.0)
        return c

    def show_toast(self, msg, color, now, duration=1.8):
        self.toast = (msg, color, now + duration)

    def emit(self, char=None, backspace=False):
        if backspace:
            self.text = self.text[:-1]
            if self.typer:
                self.typer.press("backspace")
        elif char is not None:
            self.text += char
            if self.typer:
                if char == " ":
                    self.typer.press("space")
                else:
                    self.typer.write(char)

    def _smooth_tip(self, hand, now):
        if hand is None:
            self.filter.reset()
            self.smooth = None
            return None
        self.smooth = self.filter(hand.tip, now)
        return self.smooth

    def select_tool(self, idx, now):
        if PALETTE[self.color_idx][1] != "eraser":
            self.prev_color_idx = self.color_idx
        self.color_idx = idx
        if PALETTE[idx][1] == "eraser":
            self.show_toast("Eraser on - your index finger erases", TEXT, now)
        else:
            self.show_toast(f"Colour: {PALETTE[idx][0]}", TEXT, now)

    def _swatch_at(self, tip):
        for i, (cx, cy, r) in enumerate(self.swatches):
            if math.hypot(tip[0] - cx, tip[1] - cy) <= r + 14 * self.s:
                return i
        return None

    def update(self, hand, now):
        s = self.s
        dt = 1 / 30
        if self._last_t is not None:
            dt = max(now - self._last_t, 1e-3)
            self.fps = (0.9 * self.fps + 0.1 / dt) if self.fps else 1.0 / dt
        self._last_t = now

        raw = classify_gesture(hand.fingers) if hand is not None else "NO HAND"
        g = self.gesture = self.stab.update(raw)
        tip = self._smooth_tip(hand, now)
        eraser_tool = PALETTE[self.color_idx][1] == "eraser"
        in_canvas = tip is not None and tip[1] > self.top_h + 8 * s
        self.erase_mode = tip is not None and (g == "ERASE" or (g == "WRITE" and eraser_tool))

        # ---- erasing
        if self.erase_mode and in_canvas:
            pt = ip(tip)
            self.canvas.erase(self.erase_prev or pt, pt, int(ERASE_RADIUS * s))
            self.erase_prev = pt
            self.last_active = now
        else:
            self.erase_prev = None

        # ---- writing
        was_drawing = self.prev_pt is not None
        drawing = g == "WRITE" and not eraser_tool and in_canvas
        if drawing:
            pt = ip(tip)
            col = self.ink_color(now)
            if self.prev_pt is None:
                self.canvas.begin_stroke()
                self.canvas.add_point(pt, col, max(3, int(8 * s)))
                self.prev_pt = self.anchor = pt
                self.last_active = now
            else:
                dist = math.hypot(pt[0] - self.prev_pt[0], pt[1] - self.prev_pt[1])
                if dist >= 1.5 * s:                       # ignore sub-pixel jitter
                    speed = dist / dt                     # px per second, independent of FPS
                    thick = max(3, int(round(min(max(13 - speed * 0.005, 5), 12) * s)))
                    self.canvas.add_point(pt, col, thick)
                    self.canvas.path_len += dist
                    self.prev_pt = pt
                if math.hypot(pt[0] - self.anchor[0], pt[1] - self.anchor[1]) > MOVE_TOLERANCE * s:
                    self.anchor = pt
                    self.last_active = now
            self.last_tip = pt
        else:
            self.prev_pt = None
            if was_drawing:
                self.canvas.trim(LEAVE_TRIM)

        # ---- colour / tool picking (hover over a swatch)
        self.dwell_prog = 0.0
        hit = self._swatch_at(tip) if (g == "HOVER" and tip is not None) else None
        if hit is None:
            self.dwell_idx = None
        else:
            if hit != self.dwell_idx:
                self.dwell_idx, self.dwell_start = hit, now
            if hit != self.color_idx:
                self.dwell_prog = min((now - self.dwell_start) / DWELL_SECONDS, 1.0)
                if self.dwell_prog >= 1.0:
                    self.select_tool(hit, now)
                    self.dwell_prog = 0.0

        # ---- hold gestures: space / backspace / clear
        self.hold_prog = 0.0
        if g in HOLD_LABEL:
            if g != self.hold_gesture:
                self.hold_gesture, self.hold_start, self.hold_fired = g, now, False
            if not self.hold_fired:
                self.hold_prog = min((now - self.hold_start) / HOLD_SECONDS, 1.0)
                if self.hold_prog >= 1.0:
                    self.hold_fired = True
                    self.hold_prog = 0.0
                    self._fire_hold(g, now)
        else:
            self.hold_gesture, self.hold_fired = None, False

        # ---- recognise after a pause
        self.ink_px = cv2.countNonZero(self.canvas.mask)
        self.idle = now - self.last_active
        if self.ink_px and self.idle > COMMIT_DELAY and not self.erase_mode:
            if self.ink_px >= MIN_INK_PIXELS:
                self._commit(now)
            else:
                self.canvas.discard_stroke()
            self.ink_px = 0

    def _fire_hold(self, g, now):
        if g == "PALM":
            self.canvas.clear()
            self.prev_pt = None
            self.show_toast("Canvas cleared", TEXT, now)
        elif g == "FIST":
            if self.text:
                self.emit(backspace=True)
                self.show_toast("Last character deleted", AMBER, now)
            else:
                self.show_toast("Nothing to delete", MUTED, now)
        elif g == "SPACE":
            self.emit(" ")
            self.show_toast("Space added", TEXT, now)

    def _commit(self, now):
        ys, xs = np.nonzero(self.canvas.mask)
        center = (int(xs.mean()), int(ys.mean()))
        char, conf, alts = self.rec.predict(self.canvas.mask)
        self.last_pred = (char, conf, alts, now)
        if conf >= MIN_CONFIDENCE:
            self.emit(char)
            self.popup = (char, center, now)
        else:
            self.show_toast("Not sure - write it larger and slower", AMBER, now)
        self.canvas.commit_stroke()
        self.prev_pt = None

    def handle_key(self, key, now):
        """Returns True when the app should quit."""
        if key == 255:
            return False
        if key == 27:
            return True
        if key == 9:
            self.show_panel = not self.show_panel
            return False
        if self.typer is not None:
            return False
        ch = chr(key) if 0 <= key < 256 else ""
        if ch == "q":
            return True
        if ch == "c":
            self.canvas.clear()
            self.prev_pt = None
            self.show_toast("Canvas cleared", TEXT, now)
        elif ch == "x":
            self.text = ""
            self.show_toast("Text cleared", TEXT, now)
        elif ch == "s":
            Path("output.txt").write_text(self.text, encoding="utf-8")
            self.show_toast("Saved to output.txt", GREEN, now)
        elif ch == "e":
            if PALETTE[self.color_idx][1] == "eraser":
                self.select_tool(self.prev_color_idx, now)
            else:
                self.select_tool(len(PALETTE) - 1, now)
        elif ch.isdigit() and 1 <= int(ch) <= len(PALETTE):
            self.select_tool(int(ch) - 1, now)
        return False

    # ------------------------------------------------------------------ rendering
    def render(self, frame, hand, now):
        out = cv2.multiply(frame, self.vignette, scale=1 / 255.0)
        out = cv2.add(out, self.canvas.glow())
        if hand is not None:
            self._draw_hand(out, hand)
        self._draw_cursor(out, hand, now)
        self._draw_hint(out, hand)
        self._draw_popup(out, now)
        self._draw_top_bar(out, now)
        if self.show_panel:
            self._draw_side_panel(out)
        self._draw_output(out, now)
        self._draw_toast(out, now)
        return out

    def _draw_hand(self, out, hand):
        s = self.s
        ov = out.copy()
        pts = [ip(p) for p in hand.pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(ov, pts[a], pts[b], (235, 228, 220), max(1, int(2 * s)), AA)
        for i, p in enumerate(pts):
            if i in FINGERTIPS:
                cv2.circle(ov, p, max(2, int(5 * s)), ACCENT, -1, AA)
            else:
                cv2.circle(ov, p, max(2, int(3 * s)), (255, 255, 255), -1, AA)
        cv2.addWeighted(ov, 0.5, out, 0.5, 0, out)

    def _draw_cursor(self, out, hand, now):
        s = self.s
        if hand is not None and self.smooth is not None:
            c = ip(self.smooth)
            if self.erase_mode:
                r = int(ERASE_RADIUS * s)
                cv2.circle(out, c, r, (255, 255, 255), 2, AA)
                cv2.circle(out, c, int(3 * s), RED, -1, AA)
                put_text(out, "ERASER", (c[0], c[1] + r + int(24 * s)), 0.5 * s, RED, anchor="c", shadow=True)
            elif self.gesture == "WRITE":
                cv2.circle(out, c, int(17 * s), self.ink_color(now), 2, AA)
                cv2.circle(out, c, int(6 * s), (255, 255, 255), -1, AA)
            elif self.gesture == "HOVER":
                cv2.circle(out, c, int(17 * s), (255, 255, 255), 2, AA)
                cv2.circle(out, c, int(3 * s), (255, 255, 255), -1, AA)
            if self.hold_prog > 0 and self.hold_gesture in HOLD_LABEL:
                ctr, r = ip(hand.pts[9]), int(48 * s)
                arc(out, ctr, r, self.hold_prog, AMBER, 4, TRACK)
                put_text(out, HOLD_LABEL[self.hold_gesture], (ctr[0], ctr[1] + r + int(28 * s)),
                         0.6 * s, AMBER, anchor="c", shadow=True)
        # countdown ring while the character is about to be recognised
        if (self.ink_px >= MIN_INK_PIXELS and self.idle > 0.25 and self.last_tip is not None
                and self.hold_gesture is None):
            arc(out, self.last_tip, 26 * s, self.idle / COMMIT_DELAY, AMBER, 3, TRACK)
            put_text(out, "recognising", (self.last_tip[0], self.last_tip[1] + int(48 * s)),
                     0.45 * s, AMBER, font=FONT_S, anchor="c", shadow=True)

    def _draw_hint(self, out, hand):
        if self.text or self.ink_px or self.canvas.keep_any or self.gesture == "WRITE":
            return
        msg = ("Show your hand to the camera" if hand is None
               else "Raise only your index finger and write a letter or digit")
        s = self.s
        shift = int(150 * s) if self.show_panel else 0
        cx = (self.W - shift) // 2
        y = self.H - self.out_h - int(76 * s)
        tw = text_width(msg, 0.6 * s)
        panel(out, cx - tw // 2 - int(24 * s), y - int(26 * s), cx + tw // 2 + int(24 * s), y + int(16 * s),
              alpha=0.7, radius=int(20 * s))
        put_text(out, msg, (cx, y), 0.6 * s, TEXT, anchor="c")

    def _draw_popup(self, out, now):
        if self.popup is None:
            return
        ch, (cx, cy), t0 = self.popup
        age = now - t0
        if age > 1.0:
            self.popup = None
            return
        s = self.s
        scale = (3.2 + 0.8 * age) * s
        th = cv2.getTextSize(ch, FONT, scale, 3)[0][1]
        rise = int(70 * s * age)
        ov = out.copy()
        put_text(ov, ch, (cx, cy + th // 2 - rise), scale, ACCENT, thick=3, anchor="c", shadow=True)
        a = 0.92 * (1.0 - age)
        cv2.addWeighted(ov, a, out, 1.0 - a, 0, out)

    def _draw_top_bar(self, out, now):
        s, W, th = self.s, self.W, self.top_h
        bar(out, 0, th)
        cv2.line(out, (0, th), (W, th), BORDER, 1)

        # brand
        lx, ly = int(32 * s), th // 2
        cv2.circle(out, (lx, ly), int(14 * s), ACCENT, 2, AA)
        cv2.circle(out, (lx, ly), int(5 * s), ACCENT, -1, AA)
        put_text(out, "AIR WRITER", (lx + int(26 * s), ly - int(1 * s)), 0.74 * s, TEXT)
        put_text(out, "gesture text input", (lx + int(27 * s), ly + int(18 * s)), 0.4 * s, MUTED, font=FONT_S)

        # colour palette
        for i, (cx, cy, r) in enumerate(self.swatches):
            c = PALETTE[i][1]
            if c is None:
                draw_rainbow(out, (cx, cy), r)
            elif c == "eraser":
                draw_eraser_icon(out, (cx, cy), r)
            else:
                cv2.circle(out, (cx, cy), r, c, -1, AA)
            cv2.circle(out, (cx, cy), r, BORDER, 1, AA)
            if i == self.color_idx:
                cv2.circle(out, (cx, cy), r + int(6 * s), (255, 255, 255), 2, AA)
            if i == self.dwell_idx and self.dwell_prog > 0:
                arc(out, (cx, cy), r + int(11 * s), self.dwell_prog, ACCENT, 3)

        # status chip + fps
        label, col = STATUS.get("ERASE" if self.erase_mode else self.gesture, STATUS["NONE"])
        chip_h = int(30 * s)
        chip_w = text_width(label, 0.5 * s) + int(50 * s)
        x2 = W - int(24 * s)
        x1, y1 = x2 - chip_w, (th - chip_h) // 2
        panel(out, x1, y1, x2, y1 + chip_h, color=(44, 36, 32), alpha=0.92, radius=chip_h // 2, border=col)
        cv2.circle(out, (x1 + int(17 * s), y1 + chip_h // 2), int(5 * s), col, -1, AA)
        put_text(out, label, (x1 + int(31 * s), y1 + chip_h // 2 + int(5 * s)), 0.5 * s, TEXT)
        put_text(out, f"{self.fps:.0f} FPS", (x1 - int(14 * s), y1 + chip_h // 2 + int(5 * s)),
                 0.44 * s, MUTED, font=FONT_S, anchor="r")

    def _draw_side_panel(self, out):
        s, W, H = self.s, self.W, self.H
        pw = int(286 * s)
        x2 = W - int(20 * s)
        x1 = x2 - pw
        y1 = self.top_h + int(16 * s)
        pad = int(20 * s)
        row_h = int(40 * s)
        y0 = y1 + int(50 * s)
        ly = y0 + len(GESTURE_ROWS) * row_h + int(26 * s)
        has_result = self.last_pred is not None
        y2 = ly + int(18 * s) + int(78 * s) + int(22 * s) if has_result else ly + int(52 * s)
        panel(out, x1, y1, x2, y2, radius=int(16 * s))

        put_text(out, "GESTURES", (x1 + pad, y1 + int(32 * s)), 0.46 * s, MUTED, font=FONT_S)
        for i, (gid, title, desc) in enumerate(GESTURE_ROWS):
            ry = y0 + i * row_h
            active = ("ERASE" if self.erase_mode else self.gesture) == gid
            if active:
                panel(out, x1 + int(8 * s), ry - int(6 * s), x2 - int(8 * s), ry + row_h - int(8 * s),
                      color=(70, 56, 24), alpha=0.55, radius=int(9 * s), border=ACCENT)
            put_text(out, title, (x1 + pad, ry + int(11 * s)), 0.5 * s, TEXT if active else TEXT_DIM)
            put_text(out, desc, (x1 + pad, ry + int(28 * s)), 0.42 * s, ACCENT if active else MUTED, font=FONT_S)

        cv2.line(out, (x1 + pad, ly - int(18 * s)), (x2 - pad, ly - int(18 * s)), BORDER, 1)
        put_text(out, "LAST RESULT", (x1 + pad, ly + int(4 * s)), 0.46 * s, MUTED, font=FONT_S)

        if self.last_pred is None:
            put_text(out, "Nothing recognised yet", (x1 + pad, ly + int(36 * s)), 0.5 * s, MUTED, font=FONT_S)
            return
        char, conf, alts, _ = self.last_pred
        bx, by = x1 + pad, ly + int(18 * s)
        box = int(78 * s)
        panel(out, bx, by, bx + box, by + box, color=(46, 38, 34), alpha=0.95, radius=int(12 * s))
        put_text(out, char, (bx + box // 2, by + box // 2 + int(20 * s)), 2.0 * s, ACCENT, thick=2, anchor="c")
        for i, (c, p) in enumerate(alts[:3]):
            yy = by + int(16 * s) + i * int(26 * s)
            put_text(out, c, (bx + box + int(16 * s), yy + int(6 * s)), 0.6 * s, TEXT)
            bx0 = bx + box + int(40 * s)
            bw = (x2 - pad) - bx0 - int(46 * s)
            cv2.rectangle(out, (bx0, yy - int(4 * s)), (bx0 + bw, yy + int(4 * s)), TRACK, -1)
            fill = int(bw * max(0.0, min(p, 1.0)))
            if fill > 0:
                cv2.rectangle(out, (bx0, yy - int(4 * s)), (bx0 + fill, yy + int(4 * s)),
                              ACCENT if i == 0 else MUTED, -1)
            put_text(out, f"{p * 100:.0f}%", (x2 - pad, yy + int(5 * s)), 0.42 * s,
                     TEXT if i == 0 else MUTED, font=FONT_S, anchor="r")

    def _draw_output(self, out, now):
        s, W, H = self.s, self.W, self.H
        x1, x2 = int(20 * s), W - int(20 * s)
        y2 = H - int(20 * s)
        y1 = y2 - self.out_h
        pad = int(22 * s)
        panel(out, x1, y1, x2, y2, alpha=0.72, radius=int(16 * s))
        put_text(out, "OUTPUT", (x1 + pad, y1 + int(26 * s)), 0.46 * s, MUTED, font=FONT_S)

        right = x2 - pad
        if self.typer is not None:
            label = "OS TYPING ON"
            tw = text_width(label, 0.42 * s, 1, FONT_S)
            panel(out, right - tw - int(24 * s), y1 + int(10 * s), right, y1 + int(34 * s),
                  color=(20, 52, 70), alpha=0.9, radius=int(12 * s), border=AMBER)
            put_text(out, label, (right - int(12 * s), y1 + int(27 * s)), 0.42 * s, AMBER, font=FONT_S, anchor="r")
            right -= tw + int(40 * s)
        n = len(self.text)
        put_text(out, f"{n} char" + ("" if n == 1 else "s"), (right, y1 + int(26 * s)), 0.46 * s, MUTED,
                 font=FONT_S, anchor="r")

        max_w = (x2 - x1) - 2 * pad - int(10 * s)
        scale = 1.05 * s
        shown = self.text[-120:]
        trimmed = len(shown) < len(self.text)
        while shown and text_width(("..." if trimmed else "") + shown, scale) > max_w:
            shown = shown[1:]
            trimmed = True
        shown = ("..." if trimmed and shown else "") + shown
        base_y = y2 - int(20 * s)
        if shown:
            tw = put_text(out, shown, (x1 + pad, base_y), scale, TEXT)
        else:
            tw = 0
            put_text(out, "Your text will appear here", (x1 + pad + int(16 * s), base_y), 0.8 * s, MUTED,
                     font=FONT_S)
        if int(now * 1.6) % 2 == 0:
            cx = x1 + pad + tw + int(5 * s)
            cv2.rectangle(out, (cx, base_y - int(26 * s)), (cx + max(2, int(3 * s)), base_y + int(4 * s)), ACCENT, -1)

    def _draw_toast(self, out, now):
        if self.toast is None:
            return
        msg, col, until = self.toast
        left = until - now
        if left <= 0:
            self.toast = None
            return
        s = self.s
        a = min(1.0, left / 0.35)
        shift = int(150 * s) if self.show_panel else 0
        cx = (self.W - shift) // 2
        y = self.top_h + int(44 * s)
        tw = text_width(msg, 0.62 * s)
        ov = out.copy()
        panel(ov, cx - tw // 2 - int(26 * s), y - int(26 * s), cx + tw // 2 + int(26 * s), y + int(14 * s),
              color=(40, 32, 28), alpha=0.92, radius=int(20 * s), border=col)
        put_text(ov, msg, (cx, y), 0.62 * s, col, anchor="c")
        cv2.addWeighted(ov, a, out, 1.0 - a, 0, out)


# ----------------------------------------------------------------------------- main
def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}. Is another app using it?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    parser = argparse.ArgumentParser(description="Write in the air, get text on screen")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="emnist_cnn.pt")
    parser.add_argument("--type", action="store_true",
                        help="also type recognised characters into the focused application (pyautogui)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    recognizer = CharRecognizer(Path(args.model))

    typer = None
    if args.type:
        import pyautogui  # imported lazily so the app runs without it

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0
        typer = pyautogui

    cap = open_camera(args.camera, args.width, args.height)
    tracker = HandTracker()
    app = AirWriterApp(recognizer, args.width, args.height, typer)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Air Writer started. Press ESC (or q) in the window to quit.")
    if typer is not None:
        print("OS typing is ON: click the application you want to type into.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = fit_cover(cv2.flip(frame, 1), args.width, args.height)  # mirror view
        now = time.time()
        hand = tracker.process(frame)
        app.update(hand, now)
        cv2.imshow(WINDOW, app.render(frame, hand, now))

        if app.handle_key(cv2.waitKey(1) & 0xFF, now):
            break
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()