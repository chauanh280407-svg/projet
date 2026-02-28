"""
Hand Fighter – camera-overlay edition.

One camera, split down the middle: left half = P1, right half = P2.
Projectiles travel across the screen and hit when they collide with the
opponent's detected face. Players can dodge by moving their head.
"""

import sys
import cv2
import numpy as np

from hand_detector import HandDetector
from gestures import GestureRecognizer
from player import Player, HOLD_FRAMES
from projectile import Projectile

WIN_TITLE = "Hand Fighter  |  Q = quit  |  R = reset"
FPS       = 30
ROUND_SEC = 60

_ATTACK_LABEL = {
    "jab":   "[ 1 finger ] QUICK",
    "kick":  "[ 2 fingers ] DOUBLE",
    "rock":  "[ rock sign ] SPREAD",
    "punch": "[ fist ] HEAVY",
    "block": "[ palm ] SHIELD",
}
_ATTACK_COLOR = {
    "quick":  (0, 255, 255),
    "double": (0, 140, 255),
    "spread": (180, 0, 255),
    "heavy":  (0, 50, 255),
}
_GESTURE_TO_PTYPE = {
    "jab": "quick", "kick": "double", "rock": "spread", "punch": "heavy"
}


class FightingGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.det1 = HandDetector(max_hands=1)
        self.det2 = HandDetector(max_hands=1)
        self.rec  = GestureRecognizer()

        # Face detector (ships with OpenCV, no download needed)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Last known face position in combined-frame coords: (cx, cy, r) or None
        self._face: dict[int, tuple | None] = {0: None, 1: None}

        self.scores = {0: 0, 1: 0}
        self._effects: list[list] = []
        self._new_round()

    # ------------------------------------------------------------------ rounds
    def _new_round(self):
        self.p1 = Player(0)
        self.p2 = Player(1)
        self.projectiles: list[Projectile] = []
        self._effects.clear()
        self.frame_timer = ROUND_SEC * FPS
        self.phase = "playing"
        self.round_msg = ""
        self.round_end_timer = 0

    def _end_round(self, msg: str, winner: int | None = None):
        self.phase = "round_end"
        self.round_msg = msg
        self.round_end_timer = FPS * 3
        if winner is not None:
            self.scores[winner] += 1

    # ----------------------------------------------------------- camera / face
    def _read_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), np.uint8)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        return frame[:, :w // 2].copy(), frame[:, w // 2:].copy()

    def _detect_face(self, frame) -> tuple | None:
        """Return (cx, cy, r) of the largest detected face, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return x + w // 2, y + h // 2, max(w, h) // 2

    def _detect_hand(self, frame, detector, player_id: int):
        """Return (gesture, wrist_y, annotated_frame)."""
        hands, annotated = detector.detect(frame)
        if hands:
            return self.rec.recognize(hands[0], player_id), hands[0]["wrist"][1], annotated
        return "idle", frame.shape[0] // 2, annotated

    # --------------------------------------------------------------- firing
    def _fire(self, player_id: int, attack: str, mid: int, wrist_y: int):
        ptype = _GESTURE_TO_PTYPE.get(attack, "quick")
        x = mid - 5 if player_id == 0 else mid + 5
        direction = +1 if player_id == 0 else -1

        offsets = {
            "quick":  [(0,)],
            "double": [(-18,), (+18,)],
            "spread": [(0,), (-38,), (+38,)],
            "heavy":  [(0,)],
        }
        for (dy,) in offsets[ptype]:
            self.projectiles.append(
                Projectile(x, wrist_y + dy, direction, ptype, player_id)
            )

    # --------------------------------------------------------- hit detection
    def _check_hits(self, mid: int, total_w: int):
        remove = []
        for i, proj in enumerate(self.projectiles):
            px, py = int(proj.x), int(proj.y)

            # Discard if off-screen
            if px < 0 or px > total_w:
                remove.append(i)
                continue

            # Only check collision once the projectile has entered opponent's half
            if proj.owner == 0 and px > mid:
                face = self._face[1]
                if face and self._collides(px, py, proj.radius, face):
                    self._resolve_hit(proj, self.p2, px, py, winner=0)
                    remove.append(i)

            elif proj.owner == 1 and px < mid:
                face = self._face[0]
                if face and self._collides(px, py, proj.radius, face):
                    self._resolve_hit(proj, self.p1, px, py, winner=1)
                    remove.append(i)

        for i in reversed(remove):
            self.projectiles.pop(i)

    @staticmethod
    def _collides(px: int, py: int, pr: int, face: tuple) -> bool:
        cx, cy, fr = face
        return (px - cx) ** 2 + (py - cy) ** 2 <= (fr + pr) ** 2

    def _resolve_hit(self, proj: Projectile, defender: Player,
                     x: int, y: int, winner: int):
        face = self._face[defender.player_id]
        fx, fy = (face[0], face[1]) if face else (x, y)
        if defender.shield_active:
            self._effects.append([fx, fy, proj.radius * 2, (50, 255, 80), 18])
        else:
            ko = defender.take_damage(proj.damage)
            self._effects.append([fx, fy, proj.radius * 5, (0, 60, 255), 25])
            if ko:
                self._end_round(f"Player {winner + 1} wins!", winner=winner)

    # --------------------------------------------------------------- drawing
    def _draw_faces(self, frame, mid: int):
        """Draw face-tracking ring (and shield ring if blocking)."""
        players = [(self.p1, self._face[0], 0), (self.p2, self._face[1], mid)]
        for player, face_local, x_offset in players:
            if face_local is None:
                continue
            cx, cy, r = face_local
            # Offset P2's face into combined-frame space for drawing
            draw_cx = cx + x_offset

            if player.shield_active:
                # Thick green shield ring
                cv2.circle(frame, (draw_cx, cy), r + 18, (50, 255, 80), 4)
                cv2.circle(frame, (draw_cx, cy), r + 28, (50, 255, 80), 1)
            else:
                # Thin tracking ring
                cv2.circle(frame, (draw_cx, cy), r + 8, (200, 200, 200), 1)

    def _draw_overlay(self, frame, total_w: int, h: int, mid: int):
        # Dark header band
        roi = frame[0:82, :]
        cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0, roi)

        self._draw_hp_bar(frame, self.p1, 10,       10, mid - 20, 26, left=True)
        self._draw_hp_bar(frame, self.p2, mid + 10, 10, mid - 20, 26, left=False)

        # Timer
        secs = max(0, self.frame_timer // FPS)
        t = str(secs)
        (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
        cv2.putText(frame, t, (mid - tw // 2, 54),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 215, 0), 3)

        # Score
        sc = f"{self.scores[0]}  –  {self.scores[1]}"
        (sw, _), _ = cv2.getTextSize(sc, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, sc, (mid - sw // 2, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 160), 2)

        # Dividing line
        cv2.line(frame, (mid, 0), (mid, h), (200, 200, 200), 2)

        # Gesture charge HUD
        self._draw_gesture_hud(frame, self.p1, 10,       h)
        self._draw_gesture_hud(frame, self.p2, mid + 10, h)

        # Hit-flash overlay
        for player, x0, x1 in [(self.p1, 0, mid), (self.p2, mid, total_w)]:
            if player.hit_flash > 0:
                roi = frame[:, x0:x1]
                flash = np.zeros_like(roi)
                flash[:] = (0, 0, 180)
                alpha = player.hit_flash / 18 * 0.5
                cv2.addWeighted(flash, alpha, roi, 1 - alpha, 0, roi)
                frame[:, x0:x1] = roi

    def _draw_hp_bar(self, frame, player: Player,
                     x: int, y: int, w: int, h: int, left: bool):
        ratio = player.hp / player.max_hp
        filled = int(w * ratio)
        col = (30, 200, 30) if ratio > 0.5 else (0, 200, 200) if ratio > 0.25 else (0, 0, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
        if left:
            cv2.rectangle(frame, (x, y), (x + filled, y + h), col, -1)
        else:
            cv2.rectangle(frame, (x + w - filled, y), (x + w, y + h), col, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 2)
        cv2.putText(frame, f"P{player.player_id + 1}  {player.hp} HP",
                    (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 225, 255), 1)

    def _draw_gesture_hud(self, frame, player: Player, x: int, h: int):
        gesture = player.charge_attack
        if player.shield_active:
            cv2.putText(frame, _ATTACK_LABEL["block"], (x, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 80), 2)
            return
        if not gesture:
            return
        label = _ATTACK_LABEL.get(gesture, "")
        if not label:
            return
        ptype = _GESTURE_TO_PTYPE.get(gesture, "quick")
        col = _ATTACK_COLOR[ptype]
        bar_w = 160
        filled = int(bar_w * player.charge / HOLD_FRAMES)
        by = h - 36
        cv2.rectangle(frame, (x, by), (x + bar_w, by + 14), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, by), (x + filled, by + 14), col, -1)
        cv2.rectangle(frame, (x, by), (x + bar_w, by + 14), (140, 140, 140), 1)
        cv2.putText(frame, label, (x, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    def _draw_effects(self, frame):
        next_fx = []
        for fx in self._effects:
            x, y, r, color, frames = fx
            cv2.circle(frame, (x, y), r, color, 3)
            cv2.circle(frame, (x, y), r // 2,
                       tuple(min(255, c + 80) for c in color), -1)
            if frames > 1:
                fx[2] += 3
                fx[4] -= 1
                next_fx.append(fx)
        self._effects = next_fx

    def _draw_round_end(self, frame, total_w: int, h: int):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (total_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        (tw, _), _ = cv2.getTextSize(
            self.round_msg, cv2.FONT_HERSHEY_DUPLEX, 1.9, 4)
        cv2.putText(frame, self.round_msg,
                    (total_w // 2 - tw // 2, h // 2 - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.9, (0, 215, 255), 4)
        sub = "Next round starting..."
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        cv2.putText(frame, sub, (total_w // 2 - sw // 2, h // 2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
        sc = f"Score   P1: {self.scores[0]}   P2: {self.scores[1]}"
        (ssw, _), _ = cv2.getTextSize(sc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(frame, sc, (total_w // 2 - ssw // 2, h // 2 + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 160), 2)

    # ---------------------------------------------------------------- loop
    def run(self):
        while True:
            f1, f2 = self._read_frames()
            h, mid = f1.shape[0], f1.shape[1]

            # Detect hands
            g1, wy1, f1 = self._detect_hand(f1, self.det1, 0)
            g2, wy2, f2 = self._detect_hand(f2, self.det2, 1)

            # Detect faces (keep last known position if not detected this frame)
            face1 = self._detect_face(f1)
            face2 = self._detect_face(f2)
            if face1:
                self._face[0] = face1
            if face2:
                # Offset into combined-frame coords for hit detection
                cx, cy, r = face2
                self._face[1] = (cx + mid, cy, r)

            combined = np.hstack([f1, f2])
            total_w = combined.shape[1]

            if self.phase == "playing":
                fire1 = self.p1.update_gesture(g1)
                fire2 = self.p2.update_gesture(g2)

                if fire1:
                    self._fire(0, fire1, mid, wy1)
                if fire2:
                    self._fire(1, fire2, mid, wy2)

                for proj in self.projectiles:
                    proj.update()
                self._check_hits(mid, total_w)

                self.frame_timer -= 1
                if self.frame_timer <= 0:
                    if self.p1.hp > self.p2.hp:
                        self._end_round("Player 1 wins! (Time)", winner=0)
                    elif self.p2.hp > self.p1.hp:
                        self._end_round("Player 2 wins! (Time)", winner=1)
                    else:
                        self._end_round("Draw!")

            elif self.phase == "round_end":
                self.round_end_timer -= 1
                if self.round_end_timer <= 0:
                    self._new_round()

            # Draw game elements on top of camera feed
            self._draw_faces(combined, mid)
            for proj in self.projectiles:
                proj.draw(combined)
            self._draw_effects(combined)
            self._draw_overlay(combined, total_w, h, mid)

            if self.phase == "round_end":
                self._draw_round_end(combined, total_w, h)

            cv2.imshow(WIN_TITLE, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                self.scores = {0: 0, 1: 0}
                self._new_round()

        self._quit()

    def _quit(self):
        self.cap.release()
        self.det1.release()
        self.det2.release()
        cv2.destroyAllWindows()
        sys.exit()
