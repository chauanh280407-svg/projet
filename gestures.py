"""
Sign-based gesture recognizer (no swipe velocity needed).

Hand signs → game actions:
  ☝  Index only              → jab    (quick shot)
  ✌  Index + Middle          → kick   (double shot)
  🤘  Index + Pinky           → rock   (spread shot)
  ✊  Fist (0-1 fingers)      → punch  (heavy shot)
  🖐  Open palm (4-5 fingers) → block  (shield)
"""


class GestureRecognizer:
    def _finger_states(self, lm: list) -> list[bool]:
        """[thumb, index, middle, ring, pinky] — True = extended."""
        thumb = lm[4][0] < lm[3][0]
        others = [lm[tip][1] < lm[pip][1]
                  for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]]
        return [thumb] + others

    def recognize(self, hand_data: dict, player_id: int) -> str:
        fingers = self._finger_states(hand_data["landmarks"])
        n_up = sum(fingers)
        _, idx, mid, ring, pinky = fingers

        if n_up >= 4:
            return "block"

        if n_up <= 1:
            return "punch"

        # Index + pinky (rock horns 🤘)
        if idx and not mid and not ring and pinky:
            return "rock"

        # Index + middle (peace ✌)
        if idx and mid and not ring and not pinky:
            return "kick"

        # Index only (point ☝)
        if idx and not mid and not ring and not pinky:
            return "jab"

        return "idle"
