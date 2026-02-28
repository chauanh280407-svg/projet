"""Player state: HP, shield, attack cooldown, gesture-hold charge."""

_COOLDOWNS = {"quick": 20, "double": 35, "spread": 40, "heavy": 55}
_GESTURE_TO_ATTACK = {
    "jab":   "quick",
    "kick":  "double",
    "rock":  "spread",
    "punch": "heavy",
}
HOLD_FRAMES = 14   # frames a sign must be held before it fires


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hp = 100
        self.max_hp = 100
        self.hit_flash = 0
        self.shield_active = False
        self.cooldown = 0
        self.charge = 0          # current hold-counter (0..HOLD_FRAMES)
        self.charge_attack = ""  # which attack is charging

    def take_damage(self, damage: int) -> bool:
        """Returns True if KO'd."""
        self.hp = max(0, self.hp - damage)
        self.hit_flash = 18
        return self.hp == 0

    def update_gesture(self, gesture: str) -> str | None:
        """
        Call each frame with the recognised gesture string.
        Returns the attack type to fire, or None.
        """
        self.shield_active = (gesture == "block")

        if self.cooldown > 0:
            self.cooldown -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1

        attack = _GESTURE_TO_ATTACK.get(gesture)
        if attack and self.cooldown == 0:
            if gesture == self.charge_attack:
                self.charge += 1
            else:
                self.charge_attack = gesture
                self.charge = 1

            if self.charge >= HOLD_FRAMES:
                self.charge = 0
                self.cooldown = _COOLDOWNS[attack]
                return attack
        else:
            self.charge = 0
            self.charge_attack = ""

        return None
