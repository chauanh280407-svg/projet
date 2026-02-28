import cv2


class Projectile:
    TYPES = {
        "quick":  {"speed": 22, "damage": 8,  "radius": 8,  "color": (0, 255, 255)},
        "double": {"speed": 15, "damage": 12, "radius": 11, "color": (0, 140, 255)},
        "spread": {"speed": 13, "damage": 10, "radius": 9,  "color": (180, 0, 255)},
        "heavy":  {"speed": 7,  "damage": 25, "radius": 20, "color": (0, 50, 255)},
    }

    def __init__(self, x: float, y: float, direction: int, ptype: str, owner: int):
        self.x = float(x)
        self.y = float(y)
        self.direction = direction   # +1 = right (P1), -1 = left (P2)
        self.ptype = ptype
        self.owner = owner
        data = self.TYPES[ptype]
        self.speed  = data["speed"]
        self.damage = data["damage"]
        self.radius = data["radius"]
        self.color  = data["color"]
        self._trail: list[tuple[int, int]] = []

    def update(self):
        self._trail.append((int(self.x), int(self.y)))
        if len(self._trail) > 7:
            self._trail.pop(0)
        self.x += self.speed * self.direction

    def draw(self, frame):
        for i, pt in enumerate(self._trail):
            alpha = (i + 1) / len(self._trail)
            r = max(2, int(self.radius * 0.55 * alpha))
            col = tuple(int(c * alpha * 0.45) for c in self.color)
            cv2.circle(frame, pt, r, col, -1)

        pos = (int(self.x), int(self.y))
        cv2.circle(frame, pos, self.radius, self.color, -1)
        cv2.circle(frame, pos, max(2, self.radius // 2), (255, 255, 255), -1)
        cv2.circle(frame, pos, self.radius + 5,
                   tuple(c // 3 for c in self.color), 2)
