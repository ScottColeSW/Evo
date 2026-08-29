import numpy as np


class AncestralTraumaMatrix:
    """A per-tile emotional scar layer, separate from terrain.

    Triumphant events raise a tile's score, traumatic events lower it, and both decay
    outward with a quadratic falloff so the epicenter is affected far more than the
    edge of the radius. The resulting score is translated into a short bias string
    injected into a tribe's prompt near that coordinate — it nudges temperature and
    framing, it never dictates the action directly.
    """

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.ghost_tensor = np.zeros((grid_size, grid_size), dtype=np.float32)

    def radiate_event_wave(self, x_e: int, y_e: int, magnitude: float, radius: int = 5) -> None:
        x_min, x_max = max(0, x_e - radius), min(self.grid_size, x_e + radius + 1)
        y_min, y_max = max(0, y_e - radius), min(self.grid_size, y_e + radius + 1)
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                distance = ((x - x_e) ** 2 + (y - y_e) ** 2) ** 0.5
                if distance <= radius:
                    attenuation = (1.0 - (distance / radius)) ** 2
                    self.ghost_tensor[x, y] = np.clip(
                        self.ghost_tensor[x, y] + magnitude * attenuation, -1.0, 1.0
                    )

    def bias_string(self, x: int, y: int) -> str:
        score = float(self.ghost_tensor[x, y])
        if score <= -0.35:
            return (
                f"[ANCESTRAL DREAD // ghost score {score:.2f}] Your lineage remembers "
                "suffering at this ground. Favor fortification (CONSTRUCT_WALL, BUILD_FIRE) "
                "or move away from these coordinates."
            )
        if score >= 0.35:
            return (
                f"[ANCESTRAL PRIDE // ghost score {score:.2f}] Your lineage remembers "
                "triumph at this ground. You feel emboldened to settle and defend it."
            )
        return ""
