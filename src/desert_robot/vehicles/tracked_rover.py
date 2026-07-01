from dataclasses import dataclass
from math import radians, tan


@dataclass(frozen=True)
class TrackedRoverConfig:
    """Physical capability envelope for the desert planting tracked rover."""

    body_length_m: float = 1.2
    body_width_m: float = 0.8
    track_width_m: float = 0.18
    ground_clearance_m: float = 0.18
    min_turning_radius_m: float = 0.75
    max_safe_climb_deg: float = 18.0
    max_limit_climb_deg: float = 28.0
    max_side_slope_deg: float = 16.0
    obstacle_clearance_m: float = 0.16
    slope_safety_margin_deg: float = 4.0
    soft_sand_slip_margin: float = 0.2

    @property
    def safe_climb_slope(self) -> float:
        return tan(radians(self.max_safe_climb_deg))

    @property
    def limit_climb_slope(self) -> float:
        return tan(radians(self.max_limit_climb_deg))

    @property
    def side_slope_limit(self) -> float:
        return tan(radians(self.max_side_slope_deg))

    @property
    def conservative_climb_slope(self) -> float:
        return tan(radians(max(0.0, self.max_safe_climb_deg - self.slope_safety_margin_deg)))


def default_tracked_rover() -> TrackedRoverConfig:
    return TrackedRoverConfig()
