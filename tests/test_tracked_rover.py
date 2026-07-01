from desert_robot.vehicles import TrackedRoverConfig, default_tracked_rover


def test_default_tracked_rover_has_ordered_slope_limits() -> None:
    rover = default_tracked_rover()

    assert isinstance(rover, TrackedRoverConfig)
    assert rover.conservative_climb_slope < rover.safe_climb_slope < rover.limit_climb_slope
    assert rover.body_width_m > 2.0 * rover.track_width_m
    assert rover.ground_clearance_m >= rover.obstacle_clearance_m


def test_tracked_rover_rejects_negative_conservative_margin_by_clamping() -> None:
    rover = TrackedRoverConfig(max_safe_climb_deg=3.0, slope_safety_margin_deg=6.0)

    assert rover.conservative_climb_slope == 0.0
