"""Tests for physics calculations."""



from webcam_putting.physics import (
    GOLF_BALL_RADIUS_MM,
    calculate_angle,
    calculate_shot,
    fit_trajectory,
    pixel_to_mm_ratio,
)


class TestPixelToMmRatio:
    def test_standard_ball(self):
        # If detected radius is 21 pixels, ratio = 21 / 21.335 ≈ 0.984
        ratio = pixel_to_mm_ratio(21)
        assert abs(ratio - 21 / GOLF_BALL_RADIUS_MM) < 0.001

    def test_zero_radius(self):
        assert pixel_to_mm_ratio(0) == 0.0

    def test_negative_radius(self):
        assert pixel_to_mm_ratio(-5) == 0.0


class TestCalculateAngle:
    def test_straight_right(self):
        """Ball going straight right should be ~0 degrees."""
        angle = calculate_angle((100, 200), (200, 200))
        assert abs(angle) < 1.0

    def test_straight_up(self):
        """Ball going straight up (decreasing Y) should be ~90 degrees."""
        angle = calculate_angle((100, 200), (100, 100))
        assert abs(angle - 90.0) < 1.0

    def test_straight_down(self):
        """Ball going straight down should be ~-90 degrees."""
        angle = calculate_angle((100, 100), (100, 200))
        assert abs(angle - (-90.0)) < 1.0

    def test_flip_inverts_angle(self):
        angle_normal = calculate_angle((100, 200), (200, 100))
        angle_flipped = calculate_angle((100, 200), (200, 100), flip=True)
        assert abs(angle_normal + angle_flipped) < 0.1


class TestFitTrajectory:
    def test_straight_line(self):
        """Perfectly straight trajectory should have no outliers removed."""
        positions = [(100, 200, 0.0), (200, 200, 0.1), (300, 200, 0.2), (400, 200, 0.3)]
        result = fit_trajectory(positions)
        assert result is not None
        x, y, t = result
        assert len(x) == 4

    def test_with_outlier(self):
        """One outlier should be removed."""
        positions = [
            (100, 200, 0.0),
            (200, 200, 0.1),
            (300, 500, 0.2),  # Outlier: y jumps to 500
            (400, 200, 0.3),
            (500, 200, 0.4),
        ]
        result = fit_trajectory(positions)
        assert result is not None
        x, y, t = result
        assert len(x) < 5  # Outlier should be removed

    def test_insufficient_points(self):
        assert fit_trajectory([(100, 200, 0.0)]) is None
        assert fit_trajectory([]) is None

    def test_two_points_no_rejection(self):
        """With only 2 points, no outlier rejection possible."""
        result = fit_trajectory([(100, 200, 0.0), (200, 210, 0.1)])
        assert result is not None
        x, y, t = result
        assert len(x) == 2


class TestCalculateShot:
    def test_straight_putt(self):
        """Ball going straight right at known speed."""
        # Ball travels 100mm in 0.5 seconds
        # Speed: 100mm / 0.5s = 200mm/s = 0.2m/s = 0.72km/h = 0.447 mph
        ratio = pixel_to_mm_ratio(21)  # ~0.984 px/mm
        px_distance = 100 * ratio  # pixels for 100mm

        start = (100, 200)
        end = (100 + int(px_distance), 200)

        result = calculate_shot(
            start_pos=start,
            end_pos=end,
            entry_time=0.0,
            exit_time=0.5,
            px_mm_ratio=ratio,
        )

        assert result is not None
        assert abs(result.distance_mm - 100.0) < 5.0  # Within 5mm
        assert abs(result.hla_degrees) < 2.0  # Straight putt
        assert result.speed_mph > 0

    def test_angled_putt(self):
        """Ball going at 45 degrees should report ~45 degree HLA."""
        ratio = pixel_to_mm_ratio(21)
        start = (100, 200)
        # Move 100px right and 100px up (negative Y direction)
        end = (200, 100)

        result = calculate_shot(
            start_pos=start,
            end_pos=end,
            entry_time=0.0,
            exit_time=0.5,
            px_mm_ratio=ratio,
        )

        assert result is not None
        assert abs(result.hla_degrees - 45.0) < 2.0

    def test_zero_time_returns_none(self):
        result = calculate_shot(
            start_pos=(100, 200),
            end_pos=(200, 200),
            entry_time=1.0,
            exit_time=1.0,
            px_mm_ratio=1.0,
        )
        assert result is None

    def test_zero_ratio_returns_none(self):
        result = calculate_shot(
            start_pos=(100, 200),
            end_pos=(200, 200),
            entry_time=0.0,
            exit_time=0.5,
            px_mm_ratio=0.0,
        )
        assert result is None

    def test_with_trajectory_positions(self):
        """Using multi-point positions should still produce valid results."""
        ratio = pixel_to_mm_ratio(21)
        positions = [
            (100, 200, 0.0),
            (150, 198, 0.1),
            (200, 197, 0.2),
            (250, 196, 0.3),
            (300, 195, 0.4),
        ]

        result = calculate_shot(
            start_pos=(100, 200),
            end_pos=(300, 195),
            entry_time=0.0,
            exit_time=0.4,
            px_mm_ratio=ratio,
            positions=positions,
        )

        assert result is not None
        assert result.speed_mph > 0
        assert result.distance_mm > 0

    def test_reverse_x_straight_putt(self):
        """RtL straight putt (decreasing x) should produce valid speed with ~0 HLA."""
        ratio = pixel_to_mm_ratio(21)
        px_distance = 100 * ratio

        # Ball moves LEFT (decreasing x)
        start = (500, 200)
        end = (500 - int(px_distance), 200)

        result = calculate_shot(
            start_pos=start,
            end_pos=end,
            entry_time=0.0,
            exit_time=0.5,
            px_mm_ratio=ratio,
            reverse_x=True,
        )

        assert result is not None
        assert abs(result.distance_mm - 100.0) < 5.0
        assert abs(result.hla_degrees) < 2.0
        assert result.speed_mph > 0

    def test_reverse_x_angled_putt(self):
        """RtL angled putt should report correct HLA."""
        ratio = pixel_to_mm_ratio(21)
        # Ball moves left and up (decreasing x, decreasing y)
        start = (500, 200)
        end = (400, 100)

        result = calculate_shot(
            start_pos=start,
            end_pos=end,
            entry_time=0.0,
            exit_time=0.5,
            px_mm_ratio=ratio,
            reverse_x=True,
        )

        assert result is not None
        assert abs(result.hla_degrees - 45.0) < 2.0
