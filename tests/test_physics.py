"""Tests for physics calculations."""



from birdman_putting.physics import (
    GOLF_BALL_RADIUS_MM,
    calculate_angle,
    calculate_shot,
    estimate_putt_distance_feet,
    fit_trajectory,
    pixel_to_mm_ratio,
    pixel_x_to_feet,
    ppf_from_ball_radius,
    speed_from_launch_velocity,
    speed_from_trajectory_fit,
    target_speed_for_distance,
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


class TestEstimatePuttDistance:
    def test_stimp_10_moderate_putt(self) -> None:
        """5 MPH on stimp 10 should be roughly 14-15 feet."""
        dist = estimate_putt_distance_feet(5.0, stimpmeter=10.0)
        assert 14.0 < dist < 16.0

    def test_stimp_10_short_putt(self) -> None:
        """3 MPH on stimp 10 should be roughly 5 feet."""
        dist = estimate_putt_distance_feet(3.0, stimpmeter=10.0)
        assert 4.5 < dist < 6.0

    def test_higher_stimp_rolls_farther(self) -> None:
        dist_10 = estimate_putt_distance_feet(5.0, stimpmeter=10.0)
        dist_13 = estimate_putt_distance_feet(5.0, stimpmeter=13.0)
        assert dist_13 > dist_10

    def test_zero_speed(self) -> None:
        assert estimate_putt_distance_feet(0.0) == 0.0


class TestSpeedFromLaunchVelocity:
    """Verify velocity-based speed measurement on synthetic position tracks.

    Setup matches Greg's rig: pixels_per_foot ≈ 30, camera at 60fps,
    stimp 11. The ball rolls horizontally (along +x).
    """

    PPF = 30.0
    FPS = 60.0
    DT = 1.0 / FPS  # 16.67ms per frame

    def _track(self, launch_mph: float, frames: int = 10) -> list[tuple[int, int, float]]:
        """Build a synthetic position track for a putt at a given launch velocity.

        First 3 frames are at rest (ball stable), then ball accelerates
        from rest to launch velocity over one frame, then constant-velocity
        thereafter (deceleration is negligible over ~100ms).
        """
        positions: list[tuple[int, int, float]] = []
        # Rest frames
        for i in range(3):
            positions.append((100, 200, i * self.DT))
        # Motion frames: constant-velocity roll at launch speed
        launch_fps = launch_mph * 5280.0 / 3600.0
        px_per_frame = launch_fps * self.PPF * self.DT
        for i in range(frames):
            x = int(100 + (i + 1) * px_per_frame)
            t = (3 + i) * self.DT
            positions.append((x, 200, t))
        return positions

    def test_zero_with_no_positions(self) -> None:
        speed, _dbg = speed_from_launch_velocity([], self.PPF)
        assert speed == 0.0

    def test_zero_with_invalid_ppf(self) -> None:
        track = self._track(4.0)
        assert speed_from_launch_velocity(track, 0.0)[0] == 0.0
        assert speed_from_launch_velocity(track, -1.0)[0] == 0.0

    def test_zero_when_ball_never_moves(self) -> None:
        stationary = [(100, 200, i * self.DT) for i in range(10)]
        speed, _dbg = speed_from_launch_velocity(stationary, self.PPF)
        assert speed == 0.0

    def test_short_putt_roundtrip(self) -> None:
        """A 5 ft putt at stimp 11 needs ~2.79 MPH — should measure ~same back."""
        launch_mph = target_speed_for_distance(5.0, stimpmeter=11.0)
        track = self._track(launch_mph, frames=10)
        measured, _dbg = speed_from_launch_velocity(track, self.PPF)
        assert abs(measured - launch_mph) < 0.3, (
            f"measured {measured:.2f} vs expected {launch_mph:.2f}"
        )

    def test_long_putt_does_not_saturate(self) -> None:
        """A 30 ft putt at stimp 11 needs ~6.82 MPH.

        The key property: velocity-based reads launch speed directly from
        early frames, not total roll distance. So 30 ft and 15 ft putts
        produce different measurements even when both exit the frame.
        """
        launch_mph = target_speed_for_distance(30.0, stimpmeter=11.0)
        track = self._track(launch_mph, frames=10)
        measured, _dbg = speed_from_launch_velocity(track, self.PPF)
        assert abs(measured - launch_mph) < 0.3, (
            f"measured {measured:.2f} vs expected {launch_mph:.2f}"
        )

    def test_harder_putt_reads_higher(self) -> None:
        """Sanity: hitting harder must produce a higher reading."""
        soft = self._track(target_speed_for_distance(10.0, stimpmeter=11.0))
        hard = self._track(target_speed_for_distance(40.0, stimpmeter=11.0))
        v_soft, _ = speed_from_launch_velocity(soft, self.PPF)
        v_hard, _ = speed_from_launch_velocity(hard, self.PPF)
        assert v_hard > v_soft * 1.5

    def test_sparse_hard_putt_jump_past_gateway(self) -> None:
        """Hard putt that jumps past gateway in one frame still measures."""
        # 3 rest frames + 1 detection mid-flight — simulates the
        # "jumped past gateway" path where detection misses intermediate frames
        rest = [(100, 200, i * self.DT) for i in range(3)]
        # Ball at ~6 MPH moves ~4 px/frame at ppf=30 — but if we miss 2 frames,
        # the one detection shows 12px of travel over 3 frame intervals (50ms)
        one_motion = [(100 + 12, 200, 5 * self.DT)]
        track = rest + one_motion
        measured, dbg = speed_from_launch_velocity(track, self.PPF)
        # Should produce SOME positive reading using (last rest -> motion) anchor
        assert measured > 0.0, f"expected positive reading, got {measured} ({dbg})"
        # And it should be physically reasonable (< 20 MPH cap)
        assert measured < 20.0

    def test_stale_rest_timestamp_does_not_dilute_velocity(self) -> None:
        """Rest frame timestamp is older than motion onset (re-start fired
        long before user putted). Motion-only window must ignore it."""
        # Rest frame with old timestamp (0.5s before motion starts)
        track: list[tuple[int, int, float]] = [(100, 200, 0.0)]
        # Then 6 motion frames, ball moving at 4 MPH = 5.87 fps
        # = 5.87 × 30 ppf / 60 fps = 2.93 px/frame → use 8 px/frame for clarity
        # Real motion launch velocity computation:
        launch_mph = target_speed_for_distance(15.0, stimpmeter=11.0)
        launch_fps = launch_mph * 5280 / 3600
        px_per_frame = launch_fps * self.PPF * self.DT
        # Motion starts at t=0.5s (0.5s of stale rest time)
        for i in range(6):
            x = int(100 + (i + 1) * px_per_frame)
            t = 0.5 + (i + 1) * self.DT
            track.append((x, 200, t))

        measured, dbg = speed_from_launch_velocity(track, self.PPF)

        # Should read close to actual launch velocity, NOT diluted by the
        # 0.5s stale rest gap. Old (buggy) code would compute over 0.5s+
        # window. New code uses motion-only frames.
        assert abs(measured - launch_mph) < 0.5, (
            f"stale rest dilution: measured {measured:.2f} vs expected "
            f"{launch_mph:.2f} (window dt={dbg['window_dt']:.3f}s)"
        )

    def test_rejects_dropped_frame_artifact(self) -> None:
        """If detection skips 20+ frames then pops up 400px away in 1 frame
        interval, velocity would be absurdly high — should be rejected."""
        # Rest frames at t=0..0.16s, then 400px jump at the very next frame
        rest = [(100, 200, i * self.DT) for i in range(10)]
        jump = [(500, 200, 10 * self.DT + self.DT)]  # 1/60s after last rest
        track = rest + jump
        measured, dbg = speed_from_launch_velocity(track, self.PPF)
        # 400px / 30 ppf = 13.3 ft in 0.0167s = 800 fps = 545 MPH → rejected
        assert measured == 0.0
        assert "cap" in dbg["reason"]

    def test_dt_below_minimum_returns_zero(self) -> None:
        """If all early-motion samples land within 5ms, timing is unreliable."""
        # Build a track where all motion frames are within 1ms of each other
        positions: list[tuple[int, int, float]] = []
        for i in range(3):
            positions.append((100, 200, i * 0.0001))
        for i in range(6):
            positions.append((100 + (i + 1) * 15, 200, 0.0003 + (i + 1) * 0.0001))
        speed, _dbg = speed_from_launch_velocity(positions, self.PPF)
        assert speed == 0.0


class TestPpfFromBallRadius:
    """Verify auto-calibration math from detected ball radius."""

    def test_zero_radius_returns_zero(self) -> None:
        assert ppf_from_ball_radius(0) == 0.0

    def test_negative_radius_returns_zero(self) -> None:
        assert ppf_from_ball_radius(-5) == 0.0

    def test_known_radius_gives_expected_ppf(self) -> None:
        """A detected radius of 10 px → ppf ≈ 142.86."""
        ppf = ppf_from_ball_radius(10.0)
        # 304.8 mm/ft / 21.335 mm/radius * 10 px/radius = 142.86 px/ft
        assert abs(ppf - 142.86) < 0.1

    def test_typical_greg_rig_radius(self) -> None:
        """Greg's ball radius fluctuates 7-13 px.  Sanity-check both ends."""
        ppf_small = ppf_from_ball_radius(7.0)
        ppf_large = ppf_from_ball_radius(13.0)
        # ~100 and ~186 — both physically plausible for a mat-top webcam
        assert 90 < ppf_small < 110
        assert 170 < ppf_large < 200
        # And scales linearly
        assert abs(ppf_large / ppf_small - 13 / 7) < 0.01

    def test_ppf_roundtrips_through_stimp_math(self) -> None:
        """Using ppf-from-radius should give reasonable putt projections.

        Simulate a 10ft putt by building a position track at the right
        launch velocity, then verify the measurement round-trips.
        """
        radius = 10  # px — gives ppf ≈ 142.86
        ppf = ppf_from_ball_radius(radius)

        # Real 10 ft putt at stimp 11 needs ~3.95 MPH launch
        launch_mph = target_speed_for_distance(10.0, stimpmeter=11.0)
        launch_fps = launch_mph * 5280 / 3600
        dt = 1 / 60
        px_per_frame = launch_fps * ppf * dt

        positions: list[tuple[int, int, float]] = []
        for i in range(3):
            positions.append((100, 200, i * dt))
        for i in range(10):
            positions.append((int(100 + (i + 1) * px_per_frame), 200, (3 + i) * dt))

        measured, _ = speed_from_launch_velocity(positions, ppf)
        projected_ft = estimate_putt_distance_feet(measured, stimpmeter=11.0)

        # Should project back to ~10 ft (within reason)
        assert 9.0 < projected_ft < 11.0, (
            f"expected ~10ft projection, got {projected_ft:.1f}ft "
            f"(measured {measured:.2f} MPH)"
        )


class TestSpeedFromTrajectoryFit:
    """The trajectory-fit estimator solves x(t) = v0·t − ½·a·t² over
    all motion samples. Should be accurate when given a realistic
    decelerating trajectory.
    """

    PPF = 60.0
    FPS = 60.0
    STIMP = 11.0
    DT = 1.0 / FPS

    def _decelerating_track(
        self, launch_mph: float, n_motion_frames: int = 20,
        n_rest_frames: int = 3,
    ) -> list[tuple[int, int, float]]:
        """Synthesize a track of a ball decelerating at the configured stimp."""
        import math as _math

        a = (6.08 ** 2) / (2 * self.STIMP)  # ft/s² decel
        launch_fps = launch_mph * 5280 / 3600

        positions: list[tuple[int, int, float]] = []
        # Rest frames at x=100, y=200
        for i in range(n_rest_frames):
            positions.append((100, 200, i * self.DT))

        # Motion frames: x(t) = 100 + (v0·t − ½·a·t²) · ppf
        t_start = n_rest_frames * self.DT
        for i in range(n_motion_frames):
            tau = (i + 1) * self.DT  # seconds since first motion frame
            travel_ft = launch_fps * tau - 0.5 * a * tau * tau
            x = int(100 + travel_ft * self.PPF)
            positions.append((x, 200, t_start + tau))
        return positions

    def _bare_check(self, launch_mph: float, n_motion: int = 20,
                    tolerance_mph: float = 0.4) -> None:
        track = self._decelerating_track(launch_mph, n_motion_frames=n_motion)
        measured, dbg = speed_from_trajectory_fit(track, self.PPF, self.STIMP)
        assert dbg["reason"] == "ok", f"unexpected reason: {dbg}"
        assert abs(measured - launch_mph) < tolerance_mph, (
            f"measured {measured:.2f} vs expected {launch_mph:.2f} ({dbg})"
        )

    def test_short_putt_3mph(self) -> None:
        self._bare_check(3.0)

    def test_medium_putt_5mph(self) -> None:
        self._bare_check(5.0)

    def test_hard_putt_8mph(self) -> None:
        self._bare_check(8.0)

    def test_zero_with_no_positions(self) -> None:
        speed, dbg = speed_from_trajectory_fit([], self.PPF, self.STIMP)
        assert speed == 0.0
        assert "2+" in dbg["reason"]

    def test_zero_with_invalid_ppf(self) -> None:
        track = self._decelerating_track(4.0)
        speed, dbg = speed_from_trajectory_fit(track, 0.0, self.STIMP)
        assert speed == 0.0
        assert "ppf" in dbg["reason"]

    def test_zero_when_ball_never_moves(self) -> None:
        positions = [(100, 200, i * self.DT) for i in range(15)]
        speed, dbg = speed_from_trajectory_fit(positions, self.PPF, self.STIMP)
        assert speed == 0.0
        assert "no motion" in dbg["reason"]

    def test_too_few_motion_frames(self) -> None:
        track = self._decelerating_track(4.0, n_motion_frames=2)
        speed, dbg = speed_from_trajectory_fit(track, self.PPF, self.STIMP)
        # Only 2 motion frames is below _FIT_MIN_FRAMES (4)
        assert speed == 0.0
        assert "motion frames" in dbg["reason"]

    def test_round_trips_through_distance_estimate(self) -> None:
        """Synthesize a 10-ft putt, fit it, project back — should give ~10ft."""
        target_ft = 10.0
        launch_mph = target_speed_for_distance(target_ft, stimpmeter=self.STIMP)
        # Real putts only show ~3 ft of motion before exiting frame; simulate
        track = self._decelerating_track(launch_mph, n_motion_frames=15)
        measured, _ = speed_from_trajectory_fit(track, self.PPF, self.STIMP)
        projected = estimate_putt_distance_feet(measured, stimpmeter=self.STIMP)
        assert abs(projected - target_ft) < 0.3, (
            f"expected ~{target_ft}ft, got {projected:.2f}ft"
        )

    def test_fit_is_more_consistent_than_window_estimator(self) -> None:
        """Two synthetic putts at the SAME launch speed but with
        different trail-density patterns should both fit close to the
        true launch speed.  Window estimator can vary by 25%+ across
        these patterns."""
        launch_mph = 4.0
        # Pattern A: 20 dense motion frames
        track_dense = self._decelerating_track(launch_mph, n_motion_frames=20)
        # Pattern B: same putt but only every-other motion frame captured
        track_sparse_full = self._decelerating_track(launch_mph, n_motion_frames=20)
        rest = [p for p in track_sparse_full if p[0] == 100]  # 3 rest frames
        motion = [p for p in track_sparse_full if p[0] != 100]
        track_sparse = rest + motion[::2]  # every other motion frame

        v_dense, _ = speed_from_trajectory_fit(track_dense, self.PPF, self.STIMP)
        v_sparse, _ = speed_from_trajectory_fit(track_sparse, self.PPF, self.STIMP)
        # Both should be very close to the true launch
        assert abs(v_dense - launch_mph) < 0.3
        assert abs(v_sparse - launch_mph) < 0.5  # slightly more tolerance for sparse


class TestPixelXToFeet:
    """Verify the per-x calibration helper handles fisheye correctly."""

    def test_no_markers_uses_flat_ppf(self) -> None:
        """Without markers, return x / fallback_ppf."""
        assert pixel_x_to_feet(60.0, None, 60.0) == 1.0
        assert pixel_x_to_feet(120.0, None, 60.0) == 2.0
        assert pixel_x_to_feet(0.0, [], 60.0) == 0.0

    def test_uniform_markers(self) -> None:
        """Markers at exactly 50px spacing → ft = (x - markers[0]) / 50."""
        markers = [100.0, 150.0, 200.0, 250.0]
        assert pixel_x_to_feet(100.0, markers, 60.0) == 0.0
        assert pixel_x_to_feet(150.0, markers, 60.0) == 1.0
        assert pixel_x_to_feet(200.0, markers, 60.0) == 2.0
        assert pixel_x_to_feet(250.0, markers, 60.0) == 3.0
        # Mid-segment interpolation
        assert pixel_x_to_feet(125.0, markers, 60.0) == 0.5
        assert pixel_x_to_feet(175.0, markers, 60.0) == 1.5

    def test_fisheye_pattern(self) -> None:
        """Greg's actual cork data: spacings 53, 57, 57.5, 57.5, 55, 51, 45.5.

        Marker at index i is at exactly i feet from the first marker;
        we just have non-uniform pixel spacing.
        """
        # Starting at x=172 (Greg's leftmost cork)
        markers = [172.0, 225.0, 282.0, 339.5, 397.0, 452.0, 503.0, 548.5]
        # Each marker is i feet from the first
        for i, m in enumerate(markers):
            assert abs(pixel_x_to_feet(m, markers, 60.0) - i) < 0.001, (
                f"marker at x={m} should be at {i}ft"
            )

    def test_extrapolate_left(self) -> None:
        """X before the first marker — extrapolate using leftmost segment."""
        markers = [100.0, 150.0, 200.0]  # 50px/ft uniform
        # x=50 is 50px before marker[0]; segment 0 is 50px = 1ft → result -1ft
        assert pixel_x_to_feet(50.0, markers, 60.0) == -1.0

    def test_extrapolate_right(self) -> None:
        """X past the last marker — extrapolate with rightmost segment."""
        markers = [100.0, 150.0, 200.0, 250.0]
        # Rightmost segment is 50px = 1ft.  x=300 is 50px past last → 4ft
        assert pixel_x_to_feet(300.0, markers, 60.0) == 4.0
        # x=350 → 5ft
        assert pixel_x_to_feet(350.0, markers, 60.0) == 5.0

    def test_fisheye_extrapolation_uses_local_ppf(self) -> None:
        """Greg's right-edge spacing is 45.5 px/ft (vs 57 in middle).
        Extrapolating past the last cork should use the 45.5 ppf."""
        markers = [172.0, 225.0, 282.0, 339.5, 397.0, 452.0, 503.0, 548.5]
        # Last marker at x=548.5 is 7ft.  Right-edge ppf = 548.5-503 = 45.5.
        # x=594 should be 7 + (594-548.5)/45.5 = 7 + 1.0 = 8ft
        assert abs(pixel_x_to_feet(594.0, markers, 60.0) - 8.0) < 0.05

    def test_invalid_markers_falls_back_to_ppf(self) -> None:
        """Single marker is not enough — falls back to flat ppf."""
        assert pixel_x_to_feet(60.0, [100.0], 30.0) == 2.0


class TestTrajectoryFitWithMarkers:
    """End-to-end: synthesize a putt with non-uniform ppf and verify the
    fit reads correct launch velocity when using calibration markers."""

    STIMP = 11.0
    DT = 1.0 / 60.0

    def _track_with_fisheye(
        self, launch_mph: float, n_motion: int = 30,
    ) -> tuple[list[tuple[int, int, float]], list[float]]:
        """Build a synthetic putt where pixel-per-foot varies linearly
        across the frame (simulating fisheye).  Returns (positions, markers)."""
        import math as _math

        # Define non-uniform ppf via markers at each foot — this is what
        # OBS Cal would store.
        # Markers go from x=100 (0ft) to x=560 (8ft) with spacing
        # decreasing linearly from 60 px/ft (left) to 50 px/ft (right).
        # Cumulative positions:
        markers: list[float] = [100.0]
        spacing_now = 60.0
        for _ in range(8):
            markers.append(markers[-1] + spacing_now)
            spacing_now -= 1.25  # gradient across 8 feet → 60→50

        # Real ball motion in feet (decelerating)
        a = (6.08 ** 2) / (2 * self.STIMP)
        launch_fps = launch_mph * 5280 / 3600

        positions: list[tuple[int, int, float]] = []
        # Rest frames
        for i in range(3):
            positions.append((100, 200, i * self.DT))
        t_start = 3 * self.DT
        # Motion: convert feet → pixel using the marker mapping
        for i in range(n_motion):
            tau = (i + 1) * self.DT
            travel_ft = launch_fps * tau - 0.5 * a * tau * tau
            if travel_ft < 0:
                break
            # Find pixel x for this many feet
            # Walk through markers
            if travel_ft >= 8:
                # extrapolate right
                local_ppf = markers[-1] - markers[-2]
                px = markers[-1] + (travel_ft - 8) * local_ppf
            else:
                lo = int(travel_ft)
                local_ppf = markers[lo + 1] - markers[lo]
                px = markers[lo] + (travel_ft - lo) * local_ppf
            positions.append((int(px), 200, t_start + tau))
        return positions, markers

    def test_per_x_fit_corrects_fisheye(self) -> None:
        """When we use per-x markers, fit reads the true launch velocity
        despite the synthetic fisheye gradient.  Without markers (flat
        ppf), fit is biased."""
        launch_mph = 4.0
        positions, markers = self._track_with_fisheye(launch_mph)

        # Use the central ppf (~55 px/ft) as the flat fallback
        flat_ppf = 55.0

        # Without markers: biased reading
        v_flat, _ = speed_from_trajectory_fit(
            positions, flat_ppf, stimpmeter=self.STIMP,
        )
        # With per-x markers: corrected reading
        v_perx, dbg = speed_from_trajectory_fit(
            positions, flat_ppf, stimpmeter=self.STIMP,
            calibration_markers=markers,
        )

        assert dbg["uses_markers"] is True
        # Per-x reading should be much closer to the true launch
        err_flat = abs(v_flat - launch_mph)
        err_perx = abs(v_perx - launch_mph)
        assert err_perx < 0.3, f"per-x fit error {err_perx:.2f} (read {v_perx:.2f})"
        # The per-x fit should be at least as accurate as the flat fit
        assert err_perx <= err_flat + 0.05, (
            f"per-x ({v_perx:.2f}) should be no worse than flat ({v_flat:.2f}) "
            f"vs truth {launch_mph}"
        )

    def test_markers_disabled_falls_back_cleanly(self) -> None:
        """An empty marker list should be treated as no calibration."""
        launch_mph = 4.0
        positions, _ = self._track_with_fisheye(launch_mph)
        v, dbg = speed_from_trajectory_fit(
            positions, 55.0, stimpmeter=self.STIMP,
            calibration_markers=[],  # empty list
        )
        assert dbg["uses_markers"] is False
        assert v > 0  # still produces a (biased) result
