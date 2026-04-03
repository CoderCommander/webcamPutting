#!/usr/bin/env python
"""Benchmark CPU vs CuPy GPU ball detection.

Usage:
    python scripts/benchmark_cuda.py

Requires: pip install cupy-cuda12x
"""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np

from birdman_putting.color_presets import get_preset
from birdman_putting.detection import BallDetector


def create_synthetic_frame(width: int = 200, height: int = 250) -> np.ndarray:
    """Create a synthetic BGR frame with a colored circle (simulated ball)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 60, 30)  # Dark green background
    cv2.circle(frame, (100, 125), 10, (0, 140, 255), -1)  # Orange ball
    return frame


def benchmark_detector(
    detector: BallDetector,
    frame: np.ndarray,
    iterations: int = 1000,
    label: str = "Detector",
) -> list[float]:
    """Run detection N times and return per-iteration times in ms."""
    times: list[float] = []
    # Warmup (important for GPU to initialize kernels)
    for _ in range(50):
        detector.detect(frame, 0, frame.shape[1], 0, frame.shape[0], time.perf_counter())

    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(frame, 0, frame.shape[1], 0, frame.shape[0], t0)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    median = statistics.median(times)
    p99 = sorted(times)[int(len(times) * 0.99)]
    mean = statistics.mean(times)
    print(f"\n{label}:")
    print(f"  Iterations: {iterations}")
    print(f"  Mean:   {mean:.3f} ms")
    print(f"  Median: {median:.3f} ms")
    print(f"  P99:    {p99:.3f} ms")
    print(f"  Min:    {min(times):.3f} ms")
    print(f"  Max:    {max(times):.3f} ms")
    print(f"  Theoretical FPS: {1000 / median:.0f}")
    return times


def main() -> None:
    print("=" * 60)
    print("Birdman Putting: CPU vs CuPy GPU Detection Benchmark")
    print("=" * 60)

    frame = create_synthetic_frame()
    print(f"\nFrame size: {frame.shape[1]}x{frame.shape[0]} (simulates cropped ROI)")

    hsv_range = get_preset("orange2")

    # CPU benchmark
    cpu_detector = BallDetector(hsv_range=hsv_range)
    cpu_times = benchmark_detector(cpu_detector, frame, label="CPU (BallDetector)")

    # CuPy GPU benchmark
    try:
        from birdman_putting.cuda_detection import CupyBallDetector
        from birdman_putting.gpu import get_device_name, init_cuda

        if not init_cuda("RTX 3000"):
            print("\nCUDA init failed — cannot benchmark GPU")
            return

        import cupy as _cp

        dev_id = _cp.cuda.Device().id
        print(f"\nGPU: {get_device_name(dev_id)} (device {dev_id})")

        gpu_detector = CupyBallDetector(hsv_range=hsv_range)
        gpu_times = benchmark_detector(gpu_detector, frame, label="CuPy GPU (CupyBallDetector)")

        # Comparison
        cpu_median = statistics.median(cpu_times)
        gpu_median = statistics.median(gpu_times)
        speedup = cpu_median / gpu_median if gpu_median > 0 else 0

        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"  CPU median:  {cpu_median:.3f} ms")
        print(f"  GPU median:  {gpu_median:.3f} ms")
        print(f"  Speedup:     {speedup:.2f}x")
        if speedup > 1.0:
            print("  VERDICT: GPU is FASTER — worth using!")
        else:
            print("  VERDICT: GPU is SLOWER — stick with CPU for this ROI size")
            print("  (GPU shines with larger frames — try 720p or 1080p)")

    except ImportError as e:
        print(f"\nCuPy not available: {e}")
        print("Install with: pip install cupy-cuda12x")


if __name__ == "__main__":
    main()
