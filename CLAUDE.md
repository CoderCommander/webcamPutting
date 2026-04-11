# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Webcam-based golf putting simulator for GSPro. Uses a standard webcam + OpenCV to detect ball movement, calculate ball speed (MPH) and horizontal launch angle (HLA), then sends shot data to GSPro golf simulator. Supports both direct GSPro Open Connect v1 API (TCP socket, port 921) and legacy HTTP middleware (port 8888).

This repo contains two versions:
- `cam-putting-py/` — Original upstream single-file application (reference only)
- `src/birdman_putting/` — Modernized modular rewrite (active development)

## Webcam Putting App 

When fixing bugs in the webcam putting app, always verify camera feed still works after changes. Never modify camera backend/capture pipeline without testing that frames are still being received and displayed correctly.

### Camera Backend — CRITICAL

The camera MUST use the default MSMF backend via `cv2.VideoCapture(index)`. Do NOT:
- Switch to DirectShow (`CAP_DSHOW`) — it resets Razer Kiyo Pro firmware settings, causing black frames
- Set `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` — this disables MSMF's color conversion pipeline entirely, preventing the camera from opening
- Use `cv2.VideoCapture(index, cv2.CAP_MSMF, [params])` — the 3-arg constructor fails with "can't capture by index" on some systems
- Reopen the camera on the grab thread — reopening resets firmware state (black frames on Kiyo Pro)

**GPU throttling with GSPro:** MSMF creates a D3D11 device for hardware-accelerated color conversion. When GSPro (a DirectX game) saturates the GPU, MSMF starves → 2fps. The fix is `cv2.VideoCapture(index, cv2.CAP_ANY, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])` which forces CPU-only capture. If this constructor fails on the user's system, fall back to default open + `cap.set(CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_NONE)` post-open. Never remove this — without it, FPS drops to 2 when GSPro is in the foreground.

### Shot Detection Pipeline — DO NOT BREAK

The putting detection pipeline has been carefully tuned. Do not change these without testing real putts:

- **Two-pass detection in STARTED state**: Pass 1 searches start zone only, Pass 2 searches FULL frame width. Never cap Pass 2 to a narrow band — fast putts jump past the gateway in one frame.
- **Y-range expansion**: STARTED state gets ±30px Y margin, ENTERED gets ±50px. The detection zone is only ~48px tall — ball drifts vertically during roll.
- **"Jumped past gateway" logic** (tracking.py): When ball skips from start zone past `gateway_x2` in one frame, complete the shot immediately. Without this, fast putts never register.
- **ENTERED timeout completes shot**: When ball enters gateway but then goes off-screen, complete the shot with last known position instead of discarding it.
- **Rotation applied BEFORE detection**: `apply_rotation()` must run before `detect()` so coordinates are consistent between detection, overlay, and zone drawing. Moving rotation after detection causes the ball ring to be offset from the actual ball.
- **`positions` list includes rest frames**: The physics `calculate_shot()` must handle positions where the ball sits still before moving. Movement-onset filtering is only used when 6+ positions AND 4+ moving points exist, with sanity checks.

### FS Golf PC Key Sending — CRITICAL

FS Golf PC chipping/full swing mode switching uses `keybd_event` (NOT `SendInput`). FS Golf PC ignores `SendInput`. The proven pattern (commit b46466f):
1. `AttachThreadInput` to foreground thread
2. `SetForegroundWindow` to FS Golf PC
3. `keybd_event` key down/up
4. Restore previous foreground window

Do NOT replace `keybd_event` with `SendInput` or `PostMessageW` to the main window — both have been tried and fail with FS Golf PC.

### Performance Changes

Before applying performance optimizations (FPS, threading, COM initialization), create a checkpoint commit. If the optimization breaks existing functionality, revert immediately rather than iterating on a broken approach.

### Debugging

When modifying image processing pipelines (OpenCV, HSV conversion, overlays), test each stage independently. Log intermediate frame values to catch issues like double HSV conversion early.

## Commands

```bash
# Install (editable mode with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run the application (python -m is most reliable on Windows)
python -m birdman_putting                         # uses default config
python -m birdman_putting -c orange2 -w 1         # orange ball, webcam 1
python -m birdman_putting -v path/to/video.mp4    # test with video file
python -m birdman_putting -d                      # debug mode (shows mask)
python -m birdman_putting --no-gui                # headless mode (OpenCV windows only)
python -m birdman_putting --migrate-ini cam-putting-py/config.ini  # migrate old config

# Lint and type-check
ruff check src/ tests/
mypy src/birdman_putting/

# Build standalone executable (Windows)
pip install -e ".[build]"
pyinstaller birdman_putting.spec
```

## Architecture

### Module Structure (`src/birdman_putting/`)

- `config.py` — Dataclass config model (`AppConfig` with sections: `DetectionZone`, `CameraSettings`, `BallSettings`, `ShotSettings`, `ConnectionSettings`, `ReplaySettings`). TOML load/save via `platformdirs` for cross-platform config directory. Includes `migrate_from_ini()` for old config.ini migration.
- `color_presets.py` — `HSVRange` dataclass and `PRESETS` dictionary with 12 named presets (red, white, yellow, green, orange variants for different lighting)
- `detection.py` — `BallDetector` class: HSV color filtering + contour analysis. Fixes the double HSV conversion bug from the original code. Also contains `resize_with_aspect_ratio()` (replaces imutils dependency).
- `tracking.py` — `BallTracker` with `ShotState` enum state machine (`IDLE → BALL_DETECTED → STARTED → ENTERED → LEFT`). Uses position clustering with tolerance for start detection instead of exact pixel match.
- `physics.py` — `calculate_shot()` with multi-point trajectory fitting via least-squares, outlier rejection (2σ), and `time.perf_counter()` for microsecond timing. Converts pixel distance to mm using known golf ball radius (21.335mm).
- `camera.py` — `Camera` class wrapping `cv2.VideoCapture`: handles MJPEG codec, DirectShow, FPS override, PS4 Eye decoding, camera property management.
- `gspro_client.py` — `GSProClient`: TCP socket client for GSPro Open Connect v1 API with heartbeat thread (5s interval), exponential backoff reconnection. Falls back to HTTP POST for legacy middleware mode.
- `app.py` — `PuttingApp` orchestrator wiring camera → detection → tracking → physics → GSPro client. Two modes: GUI (CustomTkinter, threaded processing) and headless (`--no-gui`, cv2.imshow). Processing thread feeds annotated frames into a queue polled by the UI.
- `__main__.py` — CLI entry point with argparse, preserving original `-c`, `-w`, `-v`, `-d` flags plus `--no-gui`, `--config`, `--migrate-ini`, `--log-level`.
- `ui/overlay.py` — HUD renderer: draws detection zones, ball marker, trajectory line, status text onto BGR frames. No tkinter dependency.
- `ui/video_panel.py` — `VideoPanel` (CTkLabel subclass): polls frame queue at ~60fps, converts BGR→RGB→PIL→CTkImage.
- `ui/settings_panel.py` — `SettingsPanel` (CTkToplevel): 4 tabs (Detection zone sliders, Camera, Ball Color preset, Connection mode/host/port). Saves to TOML only on dialog close.
- `ui/main_window.py` — `MainWindow` (CTk root, 890x460, dark mode): video panel left 70%, status panel right 30% (connection, last shot, shot history, FPS/state), control bar bottom (color dropdown, start/stop, settings).

### Threading Model (GUI mode)
- **Main thread**: CustomTkinter event loop. VideoPanel polls frame_queue via `after(16ms)`.
- **Processing thread**: `camera.read()` → detect → track → annotate → `frame_queue.put()`. UI label updates scheduled via `window.after()` at ~4Hz.
- **Heartbeat thread**: GSProClient sends keepalive every 5s.
- In headless mode (`--no-gui`), everything runs synchronously on the main thread with `cv2.imshow`.

### Detection Pipeline
1. Camera capture (MJPEG or YUY2 codec)
2. Gaussian blur + single BGR→HSV conversion (bug fix: original converted twice)
3. `cv2.inRange` mask → contour analysis → `minEnclosingCircle`
4. State machine tracks ball: stable start → gateway entry → gateway exit
5. Physics: pixel distance/time → mm → MPH speed, trajectory fit → HLA angle
6. Send to GSPro via socket or HTTP

### Configuration
TOML format stored at platform config dir (`%APPDATA%/birdman-putting/config.toml` on Windows, `~/Library/Application Support/` on macOS). All ~30 previously hardcoded thresholds are now configurable in `ShotSettings` and `BallSettings` dataclasses.

### Testing
Tests use synthetic frames (`np.zeros` + `cv2.circle`) — no camera needed. Run `pytest tests/ -v`. Test files mirror source modules: `test_config.py`, `test_detection.py`, `test_tracking.py`, `test_physics.py`, `test_gspro_client.py`.

## Key Technical Details

- Golf ball radius: 21.335mm. `pixel_to_mm_ratio = detected_radius_px / 21.335`
- GSPro Open Connect v1: TCP socket on port 921, JSON messages with `DeviceID`, `ShotNumber`, `APIversion`, `BallData`, `ShotDataOptions`. Heartbeat every 5 seconds with `IsHeartBeat: true`.
- The detection zone (yellow rectangle) defines ball start area. The gateway (red rectangle, 10px past start zone) detects entry/exit.
- 12 HSV presets for different ball colors and lighting conditions. Custom HSV values override presets.
