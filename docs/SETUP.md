# Birdman Putting Setup Guide

Complete setup guide for Birdman Putting — a webcam-based golf putting simulator with Flightscope Mevo integration for GSPro.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [GSPro Setup](#gspro-setup)
- [Webcam Putting Setup](#webcam-putting-setup)
- [Mevo Launch Monitor Setup](#mevo-launch-monitor-setup)
- [OBS Integration](#obs-integration)
- [Windows Display Settings](#windows-display-settings)
- [Launching Birdman](#launching-birdman)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### For Webcam Putting
- USB webcam (720p @ 60fps recommended — e.g., Logitech C922 Pro)
- Golf ball (any standard 1.68" diameter)
- Well-lit putting surface with contrast between ball and background
- Overhead or angled camera mount looking down at the putting area

### For Full Swings (Mevo Integration)
- Flightscope Mevo Gen 2 launch monitor
- Windows PC (FS Golf PC is Windows-only)

### For Streaming/Projection (OBS)
- OBS Studio (v28+ for built-in WebSocket support)
- Optional: projector or second display for overlay output

---

## Software Requirements

| Software | Required For | Download |
|----------|-------------|----------|
| Python 3.10+ | Everything | [python.org](https://www.python.org/downloads/) |
| GSPro | Simulator | [gsprogolf.com](https://gsprogolf.com) |
| FS Golf PC | Mevo integration | Flightscope app store |
| Tesseract OCR | Mevo integration | [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) |
| OBS Studio | Streaming/projection | [obsproject.com](https://obsproject.com) |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/CoderCommander/webcamPutting.git
cd webcamPutting

# Install core package
pip install -e .

# With Mevo OCR support
pip install -e ".[mevo]"

# With OBS integration
pip install -e ".[obs]"

# With everything (dev + mevo + obs)
pip install -e ".[dev]"
```

### Tesseract OCR (Mevo only)

If using the Mevo integration, install Tesseract OCR separately:

- **Windows**: Download and run the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Make sure it's added to your PATH.
- **macOS**: `brew install tesseract`

Verify: `tesseract --version`

---

## GSPro Setup

1. Open GSPro
2. Go to **Settings > Open API**
3. Select **BirdmanPutting** as the launch monitor (it appears after the first connection)
4. Ensure the Open API shows **"Ready"** status on port **921**

Birdman connects to GSPro via the Open Connect v1 API (TCP socket on port 921). Both applications must run on the same machine, or update `gspro_host` in config.toml for network play.

---

## Webcam Putting Setup

### 1. Camera Positioning

Mount your webcam overhead or at an angle looking down at the putting surface. The camera should see:
- The ball's starting position
- The ball's path through the detection gateway

### 2. First Launch

```bash
python -m birdman_putting -w 1 -c orange2
```

Replace `-w 1` with your webcam index (0, 1, 2...) and `-c orange2` with your ball color preset.

**Available color presets:** `yellow`, `yellow_bright`, `orange`, `orange2`, `orange_dark`, `red`, `red_dark`, `white`, `green`, `green_dark`

### 3. Detection Zone Setup

Click **"Auto Zone"** to auto-calibrate the detection zone, or click **"Edit Zone"** to manually drag the zone into position.

The yellow rectangle is the **start zone** (where the ball rests). The red rectangle is the **gateway** (the ball must pass through it to register a shot).

### 4. Angle Calibration

If your putts consistently read a slight angle offset:

1. Make sure the projected line is **OFF** (or no line is visible)
2. Click **"Angle Cal"** — Birdman captures the background
3. **Turn on** your projected straight line (or place a straight edge)
4. Wait for 30 consistent readings (~0.5 seconds) — auto-applies

### 5. Verify

Putt a ball through the gateway. You should see:
- Speed (MPH) and HLA (degrees) in the Last Shot panel
- Estimated distance (feet) based on stimpmeter setting
- Shot count incrementing
- GSPro receiving the shot data

---

## Mevo Launch Monitor Setup

The Mevo integration reads shot data from FS Golf PC via screen capture and OCR, then relays it to GSPro through the Open API.

### 1. Start FS Golf PC

Launch FS Golf PC and connect your Mevo Gen 2. Ensure shots are being displayed in the app.

### 2. Calibrate ROIs

Run the interactive calibration tool:

```bash
python -m birdman_putting --calibrate-mevo
```

This walks you through selecting regions for each metric on the FS Golf PC screen:

- **Draw a rectangle** around each metric value (click and drag)
- **Press Enter** to confirm each region
- **Press 's'** to skip optional metrics
- **Press 'r'** to redo the current region
- **Press ESC** to abort

**Required metrics:** ball_speed, launch_angle, launch_direction
**Optional metrics:** spin_rate, spin_axis, club_speed, smash_factor, carry_distance, total_distance, apex_height, flight_time, descent_angle, curve, roll_distance, aoa, club_path, dynamic_loft, face_to_target, lateral_impact, vertical_impact

**Important:** Make ROIs wide enough to capture R/L direction suffixes (e.g., "8.5 R") for signed metrics like launch_direction, spin_axis, club_path, and face_to_target.

### 3. Launch with Mevo Enabled

```bash
python -m birdman_putting -w 1 -c orange2 --mevo --obs
```

### 4. Chipping Mode

Birdman automatically switches FS Golf PC between **Chipping** and **Full Swing** modes based on club selection from GSPro:

- **Wedges** (SW, LW, AW, GW, PW) → Chipping mode
- **All other clubs** (except putter) → Full Swing mode
- **Putter** → Webcam putting (Mevo not used)

This requires FS Golf PC to be running. There will be a brief window focus flash when switching modes — see [Windows Display Settings](#windows-display-settings) to minimize this.

---

## OBS Integration

Birdman can automatically switch OBS scenes based on what's happening in GSPro.

### 1. Enable OBS WebSocket Server

In OBS: **Tools > WebSocket Server Settings**
- Check **"Enable WebSocket Server"**
- Port: `4455` (default)
- Set a password if desired

### 2. Create OBS Scenes

Create these scenes (names must match your config.toml):

| Scene Name | Purpose |
|-----------|---------|
| **Main** | Default view — GSPro gameplay, full swing camera, etc. |
| **Putt Data** | Webcam putting view — shown when putter is selected |
| **Mevo Shot Data** | Full swing data overlay (text sources auto-created) |

### 3. Configure Birdman

Add to your `config.toml`:

```toml
[obs]
enabled = true
host = "localhost"
port = 4455
password = "your_password"
mevo_scene = "Mevo Shot Data"
putt_scene = "Putt Data"
idle_scene = "Main"
display_duration = 18.0
auto_scene_switch = true
```

### 4. Use Preview Projector for Output

To display on a projector or second screen that follows scene changes:

1. Right-click the **main preview area** in OBS (the big preview panel)
2. Select **"Open Preview Projector"**
3. Choose your projector/display

**Do NOT use "Open Scene Projector"** — that locks to a single scene and won't follow automatic scene switches.

### 5. Auto Scene Switching Behavior

With `auto_scene_switch = true`:

| GSPro Event | OBS Action |
|------------|------------|
| Putter selected | Switch to "Putt Data" scene |
| Non-putter selected | Switch to "Main" scene |
| Mevo shot detected | Switch to "Mevo Shot Data" for 18 seconds, then back to current |

---

## Windows Display Settings

### Disable Background Power Throttling (Required)

Windows 11 throttles background applications, which causes the webcam to drop from 60fps to 2fps when you click on GSPro or any other window. **This must be disabled for Birdman to work properly.**

1. Press **Win + R**, type `regedit`, press Enter
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Power`
3. Right-click the **Power** key → **New > Key** → name it `PowerThrottling`
4. Right-click the **PowerThrottling** key → **New > DWORD (32-bit) Value** → name it `PowerThrottlingOff`
5. Double-click `PowerThrottlingOff` and set the value to `1`
6. **Restart your computer**

This disables power throttling for all background apps, ensuring the webcam maintains full frame rate when Birdman is behind GSPro.

### Taskbar on Laptop Only

When Birdman switches FS Golf PC modes (chipping/full swing), a brief window focus flash occurs. To prevent the taskbar from appearing on projectors:

1. **Settings > Personalization > Taskbar > Taskbar behaviors**
2. Turn **OFF** "Show my taskbar on all displays"

This keeps the taskbar on your primary laptop screen only — projectors won't show it.

---

## Launching Birdman

### Recommended Launch Command

```bash
python -m birdman_putting -c orange2 -w 1 --mevo --obs
```

### Launch Order

1. Start **OBS** (so WebSocket server is ready)
2. Start **GSPro** (so Open API is listening on port 921)
3. Start **FS Golf PC** and connect your Mevo (if using Mevo)
4. Start **Birdman Putting**

### Full Command Reference

```bash
python -m birdman_putting [OPTIONS]

Options:
  -c, --ballcolor NAME    Ball color preset (yellow, orange2, red, white, etc.)
  -w, --camera INDEX      Webcam index (0, 1, 2...)
  -v, --video PATH        Test with a video file instead of webcam
  -d, --debug             Show color detection mask window
  --mevo                  Enable Mevo launch monitor via OCR
  --calibrate-mevo        Interactive Mevo ROI calibration
  --obs                   Enable OBS WebSocket integration
  --no-gui                Headless mode (OpenCV windows only)
  --config PATH           Custom config.toml path
  --log-level LEVEL       DEBUG, INFO, WARNING, or ERROR
  --migrate-ini PATH      Migrate from old config.ini format
```

---

## Configuration Reference

Settings are stored in TOML format:
- **Windows**: `%APPDATA%/birdman-putting/config.toml`
- **macOS**: `~/Library/Application Support/birdman-putting/config.toml`

Edit directly with a text editor or use the Settings tab in the Birdman UI.

### Key Settings

| Setting | Section | Default | Description |
|---------|---------|---------|-------------|
| `stimpmeter` | `[shot]` | `10.0` | Green speed for distance estimation (7=slow, 13=fast tour) |
| `direction` | `[detection_zone]` | `left_to_right` | Ball travel direction |
| `gateway_width` | `[detection_zone]` | `15` | Gateway detection width in pixels |
| `color_preset` | `[ball]` | `yellow` | HSV color preset for ball detection |
| `rotation` | `[camera]` | `0.0` | Camera rotation correction (degrees) |
| `auto_scene_switch` | `[obs]` | `true` | Auto-switch OBS scenes on club change |
| `send_to_gspro` | `[mevo]` | `true` | Relay Mevo shots to GSPro |
| `trail_color` | `[overlay]` | `cyan` | Ball tracer color |
| `obs_overlay_mode` | `[overlay]` | `false` | Black background with tracer only |
| `projected_trail` | `[overlay]` | `false` | Calculated trajectory vs camera tracking |

---

## Troubleshooting

### Ball Not Detected
- Check lighting — the ball color must contrast with the background
- Try debug mode (`-d` flag) to see the color detection mask
- Adjust `min_radius` and `min_circularity` in config.toml
- Ensure the camera is set to the correct index (`-w 0`, `-w 1`, etc.)

### GSPro Not Receiving Shots
- Verify GSPro Open API shows "Ready" on port 921
- Check that `gspro_host` and `gspro_port` are correct in config.toml
- Look for "Shot sent to GSPro" in the logs

### FPS Drops Below 10
- The FPS watchdog auto-resets after 2 seconds below 10 FPS
- Click "Reset Putt" to manually clear state
- Reduce `max_trail_points` in config.toml if trail rendering is slow
- Close unnecessary applications competing for CPU

### Mevo OCR Reads Wrong Values
- Re-run `--calibrate-mevo` to recalibrate ROIs
- Ensure FS Golf PC window is not obscured by other windows
- Check that ROIs are wide enough for signed metrics (R/L suffixes)
- Review OCR logs with `--log-level DEBUG`

### OBS Scenes Not Switching
- Verify OBS WebSocket is enabled (Tools > WebSocket Server Settings)
- Check scene names match exactly between OBS and config.toml
- Use Preview Projector (not Scene Projector) for display output
- Check logs for "OBS: Connected" and "OBS scenes available"

### FS Golf PC Chipping Mode Not Switching
- FS Golf PC must be running and visible (can be behind other windows)
- Check logs for "Sent key 'c' to window 'FS Golf PC'"
- The `window_title` in config.toml must match the FS Golf window title
