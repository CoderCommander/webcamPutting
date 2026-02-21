# Birdman Putting

Webcam-based golf putting simulator for GSPro. Uses a standard webcam to detect ball movement, calculate ball speed (MPH) and horizontal launch angle (HLA), and sends shot data directly to GSPro via the Open Connect v1 API.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run with defaults
python -m birdman_putting

# Specify ball color and webcam
python -m birdman_putting -c orange2 -w 1

# Test with a video file
python -m birdman_putting -v path/to/video.mp4

# Debug mode (shows color detection mask)
python -m birdman_putting -d

# Headless mode (OpenCV windows only, no GUI)
python -m birdman_putting --no-gui

# Migrate from old config.ini
python -m birdman_putting --migrate-ini path/to/config.ini
```

> **Note:** On some systems, `birdman-putting` may also work as a shortcut command.
> If it doesn't (common on Windows), use `python -m birdman_putting`.

## Configuration

Settings are stored in TOML format at your platform's config directory:
- **Windows**: `%APPDATA%/birdman-putting/config.toml`
- **macOS**: `~/Library/Application Support/birdman-putting/config.toml`

## Original Project

Forked and modernized from [cam-putting-py](https://github.com/alleexx/cam-putting-py) by alleexx.
