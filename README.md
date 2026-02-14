# Webcam Putting

Webcam-based golf putting simulator for GSPro. Uses a standard webcam to detect ball movement, calculate ball speed (MPH) and horizontal launch angle (HLA), and sends shot data directly to GSPro via the Open Connect v1 API.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Run with defaults
webcam-putting

# Specify ball color and webcam
webcam-putting -c orange2 -w 1

# Test with a video file
webcam-putting -v path/to/video.mp4

# Debug mode (shows color detection mask)
webcam-putting -d

# Migrate from old config.ini
webcam-putting --migrate-ini path/to/config.ini
```

## Configuration

Settings are stored in TOML format at your platform's config directory:
- **Windows**: `%APPDATA%/webcam-putting/config.toml`
- **macOS**: `~/Library/Application Support/webcam-putting/config.toml`

## Original Project

Forked and modernized from [cam-putting-py](https://github.com/alleexx/cam-putting-py) by alleexx.
