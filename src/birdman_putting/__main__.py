"""CLI entry point for birdman-putting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from birdman_putting.config import CONFIG_FILE, load_config, migrate_from_ini, save_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Webcam-based golf putting simulator for GSPro. "
        "Run with: python -m birdman_putting",
    )
    parser.add_argument(
        "-v", "--video",
        help="Path to video file (for testing)",
    )
    parser.add_argument(
        "-i", "--img",
        help="Path to image file (for testing)",
    )
    parser.add_argument(
        "-b", "--buffer", type=int, default=64,
        help="Max tracking buffer size (default: 64)",
    )
    parser.add_argument(
        "-w", "--camera", type=int, default=None,
        help="Webcam index number (default: from config)",
    )
    parser.add_argument(
        "-c", "--ballcolor",
        help="Ball color preset (e.g., yellow, orange2, red, white)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true",
        help="Enable debug mode with color mask display",
    )
    parser.add_argument(
        "-r", "--resize", type=int, default=640,
        help="Display width in pixels (default: 640)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.toml file",
    )
    parser.add_argument(
        "--migrate-ini", type=str, default=None,
        help="Migrate settings from old config.ini file",
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run in headless mode with OpenCV windows (no CustomTkinter UI)",
    )
    parser.add_argument(
        "--mevo", action="store_true",
        help="Enable Flightscope Mevo launch monitor via screenshot OCR",
    )
    parser.add_argument(
        "--calibrate-mevo", action="store_true",
        help="Interactive ROI calibration for Mevo OCR",
    )
    parser.add_argument(
        "--obs", action="store_true",
        help="Enable OBS WebSocket integration for shot data overlay",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Handle INI migration
    if args.migrate_ini:
        ini_path = Path(args.migrate_ini)
        if ini_path.exists():
            config = migrate_from_ini(ini_path, config)
            save_config(config, config_path)
            logging.info("Migrated config from %s to %s", ini_path, config_path or CONFIG_FILE)
        else:
            logging.error("INI file not found: %s", ini_path)
            sys.exit(1)

    # Auto-detect old config.ini for migration
    old_ini = Path("config.ini")
    if old_ini.exists() and not (config_path or CONFIG_FILE).exists():
        logging.info("Found old config.ini, auto-migrating...")
        config = migrate_from_ini(old_ini, config)
        save_config(config, config_path)

    # Apply CLI overrides
    if args.camera is not None:
        config.camera.webcam_index = args.camera

    if args.ballcolor:
        config.ball.color_preset = args.ballcolor
        config.ball.custom_hsv = None  # CLI color overrides custom HSV

    if args.mevo:
        config.mevo.enabled = True

    if args.obs:
        config.obs.enabled = True

    # Handle calibration mode
    if args.calibrate_mevo:
        from birdman_putting.mevo.calibrate import run_calibration

        run_calibration(config)
        sys.exit(0)

    # Run app
    from birdman_putting.app import PuttingApp

    app = PuttingApp(
        config=config,
        video_path=args.video,
        debug=args.debug,
        headless=args.no_gui,
    )
    app.run()


if __name__ == "__main__":
    main()
