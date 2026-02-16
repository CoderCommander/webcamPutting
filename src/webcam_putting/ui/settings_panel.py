"""Settings panel — two-tab dialog: Setup (essentials) and Advanced."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from webcam_putting.color_presets import PRESET_DESCRIPTIONS, PRESETS
from webcam_putting.config import AppConfig, save_config

logger = logging.getLogger(__name__)


class SettingsPanel(ctk.CTkToplevel):
    """Settings dialog with two tabs: Setup and Advanced.

    Setup tab: ball color, webcam index, flip image, connection mode/host.
    Advanced tab: MJPEG, flip view, FPS, darkness, fixed radius, gateway width,
                  port, device ID, HTTP URL.

    Detection zone is NOT here — it's edited via mouse drag on the video panel.
    """

    def __init__(
        self,
        master: Any,
        config: AppConfig,
        on_close: Callable[[], None] | None = None,
        **kwargs: Any,
    ):
        super().__init__(master, **kwargs)
        self.title("Settings")
        self.geometry("480x480")
        self.resizable(False, False)

        self._config = config
        self._on_close = on_close

        # Tabview
        self._tabview = ctk.CTkTabview(self, width=460, height=400)
        self._tabview.pack(padx=10, pady=(10, 5), fill="both", expand=True)

        self._tabview.add("Setup")
        self._tabview.add("Advanced")

        self._build_setup_tab()
        self._build_advanced_tab()

        # Save & Close button
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(padx=10, pady=(0, 10), fill="x")

        ctk.CTkButton(
            btn_frame, text="Save & Close", command=self._save_and_close,
            width=140,
        ).pack(side="right")

        ctk.CTkButton(
            btn_frame, text="Cancel", command=self.destroy,
            width=100, fg_color="gray40", hover_color="gray30",
        ).pack(side="right", padx=(0, 8))

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._save_and_close)

    # ---- Setup Tab ----

    def _build_setup_tab(self) -> None:
        tab = self._tabview.tab("Setup")
        b = self._config.ball
        c = self._config.camera
        conn = self._config.connection

        # Ball Color
        ctk.CTkLabel(
            tab, text="Ball Color", font=("", 14, "bold"),
        ).pack(anchor="w", pady=(5, 5))

        preset_names = list(PRESETS.keys())
        preset_labels = [PRESET_DESCRIPTIONS.get(n, n) for n in preset_names]
        self._label_to_preset = dict(
            zip(preset_labels, preset_names, strict=True),
        )
        self._preset_to_label = dict(
            zip(preset_names, preset_labels, strict=True),
        )
        current_label = self._preset_to_label.get(
            b.color_preset, "Yellow (bright)",
        )
        self._color_var = ctk.StringVar(value=current_label)
        ctk.CTkOptionMenu(
            tab, variable=self._color_var, values=preset_labels, width=250,
        ).pack(anchor="w", pady=(0, 10))

        # Camera
        ctk.CTkLabel(
            tab, text="Camera", font=("", 14, "bold"),
        ).pack(anchor="w", pady=(10, 5))

        idx_frame = ctk.CTkFrame(tab, fg_color="transparent")
        idx_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(
            idx_frame, text="Webcam Index:", width=120, anchor="w",
        ).pack(side="left")
        self._cam_index = ctk.CTkEntry(idx_frame, width=60)
        self._cam_index.insert(0, str(c.webcam_index))
        self._cam_index.pack(side="left", padx=5)

        self._flip_var = ctk.BooleanVar(value=c.flip_image)
        ctk.CTkCheckBox(
            tab, text="Flip Image (Left-handed)", variable=self._flip_var,
        ).pack(anchor="w", pady=3)

        # Connection
        ctk.CTkLabel(
            tab, text="Connection", font=("", 14, "bold"),
        ).pack(anchor="w", pady=(10, 5))

        self._mode_var = ctk.StringVar(value=conn.mode)
        ctk.CTkRadioButton(
            tab, text="Direct GSPro (port 921)",
            variable=self._mode_var, value="gspro_direct",
        ).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(
            tab, text="HTTP Middleware",
            variable=self._mode_var, value="http_middleware",
        ).pack(anchor="w", pady=2)

        host_frame = ctk.CTkFrame(tab, fg_color="transparent")
        host_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            host_frame, text="Host:", width=80, anchor="w",
        ).pack(side="left")
        self._host_entry = ctk.CTkEntry(host_frame, width=180)
        self._host_entry.insert(0, conn.gspro_host)
        self._host_entry.pack(side="left", padx=5)

    # ---- Advanced Tab ----

    def _build_advanced_tab(self) -> None:
        tab = self._tabview.tab("Advanced")
        c = self._config.camera
        b = self._config.ball
        z = self._config.detection_zone
        conn = self._config.connection

        ctk.CTkLabel(
            tab, text="Advanced Settings", font=("", 14, "bold"),
        ).pack(anchor="w", pady=(5, 10))

        # Camera advanced
        self._mjpeg_var = ctk.BooleanVar(value=c.mjpeg)
        ctk.CTkCheckBox(
            tab, text="MJPEG Codec", variable=self._mjpeg_var,
        ).pack(anchor="w", pady=3)

        self._flip_view_var = ctk.BooleanVar(value=c.flip_view)
        ctk.CTkCheckBox(
            tab, text="Flip View", variable=self._flip_view_var,
        ).pack(anchor="w", pady=3)

        self._fps_override = self._add_slider(
            tab, "FPS Override (0=auto)", 0, 120, c.fps_override,
        )
        self._darkness = self._add_slider(
            tab, "Darkness", 0, 200, c.darkness,
        )

        # Ball advanced
        self._fixed_radius = self._add_slider(
            tab, "Fixed Radius (0=auto)", 0, 50, b.fixed_radius,
        )

        # Detection advanced
        self._gateway_w = self._add_slider(
            tab, "Gateway Width", 5, 50, z.gateway_width,
        )

        # Connection advanced
        port_frame = ctk.CTkFrame(tab, fg_color="transparent")
        port_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(
            port_frame, text="Port:", width=80, anchor="w",
        ).pack(side="left")
        self._port_entry = ctk.CTkEntry(port_frame, width=80)
        self._port_entry.insert(0, str(conn.gspro_port))
        self._port_entry.pack(side="left", padx=5)

        did_frame = ctk.CTkFrame(tab, fg_color="transparent")
        did_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(
            did_frame, text="Device ID:", width=80, anchor="w",
        ).pack(side="left")
        self._device_id_entry = ctk.CTkEntry(did_frame, width=180)
        self._device_id_entry.insert(0, conn.device_id)
        self._device_id_entry.pack(side="left", padx=5)

        url_frame = ctk.CTkFrame(tab, fg_color="transparent")
        url_frame.pack(fill="x", pady=3)
        ctk.CTkLabel(
            url_frame, text="HTTP URL:", width=80, anchor="w",
        ).pack(side="left")
        self._http_url_entry = ctk.CTkEntry(url_frame, width=280)
        self._http_url_entry.insert(0, conn.http_url)
        self._http_url_entry.pack(side="left", padx=5)

    # ---- Helpers ----

    def _add_slider(
        self,
        parent: Any,
        label: str,
        from_: int,
        to: int,
        initial: int,
    ) -> ctk.CTkSlider:
        """Add a labeled slider with value display."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=3)

        ctk.CTkLabel(frame, text=label, width=160, anchor="w").pack(side="left")

        value_label = ctk.CTkLabel(frame, text=str(initial), width=40)
        value_label.pack(side="right")

        slider = ctk.CTkSlider(
            frame, from_=from_, to=to, number_of_steps=to - from_,
            width=180,
            command=lambda v, lbl=value_label: lbl.configure(text=str(int(v))),
        )
        slider.set(initial)
        slider.pack(side="right", padx=5)

        return slider

    def _apply_to_config(self) -> None:
        """Read all widget values into the config object."""
        # Detection zone (only gateway width — zone position is via mouse drag)
        self._config.detection_zone.gateway_width = int(self._gateway_w.get())

        # Camera
        c = self._config.camera
        with contextlib.suppress(ValueError):
            c.webcam_index = int(self._cam_index.get())
        c.mjpeg = self._mjpeg_var.get()
        c.flip_image = self._flip_var.get()
        c.flip_view = self._flip_view_var.get()
        c.fps_override = int(self._fps_override.get())
        c.darkness = int(self._darkness.get())

        # Ball
        b = self._config.ball
        selected_label = self._color_var.get()
        b.color_preset = self._label_to_preset.get(
            selected_label, b.color_preset,
        )
        b.fixed_radius = int(self._fixed_radius.get())
        b.custom_hsv = None  # Changing preset clears custom HSV

        # Connection
        conn = self._config.connection
        conn.mode = self._mode_var.get()
        conn.gspro_host = self._host_entry.get().strip()
        with contextlib.suppress(ValueError):
            conn.gspro_port = int(self._port_entry.get())
        conn.device_id = self._device_id_entry.get().strip()
        conn.http_url = self._http_url_entry.get().strip()

    def _save_and_close(self) -> None:
        """Apply settings, save to disk, and close."""
        self._apply_to_config()
        try:
            save_config(self._config)
            logger.info("Settings saved")
        except Exception:
            logger.error("Failed to save settings", exc_info=True)

        if self._on_close:
            self._on_close()
        self.destroy()
