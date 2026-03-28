"""Main application window — CustomTkinter root with video, status, and inline settings."""

from __future__ import annotations

import contextlib
import logging
import platform
import queue
import subprocess
import tkinter as tk
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from birdman_putting.color_presets import PRESET_DESCRIPTIONS
from birdman_putting.config import CONFIG_FILE, AppConfig, save_config
from birdman_putting.ui import theme
from birdman_putting.ui.video_panel import VideoPanel

logger = logging.getLogger(__name__)


class ShotHistoryEntry:
    """Single entry in shot history."""

    def __init__(self, number: int, speed: float, hla: float):
        self.number = number
        self.speed = speed
        self.hla = hla


class MainWindow(ctk.CTk):
    """Root application window with inline settings.

    Layout:
    ┌───────────────────────────┬───────────────────┐
    │                           │ [Status][Settings] │
    │      Video Panel          │                   │
    │      (640x360)            │  (tab content)    │
    │                           │                   │
    ├───────────────────────────┴───────────────────┤
    │  [Color ▼]  [Start/Stop]  [Edit Zone]         │
    └───────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: AppConfig,
        frame_queue: queue.Queue,  # type: ignore[type-arg]
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_color_change: Callable[[str], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
        on_auto_zone: Callable[[], None] | None = None,
        on_reset_putt: Callable[[], None] | None = None,
        on_angle_cal: Callable[[], None] | None = None,
    ):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Load Space Grotesk font
        self._font_family = theme.load_font()

        self.title("Birdman Putting")
        self.geometry("940x480")
        self.minsize(700, 400)
        self.resizable(True, True)
        self.configure(fg_color=theme.BG_ROOT)

        self._config = config
        self._frame_queue = frame_queue
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_color_change = on_color_change
        self._on_settings_changed = on_settings_changed
        self._on_auto_zone = on_auto_zone
        self._on_reset_putt = on_reset_putt
        self._on_angle_cal = on_angle_cal
        self._is_running = False
        self._edit_zone_active = False
        self._auto_zone_active = False
        self._angle_cal_active = False
        self._shot_history: list[ShotHistoryEntry] = []
        self._save_after_id: str | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)
        self.bind("<Left>", self._on_rotation_left)
        self.bind("<Right>", self._on_rotation_right)

    # ---- UI Construction ----

    def _build_ui(self) -> None:
        """Construct all UI elements using grid for proper resize behavior."""
        # Root grid: row 0 = main content (expands), row 1 = control bar (fixed)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        # Main frame grid: col 0 = video (expands), col 1 = right panel (fixed)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=0, minsize=270)

        # Left: video panel (expands to fill available space)
        video_container = ctk.CTkFrame(
            main_frame, corner_radius=theme.CORNER_RADIUS,
            fg_color=theme.BG_PANEL, border_width=1, border_color=theme.BORDER_SUBTLE,
        )
        video_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        self._video_panel = VideoPanel(
            video_container, self._frame_queue, width=640, height=360,
        )
        self._video_panel.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        # Right: tabbed panel (Status + Settings)
        right_frame = ctk.CTkFrame(
            main_frame, width=270,
            fg_color=theme.BG_PANEL, corner_radius=theme.CORNER_RADIUS,
            border_width=1, border_color=theme.BORDER_SUBTLE,
        )
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_propagate(False)

        self._right_tabs = ctk.CTkTabview(
            right_frame, width=260,
            fg_color=theme.BG_PANEL,
            segmented_button_fg_color=theme.BG_CARD,
            segmented_button_selected_color=theme.ACCENT_BLUE,
            segmented_button_unselected_color=theme.BG_CARD,
            segmented_button_selected_hover_color=theme.ACCENT_BLUE_HOVER,
            segmented_button_unselected_hover_color=theme.BG_CARD_HOVER,
            corner_radius=theme.CORNER_RADIUS,
        )
        self._right_tabs.pack(fill="both", expand=True, padx=4, pady=4)
        self._right_tabs.add("Status")
        self._right_tabs.add("Settings")

        self._build_status_tab()
        self._build_settings_tab()

        # Bottom: control bar (fixed height)
        control_bar = ctk.CTkFrame(
            self, height=44, fg_color=theme.BG_PANEL,
            corner_radius=theme.CORNER_RADIUS,
            border_width=1, border_color=theme.BORDER_SUBTLE,
        )
        control_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self._build_control_bar(control_bar)

    def _build_status_tab(self) -> None:
        """Build the Status tab content."""
        tab = self._right_tabs.tab("Status")
        tab.configure(fg_color=theme.BG_PANEL)

        def _card(parent: Any, **kw: Any) -> ctk.CTkFrame:
            return ctk.CTkFrame(
                parent, corner_radius=theme.CORNER_RADIUS_SM,
                fg_color=theme.BG_CARD, border_width=1,
                border_color=theme.BORDER_SUBTLE, **kw,
            )

        def _header(parent: Any, text: str) -> ctk.CTkLabel:
            return ctk.CTkLabel(
                parent, text=text, font=theme.font(9, "bold"),
                text_color=theme.TEXT_HEADER,
            )

        # Camera
        cam_frame = _card(tab)
        cam_frame.pack(fill="x", padx=4, pady=(4, 3))
        _header(cam_frame, "CAMERA").pack(anchor="w", padx=8, pady=(6, 1))

        self._cam_indicator = ctk.CTkLabel(
            cam_frame, text="  Idle",
            font=theme.font(13), text_color=theme.TEXT_SECONDARY,
        )
        self._cam_indicator.pack(anchor="w", padx=8, pady=(0, 6))

        # Connection
        conn_frame = _card(tab)
        conn_frame.pack(fill="x", padx=4, pady=(0, 3))
        _header(conn_frame, "GSPRO CONNECTION").pack(anchor="w", padx=8, pady=(6, 1))

        self._conn_indicator = ctk.CTkLabel(
            conn_frame, text="  Disconnected",
            font=theme.font(13), text_color=theme.STATUS_ERROR,
        )
        self._conn_indicator.pack(anchor="w", padx=8, pady=(0, 6))

        # Mevo
        mevo_frame = _card(tab)
        mevo_frame.pack(fill="x", padx=4, pady=(0, 3))
        _header(mevo_frame, "MEVO").pack(anchor="w", padx=8, pady=(6, 1))

        self._mevo_indicator = ctk.CTkLabel(
            mevo_frame, text="  Disabled",
            font=theme.font(13), text_color=theme.TEXT_SECONDARY,
        )
        self._mevo_indicator.pack(anchor="w", padx=8, pady=(0, 6))

        # Last shot
        shot_frame = _card(tab)
        shot_frame.pack(fill="x", padx=4, pady=3)
        _header(shot_frame, "LAST SHOT").pack(anchor="w", padx=8, pady=(6, 1))

        self._speed_label = ctk.CTkLabel(
            shot_frame, text="-- MPH",
            font=theme.font(22, "bold"), text_color=theme.ACCENT_GREEN,
        )
        self._speed_label.pack(anchor="w", padx=8)

        self._hla_label = ctk.CTkLabel(
            shot_frame, text="-- HLA",
            font=theme.font(16), text_color=theme.TEXT_SECONDARY,
        )
        self._hla_label.pack(anchor="w", padx=8, pady=(0, 6))

        # Shot history
        history_frame = _card(tab)
        history_frame.pack(fill="both", expand=True, padx=4, pady=3)
        _header(history_frame, "SESSION HISTORY").pack(anchor="w", padx=8, pady=(6, 1))

        self._history_text = ctk.CTkTextbox(
            history_frame, font=theme.font(11), height=100,
            state="disabled", fg_color=theme.BG_INPUT,
            text_color=theme.TEXT_PRIMARY,
            border_width=0, corner_radius=theme.CORNER_RADIUS_SM,
        )
        self._history_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # FPS / state / shot count
        info_frame = _card(tab)
        info_frame.pack(fill="x", padx=4, pady=(3, 4))

        # Row 1: FPS + State
        row1 = ctk.CTkFrame(info_frame, fg_color="transparent")
        row1.pack(fill="x", padx=8, pady=(6, 2))

        self._fps_label = ctk.CTkLabel(
            row1, text="FPS: --", font=theme.font(12), text_color=theme.TEXT_PRIMARY,
        )
        self._fps_label.pack(side="left")

        self._state_label = ctk.CTkLabel(
            row1, text="idle", font=theme.font(12), text_color=theme.STATUS_IDLE,
        )
        self._state_label.pack(side="right")

        # Row 2: Shot count
        self._shot_count_label = ctk.CTkLabel(
            info_frame, text="Shots: 0", font=theme.font(12),
            text_color=theme.TEXT_PRIMARY,
        )
        self._shot_count_label.pack(anchor="w", padx=8, pady=(0, 6))

    def _build_settings_tab(self) -> None:
        """Build the Settings tab with real-time controls."""
        tab = self._right_tabs.tab("Settings")
        tab.configure(fg_color=theme.BG_PANEL)
        c = self._config.camera
        b = self._config.ball
        z = self._config.detection_zone
        conn = self._config.connection

        scroll = ctk.CTkScrollableFrame(
            tab, fg_color="transparent",
            scrollbar_button_color=theme.BG_CARD,
            scrollbar_button_hover_color=theme.BG_CARD_HOVER,
        )
        scroll.pack(fill="both", expand=True)

        # --- CAMERA ---
        self._section_label(scroll, "CAMERA")

        self._flip_var = self._add_live_checkbox(
            scroll, "Flip Image (Left-handed)", c.flip_image,
            lambda v: setattr(self._config.camera, "flip_image", v),
        )

        self._rotation_entry = self._add_entry(
            scroll, "Rotation:", str(c.rotation), width=60,
        )
        ctk.CTkLabel(
            self._rotation_entry.master, text="\u00b0 (-45 to 45)",
            font=theme.font(10), text_color=theme.TEXT_MUTED,
        ).pack(side="left", padx=2)
        self._bind_entry_apply(self._rotation_entry, self._apply_rotation)

        self._autofocus_var = self._add_live_checkbox(
            scroll, "Autofocus", bool(c.autofocus),
            self._on_autofocus_toggle,
        )

        self._focus_slider = self._add_live_slider(
            scroll, "Manual Focus", 0, 255, int(c.focus),
            lambda v: setattr(self._config.camera, "focus", float(v)),
        )
        if self._autofocus_var.get():
            self._focus_slider.configure(state="disabled")

        auto_exp_on = c.auto_exposure != 1.0
        self._auto_exposure_var = self._add_live_checkbox(
            scroll, "Auto-Exposure", auto_exp_on,
            self._on_auto_exposure_toggle,
        )

        initial_exp = int(c.exposure) if c.exposure != 0.0 else -6
        self._exposure_slider = self._add_live_slider(
            scroll, "Exposure", -13, -1, initial_exp,
            lambda v: setattr(self._config.camera, "exposure", float(v)),
        )
        if auto_exp_on:
            self._exposure_slider.configure(state="disabled")

        # --- DETECTION ---
        self._section_label(scroll, "DETECTION")

        # Roll direction dropdown
        dir_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        dir_frame.pack(fill="x", pady=1)
        ctk.CTkLabel(
            dir_frame, text="Roll Direction", width=110, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")

        dir_display = {
            "left_to_right": "Left to Right",
            "right_to_left": "Right to Left",
        }
        self._dir_to_value = {v: k for k, v in dir_display.items()}
        current_dir = dir_display.get(z.direction, "Left to Right")
        self._direction_var = ctk.StringVar(value=current_dir)
        ctk.CTkOptionMenu(
            dir_frame, variable=self._direction_var,
            values=list(dir_display.values()),
            command=self._on_direction_changed, width=130,
            font=theme.font(11),
            fg_color=theme.BG_CARD, button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
            dropdown_fg_color=theme.BG_CARD,
            dropdown_hover_color=theme.BG_CARD_HOVER,
            dropdown_text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.CORNER_RADIUS_SM,
        ).pack(side="left", padx=2)

        self._gateway_w = self._add_live_slider(
            scroll, "Gateway Width", 5, 50, z.gateway_width,
            lambda v: setattr(self._config.detection_zone, "gateway_width", v),
        )

        # Zone color dropdowns
        zone_colors = [
            "Yellow", "Red", "Green", "Cyan", "White",
            "Orange", "Magenta", "Blue", "Gray",
        ]

        zone_color_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        zone_color_frame.pack(fill="x", pady=1)
        ctk.CTkLabel(
            zone_color_frame, text="Zone Color", width=110, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")
        self._zone_color_var = ctk.StringVar(value=z.zone_color.capitalize())
        ctk.CTkOptionMenu(
            zone_color_frame, variable=self._zone_color_var,
            values=zone_colors, command=self._on_zone_color_changed, width=100,
            font=theme.font(11),
            fg_color=theme.BG_CARD, button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
            dropdown_fg_color=theme.BG_CARD,
            dropdown_hover_color=theme.BG_CARD_HOVER,
            dropdown_text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.CORNER_RADIUS_SM,
        ).pack(side="left", padx=2)

        gw_color_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        gw_color_frame.pack(fill="x", pady=1)
        ctk.CTkLabel(
            gw_color_frame, text="Gateway Color", width=110, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")
        self._gw_color_var = ctk.StringVar(value=z.gateway_color.capitalize())
        ctk.CTkOptionMenu(
            gw_color_frame, variable=self._gw_color_var,
            values=zone_colors, command=self._on_gateway_color_changed, width=100,
            font=theme.font(11),
            fg_color=theme.BG_CARD, button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
            dropdown_fg_color=theme.BG_CARD,
            dropdown_hover_color=theme.BG_CARD_HOVER,
            dropdown_text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.CORNER_RADIUS_SM,
        ).pack(side="left", padx=2)

        self._fixed_radius = self._add_live_slider(
            scroll, "Fixed Radius", 0, 50, b.fixed_radius,
            lambda v: setattr(self._config.ball, "fixed_radius", v),
        )

        # --- ADVANCED (some require restart) ---
        self._section_label(scroll, "ADVANCED")

        self._flip_view_var = self._add_live_checkbox(
            scroll, "Flip View", c.flip_view,
            lambda v: setattr(self._config.camera, "flip_view", v),
        )

        self._darkness_slider = self._add_live_slider(
            scroll, "Darkness", 0, 200, c.darkness,
            lambda v: setattr(self._config.camera, "darkness", v),
        )

        ctk.CTkLabel(
            scroll, text="Restart required for changes below:",
            font=theme.font(9), text_color=theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(6, 2))

        self._mjpeg_var = self._add_live_checkbox(
            scroll, "MJPEG Codec", c.mjpeg,
            lambda v: setattr(self._config.camera, "mjpeg", v),
            live=False,
        )

        self._fps_override_slider = self._add_live_slider(
            scroll, "FPS Override", 0, 120, c.fps_override,
            lambda v: setattr(self._config.camera, "fps_override", v),
            live=False,
        )

        self._cam_index_entry = self._add_entry(
            scroll, "Webcam:", str(c.webcam_index), width=50,
        )
        self._bind_entry_apply(self._cam_index_entry, self._apply_webcam_index)

        self._host_entry = self._add_entry(
            scroll, "Host:", conn.gspro_host, width=140,
        )
        self._bind_entry_apply(
            self._host_entry,
            lambda v: setattr(self._config.connection, "gspro_host", v.strip()),
        )

        self._port_entry = self._add_entry(
            scroll, "Port:", str(conn.gspro_port), width=60,
        )
        self._bind_entry_apply(self._port_entry, self._apply_port)

        self._device_id_entry = self._add_entry(
            scroll, "Dev ID:", conn.device_id, width=140,
        )
        self._bind_entry_apply(
            self._device_id_entry,
            lambda v: setattr(self._config.connection, "device_id", v.strip()),
        )

        self._http_url_entry = self._add_entry(
            scroll, "URL:", conn.http_url, width=170,
        )
        self._bind_entry_apply(
            self._http_url_entry,
            lambda v: setattr(self._config.connection, "http_url", v.strip()),
        )

        # Connection mode
        mode_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        mode_frame.pack(fill="x", pady=3)
        self._mode_var = ctk.StringVar(value=conn.mode)
        ctk.CTkRadioButton(
            mode_frame, text="Direct GSPro",
            variable=self._mode_var, value="gspro_direct",
            command=self._on_mode_changed, font=theme.font(11),
            text_color=theme.TEXT_SECONDARY,
            fg_color=theme.ACCENT_BLUE, hover_color=theme.ACCENT_BLUE_HOVER,
        ).pack(anchor="w")
        ctk.CTkRadioButton(
            mode_frame, text="HTTP Middleware",
            variable=self._mode_var, value="http_middleware",
            command=self._on_mode_changed, font=theme.font(11),
            text_color=theme.TEXT_SECONDARY,
            fg_color=theme.ACCENT_BLUE, hover_color=theme.ACCENT_BLUE_HOVER,
        ).pack(anchor="w")

        # --- MEVO ---
        self._section_label(scroll, "MEVO (LAUNCH MONITOR)")

        self._mevo_enabled_var = self._add_live_checkbox(
            scroll, "Enable Mevo OCR (display + OBS)", self._config.mevo.enabled,
            lambda v: setattr(self._config.mevo, "enabled", v),
            live=False,
        )

        self._mevo_send_var = self._add_live_checkbox(
            scroll, "Send Mevo data to GSPro",
            self._config.mevo.send_to_gspro,
            lambda v: setattr(self._config.mevo, "send_to_gspro", v),
            live=False,
        )
        ctk.CTkLabel(
            scroll, text="Disable if LM connects to GSPro directly",
            font=theme.font(9), text_color=theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 2))

        self._mevo_window_entry = self._add_entry(
            scroll, "Window:", self._config.mevo.window_title, width=140,
        )
        self._bind_entry_apply(
            self._mevo_window_entry,
            lambda v: setattr(self._config.mevo, "window_title", v.strip()),
        )

        self._mevo_tessdata_entry = self._add_entry(
            scroll, "Tessdata:", self._config.mevo.tessdata_dir, width=140,
        )
        self._bind_entry_apply(
            self._mevo_tessdata_entry,
            lambda v: setattr(self._config.mevo, "tessdata_dir", v.strip()),
        )

        ctk.CTkLabel(
            scroll, text="Configure ROIs in config.toml",
            font=theme.font(9), text_color=theme.TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        # --- OVERLAY / OBS ---
        self._section_label(scroll, "OVERLAY (OBS)")

        ov = self._config.overlay
        self._obs_overlay_var = self._add_live_checkbox(
            scroll, "OBS Overlay Mode (black bg, tracer only)",
            ov.obs_overlay_mode,
            lambda v: setattr(self._config.overlay, "obs_overlay_mode", v),
        )

        self._obs_show_zones_var = self._add_live_checkbox(
            scroll, "Show Zones in OBS Mode",
            ov.obs_show_zones,
            lambda v: setattr(self._config.overlay, "obs_show_zones", v),
        )

        self._projected_trail_var = self._add_live_checkbox(
            scroll, "Projected Trail (calculated trajectory)",
            ov.projected_trail,
            lambda v: setattr(self._config.overlay, "projected_trail", v),
        )

        self._trail_points_slider = self._add_live_slider(
            scroll, "Trail Length", 50, 500, ov.max_trail_points,
            self._on_trail_length_changed,
        )

        # Trail color dropdowns
        trail_colors = [
            "Cyan", "Green", "Yellow", "Red", "White",
            "Orange", "Magenta", "Blue",
        ]

        trail_color_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        trail_color_frame.pack(fill="x", pady=1)
        ctk.CTkLabel(
            trail_color_frame, text="Trail Color", width=110, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")
        self._trail_color_var = ctk.StringVar(value=ov.trail_color.capitalize())
        ctk.CTkOptionMenu(
            trail_color_frame, variable=self._trail_color_var,
            values=trail_colors, command=self._on_trail_color_changed, width=100,
            font=theme.font(11),
            fg_color=theme.BG_CARD, button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
            dropdown_fg_color=theme.BG_CARD,
            dropdown_hover_color=theme.BG_CARD_HOVER,
            dropdown_text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.CORNER_RADIUS_SM,
        ).pack(side="left", padx=2)

        active_color_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        active_color_frame.pack(fill="x", pady=1)
        ctk.CTkLabel(
            active_color_frame, text="Active Color", width=110, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")
        self._active_trail_color_var = ctk.StringVar(
            value=ov.active_trail_color.capitalize(),
        )
        ctk.CTkOptionMenu(
            active_color_frame, variable=self._active_trail_color_var,
            values=trail_colors,
            command=self._on_active_trail_color_changed, width=100,
            font=theme.font(11),
            fg_color=theme.BG_CARD, button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
            dropdown_fg_color=theme.BG_CARD,
            dropdown_hover_color=theme.BG_CARD_HOVER,
            dropdown_text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.CORNER_RADIUS_SM,
        ).pack(side="left", padx=2)

    def _on_trail_length_changed(self, val: int) -> None:
        """Update trail length in config and resize tracker deque."""
        self._config.overlay.max_trail_points = val

    def _on_trail_color_changed(self, label: str) -> None:
        """Handle trail color dropdown change."""
        self._config.overlay.trail_color = label.lower()
        self._on_setting_changed()

    def _on_active_trail_color_changed(self, label: str) -> None:
        """Handle active trail color dropdown change."""
        self._config.overlay.active_trail_color = label.lower()
        self._on_setting_changed()

    def _build_control_bar(self, parent: ctk.CTkFrame) -> None:
        """Build the bottom control bar."""
        # Ball color dropdown
        preset_labels = list(PRESET_DESCRIPTIONS.values())
        self._label_to_preset = {v: k for k, v in PRESET_DESCRIPTIONS.items()}

        current_preset = self._config.ball.color_preset
        current_label = PRESET_DESCRIPTIONS.get(current_preset, "Yellow (bright)")

        self._color_var = ctk.StringVar(value=current_label)
        self._color_menu = ctk.CTkOptionMenu(
            parent, variable=self._color_var, values=preset_labels,
            command=self._on_color_selected, width=180,
            font=theme.font(11),
            fg_color=theme.BG_CARD, button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
            dropdown_fg_color=theme.BG_CARD,
            dropdown_hover_color=theme.BG_CARD_HOVER,
            dropdown_text_color=theme.TEXT_PRIMARY,
            corner_radius=theme.CORNER_RADIUS,
        )
        self._color_menu.pack(side="left", padx=(8, 8))

        # Start/Stop
        self._start_btn = ctk.CTkButton(
            parent, text="Start", command=self._toggle_running,
            width=100, font=theme.font(12, "bold"),
            fg_color=theme.BTN_SUCCESS[0], hover_color=theme.BTN_SUCCESS[1],
            corner_radius=theme.CORNER_RADIUS,
        )
        self._start_btn.pack(side="left", padx=(0, 8))

        # Edit Zone toggle
        self._edit_zone_btn = ctk.CTkButton(
            parent, text="Edit Zone", command=self._toggle_edit_zone,
            width=100, font=theme.font(11),
            fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            corner_radius=theme.CORNER_RADIUS,
        )
        self._edit_zone_btn.pack(side="left")

        # Auto Zone button
        self._auto_zone_btn = ctk.CTkButton(
            parent, text="Auto Zone", command=self._on_auto_zone_clicked,
            width=100, font=theme.font(11),
            fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            corner_radius=theme.CORNER_RADIUS,
        )
        self._auto_zone_btn.pack(side="left", padx=(8, 0))

        # Reset Putt button
        self._reset_putt_btn = ctk.CTkButton(
            parent, text="Reset Putt", command=self._on_reset_putt_clicked,
            width=90, font=theme.font(11),
            fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            corner_radius=theme.CORNER_RADIUS,
        )
        self._reset_putt_btn.pack(side="left", padx=(8, 0))

        # Edit Config button
        self._edit_config_btn = ctk.CTkButton(
            parent, text="Edit Config", command=self._on_edit_config,
            width=100, font=theme.font(11),
            fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            corner_radius=theme.CORNER_RADIUS,
        )
        self._edit_config_btn.pack(side="left", padx=(8, 0))

        # Angle Cal button
        self._angle_cal_btn = ctk.CTkButton(
            parent, text="Angle Cal", command=self._on_angle_cal_clicked,
            width=90, font=theme.font(11),
            fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            corner_radius=theme.CORNER_RADIUS,
        )
        self._angle_cal_btn.pack(side="left", padx=(8, 0))

        # Mode label (right side)
        mode_text = self._config.connection.mode.replace("_", " ").title()
        self._mode_label = ctk.CTkLabel(
            parent, text=mode_text, font=theme.font(11), text_color=theme.TEXT_MUTED,
        )
        self._mode_label.pack(side="right", padx=8)

    # ---- Widget Helpers ----

    @staticmethod
    def _section_label(parent: Any, text: str) -> None:
        """Add a bold section header."""
        ctk.CTkLabel(
            parent, text=text, font=theme.font(9, "bold"),
            text_color=theme.TEXT_HEADER,
        ).pack(anchor="w", pady=(theme.SECTION_PAD_TOP, 3))

    def _add_live_slider(
        self,
        parent: Any,
        label: str,
        from_: int,
        to: int,
        initial: int,
        setter: Callable[[int], None],
        live: bool = True,
    ) -> ctk.CTkSlider:
        """Add a slider that applies changes in real-time."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=1)

        ctk.CTkLabel(
            frame, text=label, width=110, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")

        value_label = ctk.CTkLabel(
            frame, text=str(initial), width=30,
            font=theme.font(11), text_color=theme.TEXT_PRIMARY,
        )
        value_label.pack(side="right")

        def on_change(v: float) -> None:
            val = int(v)
            value_label.configure(text=str(val))
            setter(val)
            if live:
                self._on_setting_changed()
            else:
                self._schedule_save()

        slider = ctk.CTkSlider(
            frame, from_=from_, to=to,
            number_of_steps=max(1, abs(to - from_)),
            width=90, command=on_change,
            fg_color=theme.SLIDER_BG,
            progress_color=theme.SLIDER_PROGRESS,
            button_color=theme.ACCENT_BLUE,
            button_hover_color=theme.ACCENT_BLUE_HOVER,
        )
        slider.set(initial)
        slider.pack(side="right", padx=2)

        return slider

    def _add_live_checkbox(
        self,
        parent: Any,
        text: str,
        initial: bool,
        setter: Callable[[bool], None],
        live: bool = True,
    ) -> ctk.BooleanVar:
        """Add a checkbox that applies changes in real-time."""
        var = ctk.BooleanVar(value=initial)

        def on_change() -> None:
            setter(var.get())
            if live:
                self._on_setting_changed()
            else:
                self._schedule_save()

        ctk.CTkCheckBox(
            parent, text=text, variable=var,
            command=on_change, font=theme.font(11),
            text_color=theme.TEXT_SECONDARY,
            fg_color=theme.ACCENT_BLUE,
            hover_color=theme.ACCENT_BLUE_HOVER,
            border_color=theme.BORDER_SUBTLE,
            checkmark_color=theme.TEXT_PRIMARY,
            corner_radius=4,
        ).pack(anchor="w", pady=1)
        return var

    @staticmethod
    def _add_entry(
        parent: Any, label: str, initial: str, width: int = 120,
    ) -> ctk.CTkEntry:
        """Add a labeled text entry."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=1)

        ctk.CTkLabel(
            frame, text=label, width=70, anchor="w",
            font=theme.font(11), text_color=theme.TEXT_SECONDARY,
        ).pack(side="left")

        entry = ctk.CTkEntry(
            frame, width=width, font=theme.font(11), height=26,
            fg_color=theme.BG_INPUT, text_color=theme.TEXT_PRIMARY,
            border_color=theme.BORDER_SUBTLE, border_width=1,
            corner_radius=theme.CORNER_RADIUS_SM,
        )
        entry.insert(0, initial)
        entry.pack(side="left", padx=2)
        return entry

    def _bind_entry_apply(
        self, entry: ctk.CTkEntry, setter: Callable[[str], None],
    ) -> None:
        """Bind Return and FocusOut to apply an entry's value and schedule save."""
        def apply(event: Any = None) -> None:
            setter(entry.get())
            self._schedule_save()

        entry.bind("<Return>", apply)
        entry.bind("<FocusOut>", apply)

    # ---- Settings Callbacks ----

    def _on_autofocus_toggle(self, checked: bool) -> None:
        """Handle autofocus checkbox change."""
        self._config.camera.autofocus = 1.0 if checked else 0.0
        if checked:
            self._focus_slider.configure(state="disabled")
            self._config.camera.focus = 0.0
        else:
            self._focus_slider.configure(state="normal")

    def _on_auto_exposure_toggle(self, checked: bool) -> None:
        """Handle auto-exposure checkbox change."""
        self._config.camera.auto_exposure = 3.0 if checked else 1.0
        if checked:
            self._exposure_slider.configure(state="disabled")
            self._config.camera.exposure = 0.0
        else:
            self._exposure_slider.configure(state="normal")

    def _on_mode_changed(self) -> None:
        """Handle connection mode radio change."""
        self._config.connection.mode = self._mode_var.get()
        mode_text = self._mode_var.get().replace("_", " ").title()
        self._mode_label.configure(text=mode_text)
        self._schedule_save()

    def _apply_webcam_index(self, value: str) -> None:
        """Parse and apply webcam index from entry."""
        with contextlib.suppress(ValueError):
            self._config.camera.webcam_index = int(value)

    def _apply_rotation(self, value: str) -> None:
        """Parse and apply rotation from entry (float, clamped to [-45, 45])."""
        try:
            v = float(value)
        except ValueError:
            return
        v = max(-45.0, min(45.0, v))
        self._config.camera.rotation = v
        self._rotation_entry.delete(0, "end")
        self._rotation_entry.insert(0, str(v))
        self._on_setting_changed()

    def _on_rotation_left(self, event: Any) -> None:
        """Decrease rotation by 0.5 degrees (Left arrow key)."""
        if isinstance(self.focus_get(), (ctk.CTkEntry, tk.Entry)):
            return
        current = float(self._rotation_entry.get() or "0")
        self._apply_rotation(str(current - 0.5))

    def _on_rotation_right(self, event: Any) -> None:
        """Increase rotation by 0.5 degrees (Right arrow key)."""
        if isinstance(self.focus_get(), (ctk.CTkEntry, tk.Entry)):
            return
        current = float(self._rotation_entry.get() or "0")
        self._apply_rotation(str(current + 0.5))

    def _apply_port(self, value: str) -> None:
        """Parse and apply port from entry."""
        with contextlib.suppress(ValueError):
            self._config.connection.gspro_port = int(value)

    # ---- Real-time Apply + Debounced Save ----

    def _on_setting_changed(self) -> None:
        """Apply settings to running systems and schedule a debounced save."""
        if self._on_settings_changed:
            self._on_settings_changed()
        self._schedule_save()

    def _schedule_save(self) -> None:
        """Schedule config save 2s after the last change."""
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
        self._save_after_id = self.after(2000, self._do_save)

    def _do_save(self) -> None:
        """Save config to disk."""
        self._save_after_id = None
        try:
            save_config(self._config)
            logger.info("Config auto-saved")
        except Exception:
            logger.error("Failed to auto-save config", exc_info=True)

    # ---- Public Update Methods (called from processing thread via after()) ----

    def update_camera_status(self, status: str, state: str = "idle") -> None:
        """Update the camera status indicator.

        Args:
            status: Human-readable status text.
            state: One of "ok", "error", or "idle".
        """
        colors = {"ok": theme.STATUS_OK, "error": theme.STATUS_ERROR, "idle": theme.STATUS_IDLE}
        self._cam_indicator.configure(
            text=f"  {status}", text_color=colors.get(state, theme.STATUS_IDLE),
        )

    def update_connection_status(self, connected: bool) -> None:
        """Update the connection indicator."""
        if connected:
            self._conn_indicator.configure(text="  Connected", text_color=theme.STATUS_OK)
        else:
            self._conn_indicator.configure(text="  Disconnected", text_color=theme.STATUS_ERROR)

    def update_mevo_status(self, text: str, state: str = "disabled") -> None:
        """Update the Mevo status indicator.

        Args:
            text: Human-readable status text.
            state: One of "ok", "error", "watching", or "disabled".
        """
        colors = {
            "ok": theme.STATUS_OK,
            "error": theme.STATUS_ERROR,
            "watching": theme.STATUS_WARNING,
            "disabled": theme.STATUS_IDLE,
        }
        self._mevo_indicator.configure(
            text=f"  {text}", text_color=colors.get(state, theme.STATUS_IDLE),
        )

    def update_shot(self, speed: float, hla: float, shot_number: int) -> None:
        """Update last shot display and add to history."""
        self._speed_label.configure(text=f"{speed:.1f} MPH")

        hla_text = f"{hla:+.1f}\u00b0 HLA"
        hla_color = (
            theme.STATUS_OK if abs(hla) < 5
            else (theme.STATUS_WARNING if abs(hla) < 15 else theme.STATUS_ERROR)
        )
        self._hla_label.configure(text=hla_text, text_color=hla_color)

        entry = ShotHistoryEntry(shot_number, speed, hla)
        self._shot_history.insert(0, entry)
        self._shot_history = self._shot_history[:50]

        self._history_text.configure(state="normal")
        self._history_text.delete("1.0", "end")
        for e in self._shot_history:
            self._history_text.insert(
                "end",
                f"#{e.number:>3d}  {e.speed:5.1f} mph  {e.hla:+6.1f}\u00b0\n",
            )
        self._history_text.configure(state="disabled")

    def update_fps(self, fps: float) -> None:
        """Update FPS display."""
        self._fps_label.configure(text=f"FPS: {fps:.0f}")

    def update_state(self, state_text: str) -> None:
        """Update state display."""
        self._state_label.configure(text=state_text)

    def update_shot_count(self, count: int) -> None:
        """Update the shot count display."""
        self._shot_count_label.configure(text=f"Shots: {count}")

    def start_video(self) -> None:
        """Begin video panel polling."""
        self._video_panel.start()

    def stop_video(self) -> None:
        """Stop video panel polling."""
        self._video_panel.stop()

    @property
    def edit_zone_mode(self) -> bool:
        """Whether zone editing is active."""
        return self._edit_zone_active

    # ---- Internal Callbacks ----

    def _toggle_edit_zone(self) -> None:
        """Toggle detection zone edit mode."""
        if self._edit_zone_active:
            self._edit_zone_active = False
            self._edit_zone_btn.configure(
                text="Edit Zone",
                fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            )
            self._video_panel.set_edit_mode(False)
            self._on_setting_changed()
        else:
            self._edit_zone_active = True
            self._edit_zone_btn.configure(
                text="Done Editing",
                fg_color=theme.BTN_WARNING[0], hover_color=theme.BTN_WARNING[1],
            )
            self._video_panel.set_edit_mode(
                True, self._config.detection_zone,
                on_zone_changed=self._on_setting_changed,
            )

    def _toggle_running(self) -> None:
        """Toggle start/stop."""
        if self._is_running:
            self._is_running = False
            self._start_btn.configure(
                text="Start",
                fg_color=theme.BTN_SUCCESS[0], hover_color=theme.BTN_SUCCESS[1],
            )
            if self._on_stop:
                self._on_stop()
        else:
            self._is_running = True
            self._start_btn.configure(
                text="Stop",
                fg_color=theme.BTN_DANGER[0], hover_color=theme.BTN_DANGER[1],
            )
            if self._on_start:
                self._on_start()

    def _on_color_selected(self, label: str) -> None:
        """Handle ball color dropdown change."""
        preset_name = self._label_to_preset.get(label)
        if preset_name:
            self._config.ball.color_preset = preset_name
            self._config.ball.custom_hsv = None
            logger.info("Ball color changed to: %s", preset_name)
            if self._on_color_change:
                self._on_color_change(preset_name)
            self._schedule_save()

    def _on_reset_putt_clicked(self) -> None:
        """Handle Reset Putt button click."""
        if self._on_reset_putt:
            self._on_reset_putt()

    def _on_auto_zone_clicked(self) -> None:
        """Handle Auto Zone button click."""
        if self._on_auto_zone:
            self._on_auto_zone()

    def set_auto_zone_state(self, active: bool) -> None:
        """Update Auto Zone button appearance."""
        self._auto_zone_active = active
        if active:
            self._auto_zone_btn.configure(
                text="Cancel Cal.",
                fg_color=theme.BTN_WARNING[0], hover_color=theme.BTN_WARNING[1],
            )
        else:
            self._auto_zone_btn.configure(
                text="Auto Zone",
                fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            )

    def _on_angle_cal_clicked(self) -> None:
        """Handle Angle Cal button click."""
        if self._on_angle_cal:
            self._on_angle_cal()

    def set_angle_cal_state(self, active: bool) -> None:
        """Update Angle Cal button appearance."""
        self._angle_cal_active = active
        if active:
            self._angle_cal_btn.configure(
                text="Cancel Cal.",
                fg_color=theme.BTN_WARNING[0], hover_color=theme.BTN_WARNING[1],
            )
        else:
            self._angle_cal_btn.configure(
                text="Angle Cal",
                fg_color=theme.BTN_SECONDARY[0], hover_color=theme.BTN_SECONDARY[1],
            )

    def _on_edit_config(self) -> None:
        """Open config.toml in the system's default text editor."""
        try:
            save_config(self._config)
        except Exception:
            logger.error("Failed to save config before opening editor", exc_info=True)
        path = str(CONFIG_FILE)
        try:
            system = platform.system()
            if system == "Windows":
                import os
                os.startfile(path)  # type: ignore[attr-defined]  # noqa: S606
            elif system == "Darwin":
                subprocess.Popen(["open", path])  # noqa: S603
            else:
                subprocess.Popen(["xdg-open", path])  # noqa: S603
        except Exception:
            logger.error("Failed to open config file in editor", exc_info=True)

    def _on_zone_color_changed(self, label: str) -> None:
        """Handle zone color dropdown change."""
        self._config.detection_zone.zone_color = label.lower()
        self._on_setting_changed()

    def _on_gateway_color_changed(self, label: str) -> None:
        """Handle gateway color dropdown change."""
        self._config.detection_zone.gateway_color = label.lower()
        self._on_setting_changed()

    def _on_direction_changed(self, label: str) -> None:
        """Handle roll direction dropdown change."""
        value = self._dir_to_value.get(label, "left_to_right")
        self._config.detection_zone.direction = value
        self._on_setting_changed()

    def _on_window_close(self) -> None:
        """Handle window close — flush pending save, clean up."""
        if self._edit_zone_active:
            self._toggle_edit_zone()
        # Flush any pending debounced save
        if self._save_after_id is not None:
            self.after_cancel(self._save_after_id)
            self._do_save()
        self._video_panel.stop()
        if self._on_stop:
            self._on_stop()
        self.destroy()
