"""Main application window — CustomTkinter root with video, status, and inline settings."""

from __future__ import annotations

import contextlib
import logging
import queue
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from webcam_putting.color_presets import PRESET_DESCRIPTIONS
from webcam_putting.config import AppConfig, save_config
from webcam_putting.ui.video_panel import VideoPanel

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
    ):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Webcam Putting")
        self.geometry("940x480")
        self.resizable(False, False)

        self._config = config
        self._frame_queue = frame_queue
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_color_change = on_color_change
        self._on_settings_changed = on_settings_changed
        self._is_running = False
        self._edit_zone_active = False
        self._shot_history: list[ShotHistoryEntry] = []
        self._save_after_id: str | None = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    # ---- UI Construction ----

    def _build_ui(self) -> None:
        """Construct all UI elements."""
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: video panel
        video_container = ctk.CTkFrame(
            main_frame, width=644, height=364, corner_radius=6,
        )
        video_container.pack(side="left", fill="both", padx=(0, 8))
        video_container.pack_propagate(False)

        self._video_panel = VideoPanel(
            video_container, self._frame_queue, width=640, height=360,
        )
        self._video_panel.pack(padx=2, pady=2)

        # Right: tabbed panel (Status + Settings)
        right_frame = ctk.CTkFrame(main_frame, width=270)
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        self._right_tabs = ctk.CTkTabview(right_frame, width=260)
        self._right_tabs.pack(fill="both", expand=True, padx=4, pady=4)
        self._right_tabs.add("Status")
        self._right_tabs.add("Settings")

        self._build_status_tab()
        self._build_settings_tab()

        # Bottom: control bar
        control_bar = ctk.CTkFrame(self, height=44, fg_color="transparent")
        control_bar.pack(fill="x", padx=8, pady=(0, 8))
        self._build_control_bar(control_bar)

    def _build_status_tab(self) -> None:
        """Build the Status tab content."""
        tab = self._right_tabs.tab("Status")

        # Connection
        conn_frame = ctk.CTkFrame(tab, corner_radius=6)
        conn_frame.pack(fill="x", padx=4, pady=(4, 3))

        ctk.CTkLabel(
            conn_frame, text="CONNECTION", font=("", 10, "bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=6, pady=(4, 1))

        self._conn_indicator = ctk.CTkLabel(
            conn_frame, text="  Disconnected",
            font=("", 12), text_color="#ff4444",
        )
        self._conn_indicator.pack(anchor="w", padx=6, pady=(0, 4))

        # Last shot
        shot_frame = ctk.CTkFrame(tab, corner_radius=6)
        shot_frame.pack(fill="x", padx=4, pady=3)

        ctk.CTkLabel(
            shot_frame, text="LAST SHOT", font=("", 10, "bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=6, pady=(4, 1))

        self._speed_label = ctk.CTkLabel(
            shot_frame, text="-- MPH",
            font=("", 22, "bold"), text_color="white",
        )
        self._speed_label.pack(anchor="w", padx=6)

        self._hla_label = ctk.CTkLabel(
            shot_frame, text="-- HLA",
            font=("", 16), text_color="gray70",
        )
        self._hla_label.pack(anchor="w", padx=6, pady=(0, 4))

        # Shot history
        history_frame = ctk.CTkFrame(tab, corner_radius=6)
        history_frame.pack(fill="both", expand=True, padx=4, pady=3)

        ctk.CTkLabel(
            history_frame, text="SHOT HISTORY", font=("", 10, "bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=6, pady=(4, 1))

        self._history_text = ctk.CTkTextbox(
            history_frame, font=("Courier", 11), height=100,
            state="disabled", fg_color="gray14",
        )
        self._history_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # FPS / state
        fps_frame = ctk.CTkFrame(tab, corner_radius=6, height=32)
        fps_frame.pack(fill="x", padx=4, pady=(3, 4))
        fps_frame.pack_propagate(False)

        self._fps_label = ctk.CTkLabel(
            fps_frame, text="FPS: --", font=("", 12), text_color="gray60",
        )
        self._fps_label.pack(side="left", padx=6, pady=3)

        self._state_label = ctk.CTkLabel(
            fps_frame, text="idle", font=("", 11), text_color="gray50",
        )
        self._state_label.pack(side="right", padx=6, pady=3)

        self._shot_count_label = ctk.CTkLabel(
            fps_frame, text="Shots: 0", font=("", 12), text_color="gray60",
        )
        self._shot_count_label.pack(side="left", padx=6, pady=3)

    def _build_settings_tab(self) -> None:
        """Build the Settings tab with real-time controls."""
        tab = self._right_tabs.tab("Settings")
        c = self._config.camera
        b = self._config.ball
        z = self._config.detection_zone
        conn = self._config.connection

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # --- CAMERA ---
        self._section_label(scroll, "CAMERA")

        self._flip_var = self._add_live_checkbox(
            scroll, "Flip Image (Left-handed)", c.flip_image,
            lambda v: setattr(self._config.camera, "flip_image", v),
        )

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

        self._gateway_w = self._add_live_slider(
            scroll, "Gateway Width", 5, 50, z.gateway_width,
            lambda v: setattr(self._config.detection_zone, "gateway_width", v),
        )

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
            font=("", 9), text_color="gray50",
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
            command=self._on_mode_changed, font=("", 11),
        ).pack(anchor="w")
        ctk.CTkRadioButton(
            mode_frame, text="HTTP Middleware",
            variable=self._mode_var, value="http_middleware",
            command=self._on_mode_changed, font=("", 11),
        ).pack(anchor="w")

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
        )
        self._color_menu.pack(side="left", padx=(0, 8))

        # Start/Stop
        self._start_btn = ctk.CTkButton(
            parent, text="Start", command=self._toggle_running,
            width=100, fg_color="#2d8f2d", hover_color="#248f24",
        )
        self._start_btn.pack(side="left", padx=(0, 8))

        # Edit Zone toggle
        self._edit_zone_btn = ctk.CTkButton(
            parent, text="Edit Zone", command=self._toggle_edit_zone,
            width=100, fg_color="gray35", hover_color="gray30",
        )
        self._edit_zone_btn.pack(side="left")

        # Mode label (right side)
        mode_text = self._config.connection.mode.replace("_", " ").title()
        self._mode_label = ctk.CTkLabel(
            parent, text=mode_text, font=("", 11), text_color="gray50",
        )
        self._mode_label.pack(side="right", padx=8)

    # ---- Widget Helpers ----

    @staticmethod
    def _section_label(parent: Any, text: str) -> None:
        """Add a bold section header."""
        ctk.CTkLabel(
            parent, text=text, font=("", 10, "bold"), text_color="gray60",
        ).pack(anchor="w", pady=(8, 3))

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
            frame, text=label, width=110, anchor="w", font=("", 11),
        ).pack(side="left")

        value_label = ctk.CTkLabel(frame, text=str(initial), width=30, font=("", 11))
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
            command=on_change, font=("", 11),
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
            frame, text=label, width=70, anchor="w", font=("", 11),
        ).pack(side="left")

        entry = ctk.CTkEntry(frame, width=width, font=("", 11), height=26)
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

    def update_connection_status(self, connected: bool) -> None:
        """Update the connection indicator."""
        if connected:
            self._conn_indicator.configure(text="  Connected", text_color="#44cc44")
        else:
            self._conn_indicator.configure(text="  Disconnected", text_color="#ff4444")

    def update_shot(self, speed: float, hla: float, shot_number: int) -> None:
        """Update last shot display and add to history."""
        self._speed_label.configure(text=f"{speed:.1f} MPH")

        hla_text = f"{hla:+.1f}\u00b0 HLA"
        hla_color = (
            "#44cc44" if abs(hla) < 5
            else ("#ffaa00" if abs(hla) < 15 else "#ff4444")
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
                text="Edit Zone", fg_color="gray35", hover_color="gray30",
            )
            self._video_panel.set_edit_mode(False)
            self._on_setting_changed()
        else:
            self._edit_zone_active = True
            self._edit_zone_btn.configure(
                text="Done Editing", fg_color="#cc7700", hover_color="#aa6600",
            )
            self._video_panel.set_edit_mode(
                True, self._config.detection_zone,
            )

    def _toggle_running(self) -> None:
        """Toggle start/stop."""
        if self._is_running:
            self._is_running = False
            self._start_btn.configure(
                text="Start", fg_color="#2d8f2d", hover_color="#248f24",
            )
            if self._on_stop:
                self._on_stop()
        else:
            self._is_running = True
            self._start_btn.configure(
                text="Stop", fg_color="#cc3333", hover_color="#aa2222",
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
