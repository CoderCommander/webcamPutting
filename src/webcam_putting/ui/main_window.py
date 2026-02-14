"""Main application window — CustomTkinter root with video, status, and controls."""

from __future__ import annotations

import logging
import queue
from collections.abc import Callable

import customtkinter as ctk

from webcam_putting.color_presets import PRESET_DESCRIPTIONS
from webcam_putting.config import AppConfig
from webcam_putting.ui.settings_panel import SettingsPanel
from webcam_putting.ui.video_panel import VideoPanel

logger = logging.getLogger(__name__)


class ShotHistoryEntry:
    """Single entry in shot history."""

    def __init__(self, number: int, speed: float, hla: float):
        self.number = number
        self.speed = speed
        self.hla = hla


class MainWindow(ctk.CTk):
    """Root application window.

    Layout:
    ┌───────────────────────────┬──────────────┐
    │                           │  Connection  │
    │      Video Panel          │  Status      │
    │      (640x360)            │  Last Shot   │
    │                           │  Shot History│
    │                           │  FPS         │
    ├───────────────────────────┴──────────────┤
    │  [Color ▼]  [Start/Stop]  [Settings]     │
    └──────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: AppConfig,
        frame_queue: queue.Queue,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        on_color_change: Callable[[str], None] | None = None,
        on_settings_changed: Callable[[], None] | None = None,
    ):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Webcam Putting")
        self.geometry("890x460")
        self.resizable(False, False)

        self._config = config
        self._frame_queue = frame_queue
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_color_change = on_color_change
        self._on_settings_changed = on_settings_changed
        self._is_running = False
        self._settings_window: SettingsPanel | None = None
        self._shot_history: list[ShotHistoryEntry] = []

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _build_ui(self) -> None:
        """Construct all UI elements."""
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: video panel
        video_container = ctk.CTkFrame(main_frame, width=644, height=364, corner_radius=6)
        video_container.pack(side="left", fill="both", padx=(0, 8))
        video_container.pack_propagate(False)

        self._video_panel = VideoPanel(
            video_container,
            self._frame_queue,
            width=640,
            height=360,
        )
        self._video_panel.pack(padx=2, pady=2)

        # Right: status panel
        right_frame = ctk.CTkFrame(main_frame, width=220)
        right_frame.pack(side="right", fill="y")
        right_frame.pack_propagate(False)

        self._build_status_panel(right_frame)

        # Bottom: control bar
        control_bar = ctk.CTkFrame(self, height=44, fg_color="transparent")
        control_bar.pack(fill="x", padx=8, pady=(0, 8))

        self._build_control_bar(control_bar)

    def _build_status_panel(self, parent: ctk.CTkFrame) -> None:
        """Build the right-side status panel."""
        # Connection status
        conn_frame = ctk.CTkFrame(parent, corner_radius=6)
        conn_frame.pack(fill="x", padx=6, pady=(6, 4))

        ctk.CTkLabel(
            conn_frame, text="CONNECTION", font=("", 10, "bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=8, pady=(6, 2))

        self._conn_indicator = ctk.CTkLabel(
            conn_frame, text="  Disconnected",
            font=("", 12), text_color="#ff4444",
        )
        self._conn_indicator.pack(anchor="w", padx=8, pady=(0, 6))

        # Last shot
        shot_frame = ctk.CTkFrame(parent, corner_radius=6)
        shot_frame.pack(fill="x", padx=6, pady=4)

        ctk.CTkLabel(
            shot_frame, text="LAST SHOT", font=("", 10, "bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=8, pady=(6, 2))

        self._speed_label = ctk.CTkLabel(
            shot_frame, text="-- MPH",
            font=("", 22, "bold"), text_color="white",
        )
        self._speed_label.pack(anchor="w", padx=8)

        self._hla_label = ctk.CTkLabel(
            shot_frame, text="-- HLA",
            font=("", 16), text_color="gray70",
        )
        self._hla_label.pack(anchor="w", padx=8, pady=(0, 6))

        # Shot history
        history_frame = ctk.CTkFrame(parent, corner_radius=6)
        history_frame.pack(fill="both", expand=True, padx=6, pady=4)

        ctk.CTkLabel(
            history_frame, text="SHOT HISTORY", font=("", 10, "bold"),
            text_color="gray60",
        ).pack(anchor="w", padx=8, pady=(6, 2))

        self._history_text = ctk.CTkTextbox(
            history_frame, font=("Courier", 11), height=120,
            state="disabled", fg_color="gray14",
        )
        self._history_text.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # FPS
        fps_frame = ctk.CTkFrame(parent, corner_radius=6, height=36)
        fps_frame.pack(fill="x", padx=6, pady=(4, 6))
        fps_frame.pack_propagate(False)

        self._fps_label = ctk.CTkLabel(
            fps_frame, text="FPS: --",
            font=("", 12), text_color="gray60",
        )
        self._fps_label.pack(side="left", padx=8, pady=4)

        self._state_label = ctk.CTkLabel(
            fps_frame, text="idle",
            font=("", 11), text_color="gray50",
        )
        self._state_label.pack(side="right", padx=8, pady=4)

    def _build_control_bar(self, parent: ctk.CTkFrame) -> None:
        """Build the bottom control bar."""
        # Ball color dropdown
        preset_labels = list(PRESET_DESCRIPTIONS.values())
        self._label_to_preset = {v: k for k, v in PRESET_DESCRIPTIONS.items()}

        current_preset = self._config.ball.color_preset
        current_label = PRESET_DESCRIPTIONS.get(current_preset, "Yellow (bright)")

        self._color_var = ctk.StringVar(value=current_label)
        self._color_menu = ctk.CTkOptionMenu(
            parent,
            variable=self._color_var,
            values=preset_labels,
            command=self._on_color_selected,
            width=180,
        )
        self._color_menu.pack(side="left", padx=(0, 8))

        # Start/Stop button
        self._start_btn = ctk.CTkButton(
            parent, text="Start", command=self._toggle_running,
            width=100, fg_color="#2d8f2d", hover_color="#248f24",
        )
        self._start_btn.pack(side="left", padx=(0, 8))

        # Settings button
        ctk.CTkButton(
            parent, text="Settings", command=self._open_settings,
            width=90, fg_color="gray35", hover_color="gray30",
        ).pack(side="left")

        # Mode label (right side)
        mode_text = self._config.connection.mode.replace("_", " ").title()
        self._mode_label = ctk.CTkLabel(
            parent, text=mode_text,
            font=("", 11), text_color="gray50",
        )
        self._mode_label.pack(side="right", padx=8)

    # ---- Public update methods (called from processing thread via after()) ----

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
        hla_color = "#44cc44" if abs(hla) < 5 else ("#ffaa00" if abs(hla) < 15 else "#ff4444")
        self._hla_label.configure(text=hla_text, text_color=hla_color)

        # Add to history
        entry = ShotHistoryEntry(shot_number, speed, hla)
        self._shot_history.insert(0, entry)
        self._shot_history = self._shot_history[:50]  # Keep last 50

        # Update history text
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

    def start_video(self) -> None:
        """Begin video panel polling."""
        self._video_panel.start()

    def stop_video(self) -> None:
        """Stop video panel polling."""
        self._video_panel.stop()

    # ---- Internal callbacks ----

    def _toggle_running(self) -> None:
        """Toggle start/stop."""
        if self._is_running:
            self._is_running = False
            self._start_btn.configure(text="Start", fg_color="#2d8f2d", hover_color="#248f24")
            if self._on_stop:
                self._on_stop()
        else:
            self._is_running = True
            self._start_btn.configure(text="Stop", fg_color="#cc3333", hover_color="#aa2222")
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

    def _open_settings(self) -> None:
        """Open the settings dialog."""
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.focus()
            return

        self._settings_window = SettingsPanel(
            self,
            self._config,
            on_close=self._on_settings_closed,
        )
        self._settings_window.focus()

    def _on_settings_closed(self) -> None:
        """Handle settings dialog close."""
        self._settings_window = None
        # Update mode label
        mode_text = self._config.connection.mode.replace("_", " ").title()
        self._mode_label.configure(text=mode_text)
        if self._on_settings_changed:
            self._on_settings_changed()

    def _on_window_close(self) -> None:
        """Handle window close."""
        self._video_panel.stop()
        if self._on_stop:
            self._on_stop()
        self.destroy()
