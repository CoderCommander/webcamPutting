"""Video panel — displays OpenCV frames inside a CustomTkinter widget."""

from __future__ import annotations

import enum
import queue
from collections.abc import Callable
from typing import Any

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from birdman_putting.config import DetectionZone

_HANDLE_HALF = 6  # Half-size of handle hit area in display pixels


class _DragMode(enum.Enum):
    """Which part of the zone the user is dragging."""

    NONE = "none"
    MOVE = "move"
    TOP_LEFT = "tl"
    TOP_RIGHT = "tr"
    BOTTOM_LEFT = "bl"
    BOTTOM_RIGHT = "br"
    TOP = "t"
    BOTTOM = "b"
    LEFT = "l"
    RIGHT = "r"


class VideoPanel(ctk.CTkLabel):
    """Displays live video frames from a queue.

    Frames are BGR numpy arrays placed into a queue by the processing thread.
    This widget polls the queue at ~60fps using tkinter's after() mechanism.

    Supports an edit mode where the user can drag detection zone handles
    directly on the video.
    """

    POLL_INTERVAL_MS = 16  # ~60 fps

    def __init__(
        self,
        master: Any,
        frame_queue: queue.Queue,  # type: ignore[type-arg]
        width: int = 640,
        height: int = 360,
        **kwargs: Any,
    ):
        # Create a blank placeholder image
        placeholder = ctk.CTkImage(
            light_image=Image.new("RGB", (width, height), (30, 30, 30)),
            dark_image=Image.new("RGB", (width, height), (30, 30, 30)),
            size=(width, height),
        )
        super().__init__(master, image=placeholder, text="", **kwargs)

        self._frame_queue = frame_queue
        self._display_width = width
        self._display_height = height
        self._running = False
        self._after_id: str | None = None
        self._current_image: ImageTk.PhotoImage | None = None
        self._ctk_image: ctk.CTkImage | None = placeholder

        # Color pick mode state
        self._pick_mode = False
        self._current_frame: np.ndarray | None = None  # Latest BGR frame (640px wide)
        self._on_color_picked: Callable[[int, int, np.ndarray], None] | None = None

        # Edit mode state
        self._edit_mode = False
        self._zone: DetectionZone | None = None
        self._on_zone_changed: Callable[[], None] | None = None
        self._frame_height: int = height  # Updated from actual frames
        self._drag_mode = _DragMode.NONE
        self._drag_start: tuple[int, int] = (0, 0)
        self._drag_zone_snapshot: tuple[int, int, int, int] = (0, 0, 0, 0)

    # ---- Public API ----

    def start(self) -> None:
        """Begin polling the frame queue."""
        self._running = True
        self._poll()

    def stop(self) -> None:
        """Stop polling."""
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def set_edit_mode(
        self,
        enabled: bool,
        zone: DetectionZone | None = None,
        on_zone_changed: Callable[[], None] | None = None,
    ) -> None:
        """Enable or disable interactive zone editing via mouse drag."""
        self._edit_mode = enabled
        self._zone = zone
        self._on_zone_changed = on_zone_changed

        if enabled:
            self.bind("<Button-1>", self._on_mouse_down)
            self.bind("<B1-Motion>", self._on_mouse_drag)
            self.bind("<ButtonRelease-1>", self._on_mouse_up)
            self.bind("<Motion>", self._on_mouse_hover)
        else:
            self.unbind("<Button-1>")
            self.unbind("<B1-Motion>")
            self.unbind("<ButtonRelease-1>")
            self.unbind("<Motion>")
            self._drag_mode = _DragMode.NONE
            self.configure(cursor="arrow")

    def set_color_pick_mode(
        self,
        enabled: bool,
        on_color_picked: Callable[[int, int, np.ndarray], None] | None = None,
    ) -> None:
        """Enable or disable color pick mode.

        When enabled, clicking on the video samples that pixel's color.
        Mutually exclusive with edit mode — caller should exit edit mode first.
        """
        self._pick_mode = enabled
        self._on_color_picked = on_color_picked

        if enabled:
            self.bind("<Button-1>", self._on_pick_click)
            self.configure(cursor="crosshair")
        else:
            self.unbind("<Button-1>")
            self.configure(cursor="arrow")

    @property
    def pick_mode(self) -> bool:
        """Whether color pick mode is active."""
        return self._pick_mode

    def _on_pick_click(self, event: Any) -> None:
        """Handle click during color pick mode — sample color at click point."""
        if not self._pick_mode or self._current_frame is None:
            return

        # Display coords map 1:1 on X (both 640px), Y needs scaling
        frame_x = event.x
        frame_y = self._display_y_to_frame(event.y)

        # Clamp to frame bounds
        fh, fw = self._current_frame.shape[:2]
        frame_x = max(0, min(fw - 1, frame_x))
        frame_y = max(0, min(fh - 1, frame_y))

        if self._on_color_picked:
            self._on_color_picked(frame_x, frame_y, self._current_frame)

    @property
    def edit_mode(self) -> bool:
        """Whether zone editing is active."""
        return self._edit_mode

    # ---- Frame display ----

    def _poll(self) -> None:
        """Poll the queue for new frames and display them."""
        if not self._running:
            return

        try:
            # Drain queue to get latest frame (skip stale frames)
            frame = None
            while True:
                try:
                    frame = self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            if frame is not None:
                self._display_frame(frame)

        except Exception:
            pass  # Don't let display errors crash the UI

        self._after_id = self.after(self.POLL_INTERVAL_MS, self._poll)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Convert BGR frame to CTkImage and display."""
        # Track frame dimensions for coordinate mapping
        self._frame_height = frame.shape[0]

        # Cache the BGR frame for color picking
        self._current_frame = frame

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Resize to display dimensions
        pil_image = pil_image.resize(
            (self._display_width, self._display_height),
            Image.LANCZOS,  # type: ignore[attr-defined]
        )

        # Update the CTkImage
        self._ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(self._display_width, self._display_height),
        )
        self.configure(image=self._ctk_image)

    # ---- Coordinate mapping (display ↔ frame space) ----

    def _display_y_to_frame(self, dy: int) -> int:
        """Convert display-space Y to frame-space Y."""
        return int(dy * self._frame_height / self._display_height)

    def _frame_y_to_display(self, fy: int) -> int:
        """Convert frame-space Y to display-space Y."""
        return int(fy * self._display_height / self._frame_height)

    # X coordinates are 1:1 (both 640px wide)

    # ---- Hit testing ----

    def _hit_test(self, mx: int, my: int) -> _DragMode:
        """Determine what zone element the mouse is over."""
        if not self._zone:
            return _DragMode.NONE

        z = self._zone
        x1, x2 = z.start_x1, z.start_x2
        dy1 = self._frame_y_to_display(z.y1)
        dy2 = self._frame_y_to_display(z.y2)
        h = _HANDLE_HALF

        # Check corners first (they overlap edge midpoints)
        corners = [
            (_DragMode.TOP_LEFT, x1, dy1),
            (_DragMode.TOP_RIGHT, x2, dy1),
            (_DragMode.BOTTOM_LEFT, x1, dy2),
            (_DragMode.BOTTOM_RIGHT, x2, dy2),
        ]
        for mode, hx, hy in corners:
            if abs(mx - hx) <= h and abs(my - hy) <= h:
                return mode

        # Edge midpoints
        mid_x = (x1 + x2) // 2
        mid_y = (dy1 + dy2) // 2
        edges = [
            (_DragMode.TOP, mid_x, dy1),
            (_DragMode.BOTTOM, mid_x, dy2),
            (_DragMode.LEFT, x1, mid_y),
            (_DragMode.RIGHT, x2, mid_y),
        ]
        for mode, hx, hy in edges:
            if abs(mx - hx) <= h and abs(my - hy) <= h:
                return mode

        # Interior → move entire zone
        if x1 <= mx <= x2 and dy1 <= my <= dy2:
            return _DragMode.MOVE

        return _DragMode.NONE

    @staticmethod
    def _cursor_for(mode: _DragMode) -> str:
        """Return a cursor name for the given drag mode."""
        if mode == _DragMode.MOVE:
            return "fleur"
        if mode == _DragMode.NONE:
            return "arrow"
        return "crosshair"

    # ---- Mouse event handlers ----

    def _on_mouse_hover(self, event: Any) -> None:
        """Update cursor on hover (only when not dragging)."""
        if self._drag_mode != _DragMode.NONE:
            return
        mode = self._hit_test(event.x, event.y)
        self.configure(cursor=self._cursor_for(mode))

    def _on_mouse_down(self, event: Any) -> None:
        """Start a drag operation."""
        mode = self._hit_test(event.x, event.y)
        if mode == _DragMode.NONE:
            return
        self._drag_mode = mode
        self._drag_start = (event.x, event.y)
        if self._zone:
            z = self._zone
            self._drag_zone_snapshot = (z.start_x1, z.start_x2, z.y1, z.y2)

    def _on_mouse_drag(self, event: Any) -> None:
        """Update zone coordinates during drag."""
        if self._drag_mode == _DragMode.NONE or not self._zone:
            return

        dx = event.x - self._drag_start[0]
        dy_frame = (
            self._display_y_to_frame(event.y)
            - self._display_y_to_frame(self._drag_start[1])
        )

        z = self._zone
        ox1, ox2, oy1, oy2 = self._drag_zone_snapshot
        mode = self._drag_mode

        if mode == _DragMode.MOVE:
            z.start_x1 = ox1 + dx
            z.start_x2 = ox2 + dx
            z.y1 = oy1 + dy_frame
            z.y2 = oy2 + dy_frame
        else:
            if mode in (_DragMode.TOP_LEFT, _DragMode.BOTTOM_LEFT, _DragMode.LEFT):
                z.start_x1 = ox1 + dx
            if mode in (_DragMode.TOP_RIGHT, _DragMode.BOTTOM_RIGHT, _DragMode.RIGHT):
                z.start_x2 = ox2 + dx
            if mode in (_DragMode.TOP_LEFT, _DragMode.TOP_RIGHT, _DragMode.TOP):
                z.y1 = oy1 + dy_frame
            if mode in (_DragMode.BOTTOM_LEFT, _DragMode.BOTTOM_RIGHT, _DragMode.BOTTOM):
                z.y2 = oy2 + dy_frame

        # Clamp to frame bounds
        z.start_x1 = max(0, min(640, z.start_x1))
        z.start_x2 = max(0, min(640, z.start_x2))
        z.y1 = max(0, min(self._frame_height, z.y1))
        z.y2 = max(0, min(self._frame_height, z.y2))

        if self._on_zone_changed:
            self._on_zone_changed()

    def _on_mouse_up(self, event: Any) -> None:
        """End drag — normalize zone coordinates."""
        if self._drag_mode == _DragMode.NONE:
            return
        self._drag_mode = _DragMode.NONE

        # Ensure start_x1 <= start_x2 and y1 <= y2
        if self._zone:
            z = self._zone
            if z.start_x1 > z.start_x2:
                z.start_x1, z.start_x2 = z.start_x2, z.start_x1
            if z.y1 > z.y2:
                z.y1, z.y2 = z.y2, z.y1

        if self._on_zone_changed:
            self._on_zone_changed()
