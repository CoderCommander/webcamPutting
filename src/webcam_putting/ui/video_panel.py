"""Video panel — displays OpenCV frames inside a CustomTkinter widget."""

from __future__ import annotations

import queue
from typing import Any

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk


class VideoPanel(ctk.CTkLabel):
    """Displays live video frames from a queue.

    Frames are BGR numpy arrays placed into a queue by the processing thread.
    This widget polls the queue at ~60fps using tkinter's after() mechanism.

    Usage:
        panel = VideoPanel(parent, frame_queue)
        panel.start()   # begin polling
        panel.stop()    # stop polling
    """

    POLL_INTERVAL_MS = 16  # ~60 fps

    def __init__(
        self,
        master: Any,
        frame_queue: queue.Queue,
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
