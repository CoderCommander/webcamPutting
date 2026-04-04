"""Quick camera diagnostic — tests all backends and indices."""
import cv2
import time
import sys

print(f"OpenCV version: {cv2.__version__}")
print(f"Build info backends: {cv2.videoio_registry.getBackendName(cv2.CAP_DSHOW)}, "
      f"{cv2.videoio_registry.getBackendName(cv2.CAP_MSMF)}")
print()

backends = [
    ("default", None),
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
]

for idx in [0, 1]:
    for name, backend in backends:
        print(f"Camera {idx} via {name}...", end=" ", flush=True)
        t0 = time.time()
        if backend is None:
            cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx, backend)
        elapsed = time.time() - t0
        opened = cap.isOpened()
        if opened:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            print(f"OPEN {w}x{h}@{fps:.0f}fps read={'OK' if ret else 'FAIL'} ({elapsed:.1f}s)")
        else:
            print(f"FAILED ({elapsed:.1f}s)")
        cap.release()
        time.sleep(0.5)
    print()
