# Mevo Full-Swing Latency — Findings & Plan

## What was delivered (commit `21bc630`, testable now)
The Mevo stability gate (3 matching OCR reads before sending — the cure for the
L-wedge "across the green" misread) ran at the normal 0.5 s poll cadence, adding
~1.5 s of latency before a full swing reached GSPro.

**Fix:** the detector now exposes `is_confirming`, and the Mevo poll loop polls at
a fast `confirm_poll_interval` (0.12 s, configurable under `[mevo]`) **only while a
shot's display values are settling**. Full K=3 misread protection is kept, but the
gate's added latency drops from ~1.5 s to ~0.4 s. A new log line
`Mevo shot confirmed in X.XXs (N polls)` lets you measure it.

**To test tomorrow:** hit a full swing, watch the GSPro reaction feel snappier, and
check the log for `Mevo shot confirmed in ...` — should be ~0.4–0.6 s, not ~1.5 s.

**If you want it even faster:** drop `STABILITY_K` 3→2 in `mevo/detector.py`
(saves another poll, small misread-risk trade) and/or lower `confirm_poll_interval`.

## The latency floor (inherent to screen-reading)
FS Golf's display *animates* the numbers into place after a shot (~1–2 s). We have
to wait for them to settle before reading, or we're back to misreads. The
fast-confirm fix minimizes **birdman's** added latency; it cannot remove FS Golf's
own animation time. That floor only goes away with a non-screen data source.

## Direct FS Golf data path — investigated, NOT viable (encrypted)
Idea: read the shot from FS Golf directly instead of OCR'ing its animated screen.
FS Golf PC *does* write real-time data under `C:\Users\<you>\AppData\Local\FS Golf PC\`:
- `Sessions/.../ShotsDetails.json` — **index only** (GUID + per-shot `.fsd` filenames),
  no metrics.
- `Sessions/.../*.fsd` (result/realtime-prc/eprc/cprc) — **proprietary FlightScope
  format**, per shot.
- `FSGolfPCLive.db3` (+ `-wal`) — a live database, BUT its header is `E6 1B BF 8F...`,
  **not** `SQLite format 3` → the DB is **encrypted**. `sqlite3` reports
  "file is not a database".

Conclusion: every real-time source FlightScope writes is encrypted or proprietary.
Reading it directly would mean breaking encryption / reverse-engineering `.fsd` —
high effort, fragile, not worth it. **The OCR + fast-confirm approach is the path.**

## The cleaner long-term architecture (for a future session, not code tonight)
From research: **GSPro natively supports the Mevo+ directly** (low-latency,
plug-and-play). Today birdman bridges full shots (OCR of FS Golf → GSPro) because
birdman owns the single GSPro Open-Connect connection (for putts). Two future options:
1. **GSPro-Connect proxy:** birdman listens as if it were GSPro; FS Golf connects to
   birdman and birdman forwards structured JSON to the real GSPro (+ injects putts).
   No OCR, no animation wait, no misreads — but a real rearchitecture.
2. **Direct Mevo+ → GSPro for full shots**, webcam/birdman for putts only — depends on
   whether your GSPro setup can accept two sources. A setup question, not just code.

Both are bigger projects than tonight's latency trim; flagged for when you want to
pursue zero-latency full shots.

## Open items
- Task #8: processing-FPS optimization for the hardest putts (software, Kiyo Pro,
  no new hardware). Decouple the tracer-overlay render from detection so detection
  runs at the full ~54 fps the camera already delivers.
