# Audit Remediation — Acceptance Spec

Branch: `fix/audit-remediation` (baseline checkpoint `b64fcd0`, `main` untouched).
Green baseline before work: **225 passed**.

Work is decomposed into file-disjoint groups so agents cannot collide. Each group: write
acceptance tests FIRST, implement to green, keep the full suite green. Commit checkpoint per group.

Wave 1 (parallel, disjoint): A Mevo · B GSPro · C Physics
Wave 2 (parallel, disjoint): D Detection+Tracking · E Camera
Wave 3 (after D+E): F App orchestration

---

## Group A — Mevo OCR robustness
**Files owned:** `src/birdman_putting/mevo/ocr.py`, `src/birdman_putting/mevo/detector.py`, `tests/test_mevo_*.py`
**Root cause (tonight's L-wedge across the green):** blind ÷10 decimal guess + single-frame OCR committed mid-animation + range checks too wide to reject (VLA 26.5°, spin 2013 on a 60° wedge were sent).

**Anticipated output**
1. Decimal parsing is position/mask-based, not ÷10. A no-dot read is mapped to the field's known
   decimal mask (e.g. VLA `NN.N`: `526`→`52.6`); a digit-count mismatch is REJECTED (returns None), not guessed.
2. A shot is emitted only after values are STABLE across K consecutive polls (K≥3); mid-animation
   frames never commit, and a transient misread does not latch `_prev_metrics`.
3. An internal-consistency / plausibility gate runs before send. Implausible or club-inconsistent
   shots (e.g. lofted-wedge VLA < ~35° or spin < ~3000 rpm) are DISCARDED with a WARNING (raw OCR
   strings logged), never forwarded to GSPro.
4. A non-zero signed field with no R/L suffix is a parse failure (held/discarded), not assumed positive.
5. A failed optional field is not coalesced to a fabricated `0.0` that gets sent as a real zero.

**Acceptance tests**
- `parse "526" mask NN.N → 52.6`; `parse "5260" → None` (mask mismatch, NOT 52.6 via ÷10).
- mid-animation sequence (frame1 garbage VLA → frames2-4 stable) emits exactly one shot = the stable values.
- VLA=26.5 & spin=2013 with club=LW → `_validate`/consistency returns reject; assert NOT sent.
- non-zero `curve`/`launch_direction` with missing R/L → parse failure path (not positive value).
- full suite green.

## Group B — GSPro client robustness
**Files owned:** `src/birdman_putting/gspro_client.py`, `tests/test_gspro_client.py`
**Root cause:** listener `recv()` on a socket another thread `close()`s → WinError 10038; `ConnectionReset`
logged with full traceback instead of clean reconnect; `}{`-split framing drops messages split across packets.

**Anticipated output**
1. Listener treats expected disconnects (`ConnectionResetError`, `ConnectionAbortedError`, `TimeoutError`,
   `OSError` winerror ∈ {10053,10054,10038,10058}) as INFO-level "dropped, reconnecting" + clear connected
   flag + continue. No traceback. `exc_info=True` reserved for genuinely unexpected types.
2. Socket teardown uses `shutdown()` before `close()` so a blocked reader unblocks cleanly; cross-thread
   close no longer produces 10038. A single "reconnect in progress" guard prevents concurrent reconnect drivers.
3. Streaming JSON framing: a persistent buffer accumulates `recv()` bytes and extracts complete objects via
   `raw_decode`/brace-scan; partial trailing bytes are retained; coalesced objects all parse. No `}{` string-split.
4. Dead heartbeat code (`_send_heartbeat_json`, `_build_heartbeat_message`, unused `_shot_cooldown`) removed;
   `_heartbeat_thread` renamed to `_listener_thread`.

**Acceptance tests**
- mock socket whose `recv` raises `ConnectionResetError` → listener loop exits cleanly, `is_connected()` False,
  no exception propagates, reconnect attempted; assert not logged at ERROR with traceback.
- one JSON object split across two `recv()` chunks → exactly one message parsed/dispatched.
- two concatenated objects in one `recv()` → two messages parsed.
- `send_shot`/`send_full_shot` JSON shape unchanged (existing tests pass).
- full suite green.

## Group C — Physics & calibration
**Files owned:** `src/birdman_putting/physics.py`, `src/birdman_putting/ppf_calibration.py`, `tests/test_physics.py`, `tests/test_ppf_calibration.py`
**Root cause:** `speed_from_trajectory_fit` has NO outlier rejection (one noise point skews speed → wildly-short
putts); `pixel_x_to_feet` derives feet from array index assuming every marker gap = 1 ft (missing marker → 2 ft short).

**Anticipated output**
1. `speed_from_trajectory_fit` does iterative residual-based outlier rejection (~2σ) and enforces monotonic
   non-decreasing forward travel; a single injected outlier no longer skews the fitted speed materially.
   Reject the fit (low-confidence/None) if too few points survive.
2. Calibration validates marker spacing: adjacent gaps compared to robust median; a gap deviating > ~35%
   flags non-uniform/missing markers. `calibration_markers` are stored/used so a missing interior marker
   cannot silently shift the foot scale (e.g. validate monotonic ~1-ft progression or store explicit pairs).
3. A "too few motion frames" path returns a low-confidence signal rather than a fabricated number.

**Acceptance tests**
- clean synthetic trajectory → fitted speed S. Inject one gross outlier point → fitted speed within ~5% of S
  (outlier rejected). (Document current-vs-new in the test.)
- markers `[0,57,114,228]` (missing 171 → 114→228 gap) → calibration flagged invalid/repaired; `pixel_x_to_feet(228)` ≈ 4 ft, not 3 ft.
- uniform markers → unchanged mapping (regression).
- < FIT_MIN_FRAMES motion points → low-confidence/None per contract.
- existing physics tests still pass (they encode current correct behavior — preserve it).
- full suite green.

## Group D — Detection & tracking
**Files owned:** `src/birdman_putting/detection.py`, `src/birdman_putting/tracking.py`, `tests/test_detection.py`, `tests/test_tracking.py`
**Root cause:** detection returns the LARGEST passing contour (hand blob wins over ball); circularity disabled
(0.0) during STARTED/ENTERED; watchdog measures time-since-state-change so a stationary ball loops `Re-start`
+ `stuck in started >10s`; forward noise blobs pollute the trajectory.

**Anticipated output**
1. When a prior position exists, detection picks the best-scoring contour (distance-to-last-position + radius
   match + circularity), not simply the largest. Radius tolerance tightened from the flat ±50 px.
2. Circularity gets a relaxed floor (e.g. ~0.35) during STARTED/ENTERED instead of 0.0.
3. Tracker suppresses redundant `Re-start` when the new start is within `start_position_tolerance` of the
   current start, and exposes a "last meaningful activity" timestamp the app watchdog can use.
4. During ENTERED, a forward detection implying > max-plausible putt velocity (px/frame via ppf+dt) is rejected.

**Acceptance tests**
- two contours: a ball-sized blob near the last position + a LARGER blob far away → detector returns the ball.
- stationary ball stabilizing repeatedly → at most one `Re-start` emission within tolerance (no spam).
- forward noise blob implying impossibly high velocity during ENTERED → rejected (not appended).
- existing tracking state-machine + detection tests still pass.
- full suite green.

## Group E — Camera
**Files owned:** `src/birdman_putting/camera.py`, `tests/test_camera*.py` (create if absent; use a fake/mock VideoCapture — no real device)
**Root cause:** `read()` returns `_latest_frame` by reference (torn-frame race) and conflates "no new frame" with
"failed"; one-shot UI captures need a freshness-agnostic primitive; grab loop busy-spins on read failure.

**Anticipated output**
1. `read()` returns a caller-owned COPY under the lock. Mutating the returned frame must not affect `_latest_frame`.
2. New `read_latest()` returns a copy of the most-recent frame regardless of `_frame_new`, or None only if no
   frame has ever been captured. (Consumed by Group F for one-shot calibration captures.)
3. Grab loop: on read failure, sleeps (no busy-spin) and counts consecutive failures, setting a failure/liveness
   flag after a threshold. Property application (`update_settings`) serializes with the grab thread.
4. `release()` idempotent.

**Acceptance tests**
- after a `read()` consumes the frame (`_frame_new` False), `read_latest()` still returns the frame.
- mutating the array returned by `read()`/`read_latest()` does not change `_latest_frame` (copy semantics).
- simulated repeated read failures → grab loop does not busy-spin (sleep invoked) and sets the failure flag.
- full suite green.

## Group F — App orchestration (depends on E `read_latest`, D activity timestamp)
**Files owned:** `src/birdman_putting/app.py` (and only app.py)
**Root cause:** per-frame loop has no try/except (one exception silently kills capture → "it froze"); `min_circularity`
save/restore not in `finally`; watchdog uses last-state-change (fires on stationary ball); OBS Auto Cal still does a
single unguarded `read()`; `calibration_markers` list read by physics can be mutated mid-shot by UI.

**Anticipated output**
1. The per-frame body is wrapped so one bad frame is logged (`logger.exception`) and skipped, not fatal; the
   processing loop has an outer guard that flips `_running=False` + surfaces an error on unexpected exit.
   `min_circularity` restored in a `finally`. (Refactor the per-frame body into a testable `_process_frame` if practical.)
2. Watchdog uses the tracker's last-activity timestamp (Group D) so a stationary ball no longer trips it.
3. OBS Auto Cal (and cork/angle cal) obtain frames via `read_latest()` (Group E); ad-hoc retry loops removed.
4. Config values used by the shot (esp. `calibration_markers`, ppf, stimp) are snapshotted at shot start
   (immutable copy) so UI edits can't corrupt an in-flight fit.

**Acceptance tests**
- injected fault in the per-frame path is caught and the loop continues (test the extracted `_process_frame`).
- OBS Auto Cal obtains a frame via `read_latest` (no "no frame" when grab thread healthy) — unit/mocked.
- `import birdman_putting.app` OK; headless smoke (`--no-gui` on a short synthetic/video source) runs without crashing.
- full suite green.

---

## Global definition of done
- All groups merged on `fix/audit-remediation`, full suite green (≥ 225, plus new tests).
- `ruff check src tests` clean (or no new violations); `python -c "import birdman_putting.app"` OK.
- Headless smoke run completes without exception.
- Morning summary written: per-group commit list, what changed, residual risks, anything needing human judgment.
