# Birdman — New Laptop Migration Runbook

This is a **migration runbook for moving Birdman to a new Windows laptop**. The same
physical hardware (Razer Kiyo Pro, Flightscope Mevo Gen 2, projector) is relocating with it.

> **If you are Claude on the new laptop:** follow this top to bottom. It assumes you have
> shell access and the repo is (or will be) cloned. For the *deep* explanation of any step,
> read [`docs/SETUP.md`](SETUP.md) — this file only covers what's specific to migrating an
> already-working rig. Do not change the camera capture pipeline (see repo `CLAUDE.md`).

---

## 0. What lives where (why this runbook exists)

| Thing | Location | Transfers via |
| --- | --- | --- |
| Code + full git history | GitHub `CoderCommander/webcamPutting` | `git clone` |
| Active branch (26 commits) | `fix/audit-remediation` (pushed) | `git checkout` |
| Experimental branch | `claude/laughing-jang` (pushed) | optional |
| **All tuning** (camera, zone, HSV, 18 Mevo ROIs, OBS scenes + password, GSPro-watcher ROI) | `config.toml` — **NOT in git** | OneDrive bundle |
| Uncommitted WIP (calibration-grid overlay hook) | `dreamy-napier-overlay-WIP.patch` | OneDrive bundle (optional) |
| Distance-calibration helper | `measure_ppf.py`, `ppf_calibration_frame.png` | OneDrive bundle |

**OneDrive transfer bundle:** `OneDrive\birdman-migration\` (sign into the same Microsoft
account on the new laptop and it syncs down automatically).

---

## 0.5 Two-machine topology (READ FIRST)

Birdman and GSPro are now **split across two machines** (they used to share one PC over
`127.0.0.1`):

| Machine | Hostname / IP | Runs |
| --- | --- | --- |
| **Primary desktop** | `GolfSim` — **`192.168.68.85`** (set as a static DHCP reservation in the router) | **GSPro** |
| **New laptop** | this machine | **Birdman + OBS + projector** |

This means the localhost wiring must become real network addresses:

1. **GSPro host** — in the restored `config.toml`, `[connection] gspro_host` must be
   **`192.168.68.85`** (not `127.0.0.1`), port `921`. See step 4.
2. **Firewall on GolfSim** — GolfSim must allow inbound TCP 921 from the LAN. Run this in an
   **Administrator PowerShell on GolfSim** (already attempted; rerun there if not present):
   ```powershell
   New-NetFirewallRule -DisplayName "GSPro Open API 921 (Birdman)" -Direction Inbound `
     -Protocol TCP -LocalPort 921 -Action Allow -RemoteAddress LocalSubnet -Profile Private
   ```
3. **Static IP** — `192.168.68.85` is reserved for GolfSim via the router, so Birdman can rely
   on it. If it ever changes, update `gspro_host` to match.

### OBS & the projector
OBS runs **on this laptop** alongside Birdman, so Birdman → OBS stays `localhost:4455`
(no change to `[obs] host`). The projector plugs into **this laptop**. Two cases:

- **Projector shows only Birdman's output** (putt tracer + Mevo/shot data + camera scenes):
  OBS here is fully self-contained — the *only* cross-machine link is Birdman → GSPro:921.
- **You want GSPro gameplay inside an OBS scene:** GSPro renders on GolfSim, so pipe its
  window over the LAN with **NDI** — install an NDI sender on GolfSim (OBS + the DistroAV
  / obs-ndi plugin, or the standalone **NDI Tools → Screen Capture**) and add it as an **NDI
  source** in this laptop's OBS. (Confirm with Greg which case applies before relying on it.)

### Remote-controlling this laptop from GolfSim
Use **RustDesk in direct-LAN mode** (installed in step 1) — it mirrors the real console
session, so Birdman's MSMF/D3D11 camera pipeline and the projector output keep running while
you view/control from GolfSim. **Do not use plain Windows RDP** here: it creates a virtual
display, disconnects the console, and can break the GPU camera capture and projector output.

---

## 1. Install prerequisites

| Software | Purpose | Notes |
| --- | --- | --- |
| **Claude Desktop** | run this setup | claude.ai/download |
| **Git** | clone repo | git-scm.com |
| **Python 3.10+** | run Birdman | This rig was validated on **Python 3.14.3** |
| **GSPro** | simulator | re-login with existing license |
| **FS Golf PC** | Mevo OCR source | Flightscope app — Windows only |
| **Tesseract OCR** | Mevo OCR engine | UB Mannheim build; install to default `C:\Program Files\Tesseract-OCR\` |
| **OBS Studio** (v28+) | scene switching / projector | WebSocket built in |
| **Razer Synapse** | Kiyo Pro firmware | **turn HDR OFF** (required for 60fps); log into Razer account to pull the saved profile |
| **RustDesk** | remote control from GolfSim | rustdesk.com — used in direct-LAN mode (see §0.5) |
| **NDI Tools / DistroAV** | *only if* OBS needs GSPro gameplay | see §0.5 — install the sender on GolfSim |

Verify Tesseract after install: `tesseract --version`

---

## 2. Clone the repo and select the active branch

```powershell
cd ~\Documents
git clone https://github.com/CoderCommander/webcamPutting.git
cd webcamPutting
git checkout fix/audit-remediation   # the active line of work (26 commits ahead of main)
```

> `fix/audit-remediation` is the branch Greg runs day-to-day; it is **not** merged to `main`.

---

## 3. Install Python dependencies

```powershell
pip install -e ".[dev,mevo,obs]"
```

This pulls the core app + Mevo OCR (`pytesseract`) + OBS (`obsws-python`) + dev tools.
Then sanity-check the import:

```powershell
python -c "import birdman_putting; print('ok')"
pytest tests/ -q
```

---

## 4. Restore the tuned config (the critical step)

The config is **not** in git. Copy it from the OneDrive bundle to the platformdirs location.
On Windows the app reads from `%LOCALAPPDATA%\birdman-putting\birdman-putting\config.toml`
(the folder name is doubled — that is correct, it is `appauthor\appname`).

```powershell
$dst = "$env:LOCALAPPDATA\birdman-putting\birdman-putting"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$env:OneDrive\birdman-migration\config.toml" "$dst\config.toml" -Force
```

This restores: detection zone, camera rotation/exposure/HSV (`orange3` preset),
`pixels_per_foot`, the 18 Mevo OCR ROIs, the GSPro-watcher club ROI, and the OBS
scenes + WebSocket password.

**Then make the split-machine edit (see §0.5):** GSPro is on GolfSim now, not localhost.

```powershell
$cfg = "$dst\config.toml"
(Get-Content $cfg) -replace 'gspro_host = "127\.0\.0\.1"', 'gspro_host = "192.168.68.85"' |
  Set-Content $cfg
Select-String -Path $cfg -Pattern 'gspro_host'   # verify it now reads 192.168.68.85
```

Leave `[obs] host = "localhost"` as-is — OBS runs on this laptop.

> **Secret note:** `config.toml` contains the OBS WebSocket password in plaintext. It is
> intentionally kept out of git. Only move it through OneDrive / USB, never commit it.

---

## 5. Windows display & power tuning

These prevent the Kiyo Pro dropping to 2fps when GSPro is foreground. Full steps are in
[`docs/SETUP.md`](SETUP.md#windows-display-settings). Summary — apply all, then **reboot**:

1. Power mode → **Best Performance**
2. Registry: `HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerThrottling\PowerThrottlingOff = 1`
3. USB selective suspend → **Disabled**
4. Exempt python from throttling:
   `powercfg /powerthrottling disable /path "<path-to>\python.exe"` (use `where.exe python`)
5. Taskbar → off on secondary displays (keeps it off the projector)

---

## 6. Recalibration checklist (do after config restore)

The same hardware is moving, so HSV/zone/rotation should mostly hold — **but several values
are pinned to this rig's geometry and must be re-verified.** Display resolution on the new
laptop is the key variable.

- [ ] **Camera index** — USB enumeration may differ. Launch and confirm the feed; if wrong,
      try `-w 0`, `-w 1`, `-w 2`. Update `webcam_index` in `[camera]`.
- [ ] **Camera feed sanity** — confirm 60fps and no black frames. If black frames: check
      Razer Synapse **HDR is OFF**. Do **not** switch the backend to DirectShow (see `CLAUDE.md`).
- [ ] **Detection zone / rotation** — relocating the mount can shift framing slightly. Use
      **Auto Zone**, then putt a few balls to confirm capture.
- [ ] **Distance (`pixels_per_foot`)** — re-verify with `measure_ppf.py` +
      `ppf_calibration_frame.png` from the bundle, or the in-app distance calibration. Mount
      height changes this.
- [ ] **Mevo OCR ROIs** — calibrated to a **1904×1041** FS Golf window. If the new screen
      resolution differs, the ROIs will miss. Re-run `python -m birdman_putting --calibrate-mevo`.
      (Make ROIs wide enough for R/L suffixes on signed metrics.)
- [ ] **GSPro-watcher club ROI** — calibrated to a **3840×2160 (4K)** display. If the new
      laptop is not 4K, re-capture the club-name region (`[gspro_watcher] club_roi`).
- [ ] **OBS** — recreate/import scenes named exactly `Main`, `Putt Data`, `Mevo Shot Data`,
      `Calibration`; enable WebSocket on port `4455` with the password from the restored config.

---

## 7. (Optional) re-apply uncommitted WIP

There was a small uncommitted change on the `claude/dreamy-napier` worktree (a calibration-grid
overlay hook in `overlay.py`). It's saved as a patch in the bundle. Only apply if you want to
continue that experiment:

```powershell
git checkout claude/dreamy-napier
git apply "$env:OneDrive\birdman-migration\dreamy-napier-overlay-WIP.patch"
```

Note: the hook calls `draw_calibration_grid(frame)` which may not be fully implemented — this
is unfinished WIP, not a working feature.

---

## 8. Launch & verify

```powershell
python -m birdman_putting -c orange3 -w 1 --mevo --obs
```

**Launch order (two machines):**
1. On **GolfSim**: start GSPro, confirm Open API shows "Ready" on port 921.
2. On **this laptop**: start OBS → FS Golf PC + Mevo → Birdman.
3. Confirm Birdman connects to GSPro across the LAN (log: `Connected to GSPro at 192.168.68.85:921`).

Verify end-to-end:
- [ ] Camera feed at ~60fps, ball detected
- [ ] Putt registers Speed (MPH) + HLA, GSPro shows the shot
- [ ] Select a non-putter club in GSPro → OBS switches scenes, Mevo OCR reads a full swing
- [ ] FPS stays ≥ 30 with GSPro in the foreground (proves the power tuning took)

---

## Quick reference

- Config path: `%LOCALAPPDATA%\birdman-putting\birdman-putting\config.toml`
- GSPro Open Connect: TCP `127.0.0.1:921` · OBS WebSocket: `localhost:4455`
- Active branch: `fix/audit-remediation` · Ball preset: `orange3`
- Deep setup docs: [`docs/SETUP.md`](SETUP.md) · Project rules: repo `CLAUDE.md`
