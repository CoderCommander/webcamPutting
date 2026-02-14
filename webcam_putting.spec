# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for webcam-putting.

Build with:
    pip install pyinstaller
    pyinstaller webcam_putting.spec

Output: dist/webcam-putting/ (one-dir) or dist/webcam-putting.exe (one-file)
"""

import os
import sys
from pathlib import Path

import customtkinter

# Paths
SRC_DIR = os.path.join("src", "webcam_putting")
ASSETS_DIR = "assets"
CTK_DIR = os.path.dirname(customtkinter.__file__)

block_cipher = None

a = Analysis(
    [os.path.join(SRC_DIR, "__main__.py")],
    pathex=[os.path.join("src")],
    binaries=[],
    datas=[
        # Default config
        (os.path.join(ASSETS_DIR, "default_config.toml"), "assets"),
        # CustomTkinter theme files (required at runtime)
        (os.path.join(CTK_DIR, "assets"), os.path.join("customtkinter", "assets")),
    ],
    hiddenimports=[
        "customtkinter",
        "PIL._tkinter_finder",
        "webcam_putting",
        "webcam_putting.app",
        "webcam_putting.camera",
        "webcam_putting.color_presets",
        "webcam_putting.config",
        "webcam_putting.detection",
        "webcam_putting.gspro_client",
        "webcam_putting.physics",
        "webcam_putting.tracking",
        "webcam_putting.ui",
        "webcam_putting.ui.main_window",
        "webcam_putting.ui.overlay",
        "webcam_putting.ui.settings_panel",
        "webcam_putting.ui.video_panel",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="webcam-putting",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for logging output
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="webcam-putting",
)
