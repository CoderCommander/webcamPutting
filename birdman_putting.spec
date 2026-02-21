# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for birdman-putting.

Build with:
    pip install pyinstaller
    pyinstaller birdman_putting.spec

Output: dist/birdman-putting/ (one-dir) or dist/birdman-putting.exe (one-file)
"""

import os
import sys
from pathlib import Path

import customtkinter

# Paths
SRC_DIR = os.path.join("src", "birdman_putting")
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
        "birdman_putting",
        "birdman_putting.app",
        "birdman_putting.camera",
        "birdman_putting.color_presets",
        "birdman_putting.config",
        "birdman_putting.detection",
        "birdman_putting.gspro_client",
        "birdman_putting.physics",
        "birdman_putting.tracking",
        "birdman_putting.ui",
        "birdman_putting.ui.main_window",
        "birdman_putting.ui.overlay",
        "birdman_putting.ui.settings_panel",
        "birdman_putting.ui.video_panel",
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
    name="birdman-putting",
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
    name="birdman-putting",
)
