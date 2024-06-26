# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path
import focusfeedbackgui


block_cipher = None
path = Path(focusfeedbackgui.__file__).parent


a = Analysis(
    [path / '__main__.py'],
    pathex=[],
    binaries=[],
    datas=[(path / 'stylesheet.qss', 'focusfeedbackgui'), (path / 'conf.yml', 'focusfeedbackgui')],
    hiddenimports=(collect_submodules('focusfeedbackgui') +
                   collect_submodules('focusfeedbackgui_rs') +
                   collect_submodules('xsdata_pydantic_basemodel.hooks') +
                   collect_submodules('ndbioimage.readers')),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='focusfeedbackgui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Icon.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='focusfeedbackgui',
)
