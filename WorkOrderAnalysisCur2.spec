# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['WorkOrderAnalysisCur2.py'],
    pathex=[],
    binaries=[],
    datas=[('app_config.json', '.'), ('risk_presets.json', '.'), ('failure_mode_dictionary_.xlsx', '.'), ('failure_mode_dictionary_2.xlsx', '.'), ('sample_failure_dictionary.xlsx', '.'), ('icons', 'icons'), ('ai_classification_cache.json', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='WorkOrderAnalysisCur2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons\\app_icon.ico'],
)
