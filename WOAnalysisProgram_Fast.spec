# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['WorkOrderAnalysisCur2_optimized.py'],
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
    [],
    exclude_binaries=True,
    name='WOAnalysisProgram_Fast',
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
    icon=['icons\\app_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WOAnalysisProgram_Fast',
)
