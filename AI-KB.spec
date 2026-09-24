# -*- mode: python ; coding: utf-8 -*-
"""
AI 知识库 —— PyInstaller 打包配置
打包为单文件 EXE（onefile=True）
"""
from PyInstaller.utils.hooks import collect_submodules

datas = [
    ('frontend', 'frontend'),
]

binaries = []

# 项目实际需要的 hidden imports
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'langchain',
    'langchain.agents',
    'langchain_openai',
    'langchain_text_splitters',
    'langchain_core',
    'langgraph',
    'langgraph.graph',
    'pypdf',
    'docx',
    'docx2txt',
    'numpy',
] + collect_submodules('langchain') + collect_submodules('langgraph')

# 明确排除不需要的依赖（减少包体积）
excludes = [
    'torch',
    'torchvision',
    'torchaudio',
    'transformers',
    'sentence_transformers',
    'tensorflow',
    'matplotlib',
    'PIL',
    'pandas',
    'scipy',
    'sklearn',
    'chromadb',
    'langchain_chroma',
    'opentelemetry',
    'hnswlib',
    'duckdb',
    'onnxruntime',
    'tokenizers',
    'safetensors',
    'huggingface_hub',
    'accelerate',
    'bitsandbytes',
]


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook_disable_telemetry.py'],
    excludes=excludes,
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
    name='AI-KB',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
