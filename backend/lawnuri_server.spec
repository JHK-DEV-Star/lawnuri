# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for LawNuri backend server."""

import os
import sys
import importlib

block_cipher = None

# --- Paths ---
BACKEND_DIR = os.path.abspath('.')
APP_DIR = os.path.join(BACKEND_DIR, 'app')

# --- Hidden imports that PyInstaller cannot auto-detect ---
hiddenimports = [
    # FastAPI / Starlette
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'multipart',
    'multipart.multipart',
    # App modules
    'app',
    'app.main',
    'app.config',
    'app.api',
    'app.api.debate',
    'app.api.rag',
    'app.api.report',
    'app.api.settings',
    'app.graph',
    'app.graph.state',
    'app.graph.main_graph',
    'app.graph.team_subgraph',
    'app.graph.nodes',
    'app.graph.edges',
    'app.rag',
    'app.agents',
    'app.agents.debater',
    'app.agents.judge',
    'app.models',
    'app.utils',
    'app.utils.llm_client',
    'app.utils.embedding_client',
    'app.utils.file_parser',
    'app.utils.logger',
    'app.utils.retry',
    # LangGraph
    'langgraph',
    'langgraph.graph',
    'langgraph.graph.state',
    'langgraph.checkpoint',
    # ChromaDB
    'chromadb',
    'chromadb.config',
    # NetworkX
    'networkx',
    # OpenAI
    'openai',
    # Others
    'pydantic',
    'dotenv',
    'httpx',
    'cryptography',
    'charset_normalizer',
    'chardet',
    'fitz',  # PyMuPDF
]

# --- Collect data files ---
datas = [
    (os.path.join(BACKEND_DIR, 'app', 'db', 'schema.sql'), 'app/db'),
    (os.path.join(BACKEND_DIR, 'app', 'agents', 'legal_index', 'korea.txt'), 'app/agents/legal_index'),
]

# Collect chromadb migrations and default configs
try:
    import chromadb
    chromadb_dir = os.path.dirname(chromadb.__file__)
    datas.append((chromadb_dir, 'chromadb'))
except ImportError:
    pass

a = Analysis(
    ['run.py'],
    pathex=[BACKEND_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', '_tkinter', 'matplotlib', 'scipy', 'numpy.tests',
        'IPython', 'jupyter', 'notebook',
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
    name='lawnuri_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='server',
)
