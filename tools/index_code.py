#!/usr/bin/env python3
"""
index_code.py
────────────────────────────────────────────────────────────────────
Create a new Chroma DB vector index and index local *.py files
inside the repository (or whichever directory `ROOT_DIR` points to).

Design goals
------------
1. **Single embedding model** — use Sentence-Transformers
   *all-MiniLM-L6-v2* for *both* documentation (PDFs) and code so all
   vectors live in the **same semantic space**.
2. **CPU-friendly** — MiniLM is <100 MB and runs quickly without a GPU
   or Ollama server.
3. **Line-aware chunking** — never split a line of code; try to break
   on blank lines; guarantee ≤ 500 GPT-3.5 tokens per chunk.

Output
------
• `./chroma_db/` — on-disk Chroma database (overwritten on each run)  
• Collection name `"codebase"`  
• One vector per code chunk, metadata keeps file path + chunk index
"""

from __future__ import annotations

# ─── standard library ─────────────────────────────────────────────
import os
import shutil
from pathlib import Path
from typing import Iterable, List

# ─── third-party ---------------------------------------------------
from sentence_transformers import SentenceTransformer          # local embeddings
from tiktoken import encoding_for_model                        # token counter
from chromadb import PersistentClient                          # Chroma client
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# ╔════════════════════════════════════════════════════════════════╗
# 1.  Configuration                                                ║
# ╚════════════════════════════════════════════════════════════════╝
ROOT_DIR         = Path(".")                    # directory tree to scan
CHROMA_PATH      = Path("./chroma_db")          # where vectors are stored
COLLECTION_NAME  = "codebase"                   # logical collection name
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"           # SBERT model
MAX_TOKENS       = 500                          # ≤500 GPT-3.5 tokens/chunk

# Folder names we *never* descend into
SKIP_DIRS = {
    ".git", ".hg", ".svn",                     # VCS metadata
    "__pycache__", "node_modules",             # build / npm garbage
    ".venv", "venv", "env", "py_env",          # virtual-envs
    "site-packages",                           # installed libs
}

# ╔════════════════════════════════════════════════════════════════╗
# 2.  Chunking helper (Python-code aware)                          ║
# ╚════════════════════════════════════════════════════════════════╝
def chunk_python_code(code: str, max_tokens: int = MAX_TOKENS) -> Iterable[str]:
    """
    Yield contiguous code blocks (≤ `max_tokens`) **without breaking lines.**

    Strategy
    --------
    1. Count GPT-3.5 tokens for *every* physical line (`tiktoken`).
    2. Accumulate lines until:
         • adding the next line would exceed `max_tokens`, OR
         • we hit a *blank* line and already have content (makes blocks
           roughly correspond to logical sections).
    3. Yield the current chunk, reset counters, and continue.

    Returns
    -------
    Iterable[str]
        Each yielded string is a code chunk ready for embedding.
    """
    enc = encoding_for_model("gpt-3.5-turbo")

    current_lines: List[str] = []
    token_count = 0

    for line in code.splitlines():
        line_tokens = len(enc.encode(line + "\n"))

        # Hard break: next line would overflow token budget
        if current_lines and token_count + line_tokens > max_tokens:
            yield "\n".join(current_lines)
            current_lines, token_count = [], 0

        # Soft break: blank line signals logical separation
        if not line.strip() and current_lines:
            yield "\n".join(current_lines)
            current_lines, token_count = [], 0
            continue                      # do *not* include the blank line

        current_lines.append(line)
        token_count += line_tokens

    if current_lines:                     # last chunk (file may not end with \n)
        yield "\n".join(current_lines)

# ╔════════════════════════════════════════════════════════════════╗
# 3.  Fresh-DB helper                                              ║
# ╚════════════════════════════════════════════════════════════════╝
def reset_chroma(db_path: Path) -> None:
    """
    Delete any existing Chroma folder so we *always* start from scratch.
    Avoids mixed embeddings if you tweak chunking rules or the model.
    """
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.mkdir(parents=True, exist_ok=True)

# ╔════════════════════════════════════════════════════════════════╗
# 4.  Main routine                                                 ║
# ╚════════════════════════════════════════════════════════════════╝
def index_python_sources() -> None:
    """
    Walk the directory tree under `ROOT_DIR`, embed every `.py` file,
    and store vectors + metadata in a fresh Chroma database.
    """
    if not ROOT_DIR.exists():
        print(f"[ERROR] {ROOT_DIR.resolve()} does not exist.")
        return

    # ── 1. Load embedding model (once) ────────────────────────────
    print(f"Embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # ── 2. Fresh on-disk DB ───────────────────────────────────────
    reset_chroma(CHROMA_PATH)

    # ── 3. Connect to persistent Chroma client ────────────────────
    client = PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(),                # default Chroma settings
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    collection = client.get_or_create_collection(COLLECTION_NAME)

    file_counter = 0

    # ── 4. Recursively scan .py files ─────────────────────────────
    for root, dirs, files in os.walk(ROOT_DIR):
        # In-place filter to stop os.walk() descending into skip folders
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for name in files:
            if not name.endswith(".py"):
                continue

            file_path = Path(root) / name

            # Read file
            try:
                code_text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as err:
                print(f"[WARN] Could not read {file_path}: {err}")
                continue

            # Chunk → embed → add to collection
            for idx, chunk in enumerate(chunk_python_code(code_text)):
                vector = embed_model.encode(chunk).tolist()

                collection.add(
                    ids        =[f"{file_path}-{idx}"],
                    embeddings =[vector],
                    documents  =[chunk],
                    metadatas  =[{"path": str(file_path), "chunk_index": idx}],
                )

            file_counter += 1
            print(f"Indexed {file_path}")

    # ── 5. Done ───────────────────────────────────────────────────
    print(
        f"Indexing complete: {file_counter} Python files processed.\n"
        "New vector DB saved to ./chroma_db"
    )

# ╔════════════════════════════════════════════════════════════════╗
# 5.  Entry point                                                  ║
# ╚════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    index_python_sources()
