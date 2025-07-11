#!/usr/bin/env python3
"""
rag_agent.py
────────────────────────────────────────────────────────────────────
A *Retrieval-Augmented Generation* (RAG) demo + tool-calling agent.

Pipeline
--------
1. **Vector search (local Chroma)**  
   Looks for office metadata inside a pre-built “codebase” collection.  
   A chunk might say:

       "Paris Office 88 Champs-Élysées, Paris, France …"

2. **Information extraction**  
   • Prefer explicit   latitude/longitude in the text.  
   • Else pull a city name (“City, ST”, “City, Country”, or fallback).  
   • If we have a city but no coords, use Open-Meteo’s geocoding API.

3. **Tool calls over FastMCP**  
   • `get_weather(lat, lon)` → °C + conditions.  
   • `convert_c_to_f(c)`     → °F.

4. **Final answer** printed to stdout.

Dependencies
------------
    pip install sentence-transformers chromadb fastmcp requests tiktoken
"""

# ────────────────────────── standard libs ───────────────────────────
import asyncio
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

# ────────────────────────── third-party libs ────────────────────────
import requests
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from sentence_transformers import SentenceTransformer
from fastmcp import Client
from fastmcp.exceptions import ToolError

# ╔══════════════════════════════════════════════════════════════════╗
# 1.  Config / constants                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
CHROMA_PATH      = Path("./chroma_db")          # where the vector DB lives
COLLECTION_NAME  = "codebase"                   # collection inside Chroma
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"           # SBERT model for queries
MCP_ENDPOINT     = "http://127.0.0.1:8000/mcp/" # FastMCP server
TOP_K            = 5                            # RAG: retrieve top-5 chunks

# Regex patterns for lat/lon and various city formats
COORD_RE        = re.compile(r"\b(-?\d{1,2}(?:\.\d+)?)[,\s]+(-?\d{1,3}(?:\.\d+)?)\b")
CITY_STATE_RE   = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*),\s*([A-Z]{2})\b")
CITY_COUNTRY_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*),\s*([A-Z][a-z]{2,})\b")
CITY_RE         = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b")

STOPWORDS = {"office", "hq", "center", "centre"}  # ignore tokens like “HQ”

# ╔══════════════════════════════════════════════════════════════════╗
# 2.  Vector search helpers                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
def open_collection() -> chromadb.Collection:
    """Return (or create) the Chroma collection."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    return client.get_or_create_collection(COLLECTION_NAME)


def rag_search(query: str,
               model: SentenceTransformer,
               coll: chromadb.Collection) -> List[str]:
    """
    Embed the *query*, search the vector DB, and return the text of the
    top-k chunks (empty list if collection is empty).
    """
    q_emb = model.encode(query).tolist()
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents"],
    )
    return res["documents"][0] if res["documents"] else []

# ╔══════════════════════════════════════════════════════════════════╗
# 3.  Information-extraction helpers                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
def find_coords(texts: List[str]) -> Optional[Tuple[float, float]]:
    """First valid lat/lon in the supplied texts, else None."""
    for txt in texts:
        for m in COORD_RE.finditer(txt):
            lat, lon = map(float, m.groups())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None


def find_city_state(texts: List[str]) -> Optional[str]:
    """Return first “City, ST” (US/CA style) found."""
    for txt in texts:
        if (m := CITY_STATE_RE.search(txt)):
            return m.group(0)
    return None


def find_city_country(texts: List[str]) -> Optional[str]:
    """Return first “City, Country” found."""
    for txt in texts:
        if (m := CITY_COUNTRY_RE.search(txt)):
            return m.group(0)
    return None


def guess_city(texts: List[str]) -> Optional[str]:
    """
    Fallback: first capitalised token not in STOPWORDS and >2 chars.
    """
    for txt in texts:
        for m in CITY_RE.finditer(txt):
            token = m.group(1).strip()
            if len(token) > 2 and token.split()[-1].lower() not in STOPWORDS:
                return token
    return None


def geocode(name: str) -> Optional[Tuple[float, float]]:
    """
    Ask Open-Meteo’s geocoding API for coords.
    If “City, XX” fails, retry with just “City”.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"

    def _lookup(n: str):
        try:
            r = requests.get(url, params={"name": n, "count": 1}, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("results"):
                hit = data["results"][0]
                return hit["latitude"], hit["longitude"]
        except Exception:
            pass
        return None

    coords = _lookup(name)
    if coords:
        return coords
    if "," in name:                              # retry with simpler string
        return _lookup(name.split(",", 1)[0].strip())
    return None

# ╔══════════════════════════════════════════════════════════════════╗
# 4.  unwrap(): CallToolResult → plain Python                        ║
# ╚══════════════════════════════════════════════════════════════════╝
def unwrap(obj):
    """
    FastMCP returns a CallToolResult wrapper.  This helper converts *any*
    shape used across FastMCP versions into plain dict / number / string.
    """
    if hasattr(obj, "structured_content") and obj.structured_content:
        return unwrap(obj.structured_content)
    if hasattr(obj, "data") and obj.data:
        return unwrap(obj.data)
    if isinstance(obj, list) and len(obj) == 1:
        return unwrap(obj[0])              # unwrap single-element list
    if isinstance(obj, dict):
        numeric_vals = [v for v in obj.values() if isinstance(v, (int, float))]
        if len(numeric_vals) == 1:         # {'value': 78.8}
            return numeric_vals[0]
    return obj

# ╔══════════════════════════════════════════════════════════════════╗
# 5.  Main workflow (async)                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
async def run(prompt: str) -> None:
    """
    0. User prompt ➜ vector search ➜ possible office chunk.
    1. Extract coordinates *or* city name.
    2. If only a city, geocode to lat/lon.
    3. Call MCP tools: get_weather → convert_c_to_f.
    4. Print final weather.
    """
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    coll        = open_collection()

    # Vector search
    rag_hits = rag_search(prompt, embed_model, coll)
    top_hit  = rag_hits[0] if rag_hits else ""
    if top_hit:
        print("\nTop RAG hit:\n", top_hit, "\n")

    # — step 1: direct coordinates? —
    coords = find_coords([top_hit, prompt])

    # — step 2: if no coords, derive city then geocode —
    if not coords:
        city_str = (
            find_city_state([top_hit, prompt])
            or find_city_country([top_hit, prompt])
            or guess_city([top_hit, prompt])
        )
        if city_str:
            print(f"No coords found; geocoding '{city_str}'.")
            coords = geocode(city_str)

    if not coords:
        print("Could not determine latitude/longitude.\n")
        return

    lat, lon = coords
    print(f"Using coordinates: {lat:.4f}, {lon:.4f}\n")

    # — step 3: call MCP tools —
    async with Client(MCP_ENDPOINT) as mcp:
        try:
            w_raw = await mcp.call_tool("get_weather", {"lat": lat, "lon": lon})
        except ToolError as e:
            print(f"Error calling get_weather: {e}")
            return

        weather = unwrap(w_raw)
        if not isinstance(weather, dict):
            print(f"Unexpected get_weather result: {weather}")
            return

        temp_c = weather.get("temperature")
        cond   = weather.get("conditions", "Unknown")

        try:
            tf_raw = await mcp.call_tool("convert_c_to_f", {"c": temp_c})
            temp_f = float(unwrap(tf_raw))
        except (ToolError, ValueError) as e:
            print(f"Temperature conversion failed: {e}")
            return

    # — step 4: print result —
    print(f"Weather: {cond}, {temp_f:.1f} °F\n")

# ╔══════════════════════════════════════════════════════════════════╗
# 6.  Command-line REPL                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("Office-aware weather agent. Type 'exit' to quit.\n")
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() == "exit":
            break
        if prompt:
            asyncio.run(run(prompt))
