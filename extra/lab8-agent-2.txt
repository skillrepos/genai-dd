#!/usr/bin/env python3
"""
rag_agent.py — now ends with an LLM-generated, human-friendly summary

Pipeline
--------
1. Vector search (local Chroma)
2. Information extraction
3. Weather fetch via Open-Meteo API
4. Celsius-to-Fahrenheit conversion
5. Final summary using Llama3.2 via LangChain-Ollama

Dependencies
------------
    pip install sentence-transformers chromadb requests langchain langchain-community langchain-ollama
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

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ────────────────────────── console colors ──────────────────────────
class Color:
    GREEN = "\033[92m"
    BLUE  = "\033[94m"
    CYAN  = "\033[96m"
    YELL  = "\033[93m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

# ╔══════════════════════════════════════════════════════════════════╗
# 1.  Config / constants                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
CHROMA_PATH      = Path("./chroma_db")          # where the vector DB lives
COLLECTION_NAME  = "codebase"                   # collection inside Chroma
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"           # SBERT model for queries
TOP_K            = 5                            # RAG: retrieve top-5 chunks

# Regex patterns for lat/lon and various city formats
COORD_RE        = re.compile(r"\b(-?\d{1,2}(?:\.\d+)?)[,\s]+(-?\d{1,3}(?:\.\d+)?)\b")
CITY_STATE_RE   = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*),\s*([A-Z]{2})\b")
CITY_COUNTRY_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*),\s*([A-Z][a-z]{2,})\b")
CITY_RE         = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b")

STOPWORDS = {"office", "hq", "center", "centre"}

# Open-Meteo weather code mapping
WEATHER_CODE_MAP = {
    0:  "Clear sky",
    1:  "Mainly clear",
    2:  "Partly cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

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
# 4.  Weather & LLM tools                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
def get_weather(lat: float, lon: float, retries: int = 3) -> Optional[dict]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True
    }

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            weather = data["current_weather"]
            code = weather["weathercode"]
            return {
                "temperature": weather["temperature"],
                "conditions": WEATHER_CODE_MAP.get(code, f"Unknown ({code})")
            }
        except Exception as e:
            if attempt == retries - 1:
                print(Color.YELL + f"Failed to fetch weather: {e}" + Color.RESET)
            else:
                print(Color.CYAN + f"Retrying weather fetch... ({attempt + 1})" + Color.RESET)
    return None

def convert_c_to_f(c: float) -> float:
    return c * 9 / 5 + 32

async def summarize_with_llm(top_line: str, city: str, country: str, weather: dict) -> str:
    temp_f = convert_c_to_f(weather["temperature"])
    cond   = weather["conditions"]

    prompt = ChatPromptTemplate.from_template("""
You are an assistant helping generate friendly summaries about weather and office locations.

Given this info:
- Raw text: {top_line}
- City: {city}
- Country: {country}
- Conditions: {conditions}
- Temperature °F: {temp_f}

Write a **short summary** that:
1. Mentions the office location and city/country.
2. States current weather (conditions and temperature).
3. Includes ONE fun or historical fact about the city.
Use markdown formatting for headings or emphasis.
""")

    llm = Ollama(model="llama3.2")
    chain = prompt | llm | StrOutputParser()

    return await chain.ainvoke({
        "top_line": top_line,
        "city": city,
        "country": country,
        "conditions": cond,
        "temp_f": temp_f,
    })

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

    # Vector search
    print(Color.CYAN + "Searching office metadata..." + Color.RESET)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    coll        = open_collection()

    rag_hits = rag_search(prompt, embed_model, coll)
    top_hit  = rag_hits[0] if rag_hits else ""

    if top_hit:
        print(Color.GREEN + "\nTop RAG hit:\n" + Color.RESET + top_hit + "\n")
    else:
        print(Color.YELL + "No relevant chunks found in the collection." + Color.RESET)
        return

    # — step 1: direct coordinates? —
    coords = find_coords([top_hit, prompt])
    city_str = None

    # — step 2: if no coords, derive city then geocode —
    if not coords:
        city_str = (
            find_city_state([top_hit, prompt])
            or find_city_country([top_hit, prompt])
            or guess_city([top_hit, prompt])
        )
        if city_str:
            print(Color.CYAN + f"No coordinates found. Geocoding '{city_str}'..." + Color.RESET)
            coords = geocode(city_str)

    if not coords:
        print(Color.YELL + "Could not determine latitude/longitude." + Color.RESET)
        return

    lat, lon = coords
    print(Color.CYAN + f"Using coordinates: {lat:.4f}, {lon:.4f}" + Color.RESET)

    weather = get_weather(lat, lon)
    if not weather:
        print(Color.YELL + "Weather fetch failed." + Color.RESET)
        return

    print(Color.CYAN + "Generating final summary using Llama 3.2..." + Color.RESET)

    city_str = city_str or guess_city([top_hit]) or "Unknown"
    country_str = "Unknown"
    if "," in city_str:
        city_str, country_str = [s.strip() for s in city_str.split(",", 1)]

    summary = await summarize_with_llm(top_hit, city_str, country_str, weather)

    print(Color.BOLD + "\nFinal summary:\n" + Color.RESET + summary.strip() + "\n")

# ╔══════════════════════════════════════════════════════════════════╗
# 6.  Command-line REPL                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print(Color.BOLD + "Office-aware weather agent. Type 'exit' to quit.\n" + Color.RESET)
    while True:
        prompt = input(Color.BLUE + "Prompt: " + Color.RESET).strip()
        if prompt.lower() == "exit":
            break
        if prompt:
            asyncio.run(run(prompt))
