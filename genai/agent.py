#!/usr/bin/env python3
"""
────────────────────────────────────────────────────────────────────────
A **tiny demonstration** of the classic Thought-Action-Observation (TAO)
loop using:

* **Open-Meteo** – free REST weather API
* **LangChain + Ollama** – local Llama-3.2 language model
* **Two “tools”** defined in pure Python 

The agent:

  1. Receives a natural-language question such as  
     “What is the predicted weather today for Paris?”
  2. Thinks (“Thought: …”) and selects a tool (“Action: get_weather …”).
  3. We (the host program) call the Python function, show the *Observation*,
     and feed that back to the LLM.
  4. The LLM decides to convert °C → °F, we run that tool, feed the result
     back, and finally print a concise forecast.

Run the script, type a city, watch the full trace, and get the answer.
"""

# ───────────────────────── standard library ─────────────────────────
import json
import textwrap

from typing import Dict

import requests                           # simple HTTP client

# ───────────────────────── 3rd-party libraries ──────────────────────
# *langchain-ollama* publishes a drop-in wrapper so we can call the
# local Ollama server like any other LangChain LLM.
from langchain_ollama import ChatOllama

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 1.  Static lookup: WMO weather-code → friendly description       ║
# ╚══════════════════════════════════════════════════════════════════╝
WEATHER_CODES: Dict[int, str] = {
    0:  "Clear sky",                     1:  "Mainly clear",
    2:  "Partly cloudy",                 3:  "Overcast",
    45: "Fog",                           48: "Depositing rime fog",
    51: "Light drizzle",                 53: "Moderate drizzle",
    55: "Dense drizzle",                 56: "Light freezing drizzle",
    57: "Dense freezing drizzle",        61: "Slight rain",
    63: "Moderate rain",                 65: "Heavy rain",
    66: "Light freezing rain",           67: "Heavy freezing rain",
    71: "Slight snow fall",              73: "Moderate snow fall",
    75: "Heavy snow fall",               77: "Snow grains",
    80: "Slight rain showers",           81: "Moderate rain showers",
    82: "Violent rain showers",          85: "Slight snow showers",
    86: "Heavy snow showers",            95: "Thunderstorm",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 2.  “Tool” functions (simple Python, no server needed)           ║
# ╚══════════════════════════════════════════════════════════════════╝
def get_weather(lat: float, lon: float) -> dict:
    """
    Query the **daily** endpoint of Open-Meteo and return *today’s*
    max / min temperature and the weather-code description.

    Returned dict always has the same keys so the LLM can rely on them.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&forecast_days=1&timezone=auto"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()                       # raise for any 4xx / 5xx
    daily = r.json()["daily"]

    return {
        "high":       daily["temperature_2m_max"][0],
        "low":        daily["temperature_2m_min"][0],
        "conditions": WEATHER_CODES.get(daily["weathercode"][0], "Unknown"),
    }


def convert_c_to_f(c: float) -> float:
    """Classic °C → °F conversion."""
    return c * 9 / 5 + 32

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 3.  Local LLM wrapper (LangChain + Ollama)                       ║
# ╚══════════════════════════════════════════════════════════════════╝
# The model is fetched from your local Ollama instance; set temperature
# to 0.0 for deterministic planning.
llm = ChatOllama(model="llama3.2", temperature=0.0)

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 4.  “System” prompt that defines the tools and the TAO protocol  ║
# ╚══════════════════════════════════════════════════════════════════╝
SYSTEM = textwrap.dedent("""
You are an agent with two tools:

get_weather(lat:float, lon:float)
    → {"high": float, "low": float, "conditions": str}

convert_c_to_f(c:float) → float

When you plan, emit exactly:

Thought: <your thought>
Action: <tool name>
Args: {"lat":X,"lon":Y}   or   {"c":Z}

Do NOT output anything else.
""").strip()

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 5.  Helper that runs a single TAO episode and prints the trace   ║
# ╚══════════════════════════════════════════════════════════════════╝
def run(question: str) -> str:
    """
    Execute *one* two-step TAO loop:

        1. Ask Llama-3 to choose coordinates and call get_weather().
        2. Feed back the observation, ask it whether to convert; if so,
           call convert_c_to_f() for both temps.
        3. Print the entire trace (Thought / Action / Observation) and
           return the final sentence so the caller could display it.
    """
    # Initial conversation history
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": question},
    ]

    print("\n--- Thought → Action → Observation → Final ---\n")

    # ── First planning step: choose coordinates ────────────────────
    reply1 = llm.invoke(messages)
    plan1  = reply1.content.strip()
    print(plan1 + "\n")

    # Extract JSON args from the “Args: …” line
    coords = json.loads(plan1.split("Args:")[1].strip())

    # Call the first tool and show observation
    obs1 = get_weather(**coords)
    print(f"Observation: {obs1}\n")

    # ── Second planning step: decide whether to convert units ──────
    messages += [
        {"role": "assistant", "content": plan1},          # what the LLM “said”
        {"role": "user",      "content": f"Observation: {obs1}"},
    ]
    reply2 = llm.invoke(messages)
    plan2  = reply2.content.strip()
    print(plan2 + "\n")

    # In this toy protocol the second tool call is always °C→°F
    high_f = convert_c_to_f(obs1["high"])
    low_f  = convert_c_to_f(obs1["low"])
    obs2   = {"high_f": high_f, "low_f": low_f}
    print(f"Observation: {obs2}\n")

    # ── Compose the final human-readable answer ────────────────────
    final = (
        f"Today will be **{obs1['conditions']}** with a high of "
        f"**{high_f:.1f} °F** and a low of **{low_f:.1f} °F**."
    )
    print(f"Final: {final}\n")
    return final

# ╔══════════════════════════════════════════════════════════════════╗
# ║ 6.  Simple REPL so you can type locations interactively          ║
# ╚══════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("Weather-forecast agent (type 'exit' to quit)\n")
    while True:
        loc = input("Location (or 'exit'): ").strip()
        if loc.lower() == "exit":
            print("Goodbye!")
            break

        # Build a user prompt that asks the agent for a forecast in °F
        query = (
            f"What is the predicted weather today for {loc}? "
            "Include conditions plus high/low in °F."
        )

        try:
            run(query)              # full TAO trace is printed inside run()
        except Exception as e:
            print(f"⚠️  Error: {e}\n")
