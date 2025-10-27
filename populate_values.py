"""
populate_values.py
------------------
Fetches the latest Azure Translator languages and Azure Neural TTS voices,
then writes normalized JSON datasets used by the UI without touching runtime code.

Usage:
    python populate_values.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import requests
from dotenv import load_dotenv


LANGUAGES_URL = "https://api.cognitive.microsofttranslator.com/languages"
VOICES_URL_TEMPLATE = (
    "https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
)

LANGUAGES_FILE = Path(__file__).resolve().parent / "azure_languages.json"
VOICES_FILE = Path(__file__).resolve().parent / "azure_voices.json"


def fetch_languages() -> Dict[str, Dict[str, str]]:
    params = {"api-version": "3.0"}
    response = requests.get(LANGUAGES_URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    translation = data.get("translation", {})
    return {
        code: {
            "name": details.get("name", code),
            "nativeName": details.get("nativeName", details.get("name", code)),
        }
        for code, details in translation.items()
    }


def fetch_voices(region: str) -> Dict[str, List[Dict[str, str]]]:
    load_dotenv()
    speech_key = os.getenv("AZURE_SPEECH_KEY")

    if not speech_key:
        raise RuntimeError("AZURE_SPEECH_KEY must be set to fetch voices.")

    url = VOICES_URL_TEMPLATE.format(region=region)
    headers = {"Ocp-Apim-Subscription-Key": speech_key}

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    voices = response.json()

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for voice in voices:
        locale = voice.get("Locale")
        short_name = voice.get("ShortName")
        if not locale or not short_name:
            continue
        grouped.setdefault(locale, []).append(
            {
                "short_name": short_name,
                "gender": voice.get("Gender") or "Unknown",
                "name": voice.get("DisplayName")
                or voice.get("LocalName")
                or voice.get("FriendlyName")
                or short_name,
            }
        )

    for locale in grouped:
        grouped[locale].sort(key=lambda entry: entry["short_name"].lower())

    return grouped


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(region: str = "eastus") -> None:
    print(f"[populate] Fetching translator languages...")
    languages = fetch_languages()

    print(f"[populate] Fetching TTS voices for region '{region}'...")
    voices = fetch_voices(region)

    print(f"[populate] Writing {LANGUAGES_FILE.name} ({len(languages)} entries)")
    write_json(LANGUAGES_FILE, languages)

    print(f"[populate] Writing {VOICES_FILE.name} ({len(voices)} locales)")
    write_json(VOICES_FILE, voices)

    print("[populate] Completed successfully.")


if __name__ == "__main__":
    region_arg = sys.argv[1] if len(sys.argv) > 1 else os.getenv(
        "AZURE_SPEECH_REGION", "eastus"
    )
    main(region_arg)
