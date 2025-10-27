"""
azure_metadata.py
-----------------
Fetches language and voice metadata from Azure Translator and Speech services.
Provides a cached view for populating UI controls and configuring the pipeline.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class VoiceOption:
    short_name: str
    locale: str
    gender: str
    name: str


@dataclass(frozen=True)
class LanguageOption:
    code: str
    name: str
    native_name: str
    locales: List[str] = field(default_factory=list)
    default_locale: str = ""
    voices: List[VoiceOption] = field(default_factory=list)

    def voice_names(self) -> List[str]:
        return [voice.short_name for voice in self.voices]


def get_azure_tts_voices(region: str = "eastus") -> Dict[str, List[Dict[str, str]]]:
    """
    Fetch the Azure Neural TTS voice catalog for a region and group by locale.

    Returns:
        dict: { locale: [{short_name, gender, name}, ...], ... }
    """
    load_dotenv()
    speech_key = os.getenv("AZURE_SPEECH_KEY")

    if not speech_key:
        raise ValueError("Missing AZURE_SPEECH_KEY in environment.")

    url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    headers = {"Ocp-Apim-Subscription-Key": speech_key}

    print(f"[AzureMetadata] Fetching TTS voices for region '{region}'...")
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    voices = response.json()

    voices_by_locale: Dict[str, List[Dict[str, str]]] = {}
    for voice in voices:
        locale = voice.get("Locale")
        short_name = voice.get("ShortName")
        if not locale or not short_name:
            continue

        voices_by_locale.setdefault(locale, []).append(
            {
                "short_name": short_name,
                "gender": voice.get("Gender") or "Unknown",
                "name": voice.get("DisplayName")
                or voice.get("LocalName")
                or short_name,
            }
        )

    print(
        f"[AzureMetadata] Retrieved {len(voices)} voices across "
        f"{len(voices_by_locale)} locales."
    )
    return voices_by_locale


class AzureMetadata:
    LANGUAGES_URL = "https://api.cognitive.microsofttranslator.com/languages"

    def __init__(self) -> None:
        load_dotenv()
        self._speech_key = os.getenv("AZURE_SPEECH_KEY")
        self._speech_region = os.getenv("AZURE_SPEECH_REGION")

        self._languages: Dict[str, LanguageOption] = {}
        self._voices_by_name: Dict[str, VoiceOption] = {}
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------
    def get_languages(self) -> List[LanguageOption]:
        self._ensure_loaded()
        return sorted(self._languages.values(), key=lambda item: item.name.lower())

    def get_language(self, code: str) -> Optional[LanguageOption]:
        self._ensure_loaded()
        return self._languages.get(code)

    def get_language_codes(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._languages.keys())

    def get_voice(self, short_name: Optional[str]) -> Optional[VoiceOption]:
        self._ensure_loaded()
        if not short_name:
            return None
        return self._voices_by_name.get(short_name)

    def get_default_language(self) -> Optional[LanguageOption]:
        self._ensure_loaded()
        if not self._languages:
            return None
        return next(iter(self.get_languages()), None)

    def get_default_voice_for_language(self, language_code: str) -> Optional[str]:
        option = self.get_language(language_code)
        if not option or not option.voices:
            return None
        return option.voices[0].short_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            try:
                self._load_from_api()
            except Exception as exc:  # pragma: no cover - defensive fallback
                # Fall back to static list so the UI remains usable.
                self._languages = self._fallback_languages()
                self._voices_by_name = {
                    voice.short_name: voice
                    for option in self._languages.values()
                    for voice in option.voices
                }
                self._append_debug_log(
                    f"Falling back to embedded metadata due to: {exc}"
                )

            self._loaded = True

    def _load_from_api(self) -> None:
        translation_data = self._fetch_translation_languages()
        voices_by_locale = get_azure_tts_voices(
            region=self._speech_region or "eastus"
        )

        languages: Dict[str, LanguageOption] = {}
        for code, details in translation_data.items():
            base_language = code.split("-")[0].lower()
            matching_locales = [
                locale
                for locale in voices_by_locale.keys()
                if locale.split("-")[0].lower() == base_language
            ]
            if not matching_locales:
                continue

            voice_options: List[VoiceOption] = []
            for locale in sorted(matching_locales):
                for voice in voices_by_locale[locale]:
                    voice_options.append(
                        VoiceOption(
                            short_name=voice["short_name"],
                            locale=locale,
                            gender=voice["gender"],
                            name=voice["name"],
                        )
                    )

            if not voice_options:
                continue

            locales = sorted({voice.locale for voice in voice_options})
            default_locale = locales[0] if locales else ""

            languages[code] = LanguageOption(
                code=code,
                name=details.get("name", code),
                native_name=details.get("nativeName", details.get("name", code)),
                locales=locales,
                default_locale=default_locale,
                voices=sorted(voice_options, key=lambda v: v.short_name.lower()),
            )

        if not languages:
            raise RuntimeError("No overlapping languages between translation and TTS.")

        self._languages = languages
        self._voices_by_name = {
            voice.short_name: voice
            for option in languages.values()
            for voice in option.voices
        }

    def _fetch_translation_languages(self) -> Dict[str, dict]:
        params = {"api-version": "3.0"}
        response = requests.get(self.LANGUAGES_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()

        return payload.get("translation", {})

    @staticmethod
    def _append_debug_log(message: str) -> None:
        # Placeholder for future logging hook. For now we simply print.
        print(f"[AzureMetadata] {message}")

    @staticmethod
    def _fallback_languages() -> Dict[str, LanguageOption]:
        fallback_definitions = [
            {
                "code": "en",
                "name": "English",
                "native_name": "English",
                "voices": [
                    ("en-US-JennyNeural", "en-US", "Female", "Jenny"),
                    ("en-US-GuyNeural", "en-US", "Male", "Guy"),
                ],
            },
            {
                "code": "fr",
                "name": "French",
                "native_name": "Francais",
                "voices": [
                    ("fr-FR-DeniseNeural", "fr-FR", "Female", "Denise"),
                    ("fr-FR-HenriNeural", "fr-FR", "Male", "Henri"),
                ],
            },
            {
                "code": "es",
                "name": "Spanish",
                "native_name": "Espanol",
                "voices": [
                    ("es-ES-ElviraNeural", "es-ES", "Female", "Elvira"),
                    ("es-ES-SergioNeural", "es-ES", "Male", "Sergio"),
                ],
            },
            {
                "code": "de",
                "name": "German",
                "native_name": "Deutsch",
                "voices": [
                    ("de-DE-KatjaNeural", "de-DE", "Female", "Katja"),
                    ("de-DE-ConradNeural", "de-DE", "Male", "Conrad"),
                ],
            },
            {
                "code": "hi",
                "name": "Hindi",
                "native_name": "Hindi",
                "voices": [
                    ("hi-IN-MadhurNeural", "hi-IN", "Male", "Madhur"),
                    ("hi-IN-SwaraNeural", "hi-IN", "Female", "Swara"),
                ],
            },
            {
                "code": "zh-Hans",
                "name": "Chinese Simplified",
                "native_name": "Zhongwen (Simplified)",
                "voices": [
                    ("zh-CN-XiaoxiaoNeural", "zh-CN", "Female", "Xiaoxiao"),
                    ("zh-CN-YunxiNeural", "zh-CN", "Male", "Yunxi"),
                ],
            },
        ]

        languages: Dict[str, LanguageOption] = {}
        for entry in fallback_definitions:
            voices = [
                VoiceOption(
                    short_name=short,
                    locale=locale,
                    gender=gender,
                    name=name,
                )
                for short, locale, gender, name in entry["voices"]
            ]
            locales = sorted({voice.locale for voice in voices})
            default_locale = locales[0] if locales else ""
            languages[entry["code"]] = LanguageOption(
                code=entry["code"],
                name=entry["name"],
                native_name=entry["native_name"],
                locales=locales,
                default_locale=default_locale,
                voices=voices,
            )
        return languages
