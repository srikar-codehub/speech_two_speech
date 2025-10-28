import os
from typing import Optional

import requests
from dotenv import load_dotenv


class AzureTranslator:
    def __init__(self, target_lang: str = "fr") -> None:
        """
        Initializes the Azure Translator client using environment variables.

        Expected .env values:
            AZURE_TRANSLATE_KEY
            AZURE_TRANSLATE_ENDPOINT
            AZURE_TRANSLATE_REGION (optional)
        """
        load_dotenv()
        self.endpoint = os.getenv("AZURE_TRANSLATE_ENDPOINT")
        self.key = os.getenv("AZURE_TRANSLATE_KEY")
        self.region = os.getenv("AZURE_TRANSLATE_REGION")  # optional

        if not self.endpoint or not self.key:
            raise ValueError("Missing AZURE_TRANSLATE_KEY or AZURE_TRANSLATE_ENDPOINT in .env")

        self.path = "/translate?api-version=3.0"
        self.target_lang = target_lang
        self._update_url()

    def _update_url(self) -> None:
        self.url = f"{self.endpoint}{self.path}&to={self.target_lang}"

    def set_target_language(self, target_lang: str) -> None:
        """
        Updates the target translation language without recreating the client.
        """
        if not target_lang or target_lang == self.target_lang:
            return
        self.target_lang = target_lang
        self._update_url()

    def translate_text(self, text: str) -> Optional[str]:
        """
        Translates the provided text. Returns the translated string or None on failure.
        """
        if not text or not text.strip():
            return None

        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }
        if self.region:
            headers["Ocp-Apim-Subscription-Region"] = self.region

        body = [{"text": text}]

        try:
            response = requests.post(self.url, headers=headers, json=body, timeout=10)
            response.raise_for_status()
            result = response.json()
            translated = result[0]["translations"][0]["text"]
            print(f"[AzureTranslator] Translated ({self.target_lang}): {translated}")
            return translated
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[AzureTranslator] Error: {exc}")
            return None
