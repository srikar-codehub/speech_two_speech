# translate_azure.py
import os
import requests
from dotenv import load_dotenv

class AzureTranslator:
    def __init__(self, target_lang="fr"):
        """
        Initializes the Azure Translator client.
        Requires in .env:
            AZURE_TRANSLATE_KEY
            AZURE_TRANSLATE_ENDPOINT
        Args:
            target_lang: Target translation language code (e.g. 'fr', 'es', 'de')
        """
        load_dotenv()
        self.endpoint = os.getenv("AZURE_TRANSLATE_ENDPOINT")
        self.key = os.getenv("AZURE_TRANSLATE_KEY")
        self.region = os.getenv("AZURE_TRANSLATE_REGION")  # optional

        if not self.endpoint or not self.key:
            raise ValueError("❌ Missing AZURE_TRANSLATE_KEY or AZURE_TRANSLATE_ENDPOINT in .env")

        self.target_lang = target_lang
        self.path = "/translate?api-version=3.0"
        self.url = f"{self.endpoint}{self.path}&to={self.target_lang}"

    def translate_text(self, text):
        """
        Translates the given text using Azure Translator.
        Returns:
            str: translated text or None
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
            print(f"🌍 Translated ({self.target_lang}): {translated}")
            return translated
        except Exception as e:
            print(f"⚠️ Azure Translate error: {e}")
            return None
