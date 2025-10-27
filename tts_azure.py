# tts_azure.py
import os

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv


class AzureTTS:
    def __init__(self, language: str = "fr-FR", voice_name: str = "fr-FR-DeniseNeural"):
        """
        Azure Text-to-Speech helper class.

        Args:
            language: Speech synthesis language code (default: 'fr-FR')
            voice_name: Azure voice name (default: 'fr-FR-DeniseNeural')

        Requires in .env:
            AZURE_SPEECH_KEY
            AZURE_SPEECH_REGION
        """
        load_dotenv()
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")

        if not self.speech_key or not self.speech_region:
            raise ValueError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION in .env")

        # Configure Azure Speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region,
        )

        # Set language and voice
        self.speech_config.speech_synthesis_language = language
        self.speech_config.speech_synthesis_voice_name = voice_name

        # Default output to system speakers
        self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=self.audio_config,
        )

        print(f"[AzureTTS] Initialized ({language}, {voice_name})")

    def speak(self, text: str) -> None:
        """
        Speaks the provided text aloud.
        """
        if not text or not text.strip():
            return

        print(f"[AzureTTS] Speaking: {text}")
        result = self.synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("[AzureTTS] Playback completed.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"[AzureTTS] Playback canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"[AzureTTS] Error details: {cancellation.error_details}")

    def stop(self) -> None:
        """
        Cancels any ongoing speech playback immediately.
        """
        try:
            self.synthesizer.stop_speaking_async().get()
            print("[AzureTTS] Playback stopped.")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[AzureTTS] Stop error: {exc}")

    def save_to_file(self, text: str, filename: str = "output.wav") -> None:
        """
        Synthesizes text to a WAV file instead of playing it.
        """
        if not text or not text.strip():
            return

        print(f"[AzureTTS] Saving synthesized speech to {filename}")
        file_config = speechsdk.audio.AudioOutputConfig(filename=filename)

        file_synth = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=file_config,
        )

        result = file_synth.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"[AzureTTS] Audio saved successfully: {filename}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"[AzureTTS] Save canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"[AzureTTS] Error details: {cancellation.error_details}")
