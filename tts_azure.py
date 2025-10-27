# tts_azure.py
import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

class AzureTTS:
    def __init__(self, language="fr-FR", voice_name="fr-FR-DeniseNeural"):
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
            raise ValueError("❌ Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION in .env")

        # Configure Azure Speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )

        # Set language and voice
        self.speech_config.speech_synthesis_language = language
        self.speech_config.speech_synthesis_voice_name = voice_name

        # Default output to system speakers
        self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=self.audio_config
        )

        print(f"🔹 Azure TTS initialized ({language}, {voice_name})")

    def speak(self, text):
        """
        Speaks the provided text aloud.
        """
        if not text or not text.strip():
            return

        print(f"🔊 Speaking: {text}")
        result = self.synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("✅ TTS playback completed.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"⚠️ TTS canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation.error_details}")

    def save_to_file(self, text, filename="output.wav"):
        """
        Synthesizes text to a WAV file instead of playing it.
        """
        if not text or not text.strip():
            return

        print(f"💾 Saving synthesized speech to {filename}")
        file_config = speechsdk.audio.AudioOutputConfig(filename=filename)

        file_synth = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=file_config
        )

        result = file_synth.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"✅ Audio saved successfully: {filename}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"⚠️ TTS save canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation.error_details}")
