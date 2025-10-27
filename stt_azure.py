# stt_azure.py
import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

class AzureSTT:
    def __init__(self):
        """
        Initializes Azure Speech SDK recognizer using environment variables.
        Requires:
            AZURE_SPEECH_KEY
            AZURE_SPEECH_REGION
        """
        load_dotenv()
        self.speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_REGION")

        if not self.speech_key or not self.speech_region:
            raise ValueError("❌ Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION in .env")

        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        self.speech_config.speech_recognition_language = "en-US"

    def transcribe_chunk(self, audio_chunk, sample_rate=16000):
        """
        Transcribes a given audio tensor or numpy array (float32, mono) using Azure Speech-to-Text.
        Returns:
            str: Transcribed text or None if failed.
        """
        # ensure numpy
        import numpy as np
        if isinstance(audio_chunk, type(None)) or len(audio_chunk) == 0:
            return None
        if hasattr(audio_chunk, "numpy"):
            audio_data = audio_chunk.numpy()
        else:
            audio_data = np.array(audio_chunk, dtype=np.float32)

        # Convert to bytes
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()

        # Configure push stream
        stream = speechsdk.audio.PushAudioInputStream()
        stream.write(audio_bytes)
        stream.close()

        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        print("🌀 Sending segment to Azure STT...")
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"✅ Recognized: {result.text}")
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("❌ No speech could be recognized.")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"⚠️ Canceled: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation.error_details}")
        return None
