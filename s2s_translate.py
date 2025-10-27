# s2s_translate.py
from silero_vadhelper import SileroVADHelper
from stt_azure import AzureSTT
from translate_azure import AzureTranslator
from tts_azure import AzureTTS

def main():
    """
    Full Speech → Text → Translate → Speech pipeline.
    Detects speech in real time using Silero VAD,
    sends speech to Azure STT for transcription,
    translates it using Azure Translator,
    and finally speaks it using Azure TTS.
    """
    # Initialize components
    vad = SileroVADHelper()
    azure_stt = AzureSTT()
    translator = AzureTranslator(target_lang="fr")  # Target translation: French
    tts = AzureTTS(language="fr-FR", voice_name="fr-FR-DeniseNeural")  # French voice

    print("\n🚀 Starting Speech → Text → Translate → Speech pipeline...\n")
    print("🎙️ Speak now! (3s silence threshold)\n")

    # Stream from microphone and process
    for chunk in vad.start():
        text = azure_stt.transcribe_chunk(chunk)

        if text:
            print(f"🗒️ Transcribed: {text}")

            translated = translator.translate_text(text)
            if translated:
                print(f"💬 Final Output: {translated}")
                tts.speak(translated)
                print("🎧 Playback done.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Exiting gracefully.")
