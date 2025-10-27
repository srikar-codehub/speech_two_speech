# s2s_translate.py
from silero_vadhelper import SileroVADHelper
from stt_azure import AzureSTT

def main():
    vad = SileroVADHelper()
    azure_stt = AzureSTT()

    print("🚀 Starting speech-to-speech (S2S) transcription...")
    for chunk in vad.start():
        text = azure_stt.transcribe_chunk(chunk)
        if text:
            print(f"🗒️ Transcribed: {text}\n")

if __name__ == "__main__":
    main()
