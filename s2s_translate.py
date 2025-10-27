# s2s_translate.py
from silero_vadhelper import SileroVADHelper
from stt_azure import AzureSTT
from translate_azure import AzureTranslator

def main():
    vad = SileroVADHelper()
    azure_stt = AzureSTT()
    translator = AzureTranslator(target_lang="es")  # Change target language here

    print("🚀 Starting speech → text → translation pipeline...")
    for chunk in vad.start():
        text = azure_stt.transcribe_chunk(chunk)
        if text:
            print(f"🗒️ Transcribed: {text}")
            translated = translator.translate_text(text)
            if translated:
                print(f"💬 Final Output: {translated}\n")

if __name__ == "__main__":
    main()
