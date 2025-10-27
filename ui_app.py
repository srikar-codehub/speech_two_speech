import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import audioop  # type: ignore[attr-defined]  # Python <3.13
except ModuleNotFoundError:
    import sys

    import audioop_lts as audioop  # type: ignore

    sys.modules["audioop"] = audioop

import gradio as gr

from silero_vadhelper import SileroVADHelper
from stt_azure import AzureSTT
from translate_azure import AzureTranslator
from tts_azure import AzureTTS


@dataclass(frozen=True)
class LanguageProfile:
    label: str
    stt_locale: str
    translator_code: str
    tts_language: str
    voices: List[str]


LANGUAGE_PROFILES: Dict[str, LanguageProfile] = {
    "en": LanguageProfile(
        label="English (US)",
        stt_locale="en-US",
        translator_code="en",
        tts_language="en-US",
        voices=[
            "en-US-AnaNeural",
            "en-US-GuyNeural",
            "en-US-JennyNeural",
        ],
    ),
    "fr": LanguageProfile(
        label="French (France)",
        stt_locale="fr-FR",
        translator_code="fr",
        tts_language="fr-FR",
        voices=[
            "fr-FR-AlainNeural",
            "fr-FR-DeniseNeural",
            "fr-FR-HenriNeural",
        ],
    ),
    "es": LanguageProfile(
        label="Spanish (Spain)",
        stt_locale="es-ES",
        translator_code="es",
        tts_language="es-ES",
        voices=[
            "es-ES-ElviraNeural",
            "es-ES-SergioNeural",
            "es-ES-AlvaroNeural",
        ],
    ),
    "de": LanguageProfile(
        label="German (Germany)",
        stt_locale="de-DE",
        translator_code="de",
        tts_language="de-DE",
        voices=[
            "de-DE-ConradNeural",
            "de-DE-KatjaNeural",
            "de-DE-KillianNeural",
        ],
    ),
    "hi": LanguageProfile(
        label="Hindi (India)",
        stt_locale="hi-IN",
        translator_code="hi",
        tts_language="hi-IN",
        voices=[
            "hi-IN-MadhurNeural",
            "hi-IN-SwaraNeural",
            "hi-IN-MohanNeural",
        ],
    ),
    "zh": LanguageProfile(
        label="Chinese (Mainland)",
        stt_locale="zh-CN",
        translator_code="zh-Hans",
        tts_language="zh-CN",
        voices=[
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunjianNeural",
            "zh-CN-YunxiNeural",
        ],
    ),
}

DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "fr"
DEFAULT_SILENCE_SECONDS = 3.0


class SpeechTranslationController:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._vad_stream = None

        self._status = "Stopped"
        self._last_transcription = ""
        self._last_translation = ""
        self._log_lines: List[str] = []
        self._max_log_lines = 200
        self._state_lock = threading.Lock()

    def start(
        self,
        source_lang: str,
        target_lang: str,
        voice_name: str,
        silence_seconds: float,
    ) -> str:
        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                return "Already running"

            source_profile = LANGUAGE_PROFILES[source_lang]
            target_profile = LANGUAGE_PROFILES[target_lang]
            silence_seconds = max(0.5, float(silence_seconds))

            self._reset_state()
            self._append_log(
                f"Initializing pipeline with STT {source_profile.stt_locale}, "
                f"translator {target_profile.translator_code}, "
                f"TTS voice {voice_name}, silence duration {silence_seconds:.1f}s"
            )
            self._set_status("Initializing...")

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_pipeline,
                args=(source_profile, target_profile, voice_name, silence_seconds),
                daemon=True,
            )
            self._thread.start()
            return "Initializing..."

    def stop(self) -> str:
        with self._thread_lock:
            if not self._thread or not self._thread.is_alive():
                self._set_status("Stopped")
                return "Already stopped"

            self._set_status("Stopping...")
            self._stop_event.set()

            if self._vad_stream:
                try:
                    self._vad_stream.close()
                except Exception:
                    pass
                self._vad_stream = None

            self._append_log("Stop requested by user.")

            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._append_log("Background thread is taking longer to stop...")
            else:
                self._append_log("Pipeline stopped.")

            self._set_status("Stopped")
            return "Stopped"

    def snapshot(self):
        with self._state_lock:
            log_text = "\n".join(self._log_lines[-self._max_log_lines :])
            return (
                log_text,
                self._last_transcription,
                self._last_translation,
                self._status,
            )

    def _run_pipeline(
        self,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        voice_name: str,
        silence_seconds: float,
    ) -> None:
        try:
            vad = SileroVADHelper(silence_duration=silence_seconds)
            stt = AzureSTT()
            stt.speech_config.speech_recognition_language = source_profile.stt_locale

            translator = AzureTranslator(target_lang=target_profile.translator_code)
            tts = AzureTTS(
                language=target_profile.tts_language,
                voice_name=voice_name,
            )

            self._append_log("Components initialized.")
            self._set_status("Listening...")

            self._vad_stream = vad.start()

            for speech_chunk in self._vad_stream:
                if self._stop_event.is_set():
                    break

                self._set_status("Transcribing...")
                text = stt.transcribe_chunk(speech_chunk, sample_rate=vad.sample_rate)
                if not text:
                    self._append_log("No transcription returned.")
                    self._set_status("Listening...")
                    continue

                self._record_transcription(text)
                self._append_log(f"Recognized text: {text}")

                self._set_status("Translating...")
                translated = translator.translate_text(text)
                if not translated:
                    self._append_log("Translation failed or empty.")
                    self._set_status("Listening...")
                    continue

                self._record_translation(translated)
                self._append_log(f"Translation: {translated}")

                if self._stop_event.is_set():
                    break

                self._set_status("Speaking...")
                tts.speak(translated)
                self._append_log("TTS playback completed.")
                self._set_status("Listening...")

        except Exception as exc:
            self._append_log(f"Pipeline error: {exc}")
            self._set_status("Error")
        finally:
            self._stop_event.clear()
            self._vad_stream = None
            self._set_status("Stopped")

    def _append_log(self, message: str) -> None:
        with self._state_lock:
            self._log_lines.append(message)

    def _record_transcription(self, text: str) -> None:
        with self._state_lock:
            self._last_transcription = text

    def _record_translation(self, text: str) -> None:
        with self._state_lock:
            self._last_translation = text

    def _set_status(self, status: str) -> None:
        with self._state_lock:
            self._status = status

    def _reset_state(self) -> None:
        with self._state_lock:
            self._log_lines.clear()
            self._last_transcription = ""
            self._last_translation = ""
            self._status = "Stopped"


controller = SpeechTranslationController()


def start_pipeline(
    source_lang: str,
    target_lang: str,
    voice_name: str,
    silence_seconds: float,
):
    return controller.start(
        source_lang,
        target_lang,
        voice_name,
        silence_seconds,
    )


def stop_pipeline():
    return controller.stop()


def refresh_outputs():
    logs, transcription, translation, status = controller.snapshot()
    return logs, transcription, translation, status


def update_voices(selected_target: str):
    profile = LANGUAGE_PROFILES[selected_target]
    default_voice = profile.voices[0] if profile.voices else None
    return gr.update(choices=profile.voices, value=default_voice, interactive=True)


def build_interface():
    default_voice = LANGUAGE_PROFILES[DEFAULT_TARGET].voices[0]

    with gr.Blocks(title="Speech Translation UI") as demo:
        gr.Markdown(
            "## Real-time Speech Translation\n"
            "Run the Silero VAD ➜ Azure STT ➜ Azure Translator ➜ Azure TTS loop."
        )

        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=list(LANGUAGE_PROFILES.keys()),
                value=DEFAULT_SOURCE,
                label="Source language",
                info=LANGUAGE_PROFILES[DEFAULT_SOURCE].label,
            )
            target_dropdown = gr.Dropdown(
                choices=list(LANGUAGE_PROFILES.keys()),
                value=DEFAULT_TARGET,
                label="Target language",
                info=LANGUAGE_PROFILES[DEFAULT_TARGET].label,
            )
            voice_dropdown = gr.Dropdown(
                choices=LANGUAGE_PROFILES[DEFAULT_TARGET].voices,
                value=default_voice,
                label="Azure neural voice",
                info="Voices update based on target language.",
            )

        silence_slider = gr.Slider(
            minimum=0.5,
            maximum=10.0,
            value=DEFAULT_SILENCE_SECONDS,
            step=0.5,
            label="Silence duration (seconds)",
            info="Silero waits this long after speech before finalizing a segment.",
        )

        with gr.Row():
            start_button = gr.Button("▶️ Start", variant="primary")
            stop_button = gr.Button("⏹️ Stop")

        status_box = gr.Textbox(
            label="Status",
            value="Stopped",
            lines=1,
            interactive=False,
        )
        transcription_box = gr.Textbox(
            label="Live transcription",
            lines=4,
            interactive=False,
            placeholder="Recognized text will appear here.",
        )
        translation_box = gr.Textbox(
            label="Live translation",
            lines=4,
            interactive=False,
            placeholder="Translated text will appear here.",
        )
        log_box = gr.Textbox(
            label="Logs",
            lines=12,
            interactive=False,
            placeholder="Pipeline logs will appear here.",
        )

        target_dropdown.change(
            fn=update_voices,
            inputs=target_dropdown,
            outputs=voice_dropdown,
        )

        start_button.click(
            fn=start_pipeline,
            inputs=[
                source_dropdown,
                target_dropdown,
                voice_dropdown,
                silence_slider,
            ],
            outputs=status_box,
        )
        stop_button.click(
            fn=stop_pipeline,
            outputs=status_box,
        )

        demo.load(
            fn=refresh_outputs,
            outputs=[log_box, transcription_box, translation_box, status_box],
            every=1.0,
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()
