import threading
from typing import List, Optional

try:
    import audioop  # type: ignore[attr-defined]  # Python <3.13
except ModuleNotFoundError:
    import sys

    import audioop_lts as audioop  # type: ignore

    sys.modules["audioop"] = audioop

import gradio as gr

from azure_metadata import AzureMetadata, LanguageOption, VoiceOption
from silero_vadhelper import SileroVADHelper
from stt_azure import AzureSTT
from translate_azure import AzureTranslator
from tts_azure import AzureTTS


DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "fr"
DEFAULT_SILENCE_SECONDS = 3.0


metadata_provider = AzureMetadata()


class SpeechTranslationController:
    def __init__(self, metadata: AzureMetadata):
        self._metadata = metadata
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

            source_option = self._metadata.get_language(source_lang)
            target_option = self._metadata.get_language(target_lang)
            if not source_option or not target_option:
                self._append_log("Invalid language selection.")
                self._set_status("Error")
                return "Invalid language selection"

            voice_option = self._metadata.get_voice(voice_name)
            if not voice_option or voice_option.short_name not in {
                v.short_name for v in target_option.voices
            }:
                default_voice_name = self._metadata.get_default_voice_for_language(
                    target_lang
                )
                voice_option = (
                    self._metadata.get_voice(default_voice_name)
                    if default_voice_name
                    else None
                )
                voice_name = voice_option.short_name if voice_option else voice_name

            if not voice_option:
                self._append_log("No Azure voice available for the selected language.")
                self._set_status("Error")
                return "Voice unavailable"

            silence_seconds = max(0.5, float(silence_seconds))

            self._reset_state()
            self._append_log(
                "Initializing pipeline with "
                f"STT locale {source_option.default_locale}, "
                f"translator code {target_option.code}, "
                f"TTS voice {voice_option.short_name}, "
                f"silence duration {silence_seconds:.1f}s"
            )
            self._set_status("Initializing...")

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_pipeline,
                args=(
                    source_option,
                    target_option,
                    voice_option,
                    silence_seconds,
                ),
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
        source_option: LanguageOption,
        target_option: LanguageOption,
        voice_option: VoiceOption,
        silence_seconds: float,
    ) -> None:
        try:
            vad = SileroVADHelper(silence_duration=silence_seconds)
            stt = AzureSTT()
            stt_language = source_option.default_locale or source_option.locales[0]
            stt.speech_config.speech_recognition_language = stt_language

            translator = AzureTranslator(target_lang=target_option.code)
            tts = AzureTTS(
                language=voice_option.locale,
                voice_name=voice_option.short_name,
            )

            self._append_log(
                "Components initialized "
                f"(STT {stt_language}, translation {target_option.code}, "
                f"TTS {voice_option.short_name})."
            )
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


controller = SpeechTranslationController(metadata_provider)


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
    option = metadata_provider.get_language(selected_target)
    if not option or not option.voices:
        return gr.update(choices=[], value=None, interactive=False)

    default_voice = option.voices[0].short_name
    choices = [voice.short_name for voice in option.voices]
    return gr.update(choices=choices, value=default_voice, interactive=True)


def describe_language(code: str) -> str:
    option = metadata_provider.get_language(code)
    if not option:
        return f"**{code}** — unavailable in metadata."

    display_native = option.native_name or option.name
    if display_native and display_native.lower() == option.name.lower():
        display_name = option.name
    else:
        display_name = f"{option.name} / {display_native}"

    locales = ", ".join(option.locales) if option.locales else "n/a"
    voice_count = len(option.voices)
    return (
        f"**{code}** — {display_name}<br/>"
        f"STT locales: {locales}<br/>"
        f"Voices available: {voice_count}"
    )


def describe_voice(short_name: Optional[str]) -> str:
    if not short_name:
        return "Voice: unavailable."
    voice = metadata_provider.get_voice(short_name)
    if not voice:
        return f"Voice **{short_name}** — metadata unavailable."
    return (
        f"**{voice.short_name}** — {voice.name} "
        f"({voice.gender}, locale {voice.locale})"
    )


def describe_default_voice(language_code: str) -> str:
    default_voice = metadata_provider.get_default_voice_for_language(language_code)
    return describe_voice(default_voice)


def build_interface():
    language_options = metadata_provider.get_languages()
    if not language_options:
        raise RuntimeError("Azure metadata did not return any languages.")

    language_codes = [option.code for option in language_options]
    language_labels = {
        option.code: f"{option.name} ({option.native_name})"
        if option.native_name and option.native_name.lower() != option.name.lower()
        else option.name
        for option in language_options
    }

    default_source_code = (
        DEFAULT_SOURCE
        if metadata_provider.get_language(DEFAULT_SOURCE)
        else language_codes[0]
    )
    default_target_code = (
        DEFAULT_TARGET
        if metadata_provider.get_language(DEFAULT_TARGET)
        else language_codes[0]
    )

    target_option = metadata_provider.get_language(default_target_code)
    voice_choices = (
        [voice.short_name for voice in target_option.voices] if target_option else []
    )
    default_voice = voice_choices[0] if voice_choices else None

    with gr.Blocks(title="Speech Translation UI") as demo:
        gr.Markdown(
            "## Real-time Speech Translation\n"
            "Run the Silero VAD -> Azure STT -> Azure Translator -> Azure TTS loop."
        )

        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=language_codes,
                value=default_source_code,
                label="Source language (translation code)",
                info=language_labels.get(default_source_code, ""),
            )
            target_dropdown = gr.Dropdown(
                choices=language_codes,
                value=default_target_code,
                label="Target language (translation code)",
                info=language_labels.get(default_target_code, ""),
            )
            voice_dropdown = gr.Dropdown(
                choices=voice_choices,
                value=default_voice,
                label="Azure neural voice",
                info="Voices update based on target language.",
                interactive=bool(voice_choices),
            )

        with gr.Row():
            source_info = gr.Markdown(describe_language(default_source_code))
            target_info = gr.Markdown(describe_language(default_target_code))
            voice_info = gr.Markdown(describe_voice(default_voice))

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

        source_dropdown.change(
            fn=describe_language,
            inputs=source_dropdown,
            outputs=source_info,
        )
        target_dropdown.change(
            fn=describe_language,
            inputs=target_dropdown,
            outputs=target_info,
        )
        target_dropdown.change(
            fn=update_voices,
            inputs=target_dropdown,
            outputs=voice_dropdown,
        )
        target_dropdown.change(
            fn=describe_default_voice,
            inputs=target_dropdown,
            outputs=voice_info,
        )
        voice_dropdown.change(
            fn=describe_voice,
            inputs=voice_dropdown,
            outputs=voice_info,
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
