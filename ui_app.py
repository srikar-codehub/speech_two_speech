import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class VoiceOption:
    short_name: str
    locale: str
    gender: str
    name: str


@dataclass(frozen=True)
class LanguageOption:
    code: str
    name: str
    native_name: str
    locales: List[str]
    voices: List[VoiceOption]
    default_locale: str


DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "fr"
DEFAULT_SILENCE_SECONDS = 3.0


def _load_json(filename: str):
    path = Path(__file__).resolve().parent / filename
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _matches_language(locale: str, code: str) -> bool:
    locale_lower = locale.lower()
    code_lower = code.lower()
    if locale_lower == code_lower:
        return True
    return locale_lower.split("-")[0] == code_lower.split("-")[0]


def _build_language_options() -> Tuple[
    Dict[str, LanguageOption], Dict[str, VoiceOption]
]:
    languages_data = _load_json("azure_languages.json")
    voices_data = _load_json("azure_voices.json")

    language_options: Dict[str, LanguageOption] = {}
    voices_by_name: Dict[str, VoiceOption] = {}

    for code, details in languages_data.items():
        voice_options: List[VoiceOption] = []
        for locale, voice_entries in voices_data.items():
            if not _matches_language(locale, code):
                continue
            for entry in voice_entries:
                short_name = entry.get("short_name")
                if not short_name:
                    continue
                voice = VoiceOption(
                    short_name=short_name,
                    locale=locale,
                    gender=entry.get("gender", "Unknown"),
                    name=entry.get("name", short_name),
                )
                voice_options.append(voice)

        if not voice_options:
            continue

        voice_options.sort(key=lambda v: v.short_name.lower())
        locales = sorted({voice.locale for voice in voice_options})
        default_locale = locales[0] if locales else ""

        option = LanguageOption(
            code=code,
            name=details.get("name", code),
            native_name=details.get("nativeName", details.get("name", code)),
            locales=locales,
            voices=voice_options,
            default_locale=default_locale,
        )
        language_options[code] = option
        voices_by_name.update({voice.short_name: voice for voice in voice_options})

    if not language_options:
        raise RuntimeError(
            "No language/voice combinations available. Run populate_values.py."
        )

    return language_options, voices_by_name


LANGUAGE_OPTIONS, VOICES_BY_NAME = _build_language_options()


class SpeechTranslationController:
    def __init__(
        self, languages: Dict[str, LanguageOption], voices: Dict[str, VoiceOption]
    ):
        self._languages = languages
        self._voices = voices
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
        self._current_tts: Optional[AzureTTS] = None

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

            source_option = self._languages.get(source_lang)
            target_option = self._languages.get(target_lang)
            if not source_option or not target_option:
                self._append_log("Invalid language selection.")
                self._set_status("Error")
                return "Invalid language selection"

            voice_option = self._voices.get(voice_name)
            if not voice_option or voice_option.short_name not in {
                v.short_name for v in target_option.voices
            }:
                voice_option = target_option.voices[0]
                voice_name = voice_option.short_name

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

    def _stop_pipeline(self, force: bool) -> str:
        with self._thread_lock:
            thread = self._thread
            if not thread or not thread.is_alive():
                self._set_status("Stopped")
                return "Already stopped"

            self._set_status("Hard stopping..." if force else "Stopping...")
            self._stop_event.set()
            current_tts = self._current_tts

        if self._vad_stream:
            try:
                self._vad_stream.close()
            except Exception:
                pass
            finally:
                self._vad_stream = None

        if force and current_tts:
            try:
                current_tts.stop()
            except Exception as exc:  # pragma: no cover - defensive log
                self._append_log(f"TTS stop error: {exc}")

        self._append_log(
            "Hard stop requested by user." if force else "Stop requested by user."
        )

        if thread:
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._append_log("Background thread is taking longer to stop...")
                thread.join()

        with self._thread_lock:
            self._thread = None

        self._append_log("Pipeline stopped.")
        self._set_status("Stopped")
        return "Stopped"

    def stop(self) -> str:
        return self._stop_pipeline(force=False)

    def hard_stop(self) -> str:
        return self._stop_pipeline(force=True)

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
            with self._thread_lock:
                self._current_tts = tts

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
            with self._thread_lock:
                self._current_tts = None
                if threading.current_thread() is self._thread:
                    self._thread = None

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

    def is_running(self) -> bool:
        with self._thread_lock:
            return self._thread is not None and self._thread.is_alive()

    def restart_with(
        self,
        source_lang: str,
        target_lang: str,
        voice_name: str,
        silence_seconds: float,
    ) -> str:
        self.stop()
        return self.start(source_lang, target_lang, voice_name, silence_seconds)


controller = SpeechTranslationController(LANGUAGE_OPTIONS, VOICES_BY_NAME)


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


def hard_stop_pipeline():
    return controller.hard_stop()


def refresh_outputs():
    logs, transcription, translation, status = controller.snapshot()
    return logs, transcription, translation, status


def update_voices(selected_target: str):
    option = LANGUAGE_OPTIONS.get(selected_target)
    if not option or not option.voices:
        return gr.update(choices=[], value=None, interactive=False)

    default_voice = option.voices[0].short_name
    choices = [voice.short_name for voice in option.voices]
    return gr.update(choices=choices, value=default_voice, interactive=True)


def describe_language(code: str) -> str:
    option = LANGUAGE_OPTIONS.get(code)
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
    voice = VOICES_BY_NAME.get(short_name)
    if not voice:
        return f"Voice **{short_name}** — metadata unavailable."
    return (
        f"**{voice.short_name}** — {voice.name} ({voice.gender}, locale {voice.locale})"
    )


def describe_default_voice(language_code: str) -> str:
    option = LANGUAGE_OPTIONS.get(language_code)
    default_voice = option.voices[0].short_name if option and option.voices else None
    return describe_voice(default_voice)


def render_language_card(code: str) -> str:
    option = LANGUAGE_OPTIONS.get(code)
    if not option:
        return (
            "<div class='info-card'>"
            f"<div class='info-card__title'>{code}</div>"
            "<div class='info-card__meta'>Metadata unavailable.</div>"
            "</div>"
        )

    locales = ", ".join(option.locales) if option.locales else "n/a"
    voice_count = len(option.voices)

    card_parts = [
        "<div class='info-card'>",
        f"<div class='info-card__title'>{option.name} — {option.code}</div>",
    ]
    if option.native_name and option.native_name.lower() != option.name.lower():
        card_parts.append(
            f"<div class='info-card__subtitle'>{option.native_name}</div>"
        )
    card_parts.extend(
        [
            f"<div class='info-card__meta'><span>STT Locales:</span> {locales}</div>",
            f"<div class='info-card__meta'><span>Voices available:</span> {voice_count}</div>",
            "</div>",
        ]
    )
    return "".join(card_parts)


def render_voice_card(short_name: Optional[str]) -> str:
    if not short_name:
        return (
            "<div class='info-card'>"
            "<div class='info-card__title'>Voice unavailable</div>"
            "<div class='info-card__meta'>Metadata unavailable.</div>"
            "</div>"
        )
    voice = VOICES_BY_NAME.get(short_name)
    if not voice:
        return (
            "<div class='info-card'>"
            f"<div class='info-card__title'>{short_name}</div>"
            "<div class='info-card__meta'>Metadata unavailable.</div>"
            "</div>"
        )
    return (
        "<div class='info-card'>"
        f"<div class='info-card__title'>{voice.short_name} — {voice.name}</div>"
        f"<div class='info-card__meta'>{voice.gender}, locale {voice.locale}</div>"
        "</div>"
    )


def render_selected_voice(short_name: Optional[str]) -> str:
    if not short_name:
        return "<div class='selected-voice'><span>Selected voice:</span> unavailable.</div>"
    voice = VOICES_BY_NAME.get(short_name)
    if not voice:
        return (
            "<div class='selected-voice'>"
            f"<span>Selected voice:</span> {short_name} (metadata unavailable)"
            "</div>"
        )
    return (
        "<div class='selected-voice'>"
        f"<span>Selected voice:</span> {voice.short_name} — {voice.name} "
        f"({voice.gender}, locale {voice.locale})"
        "</div>"
    )


def render_default_voice_label(short_name: Optional[str]) -> str:
    if not short_name:
        return "<div class='default-voice'>Default voice: unavailable.</div>"
    return (
        f"<div class='default-voice'>Default voice: {short_name}</div>"
    )


def render_default_voice_views(language_code: str) -> Tuple[str, str, str]:
    option = LANGUAGE_OPTIONS.get(language_code)
    default_voice = option.voices[0].short_name if option and option.voices else None
    return (
        render_voice_card(default_voice),
        render_selected_voice(default_voice),
        render_default_voice_label(default_voice),
    )


def render_voice_views(short_name: Optional[str]) -> Tuple[str, str]:
    return render_voice_card(short_name), render_selected_voice(short_name)


def get_logo_source() -> str:
    logo_path = Path(__file__).resolve().parent / "download.png"
    if logo_path.exists():
        try:
            encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{encoded_logo}"
        except OSError:
            pass
    return "file=download.png"


def apply_settings(
    source_lang: str,
    target_lang: str,
    voice_name: str,
    silence_seconds: float,
) -> str:
    silence_seconds = float(silence_seconds)
    status = controller.restart_with(
        source_lang,
        target_lang,
        voice_name,
        silence_seconds,
    )
    return (
        f"Applied settings — source {source_lang}, target {target_lang}, "
        f"voice {voice_name or 'auto'}, silence {silence_seconds:.1f}s "
        f"({status})."
    )


def handle_silence_change(
    source_lang: str,
    target_lang: str,
    voice_name: str,
    silence_seconds: float,
):
    silence_seconds = float(silence_seconds)
    if controller.is_running():
        status = controller.restart_with(
            source_lang,
            target_lang,
            voice_name,
            silence_seconds,
        )
        return f"Restarted with silence duration {silence_seconds:.1f}s ({status})."

    return f"Silence duration set to {silence_seconds:.1f}s (applies on next start)."


def build_interface():
    language_options = sorted(
        LANGUAGE_OPTIONS.values(), key=lambda opt: opt.name.lower()
    )
    if not language_options:
        raise RuntimeError("Azure metadata did not return any languages.")

    language_items = [(option.name, option.code) for option in language_options]
    language_codes = [code for _, code in language_items]

    default_source_code = (
        DEFAULT_SOURCE if LANGUAGE_OPTIONS.get(DEFAULT_SOURCE) else language_codes[0]
    )
    default_target_code = (
        DEFAULT_TARGET if LANGUAGE_OPTIONS.get(DEFAULT_TARGET) else language_codes[0]
    )

    target_option = LANGUAGE_OPTIONS.get(default_target_code)
    voice_choices = (
        [voice.short_name for voice in target_option.voices] if target_option else []
    )
    default_voice = voice_choices[0] if voice_choices else None

    theme_path = Path(__file__).resolve().parent / "custom_theme.css"
    if theme_path.exists():
        custom_css = theme_path.read_text(encoding="utf-8")
    else:
        custom_css = """
:root {
    --protiviti-blue-900: #1B4E8E;
    --protiviti-blue-800: #254B73;
    --protiviti-blue-700: #1C355A;
    --protiviti-blue-600: #4A6FA5;
    --protiviti-blue-500: #6C7E92;
    --surface-light: #f5f7fb;
    --surface-white: #ffffff;
    --text-primary: #5f6368;
    --text-secondary: #7a7d80;
    --shadow-soft: 0 12px 24px rgba(27, 78, 142, 0.08);
    --radius-card: 12px;
    --radius-element: 10px;
    --transition-default: 0.25s ease;
}

body,
.gradio-container {
    background: var(--surface-light);
    color: var(--text-primary);
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
}

.app-container {
    max-width: 960px;
    margin: 0 auto 3rem auto;
    gap: 0;
}

.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface-white);
    border-radius: var(--radius-card);
    padding: 1.5rem 2rem;
    box-shadow: var(--shadow-soft);
    margin-top: 1.5rem;
}

.top-bar__title h1 {
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0 0 0.35rem 0;
}

.top-bar__title p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.top-bar__logo {
    display: flex;
    align-items: center;
    justify-content: flex-end;
}

.logo {
    height: 36px;
    width: auto;
}

.section-card {
    background: var(--surface-white);
    border-radius: var(--radius-card);
    padding: 1.75rem 2rem;
    box-shadow: var(--shadow-soft);
    margin-top: 1.75rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}

.section-title {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--text-primary);
    margin: 0 !important;
}

.section-description {
    color: var(--text-secondary);
    margin: -0.25rem 0 0.5rem 0 !important;
}

.language-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.language-row > * {
    flex: 1 1 280px;
}

.control-input select,
.control-input textarea,
.control-input input {
    border-radius: var(--radius-element) !important;
    border: 1px solid rgba(27, 78, 142, 0.15) !important;
    box-shadow: none !important;
}

.default-voice {
    font-weight: 500;
    color: var(--text-primary);
}

.section-accordion {
    border-radius: var(--radius-card) !important;
    box-shadow: var(--shadow-soft);
    background: var(--surface-white);
    margin-top: 1.75rem;
}

.section-accordion > .label {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
}

.accordion-description {
    color: var(--text-secondary);
    margin-bottom: 1rem !important;
}

.info-card-stack {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.info-card {
    background: #f4f7fc;
    border-radius: var(--radius-element);
    padding: 1rem 1.25rem;
    box-shadow: inset 0 1px 0 rgba(27, 78, 142, 0.05);
    border: 1px solid rgba(27, 78, 142, 0.08);
}

.info-card__title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.info-card__subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
}

.info-card__meta {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin: 0.15rem 0;
}

.info-card__meta span {
    font-weight: 600;
    color: var(--text-primary);
    margin-right: 0.35rem;
}

.selected-voice {
    font-weight: 500;
    color: var(--text-primary);
}

.selected-voice span {
    font-weight: 600;
    margin-right: 0.35rem;
    color: var(--text-primary);
}

.section-note {
    color: var(--text-secondary);
    margin-top: -0.75rem !important;
}

.silence-slider input[type="range"] {
    accent-color: var(--protiviti-blue-900);
}

.action-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

.action-row > * {
    flex: 1 1 140px;
}

.btn-start button {
    background-color: var(--protiviti-blue-900) !important;
    color: #ffffff !important;
    border-radius: var(--radius-element) !important;
    box-shadow: var(--shadow-soft);
    transition: var(--transition-default);
}

.btn-start button:hover {
    background-color: #225ca4 !important;
}

.btn-stop button {
    background-color: var(--protiviti-blue-800) !important;
    color: #ffffff !important;
    border-radius: var(--radius-element) !important;
    box-shadow: var(--shadow-soft);
    transition: var(--transition-default);
}

.btn-stop button:hover {
    background-color: #2d5b8a !important;
}

.btn-hardstop button {
    background-color: var(--protiviti-blue-700) !important;
    color: #ffffff !important;
    border-radius: var(--radius-element) !important;
    box-shadow: var(--shadow-soft);
    transition: var(--transition-default);
}

.btn-hardstop button:hover {
    background-color: #24426e !important;
}

.btn-apply button {
    background-color: var(--protiviti-blue-600) !important;
    color: #ffffff !important;
    border-radius: var(--radius-element) !important;
    box-shadow: var(--shadow-soft);
    transition: var(--transition-default);
}

.btn-apply button:hover {
    background-color: #567cb4 !important;
}

.btn-restart button {
    background-color: var(--protiviti-blue-500) !important;
    color: #ffffff !important;
    border-radius: var(--radius-element) !important;
    box-shadow: var(--shadow-soft);
    transition: var(--transition-default);
}

.btn-restart button:hover {
    background-color: #788a9f !important;
}

.output-section textarea,
.status-field textarea {
    border-radius: var(--radius-element) !important;
    border: 1px solid rgba(27, 78, 142, 0.15) !important;
    box-shadow: none !important;
    background: #fdfefe !important;
}

.status-field textarea {
    font-weight: 600;
    color: var(--text-primary);
}

.output-field textarea {
    color: var(--text-primary);
}

@media (max-width: 768px) {
    .top-bar {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }

    .top-bar__logo {
        width: 100%;
        justify-content: flex-start;
    }

    .language-row {
        flex-direction: column;
    }

    .action-row {
        flex-direction: column;
    }
}
        """

    default_voice_label = render_default_voice_label(default_voice)
    initial_voice_card = render_voice_card(default_voice)
    initial_voice_summary = render_selected_voice(default_voice)
    logo_source = get_logo_source()

    with gr.Blocks(
        title="Speech to Speech Translator",
        css=custom_css,
    ) as demo:
        with gr.Column(elem_classes=["app-container"]):
            gr.HTML(
                f"""
                <div class="top-bar">
                    <div class="top-bar__title">
                        <h1>Speech to Speech Translator</h1>
                        <p>Real-time Silero VAD -> Azure STT -> Azure Translator -> Azure TTS</p>
                    </div>
                    <div class="top-bar__logo">
                        <img src="{logo_source}" alt="Protiviti logo" class="logo"/>
                    </div>
                </div>
                """
            )

            with gr.Column(elem_classes=["section-card", "inputs-section"]):
                gr.Markdown(
                    "### Input Controls",
                    elem_classes=["section-title"],
                )
                gr.Markdown(
                    "Run the Silero VAD -> Azure STT -> Azure Translator -> Azure TTS loop.",
                    elem_classes=["section-description"],
                )
                with gr.Row(elem_classes=["language-row"]):
                    source_dropdown = gr.Dropdown(
                        choices=language_items,
                        value=default_source_code,
                        label="Source language (translation code)",
                        interactive=True,
                        elem_classes=["control-input"],
                    )
                    target_dropdown = gr.Dropdown(
                        choices=language_items,
                        value=default_target_code,
                        label="Target language (translation code)",
                        interactive=True,
                        elem_classes=["control-input"],
                    )

                voice_dropdown = gr.Dropdown(
                    choices=voice_choices,
                    value=default_voice,
                    label="Azure neural voice",
                    info="Voices update based on target language.",
                    interactive=bool(voice_choices),
                    elem_classes=["control-input"],
                )
                default_voice_display = gr.HTML(
                    default_voice_label,
                    elem_classes=["default-voice"],
                )

            with gr.Accordion(
                "Language & Voice Details",
                open=False,
                elem_classes=["section-accordion"],
            ):
                gr.Markdown(
                    "Metadata updates automatically as you adjust the configuration.",
                    elem_classes=["accordion-description"],
                )
                with gr.Column(elem_classes=["info-card-stack"]):
                    source_info = gr.HTML(
                        render_language_card(default_source_code),
                        elem_classes=["info-card-wrapper"],
                    )
                    target_info = gr.HTML(
                        render_language_card(default_target_code),
                        elem_classes=["info-card-wrapper"],
                    )
                    voice_info = gr.HTML(
                        initial_voice_card,
                        elem_classes=["info-card-wrapper"],
                    )

            with gr.Column(elem_classes=["section-card", "settings-section"]):
                voice_summary = gr.HTML(
                    initial_voice_summary,
                    elem_classes=["selected-voice"],
                )
                gr.Markdown(
                    "Voices update based on target language.",
                    elem_classes=["section-note"],
                )
                silence_slider = gr.Slider(
                    minimum=0.5,
                    maximum=10.0,
                    value=DEFAULT_SILENCE_SECONDS,
                    step=0.5,
                    label="Silence duration (seconds)",
                    info="Silero waits this long after speech before finalizing a segment.",
                    elem_classes=["silence-slider"],
                )
                with gr.Row(elem_classes=["action-row"]):
                    start_button = gr.Button(
                        "Start",
                        variant="primary",
                        elem_classes=["btn-start"],
                    )
                    stop_button = gr.Button(
                        "Stop",
                        elem_classes=["btn-stop"],
                    )
                    hard_stop_button = gr.Button(
                        "Hard Stop",
                        variant="stop",
                        elem_classes=["btn-hardstop"],
                    )
                    apply_button = gr.Button(
                        "Apply & Restart",
                        elem_classes=["btn-apply"],
                    )

            with gr.Column(elem_classes=["section-card", "output-section"]):
                gr.Markdown(
                    "### Real-Time Output",
                    elem_classes=["section-title"],
                )
                status_box = gr.Textbox(
                    label="Status",
                    value="Stopped",
                    lines=1,
                    interactive=False,
                    elem_classes=["status-field"],
                )
                transcription_box = gr.Textbox(
                    label="Live transcription",
                    lines=4,
                    interactive=False,
                    placeholder="Recognized text will appear here.",
                    elem_classes=["output-field"],
                )
                translation_box = gr.Textbox(
                    label="Live translation",
                    lines=4,
                    interactive=False,
                    placeholder="Translated text will appear here.",
                    elem_classes=["output-field"],
                )
                log_box = gr.Textbox(
                    label="Logs",
                    lines=12,
                    interactive=False,
                    placeholder="Pipeline logs will appear here.",
                    elem_classes=["output-field"],
                )

        source_dropdown.change(
            fn=render_language_card,
            inputs=source_dropdown,
            outputs=source_info,
        )
        target_dropdown.change(
            fn=render_language_card,
            inputs=target_dropdown,
            outputs=target_info,
        )
        target_dropdown.change(
            fn=update_voices,
            inputs=target_dropdown,
            outputs=voice_dropdown,
        )
        target_dropdown.change(
            fn=render_default_voice_views,
            inputs=target_dropdown,
            outputs=[voice_info, voice_summary, default_voice_display],
        )
        voice_dropdown.change(
            fn=render_voice_views,
            inputs=voice_dropdown,
            outputs=[voice_info, voice_summary],
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
        hard_stop_button.click(
            fn=hard_stop_pipeline,
            outputs=status_box,
        )
        apply_button.click(
            fn=apply_settings,
            inputs=[
                source_dropdown,
                target_dropdown,
                voice_dropdown,
                silence_slider,
            ],
            outputs=status_box,
        )

        silence_slider.change(
            fn=handle_silence_change,
            inputs=[
                source_dropdown,
                target_dropdown,
                voice_dropdown,
                silence_slider,
            ],
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
