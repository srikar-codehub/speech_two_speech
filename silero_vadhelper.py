"""
silero_vadhelper.py
-------------------
Detects speech vs silence in real time using Silero VAD.
Buffers audio and yields complete speech segments when 3s of silence are detected.
"""

import torch
import sounddevice as sd
import numpy as np
import time
from queue import Queue


class SileroVADHelper:
    def __init__(self, sample_rate=16000, threshold=0.6, silence_duration=3.0, min_chunk_sec=0.5):
        """
        Args:
            sample_rate: Audio sampling rate (Hz)
            threshold: Probability threshold for speech activity
            silence_duration: Seconds of silence before yielding a segment
            min_chunk_sec: Minimum valid speech duration to send to Azure
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.min_chunk_sec = min_chunk_sec

        print("🔹 Loading Silero VAD model...")
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        (_, _, _, self.VADIterator, _) = utils

        self.audio_queue = Queue()
        self.speech_active = False
        self.speech_buffer = []
        self.last_speech_time = 0

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("⚠️", status)
        self.audio_queue.put(indata.copy())

    def start(self):
        """Yields full speech segments when 3s of silence detected."""
        frame_size = 512
        rolling_buffer = np.zeros(0, dtype=np.float32)

        with sd.InputStream(channels=1, samplerate=self.sample_rate,
                            callback=self._audio_callback, dtype="float32"):
            print("🎙️ Listening (Silero VAD active, 3 s silence threshold)...\n")

            while True:
                if not self.audio_queue.empty():
                    frame = self.audio_queue.get().flatten()
                    rolling_buffer = np.concatenate((rolling_buffer, frame))

                    while len(rolling_buffer) >= frame_size:
                        chunk = rolling_buffer[:frame_size]
                        rolling_buffer = rolling_buffer[frame_size:]
                        audio_tensor = torch.from_numpy(chunk)

                        try:
                            speech_prob = self.model(audio_tensor, self.sample_rate).item()
                        except Exception as e:
                            print(f"⚠️ Silero VAD error: {e}")
                            continue

                        # speech detected
                        if speech_prob > self.threshold:
                            if not self.speech_active:
                                print("🗣️ Speech started")
                                self.speech_buffer = []
                            self.speech_active = True
                            self.speech_buffer.append(audio_tensor)
                            self.last_speech_time = time.time()

                        # silence after speech
                        elif self.speech_active and (time.time() - self.last_speech_time > self.silence_duration):
                            print(f"🤫 Silence detected for {self.silence_duration:.1f}s — finalizing segment...")
                            full_chunk = torch.cat(self.speech_buffer)
                            self.speech_active = False
                            self.speech_buffer = []

                            # skip too-short noise
                            if full_chunk.numel() < self.sample_rate * self.min_chunk_sec:
                                print("⚠️ Skipping very short/empty chunk")
                            else:
                                yield full_chunk

                time.sleep(0.01)
