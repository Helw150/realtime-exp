from __future__ import annotations

import asyncio
import numpy as np
from typing import Literal, Callable

from livekit import rtc

from livekit.agents.pipeline import transcription, utils, vad as voice_activity_detection
from realtime.log import logger

EventTypes = Literal[
    "start_of_speech",
    "vad_inference_done",
    "end_of_speech",
    "final_response",
    "interim_response",
]


class AudioBuffer:
    def __init__(self, max_seconds: float = 30.0):
        # Pre-allocate for better performance
        self.sample_rate = 16000
        self.max_samples = int(max_seconds * self.sample_rate)  # 16kHz sampling
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.current_size = 0

    def add_frame(self, frame) -> bool:
        """Returns False if buffer is full"""
        if len(self.buffer) >= self.max_samples:
            return False

        # Convert and normalize frame data
        frame_data = np.array(frame.data, dtype=np.float32)
        frame_data = np.clip(frame_data, -1.0, 1.0)
        self.buffer.extend(frame_data)
        return True

    def get_audio(self) -> np.ndarray:
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.array(self.buffer, dtype=np.float32)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


async def process_model_output(
    model_fn: Callable[[np.ndarray, List[ChatMessage]], AsyncIterable[str]],
    audio_data: np.ndarray,
    emit_fn: Callable,
    timeout: float = 30.0,
) -> None:
    """Process model output with timeout and error handling"""
    try:
        accumulated_text = ""
        async for text in model_fn(audio_data, self._chat_ctx.messages):
            if text:
                accumulated_text += text
                accumulated_text += text
                # Emit interim results
                emit_fn(
                    "interim_response",
                    type(
                        "SpeechEvent",
                        (),
                        {
                            "type": "interim_response",
                            "alternatives": [
                                type(
                                    "Alternative",
                                    (),
                                    {"text": accumulated_text, "language": "en"},
                                )
                            ],
                        },
                    ),
                )

        if accumulated_text:
            emit_fn(
                "final_response",
                type(
                    "SpeechEvent",
                    (),
                    {
                        "type": "final_response",
                        "alternatives": [type("Alternative", (), {"text": accumulated_text, "language": "en"})],
                    },
                ),
            )

    except asyncio.TimeoutError:
        logger.error("Model processing timed out")
        # Emit error event if needed
    except Exception as e:
        logger.error(f"Error processing model output: {str(e)}")
        # Emit error event if needed


class HumanInput(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        room: rtc.Room,
        chat_ctx: ChatContext,
        vad: voice_activity_detection.VAD,
        participant: rtc.RemoteParticipant,
        transcription: bool,
        model_fn: Callable[[np.ndarray, List[ChatMessage]], AsyncIterable[str]],
    ) -> None:
        super().__init__()
        self._room = room
        self._chat_ctx = chat_ctx
        self._vad = vad
        self._participant = participant
        self._transcription = transcription
        self._model_fn = model_fn
        self._subscribed_track: rtc.RemoteAudioTrack | None = None
        self._recognize_atask: asyncio.Task[None] | None = None

        self._closed = False
        self._speaking = False
        self._speech_probability = 0.0
        self._audio_buffer = AudioBuffer()

        self._room.on("track_published", self._subscribe_to_microphone)
        self._room.on("track_subscribed", self._subscribe_to_microphone)
        self._subscribe_to_microphone()

    async def aclose(self) -> None:
        if self._closed:
            raise RuntimeError("HumanInput already closed")

        self._closed = True
        self._room.off("track_published", self._subscribe_to_microphone)
        self._room.off("track_subscribed", self._subscribe_to_microphone)
        self._speaking = False

        if self._recognize_atask is not None:
            await utils.aio.gracefully_cancel(self._recognize_atask)

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def speaking_probability(self) -> float:
        return self._speech_probability

    def _subscribe_to_microphone(self, *args, **kwargs) -> None:
        """
        Subscribe to the participant microphone if found and not already subscribed.
        Do nothing if no track is found.
        """
        for publication in self._participant.track_publications.values():
            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                continue

            if not publication.subscribed:
                publication.set_subscribed(True)

            track: rtc.RemoteAudioTrack | None = publication.track  # type: ignore
            if track is not None and track != self._subscribed_track:
                self._subscribed_track = track
                if self._recognize_atask is not None:
                    self._recognize_atask.cancel()

                self._recognize_atask = asyncio.create_task(
                    self._recognize_task(rtc.AudioStream(track, sample_rate=16000))
                )
                break

    @utils.log_exceptions(logger=logger)
    async def _recognize_task(self, audio_stream: rtc.AudioStream) -> None:
        """
        Receive the frames from the user audio stream and detect voice activity.
        When speech ends, process the complete utterance with the model.
        """
        vad_stream = self._vad.stream()

        async def _audio_stream_co() -> None:
            async for ev in audio_stream:
                vad_stream.push_frame(ev.frame)
                if self._speaking:
                    if not self._audio_buffer.add_frame(ev.frame):
                        logger.warning("Audio buffer full, stopping recording")
                        self._speaking = False

        async def _vad_stream_co() -> None:
            async for ev in vad_stream:
                if ev.type == voice_activity_detection.VADEventType.START_OF_SPEECH:
                    self._speaking = True
                    self._audio_buffer.clear()
                    self.emit("start_of_speech", ev)
                elif ev.type == voice_activity_detection.VADEventType.INFERENCE_DONE:
                    self._speech_probability = ev.probability
                    self.emit("vad_inference_done", ev)
                elif ev.type == voice_activity_detection.VADEventType.END_OF_SPEECH:
                    self._speaking = False
                    self.emit("end_of_speech", ev)

                    # Process the complete utterance
                    audio_data = self._audio_buffer.get_audio()
                    if len(audio_data) > 0:
                        await process_model_output(self._model_fn, audio_data, self.emit)

        tasks = [
            asyncio.create_task(_audio_stream_co()),
            asyncio.create_task(_vad_stream_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await vad_stream.aclose()
