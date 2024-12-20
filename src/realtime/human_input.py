from __future__ import annotations

import asyncio
import numpy as np
from typing import Literal, Generator

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
    def __init__(self):
        self.buffer = []
        self.sample_rate = 16000  # Standard sample rate

    def add_frame(self, frame):
        self.buffer.extend(frame.data)

    def get_audio(self) -> np.ndarray:
        if not self.buffer:
            return np.array([])
        return np.array(self.buffer, dtype=np.float32)

    def clear(self):
        self.buffer = []


class HumanInput(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        room: rtc.Room,
        vad: voice_activity_detection.VAD,
        participant: rtc.RemoteParticipant,
        transcription: bool,
        model_fn: callable,  # The gen_from_model function
    ) -> None:
        super().__init__()
        self._room = room
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

        def _before_forward(fwd: transcription.STTSegmentsForwarder, transcription: rtc.Transcription):
            if not self._transcription:
                transcription.segments = []
            return transcription

        stt_forwarder = transcription.STTSegmentsForwarder(
            room=self._room,
            participant=self._participant,
            track=self._subscribed_track,
            before_forward_cb=_before_forward,
        )

        async def _audio_stream_co() -> None:
            async for ev in audio_stream:
                vad_stream.push_frame(ev.frame)
                if self._speaking:
                    self._audio_buffer.add_frame(ev.frame)

        async def _vad_stream_co() -> None:
            async for ev in vad_stream:
                if ev.type == voice_activity_detection.VADEventType.START_OF_SPEECH:
                    self._speaking = True
                    self._audio_buffer.clear()  # Clear buffer for new utterance
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
                        # Run the model in a separate thread to avoid blocking
                        loop = asyncio.get_event_loop()
                        generator = await loop.run_in_executor(None, self._model_fn, audio_data)

                        # Process the generator results
                        accumulated_text = ""
                        for text in generator:
                            if text:
                                accumulated_text += text
                                # Emit interim results
                                self.emit(
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
                                                    {"text": accumulated_text, "language": "en"},  # Default to English
                                                )
                                            ],
                                        },
                                    ),
                                )

                        # Emit final transcript
                        if accumulated_text:
                            self.emit(
                                "final_transcript",
                                type(
                                    "SpeechEvent",
                                    (),
                                    {
                                        "type": "final_response",
                                        "alternatives": [
                                            type("Alternative", (), {"text": accumulated_text, "language": "en"})
                                        ],
                                    },
                                ),
                            )

        tasks = [
            asyncio.create_task(_audio_stream_co()),
            asyncio.create_task(_vad_stream_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await stt_forwarder.aclose()
            await vad_stream.aclose()
