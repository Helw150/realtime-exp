from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncIterable, Callable, Union, Literal

from livekit import rtc

from livekit.agents import tokenize, utils
from livekit.agents import transcription as agent_transcription
from livekit.agents import tts as text_to_speech
from .agent_playout import AgentPlayout, PlayoutHandle
from .log import logger

SpeechSource = Union[str, AsyncIterable[str]]

EventTypes = Literal["final_response",]


class SynthesisHandle:
    """Handles synthesis of speech from text input to audio output"""

    def __init__(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        transcript_source: SpeechSource,
        agent_playout: AgentPlayout,
        tts: text_to_speech.TTS,
        transcription_fwd: agent_transcription.TTSSegmentsForwarder,
    ) -> None:
        self._tts_source = tts_source
        self._transcript_source = transcript_source
        self._agent_playout = agent_playout
        self._tts = tts
        self._tr_fwd = transcription_fwd
        self._buf_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._play_handle: PlayoutHandle | None = None
        self._interrupt_fut = asyncio.Future[None]()
        self._speech_id = speech_id

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def tts_forwarder(self) -> agent_transcription.TTSSegmentsForwarder:
        return self._tr_fwd

    @property
    def validated(self) -> bool:
        return self._play_handle is not None

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def play_handle(self) -> PlayoutHandle | None:
        return self._play_handle

    def play(self) -> PlayoutHandle:
        """Validate the speech for playout"""
        if self.interrupted:
            raise RuntimeError("synthesis was interrupted")

        self._play_handle = self._agent_playout.play(self._speech_id, self._buf_ch, transcription_fwd=self._tr_fwd)
        return self._play_handle

    def interrupt(self) -> None:
        """Interrupt the speech"""
        if self.interrupted:
            return

        logger.debug(
            "agent interrupted",
            extra={"speech_id": self.speech_id},
        )

        if self._play_handle is not None:
            self._play_handle.interrupt()

        self._interrupt_fut.set_result(None)


class AgentOutput(utils.EventEmitter[EventTypes]):
    """Handles the agent's speech output pipeline"""

    def __init__(
        self,
        *,
        room: rtc.Room,
        agent_playout: AgentPlayout,
        tts: text_to_speech.TTS,
    ) -> None:
        super().__init__()
        self._room = room
        self._agent_playout = agent_playout
        self._tts = tts
        self._tasks = set[asyncio.Task[Any]]()

    @property
    def playout(self) -> AgentPlayout:
        return self._agent_playout

    async def aclose(self) -> None:
        """Clean up resources and cancel pending tasks"""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def synthesize(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        transcript_source: SpeechSource,
        transcription: bool,
        transcription_speed: float,
        sentence_tokenizer: tokenize.SentenceTokenizer,
        word_tokenizer: tokenize.WordTokenizer,
        hyphenate_word: Callable[[str], list[str]],
    ) -> SynthesisHandle:
        """Create a new synthesis handle for the given speech source"""

        def _before_forward(
            fwd: agent_transcription.TTSSegmentsForwarder,
            transcription: rtc.Transcription,
        ):
            if not transcription:
                transcription.segments = []
            return transcription

        transcription_fwd = agent_transcription.TTSSegmentsForwarder(
            room=self._room,
            participant=self._room.local_participant,
            speed=transcription_speed,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            before_forward_cb=_before_forward,
        )

        handle = SynthesisHandle(
            speech_id=speech_id,
            tts_source=tts_source,
            transcript_source=transcript_source,
            agent_playout=self._agent_playout,
            tts=self._tts,
            transcription_fwd=transcription_fwd,
        )

        task = asyncio.create_task(self._synthesize_task(handle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)
        return handle

    def done_callback_factory(self, handle):
        def done_cb(tts_text):
            self.emit(
                "final_response",
                type(
                    "SpeechEvent",
                    (),
                    {
                        "type": "final_response",
                        "alternatives": [
                            type(
                                "Alternative",
                                (),
                                {"text": tts_text.result(), "language": "en"},
                            )
                        ],
                    },
                ),
            )

        return done_cb

    @utils.log_exceptions(logger=logger)
    async def _synthesize_task(self, handle: SynthesisHandle) -> None:
        """Process the speech source and generate audio"""
        tts_source = handle._tts_source
        transcript_source = handle._transcript_source

        # co = self.dummy_synthesis(tts_source, transcript_source, handle)

        if isinstance(tts_source, str):
            co = self._str_synthesis_task(tts_source, transcript_source, handle)
        else:
            co = self._stream_synthesis_task(tts_source, transcript_source, handle)

        synth = asyncio.create_task(co)
        synth.add_done_callback(self.done_callback_factory(handle))
        try:
            await asyncio.wait(
                [synth, handle._interrupt_fut],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            await utils.aio.gracefully_cancel(synth)

    @utils.log_exceptions(logger=logger)
    async def dummy_synthesis(
        self,
        tts_source: AsyncIterable[str],
        transcript_source: AsyncIterable[str] | str,
        handle: SynthesisHandle,
    ):
        final_response = ""
        try:
            async for resp in tts_source:
                final_response = resp
        finally:
            return final_response

    @utils.log_exceptions(logger=logger)
    async def _str_synthesis_task(
        self,
        tts_text: str,
        transcript_source: AsyncIterable[str] | str,
        handle: SynthesisHandle,
    ) -> str:
        """Synthesize speech from a complete text string"""
        first_frame = True

        tts_stream = handle._tts.synthesize(tts_text)
        try:
            async for audio in tts_stream:
                if first_frame:
                    first_frame = False

                handle._buf_ch.send_nowait(audio.frame)
                if not handle.tts_forwarder.closed:
                    handle.tts_forwarder.push_audio(audio.frame)

            if not handle.tts_forwarder.closed:
                handle.tts_forwarder.mark_audio_segment_end()

        finally:
            await tts_stream.aclose()
            return tts_text

    @utils.log_exceptions(logger=logger)
    async def _stream_synthesis_task(
        self,
        tts_source: AsyncIterable[str],
        transcript_source: AsyncIterable[str] | str,
        handle: SynthesisHandle,
    ) -> str:
        """Synthesize speech from a stream of text chunks"""

        @utils.log_exceptions(logger=logger)
        async def _read_generated_audio_task(
            tts_stream: text_to_speech.SynthesizeStream,
        ) -> None:
            try:
                async for audio in tts_stream:
                    if not handle._tr_fwd.closed:
                        handle._tr_fwd.push_audio(audio.frame)
                    handle._buf_ch.send_nowait(audio.frame)
            finally:
                if handle._tr_fwd and not handle._tr_fwd.closed:
                    handle._tr_fwd.mark_audio_segment_end()
                await tts_stream.aclose()

        tts_stream: text_to_speech.SynthesizeStream | None = None
        read_tts_atask: asyncio.Task | None = None
        tts_text = ""
        try:
            async for seg in tts_source:
                if tts_stream is None:
                    tts_stream = handle._tts.stream()
                    read_tts_atask = asyncio.create_task(_read_generated_audio_task(tts_stream))

                tts_stream.push_text(seg)
                tts_text += seg

            if tts_stream is not None:
                tts_stream.end_input()
                assert read_tts_atask
                await read_tts_atask

        finally:
            if read_tts_atask is not None:
                await utils.aio.gracefully_cancel(read_tts_atask)

            if inspect.isasyncgen(tts_source):
                await tts_source.aclose()
                return tts_text
