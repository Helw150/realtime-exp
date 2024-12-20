from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterable, Callable, Literal, Optional, List, Union

import numpy as np
from livekit import rtc

from .. import metrics, tokenize, tts, utils, vad
from ..types import ATTRIBUTE_AGENT_STATE, AgentState
from .agent_output import AgentOutput, SpeechSource, SynthesisHandle
from .agent_playout import AgentPlayout
from .human_input import HumanInput
from .log import logger
from .plotter import AssistantPlotter
from .speech_handle import SpeechHandle


@dataclass
class ChatMessage:
    text: str
    role: Literal["user", "assistant"]

    @staticmethod
    def create(text: str, role: Literal["user", "assistant"]) -> "ChatMessage":
        return ChatMessage(text=text, role=role)


class ChatContext:
    def __init__(self):
        self.messages: List[ChatMessage] = []

    def copy(self) -> "ChatContext":
        new_ctx = ChatContext()
        new_ctx.messages = self.messages.copy()
        return new_ctx


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_speech_committed",
    "agent_speech_committed",
    "agent_speech_interrupted",
    "metrics_collected",
]

BeforeTTSCallback = Callable[
    ["VoicePipelineAgent", Union[str, AsyncIterable[str]]],
    SpeechSource,
]


@dataclass(frozen=True)
class _ImplOptions:
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    min_endpointing_delay: float
    preemptive_synthesis: bool
    before_tts_cb: BeforeTTSCallback
    plotting: bool
    transcription: AgentTranscriptionOptions


@dataclass(frozen=True)
class AgentTranscriptionOptions:
    user_transcription: bool = True
    """Whether to forward the user transcription to the client"""
    agent_transcription: bool = True
    """Whether to forward the agent transcription to the client"""
    agent_transcription_speed: float = 1.0
    """The speed at which the agent's speech transcription is forwarded to the client.
    We try to mimic the agent's speech speed by adjusting the transcription speed."""
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    """The tokenizer used to split the speech into sentences.
    This is used to decide when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""


class _TurnDetector(Protocol):
    def unlikely_threshold(self) -> float:
        ...

    def supports_language(self, language: str | None) -> bool:
        ...

    async def predict_end_of_turn(self, chat_ctx: ChatContext) -> float:
        ...


class VoicePipelineAgent(utils.EventEmitter[EventTypes]):
    """
    A pipeline agent (VAD + Unified Model + TTS) implementation.
    """

    MIN_TIME_PLAYED_FOR_COMMIT = 1.5
    """Minimum time played for the user speech to be committed to the chat context"""

    def __init__(
        self,
        *,
        vad: vad.VAD,
        model_fn: Callable[[np.ndarray, List[ChatMessage]], AsyncIterable[str]],
        tts: tts.TTS,
        turn_detector: _TurnDetector | None = None,
        chat_ctx: ChatContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.5,
        interrupt_min_words: int = 0,
        min_endpointing_delay: float = 0.5,
        preemptive_synthesis: bool = False,
        transcription: AgentTranscriptionOptions = AgentTranscriptionOptions(),
        before_tts_cb: BeforeTTSCallback = lambda agent, text: text,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        self._opts = _ImplOptions(
            plotting=plotting,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            min_endpointing_delay=min_endpointing_delay,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            before_tts_cb=before_tts_cb,
        )
        self._plotter = AssistantPlotter(self._loop)

        self._vad = vad
        self._model_fn = model_fn
        self._tts = tts
        self._turn_detector = turn_detector
        self._chat_ctx = chat_ctx or ChatContext()
        self._started = False
        self._closed = False

        self._human_input: HumanInput | None = None
        self._agent_output: AgentOutput | None = None

        self._track_published_fut = asyncio.Future[None]()

        self._pending_agent_reply: SpeechHandle | None = None
        self._agent_reply_task: asyncio.Task[None] | None = None

        self._playing_speech: SpeechHandle | None = None
        self._response_text = ""

        self._deferred_validation = _DeferredReplyValidation(
            self._validate_reply_if_possible,
            self._opts.min_endpointing_delay,
            turn_detector=self._turn_detector,
            agent=self,
        )

        self._speech_q: list[SpeechHandle] = []
        self._speech_q_changed = asyncio.Event()

        self._update_state_task: asyncio.Task | None = None

        self._last_final_response_time: float | None = None
        self._last_speech_time: float | None = None

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tts(self) -> tts.TTS:
        return self._tts

    @property
    def vad(self) -> vad.VAD:
        return self._vad

    def start(self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None) -> None:
        """Start the voice assistant

        Args:
            room: the room to use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant found in the room will be selected
        """
        if self._started:
            raise RuntimeError("voice assistant already started")

        @self._tts.on("metrics_collected")
        def _on_tts_metrics(tts_metrics: metrics.TTSMetrics) -> None:
            speech_data = SpeechDataContextVar.get(None)
            if speech_data is None:
                return

            self.emit(
                "metrics_collected",
                metrics.PipelineTTSMetrics(
                    **tts_metrics.__dict__,
                    sequence_id=speech_data.sequence_id,
                ),
            )

        @self._vad.on("metrics_collected")
        def _on_vad_metrics(vad_metrics: vad.VADMetrics) -> None:
            self.emit("metrics_collected", metrics.PipelineVADMetrics(**vad_metrics.__dict__))

        self._started = True

        room.on("participant_connected", self._on_participant_connected)
        self._room = room
        self._participant = participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        self._main_atask = asyncio.create_task(self._main_task())

    def on(self, event: EventTypes, callback: Callable[[Any], None] | None = None):
        """Register a callback for an event

        Args:
            event: the event to listen to (see EventTypes)
                - user_started_speaking: the user started speaking
                - user_stopped_speaking: the user stopped speaking
                - agent_started_speaking: the agent started speaking
                - agent_stopped_speaking: the agent stopped speaking
                - user_speech_committed: the user speech was committed to the chat context
                - agent_speech_committed: the agent speech was committed to the chat context
                - agent_speech_interrupted: the agent speech was interrupted
                - function_calls_collected: received the complete set of functions to be executed
                - function_calls_finished: all function calls have been completed
            callback: the callback to call when the event is emitted
        """
        return super().on(event, callback)

    async def say(
        self,
        source: str | LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        """
        Play a speech source through the voice assistant.

        Args:
            source: The source of the speech to play.
                It can be a string, an LLMStream, or an asynchronous iterable of strings.
            allow_interruptions: Whether to allow interruptions during the speech playback.
            add_to_chat_ctx: Whether to add the speech to the chat context.

        Returns:
            The speech handle for the speech that was played, can be used to
            wait for the speech to finish.
        """
        await self._track_published_fut

        call_ctx = None
        fnc_source: str | AsyncIterable[str] | None = None
        if add_to_chat_ctx:
            try:
                call_ctx = AgentCallContext.get_current()
            except LookupError:
                # no active call context, ignore
                pass
            else:
                if isinstance(source, LLMStream):
                    logger.warning("LLMStream will be ignored for function call chat context")
                elif isinstance(source, AsyncIterable):
                    source, fnc_source = utils.aio.itertools.tee(source, 2)  # type: ignore
                else:
                    fnc_source = source

        new_handle = SpeechHandle.create_assistant_speech(
            allow_interruptions=allow_interruptions, add_to_chat_ctx=add_to_chat_ctx
        )
        synthesis_handle = self._synthesize_agent_speech(new_handle.id, source)
        new_handle.initialize(source=source, synthesis_handle=synthesis_handle)

        if self._playing_speech and not self._playing_speech.nested_speech_finished:
            self._playing_speech.add_nested_speech(new_handle)
        else:
            self._add_speech_for_playout(new_handle)

        # add the speech to the function call context if needed
        if call_ctx is not None and fnc_source is not None:
            if isinstance(fnc_source, AsyncIterable):
                text = ""
                async for chunk in fnc_source:
                    text += chunk
            else:
                text = fnc_source

            call_ctx.add_extra_chat_message(ChatMessage.create(text=text, role="assistant"))
            logger.debug(
                "added speech to function call chat context",
                extra={"text": text},
            )

        return new_handle

    def _update_state(self, state: AgentState, delay: float = 0.0):
        """Set the current state of the agent"""

        @utils.log_exceptions(logger=logger)
        async def _run_task(delay: float) -> None:
            await asyncio.sleep(delay)

            if self._room.isconnected():
                await self._room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: state})

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_run_task(delay))

    async def aclose(self) -> None:
        """Close the voice assistant"""
        if not self._started:
            return

        self._room.off("participant_connected", self._on_participant_connected)
        await self._deferred_validation.aclose()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if self._human_input is not None:
            return

        self._link_participant(participant.identity)

    def _link_participant(self, identity: str) -> None:
        participant = self._room.remote_participants.get(identity)
        if participant is None:
            logger.error("_link_participant must be called with a valid identity")
            return

        self._human_input = HumanInput(
            room=self._room,
            chat_ctx=self._chat_ctx,
            vad=self._vad,
            participant=participant,
            transcription=self._opts.transcription.user_transcription,
            model_fn=self.model_fn,
        )

        def _on_start_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_started_speaking")
            self.emit("user_started_speaking")
            self._deferred_validation.on_human_start_of_speech(ev)

        def _on_vad_inference_done(ev: vad.VADEvent) -> None:
            if not self._track_published_fut.done():
                return

            assert self._agent_output is not None

            tv = 1.0
            if self._opts.allow_interruptions:
                tv = max(0.0, 1.0 - ev.probability)
                self._agent_output.playout.target_volume = tv

            smoothed_tv = self._agent_output.playout.smoothed_volume

            self._plotter.plot_value("raw_vol", tv)
            self._plotter.plot_value("smoothed_vol", smoothed_tv)
            self._plotter.plot_value("vad_probability", ev.probability)

            if ev.speech_duration >= self._opts.int_speech_duration:
                self._interrupt_if_possible()

            if ev.raw_accumulated_speech > 0.0:
                self._last_speech_time = time.perf_counter() - ev.raw_accumulated_silence

        def _on_end_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_stopped_speaking")
            self.emit("user_stopped_speaking")
            self._deferred_validation.on_human_end_of_speech(ev)

        def _on_interim_response(ev) -> None:
            self._response_text = ev.alternatives[0].text
            if self._opts.preemptive_synthesis:
                if self._playing_speech is None or self._playing_speech.allow_interruptions:
                    self._synthesize_agent_reply()

        def _on_final_response(ev) -> None:
            response = ev.alternatives[0].text
            if not response:
                return

            self._last_final_response_time = time.perf_counter()

            # Store placeholder for user's speech
            user_msg = ChatMessage.create(text="[PLACEHOLDER]", role="user")
            self._chat_ctx.messages.append(user_msg)
            self.emit("user_speech_committed", user_msg)

            # Store the actual response
            self._response_text = response

            if self._playing_speech is None or self._playing_speech.allow_interruptions:
                self._synthesize_agent_reply()

            words = self._opts.transcription.word_tokenizer.tokenize(text=response)
            if len(words) >= 3:
                self._interrupt_if_possible()

            self._deferred_validation.on_human_final_transcript(response, ev.alternatives[0].language)

        self._human_input.on("start_of_speech", _on_start_of_speech)
        self._human_input.on("vad_inference_done", _on_vad_inference_done)
        self._human_input.on("end_of_speech", _on_end_of_speech)
        self._human_input.on("interim_response", _on_interim_response)
        self._human_input.on("final_response", _on_final_response)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        if self._opts.plotting:
            await self._plotter.start()

        self._update_state("initializing")
        audio_source = rtc.AudioSource(self._tts.sample_rate, self._tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", audio_source)
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        agent_playout = AgentPlayout(audio_source=audio_source)
        self._agent_output = AgentOutput(
            room=self._room,
            agent_playout=agent_playout,
            tts=self._tts,
        )

        def _on_playout_started() -> None:
            self._plotter.plot_event("agent_started_speaking")
            self.emit("agent_started_speaking")
            self._update_state("speaking")

        def _on_playout_stopped(interrupted: bool) -> None:
            self._plotter.plot_event("agent_stopped_speaking")
            self.emit("agent_stopped_speaking")
            self._update_state("listening")

        agent_playout.on("playout_started", _on_playout_started)
        agent_playout.on("playout_stopped", _on_playout_stopped)

        self._track_published_fut.set_result(None)

        while True:
            await self._speech_q_changed.wait()

            while self._speech_q:
                speech = self._speech_q[0]
                self._playing_speech = speech
                await self._play_speech(speech)
                self._speech_q.pop(0)
                self._playing_speech = None

            self._speech_q_changed.clear()

    def _synthesize_agent_reply(self) -> None:
        if self._pending_agent_reply is not None:
            self._pending_agent_reply.cancel()

        if self._human_input is not None and not self._human_input.speaking:
            self._update_state("thinking", 0.2)

        self._pending_agent_reply = new_handle = SpeechHandle.create_assistant_reply(
            allow_interruptions=self._opts.allow_interruptions,
            add_to_chat_ctx=True,
            user_question=self._response_text,
        )

        synthesis_handle = self._synthesize_agent_speech(new_handle.id, self._response_text)
        new_handle.initialize(source=self._response_text, synthesis_handle=synthesis_handle)
        self._add_speech_for_playout(new_handle)
        self._pending_agent_reply = None
        self._response_text = ""

    def _synthesize_agent_speech(
        self,
        speech_id: str,
        source: str | AsyncIterable[str],
    ) -> SynthesisHandle:
        assert self._agent_output is not None

        tts_source = self._opts.before_tts_cb(self, source)
        if tts_source is None:
            raise ValueError("before_tts_cb must return str or AsyncIterable[str]")

        return self._agent_output.synthesize(
            speech_id=speech_id,
            tts_source=tts_source,
            transcript_source=source,
            transcription=self._opts.transcription.agent_transcription,
            transcription_speed=self._opts.transcription.agent_transcription_speed,
            sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
            word_tokenizer=self._opts.transcription.word_tokenizer,
            hyphenate_word=self._opts.transcription.hyphenate_word,
        )

    async def _play_speech(self, speech_handle: SpeechHandle) -> None:
        try:
            await speech_handle.wait_for_initialization()
        except asyncio.CancelledError:
            return

        await self._agent_publication.wait_for_subscription()
        synthesis_handle = speech_handle.synthesis_handle

        if synthesis_handle.interrupted:
            return

        play_handle = synthesis_handle.play()
        join_fut = play_handle.join()

        collected_text = speech_handle.synthesis_handle.tts_forwarder.played_text
        interrupted = speech_handle.interrupted

        if collected_text and speech_handle.add_to_chat_ctx:
            if interrupted:
                collected_text += "..."

            msg = ChatMessage.create(text=collected_text, role="assistant")
            self._chat_ctx.messages.append(msg)

            if interrupted:
                self.emit("agent_speech_interrupted", msg)
            else:
                self.emit("agent_speech_committed", msg)

            logger.debug(
                "committed agent speech",
                extra={
                    "agent_response": collected_text,
                    "interrupted": interrupted,
                    "speech_id": speech_handle.id,
                },
            )

        # Wait for playback to complete
        while not join_fut.done():
            await asyncio.sleep(0.2)
            if speech_handle.interrupted:
                break

        speech_handle._set_done()

    def _add_speech_for_playout(self, speech_handle: SpeechHandle) -> None:
        self._speech_q.append(speech_handle)
        self._speech_q_changed.set()

    def _interrupt_if_possible(self) -> None:
        """Check whether the current assistant speech should be interrupted"""
        if self._playing_speech and self._should_interrupt():
            self._playing_speech.interrupt()

    def _should_interrupt(self) -> bool:
        if self._playing_speech is None:
            return True

        if not self._playing_speech.allow_interruptions or self._playing_speech.interrupted:
            return False

        if self._opts.int_min_words != 0:
            text = self._response_text
            interim_words = self._opts.transcription.word_tokenizer.tokenize(text=text)
            if len(interim_words) < self._opts.int_min_words:
                return False

        return True

    def _validate_reply_if_possible(self) -> None:
        """Check if the new agent speech should be played"""

        if self._playing_speech is not None:
            should_ignore_input = False
            if not self._playing_speech.allow_interruptions:
                should_ignore_input = True
                logger.debug(
                    "skipping validation, agent is speaking and does not allow interruptions",
                    extra={"speech_id": self._playing_speech.id},
                )
            elif not self._should_interrupt():
                should_ignore_input = True
                logger.debug(
                    "interrupt threshold is not met",
                    extra={"speech_id": self._playing_speech.id},
                )
            if should_ignore_input:
                self._response_text = ""
                return

        if self._pending_agent_reply is None:
            if self._opts.preemptive_synthesis or not self._response_text:
                return

            self._synthesize_agent_reply()

        assert self._pending_agent_reply is not None

        logger.debug(
            "validated agent reply",
            extra={"speech_id": self._pending_agent_reply.id},
        )

        if self._last_speech_time is not None:
            time_since_last_speech = time.perf_counter() - self._last_speech_time
            transcription_delay = max((self._last_final_response_time or 0) - self._last_speech_time, 0)

            eou_metrics = metrics.PipelineEOUMetrics(
                timestamp=time.time(),
                sequence_id=self._pending_agent_reply.id,
                end_of_utterance_delay=time_since_last_speech,
                transcription_delay=transcription_delay,
            )
            self.emit("metrics_collected", eou_metrics)

        self._add_speech_for_playout(self._pending_agent_reply)
        self._pending_agent_reply = None
        self._response_text = ""

    def _update_state(self, state: AgentState, delay: float = 0.0):
        """Set the current state of the agent"""

        @utils.log_exceptions(logger=logger)
        async def _run_task(delay: float) -> None:
            await asyncio.sleep(delay)

            if self._room.isconnected():
                await self._room.local_participant.set_attributes({ATTRIBUTE_AGENT_STATE: state})

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_run_task(delay))

    async def aclose(self) -> None:
        """Close the voice assistant"""
        if not self._started:
            return

        self._started = False
        self._closed = True

        if self._human_input is not None:
            await self._human_input.aclose()

        if self._agent_output is not None:
            await self._agent_output.aclose()

        if self._plotter is not None:
            await self._plotter.terminate()

        await self._deferred_validation.aclose()

        self._room.off("participant_connected", self._on_participant_connected)

        if self._playing_speech:
            self._playing_speech.interrupt()


class _DeferredReplyValidation:
    """This class is used to try to find the best time to validate the agent reply."""

    # if the STT gives us punctuation, we can try validate the reply faster.
    PUNCTUATION = ".!?"
    PUNCTUATION_REDUCE_FACTOR = 0.75

    # Long delay to use when the model thinks the user is still speaking
    UNLIKELY_ENDPOINT_DELAY = 6

    FINAL_TRANSCRIPT_TIMEOUT = 5

    def __init__(
        self,
        validate_fnc: Callable[[], None],
        min_endpointing_delay: float,
        turn_detector: _TurnDetector | None,
        agent: VoicePipelineAgent,
    ) -> None:
        self._turn_detector = turn_detector
        self._validate_fnc = validate_fnc
        self._validating_task: asyncio.Task | None = None
        self._last_final_transcript: str = ""
        self._last_language: str | None = None
        self._last_recv_start_of_speech_time: float = 0.0
        self._last_recv_end_of_speech_time: float = 0.0
        self._last_recv_transcript_time: float = 0.0
        self._speaking = False

        self._agent = agent
        self._end_of_speech_delay = min_endpointing_delay

    @property
    def validating(self) -> bool:
        return self._validating_task is not None and not self._validating_task.done()

    def _compute_delay(self) -> float | None:
        """Computes the amount of time to wait before validating the agent reply.

        This function should be called after the agent has received final transcript, or after VAD
        """
        # never interrupt the user while they are speaking
        if self._speaking:
            return None

        # if STT doesn't give us the final transcript after end of speech, we'll still validate the reply
        # to prevent the agent from getting "stuck"
        # in this case, the agent will not have final transcript, so it'll trigger the user input with empty
        if not self._last_final_transcript:
            return self.FINAL_TRANSCRIPT_TIMEOUT

        delay = self._end_of_speech_delay
        if self._end_with_punctuation():
            delay = delay * self.PUNCTUATION_REDUCE_FACTOR

        # the delay should be computed from end of earlier timestamp, that's the true end of user speech
        end_of_speech_time = self._last_recv_end_of_speech_time
        if (
            self._last_recv_transcript_time > 0
            and self._last_recv_transcript_time > self._last_recv_start_of_speech_time
            and self._last_recv_transcript_time < end_of_speech_time
        ):
            end_of_speech_time = self._last_recv_transcript_time

        elapsed_time = time.perf_counter() - end_of_speech_time
        if elapsed_time < delay:
            delay -= elapsed_time
        else:
            delay = 0
        return delay

    def on_human_final_transcript(self, transcript: str, language: str | None) -> None:
        self._last_final_transcript += " " + transcript.strip()
        self._last_language = language
        self._last_recv_transcript_time = time.perf_counter()

        delay = self._compute_delay()
        if delay is not None:
            self._run(delay)

    def on_human_start_of_speech(self, ev: vad.VADEvent) -> None:
        self._speaking = True
        self._last_recv_start_of_speech_time = time.perf_counter()
        if self.validating:
            assert self._validating_task is not None
            self._validating_task.cancel()

    def on_human_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._speaking = False
        self._last_recv_end_of_speech_time = time.perf_counter()

        delay = self._compute_delay()
        if delay is not None:
            self._run(delay)

    def _end_with_punctuation(self) -> bool:
        return len(self._last_final_transcript) > 0 and self._last_final_transcript[-1] in self.PUNCTUATION

    def _reset_states(self) -> None:
        self._last_final_transcript = ""
        self._last_recv_end_of_speech_time = 0.0
        self._last_recv_transcript_time = 0.0

    def _run(self, delay: float) -> None:
        @utils.log_exceptions(logger=logger)
        async def _run_task(chat_ctx: ChatContext, delay: float) -> None:
            use_turn_detector = self._last_final_transcript and not self._speaking
            if (
                use_turn_detector
                and self._turn_detector is not None
                and self._turn_detector.supports_language(self._last_language)
            ):
                start_time = time.perf_counter()
                eot_prob = await self._turn_detector.predict_end_of_turn(chat_ctx)
                unlikely_threshold = self._turn_detector.unlikely_threshold()
                elasped = time.perf_counter() - start_time
                if eot_prob < unlikely_threshold:
                    delay = self.UNLIKELY_ENDPOINT_DELAY
                delay = max(0, delay - elasped)
            await asyncio.sleep(delay)

            self._reset_states()
            self._validate_fnc()

        if self._validating_task is not None:
            self._validating_task.cancel()

        detect_ctx = self._agent.chat_ctx.copy()
        detect_ctx.messages.append(ChatMessage.create(text=self._agent._response_text, role="user"))
        self._validating_task = asyncio.create_task(_run_task(detect_ctx, delay))

    async def aclose(self) -> None:
        if self._validating_task is not None:
            self._validating_task.cancel()
