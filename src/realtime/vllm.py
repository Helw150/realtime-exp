# VLLM.py
import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from datasets import Audio
from livekit import agents, rtc
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class VLLMConfig:
    model_id: str
    base_url: str = "http://localhost:8003/v1"
    api_key: str = "empty"
    sample_rate: int = 16000
    min_audio_len: int = 8000  # Minimum audio length to process (0.5s at 16kHz)
    prompt: str = "You are a helpful voice assistant. Respond conversationally to the speech provided."


@dataclass
class VLLMState:
    conversation: list = field(default_factory=list)
    model_outs: any = None


class VLLM:
    def __init__(
        self,
        ctx: agents.JobContext,
        vad: any,
        on_response: Callable[[str], None],
        config: VLLMConfig,
    ):
        """Initialize Voice LLM agent

        Args:
            ctx: LiveKit job context
            vad: Voice Activity Detection component
            on_response: Callback function for handling responses
            config: Configuration for the VLLM instance
        """
        self.ctx = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.on_response = on_response
        self.vad = vad
        self.config = config

        # Initialize API client and audio components
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        self.resampler = Audio(sampling_rate=config.sample_rate)

        # Instance state
        self.state = VLLMState()
        self._current_state = AgentState.LISTENING
        self._audio_buffer = []
        self._is_speaking = False

        # Setup handlers
        self.ctx.room.on("track_subscribed", self._handle_track)

    def _update_state(self, state: AgentState):
        """Update agent state and metadata"""
        self._current_state = state
        asyncio.create_task(self.ctx.room.local_participant.set_attributes({"ATTRIBUTE_AGENT_STATE": state.value}))

    def set_speaking(self, is_speaking: bool):
        """Update speaking state"""
        if is_speaking:
            self._update_state(AgentState.SPEAKING)
        else:
            self._update_state(AgentState.LISTENING)

    async def start(self):
        """Start the VLLM instance"""
        self._update_state(AgentState.LISTENING)

    def _handle_track(self, track: rtc.Track, publication, participant):
        """Handle new audio tracks"""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(self._process_audio(track))

    async def _process_audio(self, track: rtc.Track):
        """Process incoming audio"""
        audio_stream = rtc.AudioStream(track)
        vad_stream = self.vad.stream()

        async for audio_event in audio_stream:
            if self._current_state != AgentState.LISTENING:
                continue

            frame = audio_event.frame
            vad_stream.push_frame(frame)
            arr = np.frombuffer(frame.data, dtype=np.int16)
            float_arr = arr.astype(np.float32) / 32768.0
            self._audio_buffer.extend(float_arr)

            async for ev in vad_stream:
                if ev.type == voice_activity_detection.VADEventType.START_OF_SPEECH:
                    self._is_speaking = True
                    self._audio_buffer = []
                elif ev.type == voice_activity_detection.VADEventType.INFERENCE_DONE:
                    self._speech_probability = ev.probability
                elif ev.type == voice_activity_detection.VADEventType.END_OF_SPEECH:
                    self._speaking = False
                    await self._generate_response(self._audio_buffer)
                    self._is_speaking = False
                    self._audio_buffer = []

    async def _generate_response(self, audio_data):
        """Generate response using API streaming"""
        self._update_state(AgentState.THINKING)

        try:
            # Process audio
            sr = self.config.sample_rate
            audio_tuple = (sr, audio_data)

            get_response, _ = api_streaming(self.config.model_id)

            complete_response = ""
            async for response in get_response(audio_tuple):
                complete_response = response

            if complete_response:
                # Send chat message
                await self.chat.send_message(complete_response)
                # Trigger TTS
                self.on_response(complete_response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self._update_state(AgentState.LISTENING)

    async def generate_text_response(self, text: str) -> str:
        """Generate response from text input"""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": self.config.prompt,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            return "I apologize, I'm having trouble processing your request."
