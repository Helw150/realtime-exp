import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
import torch
from datasets import Audio
from livekit import agents, rtc
from livekit.plugins.silero import VAD
from transformers import AutoModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


PROMPT = """Your name is DiVA. You are a voice assistant. Keep responses under 20 words."""


@dataclass
class DiVAState:
    conversation: list = field(default_factory=list)
    model_outs: any = None


class DiVA:
    def __init__(self, ctx: agents.JobContext, on_response: Callable[[str], None]):
        self.ctx = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.on_response = on_response

        # Initialize components
        self.diva_model = AutoModel.from_pretrained("WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True)
        self.resampler = Audio(sampling_rate=16_000)
        self.vad = VAD(threshold=0.5)

        # State
        self.state = DiVAState()
        self._current_state = AgentState.LISTENING
        self._audio_buffer = []
        self._is_speaking = False

        # Setup handlers
        self.ctx.room.on("track_subscribed", self._handle_track)

    def _update_state(self, state: AgentState):
        """Update agent state and metadata"""
        self._current_state = state
        metadata = json.dumps({"agent_state": state.value})
        self.ctx.create_task(self.ctx.room.local_participant.update_metadata(metadata))

    def set_speaking(self, is_speaking: bool):
        """Update speaking state from TTS manager"""
        if is_speaking:
            self._update_state(AgentState.SPEAKING)
        else:
            self._update_state(AgentState.LISTENING)

    async def start(self):
        """Start the DiVA agent"""
        self._update_state(AgentState.LISTENING)

    def _handle_track(self, track: rtc.Track, publication, participant):
        """Handle new audio tracks"""
        if track.kind == rtc.TrackKind.AUDIO:
            self.ctx.create_task(self._process_audio(track))

    @torch.no_grad()
    async def _process_audio(self, track: rtc.Track):
        """Process incoming audio"""
        audio_stream = rtc.AudioStream(track)
        vad_stream = self.vad.stream()

        async for audio_event in audio_stream:
            if self._current_state != AgentState.LISTENING:
                continue

            frame = audio_event.frame
            vad_stream.push_frame(frame)

            if vad_stream.is_speech:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._audio_buffer = []
                self._audio_buffer.extend(frame.data.flatten())
            elif self._is_speaking:
                # Speech ended, process if we have enough audio
                if len(self._audio_buffer) > 8000:  # 0.5s at 16kHz
                    await self._generate_response(np.array(self._audio_buffer))
                self._is_speaking = False
                self._audio_buffer = []

    async def _generate_response(self, audio_data: np.ndarray):
        """Generate and process response"""
        self._update_state(AgentState.THINKING)

        # Normalize and process audio
        audio_data = audio_data.astype(np.float32)
        audio_data /= np.max(np.abs(audio_data))
        audio_processed = self.resampler.decode_example({"array": audio_data, "sampling_rate": 16000})

        # Generate response
        complete_response = ""
        async for response, outs in self.diva_model.generate_stream(
            audio_processed["array"],
            PROMPT if self.state.model_outs is None else None,
            max_new_tokens=256,
            init_outputs=self.state.model_outs,
            return_outputs=True,
        ):
            if response:
                complete_response = response
                self.state.model_outs = outs

        if complete_response:
            # Send chat message
            await self.chat.send_message(complete_response)
            # Trigger TTS
            self.on_response(complete_response)
