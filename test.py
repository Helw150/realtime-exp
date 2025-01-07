import asyncio
import base64
import logging
from collections import deque
from typing import AsyncIterable, Iterable, List

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, metrics
from livekit.plugins import cartesia, silero

import realtime.streaming_helpers as sh
from realtime.pipeline_agent import ChatContext, ChatMessage, VoicePipelineAgent


load_dotenv()
logger = logging.getLogger("voice-assistant")


def prewarm(proc: JobProcess):
    """Pre-warm any models or resources needed"""
    proc.userdata["vad"] = silero.VAD.load()
    streaming_fn, _ = sh.api_streaming("WillHeld/DiVA-llama-3-v0-8b")

    async def timed_string_accumulator(resps: AsyncIterable[str], interval: float = 0.2) -> AsyncIterable[str]:
        last_emit = None
        async for resp in resps:
            if last_emit is None or asyncio.get_event_loop().time() - last_emit >= interval:
                last_emit = asyncio.get_event_loop().time()
                yield resp
            final_resp = resp
        yield final_resp

    # Example unified model function that combines STT and LLM
    async def unified_model_fn(audio_array: np.ndarray, chat_history: List[ChatMessage]) -> AsyncIterable[str]:
        messages = []
        for message in chat_history[:-1]:
            if message.text is not None:
                content = [{"type": "text", "text": message.text}]
            elif message.audio is not None:
                audio = message.audio
                y = audio.astype(np.float32)
                y /= np.max(np.abs(y))
                sf.write(f"tmp.wav", y, 16000, format="wav")
                with open(f"tmp.wav", "rb") as wav_file:
                    wav_data = wav_file.read()
                encoded_string = base64.b64encode(wav_data).decode("utf-8")
                content = [{"type": "audio", "audio_url": "data:audio/wav;base64," + encoded_string}]
            messages.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
        # messages = [messages[0]]  # Temporary Single Turn Override
        prev_seen = ""
        async for resp in timed_string_accumulator(streaming_fn((16000, audio_array), messages)):
            yield resp[len(prev_seen) :]
            prev_seen = resp

    proc.userdata["model_fn"] = unified_model_fn


async def entrypoint(ctx: JobContext):
    """Main entry point for the voice assistant"""
    initial_ctx = ChatContext()
    initial_ctx.messages.append(
        ChatMessage(
            role="system",
            text="You are a voice assistant. Your interface with users will be voice. You should tailor your response style to speech using spelled out numbers and avoiding all formatting except for periods, commas, exclamation marks, and question marks.",
        )
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for first participant
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Initialize the pipeline agent
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        model_fn=ctx.proc.userdata["model_fn"],
        tts=cartesia.TTS(
            model="sonic-english",  # Cartesia's default English TTS model
            language="en",
            sample_rate=24000,  # Cartesia's default sample rate
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",  # Default voice ID
            encoding="pcm_s16le",  # Required PCM format
        ),  # Using Cartesia's TTS implementation
        chat_ctx=initial_ctx,
        allow_interruptions=True,
        interrupt_speech_duration=0.5,
        min_endpointing_delay=0.5,
        preemptive_synthesis=True,
        plotting=False,
    )

    agent.start(ctx.room, participant)

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    # Optional: Handle text chat messages
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        await agent.say(txt, allow_interruptions=True)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    # Initial greeting
    # await agent.say("Testing 1 2 3", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
