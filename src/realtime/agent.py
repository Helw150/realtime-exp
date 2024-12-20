# agent.py
import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import cartesia, silero

from realtime.vllm import VLLM, VLLMConfig

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_CONFIGS = {
    "diva": VLLMConfig(
        model_id="WillHeld/DiVA-llama-3-v0-8b",
        base_url="https://ov228wydp5d4u7-40021.proxy.runpod.net/v1",
        prompt="You are DiVA, a voice assistant. Keep responses under 20 words.",
    ),
}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for first participant
    participant = await ctx.wait_for_participant()

    # Get model config from environment or default to DiVA
    model_name = os.getenv("VLLM_MODEL", "diva")
    model_config = MODEL_CONFIGS[model_name]
    logger.info(f"starting {model_name} for participant {participant.identity}")

    # Initialize TTS
    tts = cartesia.TTS(voice="ee7ea9f8-c0c1-498c-9279-764d6b56d189")
    audio_source = rtc.AudioSource(24000, 1)

    # Create and publish audio track
    track = rtc.LocalAudioTrack.create_audio_track("vllm-voice", audio_source)
    await ctx.room.local_participant.publish_track(track)

    async def synthesize_speech(text: str):
        stream = tts.stream()
        stream.push_text(text)

        async for event in stream:
            await audio_source.capture_frame(event.data)

        await stream.aclose()

    # Initialize VLLM
    vllm = VLLM(
        ctx=ctx,
        vad=ctx.proc.userdata["vad"],
        on_response=lambda text: asyncio.create_task(synthesize_speech(text)),
        config=model_config,
    )

    # Start VLLM
    await vllm.start()

    # Handle text chat
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        response = await vllm.generate_text_response(txt)
        await chat.send_message(response)
        await synthesize_speech(response)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    # Initial greeting
    await synthesize_speech("Hey, how can I help you today?")

    # Wait for disconnect
    await ctx.wait_for_disconnect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
