import asyncio
import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli, llm, metrics
from livekit.plugins import cartesia, silero

from realtime.diva import DiVA


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Initialize chat context
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "Your name is DiVA, which stands for Distilled Voice Assistant. "
            "You were trained with early-fusion training to merge OpenAI's Whisper and Meta AI's Llama 3 8B "
            "to provide end-to-end voice processing. Keep responses concise and under 20 words."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for first participant
    participant = await ctx.wait_for_participant()
    logger.info(f"starting DiVA for participant {participant.identity}")

    # Initialize TTS
    tts = cartesia.TTS(model_name="ee7ea9f8-c0c1-498c-9279-764d6b56d189", sample_rate=24000)
    audio_source = rtc.AudioSource(24000, 1)

    # Create and publish audio track
    track = rtc.LocalAudioTrack.create_audio_track("diva-voice", audio_source)
    await ctx.room.local_participant.publish_track(track)

    async def synthesize_speech(text: str):
        stream = tts.stream()
        stream.push_text(text)

        async for event in stream:
            await audio_source.capture_frame(event.data)

        await stream.aclose()

    # Initialize DiVA
    diva = DiVA(
        ctx=ctx, vad=ctx.proc.userdata["vad"], on_response=lambda text: ctx.create_task(synthesize_speech(text))
    )

    # Start DiVA
    await diva.start()

    # Set up metrics collection
    usage_collector = metrics.UsageCollector()

    @diva.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    # Handle text chat
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        response = await diva.generate_text_response(txt)
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
