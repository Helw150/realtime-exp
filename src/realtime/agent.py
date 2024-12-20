import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.plugins.cartesia import TTS

from realtime.diva import DiVA

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TTSManager:
    """Handles text-to-speech synthesis and audio output"""

    audio_out: rtc.AudioSource
    tts: TTS
    on_speaking: Optional[Callable] = None
    on_finished: Optional[Callable] = None

    @classmethod
    def create(cls, sample_rate: int = 24000, channels: int = 1):
        audio_out = rtc.AudioSource(sample_rate, channels)
        tts = TTS(model_name="ee7ea9f8-c0c1-498c-9279-764d6b56d189", sample_rate=sample_rate)
        return cls(audio_out=audio_out, tts=tts)

    async def speak(self, text: str):
        """Convert text to speech and output audio"""
        if self.on_speaking:
            self.on_speaking()

        stream = self.tts.stream()
        stream.push_text(text)

        async for event in stream:
            # TTS stream directly provides audio data
            await self.audio_out.capture_frame(event.data)

        await stream.aclose()

        if self.on_finished:
            self.on_finished()


async def setup_diva(ctx: agents.JobContext) -> tuple[DiVA, TTSManager]:
    """Initialize DiVA and TTS components"""
    # Create TTS manager
    tts_manager = TTSManager.create()

    # Create DiVA with TTS callback
    diva = DiVA(ctx=ctx, on_response=lambda text: ctx.create_task(tts_manager.speak(text)))

    # Connect TTS state changes to DiVA
    tts_manager.on_speaking = lambda: diva.set_speaking(True)
    tts_manager.on_finished = lambda: diva.set_speaking(False)

    # Setup audio output
    track = rtc.LocalAudioTrack.create_audio_track("diva-mic", tts_manager.audio_out)
    await ctx.room.local_participant.publish_track(track)

    return diva, tts_manager


async def run_diva(ctx: agents.JobContext):
    """Run DiVA agent with TTS"""
    try:
        diva, _ = await setup_diva(ctx)
        await diva.start()
        await ctx.wait_for_disconnect()
    except Exception as e:
        logger.error(f"Error in DiVA agent: {e}")


async def handle_job_request(job_request: agents.JobRequest):
    """Handle incoming job requests"""
    logger.info("Accepting job request")
    await job_request.accept(
        run_diva,
        identity="diva_agent",
        name="DiVA",
        auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
    )


async def main():
    """Main entry point"""
    # Get credentials from environment variables
    api_key = os.getenv("LIVEKIT_API_KEY")
    ws_url = os.getenv("LIVEKIT_URL")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([api_key, ws_url, api_secret]):
        raise ValueError(
            "LIVEKIT_API_KEY, LIVEKIT_WS_URL, and LIVEKIT_API_SECRET must be set in environment variables"
        )

    worker_opts = agents.WorkerOptions(
        entrypoint_fnc=handle_job_request,
        ws_url=ws_url,
        api_key=api_key,
        api_secret=api_secret,
    )
    worker = agents.Worker(opts=worker_opts)

    # Run the worker
    try:
        await worker.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
