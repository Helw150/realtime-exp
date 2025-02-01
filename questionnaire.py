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

# Set up logging
logger = logging.getLogger("questionnaire")
logging.basicConfig(level=logging.INFO)


async def ask_questions(agent: VoicePipelineAgent, questions: list[str]) -> None:

    user_stopped_event = asyncio.Event()

    def on_user_stopped_speaking(*args, **kwargs):
        print("Detected that the user has stopped speaking.")
        user_stopped_event.set()

    agent.on("user_stopped_speaking", on_user_stopped_speaking)

    for question in questions:
        # clear the event for each new question.
        user_stopped_event.clear()
        print(f"asking question: {question}")

        await agent.say(question, allow_interruptions=True)
        print("question finished playing. waiting for user resp...")

        await user_stopped_event.wait()
        print("-------------- x user finished responding moving to the next question.\n")


async def entrypoint(ctx):

    logger.info("Connecting to room %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    logger.info("Participant %s joined", participant.identity)

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        model_fn=ctx.proc.userdata["model_fn"],
        tts=cartesia.TTS(
            model="sonic-english",
            language="en",
            sample_rate=24000,
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab", 
            encoding="pcm_s16le",
        ),
        chat_ctx=None,
        allow_interruptions=False,
        preemptive_synthesis=False,
        plotting=False,
    )


    agent.start(ctx.room, participant)
    await asyncio.sleep(1) 

    questions = [
        "What is your name?",
        "How old are you?",
        "Where do you live?",
        "Where is Stanford University located?"
    ]

    # question loop
    await ask_questions(agent, questions)

    await agent.say("Thank you for answering all the questions Goodbye", allow_interruptions=False)
    await asyncio.sleep(2)
    print("all questions completed")
    import sys
    sys.exit(0)


def prewarm(proc: JobProcess):

    vad = silero.VAD.load(
        min_speech_duration=0.3,
        min_silence_duration=2,
        prefix_padding_duration=0.5,
        max_buffered_speech=60.0
    )
    proc.userdata["vad"] = vad

    def save_response_model_fn(audio: np.ndarray, chat_history: List[ChatMessage]):
        """
        function is called with the user audio when Silero vad detects a completed utterance.
        normalizes and trims the audio, then saves it as a wav file.
        """
        if audio is not None and len(audio) > 0:
            y = audio.astype(np.float32)
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            threshold = 0.01  
            indices = np.where(np.abs(y) > threshold)[0]
            if indices.size > 0:
                y = y[indices[0]: indices[-1] + 1]

            import time
            filename = f"user_response_{int(time.time())}.wav"
            sf.write(filename, y, 16000, format="wav", subtype="PCM_16")
            print(f"Saved user response to {filename}")

        return iter([])

    proc.userdata["model_fn"] = save_response_model_fn


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )