import asyncio
import io
import logging
import os
from dataclasses import dataclass

import aiohttp
import numpy as np


class STTStream:
    @dataclass
    class StartedEvent:
        type: str = "started"

    @dataclass
    class FinishedEvent:
        text: str
        type: str = "finished"

    def __init__(self) -> None:
        super().__init__()
        self._api_key = os.environ["HUGGINGFACE_API_KEY"]
        self._audio_buffer = np.array([], dtype=np.int16)
        self._event_queue = asyncio.Queue()
        self._closed = False
        self._main_task = asyncio.create_task(self._run())

    def push_frame(self, frame: "rtc.AudioFrame") -> None:
        if self._closed:
            raise ValueError("cannot push frame to closed stream")

        resampled = frame.remix_and_resample(16000, 1)
        frame_data = np.frombuffer(resampled.data, dtype=np.int16)
        self._audio_buffer = np.append(self._audio_buffer, frame_data)

    async def aclose(self) -> None:
        self._closed = True
        await self._main_task

    async def _run(self) -> None:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        await self._event_queue.put(self.StartedEvent())

        while not self._closed:
            await asyncio.sleep(0.1)

        # Process all collected audio
        if len(self._audio_buffer) > 0:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(API_URL, headers=headers, data=self._audio_buffer.tobytes()) as response:
                        result = await response.json()
                        if "text" in result:
                            text = result["text"].strip()
                            if text:
                                await self._event_queue.put(self.FinishedEvent(text=text))
                        else:
                            logging.error(f"Unexpected API response: {result}")
            except Exception as e:
                logging.error(f"Error processing audio: {e}")

    def __aiter__(self) -> "STTStream":
        return self

    async def __anext__(self):
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
