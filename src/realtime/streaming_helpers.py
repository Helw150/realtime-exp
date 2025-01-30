import asyncio
import base64
import json
import os
from collections import defaultdict
from pathlib import Path

import google.generativeai as genai
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
import xxhash
from datasets import Audio
from openai import AsyncOpenAI
from transformers import AutoModel, AutoProcessor, Qwen2AudioForConditionalGeneration, TextIteratorStreamer
from transformers.generation import GenerationConfig


def _get_config_for_model_name(model_id):
    if "API_MODEL_CONFIG" in os.environ:
        return json.loads(os.environ["API_MODEL_CONFIG"])[model_id]
    return {
        "pipeline/meta-llama/Meta-Llama-3-8B-Instruct": {"base_url": "http://localhost:8001/v1", "api_key": "empty"},
        "scb10x/llama-3-typhoon-audio-8b-2411": {
            "base_url": "http://localhost:8002/v1",
            "api_key": "empty",
        },
        "WillHeld/DiVA-llama-3-v0-8b": {
            "base_url": "http://localhost:8003/v1",
            "api_key": "empty",
        },
        "Qwen/Qwen2-Audio-7B-Instruct": {
            "base_url": "http://localhost:8004/v1",
            "api_key": "empty",
        },
    }[model_id]


def api_streaming(model_id):
    client = AsyncOpenAI(**_get_config_for_model_name(model_id))
    resampler = Audio(sampling_rate=16_000)

    async def get_chat_response(history):
        if len(history) == 0:
            raise StopAsyncIteration("")
        try:
            completion = await client.chat.completions.create(
                model=model_id,
                messages=history,
                stream=True,
                temperature=0.0,
            )
            text_response = []
            async for chunk in completion:
                if len(chunk.choices) > 0:
                    text_response.append(chunk.choices[0].delta.content)
                    yield "".join(text_response)
        except Exception as e:
            print(e)
            print(f"error for {model_id}")
            raise StopAsyncIteration(f"error for {model_id}")

    return get_chat_response, client
