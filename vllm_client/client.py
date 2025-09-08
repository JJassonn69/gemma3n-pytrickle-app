"""
OpenAI-compatible client for vLLM with multimodal (text + images + audio) support.

Usage:
    from vllm.client import VLLMClient
    client = VLLMClient()  # reads VLLM_BASE_URL and MODEL_ID from environment if present
    text = client.chat(prompt="Describe the scene.", images=[pil_img], audio=audio_np, audio_sr=16000)

Environment variables:
    VLLM_BASE_URL: Base URL to the vLLM OpenAI-compatible API (e.g., http://vllm:9000/v1)
    MODEL_ID: Model id (default: google/gemma-3n-E4B-it)
    VLLM_API_KEY: token; vLLM typically ignores, default is "EMPTY"
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Generator, Iterable, List, Optional

import numpy as np
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)


class VLLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:9000/v1")
        self.model_id = model_id or os.getenv("MODEL_ID", "google/gemma-3n-E2B-it")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    # ---------- Encoding helpers ----------
    @staticmethod
    def _img_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=quality)
        mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _audio_to_b64_wav(audio: np.ndarray, sr: int = 16000) -> str:
        # Ensure mono float32 [-1, 1]
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)

        # Clip to [-1, 1] before writing PCM_16
        audio = np.clip(audio, -1.0, 1.0)
        
        # Ensure minimum duration for Gemma 3n (at least 1 second to avoid kernel size issues)
        min_samples = sr  # 1 second
        if len(audio) < min_samples:
            # Pad with zeros to meet minimum duration
            padding = min_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
            logger.info(f"Padded audio from {len(audio) - padding} to {len(audio)} samples for Gemma 3n compatibility")
        
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ---------- Message builder ----------
    def _build_content(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        audio: Optional[np.ndarray] = None,
        audio_sr: int = 16000,
    ) -> List[dict]:
        content: List[dict] = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._img_to_data_url(img)}
                })
        if audio is not None and len(audio) > 0:
            content.append({
                "type": "input_audio",
                "input_audio": {
                    "data": self._audio_to_b64_wav(audio, audio_sr),
                    "format": "wav",
                },
            })
        return content

    # ---------- Chat completion ----------
    def chat(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        audio: Optional[np.ndarray] = None,
        audio_sr: int = 16000,
        max_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Call vLLM Chat Completions API.

        When stream=False, returns a full string response.
        When stream=True, returns a generator yielding token deltas.
        """
        content = self._build_content(prompt, images, audio, audio_sr)
        kwargs = dict(
            model=self.model_id,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )

        if stream:
            def _gen() -> Generator[str, None, None]:
                response = self.client.chat.completions.create(**kwargs)
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return _gen()
        else:
            resp = self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content


def get_client() -> VLLMClient:
    """Factory reading from environment."""
    return VLLMClient()
