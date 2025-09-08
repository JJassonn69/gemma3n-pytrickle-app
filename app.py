import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import librosa
from PIL import Image

from pytrickle import StreamProcessor, VideoFrame, AudioFrame

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & State Management ---

@dataclass
class AppConfig:
    # Modality selection: one of 'text', 'vision', 'audio'
    modality: str = "vision"

    # Generation params
    vision_prompt: str = """
    You are an expert video analysis provider that balances detail and broad understanding of video provided.
    Describe the video frame in context of previous frames.  First sentence should be what changed, next list details of the frame to use for reference in next frame analysis.
    Do not include leading text like 'this image shows', 'this video depicts',
    """

    audio_prompt: str = "Transcribe the audio to the language it is spoken."
    generate_every_s: float = 5
    max_buffer_s: float = 8.0
    max_new_tokens: int = 300
    request_timeout_s: float = 8.0

    # Video sampling params
    target_fps: int = 7
    max_frames: int = 10
    enable_downscale: bool = True
    downscale_px: int = 512

    # Audio params
    audio_sample_rate: int = 16000
    audio_window_s: float = 4.0

    # App server
    server_port: int = 8000

    # Transcript correction settings (post-audio generation)
    correction_enabled: bool = True
    history_max_turns: int = 4
    correction_system_hint: str = (
        "You are an ASR post-processor. Use the provided history to correct the latest transcript. "
        "Fix homophones, merge split/partial words, and prefer likely phrases based on context. "
        "Return only the corrected latest chunk unless asked otherwise."
    )


@dataclass
class AppState:
    # VLLM client
    vllm_client: Any = None

    # In-memory rolling buffers
    video_frames: deque = field(default_factory=deque)  # deque[PIL.Image]
    audio_chunks: deque = field(default_factory=deque)  # deque[np.ndarray]
    audio_samples_total: int = 0  # total samples in audio_chunks at target rate
    buffer_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Timing & Control
    last_data_time: float = 0.0
    idle_timeout_s: float = 1.0
    watcher_task: asyncio.Task | None = None
    trigger_generation: bool = True
    last_video_time: float = 0.0

    # PyTrickle server reference
    server: Any = None
    processor: Any = None

    # Diagnostics
    last_audio_log_time: float = 0.0

    # Transcription history (raw, as produced by first pass)
    generation_history: deque = field(default_factory=deque)  # deque[str]

# Global instances
config = AppConfig()
state = AppState()


# --- Core Callbacks for StreamProcessor ---

def load_model(**kwargs):
    """Initialize VLLM client."""
    logger.info("Initializing VLLM client")
    try:
        from vllm_client import get_client as get_vllm_client
        state.vllm_client = get_vllm_client()
        logger.info(f"VLLM client initialized. Base URL: {os.getenv('VLLM_BASE_URL', 'http://localhost:9000/v1')}")
        logger.info("Worker ready, performing quick warmup call to vLLM...")
        
        try:
            green_img = Image.new("RGB", (224, 224), color=(0, 255, 0))
            warmup = state.vllm_client.chat(
                prompt="Warmup.",
                images=[green_img],
                audio=None,
                max_tokens=8,
                stream=False,
            )
            logger.info(f"Warmup with vLLM complete with result: {warmup}")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize VLLM client: {e}")
        raise RuntimeError("VLLM client is required for this application") from e


async def process_video(frame: VideoFrame) -> VideoFrame:
    """Sample frames to target FPS, keep as PIL in-memory, pass-through the original frame."""
    # Skip buffering if current modality is not vision
    if config.modality != 'vision':
        return frame
    now = time.time()
    if (now - state.last_video_time) < (1.0 / config.target_fps):
        return frame

    state.last_video_time = now

    if state.trigger_generation and (state.watcher_task is None or state.watcher_task.done()):
        # Start background watcher task on the SAME loop
        loop = asyncio.get_running_loop()
        state.watcher_task = loop.create_task(run_generation_loop())
        state.trigger_generation = False

    try:
        # Convert to PIL Image
        img_np = frame.tensor.squeeze(0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np, 'RGB')

        # Optional downscaling
        if config.enable_downscale:
            img.thumbnail((config.downscale_px, config.downscale_px))

        # Add to rolling buffer with capacity enforcement
        async with state.buffer_lock:
            while len(state.video_frames) >= config.max_frames:
                state.video_frames.popleft()
            state.video_frames.append(img)
            state.last_data_time = time.time()
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")

    return frame # Pass-through


async def process_audio(audioframe: AudioFrame) -> list[AudioFrame]:
    """Buffer mono 16k audio into a sliding window and pass-through the original frame."""
    # Skip buffering if current modality is not audio
    if config.modality != 'audio':
        return [audioframe]
    # Convert to mono and resample to target sample rate

    if state.trigger_generation and (state.watcher_task is None or state.watcher_task.done()):
        # Start background watcher task on the SAME loop
        loop = asyncio.get_running_loop()
        state.watcher_task = loop.create_task(run_generation_loop())
        state.trigger_generation = False

    try:
        samples = np.asarray(audioframe.samples)
        src_sr = None
        for attr in ('rate', 'sample_rate', 'sr'):
            src_sr = getattr(audioframe, attr, None) if src_sr is None else src_sr
        src_sr = src_sr or config.audio_sample_rate
       
        # Reduce to mono
        if samples.ndim == 2:
            # Try averaging across channels; handle (C,N) or (N,C)
            if samples.shape[0] == 1:
                samples = samples[0]
            elif samples.shape[1] == 1:
                samples = samples[:, 0]
            else:
                # Choose axis with more samples as time
                if samples.shape[0] < samples.shape[1]:
                    samples = samples.mean(axis=0)
                else:
                    samples = samples.mean(axis=1)
        elif samples.ndim > 2:
            samples = samples.reshape(-1)

        # Normalize to float32 in [-1, 1] if integer
        if np.issubdtype(samples.dtype, np.integer):
            max_abs = np.iinfo(samples.dtype).max
            samples = samples.astype(np.float32) / max_abs
        else:
            samples = samples.astype(np.float32)

        # Resample if needed
        if src_sr != config.audio_sample_rate:
            try:
                samples = librosa.resample(samples, orig_sr=src_sr, target_sr=config.audio_sample_rate)
            except Exception as e:
                logger.warning(f"Audio resample failed from {src_sr} to {config.audio_sample_rate}: {e}")
                # Fallback: continue with original samples

        async with state.buffer_lock:
            # Append resampled mono chunk
            state.audio_chunks.append(samples)
            state.audio_samples_total += samples.shape[0]

            # Enforce larger sliding window for overlap (changed from 4.0 to config.max_buffer_s)
            max_samples = int(config.audio_sample_rate * config.max_buffer_s)
            while state.audio_samples_total > max_samples and state.audio_chunks:
                removed = state.audio_chunks.popleft()
                state.audio_samples_total -= len(removed)

            # Always update last_data_time on each chunk to avoid false idles
            state.last_data_time = time.time()

            now = state.last_data_time
            if now - state.last_audio_log_time > float(config.audio_window_s):
                state.last_audio_log_time = now
                chunks = len(state.audio_chunks)
                secs = state.audio_samples_total / float(config.audio_sample_rate)
                logger.info(f"Buffered audio chunks: {chunks} (~{secs:.2f}s at {config.audio_sample_rate} Hz)")
    except Exception as e:
        logger.error(f"Error buffering audio frame: {e}")
    return [audioframe] # Pass-through


async def update_params(params: dict):
    """Updates application configuration from server requests."""
    logger.info(f"Updating parameters: {params}")
    for key, value in params.items():
        if hasattr(config, key):
            try:
                # Get the type of the default value and cast the new value
                if key == 'modality':
                    mod = str(value).lower()
                    if mod not in { 'text', 'vision', 'audio' }:
                        raise ValueError("modality must be one of 'text', 'vision', 'audio'")
                    setattr(config, key, mod)
                    # Set default prompt based on modality only if prompt not provided in params
                    if 'prompt' not in params:
                        if mod == 'audio':
                            config.audio_prompt = "Transcribe the audio into the language spoken in the audio"
                        elif mod == 'vision':
                            config.vision_prompt = "Describe whats happening in the images in short sentences."
                    elif key == 'prompt':
                        # Always update prompt if provided
                        setattr(config, key, str(value))
                else:
                    target_type = type(getattr(AppConfig(), key))
                    setattr(config, key, target_type(value))
            except (TypeError, ValueError) as e:
                logger.warning(f"Could not set param '{key}' to '{value}': {e}")
    logger.info(f"New config: {asdict(config)}")


# --- Background Tasks ---

async def run_generation_loop():
    """Periodically trigger multimodal generation using in-memory buffers."""
    try:
        while True:
            await asyncio.sleep(config.generate_every_s)

            async with state.buffer_lock:
                pil_images = list(state.video_frames) if config.modality == 'vision' else []
                # Changed: Extract only the last 4 seconds for overlap
                audio_data = None
                if config.modality == 'audio':
                    audio_data = get_last_n_seconds_audio(
                        state.audio_chunks, state.audio_samples_total, 4.0, config.audio_sample_rate
                    )
                    if audio_data is None:
                        logger.debug("Not enough audio for 4s window; skipping generation.")
                        continue

            # Determine if we should run a cycle based on modality
            should_run = (
                (config.modality == 'vision' and bool(pil_images)) or
                (config.modality == 'audio' and state.audio_samples_total > 0) or
                (config.modality == 'text')
            )

            now = time.time()
            idle = (now - state.last_data_time) > state.idle_timeout_s

            if idle:
                logger.debug("No longer receiving segments breaking generation loop")
                break
            elif not should_run:
                logger.debug("NNo longer receivinelected or buffer empty.")
                break

            cycle_id = str(uuid.uuid4())
            logger.info(f"Starting generation cycle {cycle_id}")

            try:
                # Prepare multimodal inputs for VLLM
                audio_data = None

                # Prepare audio for audio modality
                if config.modality == 'audio' and state.audio_chunks:
                    try:
                        audio_data = np.concatenate(state.audio_chunks, axis=0).astype(np.float32, copy=False)
                    except Exception as e:
                        logger.warning(f"Failed to concatenate audio; skipping cycle: {e}")
                        # Skip this cycle to avoid sending empty audio to the model
                        continue
                    if audio_data is not None and audio_data.size == 0:
                        logger.info("Audio payload is empty after concatenation; skipping cycle.")
                        continue

                    # Basic sanity checks: duration and loudness
                    duration_s = audio_data.shape[0] / float(config.audio_sample_rate)
                    rms = float(np.sqrt(np.mean(audio_data**2))) if audio_data.size > 0 else 0.0
                    if duration_s < 0.2:
                        logger.info(f"Audio too short ({duration_s:.3f}s); skipping cycle.")
                        continue
                    if rms < 1e-4:
                        logger.info(f"Audio near-silent (rms {rms:.2e}); skipping cycle.")
                        continue                    

                # Generate via VLLM
                t_gen_start = time.perf_counter()
                try:
                    generated_text = ""
                    generated_text = await asyncio.wait_for(
                        asyncio.to_thread(
                            state.vllm_client.chat,
                            prompt=config.audio_prompt if config.modality == "audio" else config.vision_prompt,
                            images=pil_images if len(pil_images) != 0 else None,
                            audio= audio_data,
                            audio_sr=config.audio_sample_rate,
                            max_tokens=config.max_new_tokens,
                            stream=False,
                        ),
                        timeout=int(config.request_timeout_s),
                    )
                except asyncio.TimeoutError:
                    logger.error("vLLM chat timed out; skipping cycle.")
                    continue
                except Exception as e:
                    logger.error(f"vLLM chat failed; skipping cycle: {e}")
                    continue

                t_gen_end = time.perf_counter()

                payload = {
                    "type": "generation",
                    "timestamp_ms": int(time.time() * 1000),
                    "cycle_id": cycle_id,
                    "text": generated_text,
                    "stats": {
                        "frame_count": len(pil_images),
                        "audio_duration_ms": int((audio_data.shape[0] / config.audio_sample_rate) * 1000) if (config.modality == 'audio' and audio_data is not None) else 0,
                        "max_new_tokens": config.max_new_tokens,
                        "timings_ms": {
                            "generate": int((t_gen_end - t_gen_start) * 1000)
                        }
                    }
                }
                try:
                    # Do we send both uncorrected and corrected payload for audio?
                    if not config.correction_enabled:
                        await state.processor.send_data(json.dumps(payload))
                        logger.info("Published generation payload to data_url.")
                except Exception as e:
                    logger.error(f"Failed to publish generation payload: {e}")
                timings = payload["stats"]["timings_ms"]
                logger.info(f"Cycle {cycle_id} timings (ms): generate={timings['generate']}")

                # Follow-up: text-only correction using rolling history
                try:
                    if (
                        config.correction_enabled
                        and isinstance(generated_text, str)
                        and generated_text.strip() != ""):

                        latest_raw = generated_text.strip()

                        # Append to history and trim
                        state.generation_history.append(latest_raw)
                        while len(state.generation_history) > config.history_max_turns:
                            state.generation_history.popleft()

                        # Build correction prompt
                        prev_history = list(state.generation_history)[:-1]
                        if config.modality=="vision":
                            corr_prompt= vision_correction_prompt(
                                history=prev_history,
                                latest=latest_raw
                            )
                        else:
                            corr_prompt = build_correction_prompt(
                                history=prev_history,
                                latest=latest_raw,
                                system_hint=config.correction_system_hint,
                            )

                        t_corr_start = time.perf_counter()
                        corrected_text = await asyncio.wait_for(
                            asyncio.to_thread(
                                state.vllm_client.chat,
                                prompt=corr_prompt,
                                images=None,
                                audio=None,
                                max_tokens=min(512, config.max_new_tokens),
                                stream=False,
                        ),
                        timeout= float(config.request_timeout_s),)

                        t_corr_end = time.perf_counter()

                        corr_payload = {
                            "type": "correction",
                            "timestamp_ms": int(time.time() * 1000),
                            "cycle_id": cycle_id,
                            "history_len": len(prev_history),
                            "corrected_text": corrected_text,
                            "stats": {
                                "timings_ms": {
                                    "generate": int((t_gen_end - t_gen_start) * 1000),
                                    "correct": int((t_corr_end - t_corr_start) * 1000)
                                }
                            }
                        }

                        # Append to history and trim
                        state.generation_history.append(corrected_text)
                        while len(state.generation_history) > config.history_max_turns:
                            state.generation_history.popleft()

                        try:
                            await state.processor.send_data(json.dumps(corr_payload))
                            logger.info("Published transcript correction payload to data_url.")
                        except Exception as e:
                            logger.error(f"Failed to publish correction payload: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Error during transcript correction step: {e}", exc_info=True)
                    continue
            except Exception as e:
                logger.error(f"Error during generation cycle: {e}", exc_info=True)
                continue
    finally:
        state.trigger_generation = True


# --- Helpers ---

def build_correction_prompt(
    history: list[str],
    latest: str,
    system_hint: str | None = None,
) -> str:
    """Construct a single-string prompt for text-only correction using raw history.

    style="chunk": Ask to correct only the latest chunk using the prior history for context.
    """
    sys = (system_hint or "").strip()
    lines: list[str] = []
    if sys:
        lines.append(f"[System]\n{sys}\n")

    if history:
        lines.append("[Previous Transcript Chunks]")
        for h in history[-config.history_max_turns:]:
            if h.strip():
                lines.append(f"- {h.strip()}")
        lines.append("")

    lines.append("[Latest Chunk]")
    lines.append(latest.strip())
    lines.append("")
    lines.append(
        "Task: Using the history for context, correct ONLY the latest chunk and return just that corrected chunk. "
        "If the latest chunk is silence or noise, return an empty string."
    )
    return "\n".join(lines)

def vision_correction_prompt(
    history: list[str],
    latest: str,
) -> str:

    sys: str = (
        "You are a video narrative continuation assistant. You will receive [Previous Frames Understanding] "
        "(up to the last 4 entries) and a [Latest Chunk]. Compare them and produce a concise continuation of the "
        "same story, not a new caption. Include only genuinely new information or state changes from the latest frame; "
        "omit anything already established. Keep entity names and pronouns consistent, maintain the same tense and style, "
        "and avoid phrases like \"this frame shows\" or \"the image.\" If nothing new is present, return an empty string."
    )
    lines: list[str] = []
    if sys:
        lines.append(f"[System]\n{sys}\n")

    if history:
        lines.append("[Previous Frames Understanding]")
        for h in history[-config.history_max_turns:]:
            if h.strip():
                lines.append(f"- {h.strip()}")
        lines.append("")

    lines.append("[Latest Chunk]")
    lines.append(latest.strip())
    lines.append("")
    lines.append(
    "Task: Using [Previous Frames Understanding] for context, rewrite [Latest Chunk] as a continuation of the ongoing narrative, "
    "adding only genuinely new changes or state updates. Write the changes in narration style of david attaborough in 3-4 short sentences, no headings or bullets, no recap. "
    "If there are no new changes, return an empty string."
    )
    return "\n".join(lines)
# --- Helper Function ----

def get_last_n_seconds_audio(audio_chunks: deque, audio_samples_total: int, n_seconds: float, sample_rate: int) -> np.ndarray | None:
    """Extract the last N seconds of audio from the deque without modifying it."""
    if audio_samples_total < n_seconds * sample_rate:
        return None  # Not enough data
    
    target_samples = int(n_seconds * sample_rate)
    accumulated = []
    total_accumulated = 0

    # Iterate from the end of the deque
    for chunk in reversed(audio_chunks):
        accumulated.insert(0, chunk)  # Prepend to maintain order
        total_accumulated += len(chunk)
        if total_accumulated >= target_samples:
            break

    # Concatenate and trim to exactly target_samples
    audio_data = np.concatenate(accumulated)
    if len(audio_data) > target_samples:
        audio_data = audio_data[-target_samples:]  # Take the last part
    return audio_data    


# --- Main Execution ---

async def main():
    """Start the StreamServer and background tasks on the same asyncio loop."""
    processor = StreamProcessor(
        video_processor=process_video,
        audio_processor=process_audio,
        model_loader=load_model(),
        param_updater=update_params,
        name="multimodal-understanding-gemma3n",
        port=config.server_port,
    )
    state.server = processor.server
    state.processor = processor

    # Start HTTP server (aiohttp) on the current event loop
    await processor.server.run_forever()
    logger.info("PyTrickle server started; watcher task running.")

if __name__ == "__main__":
    asyncio.run(main())

