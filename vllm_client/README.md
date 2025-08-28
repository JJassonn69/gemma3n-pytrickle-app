# vLLM External Server Setup and Client Integration

This folder contains assets to run a vLLM OpenAI-compatible server for `google/gemma-3n-E4B-it` and a small Python client used by the app to call it with multimodal inputs (text + images + audio).

## Prerequisites

- NVIDIA GPU on the host with NVIDIA Container Toolkit installed.
- Internet egress for model weights download (or a pre-populated HF cache volume).
- vLLM requires V1 for Gemma 3n and `timm>=1.0.17` (the Dockerfile here ensures this).

## Authentication for gated models (Hugging Face token)

Gemma 3n models are gated on Hugging Face. You must:

1. Request/accept access on the model page:
   - https://huggingface.co/google/gemma-3n-E4B-it
2. Create a Read token:
   - https://huggingface.co/settings/tokens
3. Copy the example env file and set your token:
   ```bash
   cd vllm
   cp .env.example .env
   # edit .env and paste your token value for HUGGING_FACE_HUB_TOKEN
   ```
4. Start (or restart) the server so it picks up the token:
   ```bash
   docker compose down
   docker compose up -d
   ```
5. Verify health:
   ```bash
   curl -s http://localhost:9000/v1/models | jq
   ```

Notes:
- The compose passes your token to the container via multiple environment variable names for compatibility: `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`.
- Do NOT commit `.env` to version control.

## Run with Docker Compose

The provided `docker-compose.yml` defines a `vllm` service exposing the OpenAI-compatible API.

```bash
cd vllm
# Start only the vLLM service
docker compose up -d

# Check health
curl -s http://localhost:9000/v1/models | jq
```

- Default port: `9000`
- Base URL from your app: `http://vllm:9000/v1` (service name on the compose network) or `http://localhost:9000/v1` from host.

## Environment variables for the app

Set these in your app container (compose or Kubernetes):

- `USE_VLLM=1`
- `VLLM_BASE_URL=http://vllm:9000/v1`  (or your service URL)
- `MODEL_ID=google/gemma-3n-E4B-it`

## Client usage (inside your app)

Use the small wrapper client to send multimodal Chat Completions requests.

```python
from vllm.client import VLLMClient

client = VLLMClient()  # reads VLLM_BASE_URL, MODEL_ID from env if set
text = client.chat(
    prompt="Describe the scene and audio briefly.",
    images=[pil_image_1, pil_image_2],      # optional
    audio=audio_np_f32_mono_16k,            # optional (float32 mono, 16k)
    audio_sr=16000,
    max_tokens=256,
    stream=False,
)
print(text)
```

For streaming tokens, use:

```python
for token in client.chat(prompt="...", images=[...], audio=..., stream=True):
    print(token, end="", flush=True)
```

## Notes

- Keep the vLLM server on an internal network; add auth/TLS if exposed externally.
- Mount a persistent volume at `/data/hf` to cache weights.
- Adjust `--gpu-memory-utilization`, `--max-model-len` as needed.

## References

- vLLM Supported Models (Gemma 3n): https://docs.vllm.ai/en/latest/models/supported_models.html
- vLLM Multimodal Inputs (OpenAI API): https://docs.vllm.ai/en/latest/features/multimodal_inputs.html
- vLLM VLM (offline examples): https://docs.vllm.ai/en/stable/models/vlm.html
