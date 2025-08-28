# Gemma 3n Multimodal PyTrickle Server

This application provides a headless PyTrickle streaming server for the `google/gemma-3n-E4B-it` model. It processes video and audio streams to generate multimodal understanding outputs.

## Features

- **Media Passthrough**: Video and audio streams are passed through unmodified.
- **Separate Data Channel**: Generated text is sent over the Trickle `data_url` as newline-delimited JSON, not overlaid on the video.
- **Real-time Prompt Updates**: The generation prompt and other parameters can be updated in real-time.
- **Configurable Sampling**: Frame sampling, generation cadence, and token limits are all configurable.
- **Resource Aware**: Implements token gating and optional frame downscaling to manage model load and prevent OOM errors.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the server:**
    ```bash
    python app.py
    ```

The server will start on `http://localhost:8000`.

## Usage

### Start a Stream

Use `curl` to send a `POST` request to the `/api/stream/start` endpoint. You must provide URLs for the input (`subscribe_url`), output (`publish_url`), and data (`data_url`).

```bash
curl -X POST http://localhost:8000/api/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "subscribe_url": "http://<your-ingress-server>/input",
    "publish_url": "http://<your-egress-server>/output",
    "data_url": "http://<your-data-server>/data",
    "gateway_request_id": "my-stream-1",
    "params": {
      "prompt": "Describe what is happening in the video and transcribe any speech.",
      "generate_every_s": 4.0,
      "target_fps": 3,
      "max_frames": 30,
      "max_input_tokens": 10000,
      "max_new_tokens": 700
    }
  }'
```

### Update Parameters

Update the prompt or other parameters mid-stream by sending a `POST` request to `/api/stream/params`.

```bash
curl -X POST http://localhost:8000/api/stream/params \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Focus on the actions of the main person in the scene.",
    "enable_downscale": true
  }'
```

### Stop a Stream

```bash
curl -X POST http://localhost:8000/api/stream/stop
```

### Consuming Data

The generated text and metadata will be streamed as newline-delimited JSON objects to the `data_url` you provided. Each JSON object represents the output from one generation cycle.

**Example JSON record:**
```json
{
  "type": "generation",
  "timestamp_ms": 1723800000000,
  "cycle_id": "some-uuid",
  "prompt": "Describe what is happening...",
  "text": "A person is walking down a city street at night.",
  "stats": {
    "frame_count": 12,
    "audio_duration_ms": 4000,
    "input_tokens": 950,
    "max_new_tokens": 700
  }
}
```
