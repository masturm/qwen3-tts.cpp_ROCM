# qwen3-tts.cpp

HTTP API server for [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) text-to-speech, built on top of the [GGML](https://github.com/ggml-org/ggml) tensor library. Models load once at startup; requests come in over HTTP and return WAV audio.

## Features

- Load GGUF models once, serve many requests
- Text → WAV synthesis over HTTP
- Voice cloning from an uploaded reference WAV
- Reusable speaker embeddings (extract once, synthesize repeatedly)
- Named-speaker presets for CustomVoice-capable models
- Pure C++17, no runtime Python dependency (Python is only used to prepare model files)

## Prerequisites

- CMake 3.14+
- A C++17 compiler (MSVC 2022+, Clang, or GCC)
- Python 3.10+ with [uv](https://github.com/astral-sh/uv) for one-time model preparation

## Build

```bash
git clone https://github.com/predict-woo/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

cmake -S . -B build -DQWEN3_TTS_EMBED_GGML=ON
cmake --build build --config Release --parallel
```

On Windows with Visual Studio 2026:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DQWEN3_TTS_EMBED_GGML=ON
cmake --build build --config Release --parallel
# copy ggml DLLs next to the exe so it can find them
cp build/bin/Release/*.dll build/Release/
```

`build.ps1 -Clean` wraps the Windows build (VS 2022 generator by default; override with `-UseNinja` or edit the script).

### Optional build flags

- `-DQWEN3_TTS_CUDA=ON` — enable CUDA backend via GGML
- `-DQWEN3_TTS_EMBED_GGML=OFF -DQWEN3_TTS_GGML_BUILD_DIR=/path/to/ggml/build` — link against a prebuilt GGML

## Model preparation

Download and convert the Qwen3-TTS weights once:

```bash
uv venv && source .venv/bin/activate    # or `.venv\Scripts\activate` on Windows
uv pip install huggingface_hub gguf torch safetensors numpy tqdm
python scripts/setup_pipeline_models.py
```

This produces two GGUF files in `models/`:

- `models/qwen3-tts-0.6b-f16.gguf` — transformer + speaker encoder + text tokenizer
- `models/qwen3-tts-tokenizer-f16.gguf` — WavTokenizer vocoder

For the 1.7B variants (including CustomVoice with named speakers):

```bash
python scripts/setup_1.7b_model.py                  # base
python scripts/setup_1.7b_model.py --variant customvoice
```

## Run the server

```bash
./build/qwen3-tts-server -m models
# or:
./build/qwen3-tts-server -m models --host 0.0.0.0 --port 8080 -j 8
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `-m, --model <dir>` | required | Directory containing the GGUF files |
| `--model-name <name>` | auto | Explicit base name if multiple model sets live in `models/` |
| `--host <addr>` | `127.0.0.1` | Bind address (`0.0.0.0` for LAN access) |
| `--port <n>` | `8080` | TCP port |
| `-j, --threads <n>` | `4` | Default thread count for synthesis |

Models load on startup (~1–2 s for 0.6B-f16). Synthesis requests are serialized behind a mutex — one synthesis at a time per server process.

## HTTP API

All endpoints are JSON except where multipart is noted. Audio responses are `audio/wav` (16-bit PCM, 24 kHz, mono). Synthesis timings are returned in `X-Synth-*-Ms` response headers.

### `GET /health`

```json
{"status":"ok","model_loaded":true}
```

### `GET /v1/capabilities`

```json
{
  "loaded": true,
  "model_type": "base",
  "supports_voice_clone": true,
  "supports_named_speakers": false,
  "supports_instruction": true,
  "speaker_embedding_dim": 1024,
  "speaker_count": 0
}
```

### `GET /v1/speakers`

List named speakers (CustomVoice models only; empty otherwise).

```json
{"speakers":["vivian","nova","..."]}
```

### `POST /v1/synthesize`

Synthesize from text. Returns `audio/wav`.

Request body (JSON):

```json
{
  "text": "Hello from the API!",
  "speaker": "",
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 1.0,
  "max_tokens": 4096,
  "repetition_penalty": 1.05,
  "language": "en",
  "instruction": "",
  "threads": 4
}
```

Only `text` is required. `language` accepts `en`, `ru`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `it`, `pt` or a raw integer ID. For CustomVoice models, pass `"speaker": "<name>"`. To synthesize with a precomputed embedding, pass `"embedding": [<floats>]` (takes precedence over `speaker`).

Example:

```bash
curl -s http://127.0.0.1:8080/v1/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from the API!","language":"en"}' \
  --output hello.wav
```

### `POST /v1/synthesize_with_voice`

Voice-clone synthesis from an uploaded reference WAV. Multipart form:

| Field | Type | Notes |
|---|---|---|
| `text` | form field | Required |
| `reference_audio` | file | Required. WAV, any sample rate (resampled to 24 kHz server-side). |
| `params` | form field | Optional JSON blob with the same keys as `/v1/synthesize` |
| *any other* | form field | Individual params (`temperature=0.9`, `language=en`, …) |

Returns `audio/wav`.

```bash
curl -s http://127.0.0.1:8080/v1/synthesize_with_voice \
  -F "text=Cloned voice saying this." \
  -F "reference_audio=@examples/readme_clone_input.wav" \
  -F "temperature=0.8" \
  --output cloned.wav
```

### `POST /v1/speaker_embedding`

Extract and return a speaker embedding from a reference WAV. Useful for caching — send once, then call `/v1/synthesize` with the returned embedding for subsequent clips.

Multipart: `reference_audio` (file, required).

```json
{
  "embedding": [0.0123, -0.0487, ...],
  "dim": 1024,
  "encode_ms": 142
}
```

```bash
curl -s http://127.0.0.1:8080/v1/speaker_embedding \
  -F "reference_audio=@examples/readme_clone_input.wav" \
  > speaker.json

# Reuse the embedding without re-encoding:
jq -n --slurpfile s speaker.json \
  '{text:"Second clip, same voice.", embedding:$s[0].embedding}' \
| curl -s http://127.0.0.1:8080/v1/synthesize \
    -H 'Content-Type: application/json' -d @- \
    --output second.wav
```

## Error responses

Non-2xx responses return JSON:

```json
{"error":"Failed to tokenize text"}
```

Common statuses: `400` for bad request (missing fields, invalid JSON, unknown language), `500` for synthesis failures.

## Concurrency

Synthesis inside a single process is serialized behind a mutex — the GGML graph state isn't safe for concurrent inference. For parallelism, run multiple server processes on different ports behind a load balancer.

## Project layout

```
src/
  qwen3_tts.{h,cpp}           public C++ API (load-once, synthesize-many)
  pipeline/                   text → tokens → codes → audio pipeline
  transformer/, encoder/, decoder/   GGML graph + weight loaders
  common/                     WAV + speaker-embedding file I/O
  server/                     HTTP server (this project's entry point)
third_party/                  httplib.h, json.hpp (vendored)
ggml/                         submodule
scripts/                      Python tools to convert HF weights to GGUF
```

## License

See `LICENSE`.
