# LocalDev AI



**A local-first AI coding assistant for chatting with a locally installed language model and exploring a selected codebase.** The app provides a browser UI, a FastAPI backend, persistent SQLite chat history, optional project indexing, and live CPU/GPU/disk/RAM telemetry. It does not send prompts to a hosted LLM.

> Local inference requires you to install a compatible model and its runtime separately. Model weights and AMD's runtime are not distributed with this project.

## Screenshot

Example response generated in the LocalDev AI chat for **“What is linear regression?”**

![LocalDev AI answering “What is linear regression?”](assets/screenshot.png)

## Features

- **Local chat:** the server runs on `127.0.0.1`; conversations and messages are stored in SQLite.
- **Local model backends:** AMD Ryzen AI ONNX Runtime GenAI hybrid, ONNX Runtime GenAI DirectML, or local Transformers.
- **Project context:** select and index a local code project; retrieve relevant source snippets using lightweight lexical matching.
- **Chat history:** create, list, rename, export, and delete chats. Deleting a conversation also deletes its messages and saved export.
- **System dashboard:** synchronized CPU, GPU, physical-disk activity, and RAM graphs. Windows hardware counters are sampled by a separate local process so model generation does not block the graphs.
- **Readable answers:** Markdown-style lists, code blocks with copy buttons, and formatted equations.
- **No hosted fallback:** if the local model cannot be loaded or generation fails, the API reports an error instead of sending the prompt to an online service.

## Requirements

- Windows 10/11 is recommended. The included CPU/RAM sampler uses `psutil`; GPU engine and physical-disk activity readings use Windows performance counters and PowerShell.
- Python 3.10 or newer for the web application and tests.
- A compatible local model and its runtime for chat generation.
- For AMD hybrid inference: supported AMD Ryzen AI hardware, current AMD drivers, and the compatible AMD Ryzen AI / ONNX Runtime GenAI environment required by the selected model.
- Internet access is needed only to install Python packages or download model files. Chat prompts are sent to the local app/model, not a hosted LLM.

## Quick start (Windows PowerShell)

Open PowerShell in the repository folder.

1. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in the following commands.

2. Install the application dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   `torch` and `transformers` can be large downloads. For AMD hybrid or DirectML inference, use the Python environment and provider packages compatible with your AMD Ryzen AI installation instead of replacing its runtime with an incompatible generic package.

3. Create local settings:

   ```powershell
   Copy-Item .env.example .env
   ```

   The template uses automatic model-path detection. Set `LOCALDEV_MODEL_PATH` in `.env` if your model is in a different location. `.env` is ignored by Git.

4. Install or download a supported model as described in [Model setup](#model-setup).

5. Start the server:

   ```powershell
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

   Or use the repository helper:

   ```powershell
   .\scripts\start.ps1
   ```

6. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The first chat request loads the selected model and may take some time. Use `Ctrl+C` in the server terminal to stop the app.

The app listens on loopback by default. Keep it bound to `127.0.0.1` unless you have deliberately configured and secured access from other machines.

## Model setup

The repository does not include model weights. By default, the app checks these folders **beside the repository folder** (the `models` directory is a sibling of the cloned repository):

1. `../models/Qwen2.5-Coder-7B-Instruct-RyzenAI-Hybrid` — a complete AMD Ryzen AI hybrid ONNX Runtime GenAI package.
2. `../models/Qwen2.5-Coder-7B-Instruct-ONNX-DML` — a prepared ONNX Runtime GenAI DirectML model.
3. `../models/Qwen2.5-Coder-7B-Instruct` — a standard local Hugging Face Transformers checkpoint.

The hybrid package is preferred when its required files are present. It routes prompt prefill to the NPU and token generation to the integrated GPU in successive stages; it does **not** run the same model operation on CPU, GPU, and NPU simultaneously. The CPU also performs host-side work. DirectML is GPU inference. Transformers uses the device selected by PyTorch and may run on CPU or a supported CUDA device; it is not the AMD hybrid runtime.

### Build the DirectML fallback

If you already have the standard Qwen checkpoint at `../models/Qwen2.5-Coder-7B-Instruct`, activate a Python environment with the compatible `onnxruntime_genai` model builder and DirectML dependencies installed, then run:

```powershell
.\scripts\build_dml_model.ps1
```

The script converts the local checkpoint to INT4 DirectML format in the sibling `../models/Qwen2.5-Coder-7B-Instruct-ONNX-DML` directory. It does not download model files.

### AMD Ryzen AI hybrid model

Install the AMD Ryzen AI software/runtime version supported by your device and follow AMD's setup guidance for its compatible Python environment and ONNX Runtime GenAI provider. Then, from the repository folder, download the official model into the default sibling location:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='amd/Qwen2.5-Coder-7B-Instruct-onnx-ryzenai-1.7-hybrid', local_dir=r'..\models\Qwen2.5-Coder-7B-Instruct-RyzenAI-Hybrid')"
```

Run the command in the same AMD-compatible environment in which the app will run. If `huggingface_hub` is unavailable, install it in that environment first. The model repository and AMD runtime are separate third-party projects with their own requirements and licenses.

### Existing model in a different folder

Set the path to the model directory in `.env`:

```dotenv
LOCALDEV_MODEL_PATH=C:/absolute/path/to/model-directory
```

Other supported settings are documented in `.env.example`: model name/provider, execution mode, temperature, top-p, maximum generated tokens, and streaming preference. The directory must contain the files expected by its backend. ONNX Runtime GenAI model directories require a compatible `genai_config.json` and provider/runtime; Transformers directories require local model and tokenizer files. The app loads model files locally and does not download them during chat.

Check `GET /api/system/model` after startup or first use. `execution_verified` becomes true after the provider has loaded and verified its runtime device. A model directory existing on disk alone does not mean that its provider or hardware is available.

## Project indexing

Use the project API to register a folder and index its supported source and text files. The scanner skips common version-control, virtual-environment, dependency, cache, build, and generated directories, hidden files (except selected project metadata files), `.env` files, and filenames containing `secret`. Supported extensions are defined by `SAFE_TEXT_EXTENSIONS` in `main.py`.

The current index stores file metadata and hashes in SQLite. Search scores matching terms from the current file contents and paths; it is a small lexical retrieval implementation, **not** a vector/embedding database. Project roots are local paths on the machine running the server. Review the project folder and its contents before indexing.

The PDF upload API stores uploaded files locally under `local_data/uploads/`; uploading alone does not add them to chat context or build a document index.

## Resource dashboard

- **CPU and RAM:** whole-system usage from `psutil`.
- **GPU:** busiest Windows GPU engine, aggregated by adapter and engine type from Windows GPU performance counters.
- **Disk:** system physical-disk active time from the Windows `% Disk Time` counter. App-drive used/free capacity is additional detail, not the graph percentage.
- Sampling runs in a separate localhost process on port `8766`; the browser polls it independently from model generation. Hardware counters may be unavailable on non-Windows systems or Windows installations without the relevant counters; unavailable readings are reported rather than inferred.

## Data, privacy, and cleanup

- Chat database: `local_data/database/app.sqlite`
- Conversation exports: `local_data/exports/`
- Project metadata and other generated app data: `local_data/`
- Uploaded files: `local_data/uploads/`
- Downloaded model weights: keep them outside the repository, such as the sibling `../models/` folder above.

These local data folders, model files, vector stores, and environment files are excluded by `.gitignore`. Deleting a chat through the app removes its conversation and messages from SQLite and deletes its export file. The database is created automatically on startup.

The assistant generates answers locally, but third-party model files and runtime packages remain subject to their own licenses and privacy policies. This project does not bundle them.

## API overview

The interactive API reference is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) while the app is running.

| Route | Purpose |
|---|---|
| `GET /api/health` | App and database location status |
| `GET /api/conversations` | List saved conversations |
| `POST /api/conversations` | Create a conversation |
| `GET /api/conversations/{id}` | Read a conversation and its messages |
| `PATCH /api/conversations/{id}` | Rename a conversation |
| `DELETE /api/conversations/{id}` | Delete a conversation, its messages, and its export |
| `DELETE /api/conversations` | Delete all conversations and messages |
| `GET /api/conversations/{id}/export` | Write a JSON conversation export |
| `POST /api/chat` | Generate a local-model response and save both messages |
| `POST /api/chat/stream` | Server-sent-event response endpoint |
| `GET /api/projects`, `POST /api/projects` | List or register local project roots |
| `POST /api/projects/{id}/index` | Index a registered project |
| `GET /api/projects/{id}/status` | Read indexing status |
| `POST /api/rag/search` | Search indexed project files |
| `POST /api/upload` | Store an uploaded file locally |
| `GET /api/system/usage` | Read cached CPU/GPU/disk/RAM usage |
| `GET /api/system/status` | Read resource readings and model readiness |
| `GET /api/system/model` | Inspect model and execution status |
| `POST /api/system/model/load`, `/unload` | Load or unload the configured local model |
| `GET /api/system/metrics` | Read model load and generation metrics |

`/api/history` routes remain available as compatibility aliases for conversation history.

## Development and tests

Install the test-only dependencies and run the test suite:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q tests/test_local_assistant.py
```

Tests cover API behavior, chat persistence and deletion, database integrity, project indexing, resource-monitor lifecycle, and static logo serving. Hardware-counter tests accept explicit unavailable readings. GitHub Actions runs the suite on Windows with Python 3.12 for pushes and pull requests.

## Repository layout

```text
assets/                  SVG brand mark and README screenshot
scripts/setup.ps1        Install dependencies and create local-data folders
scripts/start.ps1        Start the local FastAPI server
tests/                   API, database, and telemetry tests
.env.example             Safe local settings template
index.html               Browser chat UI and resource graphs
main.py                  FastAPI app, local model provider, chat and projects
resource_monitor.py      Separate CPU/GPU/disk/RAM sampler
requirements.txt         Application dependencies
requirements-test.txt    Lightweight test dependencies
SECURITY.md              Vulnerability reporting guidance
```
