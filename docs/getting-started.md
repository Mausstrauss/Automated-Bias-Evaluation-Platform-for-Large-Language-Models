# Getting Started

This section explains how to set up and run the Automated Bias Auditing Platform. You can run the application either fully containerized via Docker (recommended) or locally using a Python virtual environment.

## Prerequisites

* **Docker & Docker Compose** installed on your machine.
* (Optional) **Python 3.9+** if you want to run it locally without Docker.
* **API Keys** for the black-box models you want to audit (e.g., OpenAI, Google Gemini).

## Step 1: Clone the Repository

```bash
git clone https://github.com/nkolev1919/Automated-Bias-Evaluation-Platform-for-Large-Language-Models.git
cd bias-eval-platform
```

## Step 2: API & Environment Configuration

Because this platform conducts continuous black-box evaluations against proprietary LLMs, it requires valid API credentials. The application utilizes a `.env` file to securely manage these keys and prevent accidental commits of sensitive data to version control.

1. **Obtain your API keys.** You must generate API keys from the respective model providers. Running automated bias benchmarks consumes tokens; ensure your accounts are funded or within free-tier limits.
   * **OpenAI (GPT-3.5 / GPT-4):** Generate a key at the OpenAI Developer Platform.
   * **Google (Gemini Pro):** Generate a key via Google AI Studio.
   * **Anthropic / Mistral / Others:** Generate keys via their respective developer consoles if you plan to extend the model mapping.

2. **Create the `.env` file.** In the root directory of the project, duplicate the `.env.example` file (if provided) or create a new file named exactly `.env`.

3. **Configure the variables.** Populate the file with your credentials using an INI-style format. The `LLMGenerator` module will automatically pick up these values via environment variables:

```text
# --- .env ---

# TARGET MODEL API KEYS (Black-Box Evaluation)
OPENAI_API_KEY=sk-proj-YourActualOpenAIKeyHere...
GOOGLE_API_KEY=AIzaSyYourActualGoogleKeyHere...

# Optional: Add further keys if you extend the platform
# ANTHROPIC_API_KEY=sk-ant-YourActualAnthropicKeyHere...
# HUGGINGFACE_API_KEY=hf_YourActualHuggingFaceKeyHere...

# FRAMEWORK CONFIGURATION
ENVIRONMENT=development
# LOG_LEVEL=INFO
```

**Security warning:** Never commit your `.env` file to Git. Ensure `.env` is explicitly listed in your `.gitignore` file. If you prefer not to store API keys on your file system, the Streamlit GUI provides dynamic password-masked input fields in the sidebar to inject keys at runtime.

## Step 3: Containerized Deployment (Recommended)

The Docker setup builds two isolated containers (`bias-gui` and `bias-scheduler`) that share a synchronized data volume (`.:/app`).

**Performance optimization note:** The Dockerfile is optimized for deployment on standard hardware. It explicitly forces installation of the **CPU-only version of PyTorch**. This reduces the container image size by omitting unnecessary NVIDIA/CUDA binaries, speeding up the build process and preventing `ReadTimeoutError` crashes on standard network connections.

To build and start the application cluster, run the following command from the project root:

```bash
docker-compose up --build
```

To run the containers in detached mode in the background, append `-d` to the command.

**System verification:**

* **Frontend GUI:** Open your web browser and navigate to `http://localhost:8501`.
* **Scheduler service:** Check the terminal output to verify the scheduler has booted and is successfully polling the `scheduler_config.json` via the shared volume.

## Step 4: Local Development Setup (Fallback)

If you need to debug the source code, modify the evaluation logic, or prefer running the microservices locally without Docker, follow these steps to set up an isolated Python environment.

1. **Initialize the virtual environment:**

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies (optimized order).** To avoid downloading massive GPU drivers locally, install the CPU version of PyTorch *before* installing the remaining framework requirements:

```bash
# First: Install PyTorch (CPU-only variant)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Second: Install remaining project dependencies (Streamlit, Transformers, TextBlob, etc.)
pip install -r requirements.txt
```

3. **Initialize NLP corpora.** The lexicon-based evaluation metrics (e.g., the Sentiment Oracle) require specific NLTK text corpora to function. Download them via the TextBlob module:

```bash
python -m textblob.download_corpora
```

4. **Start the microservices.** Because the architecture relies on two asynchronous processes communicating via the file system, you must start them in two separate terminal windows (with the virtual environment activated in both):

* **Terminal 1 (Streamlit dashboard):**

```bash
streamlit run app_gui.py
```

* **Terminal 2 (continuous auditor):**

```bash
python scheduler.py
```

## Step 5: Running the Test Suite

The project includes a comprehensive test suite covering all core modules. No API key is required — all external LLM calls are mocked.

Install pytest if not already installed:

```bash
pip install pytest --break-system-packages
```

Run the full suite:

```bash
pytest tests/ -v
```

Expected output: 93 tests passing across 9 test files.

To run only unit tests:

```bash
pytest tests/unit/ -v
```

To run only integration tests:

```bash
pytest tests/integration/ -v
```

