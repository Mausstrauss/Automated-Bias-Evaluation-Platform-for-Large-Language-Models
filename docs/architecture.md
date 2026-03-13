# System Architecture & Data Flow

## 2. Architecture & System Design

The system consists of two primary operational components running in isolated Docker containers, synchronized via a shared data volume:

### 2.1. Module 1: The Frontend (Bias GUI)

* **Technology:** `app_gui.py` (Streamlit)
* **Role:** Serves as the interactive control center for researchers and domain experts.
* **Features:**
  * Allows users to upload custom prompt datasets (CSV/JSON) for domain-specific tests (e.g., medical or legal contexts).
  * Configures the target models (e.g., OpenAI GPT-4, Google Gemini) and inputs API keys.
  * Visualizes the multi-dimensional bias profiles through interactive Heatmaps and Bar Charts.
  * Generates and writes the `scheduler_config.json`.

### 2.2. Module 2: The Continuous Auditor (Bias Scheduler)

* **Technology:** `scheduler.py` (Python background process)
* **Role:** The engine for benchmark history tracking (bias drift over time).
* **Features:**
  * Runs head-less (without a GUI) and periodically checks the configuration.
  * Automatically executes the evaluation pipelines (Sentiment, Toxicity) against the configured APIs.
  * Appends new measurement points to the results database to track how model updates impact fairness over time.

### 2.3. Data Orchestration & IPC (Inter-Process Communication)

Instead of a heavy external database, this prototype utilizes a **shared Docker volume** (`.:/app`) for lightweight orchestration.

* When a user configures a continuous test in the GUI, it updates the `scheduler_config.json`.
* The Scheduler container instantly reads this file from the shared volume.
* Once the Scheduler completes a benchmark run, it saves the raw data as a CSV in the shared volume, which the GUI then parses to render the updated visualizations.

This architecture ensures that the system is model-agnostic, extensible, and well-suited for continuous **black-box** API evaluation.

---

## 4. Data Flow & Lifecycle

The system manages data through two distinct lifecycles depending on the trigger source (User vs. System).

### 4.1 Execution Flow (Synchronous Audit via GUI)

This flow is optimized for real-time feedback and data exploration.

1. **User Trigger:** Researcher configures parameters (e.g., *Gemini 1.5*, *BOLD Dataset*) and clicks "RUN EVALUATION".
2. **Template Expansion:** The `template_loader.py` reads the source (CSV or HuggingFace stream) and standardizes it into a `List[Dict]` format.
3. **Batch Inference:** The `LLMGenerator` iterates through prompts.
   * *Rate Limiting:* A dynamic sleep timer (`time.sleep(1.0)`) is applied to prevent HTTP 429 (Too Many Requests) errors on free-tier APIs.
4. **Scoring Pipeline:** Raw responses are passed to the static methods of `BiasOracle`.
5. **In-Memory Aggregation:** Results are converted to a Pandas DataFrame for immediate rendering.
6. **Visualization:** `BiasVisualizer` generates Plotly JSON objects (radar charts, box plots) which are rendered by Streamlit's frontend engine.

### 4.2 Scheduler Flow (Asynchronous / Benchmark History Audit)

This flow is optimized for reliability and data persistence.

1. **Initialization:** The `bias-scheduler` container starts an infinite event loop.
2. **Cron Trigger:** The `schedule` library detects when system time matches `RUN_TIME` (environment variable, default: `03:00`).
3. **Headless Execution:** The pipeline runs without UI overhead.
4. **Serialization Strategy (Dual-Write):**
   * **Aggregated Metrics:** Appended to `output/evaluation_history.csv` to update the trend lines in the dashboard.
   * **Raw Audit Logs:** A timestamped file is generated (`output/logs/audit_[TIMESTAMP].csv`) containing every single prompt-response pair. This ensures full scientific reproducibility and allows for post-hoc qualitative analysis.

---

## 5. Security, Constraints & Optimization

### 5.1 API Key Management

* **Ephemeral Storage:** API keys (Google, OpenAI) are injected via environment variables (`.env`) or temporary session state in the GUI.
* **No Disk Writes:** Keys are **never** serialized to disk or included in logs.
* **Safety Settings Override:** The system explicitly disables safety filters (`BLOCK_NONE`) on Google Gemini. This is a deliberate architectural decision required to measure the model's *intrinsic* bias rather than its safety filter's efficacy.

### 5.2 Resource & Network Constraints

* **Container Footprint:**
  * The Docker image is optimized by installing the **CPU-only version of PyTorch** (`--index-url https://download.pytorch.org/whl/cpu`). This reduces the image size from ~4GB to ~800MB, preventing timeouts during the build process on standard bandwidth.
* **Memory Usage:**
  * The `toxic-bert` classifier requires approximately **500MB RAM**.
  * The application is configured to run stably on instances with **2 vCPU / 4GB RAM**.
* **Network:**
  * Containers communicate via a private bridge network.
  * Outbound HTTPS (port 443) access is required for the `bias-gui` and `bias-scheduler` containers to reach model provider APIs.

---

## 6. Directory Structure & Artifacts

To assist with auditing and extension, the project structure follows a strict separation of concerns:

```text
/app
├── core/
│   ├── blackbox/
│   │   ├── generators.py       # Strategy pattern for LLM APIs
│   │   ├── oracles.py          # Scoring logic (sentiment/toxicity)
│   │   ├── template_loader.py  # Data ingestion (CSV/HuggingFace)
│   │   └── visualizer.py       # Plotly graphing logic
├── data/
│   ├── templates/              # LangBiTe CSV templates
│   └── bold/                   # Fallback data for BOLD dataset
├── output/                     # MOUNTED VOLUME (persisted)
│   ├── logs/                   # Raw CSV logs per run
│   └── evaluation_history.csv  # Aggregated time-series data
├── app_gui.py                  # Streamlit frontend entrypoint
├── scheduler.py                # Background service entrypoint
├── Dockerfile                  # Multi-stage build definition
└── requirements.txt            # Pinned dependencies (CPU-torch)
```

