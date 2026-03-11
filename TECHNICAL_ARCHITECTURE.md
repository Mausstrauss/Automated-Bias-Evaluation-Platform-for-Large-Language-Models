# TECHNICAL_ARCHITECTURE.md

# Technical Implementation & System Architecture
**Project:** Automated Bias Evaluation Platform for LLMs
**Version:** 1.0.0 (Release Candidate)
**Date:** January 2026

---

## 1. System Overview & Design Philosophy

The Automated Bias Evaluation Platform is a containerized, microservices-based application designed to audit Large Language Models (LLMs) for social biases. The system moves beyond static benchmarks by enabling **continuous, longitudinal evaluation** of both proprietary APIs (Google Gemini, OpenAI) and open-source models.

### 1.1 Architectural Pattern
The system follows a **Service-Oriented Architecture (SOA)** managed via Docker Compose. It decouples the user interface from the execution logic to allow for independent scaling and distinct lifecycles (interactive vs. headless).

* **Frontend Service (`bias-gui`):** Handles user interaction, configuration, and real-time visualization.
* **Background Service (`bias-scheduler`):** Handles automated, periodic execution of test suites for longitudinal tracking.
* **Shared Volume (`/output`):** Acts as the persistence layer, ensuring data consistency between services without the overhead of a dedicated database server.

---

## 2. Microservices Implementation

### 2.1 Service A: The Interactive Frontend (`bias-gui`)
* **Framework:** Streamlit (Python)
* **Responsibility:** Provides the "Research Dashboard" for ad-hoc audits.
* **Key Engineering Features:**
    * **Session State Management:** Utilizes Streamlit's session state to manage API keys and temporary dataframes across re-runs.
    * **Dynamic Visualization:** Integrates `plotly.express` for interactive Radar Charts and Box Plots, enabling researchers to inspect high-dimensional bias profiles.
    * **Data Injection:** Implements a file uploader to accept custom CSV datasets, dynamically normalizing them into the internal `List[Dict]` schema.

### 2.2 Service B: The Headless Scheduler (`bias-scheduler`)
* **Framework:** Python `schedule` library + Custom Loop
* **Responsibility:** Executes the "Nightly Audit" to detect bias drift.
* **Key Engineering Features:**
    * **Deterministic Execution:** Runs a pre-configured suite of tests (e.g., "Gender Templates" against "Gemini-1.5") at a fixed time (default `03:00` UTC).
    * **Fail-Safe Logging:** Writes raw logs to disk immediately after generation to ensure data recovery in case of container termination.

---

## 3. Core Logic & Design Patterns

The core logic (`app/core/blackbox/`) is shared between both services to ensure consistent metric calculation.

### 3.1 The Inference Engine (Strategy Pattern)
**File:** `app/core/blackbox/generators.py`

To manage the heterogeneity of model providers, the system implements the **Strategy Design Pattern**. This allows the application to switch between API protocols at runtime without changing the consuming code.

* **Context:** `LLMGenerator` class.
* **Strategy Interface:** Defines `generate_batch(prompts: List[Dict]) -> List[Dict]`.
* **Concrete Strategies:**
    1.  **`GoogleGeminiStrategy`:** * Wraps `google.generativeai`.
        * **Critical Implementation Detail:** Explicitly disables safety filters (`HarmBlockThreshold.BLOCK_NONE`). This is necessary to measure the model's *inherent* bias; otherwise, the API would block the very content we aim to measure.
    2.  **`OpenAIStrategy`:** * Wraps the `openai` client (v1.0+).

### 3.2 The Oracle Pipeline (Scoring Factory)
**File:** `app/core/blackbox/oracles.py`

The system processes raw text through a chain of "Bias Oracles."

* **Sentiment Oracle:**
    * **Algorithm:** Uses `TextBlob` (Lexicon-based) to compute polarity $P \in [-1.0, 1.0]$.
    * **Metric:** Calculates the delta between demographic groups ($\Delta S$).
* **Toxicity Oracle:**
    * **Model:** `unitary/toxic-bert` (Hugging Face).
    * **Optimization:** The model is loaded once into memory (Singleton behavior) to prevent overhead during batch processing.
    * **Metric:** Returns a probability score $P(Toxic) \in [0.0, 1.0]$.

### 3.3 Robust Data Ingestion (Fail-Safe Loading)
**File:** `app/core/blackbox/template_loader.py`

To handle the unreliability of external dependencies (Hugging Face Hub) in restricted container environments, the data loader implements a **Fail-Safe Circuit Breaker**:

1.  **Attempt 1 (Streaming):** Tries to stream datasets (e.g., BOLD, RealToxicityPrompts) from Hugging Face to minimize memory footprint.
2.  **Catch Exception:** If a network timeout or HTTP 500 error occurs.
3.  **Attempt 2 (Fallback):** Automatically switches to a hardcoded, internal fallback list of prompts. This ensures the application never crashes during a demo or offline use.

---

## 4. Infrastructure & Deployment Details

### 4.1 Docker Optimization
The Docker image is heavily optimized for size and build stability.

* **Base Image:** `python:3.9-slim` (Debian-based, minimal footprint).
* **CPU-Only PyTorch:** * Standard PyTorch installs include CUDA drivers (~3GB). 
    * We enforce the CPU version via `pip install --index-url https://download.pytorch.org/whl/cpu torch`.
    * **Result:** Reduces image size by ~75% (down to ~800MB), fixing build timeouts.
* **Caching:** `requirements.txt` is copied and installed *before* the application code to leverage Docker layer caching.

### 4.2 Data Persistence Strategy
The system uses a **File-Based Persistence Model** to avoid the complexity of managing a Postgres/SQL container.

* **Raw Logs (`/output/logs/`):** * Format: `audit_[MODEL]_[TIMESTAMP].csv`.
    * Content: Every single prompt, response, and score.
    * Purpose: Scientific reproducibility and granular auditing.
* **History Index (`/output/evaluation_history.csv`):**
    * Format: Time-series CSV.
    * Content: Aggregated mean scores per run.
    * Purpose: Source data for the "History & Trends" dashboard tab.

---

## 5. Security Protocols

1.  **API Key Hygiene:**
    * Keys are accepted via Environment Variables (`.env`) or GUI input.
    * Keys are stored in ephemeral memory only; they are **never** written to disk or logs.
2.  **Network Isolation:**
    * Services communicate over a private Docker bridge network.
    * Only the Frontend (`bias-gui`) exposes a port (8501) to the host machine.
3.  **Dependency Pinning:**
    * All critical libraries (Pandas, Plotly, Google-GenAI) are pinned to specific versions in `requirements.txt` to prevent supply chain drift.

---

## 6. Directory Structure

```text
/bias-eval-platform
├── docker-compose.yml          # Orchestration config
├── .env                        # Environment variables (API Keys)
├── app/
│   ├── Dockerfile              # Multi-stage build definition
│   ├── requirements.txt        # Python dependencies (CPU-optimized)
│   ├── app_gui.py              # Entrypoint: Frontend Service
│   ├── scheduler.py            # Entrypoint: Scheduler Service
│   ├── core/
│   │   ├── blackbox/
│   │   │   ├── generators.py       # Strategy Pattern Implementation
│   │   │   ├── oracles.py          # Scoring Logic
│   │   │   ├── template_loader.py  # Data Ingestion & Fail-Safe
│   │   │   └── visualizer.py       # Plotly Helper Functions
│   └── data/
│       ├── templates/          # Local CSV Templates
│       └── bold/               # (Optional) Local BOLD Cache
└── output/                     # Persistence Volume
    ├── logs/                   # Raw Audit Trails
    └── evaluation_history.csv  # Aggregated Trend Data