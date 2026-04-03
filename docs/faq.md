# FAQ & Troubleshooting

This section addresses common orchestration, data parsing, and runtime anomalies encountered during continuous black-box evaluation.

## 7.1. API & Network Issues

**Q: The scheduler crashes with a `429 Too Many Requests` or `Quota Exceeded` error.**

* **Symptom:** The platform stops evaluating and the terminal outputs an HTTP 429 status code.
* **Resolution:** You have hit the rate limit or token quota of your black-box provider (e.g., OpenAI or Google).
  1. Check your provider's billing dashboard.
  2. If on a free tier, increase the delay between batch requests in the `LLMGenerator` to respect the provider's tokens-per-minute (TPM) limits.

**Q: How do I manage and limit the API costs incurred by the continuous scheduler?**

* **Symptom:** The platform runs continuously, potentially generating high API billing costs over time.
* **Resolution:** The `scheduler.py` is designed for continuous benchmark history auditing. If you set `interval_hours` too low (e.g., every 1 hour) across thousands of prompts, costs will scale rapidly.
  1. For testing, set the scheduler to run infrequently (e.g., `24` hours) or disable the `active` flag in the GUI when not actively tracking drift.
  2. Monitor costs directly via your respective provider dashboards (e.g., OpenAI Platform billing).

## 7.2. Environment & Docker Configurations

**Q: The Docker build fails with a `ReadTimeoutError` during `pip install`.**

* **Symptom:** The build process stalls and throws an HTTPS connection timeout while downloading `nvidia-cudnn-cu12` or `torch`.
* **Resolution:** Docker is attempting to pull the massive GPU-accelerated PyTorch binaries (~3–4 GB). Ensure your `requirements.txt` does **not** contain `torch`. The provided `Dockerfile` explicitly installs the CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`) prior to processing the requirements file.

**Q: Changes made in the GUI do not trigger the background scheduler container.**

* **Symptom:** You update the target models in the Streamlit dashboard, but the scheduler continues running the old configuration.
* **Resolution:** This is a Docker volume synchronization issue, common on Windows/WSL2 environments. Ensure your project directory resides within the WSL2 filesystem (e.g., `\\wsl$\Ubuntu\home\user\project`) rather than the mounted Windows filesystem (`/mnt/c/...`) to guarantee real-time file I/O events for the `scheduler_config.json`.

**Q: Docker fails to start because port `8501` is already in use.**

* **Symptom:** `Error starting userlandproxy: listen tcp4 0.0.0.0:8501: bind: address already in use.`
* **Resolution:** A zombie Streamlit process or another container is occupying the port. Terminate the existing process, or modify your `docker-compose.yml` to map to an alternative host port: `ports: ["8505:8501"]`.

## 7.3. Data, Parsing & Visualization Anomalies

**Q: Uploading a custom prompt CSV results in a `KeyError` or `UnicodeDecodeError`.**

* **Symptom:** The GUI crashes immediately upon uploading the `.csv` dataset.
* **Resolution:**
  1. **Encoding:** The CSV must be strictly encoded in `UTF-8`. (Avoid standard Excel CSV exports; use "CSV UTF-8".)
  2. **Headers:** The parsing engine enforces strict schema validation. The first row must exactly contain the headers: `prompt`, `variable`, and `group`.

**Q: The Streamlit GUI throws a `StreamlitDuplicateElementId` error.**

* **Symptom:** A red overlay states: "There are multiple `plotly_chart` elements with the same auto-generated ID."
* **Resolution:** Streamlit requires unique identifiers when rendering multiple charts of the same class within the same DOM scope. Verify that every `st.plotly_chart()` instantiation in `app_gui.py` possesses a unique `key` argument (e.g., `key="barchart_comparison_unique"`).

**Q: I enabled a metric but see `NaN` / missing values.**

* **Symptom:** A metric column is missing or contains `NaN`.
* **Resolution:** Ensure the selected metric is supported by the running version and that the required dependencies are installed (e.g., `transformers` for toxicity). Also verify your dataset contains the expected columns (`prompt`, `variable`).

## 7.4. Evaluation & Mathematical Determinism

**Q: I run the exact same benchmark twice but get slightly different bias scores. Why?**

* **Symptom:** The `SentimentDiff` or `Toxicity` score fluctuates slightly (e.g., `0.12` to `0.14`) across identical runs.
* **Resolution:** While the mathematical oracles are deterministic, generative LLMs are inherently probabilistic. Although the platform typically enforces low temperature values in the API payloads to minimize variance, providers (like OpenAI) state that temperature settings do not guarantee 100% determinism due to underlying GPU floating-point arithmetic. For maximum scientific rigor, we recommend running evaluations in batches of \(N > 10\) and averaging the resultant scores.

**Q: The heatmap is completely blank or returns `0.00` for all metrics unexpectedly.**

* **Symptom:** The benchmark completes rapidly, but the visual matrices show zero variance.
* **Resolution:**
  1. Verify the `.env` API keys are valid; authentication failures can result in empty strings or fallback behavior.
  2. Ensure the `variable` values in your CSV are distinct for each demographic group. Rows with the same `variable` value will be treated as a single group, producing a `0.00` difference.

