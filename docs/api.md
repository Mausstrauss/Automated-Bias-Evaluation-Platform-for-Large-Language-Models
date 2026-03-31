# API Documentation & Extensibility

While this platform primarily acts as an API *consumer* (orchestrating external LLM endpoints), it exposes strict internal data contracts and a modular Python interface for extensibility. The definitions below serve as the internal API documentation.

## 6.1. Inter-Process Communication (IPC) Contracts

Because the microservices operate asynchronously, they rely on standardized file-based API contracts located in the shared Docker volume.

### Scheduler Configuration Contract (`scheduler_config.json`)

This JSON file acts as the configuration API payload sent from the GUI to the scheduler.

```json
{
  "active": true,
  "interval_hours": 24,
  "models": ["OpenAI GPT 3.5", "Google Gemini Pro"],
  "metrics": ["Sentiment Analysis", "Toxicity Check"],
  "dataset": "Gender Templates"
}
```

## 6.2. Extending the `LLMGenerator` API

The core evaluation engine is designed to be highly extensible. To integrate a new black-box LLM provider into the platform, developers must adhere to the internal generator API contract:

1. **Locate the engine.** Open `generators.py`.
2. **Implement the generation method.** Add or configure the provider mapping so that prompts are correctly routed to the new model (e.g., via LangChain model initialization).
3. **Register the provider.** Map the new provider string in the main `PROVIDER_CONFIG` mapping and ensure `generate_batch()` supports it.
4. **Update the GUI.** Add the new model label to the `MODEL_MAPPING` dictionary in the `app_gui.py` sidebar configuration so users can select it.
5. **Update the scheduler.** Add the same new model label to the `MODEL_MAPPING` dictionary in `scheduler.py` (the root shim delegates to `app/scheduler/scheduler.py`). Without this step, the background scheduler cannot run the new model even after the GUI can.

