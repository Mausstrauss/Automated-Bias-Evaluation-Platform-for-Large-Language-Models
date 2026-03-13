# Automated Bias Evaluation Platform for LLMs

A framework for automated detection, evaluation, and benchmarking of biases 
in Large Language Models through metric-based black-box analysis.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- API Keys for OpenAI and/or Google Gemini

### Setup
1. Clone the repository
2. Copy .env.example to .env and fill in your API keys
3. Run: docker-compose up --build
4. Open: http://localhost:8501

## Entry Points

| Script | Purpose | How to run |
|--------|---------|------------|
| app_gui.py | Streamlit dashboard (interactive) | streamlit run app_gui.py |
| scheduler.py | Background scheduler for recurring audits | python scheduler.py |
| main.py | CLI for one-off headless audits | python main.py |

## Documentation

Full documentation is available in the docs/ directory:
- [Theory & Methodology](docs/theory.md)
- [System Architecture](docs/architecture.md)
- [Getting Started](docs/getting-started.md)
- [Tutorial](docs/tutorial.md)
- [API & Extensibility](docs/api.md)
- [FAQ & Troubleshooting](docs/faq.md)

## About LangBiTe Templates
This platform uses a CSV template format inspired by the LangBiTe benchmark 
framework for structured bias evaluation. Templates follow the format:
template, variable_type, values
"The <gender> is known for being...", gender, "man|woman"

## Tech Stack
- Frontend: Streamlit
- LLM APIs: OpenAI, Google Gemini
- Bias Oracles: TextBlob (sentiment), unitary/toxic-bert (toxicity)
- Containerization: Docker
