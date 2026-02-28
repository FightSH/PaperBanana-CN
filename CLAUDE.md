# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PaperBanana-CN is a Chinese-localized AI academic illustration assistant. It transforms paper methodology sections and figure captions into publication-ready diagrams/plots via a multi-agent pipeline. Built on the open-source [PaperBanana](https://github.com/dwzhu-pku/PaperBanana), optimized for mainland China users with Evolink API support.

## Common Commands

```bash
# Environment setup
python3 -m venv .venv && source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run Streamlit web UI
streamlit run demo.py --server.port 8501

# Run batch processing
python main.py --exp_mode demo_planner_critic --max_critic_rounds 3

# Run tests
pytest tests/ -v
pytest tests/test_evolink_provider.py -v

# Platform-specific one-click launchers
# macOS: double-click mac-start.command
# Windows: double-click win-start.bat
```

## Architecture

### Multi-Agent Pipeline

The core flow is orchestrated by `PaperVizProcessor` (`utils/paperviz_processor.py`):

```
Retriever → Planner → Stylist → Visualizer → Critic (iterates 1-5 rounds)
```

Pipeline modes (configured via `exp_mode`):
- `vanilla`: Direct generation, no retrieval/styling
- `dev_planner`: Retriever → Planner → Visualizer
- `dev_planner_stylist`: Adds Stylist step
- `demo_planner_critic`: Default UI mode — includes Critic iteration loop
- `demo_full`: Full pipeline with Stylist + Critic

### Agent System

All agents inherit from `BaseAgent` (`agents/base_agent.py`) with an `async process(data: Dict) -> Dict` interface. Agents communicate via a shared dict with keys following the pattern: `target_<task>_<agent>_<field><round>` (e.g., `target_diagram_critic_desc0_base64_jpg`).

### Provider Abstraction

`BaseProvider` (`providers/base.py`) defines two abstract methods: `generate_text()` and `generate_image()`. Current implementation: `EvolinkProvider` (`providers/evolink.py`) using OpenAI-compatible endpoints. To add a new provider, implement `BaseProvider` and register in `providers/__init__.py`'s `create_provider()` factory.

Agent code routes calls based on `exp_config.provider`:
```python
if self.exp_config.provider == "evolink":
    response = await call_evolink_text_with_retry_async(...)
else:
    response = await call_gemini_with_retry_async(...)
```

### Configuration

`ExpConfig` dataclass (`utils/config.py`) is the single source of truth for experiment settings (model names, task type, pipeline mode, retrieval settings, etc.). It's passed to all agents. Falls back to `configs/model_config.yaml` (copied from `model_config.template.yaml`, not committed) for API keys and model defaults.

### Entry Points

- **`demo.py`**: Streamlit web UI with two tabs — "Generate Candidates" (parallel multi-candidate generation) and "Refine Images" (image-to-image editing/upscaling)
- **`main.py`**: CLI batch processing — reads JSON datasets from `data/`, writes results to `results/`

### Key Patterns

- **Async-first**: All I/O uses `async/await` with `aiohttp.ClientSession`. Never use blocking I/O in coroutines.
- **Concurrency control**: Parallel candidate generation capped at 3 concurrent tasks to avoid Evolink 429 rate limiting.
- **Incremental persistence**: Batch processing saves results every 10 samples to avoid data loss.
- **Smart retrieval**: Default `auto` mode sends only captions (~30K tokens) instead of full papers (~800K tokens), saving 96% on API costs.

## Coding Conventions

- PEP 8, 4-space indentation
- Files: `snake_case.py`, classes: `PascalCase`, functions/variables: `snake_case`
- New providers/agents must implement the abstract base class interfaces
- Place shared logic in `utils/`, keep `agents/` focused on orchestration and strategy
- Tests use `pytest` with `pytest.mark.asyncio`; mock external API calls
- Commits: `feat:`/`fix:` prefix with Chinese description (e.g., `fix: 降低并发数避免 Evolink API 429 限流`)
