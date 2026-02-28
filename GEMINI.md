# GEMINI.md

## Project Overview

This is a Python-based application for generating academic illustrations. It uses a Streamlit web interface to allow users to input their paper's methodology and captions, and then generates multiple candidate illustrations using a pipeline of AI agents.

The core of the project is a multi-agent system that includes:
- **Retriever Agent:** Finds relevant examples from a reference set.
- **Planner Agent:** Translates the text into a detailed description of the illustration.
- **Stylist Agent:** Optimizes the aesthetic style of the illustration.
- **Visualizer Agent:** Generates the image from the description.
- **Critic Agent:** Reviews the generated image and suggests improvements.

The project is designed to be extensible, with a provider-based architecture that allows for the integration of different API services (such as Evolink and Google Gemini) for text and image generation.

## Building and Running

The project can be run using either the provided start scripts or manually.

### Quick Start (Recommended)

-   **macOS:** Double-click `mac-start.command`.
-   **Windows:** Double-click `win-start.bat`.

These scripts will automatically set up a Python virtual environment, install the required dependencies from `requirements.txt`, and launch the Streamlit application.

### Manual Installation

If the quick start scripts fail, you can set up and run the project manually:

1.  **Ensure you have Python 3.10+ installed.**

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    python3 -m venv .venv

    # Activate on macOS/Linux
    source .venv/bin/activate

    # Activate on Windows
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run demo.py --server.port 8501
    ```

The application will be available at `http://localhost:8501`.

### Batch Processing

The `main.py` script is used for batch processing of datasets. It takes several command-line arguments to configure the experiment.

Example usage:
```bash
python main.py --dataset_name PaperBananaBench --task_name diagram --split_name test
```

## Development Conventions

-   **Modular Architecture:** The project is organized into several modules, including `agents`, `providers`, `utils`, and `prompts`.
-   **Provider-Based API Integration:** API services are abstracted into `Provider` classes (see `providers/base.py` and `providers/evolink.py`). This makes it easy to add support for new APIs.
-   **Configuration:** The project uses YAML files for configuration (e.g., `configs/model_config.template.yaml`).
-   **Asynchronous Operations:** The project uses `asyncio` for handling concurrent API requests, which is crucial for the parallel generation of candidate illustrations.
-   **Dependencies:** All Python dependencies are listed in `requirements.txt`.
-   **Frontend:** The user interface is built with Streamlit (`demo.py`).
-   **Code Style:** The code follows standard Python conventions (PEP 8).
-   **Testing:** The `tests` directory contains tests for the project.
