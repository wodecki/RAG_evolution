# RAG_evolution

This project is a Python application that utilizes [uv](https://github.com/astral-sh/uv) for dependency management and virtual environments. It includes scripts for working with datasets and ChromaDB, and the main entry point is `main.py`.

## Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (a fast Python package manager)

## Setup Instructions

1. **Install uv**  
   If you don't have `uv` installed, you can install it with pipx or pip:
   ```bash
   pipx install uv
   ```
   or
   ```bash
   pip install uv
   ```

2. **Create a virtual environment**
   ```bash
   uv venv
   ```

3. **Activate the virtual environment**
   ```bash
   source ./venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   uv sync
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## Project Structure

- `main.py` — Main entry point of the application.
- `pyproject.toml`, `uv.lock` — Project dependencies and lock file.
- `datasets/` — Scripts and data for generating and handling question datasets.
- `chroma/`, `chroma_langchain_db/` — ChromaDB database files.

## Notes

- Ensure you have Python 3.8 or newer installed.
- All dependencies are managed via `uv` and specified in `pyproject.toml`.
- For additional scripts (e.g., dataset generation), see the `datasets/` directory.