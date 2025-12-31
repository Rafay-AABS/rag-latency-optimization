# Project Setup Guide

## Python Version Requirement

This project requires a stable version of Python (e.g., 3.11 or 3.12). 
**Do not use Python 3.14 (pre-release)** as it lacks binary wheels for critical dependencies like `chromadb` and `hnswlib`, requiring C++ build tools that may not be present.

## Virtual Environment Setup

If you need to recreate the environment:

1.  Remove the existing `.venv` directory.
2.  Create a new virtual environment using Python 3.12:
    ```powershell
    py -3.12 -m venv .venv
    ```
3.  Activate the environment:
    ```powershell
    & .venv\Scripts\Activate.ps1
    ```
4.  Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```

## Troubleshooting

If you see errors like `Microsoft Visual C++ 14.0 or greater is required`, ensure you are NOT using Python 3.14. Check your python version with `python --version`.
