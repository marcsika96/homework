# Notes

Working notes for ReviewGraph. Decisions, problems and their fixes go here as they happen.

## AI usage log

| Date | Tool | Used for |
|------|------|----------|
| 2026-11-11 | Claude Code | Wrote `implementation-plan.html`, the exercise-by-exercise guide I follow for this project. |
| 2026-11-12 | Claude Code | Exercise 0.1: initialised the git repository, wrote `.gitignore` and started this file. |
| 2026-11-12 | Claude Code | Exercise 0.2: created the venv, found why the lab constraints were ignored on my machine, produced `requirements.txt`. |
| 2026-11-12 | Claude Code | Exercise 0.3: package skeleton, `pyproject.toml`, `.env.example`, filled in `get_settings()` in `config.py`, wrote `tests/test_config.py`. |

## Decisions

- 2026-11-12: There is no template repository, so the repository was created with `git init` in this folder. Work happens directly on `main` instead of a `solution` branch.
- 2026-11-12: Python 3.12.7 (x64) in `.venv`. `requirements.txt` is a `pip freeze` of an install constrained by the lab 3 requirements file, so shared packages have the lab versions: langchain 1.4.2, langchain-core 1.6.5, langgraph 1.2.12, langchain-ollama 1.1.0, pydantic 2.13.5, google-genai 2.25.0. Packages the lab file does not contain: langchain-google-genai 4.4.0 (no conflict with langchain-core 1.6.5), python-dotenv 1.2.4, pytest 9.2.0.

## Problems

- 2026-11-12: `pip install -c <lab requirements URL> ...` ignored every pin and installed the newest versions (langchain 1.5.0, langgraph 1.2.20). Cause: my laptop is Windows on ARM running x64 Python, where `platform.machine()` returns `ARM64`, and every line of the lab file carries the marker `platform_machine == 'AMD64' and sys_platform == 'win32'`. Fix: evaluated the markers with `platform_machine` set to `AMD64`, wrote the matching lines to a plain constraints file without markers, and reinstalled into a fresh venv with that file. `pip check` reports no broken requirements. `requirements.txt` has no markers, so it installs the same versions on any machine.
